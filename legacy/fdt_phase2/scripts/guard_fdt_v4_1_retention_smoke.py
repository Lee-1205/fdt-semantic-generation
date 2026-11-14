from __future__ import annotations

import argparse
import ctypes
import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(r"<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM").resolve()
RUNS = (ROOT / "runs").resolve()
MIN_FREE_GIB = 40.0
MIN_FREE_COMMIT_GIB = 2.0
CRITICAL_STDERR = (
    "traceback",
    "cuda out of memory",
    "device-side assert",
    "non-finite",
    "nonfinite",
    "floatingpointerror",
    "nvlddmkm",
    "tdr",
)
MALICIOUS_MARKERS = ("woolypooly", "progpow", "stratum")


class MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("memory_load", ctypes.c_ulong),
        ("total_phys", ctypes.c_ulonglong),
        ("avail_phys", ctypes.c_ulonglong),
        ("total_page_file", ctypes.c_ulonglong),
        ("avail_page_file", ctypes.c_ulonglong),
        ("total_virtual", ctypes.c_ulonglong),
        ("avail_virtual", ctypes.c_ulonglong),
        ("avail_extended_virtual", ctypes.c_ulonglong),
    ]


def free_commit_gib() -> float:
    status = MemoryStatusEx()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError("GlobalMemoryStatusEx failed")
    return status.avail_page_file / (1024**3)


def require_owned_run(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:" or RUNS not in resolved.parents:
        raise ValueError("guardian run must be an owned C-only FDT run")
    return resolved


def append(path: Path, event: dict[str, object]) -> None:
    payload = {"time": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def request_stop(stop: Path, log: Path, reason: str, detail: dict[str, object]) -> None:
    if not stop.exists():
        temporary = stop.with_name(stop.name + ".tmp")
        temporary.write_text(reason + "\n", encoding="ascii")
        temporary.replace(stop)
    append(log, {"event": "stop_requested", "reason": reason, **detail})


def event_stop_reason(event: dict[str, object]) -> str | None:
    kind = event.get("event")
    if kind == "train":
        values = (
            event.get("loss"),
            event.get("base_grad_norm"),
            event.get("exact_pointer_grad_norm"),
        )
        if not bool(event.get("finite_loss", False)) or any(
            value is not None and not math.isfinite(float(value)) for value in values
        ):
            return "nonfinite_train"
    if kind == "validation" and not bool(event.get("finite_loss", False)):
        return "nonfinite_validation"
    if kind in {"fixed52_startup_validation", "fixed52_validation"}:
        if bool(event.get("gate_triggered", False)):
            return "fixed52_regression_gate"
        for name in ("row_mean_nll", "token_weighted_nll"):
            if not math.isfinite(float(event.get(name, math.inf))):
                return "nonfinite_fixed52"
    return None


def new_gpu_driver_event(since_iso: str) -> bool:
    escaped = since_iso.replace("'", "''")
    command = (
        f"$s=[DateTime]::Parse('{escaped}').ToUniversalTime();"
        "$e=@(Get-WinEvent -FilterHashtable @{LogName='System';StartTime=$s} "
        "-ErrorAction SilentlyContinue | Where-Object {"
        "($_.ProviderName -match 'nvlddmkm|Display') -and ($_.Id -in 153,4101)});"
        "if($e.Count -gt 0){'1'}else{'0'}"
    )
    result = subprocess.run(
        [
            r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip().endswith("1")


def malicious_process() -> str | None:
    for process in psutil.process_iter(["name", "cmdline"]):
        try:
            text = " ".join(
                [process.info.get("name") or "", *(process.info.get("cmdline") or [])]
            ).lower()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if any(marker in text for marker in MALICIOUS_MARKERS):
            return text[:256]
    return None


def supervise(run_dir: Path, poll_seconds: float) -> int:
    run = require_owned_run(run_dir)
    pid_path = run / "train.pid"
    if not pid_path.is_file():
        raise FileNotFoundError("trainer PID is missing")
    pid = int(pid_path.read_text(encoding="ascii").strip())
    train_log = run / "training_log.jsonl"
    stderr = run / "stderr.log"
    stop = run / "STOP_REQUESTED"
    guardian_log = run / "guardian_log.jsonl"
    started = datetime.now(timezone.utc).isoformat()
    offset = 0
    stderr_offset = 0
    last_driver_check = 0.0
    append(guardian_log, {"event": "guardian_start", "trainer_pid": pid})
    while True:
        free_gib = shutil.disk_usage(ROOT).free / (1024**3)
        commit_gib = free_commit_gib()
        if free_gib < MIN_FREE_GIB:
            request_stop(stop, guardian_log, "disk_floor", {"free_gib": free_gib})
            return 2
        if commit_gib < MIN_FREE_COMMIT_GIB:
            request_stop(
                stop, guardian_log, "commit_floor", {"free_commit_gib": commit_gib}
            )
            return 3

        final_seen = False
        if train_log.exists():
            with train_log.open("r", encoding="utf-8") as stream:
                stream.seek(offset)
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    reason = event_stop_reason(event)
                    if reason is not None:
                        request_stop(
                            stop,
                            guardian_log,
                            reason,
                            {"step": event.get("step"), "source_event": event.get("event")},
                        )
                        return 4
                    final_seen = final_seen or event.get("event") == "final"
                offset = stream.tell()

        if stderr.exists():
            with stderr.open("r", encoding="utf-8", errors="replace") as stream:
                stream.seek(stderr_offset)
                new_stderr = stream.read()
                stderr_offset = stream.tell()
            lowered = new_stderr.lower()
            marker = next((item for item in CRITICAL_STDERR if item in lowered), None)
            if marker is not None:
                request_stop(stop, guardian_log, "critical_stderr", {"marker": marker})
                return 5

        now = time.monotonic()
        if now - last_driver_check >= 60.0:
            if new_gpu_driver_event(started):
                request_stop(stop, guardian_log, "gpu_driver_event", {})
                return 6
            bad_process = malicious_process()
            if bad_process is not None:
                request_stop(
                    stop, guardian_log, "malicious_process_indicator", {"match": bad_process}
                )
                return 7
            last_driver_check = now

        if final_seen:
            append(
                guardian_log,
                {
                    "event": "guardian_complete",
                    "free_gib": free_gib,
                    "free_commit_gib": commit_gib,
                },
            )
            return 0
        if not psutil.pid_exists(pid):
            append(
                guardian_log,
                {
                    "event": "trainer_exit_without_final",
                    "trainer_pid": pid,
                    "free_gib": free_gib,
                    "free_commit_gib": commit_gib,
                },
            )
            return 8
        time.sleep(max(poll_seconds, 1.0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bounded FDT v4.1 smoke guardian")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(supervise(**vars(parse_args())))
