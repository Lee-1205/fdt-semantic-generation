from __future__ import annotations

import json
import math
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


ROOT = Path(r"<USER_HOME>\Documents\Codex\2026-06-26\dk\outputs\FDT_RLM").resolve()
RUN = (ROOT / "runs/fdt_v4_1_424m_alpha025_recovery_micro_20260903_r1").resolve()
TRAIN_LOG = RUN / "training_log.jsonl"
STDERR = RUN / "stderr.log"
STOP = RUN / "STOP_REQUESTED"
SUPERVISOR_LOG = RUN / "supervisor_log.jsonl"
INITIAL_VALIDATION_LOSS = 2.0763975977897644
MAX_VALIDATION_LOSS = INITIAL_VALIDATION_LOSS * 1.01
MIN_FREE_GIB = 40.0


def append(event: dict[str, object]) -> None:
    payload = {"time": datetime.now(timezone.utc).isoformat(), **event}
    with SUPERVISOR_LOG.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def request_stop(reason: str, detail: dict[str, object]) -> None:
    if not STOP.exists():
        temp = STOP.with_name(STOP.name + ".tmp")
        temp.write_text(reason + "\n", encoding="ascii")
        temp.replace(STOP)
    append({"event": "stop_requested", "reason": reason, **detail})


def main() -> int:
    if ROOT not in RUN.parents or not RUN.is_dir():
        raise RuntimeError("run escaped the FDT workspace or does not exist")
    pid = int((RUN / "train.pid").read_text(encoding="ascii").strip())
    offset = 0
    last_status = 0.0
    append({
        "event": "supervisor_start",
        "trainer_pid": pid,
        "min_free_gib": MIN_FREE_GIB,
        "initial_validation_loss": INITIAL_VALIDATION_LOSS,
        "max_validation_loss": MAX_VALIDATION_LOSS,
    })
    while True:
        free_gib = shutil.disk_usage(ROOT).free / (1024**3)
        if free_gib < MIN_FREE_GIB:
            request_stop("disk_floor", {"free_gib": free_gib})
            return 2
        if STDERR.exists() and STDERR.stat().st_size > 0:
            request_stop("stderr_nonempty", {"stderr_bytes": STDERR.stat().st_size})
            return 3

        final_seen = False
        if TRAIN_LOG.exists():
            with TRAIN_LOG.open("r", encoding="utf-8") as stream:
                stream.seek(offset)
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    kind = event.get("event")
                    if kind == "train" and not bool(event.get("finite_loss", False)):
                        request_stop("nonfinite_train", {"step": event.get("step")})
                        return 4
                    if kind == "validation":
                        loss = float(event.get("validation_loss", math.inf))
                        if not math.isfinite(loss) or loss > MAX_VALIDATION_LOSS:
                            request_stop("validation_gate", {"step": event.get("step"), "validation_loss": loss})
                            return 5
                    if kind == "final":
                        final_seen = True
                offset = stream.tell()
        if final_seen:
            append({"event": "supervisor_complete", "free_gib": free_gib})
            return 0
        if not psutil.pid_exists(pid):
            append({"event": "trainer_exit_without_final", "trainer_pid": pid, "free_gib": free_gib})
            return 6
        now = time.monotonic()
        if now - last_status >= 300:
            append({"event": "status", "trainer_pid": pid, "free_gib": free_gib})
            last_status = now
        time.sleep(15)


if __name__ == "__main__":
    raise SystemExit(main())
