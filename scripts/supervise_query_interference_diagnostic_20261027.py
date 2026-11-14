"""Owned, bounded supervisor for a single readonly GPU diagnostic."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import time

from filelock import FileLock, Timeout
from learning_io_v2 import atomic_json, atomic_text, load_json

POLL_SECONDS = 2
STOP_GRACE_SECONDS = 8
SHUTDOWN_RESERVE_SECONDS = 30
CHECKPOINT_SCHEMA = "fdt_local_checkpoint_v1"
ACTIVE_WORKERS = (
    "run_learning_102m.py", "run_learning_102m_v2.py", "run_e2_grounding_20261026.py",
    "run_role8_diffusion_20261027.py", "run_active_balanced_diffusion_20261027.py",
    "run_active_balanced_diffusion_20261027_v3.py", "run_query_mix_diffusion_20261027.py",
    "run_query_interference_diagnostic_20261027.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_digest(package_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _no_window():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def validate_plan(plan_path: Path, worker: Path, output: Path):
    plan = load_json(plan_path)
    root = Path(__file__).resolve().parents[1]
    if plan.get("schema") != "fdt_query_interference_readonly_v1":
        raise ValueError("readonly diagnostic plan schema mismatch")
    for key, actual in (("output_dir", output.resolve()), ("control_dir", (output / "control").resolve())):
        if Path(plan[key]).resolve() != actual:
            raise ValueError(f"{key} mismatch")
    if plan.get("authorized") not in (False, True):
        raise ValueError("authorized must be boolean")
    if (plan.get("wall_seconds"), plan.get("max_logical_forwards"), plan.get("seed"),
            plan.get("query_count"), plan.get("graph_count")) != (120, 200, 17, 16, 8):
        raise ValueError("readonly diagnostic bounds or fixed sample mismatch")
    if plan.get("optimizer_allowed") is not False or plan.get("checkpoint_write_allowed") is not False:
        raise ValueError("optimizer/checkpoint writes must be prohibited")
    if plan.get("e2_final_split_accessed") is not False:
        raise ValueError("E2 final holdout prohibited")
    if plan.get("selected_dev_indices") != [0, 8, 16, 24, 32, 40, 48, 56]:
        raise ValueError("fixed systematic QueryMix slice mismatch")
    if (plan.get("min_free_bytes", 0) < 20 * 1024**3 or plan.get("min_gpu_free_mib", 0) < 12288
            or plan.get("stall_seconds", 0) <= 0):
        raise ValueError("diagnostic resource/stall bounds mismatch")
    deps = {
        "worker_sha256": worker.resolve(),
        "supervisor_sha256": Path(__file__).resolve(),
        "io_sha256": Path(__file__).with_name("learning_io_v2.py"),
        "parent_marker_sha256": Path(plan["parent_marker"]),
        "dev_data_sha256": Path(plan["dev_data_path"]),
        "candidate_manifest_sha256": Path(plan["candidate_manifest_path"]),
        "data_source_sha256": root / "src/fdt_local/data.py",
        "model_source_sha256": root / "src/fdt_local/model_masked.py",
    }
    mismatches = {k: (plan.get(k), sha256(p)) for k, p in deps.items() if plan.get(k) != sha256(p)}
    if mismatches:
        raise ValueError(f"readonly diagnostic dependency digest mismatch: {mismatches}")
    if runtime_digest(root / "src/fdt_local") != plan["runtime_source_sha256"]:
        raise ValueError("FDT runtime source digest mismatch")
    if plan.get("core_sha256") != plan["runtime_source_sha256"]:
        raise ValueError("core/runtime source digest mismatch")
    if Path(plan["parent_marker"]).name != "complete_000262.json":
        raise ValueError("must use the final QueryMix checkpoint marker")
    if Path(plan["dev_data_path"]).resolve() != (root / "research/artifacts/role8_query_mix_20261027_v1/dev.json").resolve():
        raise ValueError("unexpected diagnostic dev source")
    allowed_existing = {"plan.json", "seal.json", "readiness.json", "code_snapshot",
                        "supervisor.stdout.log", "supervisor.stderr.log"}
    if output.exists():
        unexpected = {p.name for p in output.iterdir()} - allowed_existing
        if unexpected:
            raise FileExistsError(f"diagnostic output not new; unexpected entries {sorted(unexpected)}")
    return plan, output / "control", output


def worker_preflight(plan_path: Path, worker: Path, output: Path):
    args = [sys.executable, "-B", "-X", "utf8", str(worker.resolve()),
            "--plan", str(plan_path.resolve()), "--output", str(output.resolve()), "--preflight"]
    result = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            shell=False, timeout=90, creationflags=_no_window())
    if result.returncode or result.stdout.strip() != "CPU_PREFLIGHT_PASS":
        raise RuntimeError(f"worker CPU preflight failed: rc={result.returncode} out={result.stdout[-2000:]} err={result.stderr[-3000:]}")
    return {"status": "CPU_PREFLIGHT_PASS"}


def inspect_resources(plan: dict, root: Path):
    disk = shutil.disk_usage(root)
    if disk.free < plan["min_free_bytes"]:
        raise RuntimeError(f"C: free space below 20 GiB: {disk.free}")
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15, creationflags=_no_window())
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values or max(values) < plan["min_gpu_free_mib"]:
        raise RuntimeError(f"GPU free memory below 12 GiB: {values}")
    return {"disk_free_bytes": disk.free, "gpu_free_mib": max(values)}


def runs_active_worker(command: str, worker_paths) -> bool:
    """Classify the invoked script, excluding a supervisor's later --worker argument."""
    try:
        arguments = shlex.split(command, posix=False)
    except ValueError:
        return False
    scripts = (arg.strip('"').lower().replace("\\", "/") for arg in arguments)
    invoked = next((arg for arg in scripts if arg.endswith(".py")), None)
    return invoked in worker_paths


def process_inventory(root: Path):
    project = str(root.resolve()).lower().replace("\\", "/")
    script = "$self=$PID; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $self -and $_.Name -match '^python(w)?\\.exe$' } | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True, capture_output=True, text=True, timeout=5, creationflags=_no_window())
    items = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(items, dict): items = [items]
    paths = [str((root / "scripts" / name).resolve()).lower().replace("\\", "/") for name in ACTIVE_WORKERS]
    found = []
    for item in items:
        command = (item.get("CommandLine") or "").lower().replace("\\", "/")
        if project in command and runs_active_worker(item.get("CommandLine") or "", paths):
            found.append(item)
    return found


def process_identity(pid: int, expected_parent: int, worker: Path, plan: Path, output: Path):
    script = ("$p=" + str(int(pid)) + "; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -eq $p } "
              "| Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True, capture_output=True, text=True, timeout=2, creationflags=_no_window())
    item = json.loads(result.stdout) if result.stdout.strip() else None
    if not isinstance(item, dict): return None
    command = (item.get("CommandLine") or "").lower().replace("\\", "/")
    valid = (int(item.get("ProcessId", -1)) == int(pid)
        and int(item.get("ParentProcessId", -1)) == int(expected_parent)
        and str(worker.resolve()).lower().replace("\\", "/") in command
        and str(plan.resolve()).lower().replace("\\", "/") in command
        and str(output.resolve()).lower().replace("\\", "/") in command)
    item["identity_verified"] = bool(valid)
    if not valid:
        raise RuntimeError(f"worker PID identity does not match owned launch: {item}")
    return item


def _atomic_state(path, state):
    state["observed_unix"] = time.time()
    atomic_json(path, state)


def _write_stop(control: Path, reason: str):
    control.mkdir(parents=True, exist_ok=True)
    path = control / "STOP"
    if not path.exists():
        atomic_text(path, reason + "\n")


def _stop_owned_tree(child):
    if child.poll() is not None:
        return child.poll()
    subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"], check=False,
                   capture_output=True, text=True, timeout=2, creationflags=_no_window())
    try:
        return child.wait(timeout=1)
    except subprocess.TimeoutExpired:
        return None


def verify_owned_exit(child, actual, worker, plan_path, output, root, terminate=False):
    identity_error = None
    actual_after = None
    if actual is not None:
        try:
            actual_after = process_identity(actual["ProcessId"], child.pid, worker, plan_path, output)
        except Exception as exc:
            identity_error = repr(exc)
        if actual_after is not None and terminate:
            subprocess.run(["taskkill.exe", "/PID", str(actual["ProcessId"]), "/T", "/F"],
                check=False, capture_output=True, text=True, timeout=2, creationflags=_no_window())
            try:
                actual_after = process_identity(actual["ProcessId"], child.pid, worker, plan_path, output)
            except Exception as exc:
                identity_error = repr(exc)
    def owned_matches():
        matches=[]
        worker_text=str(worker.resolve()).lower().replace("\\", "/")
        plan_text=str(plan_path.resolve()).lower().replace("\\", "/")
        output_text=str(output.resolve()).lower().replace("\\", "/")
        for item in process_inventory(root):
            command=(item.get("CommandLine") or "").lower().replace("\\", "/")
            if worker_text in command and plan_text in command and output_text in command:
                matches.append(item)
        return matches
    lingering=owned_matches()
    if terminate and lingering:
        for item in lingering:
            try:
                if int(item["ParentProcessId"]) == child.pid:
                    subprocess.run(["taskkill.exe", "/PID", str(int(item["ProcessId"])), "/T", "/F"],
                        check=False, capture_output=True, text=True, timeout=2, creationflags=_no_window())
            except Exception:
                pass
        lingering=owned_matches()
    return {"proxy_exit_code":child.poll(),"actual_process_after_exit":actual_after,
        "identity_error":identity_error,"lingering_processes":lingering,
        "all_owned_processes_gone":child.poll() is not None and actual_after is None
            and identity_error is None and not lingering}


def supervise(plan_path: Path, worker: Path, output: Path):
    operation_start = time.time()
    deadline_unix = operation_start + 120
    plan, control, output = validate_plan(plan_path.resolve(), worker, output.resolve())
    if plan.get("authorized") is not True:
        raise ValueError("readonly GPU run requires a separately authorized plan")
    worker_preflight(plan_path, worker, output)
    root = Path(__file__).resolve().parents[1]
    lock_hash = hashlib.sha256(str(root).lower().encode()).hexdigest()[:20]
    lock = FileLock(str(Path(os.getenv("TEMP", ".")) / f"fdt_readonly_diag_{lock_hash}.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise RuntimeError("another FDT readonly diagnostic supervisor is active") from exc
    try:
        resources = inspect_resources(plan, root)
        duplicates = process_inventory(root)
        if duplicates:
            raise RuntimeError(f"active FDT train/diagnostic process already exists: {duplicates}")
        output.mkdir(parents=True, exist_ok=True)
        supervisor_dir = output / "supervisor"
        supervisor_dir.mkdir(parents=True, exist_ok=True)
        state_path = supervisor_dir / "state.json"
        reserved = ("worker.stdout.log", "worker.stderr.log", "worker_progress.json", "diagnostic_result.json",
                    "paired_seed17_noise.pt", "incident.json", "final_status.json")
        if state_path.exists() or any((output / name).exists() for name in reserved):
            raise FileExistsError("diagnostic output has existing execution markers/logs")
        if output.joinpath("model").exists():
            raise FileExistsError("readonly diagnostic must never create a model directory")
        launched_unix = operation_start
        worker_deadline_unix = deadline_unix - SHUTDOWN_RESERVE_SECONDS
        if time.time() >= worker_deadline_unix:
            raise TimeoutError("CPU preflight/resource checks consumed the bounded inference window")
        state = {"status": "LAUNCHING", "launch_count": 1, "launched_unix": launched_unix,
            "deadline_unix": deadline_unix, "worker_deadline_unix": worker_deadline_unix,
            "supervisor_pid": os.getpid(), "worker_proxy_pid": None, "worker_actual_process": None,
            "command": [sys.executable, "-B", "-X", "utf8", str(worker.resolve()),
                        "--plan", str(plan_path.resolve()), "--output", str(output.resolve())],
            "worker_sha256": plan["worker_sha256"], "supervisor_sha256": plan["supervisor_sha256"],
            "parent_payload_sha256": plan["parent_payload_sha256"], "resources_preflight": resources,
            "no_optimizer": True, "no_checkpoint_write": True}
        _atomic_state(state_path, state)
        stdout_path, stderr_path = output / "worker.stdout.log", output / "worker.stderr.log"
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            child = subprocess.Popen(state["command"], stdin=subprocess.DEVNULL, stdout=stdout,
                stderr=stderr, shell=False, creationflags=_no_window())
            state["worker_started_unix"] = time.time()
            state["worker_proxy_pid"] = child.pid
            state["status"] = "RUNNING"
            _atomic_state(state_path, state)
            progress_path = output / "worker_progress.json"
            previous = None
            changed_at = time.monotonic()
            stop_started = None
            stop_reason = None
            actual = None
            while True:
                now_mono = time.monotonic()
                progress = load_json(progress_path)
                if isinstance(progress, dict):
                    sig = json.dumps({k:v for k,v in progress.items() if k != "observed_unix"}, sort_keys=True)
                    if sig != previous:
                        previous, changed_at = sig, now_mono
                    pid = progress.get("worker_pid")
                    if isinstance(pid, int):
                        actual = process_identity(pid, child.pid, worker, plan_path, output)
                if stop_reason is None:
                    if time.time() >= worker_deadline_unix:
                        stop_reason = "wall_budget_shutdown_reserve"
                    elif now_mono - changed_at >= plan["stall_seconds"]:
                        stop_reason = "worker_progress_stalled"
                    elif isinstance(progress, dict) and progress.get("anomaly"):
                        stop_reason = "worker_anomaly: " + str(progress["anomaly"])
                    elif shutil.disk_usage(root).free < plan["min_free_bytes"]:
                        stop_reason = "disk_free_below_20GiB"
                    if stop_reason:
                        stop_started = now_mono
                        _write_stop(control, stop_reason)
                state.update({"status": "STOPPING" if stop_reason else "RUNNING",
                    "stop_reason": stop_reason, "worker_progress": progress,
                    "worker_actual_process": actual, "worker_proxy_alive": child.poll() is None,
                    "last_progress_change_monotonic": changed_at})
                _atomic_state(output / "supervisor/latest.json", {k:v for k,v in state.items() if k != "command"})
                code = child.poll()
                if code is not None:
                    identity_error = None
                    remaining_actual = None
                    if actual is not None:
                        try:
                            remaining_actual = process_identity(actual["ProcessId"], child.pid, worker, plan_path, output)
                        except Exception as exc:
                            identity_error = repr(exc)
                        if remaining_actual is not None:
                            subprocess.run(["taskkill.exe", "/PID", str(actual["ProcessId"]), "/T", "/F"],
                                check=False, capture_output=True, text=True, timeout=2, creationflags=_no_window())
                            try:
                                remaining_actual = process_identity(actual["ProcessId"], child.pid, worker, plan_path, output)
                            except Exception as exc:
                                identity_error = repr(exc)
                    inventory = process_inventory(root)
                    lingering = [item for item in inventory if str(worker.resolve()).lower().replace("\\", "/")
                        in (item.get("CommandLine") or "").lower().replace("\\", "/")
                        and str(plan_path.resolve()).lower().replace("\\", "/")
                        in (item.get("CommandLine") or "").lower().replace("\\", "/")]
                    if lingering:
                        for item in lingering:
                            try:
                                owned_pid = int(item["ProcessId"])
                                owned_parent = int(item["ParentProcessId"])
                                command = (item.get("CommandLine") or "").lower().replace("\\", "/")
                                expected = str(output.resolve()).lower().replace("\\", "/")
                                if owned_parent == child.pid and expected in command:
                                    subprocess.run(["taskkill.exe", "/PID", str(owned_pid), "/T", "/F"],
                                        check=False, capture_output=True, text=True, timeout=2, creationflags=_no_window())
                            except Exception:
                                pass
                        inventory = process_inventory(root)
                        lingering = [item for item in inventory if str(worker.resolve()).lower().replace("\\", "/")
                            in (item.get("CommandLine") or "").lower().replace("\\", "/")
                            and str(plan_path.resolve()).lower().replace("\\", "/")
                            in (item.get("CommandLine") or "").lower().replace("\\", "/")]
                    final = load_json(output / "diagnostic_result.json")
                    clean_exit = not lingering and remaining_actual is None and identity_error is None
                    status = ("COMPLETE" if code == 0 and final and not stop_reason and clean_exit
                              else ("UNCONFIRMED_EXIT" if not clean_exit else ("STOPPED" if stop_reason else "FAILED")))
                    state.update({"status": status, "worker_exit_code": code,
                        "worker_proxy_alive": False, "worker_actual_process_after_exit": remaining_actual,
                        "worker_identity_error_after_exit": identity_error,
                        "lingering_owned_workers_after_exit": lingering,
                        "diagnostic_result_present": bool(final), "finished_unix": time.time(),
                        "elapsed_seconds": time.time() - launched_unix})
                    _atomic_state(output / "supervisor/final.json", {k:v for k,v in state.items() if k != "command"})
                    atomic_json(output / "final_status.json", {"status": status,
                        "worker_exit_code": code, "stop_reason": stop_reason,
                        "diagnostic_result_present": bool(final), "launch_count": 1,
                        "elapsed_seconds": time.time() - launched_unix})
                    return 0 if status == "COMPLETE" else 1
                if stop_reason and now_mono - stop_started >= STOP_GRACE_SECONDS:
                    code = _stop_owned_tree(child)
                    exit_check = verify_owned_exit(child, actual, worker, plan_path, output, root, terminate=True)
                    stop_status = "STOPPED" if exit_check["all_owned_processes_gone"] else "UNCONFIRMED_EXIT"
                    state.update({"status": stop_status, "worker_exit_code": code,
                        "stop_reason": stop_reason, "worker_proxy_alive": child.poll() is None,
                        "exit_verification": exit_check,
                        "finished_unix": time.time(), "elapsed_seconds": time.time() - launched_unix})
                    _atomic_state(output / "supervisor/final.json", {k:v for k,v in state.items() if k != "command"})
                    atomic_json(output / "final_status.json", {"status": stop_status, "worker_exit_code": code,
                        "stop_reason": stop_reason, "launch_count": 1,
                        "exit_verification": exit_check,
                        "elapsed_seconds": time.time() - launched_unix})
                    return 1
                remaining = max(0., min(POLL_SECONDS, deadline_unix-time.time()))
                if remaining <= 0:
                    stop_reason = stop_reason or "wall_clock_deadline"
                    _write_stop(control, stop_reason)
                    code = _stop_owned_tree(child)
                    exit_check = verify_owned_exit(child, actual, worker, plan_path, output, root, terminate=True)
                    stop_status = "STOPPED" if exit_check["all_owned_processes_gone"] else "UNCONFIRMED_EXIT"
                    state.update({"status": stop_status, "worker_exit_code": code,
                        "stop_reason": stop_reason, "worker_proxy_alive": child.poll() is None,
                        "exit_verification": exit_check,
                        "finished_unix": time.time(), "elapsed_seconds": time.time() - launched_unix})
                    _atomic_state(output / "supervisor/final.json", {k:v for k,v in state.items() if k != "command"})
                    atomic_json(output / "final_status.json", {"status": stop_status, "worker_exit_code": code,
                        "stop_reason": stop_reason, "launch_count": 1,
                        "exit_verification": exit_check,
                        "elapsed_seconds": time.time() - launched_unix})
                    return 1
                time.sleep(remaining)
    finally:
        lock.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--validate", action="store_true", help="CPU-only; no GPU resource query or launch")
    args = parser.parse_args()
    try:
        if args.validate:
            plan, _, output = validate_plan(args.plan.resolve(), args.worker.resolve(), args.output.resolve())
            result = worker_preflight(args.plan.resolve(), args.worker.resolve(), output)
            print(json.dumps({"status":"CPU_PREFLIGHT_OK_NO_GPU_NO_LAUNCH","worker_preflight":result,
                              "plan_sha256":sha256(args.plan.resolve())}, ensure_ascii=False))
            return 0
        return supervise(args.plan, args.worker, args.output)
    except Exception as exc:
        print(f"readonly supervisor error: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
