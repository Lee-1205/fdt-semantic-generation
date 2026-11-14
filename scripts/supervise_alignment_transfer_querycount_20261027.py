"""Single-launch bounded owner for the alignment/query-count readonly diagnostic."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from filelock import FileLock, Timeout
from learning_io_v2 import atomic_json, atomic_text, load_json

TOTAL_WALL_SECONDS = 120
SHUTDOWN_RESERVE_SECONDS = 30
STOP_GRACE_SECONDS = 6
POLL_SECONDS = 1
ACTIVE_WORKERS = (
    "run_learning_102m.py", "run_learning_102m_v2.py", "run_learning_102m_recovery_v2.py", "run_e2_grounding_20261026.py",
    "run_role8_diffusion_20261027.py", "run_active_balanced_diffusion_20261027.py",
    "run_active_balanced_diffusion_20261027_v3.py", "run_query_mix_diffusion_20261027.py",
    "run_query_bound_alignment_20261027.py", "run_query_bound_alignment_20261027_v2.py",
    "run_query_interference_diagnostic_20261027.py", "run_alignment_transfer_querycount_20261027.py",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fixed_deadlines(invocation_start: float):
    absolute = invocation_start + TOTAL_WALL_SECONDS
    return absolute, absolute - SHUTDOWN_RESERVE_SECONDS


def natural_terminal_status(exit_info: dict, exit_code: int, result_present: bool, stop_reason):
    if not exit_info.get("all_owned_processes_gone", False):
        return "UNCONFIRMED_EXIT"
    if exit_code == 0 and result_present and not stop_reason:
        return "COMPLETE"
    return "FAILED"


def _no_window():
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def invoked_script(command: str):
    try:
        args = shlex.split(command, posix=False)
    except ValueError:
        return None
    for arg in args:
        value = arg.strip('"').lower().replace("\\", "/")
        if value.endswith(".py"):
            return value
    return None


def is_active_worker_command(command: str, active_paths) -> bool:
    # Only the first .py argument is the invoked script; --worker is not a launch.
    return invoked_script(command) in set(active_paths)


def process_inventory(root: Path):
    project = str(root.resolve()).lower().replace("\\", "/")
    script = "$self=$PID; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $self -and $_.Name -match '^python(w)?\\.exe$' } | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True, capture_output=True, text=True, timeout=6, creationflags=_no_window())
    items = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(items, dict):
        items = [items]
    paths = [str((root / "scripts" / name).resolve()).lower().replace("\\", "/") for name in ACTIVE_WORKERS]
    return [item for item in items if project in (item.get("CommandLine") or "").lower().replace("\\", "/")
        and is_active_worker_command(item.get("CommandLine") or "", paths)]


def process_identity(pid: int, expected_parent: int, worker: Path, plan: Path, output: Path):
    command = ("$p=" + str(int(pid)) + "; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -eq $p } "
        "| Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True, capture_output=True, text=True, timeout=4, creationflags=_no_window())
    item = json.loads(result.stdout) if result.stdout.strip() else None
    if not isinstance(item, dict):
        return None
    text = (item.get("CommandLine") or "").lower().replace("\\", "/")
    valid = (int(item.get("ProcessId", -1)) == pid and int(item.get("ParentProcessId", -1)) == expected_parent
        and str(worker.resolve()).lower().replace("\\", "/") in text
        and str(plan.resolve()).lower().replace("\\", "/") in text
        and str(output.resolve()).lower().replace("\\", "/") in text)
    if not valid:
        raise RuntimeError(f"owned worker identity mismatch: {item}")
    item["identity_verified"] = True
    return item


def validate_plan(plan_path: Path, worker: Path, output: Path):
    plan = load_json(plan_path)
    root = Path(__file__).resolve().parents[1]
    if not isinstance(plan, dict) or plan.get("schema") != "fdt_alignment_transfer_querycount_readonly_v1":
        raise ValueError("plan schema mismatch")
    if Path(plan["output_dir"]).resolve() != output.resolve():
        raise ValueError("plan output directory mismatch")
    if plan.get("authorized") not in (False, True):
        raise ValueError("authorization marker must be boolean")
    if (plan.get("wall_seconds"), plan.get("wall_shutdown_reserve_seconds"), plan.get("max_actual_batched_model_calls"),
            plan.get("seed"), plan.get("query_counts"), plan.get("graph_count"), plan.get("batch_size")) != (120, 30, 200, 17, [1, 4, 8, 16], 16, 4):
        raise ValueError("plan bounded contract mismatch")
    if plan.get("optimizer_allowed") is not False or plan.get("checkpoint_write_allowed") is not False:
        raise ValueError("readonly diagnostic must prohibit optimizer/checkpoint writes")
    if int(plan.get("min_free_bytes", 0)) < 20 * 1024**3 or int(plan.get("min_gpu_free_mib", 0)) < 12288:
        raise ValueError("minimum disk/GPU resource reserves cannot be lowered")
    if plan.get("e2_final_files_accessed") is not False:
        raise ValueError("E2 final files are prohibited")
    root_files = {"worker": worker.resolve(), "supervisor": Path(__file__).resolve(),
        "io": root / "scripts/learning_io_v2.py", "common_worker": root / "scripts/run_e2_grounding_20261026.py",
        "builder_py": root / "research/alignment_transfer_querycount_v1.py",
        "query_bound_helper": root / "research/query_bound_refiner_v1.py",
        "v2_builder": root / "research/role8_active_balanced_v2.py",
        "frozen_data": root / "src/fdt_local/data.py", "model_config": root / "src/fdt_local/model_masked.py",
        "parent_marker": Path(plan["parent_marker"]), "manifest": Path(plan["dataset_dir"]) / "manifest.json",
        "selection_registry": Path(plan["dataset_dir"]) / "selection_registry.json",
        "config": Path(plan["config_path"])}
    mismatch = {k: (plan.get("dependency_sha256", {}).get(k), sha256(v))
        for k, v in root_files.items() if plan.get("dependency_sha256", {}).get(k) != sha256(v)}
    if mismatch:
        raise ValueError(f"pinned dependency SHA mismatch: {mismatch}")
    if plan.get("worker_sha256") != sha256(worker.resolve()) or plan.get("supervisor_sha256") != sha256(Path(__file__).resolve()):
        raise ValueError("worker/supervisor SHA mismatch")
    if plan.get("runtime_source_sha256") != plan.get("core_sha256"):
        raise ValueError("core/runtime source digest mismatch")
    if plan.get("dataset_manifest_sha256") != sha256(root_files["manifest"]):
        raise ValueError("dataset manifest SHA mismatch")
    if plan.get("selection_registry_sha256") != sha256(root_files["selection_registry"]):
        raise ValueError("selection registry SHA mismatch")
    for name, path in ((f"q{q:02d}.json", Path(plan["dataset_dir"]) / f"q{q:02d}.json") for q in (1, 4, 8, 16)):
        if plan.get("data_sha256", {}).get(name) != sha256(path):
            raise ValueError(f"data file SHA mismatch: {name}")
    if plan.get("parent_payload_sha256") != "a877f316d0d97f2c497497fd9b941184463238055906d7109d3300db4c253b36":
        raise ValueError("unexpected frozen parent weights")
    existing_allowed = {"plan.json", "readiness.json", "seal.json", "code_snapshot",
        "supervisor.stdout.log", "supervisor.stderr.log"}
    if output.exists():
        unexpected = {entry.name for entry in output.iterdir()} - existing_allowed
        if unexpected:
            raise FileExistsError(f"output contains execution/prior-run entries: {sorted(unexpected)}")
    return plan, root


def worker_preflight(plan_path: Path, worker: Path, output: Path, timeout_seconds: float):
    command = [sys.executable, "-B", "-X", "utf8", str(worker.resolve()), "--plan", str(plan_path.resolve()),
        "--output", str(output.resolve()), "--preflight"]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, shell=False,
        timeout=max(1.0, timeout_seconds), creationflags=_no_window())
    if result.returncode or result.stdout.strip() != "CPU_PREFLIGHT_PASS":
        raise RuntimeError(f"worker preflight failed rc={result.returncode} stdout={result.stdout[-1000:]} stderr={result.stderr[-2000:]}")
    return {"status": "CPU_PREFLIGHT_PASS"}


def inspect_resources(plan: dict, root: Path):
    free = shutil.disk_usage(root).free
    if free < int(plan["min_free_bytes"]):
        raise RuntimeError(f"disk reserve insufficient: {free}")
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15, creationflags=_no_window())
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values or max(values) < int(plan["min_gpu_free_mib"]):
        raise RuntimeError(f"GPU memory reserve insufficient: {values}")
    return {"disk_free_bytes": free, "gpu_free_mib": max(values)}


def write_stop(control: Path, reason: str):
    control.mkdir(parents=True, exist_ok=True)
    stop = control / "STOP"
    if not stop.exists():
        atomic_text(stop, reason + "\n")


def verify_exit(child, worker, plan_path, output, root, actual, *, terminate=False):
    identity_error = None
    actual_after = None
    if actual is not None:
        try:
            actual_after = process_identity(actual["ProcessId"], child.pid, worker, plan_path, output)
        except Exception as exc:
            identity_error = repr(exc)
        if actual_after is not None and terminate:
            subprocess.run(["taskkill.exe", "/PID", str(actual["ProcessId"]), "/T", "/F"], check=False,
                capture_output=True, text=True, timeout=3, creationflags=_no_window())
            try:
                actual_after = process_identity(actual["ProcessId"], child.pid, worker, plan_path, output)
            except Exception as exc:
                identity_error = repr(exc)
    worker_path = str(worker.resolve()).lower().replace("\\", "/")
    plan_text = str(plan_path.resolve()).lower().replace("\\", "/")
    output_text = str(output.resolve()).lower().replace("\\", "/")
    def owned_matches():
        return [item for item in process_inventory(root) if worker_path in
            (item.get("CommandLine") or "").lower().replace("\\", "/") and plan_text in
            (item.get("CommandLine") or "").lower().replace("\\", "/") and output_text in
            (item.get("CommandLine") or "").lower().replace("\\", "/")]
    matching = owned_matches()
    if terminate and matching:
        for item in matching:
            if int(item.get("ParentProcessId", -1)) == child.pid:
                subprocess.run(["taskkill.exe", "/PID", str(item["ProcessId"]), "/T", "/F"], check=False,
                    capture_output=True, text=True, timeout=3, creationflags=_no_window())
        matching = owned_matches()
    return {"proxy_exit_code": child.poll(), "actual_process_after_exit": actual_after,
        "identity_error": identity_error, "lingering_owned_processes": matching,
        "all_owned_processes_gone": child.poll() is not None and actual_after is None
            and identity_error is None and not matching}


def cleanup_owned_after_exception(child, control, worker, plan_path, output, root, actual, reason):
    """Cooperatively stop and verify the exact Popen-owned subtree after monitor failure."""
    stop_error = None
    try:
        write_stop(control, reason)
    except Exception as exc:
        stop_error = repr(exc)
    if child.poll() is None:
        try:
            subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"], check=False,
                capture_output=True, text=True, timeout=4, creationflags=_no_window())
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        except Exception:
            pass
    try:
        exit_info = verify_exit(child, worker, plan_path, output, root, actual, terminate=True)
    except Exception as verify_error:
        exit_info = {"proxy_exit_code": child.poll(), "actual_process_after_exit": "UNCONFIRMED",
            "identity_error": repr(verify_error), "lingering_owned_processes": "UNCONFIRMED",
            "all_owned_processes_gone": False}
    return {"status": "STOPPED" if exit_info.get("all_owned_processes_gone") else "UNCONFIRMED_EXIT",
        "stop_reason": reason, "cooperative_stop_write_error": stop_error, "exit_verification": exit_info}


def supervise(plan_path: Path, worker: Path, output: Path):
    invocation_start = time.time()
    deadline, worker_deadline = fixed_deadlines(invocation_start)
    plan_path, worker, output = plan_path.resolve(), worker.resolve(), output.resolve()
    plan, root = validate_plan(plan_path, worker, output)
    if plan.get("authorized") is not True:
        raise ValueError("draft is not authorized; no inference launch")
    timeout = min(55.0, worker_deadline - time.time() - 1)
    if timeout <= 1:
        raise TimeoutError("preflight consumed the reserved worker window")
    cpu = worker_preflight(plan_path, worker, output, timeout)
    lock_path = Path(os.getenv("TEMP", str(root))) / ("fdt_align_transfer_" + hashlib.sha256(str(root).lower().encode()).hexdigest()[:16] + ".lock")
    lock = FileLock(str(lock_path), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise RuntimeError("another alignment transfer supervisor holds the operation lock") from exc
    try:
        resources = inspect_resources(plan, root)
        duplicates = process_inventory(root)
        if duplicates:
            raise RuntimeError(f"another FDT worker/readonly diagnostic is active: {duplicates}")
        if time.time() >= worker_deadline:
            raise TimeoutError("CPU preflight/resource checks exhausted the worker window")
        output.mkdir(parents=True, exist_ok=True)
        supervisor_dir = output / "supervisor"
        supervisor_dir.mkdir(exist_ok=True)
        state_path = supervisor_dir / "state.json"
        reserved = ("worker.stdout.log", "worker.stderr.log", "worker_progress.json", "diagnostic_result.json",
            "paired_seed17_noise.pt", "incident.json", "final_status.json")
        if state_path.exists() or any((output / name).exists() for name in reserved):
            raise FileExistsError("refusing to overwrite existing one-shot execution evidence")
        if (output / "model").exists() or (output / "checkpoint.pt").exists():
            raise FileExistsError("model/checkpoint output is prohibited")
        command = [sys.executable, "-B", "-X", "utf8", str(worker), "--plan", str(plan_path), "--output", str(output)]
        state = {"schema": "alignment_transfer_querycount_supervisor_v1", "status": "LAUNCHING",
            "launch_count": 1, "invocation_started_unix": invocation_start, "deadline_unix": deadline,
            "worker_deadline_unix": worker_deadline, "shutdown_reserve_seconds": SHUTDOWN_RESERVE_SECONDS,
            "supervisor_pid": os.getpid(), "worker_proxy_pid": None, "worker_actual_process": None,
            "command": command, "worker_sha256": plan["worker_sha256"],
            "supervisor_sha256": plan["supervisor_sha256"], "parent_payload_sha256": plan["parent_payload_sha256"],
            "worker_preflight": cpu, "resources_preflight": resources,
            "optimizer_allowed": False, "checkpoint_write_allowed": False}
        atomic_json(state_path, state)
        with (output / "worker.stdout.log").open("xb") as stdout, (output / "worker.stderr.log").open("xb") as stderr:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                shell=False, creationflags=_no_window())
            actual, stop_reason = None, None
            try:
              state.update({"status": "RUNNING", "worker_proxy_pid": child.pid, "worker_started_unix": time.time()})
              atomic_json(state_path, state)
              while True:
                progress = load_json(output / "worker_progress.json")
                if isinstance(progress, dict) and isinstance(progress.get("worker_pid"), int):
                    actual = process_identity(progress["worker_pid"], child.pid, worker, plan_path, output)
                if shutil.disk_usage(root).free < plan["min_free_bytes"]:
                    stop_reason = "disk_free_below_minimum"
                elif isinstance(progress, dict) and progress.get("anomaly"):
                    stop_reason = "worker_integrity_or_numerical_anomaly: " + str(progress["anomaly"])
                if time.time() >= worker_deadline and child.poll() is None:
                    stop_reason = stop_reason or "absolute_worker_deadline_shutdown_reserve"
                atomic_json(output / "supervisor/latest.json", {"status": "STOPPING" if stop_reason else "RUNNING",
                    "observed_unix": time.time(), "progress": progress, "actual_process": actual,
                    "proxy_alive": child.poll() is None, "stop_reason": stop_reason})
                code = child.poll()
                if code is not None:
                    exit_info = verify_exit(child, worker, plan_path, output, root, actual)
                    result_present = (output / "diagnostic_result.json").exists()
                    status = natural_terminal_status(exit_info, code, result_present, stop_reason)
                    final = {"status": status, "launch_count": 1, "worker_exit_code": code,
                        "diagnostic_result_present": result_present, "exit_verification": exit_info,
                        "elapsed_seconds": time.time() - invocation_start}
                    atomic_json(output / "supervisor/final.json", final)
                    atomic_json(output / "final_status.json", final)
                    return 0 if status == "COMPLETE" else 1
                if stop_reason:
                    write_stop(output / "control", stop_reason)
                    grace_end = min(deadline, time.time() + STOP_GRACE_SECONDS)
                    while child.poll() is None and time.time() < grace_end:
                        time.sleep(min(.4, max(0, grace_end - time.time())))
                    if child.poll() is None:
                        subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"], check=False,
                            capture_output=True, text=True, timeout=4, creationflags=_no_window())
                        try:
                            child.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            pass
                    exit_info = verify_exit(child, worker, plan_path, output, root, actual, terminate=True)
                    status = "STOPPED" if exit_info["all_owned_processes_gone"] else "UNCONFIRMED_EXIT"
                    final = {"status": status, "launch_count": 1, "worker_exit_code": child.poll(),
                        "stop_reason": stop_reason, "exit_verification": exit_info,
                        "elapsed_seconds": time.time() - invocation_start}
                    atomic_json(output / "supervisor/final.json", final)
                    atomic_json(output / "final_status.json", final)
                    return 1
                if time.time() >= deadline:
                    stop_reason = "absolute_total_wall_deadline"
                    write_stop(output / "control", stop_reason)
                    subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"], check=False,
                        capture_output=True, text=True, timeout=4, creationflags=_no_window())
                    try:
                        child.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
                    exit_info = verify_exit(child, worker, plan_path, output, root, actual, terminate=True)
                    status = "STOPPED" if exit_info["all_owned_processes_gone"] else "UNCONFIRMED_EXIT"
                    final = {"status": status, "launch_count": 1, "worker_exit_code": child.poll(),
                        "stop_reason": stop_reason, "exit_verification": exit_info,
                        "elapsed_seconds": time.time() - invocation_start}
                    atomic_json(output / "supervisor/final.json", final)
                    atomic_json(output / "final_status.json", final)
                    return 1
                time.sleep(POLL_SECONDS)
            except BaseException as exc:
                reason = f"owned_supervisor_exception:{type(exc).__name__}:{exc}"
                cleanup = cleanup_owned_after_exception(child, output / "control", worker, plan_path,
                    output, root, actual, reason)
                final = {**cleanup, "launch_count": 1, "worker_exit_code": child.poll(),
                    "exception": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": time.time() - invocation_start}
                atomic_json(output / "supervisor/final.json", final)
                atomic_json(output / "final_status.json", final)
                return 1
    finally:
        lock.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate", action="store_true", help="CPU-only; no resource query or launch")
    args = parser.parse_args()
    try:
        if args.validate:
            plan, _ = validate_plan(args.plan.resolve(), args.worker.resolve(), args.output.resolve())
            preflight = worker_preflight(args.plan.resolve(), args.worker.resolve(), args.output.resolve(), 100)
            print(json.dumps({"status": "CPU_PREFLIGHT_OK_NO_GPU_NO_LAUNCH", "worker_preflight": preflight,
                "plan_sha256": sha256(args.plan.resolve()), "authorized": plan["authorized"]}, ensure_ascii=False))
            return 0
        return supervise(args.plan, args.worker, args.output)
    except Exception as exc:
        print(f"alignment transfer supervisor error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
