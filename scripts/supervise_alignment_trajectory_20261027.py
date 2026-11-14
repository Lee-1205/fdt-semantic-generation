"""Single-launch bounded owner for the passive alignment trajectory trace."""
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

TOTAL_WALL_SECONDS = 90
SHUTDOWN_RESERVE_SECONDS = 20
STOP_GRACE_SECONDS = 5
ACTIVE_WORKERS = (
    "run_learning_102m.py", "run_learning_102m_recovery_v2.py", "run_e2_grounding_20261026.py",
    "run_role8_diffusion_20261027.py", "run_active_balanced_diffusion_20261027.py",
    "run_active_balanced_diffusion_20261027_v3.py", "run_query_mix_diffusion_20261027.py",
    "run_query_bound_alignment_20261027.py", "run_query_bound_alignment_20261027_v2.py",
    "run_query_interference_diagnostic_20261027.py", "run_alignment_transfer_querycount_20261027.py",
    "run_alignment_trajectory_20261027.py",
)


def sha256(path: Path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def is_active_worker_command(command, active_paths):
    return invoked_script(command) in set(active_paths)


def fixed_deadlines(started):
    absolute = started + TOTAL_WALL_SECONDS
    return absolute, absolute - SHUTDOWN_RESERVE_SECONDS


def natural_terminal_status(exit_info, exit_code, result_present, stop_reason):
    if not exit_info.get("all_owned_processes_gone", False):
        return "UNCONFIRMED_EXIT"
    return "COMPLETE" if exit_code == 0 and result_present and not stop_reason else "FAILED"


def process_inventory(root: Path):
    project = str(root.resolve()).lower().replace("\\", "/")
    script = "$self=$PID; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $self -and $_.Name -match '^python(w)?\\.exe$' } | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True, capture_output=True, text=True, timeout=6,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    entries = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(entries, dict):
        entries = [entries]
    active_paths = [str((root / "scripts" / name).resolve()).lower().replace("\\", "/") for name in ACTIVE_WORKERS]
    return [item for item in entries if project in (item.get("CommandLine") or "").lower().replace("\\", "/")
        and is_active_worker_command(item.get("CommandLine") or "", active_paths)]


def process_identity(pid, parent_pid, worker, plan_path, output):
    script = ("$p=" + str(int(pid)) + "; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -eq $p } "
        "| Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True, capture_output=True, text=True, timeout=4,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    item = json.loads(result.stdout) if result.stdout.strip() else None
    if not isinstance(item, dict):
        return None
    cmd = (item.get("CommandLine") or "").lower().replace("\\", "/")
    ok = (int(item.get("ProcessId", -1)) == int(pid) and int(item.get("ParentProcessId", -1)) == int(parent_pid)
        and str(worker.resolve()).lower().replace("\\", "/") in cmd
        and str(plan_path.resolve()).lower().replace("\\", "/") in cmd
        and str(output.resolve()).lower().replace("\\", "/") in cmd)
    if not ok:
        raise RuntimeError(f"owned trajectory worker identity mismatch: {item}")
    item["identity_verified"] = True
    return item


def validate_plan(plan_path: Path, worker: Path, output: Path):
    plan = load_json(plan_path)
    root = Path(__file__).resolve().parents[1]
    if not isinstance(plan, dict) or plan.get("schema") != "fdt_alignment_trajectory_readonly_v1":
        raise ValueError("trajectory plan schema mismatch")
    if Path(plan["output_dir"]).resolve() != output.resolve():
        raise ValueError("output path mismatch")
    if plan.get("authorized") not in (False, True):
        raise ValueError("authorized must be a boolean")
    if (plan.get("wall_seconds"), plan.get("shutdown_reserve_seconds"), plan.get("max_actual_encode_denoise_calls"),
            plan.get("trace_cap_bytes"), plan.get("graph_indices"), plan.get("query_counts"), plan.get("seed")) != (
            90, 20, 40, 128 * 1024 * 1024, [8, 9, 10, 11], [4, 8, 16], 17):
        raise ValueError("fixed trajectory budget/case mismatch")
    if plan.get("optimizer_allowed") is not False or plan.get("checkpoint_write_allowed") is not False or plan.get("e2_final_files_accessed") is not False:
        raise ValueError("optimizer/checkpoint/E2-final must remain prohibited")
    if int(plan.get("min_free_bytes", 0)) < 20 * 1024**3 or int(plan.get("min_gpu_free_mib", 0)) < 12288:
        raise ValueError("minimum launch resources cannot be lowered")
    deps = {"worker": worker.resolve(), "supervisor": Path(__file__).resolve(),
        "config": Path(plan["config_path"]), "io": root / "scripts/learning_io_v2.py",
        "common_worker": root / "scripts/run_e2_grounding_20261026.py",
        "builder": root / "research/alignment_transfer_querycount_v1.py",
        "query_bound_helper": root / "research/query_bound_refiner_v1.py",
        "model_config": root / "src/fdt_local/model_masked.py", "data_source": root / "src/fdt_local/data.py",
        "parent_marker": Path(plan["parent_marker"]), "dataset_manifest": Path(plan["dataset_dir"]) / "manifest.json",
        "selection_registry": Path(plan["dataset_dir"]) / "selection_registry.json",
        "source_transfer_plan": Path(plan["source_transfer_plan_path"]),
        "source_transfer_result": Path(plan["source_transfer_result_path"]),
        "source_noise": Path(plan["source_noise_path"])}
    mismatch = {k: (plan.get("dependency_sha256", {}).get(k), sha256(v))
        for k, v in deps.items() if plan.get("dependency_sha256", {}).get(k) != sha256(v)}
    if mismatch:
        raise ValueError(f"trajectory source/data SHA mismatch: {mismatch}")
    if plan.get("worker_sha256") != sha256(worker.resolve()) or plan.get("supervisor_sha256") != sha256(Path(__file__).resolve()):
        raise ValueError("worker/supervisor SHA mismatch")
    if plan.get("runtime_source_sha256") != plan.get("core_sha256"):
        raise ValueError("frozen core/runtime SHA mismatch")
    existing_allowed = {"plan.json", "readiness.json", "seal.json", "code_snapshot",
        "supervisor.stdout.log", "supervisor.stderr.log"}
    if output.exists():
        unexpected = {p.name for p in output.iterdir()} - existing_allowed
        if unexpected:
            raise FileExistsError(f"output is not a fresh one-shot directory: {sorted(unexpected)}")
    return plan, root


def worker_preflight(plan_path, worker, output, timeout):
    command = [sys.executable, "-B", "-X", "utf8", str(worker.resolve()), "--plan", str(plan_path.resolve()),
        "--output", str(output.resolve()), "--preflight"]
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True,
        shell=False, timeout=max(1.0, timeout), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode or result.stdout.strip() != "CPU_PREFLIGHT_PASS":
        raise RuntimeError(f"worker CPU preflight failed rc={result.returncode} stdout={result.stdout[-1000:]} stderr={result.stderr[-2000:]}")
    return {"status": "CPU_PREFLIGHT_PASS"}


def inspect_resources(plan, root):
    free = shutil.disk_usage(root).free
    if free < plan["min_free_bytes"]:
        raise RuntimeError(f"C: free space below reserved amount: {free}")
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values or max(values) < plan["min_gpu_free_mib"]:
        raise RuntimeError(f"GPU free memory below required reserve: {values}")
    return {"disk_free_bytes": free, "gpu_free_mib": max(values)}


def write_stop(control, reason):
    control.mkdir(parents=True, exist_ok=True)
    path = control / "STOP"
    if not path.exists():
        atomic_text(path, reason + "\n")


def verify_exit(child, worker, plan_path, output, root, actual, *, terminate=False):
    error, actual_after = None, None
    if actual is not None:
        try:
            actual_after = process_identity(actual["ProcessId"], child.pid, worker, plan_path, output)
        except Exception as exc:
            error = repr(exc)
        if actual_after is not None and terminate:
            subprocess.run(["taskkill.exe", "/PID", str(actual["ProcessId"]), "/T", "/F"], check=False,
                capture_output=True, text=True, timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                actual_after = process_identity(actual["ProcessId"], child.pid, worker, plan_path, output)
            except Exception as exc:
                error = repr(exc)
    wtxt = str(worker.resolve()).lower().replace("\\", "/")
    ptxt = str(plan_path.resolve()).lower().replace("\\", "/")
    otxt = str(output.resolve()).lower().replace("\\", "/")
    def matches():
        return [item for item in process_inventory(root) if wtxt in (item.get("CommandLine") or "").lower().replace("\\", "/")
            and ptxt in (item.get("CommandLine") or "").lower().replace("\\", "/")
            and otxt in (item.get("CommandLine") or "").lower().replace("\\", "/")]
    lingering = matches()
    if terminate and lingering:
        for item in lingering:
            if int(item.get("ParentProcessId", -1)) == child.pid:
                subprocess.run(["taskkill.exe", "/PID", str(item["ProcessId"]), "/T", "/F"], check=False,
                    capture_output=True, text=True, timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lingering = matches()
    return {"proxy_exit_code": child.poll(), "actual_process_after_exit": actual_after,
        "identity_error": error, "lingering_owned_processes": lingering,
        "all_owned_processes_gone": child.poll() is not None and actual_after is None and error is None and not lingering}


def cleanup_owned_after_exception(child, control, worker, plan_path, output, root, actual, reason):
    stop_error = None
    try:
        write_stop(control, reason)
    except Exception as exc:
        stop_error = repr(exc)
    if child.poll() is None:
        try:
            subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"], check=False,
                capture_output=True, text=True, timeout=4, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        except Exception:
            pass
    try:
        exit_info = verify_exit(child, worker, plan_path, output, root, actual, terminate=True)
    except Exception as exc:
        exit_info = {"proxy_exit_code": child.poll(), "actual_process_after_exit": "UNCONFIRMED",
            "identity_error": repr(exc), "lingering_owned_processes": "UNCONFIRMED",
            "all_owned_processes_gone": False}
    return {"status": "STOPPED" if exit_info["all_owned_processes_gone"] else "UNCONFIRMED_EXIT",
        "stop_reason": reason, "cooperative_stop_write_error": stop_error, "exit_verification": exit_info}


def supervise(plan_path: Path, worker: Path, output: Path):
    invocation_start = time.time()
    absolute_deadline, worker_deadline = fixed_deadlines(invocation_start)
    plan_path, worker, output = plan_path.resolve(), worker.resolve(), output.resolve()
    plan, root = validate_plan(plan_path, worker, output)
    if plan.get("authorized") is not True:
        raise ValueError("draft plan is not authorized; no inference launch")
    remaining = worker_deadline - time.time()
    if remaining <= 1:
        raise TimeoutError("preflight consumed the fixed worker time")
    cpu = worker_preflight(plan_path, worker, output, min(45.0, remaining - 1))
    lock_path = Path(os.getenv("TEMP", str(root))) / ("fdt_alignment_trajectory_" + hashlib.sha256(str(root).lower().encode()).hexdigest()[:16] + ".lock")
    lock = FileLock(str(lock_path), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise RuntimeError("another FDT operation owns the local lock") from exc
    try:
        resources = inspect_resources(plan, root)
        duplicates = process_inventory(root)
        if duplicates:
            raise RuntimeError(f"active FDT worker/readonly diagnostic exists: {duplicates}")
        if time.time() >= worker_deadline:
            raise TimeoutError("preflight/resource checks used the bounded window")
        output.mkdir(parents=True, exist_ok=True)
        supervisor_dir = output / "supervisor"
        supervisor_dir.mkdir(exist_ok=True)
        state_path = supervisor_dir / "state.json"
        reserved = ("worker.stdout.log", "worker.stderr.log", "worker_progress.json", "trajectory_trace.pt",
            "diagnostic_result.json", "incident.json", "final_status.json")
        if state_path.exists() or any((output / name).exists() for name in reserved):
            raise FileExistsError("refusing to overwrite trajectory run evidence")
        if (output / "model").exists() or (output / "checkpoint.pt").exists():
            raise FileExistsError("trajectory diagnostic cannot create model/checkpoint")
        command = [sys.executable, "-B", "-X", "utf8", str(worker), "--plan", str(plan_path), "--output", str(output)]
        state = {"schema": "alignment_trajectory_owned_supervisor_v1", "status": "LAUNCHING",
            "launch_count": 1, "invocation_started_unix": invocation_start,
            "deadline_unix": absolute_deadline, "worker_deadline_unix": worker_deadline,
            "shutdown_reserve_seconds": SHUTDOWN_RESERVE_SECONDS, "supervisor_pid": os.getpid(),
            "worker_proxy_pid": None, "worker_actual_process": None, "command": command,
            "worker_sha256": plan["worker_sha256"], "supervisor_sha256": plan["supervisor_sha256"],
            "parent_payload_sha256": plan["parent_payload_sha256"], "worker_preflight": cpu,
            "resources_preflight": resources, "optimizer_allowed": False, "checkpoint_write_allowed": False}
        atomic_json(state_path, state)
        with (output / "worker.stdout.log").open("xb") as stdout, (output / "worker.stderr.log").open("xb") as stderr:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            actual = None
            try:
                state.update({"status": "RUNNING", "worker_proxy_pid": child.pid, "worker_started_unix": time.time()})
                atomic_json(state_path, state)
                stop_reason = None
                while True:
                    progress = load_json(output / "worker_progress.json")
                    if isinstance(progress, dict) and isinstance(progress.get("worker_pid"), int):
                        actual = process_identity(progress["worker_pid"], child.pid, worker, plan_path, output)
                    if shutil.disk_usage(root).free < plan["min_free_bytes"]:
                        stop_reason = "disk_free_below_required_reserve"
                    elif isinstance(progress, dict) and progress.get("anomaly"):
                        stop_reason = "trace_integrity_or_numeric_anomaly: " + str(progress["anomaly"])
                    if time.time() >= worker_deadline and child.poll() is None:
                        stop_reason = stop_reason or "worker_deadline_shutdown_reserve"
                    if (output / "trajectory_trace.pt").exists() and (output / "trajectory_trace.pt").stat().st_size >= plan["trace_cap_bytes"]:
                        stop_reason = "trajectory_trace_size_cap"
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
                        grace_end = min(absolute_deadline, time.time() + STOP_GRACE_SECONDS)
                        while child.poll() is None and time.time() < grace_end:
                            time.sleep(min(.4, max(0, grace_end - time.time())))
                        if child.poll() is None:
                            subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"], check=False,
                                capture_output=True, text=True, timeout=4, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
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
                    if time.time() >= absolute_deadline:
                        stop_reason = "absolute_total_wall_deadline"
                        write_stop(output / "control", stop_reason)
                        subprocess.run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"], check=False,
                            capture_output=True, text=True, timeout=4, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
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
                    time.sleep(1)
            except BaseException as exc:
                reason = f"owned_supervisor_exception:{type(exc).__name__}:{exc}"
                cleanup = cleanup_owned_after_exception(child, output / "control", worker, plan_path,
                    output, root, actual, reason)
                final = {**cleanup, "launch_count": 1, "worker_exit_code": child.poll(),
                    "exception": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.time() - invocation_start}
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
    parser.add_argument("--validate", action="store_true", help="CPU only; no GPU query or launch")
    args = parser.parse_args()
    try:
        if args.validate:
            plan, _ = validate_plan(args.plan.resolve(), args.worker.resolve(), args.output.resolve())
            worker_result = worker_preflight(args.plan.resolve(), args.worker.resolve(), args.output.resolve(), 100)
            print(json.dumps({"status": "CPU_PREFLIGHT_OK_NO_GPU_NO_LAUNCH", "worker_preflight": worker_result,
                "plan_sha256": sha256(args.plan.resolve()), "authorized": plan["authorized"]}, ensure_ascii=False))
            return 0
        return supervise(args.plan, args.worker, args.output)
    except Exception as exc:
        print(f"trajectory supervisor error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
