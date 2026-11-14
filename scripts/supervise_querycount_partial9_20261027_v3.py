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
import tempfile
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
    "run_query_interference_diagnostic_20261027.py", "run_alignment_transfer_querycount_20261027.py", "run_alignment_trajectory_20261027.py",
    "run_querycount_alignment_20261027.py",
    "run_querycount_alignment_20261027_recovery_v2.py", "evaluate_querycount_partial9_20261027.py",
    "run_alignment_trajectory_20261027.py", "run_query_bound_alignment_20261027_v2.py",
    "run_query_mix_diffusion_20261027.py", "run_role8_diffusion_20261027.py",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run_native(args, **kwargs):
    """Use UTF-8 JSON from PowerShell and tolerate localized native-tool diagnostics."""
    args = list(args)
    if args and str(args[0]).lower().endswith("powershell.exe") and "-Command" in args:
        index = args.index("-Command") + 1
        args[index] = ("[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
                       "$OutputEncoding = [Console]::OutputEncoding; " + args[index])
    if kwargs.get("text"):
        kwargs.setdefault("encoding", "utf-8")
        kwargs.setdefault("errors", "replace")
    return subprocess.run(args, **kwargs)


def fixed_deadlines(immutable_absolute_deadline: float):
    absolute = float(immutable_absolute_deadline)
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
    result = run_native(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True, capture_output=True, text=True, timeout=6, creationflags=_no_window())
    items = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(items, dict):
        items = [items]
    paths = [str((root / "scripts" / name).resolve()).lower().replace("\\", "/") for name in ACTIVE_WORKERS]
    return [item for item in items if project in (item.get("CommandLine") or "").lower().replace("\\", "/")
        and is_active_worker_command(item.get("CommandLine") or "", paths)]


def owned_descendants(proxy_pid: int):
    """Fresh CIM snapshot of the proxy and every transitive child, regardless of command line."""
    script = "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    result = run_native(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
        check=True, capture_output=True, text=True, timeout=8, creationflags=_no_window())
    rows=json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(rows,dict): rows=[rows]
    descendants={int(proxy_pid)}; changed=True
    while changed:
        changed=False
        for row in rows:
            try: pid,parent=int(row.get("ProcessId",-1)),int(row.get("ParentProcessId",-1))
            except (TypeError,ValueError): continue
            if parent in descendants and pid not in descendants:
                descendants.add(pid); changed=True
    return [row for row in rows if int(row.get("ProcessId",-1)) in descendants]


def terminate_owned_descendants(proxy_pid: int):
    """Terminate only fresh-CIM-verified descendants of this Popen proxy."""
    rows=owned_descendants(proxy_pid)
    by_pid={int(row["ProcessId"]):row for row in rows if "ProcessId" in row}
    # Taskkill the verified proxy tree; descendants are included even if their command differs.
    for pid,row in sorted(by_pid.items(), key=lambda pair:int(pair[1].get("ParentProcessId",-1))==proxy_pid, reverse=True):
        if pid==proxy_pid: continue
        current=owned_descendants(proxy_pid)
        match=next((item for item in current if int(item.get("ProcessId",-1))==pid),None)
        if match is None: continue
        if (match.get("CreationDate")!=row.get("CreationDate") or
                int(match.get("ParentProcessId",-1))!=int(row.get("ParentProcessId",-2))):
            raise RuntimeError(f"owned descendant identity changed before termination: {pid}")
        run_native(["taskkill.exe","/PID",str(pid),"/T","/F"],check=False,capture_output=True,
            text=True,timeout=4,creationflags=_no_window())


def process_identity(pid: int, expected_parent: int, worker: Path, plan: Path, output: Path):
    command = ("$p=" + str(int(pid)) + "; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -eq $p } "
        "| Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress")
    result = run_native(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        check=True, capture_output=True, text=True, timeout=4, creationflags=_no_window())
    item = json.loads(result.stdout) if result.stdout.strip() else None
    if not isinstance(item, dict):
        return None
    # CIM can briefly return PID/parent/creation while command/path are already gone
    # during natural termination. Keep it as an unverified live row until absence
    # is observed; it is not a command mismatch and never proves exit by itself.
    if (int(item.get("ProcessId", -1)) == pid and
            int(item.get("ParentProcessId", -1)) == expected_parent and not item.get("CommandLine")):
        item["identity_verified"] = False
        item["identity_status"] = "PARTIAL_NATIVE_ROW"
        return item
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
    plan=load_json(plan_path); root=Path(__file__).resolve().parents[1]
    if not isinstance(plan,dict) or plan.get("schema")!="fdt_querycount_partial9_readonly_plan_v1": raise ValueError("partial9 assessment plan schema mismatch")
    if plan.get("authorized") not in (False,True): raise ValueError("authorization marker must be boolean")
    if not isinstance(plan.get("immutable_owner_deadline_unix"),(int,float)) or not isinstance(plan.get("worker_cutoff_unix"),(int,float)):
        raise ValueError("immutable expired-allocation timestamps are required")
    if float(plan["worker_cutoff_unix"])!=float(plan["immutable_owner_deadline_unix"])-SHUTDOWN_RESERVE_SECONDS:
        raise ValueError("worker cutoff must preserve the original shutdown reserve")
    if Path(plan["output_dir"]).resolve()!=output.resolve(): raise ValueError("output path mismatch")
    if Path(plan["control_dir"]).resolve()!=(output/"control").resolve(): raise ValueError("control directory must be output/control")
    if (plan.get("wall_seconds"),plan.get("shutdown_reserve_seconds"),plan.get("max_actual_encode_denoise_calls"),plan.get("expected_actual_calls"),plan.get("expected_readout_calls"))!=(120,30,720,700,280): raise ValueError("fixed 120s/720-call assessment contract mismatch")
    if plan.get("optimizer_allowed") is not False or plan.get("checkpoint_write_allowed") is not False or plan.get("e2_final_files_accessed") is not False: raise ValueError("assessment must remain readonly")
    if plan.get("min_free_bytes",0)<28*1024**3 or plan.get("min_gpu_free_mib",0)<12288 or plan.get("max_output_bytes")!=64*1024**2: raise ValueError("resource/output floors cannot be lowered")
    if plan.get("worker_sha256")!=sha256(worker.resolve()) or plan.get("supervisor_sha256")!=sha256(Path(__file__).resolve()): raise ValueError("worker/supervisor digest mismatch")
    for name,path in plan["dependencies"].items():
        if sha256(Path(path))!=plan["dependency_sha256"].get(name): raise ValueError(f"dependency SHA mismatch: {name}")
    if plan.get("runtime_source_sha256")!=plan.get("core_sha256"): raise ValueError("frozen source digest mismatch")
    if plan.get("checkpoint_payload_sha256")!="5494ec7c3cf9d8466d799d05fd0b782a101969dedd645678ecc52154dba93675": raise ValueError("wrong partial9 source payload")
    if set(plan.get("baseline_raw_paths",{}))!=set(plan.get("baseline_raw_sha256",{})) or len(plan["baseline_raw_paths"])!=14: raise ValueError("all fourteen baseline raw references must be pinned")
    allowed={"plan.json","readiness.json","code_snapshot","supervisor","supervisor.stdout.log","supervisor.stderr.log"}
    if output.exists():
        unexpected={x.name for x in output.iterdir()}-allowed
        if unexpected: raise FileExistsError(f"output already contains execution evidence: {sorted(unexpected)}")
    return plan,root

def worker_preflight(plan_path: Path, worker: Path, output: Path, timeout_seconds: float):
    command = [sys.executable, "-B", "-X", "utf8", str(worker.resolve()), "--plan", str(plan_path.resolve()),
        "--output", str(output.resolve()), "--preflight"]
    result = run_native(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, shell=False,
        timeout=max(1.0, timeout_seconds), creationflags=_no_window())
    if result.returncode or result.stdout.strip() != "CPU_PREFLIGHT_PASS":
        raise RuntimeError(f"worker preflight failed rc={result.returncode} stdout={result.stdout[-1000:]} stderr={result.stderr[-2000:]}")
    return {"status": "CPU_PREFLIGHT_PASS"}


def inspect_resources(plan: dict, root: Path):
    free = shutil.disk_usage(root).free
    if free < int(plan["min_free_bytes"]):
        raise RuntimeError(f"disk reserve insufficient: {free}")
    result = run_native(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15, creationflags=_no_window())
    values = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not values or max(values) < int(plan["min_gpu_free_mib"]):
        raise RuntimeError(f"GPU memory reserve insufficient: {values}")
    return {"disk_free_bytes": free, "gpu_free_mib": max(values)}


def owned_output_bytes(output: Path) -> int:
    return sum(p.stat().st_size for p in output.rglob("*") if p.is_file() and not p.is_symlink())


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
            if child.poll() is not None:
                # Bounded settling for a native exit row; continued presence remains
                # UNCONFIRMED and never turns into fabricated exit confirmation.
                for _ in range(2):
                    if actual_after is None:
                        break
                    time.sleep(.05)
                    actual_after = process_identity(actual["ProcessId"], child.pid, worker, plan_path, output)
        except Exception as exc:
            identity_error = repr(exc)
        if actual_after is not None and terminate:
            run_native(["taskkill.exe", "/PID", str(actual["ProcessId"]), "/T", "/F"], check=False,
                capture_output=True, text=True, timeout=3, creationflags=_no_window())
            try:
                actual_after = process_identity(actual["ProcessId"], child.pid, worker, plan_path, output)
            except Exception as exc:
                identity_error = repr(exc)
    inventory_error=None
    try:
        descendants=owned_descendants(child.pid)
        if terminate and any(int(item.get("ProcessId",-1))!=child.pid for item in descendants):
            terminate_owned_descendants(child.pid)
            descendants=owned_descendants(child.pid)
    except Exception as exc:
        descendants=None; inventory_error=repr(exc)
    lingering=[item for item in descendants or [] if int(item.get("ProcessId",-1))!=child.pid]
    return {"proxy_exit_code": child.poll(), "actual_process_after_exit": actual_after,
        "identity_error": identity_error,"descendant_inventory_error":inventory_error,
        "owned_descendants_after_exit":descendants,"lingering_owned_processes":lingering,
        "all_owned_processes_gone": child.poll() is not None and actual_after is None
            and identity_error is None and inventory_error is None and not lingering}


def cleanup_owned_after_exception(child, control, worker, plan_path, output, root, actual, reason):
    """Cooperatively stop and verify the exact Popen-owned subtree after monitor failure."""
    stop_error = None
    try:
        write_stop(control, reason)
    except Exception as exc:
        stop_error = repr(exc)
    if child.poll() is None:
        try:
            run_native(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"], check=False,
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
    plan_path, worker, output = plan_path.resolve(), worker.resolve(), output.resolve()
    plan, root = validate_plan(plan_path, worker, output)
    deadline, worker_deadline = fixed_deadlines(plan["immutable_owner_deadline_unix"])
    if invocation_start>=deadline:
        raise TimeoutError("the original one-shot allocation deadline has expired")
    if invocation_start>=worker_deadline:
        raise TimeoutError("the original worker cutoff has expired")
    if plan.get("authorized") is not True:
        raise ValueError("draft is not authorized; no inference launch")
    timeout = min(55.0, worker_deadline - time.time() - 1)
    if timeout <= 1:
        raise TimeoutError("preflight consumed the reserved worker window")
    cpu = worker_preflight(plan_path, worker, output, timeout)
    lock_hash=hashlib.sha256(str(root).lower().encode()).hexdigest()[:20]
    lock_path=Path(tempfile.gettempdir())/f"fdt_local_supervisor_{lock_hash}.lock"
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
        state = {"schema": "querycount_partial9_readonly_supervisor_v1", "status": "LAUNCHING",
            "launch_count": 1, "invocation_started_unix": invocation_start, "deadline_unix": deadline,
            "worker_deadline_unix": worker_deadline, "shutdown_reserve_seconds": SHUTDOWN_RESERVE_SECONDS,
            "supervisor_pid": os.getpid(), "worker_proxy_pid": None, "worker_actual_process": None,
            "command": command, "worker_sha256": plan["worker_sha256"],
            "supervisor_sha256": plan["supervisor_sha256"], "checkpoint_payload_sha256": plan["checkpoint_payload_sha256"],
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
                try: progress = load_json(output / "worker_progress.json")
                except OSError: progress = None
                if child.poll() is None and isinstance(progress, dict) and isinstance(progress.get("worker_pid"), int):
                    actual = process_identity(progress["worker_pid"], child.pid, worker, plan_path, output)
                if shutil.disk_usage(root).free < plan["min_free_bytes"]:
                    stop_reason = "disk_free_below_minimum"
                elif owned_output_bytes(output) > plan["max_output_bytes"]:
                    stop_reason = "owned_output_exceeded_64MiB"
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
                        run_native(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"], check=False,
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
                    run_native(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"], check=False,
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
