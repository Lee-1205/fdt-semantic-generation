"""Single-launch, bounded supervisor for the authorized local 102M run."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

from filelock import FileLock, Timeout
from learning_io_v2 import atomic_json, atomic_text, load_json

POLL_SECONDS = 5
GRACE_SECONDS = 120
CHECKPOINT_ESTIMATE = 2 * 1024**3
CHECKPOINT_SCHEMA = "fdt_local_checkpoint_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_digest(package_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package_dir.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def validate_plan(plan_path: Path, worker: Path, output: Path) -> tuple[dict, Path, Path]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    config = Path(plan["config_path"])
    run_dir = Path(plan["run_dir"])
    if not config.is_absolute() or not run_dir.is_absolute():
        raise ValueError("plan config_path and run_dir must be absolute paths")
    if not worker.is_absolute():
        worker = worker.resolve()
    if not output.is_absolute():
        output = output.resolve()
    plan_output = Path(plan["output_dir"])
    if not plan_output.is_absolute() or plan_output.resolve() != output.resolve():
        raise ValueError("plan output_dir must be absolute and match --output")
    try:
        relative_run = run_dir.resolve().relative_to(output.resolve())
    except ValueError as exc:
        raise ValueError("run_dir must be inside output_dir") from exc
    if not relative_run.parts:
        raise ValueError("run_dir must be a child of output_dir")
    expected = {
        "worker_sha256": sha256(worker),
        "supervisor_sha256": sha256(Path(__file__).resolve()),
        "io_sha256": sha256(Path(__file__).resolve().parent / "learning_io_v2.py"),
        "runtime_source_sha256": runtime_digest(root / "src" / "fdt_local"),
        "config_sha256": sha256(config),
    }
    mismatches = {key: (plan.get(key), value) for key, value in expected.items() if plan.get(key) != value}
    if mismatches:
        raise ValueError(f"plan digest mismatch: {mismatches}")
    if not all(isinstance(plan.get(k), int) and plan[k] > 0 for k in
               ("wall_seconds", "max_attempts", "min_free_bytes", "min_gpu_free_mib", "stall_seconds", "save_stale_seconds")):
        raise ValueError("invalid plan limits")
    if plan["wall_seconds"] > 7200 or plan["max_attempts"] > 1500:
        raise ValueError("plan exceeds authorized wall/attempt limits")
    attempts_at_start = plan.get("attempts_at_start")
    if not isinstance(attempts_at_start, int) or attempts_at_start < 0:
        raise ValueError("recovery plan requires nonnegative attempts_at_start")
    if attempts_at_start + plan["max_attempts"] > 1500:
        raise ValueError("recovery attempts exceed global 1500-attempt limit")
    if "target_successful_updates" in plan:
        successful_at_start = plan.get("successful_at_start")
        if not isinstance(successful_at_start, int) or successful_at_start < 0:
            raise ValueError("recovery plan requires nonnegative successful_at_start")
        if not isinstance(plan["target_successful_updates"], int) or plan["target_successful_updates"] <= 0:
            raise ValueError("invalid target_successful_updates")
        if successful_at_start + plan["target_successful_updates"] > 1440:
            raise ValueError("recovery successful updates exceed global 1440-update target")
    if plan.get("absolute_deadline_unix") is not None:
        absolute_deadline = float(plan["absolute_deadline_unix"])
        if not math.isfinite(absolute_deadline) or absolute_deadline <= 0:
            raise ValueError("invalid absolute_deadline_unix")
    if run_dir.exists():
        raise ValueError(f"run_dir must be a new, nonexistent path: {run_dir}")
    if output.exists() and any((output / name).exists() for name in ("STOP", "clock.json", "final.json", "final_status.json")):
        raise ValueError("output already contains terminal markers")
    if (run_dir / "STOP").exists():
        raise ValueError("run_dir already has STOP; refusing to remove or reuse it")
    return plan, run_dir, output


def inspect_resources(plan: dict, root: Path) -> dict:
    disk = shutil.disk_usage(root)
    floor = plan["min_free_bytes"]
    if disk.free < floor + 6 * 1024**3 + CHECKPOINT_ESTIMATE:
        raise RuntimeError(f"C: reserve too low: free={disk.free}, required={floor + 6 * 1024**3 + CHECKPOINT_ESTIMATE}")
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                check=True, capture_output=True, text=True, timeout=15, creationflags=_no_window())
        gpu_free = max(int(line.strip()) for line in result.stdout.splitlines() if line.strip())
    except Exception as exc:
        raise RuntimeError(f"cannot verify GPU memory with nvidia-smi: {exc}") from exc
    if gpu_free < plan["min_gpu_free_mib"]:
        raise RuntimeError(f"GPU free memory too low: {gpu_free} MiB")
    return {"disk_total": disk.total, "disk_free": disk.free, "gpu_free_mib": gpu_free}


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_existing_training(root: Path, supervisor_pid: int) -> list[dict]:
    venv = str((root / ".venv").resolve()).lower()
    worker_scripts = [str((root / "scripts" / name).resolve()).lower().replace("\\", "/")
                      for name in ("run_learning_102m.py", "run_learning_102m_v2.py")]
    supervisor_script = str(Path(__file__).resolve()).lower().replace("\\", "/")
    script = "$self=$PID; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -ne $self } | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    proc = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                          check=True, capture_output=True, text=True, timeout=20, creationflags=_no_window())
    try:
        items = json.loads(proc.stdout) if proc.stdout.strip() else []
    except ValueError as exc:
        raise RuntimeError("could not parse CIM process inventory") from exc
    if isinstance(items, dict):
        items = [items]
    found = []
    for item in items:
        if str(item.get("ProcessId")) == str(supervisor_pid):
            continue
        executable = (item.get("ExecutablePath") or "").lower()
        command = (item.get("CommandLine") or "").lower()
        normalized = command.replace("\\", "/")
        if supervisor_script in normalized:
            continue
        uses_project_venv = venv in executable.replace("/", "\\") or venv.replace("\\", "/") in normalized
        is_worker_script = any(path in normalized for path in worker_scripts)
        is_module_training = "-m fdt_local train" in normalized or "-m fdt_local resume" in normalized
        if uses_project_venv and (is_worker_script or is_module_training):
            found.append({"pid": item.get("ProcessId"), "command": item.get("CommandLine")})
    return found


def verify_marker(marker: Path, run_dir: Path) -> dict | None:
    meta = load_json(marker)
    if not isinstance(meta, dict):
        return None
    try:
        required = {"schema", "generation", "file", "bytes", "sha256", "contract", "run_id", "attempts", "updates"}
        if not required <= meta.keys() or meta["schema"] != CHECKPOINT_SCHEMA or meta.get("local_checkpoint_verified") is not True:
            return None
        if (not isinstance(meta["generation"], int) or not isinstance(meta["bytes"], int)
                or not isinstance(meta["contract"], dict) or not isinstance(meta["run_id"], str)
                or not isinstance(meta["attempts"], int) or not isinstance(meta["updates"], int)
                or not isinstance(meta["sha256"], str) or len(meta["sha256"]) != 64
                or any(ch not in "0123456789abcdef" for ch in meta["sha256"].lower())):
            return None
        name = meta["file"]
        if not isinstance(name, str) or Path(name).name != name or not name.startswith("payload_"):
            return None
        payload = run_dir / name
        if payload.stat().st_size != meta["bytes"] or sha256(payload) != meta["sha256"]:
            return None
        return meta
    except FileNotFoundError:
        return None
    except (ValueError, TypeError, KeyError):
        return None


def latest_durable(run_dir: Path, verified_cache: dict[str, dict]) -> tuple[dict | None, float | None]:
    newest = None
    newest_mtime = None
    for marker in run_dir.glob("complete_*.json"):
        try:
            current = load_json(marker)
            if current is None:
                continue
            cached = verified_cache.get(marker.name)
            # Rehash when marker metadata changes; cache only metadata whose payload passed SHA.
            if cached != current:
                meta = verify_marker(marker, run_dir)
                if meta is None:
                    verified_cache.pop(marker.name, None)
                    continue
                verified_cache[marker.name] = meta
            else:
                meta = cached
                payload = run_dir / str(meta.get("file", ""))
                if Path(payload).name != meta.get("file") or payload.stat().st_size != meta.get("bytes"):
                    verified_cache.pop(marker.name, None)
                    continue
        except FileNotFoundError:
            continue
        if meta and (newest is None or int(meta.get("generation", -1)) > int(newest["generation"])):
            try:
                newest, newest_mtime = meta, marker.stat().st_mtime
            except FileNotFoundError:
                continue
    age = None if newest_mtime is None else max(0.0, time.time() - newest_mtime)
    return newest, age


def write_stop(run_dir: Path, reason: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "STOP"
    if path.exists():
        return
    atomic_text(path, reason.strip() + "\n")


def evaluate_stop(plan: dict, now: float, deadline: float, progress: dict | None,
                  progress_change_at: float, durable: dict | None, durable_age: float | None,
                  disk_free: int) -> str | None:
    if disk_free < plan["min_free_bytes"] + CHECKPOINT_ESTIMATE:
        return "disk_reserve_below_20GiB_plus_2GiB_checkpoint"
    if now >= deadline:
        return "wall_clock_deadline"
    if now - progress_change_at > plan["stall_seconds"]:
        return "worker_progress_stalled"
    if durable is not None and durable_age is not None and durable_age > plan["save_stale_seconds"]:
        return "durable_checkpoint_stale"
    if progress and progress.get("anomaly"):
        return "worker_reported_anomaly: " + str(progress["anomaly"])
    if (progress and isinstance(progress.get("attempts"), int)
            and progress["attempts"] - int(plan.get("attempts_at_start", 0)) >= plan["max_attempts"]):
        return "maximum_attempt_budget_reached"
    if (progress and isinstance(progress.get("updates"), int)
            and "target_successful_updates" in plan
            and progress["updates"] > int(plan.get("successful_at_start", 0)) + plan["target_successful_updates"]):
        return "target_successful_update_budget_exceeded"
    return None


def deadline_unix_for_launch(launched_unix: float, wall_seconds: int,
                             absolute_deadline_unix: float | None = None) -> float:
    computed = launched_unix + wall_seconds
    return min(computed, float(absolute_deadline_unix)) if absolute_deadline_unix is not None else computed


def _state_write(state_path: Path, state: dict) -> None:
    state["observed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state_path, state)


def _write_incident(output: Path, record: dict) -> Path | None:
    """Best-effort immutable evidence if normal final-state replace fails."""
    incident_dir = output / "supervisor"
    try:
        incident_dir.mkdir(parents=True, exist_ok=True)
        target = incident_dir / f"incident_{time.time_ns()}.json"
        with target.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        return target
    except Exception as exc:
        print(f"could not preserve supervisor incident file: {exc}; record={record!r}", file=sys.stderr)
        return None


def _write_final(output: Path, supervisor_dir: Path, state: dict) -> None:
    errors = []
    for path, value in ((supervisor_dir / "state.json", state),
                         (supervisor_dir / "latest.json", {k: v for k, v in state.items() if k != "command"}),
                         (supervisor_dir / "final.json", state)):
        try:
            atomic_json(path, value)
        except Exception as exc:
            errors.append({"path": str(path), "error": repr(exc)})
    if errors:
        _write_incident(output, {"event": "final_state_write_error", "at": datetime.now(timezone.utc).isoformat(),
                                 "status": state.get("status"), "errors": errors,
                                 "worker_pid": state.get("worker_pid"),
                                 "worker_actual_process": state.get("worker_actual_process"),
                                 "last_durable": state.get("last_durable")})


def process_identity(pid: int, expected_parent: int, worker: Path, plan_path: Path, output: Path) -> dict | None:
    """Return fresh CIM evidence only when PID, parent and exact launch contract agree."""
    script = ("$p=" + str(int(pid)) + "; Get-CimInstance Win32_Process | Where-Object { $_.ProcessId -eq $p } "
              "| Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress")
    try:
        proc = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
                              check=True, capture_output=True, text=True, timeout=15, creationflags=_no_window())
        item = json.loads(proc.stdout) if proc.stdout.strip() else None
        if not isinstance(item, dict):
            return None
        command = (item.get("CommandLine") or "").lower().replace("\\", "/")
        if (int(item.get("ProcessId", -1)) != int(pid) or int(item.get("ParentProcessId", -1)) != int(expected_parent)
                or str(worker.resolve()).lower().replace("\\", "/") not in command
                or str(plan_path.resolve()).lower().replace("\\", "/") not in command
                or str(output.resolve()).lower().replace("\\", "/") not in command):
            item["identity_verified"] = False
            return item
        item["identity_verified"] = True
        return item
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"fresh CIM identity lookup failed for owned worker PID {pid}: {exc}") from exc


def terminate_owned_tree(child, *, run=subprocess.run, wait_seconds=10):
    """Terminate the process tree rooted at this live Popen-owned proxy only."""
    if child.poll() is not None:
        return child.poll(), False
    result = run(["taskkill.exe", "/PID", str(child.pid), "/T", "/F"], shell=False,
                 check=False, capture_output=True, text=True, timeout=wait_seconds,
                 creationflags=_no_window())
    try:
        code = child.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        return None, result.returncode == 0
    return code, result.returncode == 0


def supervise(plan_path: Path, worker: Path, output: Path, *, popen=subprocess.Popen,
              monotonic=time.monotonic, sleep=time.sleep, resource_check=inspect_resources,
              process_check=find_existing_training) -> int:
    plan, run_dir, output = validate_plan(plan_path.resolve(), worker, output)
    root = Path(__file__).resolve().parents[1]
    lock_hash = hashlib.sha256(str(root).lower().encode()).hexdigest()[:20]
    lock = FileLock(str(Path(tempfile.gettempdir()) / f"fdt_local_supervisor_{lock_hash}.lock"), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise RuntimeError("another FDT learning supervisor holds the global lock") from exc
    try:
        resources = resource_check(plan, root)
        existing = process_check(root, os.getpid())
        if existing:
            raise RuntimeError(f"existing project training Python process found: {existing}")
        output.mkdir(parents=True, exist_ok=True)
        supervisor_dir = output / "supervisor"
        supervisor_dir.mkdir(parents=True, exist_ok=True)
        state_path = supervisor_dir / "state.json"
        if state_path.exists() or (output / "worker.stdout.log").exists() or (output / "worker.stderr.log").exists():
            raise RuntimeError("output already appears to have been launched; refusing second launch")
        start = monotonic()
        launched_unix = time.time()
        deadline_unix = deadline_unix_for_launch(launched_unix, plan["wall_seconds"],
                                                 plan.get("absolute_deadline_unix"))
        deadline = start + max(0.0, deadline_unix - launched_unix)
        args = [sys.executable, "-B", "-X", "utf8", str(worker.resolve()), "--plan", str(plan_path.resolve()), "--output", str(output)]
        state = {"status": "LAUNCHING", "created_at": datetime.now(timezone.utc).isoformat(),
                 "launched_unix": launched_unix, "deadline_unix": deadline_unix,
                 "supervisor_pid": os.getpid(), "worker_pid": None, "deadline_monotonic": deadline,
                 "wall_seconds": plan["wall_seconds"], "worker_sha256": plan["worker_sha256"],
                 "supervisor_sha256": plan["supervisor_sha256"], "runtime_source_sha256": plan["runtime_source_sha256"],
                 "config_sha256": plan["config_sha256"], "io_sha256": plan["io_sha256"],
                 "resources_preflight": resources, "launch_count": 1,
                 "command": args}
        try:
            _state_write(state_path, state)
        except Exception as exc:
            _write_incident(output, {"event": "prelaunch_state_write_error", "error": repr(exc), "state": state})
            raise
        with (output / "worker.stdout.log").open("xb") as stdout, (output / "worker.stderr.log").open("xb") as stderr:
            child = popen(args, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                          shell=False, creationflags=_no_window())
            state["worker_pid"] = child.pid
            supervisor_io_error = None
            try:
                _state_write(state_path, state)
            except Exception as exc:
                supervisor_io_error = f"postlaunch_state_write_failed: {exc!r}"
                _write_incident(output, {"event": "postlaunch_state_write_error", "error": repr(exc),
                                         "worker_pid": child.pid, "command": args})
            progress_path = output / "worker_progress.json"
            previous_signature = None
            progress_change_at = start
            verified_markers: dict[str, dict] = {}
            actual_identity = None
            stop_started = None
            stop_reason = None
            while True:
                now = monotonic()
                try:
                    progress = load_json(progress_path)
                except OSError as exc:
                    progress = None
                    supervisor_io_error = f"progress_read_failed_after_retry: {exc!r}"
                if progress is not None and not isinstance(progress, dict):
                    supervisor_io_error = "worker_progress_json_not_object"
                    progress = None
                if progress is not None:
                    meaningful = {key: value for key, value in progress.items()
                                  if key.lower() not in {"timestamp", "observed_at", "observed_unix", "heartbeat", "updated_at"}}
                    signature = json.dumps(meaningful, sort_keys=True, ensure_ascii=False)
                else:
                    signature = None
                if signature != previous_signature:
                    progress_change_at = now
                    previous_signature = signature
                try:
                    durable, durable_age = latest_durable(run_dir, verified_markers)
                except OSError as exc:
                    durable, durable_age = state.get("last_durable"), state.get("last_durable_age_seconds")
                    supervisor_io_error = f"checkpoint_marker_read_failed_after_retry: {exc!r}"
                actual_pid = progress.get("worker_pid") if isinstance(progress, dict) else None
                if isinstance(actual_pid, int):
                    try:
                        found_identity = process_identity(actual_pid, child.pid, worker, plan_path, output)
                        if found_identity is not None:
                            actual_identity = found_identity
                    except RuntimeError as exc:
                        supervisor_io_error = f"owned_worker_identity_lookup_failed: {exc!r}"
                disk = shutil.disk_usage(root)
                reason = evaluate_stop(plan, now, deadline, progress, progress_change_at,
                                       durable, durable_age, disk.free)
                if supervisor_io_error:
                    reason = "supervisor_io_failstop: " + supervisor_io_error
                if reason and stop_reason is None:
                    stop_reason = reason
                    stop_started = now
                    try:
                        write_stop(run_dir, reason)
                    except Exception as exc:
                        supervisor_io_error = f"cooperative_stop_write_failed: {exc!r}"
                        _write_incident(output, {"event": "cooperative_stop_write_error", "error": repr(exc),
                                                 "worker_pid": child.pid, "reason": reason,
                                                 "last_durable": durable, "worker_actual_process": actual_identity})
                state.update({"status": "STOPPING" if stop_reason else "RUNNING", "stop_reason": stop_reason,
                              "worker_alive": child.poll() is None or actual_identity is not None,
                              "worker_proxy_alive": child.poll() is None, "worker_progress": progress,
                              "worker_actual_process": actual_identity,
                              "supervisor_io_error": supervisor_io_error,
                              "last_durable": durable, "last_durable_age_seconds": durable_age,
                              "disk_free_bytes": disk.free, "progress_last_changed_monotonic": progress_change_at})
                try:
                    atomic_json(supervisor_dir / "latest.json", {k: v for k, v in state.items() if k != "command"})
                    _state_write(state_path, state)
                except Exception as exc:
                    supervisor_io_error = f"supervisor_state_write_failed: {exc!r}"
                    if stop_reason is None:
                        stop_reason, stop_started = "supervisor_state_io_failstop", now
                        try:
                            write_stop(run_dir, stop_reason)
                        except Exception as stop_exc:
                            supervisor_io_error += f"; cooperative_STOP_failed: {stop_exc!r}"
                    _write_incident(output, {"event": "supervisor_state_write_error", "error": repr(exc),
                                             "worker_pid": child.pid, "worker_actual_process": actual_identity,
                                             "stop_reason": stop_reason, "last_durable": durable})
                code = child.poll()
                if code is not None:
                    worker_final = (load_json(output / "final_status.json") or
                                    load_json(run_dir / "final_status.json") or {})
                    final_status = worker_final.get("status") or (progress or {}).get("final_status")
                    actual_after = None
                    identity_check_error = None
                    if actual_identity and actual_identity.get("ProcessId") is not None:
                        try:
                            actual_after = process_identity(actual_identity["ProcessId"], child.pid, worker, plan_path, output)
                        except RuntimeError as exc:
                            identity_check_error = repr(exc)
                    if actual_after is not None or identity_check_error is not None:
                        status = "ERROR"
                    elif code == 0 and final_status == "COMPLETE" and not stop_reason:
                        status = "COMPLETE"
                    elif stop_reason or final_status == "STOPPED":
                        status = "STOPPED"
                    else:
                        status = "FAILED"
                    state.update({"status": status, "worker_final_status": final_status,
                                  "exit_code": code, "worker_alive": actual_after is not None or identity_check_error is not None,
                                  "worker_proxy_alive": False, "worker_actual_process_after_exit": actual_after,
                                  "worker_actual_identity_check_error": identity_check_error,
                                  "last_durable": durable,
                                  "last_durable_age_seconds": durable_age, "stop_reason": stop_reason,
                                  "finished_at": datetime.now(timezone.utc).isoformat()})
                    _write_final(output, supervisor_dir, state)
                    return code
                if stop_reason and now - stop_started >= GRACE_SECONDS:
                    # taskkill targets only the live Popen-owned proxy's process tree.
                    code, tree_killed = terminate_owned_tree(child)
                    actual_after = None
                    identity_check_error = None
                    if actual_identity and actual_identity.get("ProcessId") is not None:
                        try:
                            actual_after = process_identity(actual_identity["ProcessId"], child.pid, worker, plan_path, output)
                        except RuntimeError as exc:
                            identity_check_error = repr(exc)
                    state.update({"status": "STOPPED" if code is not None and actual_after is None and identity_check_error is None else "ERROR",
                                  "exit_code": code, "worker_alive": code is None or actual_after is not None or identity_check_error is not None,
                                  "worker_proxy_alive": code is None,
                                  "termination_after_grace": True, "finished_at": datetime.now(timezone.utc).isoformat(),
                                  "owned_tree_kill_succeeded": tree_killed, "worker_actual_process_after_kill": actual_after,
                                  "worker_actual_identity_check_error": identity_check_error,
                                  "last_durable": durable, "last_durable_age_seconds": durable_age})
                    _write_final(output, supervisor_dir, state)
                    return code if code is not None else 1
                sleep(POLL_SECONDS)
    finally:
        lock.release()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        return supervise(args.plan, args.worker, args.output)
    except Exception as exc:
        print(f"supervisor preflight/launch failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
