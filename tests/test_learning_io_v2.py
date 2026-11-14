import hashlib
import importlib.util
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import learning_io_v2 as io_v2


def load_supervisor():
    spec = importlib.util.spec_from_file_location("supervisor_v2", ROOT / "scripts" / "supervise_learning_102m_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def denied(winerror=5):
    exc = PermissionError(winerror, "sharing violation")
    exc.winerror = winerror
    return exc


def test_atomic_json_retries_transient_replace_denials_then_succeeds(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    real_replace = io_v2.os.replace
    calls = []
    clock = [0.0]
    def replace(src, dst):
        calls.append((src, dst))
        if len(calls) < 3:
            raise denied(32 if len(calls) == 1 else 5)
        return real_replace(src, dst)
    monkeypatch.setattr(io_v2.os, "replace", replace)
    io_v2.atomic_json(target, {"value": 7}, monotonic=lambda: clock[0],
                      sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    assert len(calls) == 3
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 7}
    assert not list(tmp_path.glob("state.json.*.tmp"))


def test_permanent_replace_denial_preserves_old_state_and_cleans_own_temp(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text('{"old": true}\n', encoding="utf-8")
    original = target.read_bytes()
    clock = [0.0]
    monkeypatch.setattr(io_v2.os, "replace", lambda *_: (_ for _ in ()).throw(denied()))
    with __import__("pytest").raises(PermissionError):
        io_v2.atomic_json(target, {"new": True}, monotonic=lambda: clock[0],
                          sleep=lambda seconds: clock.__setitem__(0, clock[0] + seconds), retry_seconds=0.08)
    assert target.read_bytes() == original
    assert not list(tmp_path.glob("state.json.*.tmp"))
    assert clock[0] >= 0.08


def test_shared_reader_survives_concurrent_atomic_replace(tmp_path):
    target = tmp_path / "progress.json"
    io_v2.atomic_json(target, {"sequence": 0})
    failures = []
    def writer():
        try:
            for n in range(1, 41):
                io_v2.atomic_json(target, {"sequence": n, "blob": "x" * 1024})
        except Exception as exc:
            failures.append(exc)
    thread = threading.Thread(target=writer)
    thread.start()
    read_count = 0
    while thread.is_alive():
        record = io_v2.load_json(target)
        if record is not None:
            assert isinstance(record["sequence"], int)
            assert record["sequence"] == 0 or record["blob"] == "x" * 1024
        read_count += 1
    thread.join()
    assert not failures and read_count > 0
    assert io_v2.load_json(target)["sequence"] == 40


def test_rotation_retries_payload_unlink_and_only_removes_old_verified_generation(tmp_path, monkeypatch):
    def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
    generations = []
    for n in range(3):
        payload = tmp_path / f"payload_{n:06d}.pt"
        payload.write_bytes(f"weight-{n}".encode())
        marker = tmp_path / f"complete_{n:06d}.json"
        meta = {"generation": n, "file": payload.name, "bytes": payload.stat().st_size,
                "sha256": sha(payload), "tag": None, "local_checkpoint_verified": True}
        marker.write_text(json.dumps(meta), encoding="utf-8")
        generations.append((n, marker, meta))
    class Store:
        directory = tmp_path
        keep = 2
        def markers(self): return list(reversed(generations))
    real_unlink = Path.unlink
    denied_count = [0]
    clock = [0.0]
    old_payload = tmp_path / "payload_000000.pt"
    def unlink(path, *args, **kwargs):
        if path == old_payload and denied_count[0] < 2:
            denied_count[0] += 1
            raise denied(32)
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(io_v2.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(io_v2.time, "sleep", lambda secs: clock.__setitem__(0, clock[0] + secs))
    io_v2.rotate_verified_generations(Store())
    assert denied_count[0] == 2
    assert not old_payload.exists()
    assert not (tmp_path / "complete_000000.json").exists()
    assert (tmp_path / "payload_000001.pt").exists() and (tmp_path / "payload_000002.pt").exists()
    assert (tmp_path / "rotation.jsonl").exists()


def test_absolute_deadline_clamps_wall_clock_and_remaining_budgets():
    sup = load_supervisor()
    assert sup.deadline_unix_for_launch(1000, 7200, 1300) == 1300
    assert sup.deadline_unix_for_launch(1000, 7200, 9000) == 8200
    plan = {"min_free_bytes": 20, "stall_seconds": 600, "save_stale_seconds": 3000,
            "max_attempts": 1390, "attempts_at_start": 110, "target_successful_updates": 1330,
            "successful_at_start": 110}
    assert sup.evaluate_stop(plan, 1, 9999, {"attempts": 1499, "successful": 1439}, 1, None, None, 10**12) is None
    assert sup.evaluate_stop(plan, 1, 9999, {"attempts": 1500, "successful": 1440}, 1, None, None, 10**12) == "maximum_attempt_budget_reached"
    assert sup.evaluate_stop(plan, 1, 9999, {"attempts": 1499, "successful": True, "updates": 1439}, 1, None, None, 10**12) is None
    assert sup.evaluate_stop(plan, 1, 9999, {"attempts": 1499, "successful": True, "updates": 1440}, 1, None, None, 10**12) is None
    assert sup.evaluate_stop(plan, 1, 9999, {"attempts": 1499, "successful": True, "updates": 1441}, 1, None, None, 10**12) == "target_successful_update_budget_exceeded"


def test_supervisor_no_restart_owned_tree_and_final_write_incident(tmp_path, monkeypatch):
    sup = load_supervisor()
    # taskkill is rooted only at the still-live Popen handle; no unrelated PID is accepted.
    calls = []
    class Child:
        pid = 222
        def poll(self): return None
        def wait(self, timeout=None): return 137
    def fake_run(args, **kwargs): calls.append((args, kwargs)); return SimpleNamespace(returncode=0)
    code, killed = sup.terminate_owned_tree(Child(), run=fake_run)
    assert code == 137 and killed
    assert calls[0][0] == ["taskkill.exe", "/PID", "222", "/T", "/F"] and calls[0][1]["shell"] is False
    class AlreadyExited:
        pid = 999
        def poll(self): return 0
    assert sup.terminate_owned_tree(AlreadyExited(), run=fake_run) == (0, False)
    assert len(calls) == 1

    # A finished failed worker is recorded exactly once and never automatically relaunched.
    worker, config = tmp_path / "worker.py", tmp_path / "config.yaml"
    worker.write_text("pass\n", encoding="utf-8")
    config.write_text("fixture: true\n", encoding="utf-8")
    output = tmp_path / "output"
    run_dir = output / "model"
    plan_path = tmp_path / "plan.json"
    script = ROOT / "scripts" / "supervise_learning_102m_v2.py"
    io_path = ROOT / "scripts" / "learning_io_v2.py"
    plan = {"worker_sha256": digest(worker), "supervisor_sha256": digest(script), "io_sha256": digest(io_path),
            "runtime_source_sha256": sup.runtime_digest(ROOT / "src" / "fdt_local"),
            "config_path": str(config), "config_sha256": digest(config), "output_dir": str(output),
            "run_dir": str(run_dir), "wall_seconds": 10, "max_attempts": 1390,
            "attempts_at_start": 110, "successful_at_start": 110, "target_successful_updates": 1330,
            "min_free_bytes": 20, "min_gpu_free_mib": 12000, "stall_seconds": 600,
            "save_stale_seconds": 3000, "absolute_deadline_unix": 9999999999}
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    class ExitedChild:
        pid = 333
        def poll(self): return 1
    launches = []
    result = sup.supervise(plan_path, worker, output, popen=lambda args, **kwargs: launches.append(args) or ExitedChild(),
        resource_check=lambda *_: {"disk_free": 10**12, "gpu_free_mib": 15000}, process_check=lambda *_: [])
    assert result == 1 and len(launches) == 1
    assert json.loads((output / "supervisor" / "final.json").read_text(encoding="utf-8"))["status"] == "FAILED"
    with __import__("pytest").raises(RuntimeError, match="already appears to have been launched"):
        sup.supervise(plan_path, worker, output, popen=lambda args, **kwargs: launches.append(args) or ExitedChild(),
            resource_check=lambda *_: {"disk_free": 10**12, "gpu_free_mib": 15000}, process_check=lambda *_: [])
    assert len(launches) == 1

    # If all normal final state replacements fail, an exclusive incident remains.
    monkeypatch.setattr(sup, "atomic_json", lambda *_: (_ for _ in ()).throw(PermissionError("denied")))
    sup._write_final(output, output / "supervisor", {"status": "ERROR", "worker_pid": 333})
    incidents = list((output / "supervisor").glob("incident_*.json"))
    assert len(incidents) == 1
    assert json.loads(incidents[0].read_text(encoding="utf-8"))["event"] == "final_state_write_error"
