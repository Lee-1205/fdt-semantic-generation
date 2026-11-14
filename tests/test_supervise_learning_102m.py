import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("supervisor", ROOT / "scripts" / "supervise_learning_102m.py")
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def basic_plan():
    return {"wall_seconds": 7200, "max_attempts": 1440, "min_free_bytes": 20 * 1024**3,
            "min_gpu_free_mib": 12000, "stall_seconds": 600, "save_stale_seconds": 3000}


def test_stop_reasons_cover_deadline_stall_storage_and_anomaly():
    plan = basic_plan()
    assert supervisor.evaluate_stop(plan, 7200, 7200, None, 0, None, None, 30 * 1024**3) == "wall_clock_deadline"
    assert supervisor.evaluate_stop(plan, 601, 9999, {"step": 1}, 0, None, None, 30 * 1024**3) == "worker_progress_stalled"
    assert supervisor.evaluate_stop(plan, 5, 9999, None, 0, None, None, 21 * 1024**3) == "disk_reserve_below_20GiB_plus_2GiB_checkpoint"
    assert supervisor.evaluate_stop(plan, 5, 9999, {"anomaly": "nan_loss"}, 5, None, None, 30 * 1024**3) == "worker_reported_anomaly: nan_loss"
    assert supervisor.evaluate_stop(plan, 5, 9999, None, 0, {"generation": 2}, 3001, 30 * 1024**3) == "durable_checkpoint_stale"
    assert supervisor.evaluate_stop(plan, 5, 9999, {"attempts": 1440}, 5, None, None, 30 * 1024**3) == "maximum_attempt_budget_reached"


def test_stop_is_atomic_and_preserves_first_reason(tmp_path):
    supervisor.write_stop(tmp_path, "first reason")
    supervisor.write_stop(tmp_path, "second reason")
    assert (tmp_path / "STOP").read_text(encoding="utf-8") == "first reason\n"


def test_marker_verification_checks_payload_size_and_hash(tmp_path):
    payload = tmp_path / "payload_000000.pt"
    payload.write_bytes(b"fixture payload")
    marker = tmp_path / "complete_000000.json"
    marker.write_text(json.dumps({"schema": "fdt_local_checkpoint_v1", "generation": 0,
        "file": payload.name, "bytes": payload.stat().st_size, "sha256": digest(payload),
        "contract": {}, "run_id": "fixture", "attempts": 1, "updates": 1,
        "local_checkpoint_verified": True}), encoding="utf-8")
    assert supervisor.verify_marker(marker, tmp_path)["generation"] == 0
    original = payload.read_bytes()
    cache = {}
    payload.write_bytes(b"tampered")
    assert supervisor.latest_durable(tmp_path, cache)[0] is None
    assert not cache  # failed first verification must never be cached
    payload.write_bytes(original)
    assert supervisor.latest_durable(tmp_path, cache)[0]["generation"] == 0
    meta = json.loads(marker.read_text(encoding="utf-8"))
    meta["sha256"] = "0" * 64
    marker.write_text(json.dumps(meta), encoding="utf-8")
    assert supervisor.latest_durable(tmp_path, cache)[0] is None  # changed marker is reverified


def test_cim_filter_matches_only_project_training_and_excludes_self(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / ".venv" / "Scripts").mkdir(parents=True)
    (root / "scripts").mkdir()
    worker = root / "scripts" / "run_learning_102m.py"
    worker.write_text("pass", encoding="utf-8")
    supervisor_file = root / "scripts" / "supervise_learning_102m.py"
    supervisor_file.write_text("pass", encoding="utf-8")
    vpy = str(root / ".venv" / "Scripts" / "python.exe")
    items = [
        {"ProcessId": 10, "ExecutablePath": vpy, "CommandLine": f'"{vpy}" "{worker}"'},
        {"ProcessId": 11, "ExecutablePath": vpy, "CommandLine": f'"{vpy}" -m fdt_local train'},
        {"ProcessId": 12, "ExecutablePath": vpy, "CommandLine": f'"{vpy}" "{supervisor_file}" --worker train'},
        {"ProcessId": 13, "ExecutablePath": vpy, "CommandLine": f'"{vpy}" unrelated.py'},
        {"ProcessId": 99, "ExecutablePath": vpy, "CommandLine": f'"{vpy}" "{worker}"'},
    ]
    monkeypatch.setattr(supervisor.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=json.dumps(items)))
    assert [p["pid"] for p in supervisor.find_existing_training(root, supervisor_pid=99)] == [10, 11]


def test_supervisor_launches_owned_child_once_and_refuses_relaunch(tmp_path, monkeypatch):
    script = ROOT / "scripts" / "supervise_learning_102m.py"
    worker, config = tmp_path / "worker.py", tmp_path / "config.yaml"
    worker.write_text("pass\n", encoding="utf-8")
    config.write_text("fixture: true\n", encoding="utf-8")
    output = tmp_path / "new-output"
    run_dir = output / "model"
    plan_path = tmp_path / "plan.json"
    plan = basic_plan() | {"worker_sha256": digest(worker), "supervisor_sha256": digest(script),
        "runtime_source_sha256": supervisor.runtime_digest(ROOT / "src" / "fdt_local"),
        "config_path": str(config), "config_sha256": digest(config), "output_dir": str(output), "run_dir": str(run_dir),
        "wall_seconds": 1}
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    class FakeClock:
        now = 0.0
        def tick(self, seconds): self.now += seconds
    clock = FakeClock()
    class Child:
        pid = 424242
        def __init__(self): self.code = None; self.terminated = 0; self.killed = 0
        def poll(self): return self.code
        def terminate(self): self.terminated += 1; self.code = 143
        def kill(self): self.killed += 1; self.code = 137
        def wait(self, timeout=None): return self.code
    child = Child()
    launches = []
    def fake_popen(args, **kwargs): launches.append((args, kwargs)); return child
    monkeypatch.setattr(supervisor.shutil, "disk_usage", lambda _: type("Disk", (), {"total": 100, "free": 50 * 1024**3})())
    result = supervisor.supervise(plan_path, worker, output, popen=fake_popen,
        monotonic=lambda: clock.now, sleep=lambda seconds: clock.tick(seconds),
        resource_check=lambda *_: {"disk_free": 50 * 1024**3, "gpu_free_mib": 16000},
        process_check=lambda *_: [])
    assert result == 143
    assert len(launches) == 1
    args, kwargs = launches[0]
    assert args[0] == supervisor.sys.executable and args[1:4] == ["-B", "-X", "utf8"]
    assert kwargs["shell"] is False
    assert child.terminated == 1 and child.killed == 0
    state = json.loads((output / "supervisor" / "final.json").read_text(encoding="utf-8"))
    assert state["worker_pid"] == 424242 and state["status"] == "STOPPED"
    assert json.loads((output / "supervisor" / "latest.json").read_text(encoding="utf-8"))["status"] == "STOPPED"
    try:
        supervisor.supervise(plan_path, worker, output, resource_check=lambda *_: {}, process_check=lambda *_: [])
    except (RuntimeError, ValueError) as exc:
        assert any(token in str(exc) for token in ("terminal markers", "already been launched", "STOP", "new, nonexistent"))
    else:
        raise AssertionError("relaunch was not rejected")
    assert len(launches) == 1
