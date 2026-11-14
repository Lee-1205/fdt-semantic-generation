import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location("role8_supervisor", ROOT / "scripts" / "supervise_role8_diffusion.py")
supervisor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(supervisor)


PREFIXES = ["slot_router.", "memory_project.", "rel_left", "rel_right", "rel_score.",
            "rel_message.", "time.", "refiner.", "refine_norm.", "velocity."]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan_files(tmp_path):
    paths = {}
    for key, name in (("config_path", "config.yaml"), ("parent_marker", "parent.json"),
                      ("train_data_path", "train.json"), ("dev_data_path", "dev.json"),
                      ("candidate_manifest_path", "manifest.json")):
        path = tmp_path / name
        path.write_text("{}\n", encoding="utf-8")
        paths[key] = str(path)
    return paths


def valid_plan(tmp_path):
    files = plan_files(tmp_path)
    worker = ROOT / "scripts" / "run_role8_diffusion_20261027.py"
    supervisor_file = ROOT / "scripts" / "supervise_role8_diffusion.py"
    io = ROOT / "scripts" / "learning_io_v2.py"
    root = ROOT
    plan = {**files, "output_dir": str(tmp_path / "output"),
        "run_dir": str(tmp_path / "output" / "model"), "wall_seconds": 1800,
        "max_attempts": 140, "min_free_bytes": 20 * 1024**3, "min_gpu_free_mib": 12000,
        "stall_seconds": 600, "save_stale_seconds": 3000, "attempts_at_start": 0,
        "successful_at_start": 0, "target_successful_updates": 128, "reserve_seconds": 180,
        "phase": 2, "phase_steps": [480, 320, 128], "seed": 21,
        "parent_updates": 128, "parent_attempts": 128, "parent_optimizer_imported": False,
        "parent_rng_imported": False, "trainable_prefixes": PREFIXES,
        "max_forward_attempts": 4096, "final_eval_budget_seconds": 150, "learning_rate": 3e-5,
        "e2_final_split_evaluated": False,
        "worker_sha256": digest(worker), "supervisor_sha256": digest(supervisor_file),
        "io_sha256": digest(io), "runtime_source_sha256": supervisor.runtime_digest(root / "src" / "fdt_local"),
        "core_sha256": supervisor.runtime_digest(root / "src" / "fdt_local"),
        "config_sha256": digest(Path(files["config_path"])),
        "parent_marker_sha256": digest(Path(files["parent_marker"])),
        "train_data_sha256": digest(Path(files["train_data_path"])),
        "dev_data_sha256": digest(Path(files["dev_data_path"])),
        "candidate_manifest_sha256": digest(Path(files["candidate_manifest_path"])),
        "common_worker_sha256": digest(ROOT / "scripts" / "run_e2_grounding_20261026.py"),
        "probe_common_sha256": digest(ROOT / "scripts" / "probe_e1_readonly.py")}
    return plan


def test_role8_plan_validation_requires_exact_stage_and_dependency_digests(tmp_path):
    worker = ROOT / "scripts" / "run_role8_diffusion_20261027.py"
    output = tmp_path / "output"
    plan = valid_plan(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    parsed, run_dir, _ = supervisor.validate_plan(plan_path, worker, output)
    assert parsed["phase"] == 2 and run_dir == output / "model"
    plan["phase"] = 1
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    try:
        supervisor.validate_plan(plan_path, worker, output)
    except ValueError as exc:
        assert "phase mismatch" in str(exc)
    else:
        raise AssertionError("wrong phase schedule passed supervisor validation")


def test_worker_preflight_is_cpu_gate_before_launch(tmp_path, monkeypatch):
    worker = ROOT / "scripts" / "run_role8_diffusion_20261027.py"
    result = "CPU_PREFLIGHT_PASS"
    seen = {}
    def fake_run(args, **kwargs):
        seen["args"] = args
        assert "--preflight" in args
        assert kwargs["shell"] is False
        return SimpleNamespace(returncode=0, stdout=result, stderr="")
    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    assert supervisor.validate_worker_preflight(tmp_path / "plan.json", worker, tmp_path / "output") == {"status": result}
    assert "nvidia-smi" not in seen["args"]
