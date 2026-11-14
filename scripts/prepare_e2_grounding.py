"""New bounded E2 stage, not a reopening of the completed E1 deadline."""
import json
from pathlib import Path
import shutil
import time
import yaml

from fdt_local.runner import source_digest
from fdt_local.storage import CheckpointStore, atomic_json, ensure_space, sha256

root = Path(__file__).resolve().parents[1]
output = root / "runs/e2_lexical_grounding_20261026_v1"
config = root / "configs/e2_grounding_authorized_20261026.yaml"
if output.exists() or config.exists():
    raise FileExistsError("explicit new stage ID required")
parent = root / "runs/e1_masked_learning_20261026_recovery_v2"
terminal = json.loads((parent / "final_status.json").read_text(encoding="utf-8"))
if terminal["status"] != "COMPLETE" or terminal["updates"] != 1440:
    raise ValueError("completed E1 is required")
marker = parent / "model/complete_001336.json"
meta, payload = CheckpointStore(marker.parent).verify(marker)
if meta["sha256"] != "d57d06ec7cc1ebb4f5ccbf365673554efc8a705043d492dd802fff1b6c7c8edb":
    raise ValueError("unexpected E1 parent")
del payload
ensure_space(root, 8 * 1024**3)
spec = yaml.safe_load((root / "configs/e1_authorized_20261026.yaml").read_text(encoding="utf-8"))
spec["plan"].update(seed=20, phase_steps=[128, 320, 640], train_graphs=512, dev_graphs=96,
                    encoder_lr=1e-5, ground_lr=3e-4)
spec["research_stage"] = "E2 grounding only; full parent model warmstart; fresh optimizer; original dev96 cached for retention"
config.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
worker = root / "scripts/run_e2_grounding_20261026.py"
supervisor = root / "scripts/supervise_e2_grounding.py"
helper = root / "scripts/learning_io_v2.py"
candidate = root / "research/artifacts/e2_lexical_20261026_v2"
plan = {"schema": "fdt_e2_lexical_grounding_stage_v1", "authorized": True,
    "human_instruction": "학습 시작해 계속 검사하며 해당 구조의 가능성을 조사하고 이상하다 싶으면 멈추고 수정해",
    "output_dir": str(output), "run_dir": str(output / "model"),
    "config_path": str(config), "config_sha256": sha256(config),
    "worker_sha256": sha256(worker), "supervisor_sha256": sha256(supervisor),
    "io_sha256": sha256(helper), "runtime_source_sha256": source_digest(), "core_sha256": source_digest(),
    "parent_marker": str(marker), "parent_sha256": meta["sha256"], "parent_updates": 1440, "parent_attempts": 1440,
    "seed": 20, "phase_steps": [128, 320, 640], "encoder_lr": 1e-5, "head_router_lr": 3e-4,
    "target_successful_updates": 128, "max_attempts": 140, "attempts_at_start": 0, "successful_at_start": 0,
    "wall_seconds": 1800, "reserve_seconds": 180, "min_free_bytes": 20 * 1024**3,
    "min_gpu_free_mib": 12000, "stall_seconds": 600, "save_stale_seconds": 3000,
    "old_retention_floor": .95, "old_initializer_retention_floor": .95,
    "checkpoint_policy": "full current model/Adam/scaler/RNG every attempt; first/milestone/final tags; own-run rotation",
    "parent_optimizer_imported": False, "parent_rng_imported": False, "fresh_optimizer": True,
    "frozen_anchors_and_prior": True, "trainable_prefixes": ["language.", "phrase_router.", "ground_head."],
    "old_e1_deadline_reopened": False, "old_e1_run_modified": False,
    "e2_final_split_evaluated": False, "diffusion_successor_automatically_promoted": False,
    "former_development_families_now_seen": ["en_exceed", "ko_numeric"],
    "created_unix": time.time(), "no_automatic_restart": True}
for key, name in (("train_data", "train.json"), ("dev_data", "dev.json"), ("candidate_manifest", "manifest.json")):
    path = candidate / name
    plan[key + "_path"] = str(path)
    plan[key + "_sha256"] = sha256(path)
output.mkdir()
atomic_json(output / "plan.json", plan)
snapshot = output / "code_snapshot"
shutil.copytree(root / "src/fdt_local", snapshot / "fdt_local", ignore=shutil.ignore_patterns("__pycache__"))
for path in (worker, supervisor, helper, config, candidate / "manifest.json"):
    shutil.copy2(path, snapshot / path.name)
atomic_json(output / "readiness.json", {"status": "SEALED_NOT_LAUNCHED", "plan_sha256": sha256(output / "plan.json"),
    "parent_full_model_verified": True, "new_stage_optimizer_attempts": 0,
    "candidate_cpu_rows_verified": 576, "worker_and_io_cpu_tests_passed": 15})
print(json.dumps({"status": "SEALED_NOT_LAUNCHED", "plan_sha256": sha256(output / "plan.json"),
                  "target_updates": 128, "max_attempts": 140, "wall_seconds": 1800}))
