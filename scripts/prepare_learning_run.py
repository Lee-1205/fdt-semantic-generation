"""Seal the one authorized E1 learning run; does not launch a GPU process."""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path
import shutil

import yaml

from fdt_local.runner import source_digest
from fdt_local.storage import atomic_json, ensure_space, sha256

root = Path(__file__).resolve().parents[1]
output = root / "runs/e1_masked_learning_20261026_first"
config = root / "configs/e1_authorized_20261026.yaml"
if output.exists() or config.exists():
    raise FileExistsError("preserve sealed runs/configs; explicit new ID required")
ensure_space(root, 8 * 1024**3)
spec = yaml.safe_load((root / "configs/local_masked_candidate.yaml").read_text(encoding="utf-8"))
spec["research_stage"] = "E1 authorized learning; masks only, baseline data/objectives retained"
config.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
output.mkdir()
worker = root / "scripts/run_learning_102m.py"
supervisor = root / "scripts/supervise_learning_102m.py"
plan = {
    "schema": "fdt_102m_authorized_learning_v1",
    "created_kst": datetime.now(timezone(timedelta(hours=9))).isoformat(),
    "human_instruction": "학습 시작해 계속 검사하며 해당 구조의 가능성을 조사하고 이상하다 싶으면 멈추고 수정해",
    "authorized": True, "output_dir": str(output), "run_dir": str(output / "model"),
    "config_path": str(config), "config_sha256": sha256(config),
    "worker_sha256": sha256(worker), "supervisor_sha256": sha256(supervisor),
    "runtime_source_sha256": source_digest(), "profile": "actual_102237197_parameters",
    "variant": "masked", "fresh_initialization": True, "seed": 19,
    "old_2240_weights_recovered": False, "phase_steps": [480, 320, 640],
    "target_successful_updates": 1440, "max_attempts": 1500,
    "wall_seconds": 7200, "reserve_seconds": 300,
    "min_free_bytes": 20 * 1024**3, "min_gpu_free_mib": 12000,
    "stall_seconds": 600, "save_stale_seconds": 3000,
    "checkpoint_policy": "full model/Adam/scaler/all RNG every attempt; verified before next attempt; own-run rotation only",
    "diagnostic_phase_steps": [0, 1, 64, 128, 256, 384, 480, 512, 640],
    "ground_gate": .9, "semantic_gate": .8,
    "anomaly_stops": ["nonfinite loss/parameters", "gradient ownership failure", "three consecutive AMP skips",
                      "more than four skips in twenty attempts", "three preclip norms above1000", "repeated loss explosion",
                      "functional phase gate", "disk reserve", "save/progress stall", "immutable deadline"],
    "evaluation_noise_seeds": [17, 29, 43],
    "evaluation_suites": {
        "new_graphs": {"graphs": 16, "seed": 60001},
        "heldout_original_phrases": {"graphs": 16, "seed": 70001, "heldout": True},
        "larger_graphs": {"graphs": 16, "seed": 80001, "n_range": [7, 8]},
    },
    "e2_candidate_train_used": False, "e2_final_selection_allowed": False,
    "matched_baseline_control_run": False,
    "interpretation": "Tests learnability of the bounded typed adapter. Not general LM completion, learned NER, or causal mask/architecture superiority.",
    "auto_restart": False,
}
atomic_json(output / "plan.json", plan)
snapshot = output / "code_snapshot"
shutil.copytree(root / "src/fdt_local", snapshot / "fdt_local", ignore=shutil.ignore_patterns("__pycache__"))
for path in (worker, supervisor, config):
    shutil.copy2(path, snapshot / path.name)
atomic_json(output / "readiness.json", {"status": "SEALED_NOT_LAUNCHED", "plan_sha256": sha256(output / "plan.json"),
    "worker_sha256": sha256(worker), "supervisor_sha256": sha256(supervisor),
    "runtime_source_sha256": source_digest(), "local_cpu_controller_tests": 4,
    "supervisor_cpu_tests": 5, "gpu_execution": False})
print(json.dumps({"status": "SEALED_NOT_LAUNCHED", "plan": str(output / "plan.json"),
                  "plan_sha256": sha256(output / "plan.json"), "output": str(output)}, ensure_ascii=False))
