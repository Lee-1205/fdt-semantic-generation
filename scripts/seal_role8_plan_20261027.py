"""Seal a new prospective role8 output; no GPU or process launch."""
from pathlib import Path
import json
import shutil
from datetime import datetime, timezone
from fdt_local.runner import source_digest
from learning_io_v2 import atomic_json
from run_role8_diffusion_20261027 import TRAINABLE_PREFIXES, sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/role8_diffusion_20261027_v1"
if OUT.exists():
    raise FileExistsError("a new immutable output is required")
paths = {
    "worker": ROOT / "scripts/run_role8_diffusion_20261027.py",
    "supervisor": ROOT / "scripts/supervise_role8_diffusion.py",
    "io": ROOT / "scripts/learning_io_v2.py",
    "common_worker": ROOT / "scripts/run_e2_grounding_20261026.py",
    "probe_common": ROOT / "scripts/probe_e1_readonly.py",
    "config": ROOT / "configs/role8_diffusion_20261027.yaml",
    "parent_marker": ROOT / "runs/e2_lexical_grounding_20261026_v1/model/complete_000129.json",
    "train_data": ROOT / "research/artifacts/role8_coverage_20261026_v1/train.json",
    "dev_data": ROOT / "research/artifacts/role8_coverage_20261026_v1/dev.json",
    "candidate_manifest": ROOT / "research/artifacts/role8_coverage_20261026_v1/manifest.json",
}
plan = {"schema": "fdt_role8_phase2_plan_v1", "status": "SEALED_NOT_EXECUTED",
    "authorized": True,
    "authorization": "User explicitly authorized continued local training with inspection and anomaly correction, and said resume; standing local hardware authorization; no invented individual reply.",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "output_dir": str(OUT), "run_dir": str(OUT / "model"),
    "config_path": str(paths["config"]), "parent_marker": str(paths["parent_marker"]),
    "train_data_path": str(paths["train_data"]), "dev_data_path": str(paths["dev_data"]),
    "candidate_manifest_path": str(paths["candidate_manifest"]),
    "parent_sha256": "938b15e963ad4f9a120aed333653e02f588594c4235482960cc4164bb1d1bac4",
    "runtime_source_sha256": source_digest(), "core_sha256": source_digest(),
    "phase": 2, "phase_steps": [480, 320, 128], "seed": 21,
    "target_successful_updates": 128, "max_attempts": 140,
    "attempts_at_start": 0, "successful_at_start": 0,
    "parent_updates": 128, "parent_attempts": 128, "ancestor_e1_updates": 1440,
    "parent_optimizer_imported": False, "parent_rng_imported": False,
    "trainable_prefixes": list(TRAINABLE_PREFIXES), "learning_rate": 3e-5,
    "wall_seconds": 1800, "reserve_seconds": 180,
    "stall_seconds": 600, "save_stale_seconds": 3000,
    "min_free_bytes": 20 * 1024**3, "min_gpu_free_mib": 12288,
    "max_forward_attempts": 4096, "final_eval_budget_seconds": 150,
    "e2_final_split_evaluated": False,
    "training_rows": 512, "training_abstract_source_graphs": 64, "dev_rows": 128,
    "dev_abstract_source_graphs": 16, "correlated_cyclic_renamings": 8,
    "train_rows_sampled_per_attempt": 8, "max_attempted_row_exposures": 1120,
    "max_attempted_queries": 17920, "max_attempted_facts": 13440,
    "exposure_units": "bounded adapter rows/queries/facts and UTF8 input bytes, not LM supervised tokens",
    "loss": "unchanged VP diffusion v_mse + .25 clean_mse + .5 weighted CE; class weights from every role8 training query",
    "retention": {"panels": ["old_normal16", "old_large16"],
        "accuracy_floor": "max(.94, same-parent same-seed baseline-.02)",
        "macro_f1_floor": "same-parent baseline-.05", "ground_initializer_floor": .95,
        "consecutive_failures_to_stop": 2, "noise_seed": 17},
    "development_goal": {"paired_new_ids_accuracy_gain": .05,
        "old_retention_pass": True, "frozen_subset_unchanged": True},
    "baseline_and_diagnostics": {"panels": ["role8_dev128", "old_normal16", "old_large16", "paired_new_ids16"],
        "steps": [0, 32, 64, 96, 128], "noise_seeds": [17], "DDIM_steps": 4,
        "old_normal_seed": 60001, "old_large_seed": 80001, "large_n_range": [7, 8],
        "paired_mapping": [6, 7, 2, 3, 4, 5, 0, 1]},
    "final_noise_seeds": [17, 29, 43], "per_attempt_verified_fullstate": True,
    "no_auto_restart": True, "old_deadlines_unchanged": True,
    "no_broad_lm_or_pure_neural_reasoning_claim": True,
}
plan.update({f"{name}_sha256": sha256(path) for name, path in paths.items()})
if source_digest() != "eb68ced289218c5a9ab1fc0481f52106da9f8c5892cb67022e70759d24a4f352":
    raise ValueError("frozen core source changed")
OUT.mkdir()
snapshot = OUT / "code_snapshot"
snapshot.mkdir()
for key in ("worker", "supervisor", "io", "common_worker", "probe_common", "config"):
    shutil.copyfile(paths[key], snapshot / paths[key].name)
atomic_json(OUT / "plan.json", plan)
atomic_json(OUT / "seal.json", {"plan_sha256": sha256(OUT / "plan.json"),
    "code_and_data_sha256": {k: v for k, v in plan.items() if k.endswith("sha256")},
    "actual_execution_started": False})
print(json.dumps({"status": "SEALED_NOT_EXECUTED", "output": str(OUT),
    "plan_sha256": sha256(OUT / "plan.json")}))
