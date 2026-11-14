"""Seal one prospective active-balanced run; CPU only, no launch."""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
from fdt_local.runner import source_digest
from learning_io_v2 import atomic_json
from run_active_balanced_diffusion_20261027 import TRAINABLE_PREFIXES, sha256

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/active_balanced_diffusion_20261027_v1"
if OUT.exists():
    raise FileExistsError("new sealed output required; do not reuse or reset")
DATA = ROOT / "research/artifacts/role8_active_balanced_20261027_v2"
paths = {
    "worker": ROOT / "scripts/run_active_balanced_diffusion_20261027.py",
    "supervisor": ROOT / "scripts/supervise_active_balanced_diffusion.py",
    "io": ROOT / "scripts/learning_io_v2.py",
    "common_worker": ROOT / "scripts/run_e2_grounding_20261026.py",
    "probe_common": ROOT / "scripts/probe_e1_readonly.py",
    "dataset_builder": ROOT / "research/role8_active_balanced_v2.py",
    "config": ROOT / "configs/active_balanced_diffusion_20261027.yaml",
    "parent_marker": ROOT / "runs/role8_diffusion_20261027_v1/model/complete_000130.json",
    "train_data": DATA / "train.json", "dev_data": DATA / "dev.json",
    "candidate_manifest": DATA / "manifest.json",
    "old_role_dev": ROOT / "research/artifacts/role8_coverage_20261026_v1/dev.json",
    "root_data_review": DATA / "root_cpu_review.json",
}
review = json.loads(paths["root_data_review"].read_text(encoding="utf-8"))
if review["status"] != "ROOT_STORED_CPU_REVIEW_PASS" or review["manifest_file_sha256"] != sha256(paths["candidate_manifest"]):
    raise ValueError("new data has not passed current root stored CPU review")
plan = {"schema": "fdt_active_balanced_phase2_plan_v1", "status": "SEALED_NOT_EXECUTED",
    "authorized": True,
    "authorization": "Human explicitly requested continued local training/inspection/anomaly correction and resume; standing local hardware authorization. No new individual reply fabricated.",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "output_dir": str(OUT), "run_dir": str(OUT / "model"),
    "config_path": str(paths["config"]), "parent_marker": str(paths["parent_marker"]),
    "train_data_path": str(paths["train_data"]), "dev_data_path": str(paths["dev_data"]),
    "candidate_manifest_path": str(paths["candidate_manifest"]),
    "old_role_dev_path": str(paths["old_role_dev"]),
    "parent_sha256": "0b5066444b9fdf0338219fba167edca8c2d211dbdda0bc7ea202bcc8b2784829",
    "runtime_source_sha256": source_digest(), "core_sha256": source_digest(),
    "phase": 2, "phase_steps": [480, 320, 256], "seed": 22,
    "target_successful_updates": 256, "max_attempts": 280,
    "attempts_at_start": 0, "successful_at_start": 0,
    "parent_updates": 128, "parent_attempts": 128,
    "parent_stage": "Role8", "ancestor_e2_updates": 128, "ancestor_e1_updates": 1440,
    "parent_optimizer_imported": False, "parent_rng_imported": False,
    "trainable_prefixes": list(TRAINABLE_PREFIXES), "learning_rate": 3e-5,
    "wall_seconds": 1800, "reserve_seconds": 180,
    "stall_seconds": 600, "save_stale_seconds": 3000,
    "min_free_bytes": 20 * 1024**3, "min_gpu_free_mib": 12288,
    "max_forward_attempts": 6144, "final_eval_budget_seconds": 150,
    "mixed_query_seed": 22017, "mixed_queries_per_graph": 16,
    "e2_final_split_evaluated": False,
    "training_rows": 1024, "training_distinct_abstract_graphs": 1024, "dev_rows": 64,
    "train_queries_per_row": 8, "train_graphs_per_query_class": 256,
    "train_rows_sampled_per_attempt": 8, "max_attempted_row_exposures": 2240,
    "max_attempted_queries": 17920, "max_attempted_facts": 26880,
    "max_input_utf8_bytes_capacity_bound": 1290240,
    "exposure_units": "adapter rows/queries/facts/UTF8 bytes, not language-model supervised tokens",
    "loss": "unchanged VP v_mse + .25 clean_mse + .5 weighted CE; all-query class counts produce weights1",
    "retention": {"panels": ["old_normal16", "old_large16", "old_role_dev128", "new_dev_mixed64"],
        "accuracy_floor": "max(.94, same-parent same-seed baseline-.02)",
        "macro_f1_floor": "same-parent baseline-.05", "ground_initializer_floor": .95,
        "consecutive_failures_to_stop": 2, "noise_seed": 17,
        "below_absolute_floor_is_not_automatically_a_regression": True},
    "development_goals": {"new_dev64_gain": .03, "paired_new_ids16_gain": .03,
        "same_noise_seed": 17, "old_and_mixed_retention_required": True,
        "baseline_ceiling_must_be_disclosed": True},
    "baseline_and_diagnostics": {"steps": [0, 64, 128, 192, 256], "noise_seeds": [17], "DDIM_steps": 4,
        "old_normal_seed": 60001, "old_large_seed": 80001, "large_n_range": [7, 8],
        "paired_mapping": [6, 7, 2, 3, 4, 5, 0, 1],
        "mixed_view_is_correlated_with_new_dev64": True},
    "final_noise_seeds": [17, 29, 43], "per_attempt_verified_fullstate": True,
    "no_auto_restart": True, "old_deadlines_unchanged": True,
    "multiple_data_exposure_schedule_changes_are_not_isolated_causal_gain": True,
    "no_broad_lm_or_pure_neural_reasoning_claim": True,
}
plan.update({f"{name}_sha256": sha256(path) for name, path in paths.items()})
if source_digest() != "eb68ced289218c5a9ab1fc0481f52106da9f8c5892cb67022e70759d24a4f352":
    raise ValueError("frozen core changed")
OUT.mkdir()
snapshot = OUT / "code_snapshot"
snapshot.mkdir()
for key in ("worker", "supervisor", "io", "common_worker", "probe_common", "dataset_builder", "config"):
    shutil.copyfile(paths[key], snapshot / paths[key].name)
atomic_json(OUT / "plan.json", plan)
atomic_json(OUT / "seal.json", {"plan_sha256": sha256(OUT / "plan.json"),
    "code_and_data_sha256": {k: v for k, v in plan.items() if k.endswith("sha256")},
    "actual_execution_started": False})
print(json.dumps({"status": "SEALED_NOT_EXECUTED", "output": str(OUT),
    "plan_sha256": sha256(OUT / "plan.json")}))
