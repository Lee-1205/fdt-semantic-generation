"""Prepare an explicit continuation, bounded by the ORIGINAL absolute deadline."""
import json
from pathlib import Path
import shutil
import time

from fdt_local.runner import source_digest
from fdt_local.storage import CheckpointStore, atomic_json, ensure_space, sha256

root = Path(__file__).resolve().parents[1]
parent_run = root / "runs/e1_masked_learning_20261026_first"
output = root / "runs/e1_masked_learning_20261026_recovery_v2"
if output.exists():
    raise FileExistsError("do not reopen or overwrite a continuation")
prior_status = json.loads((parent_run / "final_status.json").read_text(encoding="utf-8"))
if prior_status["status"] != "FAILED" or prior_status["updates"] != 110:
    raise ValueError("parent terminal state changed")
old = json.loads((parent_run / "plan.json").read_text(encoding="utf-8"))
clock = json.loads((parent_run / "clock.json").read_text(encoding="utf-8"))
remaining_seconds = int(clock["deadline_unix"] - time.time())
if remaining_seconds <= old["reserve_seconds"] + 120:
    raise RuntimeError("original deadline has no safe continuation window")
ensure_space(root, 8 * 1024**3)
parent_marker = parent_run / "model/complete_000110.json"
meta, payload = CheckpointStore(parent_marker.parent).verify(parent_marker)
if meta["sha256"] != "836815fb470703b33f98273f8cb3e37d06b5e9b09da9df6879b38e7b2d15cdc5":
    raise ValueError("parent checkpoint identity changed")
del payload
worker = root / "scripts/run_learning_102m_recovery_v2.py"
supervisor = root / "scripts/supervise_learning_102m_v2.py"
helper = root / "scripts/learning_io_v2.py"
plan = {**old, "schema": "fdt_102m_operational_recovery_v2", "output_dir": str(output),
        "run_dir": str(output / "model"), "wall_seconds": remaining_seconds,
        "absolute_deadline_unix": clock["deadline_unix"],
        "worker_sha256": sha256(worker), "supervisor_sha256": sha256(supervisor),
        "io_sha256": sha256(helper), "runtime_source_sha256": source_digest(),
        "parent_marker": str(parent_marker), "parent_sha256": meta["sha256"],
        "parent_plan_sha256": sha256(parent_run / "plan.json"),
        "imported_updates": 110, "imported_attempts": 110,
        "attempts_at_start": 110, "successful_at_start": 110,
        "max_attempts": 1390, "target_successful_updates": 1330,
        "global_max_attempts": 1500, "global_target_successful_updates": 1440,
        "remaining_successful_updates": 1330, "remaining_optimizer_attempts": 1390,
        "fresh_initialization": False, "populated_adam_and_all_rng_import": True,
        "operational_migration": "bounded Windows IO retry and owned process-tree supervision; model/data/loss unchanged",
        "strict_old_full_execution_contract_claimed": False,
        "first110_updates_retrained": False, "deadline_extended": False,
        "prepared_unix": time.time()}
output.mkdir()
atomic_json(output / "plan.json", plan)
snapshot = output / "code_snapshot"
shutil.copytree(root / "src/fdt_local", snapshot / "fdt_local", ignore=shutil.ignore_patterns("__pycache__"))
for path in (worker, supervisor, helper, Path(plan["config_path"])):
    shutil.copy2(path, snapshot / path.name)
atomic_json(output / "readiness.json", {"status": "SEALED_NOT_LAUNCHED", "plan_sha256": sha256(output / "plan.json"),
    "parent_full_checkpoint_verified": True, "parent_sha256": meta["sha256"], "original_deadline_unix": clock["deadline_unix"],
    "new_optimizer_attempts": 0, "old_run_untouched": True})
print(json.dumps({"status": "SEALED_NOT_LAUNCHED", "plan_sha256": sha256(output / "plan.json"),
                  "remaining_seconds": remaining_seconds, "remaining_success_updates": 1330,
                  "absolute_deadline_unix": clock["deadline_unix"]}))
