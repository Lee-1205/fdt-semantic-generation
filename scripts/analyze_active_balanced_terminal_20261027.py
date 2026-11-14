"""CPU-only review of the stopped trial; do not reinterpret its historical gate."""
from pathlib import Path
from collections import Counter
import json
import torch
from fdt_local.storage import CheckpointStore
from learning_io_v2 import atomic_json
from run_active_balanced_diffusion_20261027_v3 import (
    is_trainable, count_batch_exposure, flatten_batch_indices)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/active_balanced_diffusion_20261027_runtime_v3"
final = json.loads((OUT / "final_status.json").read_text(encoding="utf-8"))
meta, payload = CheckpointStore(Path(final["last_marker"]).parent).verify(Path(final["last_marker"]))
parent_marker = ROOT / "runs/role8_diffusion_20261027_v1/model/complete_000130.json"
_, parent = CheckpointStore(parent_marker.parent).verify(parent_marker)
frozen_changes = [k for k, value in parent["model"].items()
                  if not is_trainable(k) and not torch.equal(value, payload["model"][k])]
if frozen_changes:
    raise ValueError("frozen tensors changed")
if (payload["updates"], payload["attempts"], payload["phase"], payload["phase_step"]) != (128, 128, 2, 128):
    raise ValueError("terminal counter/phase mismatch")
history = payload["history"]
if [r["attempt"] for r in history] != list(range(1, 129)):
    raise ValueError("attempt history incomplete")
rows = json.loads((ROOT / "research/artifacts/role8_active_balanced_20261027_v2/train.json").read_text(encoding="utf-8"))
batches = [batch for entry in history for batch in entry["batch_indices"]]
ids = count_batch_exposure(rows, batches)
runtime = payload["active_balanced_runtime"]
if ids != runtime["attempted_id_exposure"]:
    raise ValueError("ID exposure mismatch")
flat = flatten_batch_indices(batches)
class_counts = Counter(y for index in flat for y in rows[index]["labels"])
graph_counts = Counter(flat)
if ([class_counts[i] for i in range(4)] != runtime["attempted_query_class_exposure"]
        or [graph_counts[i] for i in range(len(rows))] != runtime["unique_graph_exposure"]):
    raise ValueError("class/unique-successful-graph exposure mismatch")
required = {"optimizer", "scaler", "sampler_rng", "noise_rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng"}
if not required <= payload.keys() or not payload["optimizer"]["state"]:
    raise ValueError("missing full recovery state")
base = json.loads((OUT / "baseline/summary.json").read_text(encoding="utf-8"))
after = json.loads((OUT / "diagnostics/step128/summary.json").read_text(encoding="utf-8"))
comparisons, errors = {}, {}
for name in base:
    before = base[name]["results"]["ddim_4"]
    current = after[name]["results"]["ddim_4"]
    comparisons[name] = {"before_accuracy": before["accuracy"], "after_accuracy": current["accuracy"],
        "absolute_gain": current["accuracy"] - before["accuracy"],
        "before_macro_f1": before["macro_f1"], "after_macro_f1": current["macro_f1"],
        "baseline_relative_accuracy_pass": current["accuracy"] >= before["accuracy"] - .02,
        "baseline_relative_macro_pass": current["macro_f1"] >= before["macro_f1"] - .05,
        "historical_absolute_94_pass": current["accuracy"] >= .94,
        "ground_accuracy": after[name]["ground_accuracy"],
        "initializer_accuracy": after[name]["results"]["initializer"]["accuracy"]}
    raw = json.loads((OUT / "diagnostics/step128" / f"{name}.json").read_text(encoding="utf-8"))["raw"]
    confusion, examples = Counter(), []
    for row in raw:
        for query, gold, pred, initial in zip(row["queries"], row["gold"], row["diffusion"]["4"], row["initializer"]):
            if pred == gold:
                continue
            confusion[f"{gold}->{pred}"] += 1
            if len(examples) < 4:
                examples.append({"query": query, "gold": gold, "prediction": pred,
                    "initializer": initial, "facts": row["facts"], "seed": row["seed"], "graph_key": row["graph_key"]})
    errors[name] = {"count": sum(confusion.values()), "confusions": dict(confusion),
                    "examples": examples, "raw_graph_traces_read": len(raw)}
result = {"status": "ROOT_TERMINAL_CPU_REVIEW_COMPLETE", "optimizer_updates_in_review": 0,
    "model_forwards_in_review": 0, "historical_trial_status": final["status"],
    "historical_trial_reason": final["reason"], "historical_gate_relaxed_or_retroactively_passed": False,
    "checkpoint_sha256": meta["sha256"], "checkpoint_bytes": meta["bytes"],
    "fullstate_hash_and_safe_load_verified": True, "full_recovery_state_present": True,
    "frozen_tensor_changes": frozen_changes, "phase": payload["phase"], "updates": payload["updates"],
    "attempts": payload["attempts"], "amp_skips": sum(r["amp_skipped"] for r in history),
    "same_seed17_comparisons": comparisons, "diagnostic_raw_errors": errors,
    "actual_exposure_reconstructed_exactly": True, "sampled_rows": len(flat),
    "distinct_training_graphs_sampled": len(graph_counts), "input_utf8_bytes": payload["exposure"],
    "query_class_exposure": [class_counts[i] for i in range(4)], "ID_exposure": ids,
    "step_seconds_sum": sum(r["step_seconds"] for r in history),
    "max_preclip_gradient_norm": max(r["gradient_norm_before_clip"] for r in history),
    "original_target256_completed": False, "final_three_seed_evaluation_run": False,
    "new_balanced_dev_baseline_ceiling": base["new_dev64"]["results"]["ddim_4"]["accuracy"] >= .97,
    "interpretation": "Absolute mixed-panel qualification floor failed despite improvement; no observed old-panel regression. Preserve STOPPED and separate future retention from qualification prospectively.",
    "E2_final_holdout_used": False, "model_completion_claimed": False}
atomic_json(OUT / "root_result.json", result)
print(json.dumps({k: v for k, v in result.items() if k not in ("diagnostic_raw_errors", "ID_exposure")}, indent=2))
