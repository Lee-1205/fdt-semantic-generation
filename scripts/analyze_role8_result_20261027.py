"""CPU-only terminal evidence review; no model forward or optimizer update."""
from pathlib import Path
from collections import Counter
import json
import torch
from fdt_local.storage import CheckpointStore
from learning_io_v2 import atomic_json
from run_role8_diffusion_20261027 import count_batch_exposure, frozen_digest, is_role8_trainable, state_digest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/role8_diffusion_20261027_v1"
final = json.loads((OUT / "final_status.json").read_text(encoding="utf-8"))
marker = Path(final["last_marker"])
meta, payload = CheckpointStore(marker.parent).verify(marker)
parent_marker = ROOT / "runs/e2_lexical_grounding_20261026_v1/model/complete_000129.json"
_, parent = CheckpointStore(parent_marker.parent).verify(parent_marker)
frozen_changes = [k for k, v in parent["model"].items()
                  if not is_role8_trainable(k) and not torch.equal(v, payload["model"][k])]
changed_trainable = [k for k, v in parent["model"].items()
                     if is_role8_trainable(k) and not torch.equal(v, payload["model"][k])]
if frozen_changes:
    raise ValueError(f"frozen subset changed: {frozen_changes}")
if (payload["attempts"], payload["updates"], payload["phase"], payload["phase_step"]) != (128, 128, 2, 128):
    raise ValueError("terminal counters mismatch")
history = payload["history"]
if [entry["attempt"] for entry in history] != list(range(1, 129)):
    raise ValueError("incomplete attempt history")
rows = json.loads((ROOT / "research/artifacts/role8_coverage_20261026_v1/train.json").read_text(encoding="utf-8"))
batches = [batch for entry in history for batch in entry["batch_indices"]]
exposure = count_batch_exposure(rows, batches)
if exposure != payload["role8_runtime"]["attempted_id_exposure"]:
    raise ValueError("saved actual ID exposure disagrees with sampled history")
required = {"optimizer", "scaler", "sampler_rng", "noise_rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng"}
if not required <= payload.keys() or not payload["optimizer"]["state"]:
    raise ValueError("missing full recovery state")
base = json.loads((OUT / "baseline/summary.json").read_text(encoding="utf-8"))
after = json.loads((OUT / "diagnostics/step128/summary.json").read_text(encoding="utf-8"))
paired_gain = after["paired_new_ids16"]["results"]["ddim_4"]["accuracy"] - base["paired_new_ids16"]["results"]["ddim_4"]["accuracy"]
matched = {k: {"before_accuracy": base[k]["results"]["ddim_4"]["accuracy"],
                "after_accuracy": after[k]["results"]["ddim_4"]["accuracy"],
                "before_macro_f1": base[k]["results"]["ddim_4"]["macro_f1"],
                "after_macro_f1": after[k]["results"]["ddim_4"]["macro_f1"]} for k in base}
errors, inspected = {}, {}
for key in base:
    raw = json.loads((OUT / "final_evaluation" / f"{key}.json").read_text(encoding="utf-8"))["raw"]
    categories, examples = Counter(), []
    for row in raw:
        for query, gold, pred, initial in zip(row["queries"], row["gold"], row["diffusion"]["4"], row["initializer"]):
            if pred == gold:
                continue
            categories[f"{gold}->{pred}"] += 1
            if len(examples) < 4:
                examples.append({"query": query, "gold": gold, "predicted": pred, "initializer": initial,
                                 "facts": row["facts"], "seed": row["seed"], "graph_key": row["graph_key"]})
    errors[key] = {"count": sum(categories.values()), "confusions": dict(categories), "examples": examples}
    inspected[key] = {"raw_graph_seed_traces": len(raw), "retained_error_examples": len(examples)}
result = {"status": "ROOT_CPU_TERMINAL_REVIEW_COMPLETE", "optimizer_updates_in_review": 0,
    "forwards_in_review": 0, "checkpoint_sha256": meta["sha256"], "checkpoint_bytes": meta["bytes"],
    "fullstate_safe_load_and_hash_verified": True, "full_recovery_keys_present": sorted(required),
    "frozen_tensor_changes": frozen_changes, "changed_trainable_tensor_count": len(changed_trainable),
    "changed_trainable_examples": changed_trainable[:8],
    "matched_noise17": matched, "paired_gain_absolute": paired_gain,
    "prospective_paired_gain_target": .05, "paired_gain_goal_pass": paired_gain >= .05,
    "actual_exposure_reconstructed_exactly": True, "attempted_row_exposures": sum(map(len, batches)),
    "actual_id_exposure": exposure, "amp_skips": sum(e["amp_skipped"] for e in history),
    "input_utf8_bytes": payload["exposure"], "query_label_exposures": payload["query_exposure"],
    "training_step_seconds_sum": sum(e["step_seconds"] for e in history),
    "gradient_norm_max": max(e["gradient_norm_before_clip"] for e in history),
    "final_raw_errors": errors, "raw_review_scope": inspected,
    "e2_final_holdout_used": False, "broad_model_completion_claimed": False}
atomic_json(OUT / "root_result.json", result)
print(json.dumps({k: v for k, v in result.items() if k not in ("final_raw_errors", "full_recovery_keys_present", "raw_review_scope", "actual_id_exposure")}, indent=2))
