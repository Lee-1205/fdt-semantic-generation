"""Bounded paired E1/E2 diagnosis; zero optimizer, no E2 final rows."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import torch
from fdt_local.data import make_split
from fdt_local.model_masked import Config, FDT100
from fdt_local.runner import configure_determinism, source_digest
from fdt_local.storage import CheckpointStore, ensure_space, sha256
from learning_io_v2 import atomic_json
from probe_e1_readonly import Budget, panel

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("language.", "phrase_router.", "ground_head.")


def subset_digest(model, trainable_subset):
    result = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.startswith(PREFIXES) == trainable_subset:
            result.update(key.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def prepare(output):
    if output.exists():
        raise FileExistsError("new comparison output required")
    e1 = ROOT / "runs/e1_masked_learning_20261026_recovery_v2"
    e2 = ROOT / "runs/e2_lexical_grounding_20261026_v1"
    status = json.loads((e2 / "final_status.json").read_text(encoding="utf-8"))
    if status["status"] != "COMPLETE" or status["stage_updates"] != 128:
        raise ValueError("completed E2 required")
    paths = {"e1_marker": e1 / "model/complete_001336.json", "e2_marker": Path(status["last_marker"]),
             "dev": ROOT / "research/artifacts/e2_lexical_20261026_v2/dev.json",
             "train": ROOT / "research/artifacts/e2_lexical_20261026_v2/train.json",
             "promoted": ROOT / "research/artifacts/e2_review_20261026_v1/crossed_development_panels.json",
             "reference_new": e1 / "evaluation/new_graphs.json", "reference_large": e1 / "evaluation/larger_graphs.json",
             "common_code": ROOT / "scripts/probe_e1_readonly.py", "io_code": ROOT / "scripts/learning_io_v2.py"}
    plan = {"script_sha256": sha256(Path(__file__)), "core_sha256": source_digest(),
            "paths": {k: str(v) for k, v in paths.items()}, "file_sha256": {k: sha256(v) for k, v in paths.items()},
            "e1_checkpoint_sha256": "d57d06ec7cc1ebb4f5ccbf365673554efc8a705043d492dd802fff1b6c7c8edb",
            "e2_checkpoint_sha256": status["last_checkpoint_sha256"], "output_dir": str(output),
            "wall_seconds": 300, "max_forward_attempts": 400, "optimizer_attempts": 0,
            "noise_seeds": [17, 29, 43], "e2_final_used": False,
            "notice": "E1 dev64 baseline is measured now on its preserved frozen checkpoint; not a preregistered numerical baseline."}
    output.mkdir()
    atomic_json(output / "plan.json", plan)
    print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "plan_sha256": sha256(output / "plan.json")}))


def run(plan_path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output = Path(plan["output_dir"])
    if (output / "clock.json").exists():
        raise FileExistsError("one fixed comparison launch only")
    if sha256(Path(__file__)) != plan["script_sha256"] or source_digest() != plan["core_sha256"]:
        raise ValueError("comparison source changed")
    paths = {k: Path(v) for k, v in plan["paths"].items()}
    if any(sha256(v) != plan["file_sha256"][k] for k, v in paths.items()):
        raise ValueError("sealed comparison input changed")
    ensure_space(output, 64 * 1024**2)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("CUDA and12GiBfree required")
    configure_determinism()
    budget = Budget(plan["wall_seconds"], plan["max_forward_attempts"])
    atomic_json(output / "clock.json", {"started_unix": time.time(), "wall_seconds": 300, "optimizer_attempts": 0})
    train = json.loads(paths["train"].read_text(encoding="utf-8"))
    sampled = []
    families = sorted({r["family_id"] for r in train})
    for family in families:
        sampled.extend([r for r in train if r["family_id"] == family][:4])
    panels = {"new_dev64": json.loads(paths["dev"].read_text(encoding="utf-8")),
              "train8families32_seen_after_e2": sampled,
              "promoted_old_dev16_seen_after_e2": json.loads(paths["promoted"].read_text(encoding="utf-8"))["dev_graph_dev_family"],
              "old_new16": make_split("test", 16, seed=60001),
              "old_large16": make_split("test", 16, seed=80001, n_range=(7, 8))}
    all_results, digests = {}, {}
    for model_name in ("e1", "e2"):
        marker = paths[model_name + "_marker"]
        meta, payload = CheckpointStore(marker.parent).verify(marker)
        if meta["sha256"] != plan[model_name + "_checkpoint_sha256"]:
            raise ValueError("model checkpoint identity changed")
        model = FDT100(Config(**payload["contract"]["config"]))
        model.load_state_dict(payload["model"], strict=True)
        digests[model_name] = {"frozen_subset": subset_digest(model, False), "adapted_subset": subset_digest(model, True)}
        model.requires_grad_(False).cuda().eval()
        del payload
        all_results[model_name] = {}
        for panel_name, rows in panels.items():
            steps, seeds = ([4], plan["noise_seeds"]) if panel_name.startswith("old_") else ([], [17])
            measured = panel(model, rows, seeds, steps, budget)
            atomic_json(output / f"{model_name}_{panel_name}.json", {"checkpoint_sha256": meta["sha256"], **measured})
            all_results[model_name][panel_name] = {k: measured[k] for k in ("graphs", "ground_accuracy", "ground_macro_f1", "results")}
        del model
        gc.collect()
        torch.cuda.empty_cache()
    reproduced = True
    for name, key in (("old_new16", "reference_new"), ("old_large16", "reference_large")):
        reference = json.loads(paths[key].read_text(encoding="utf-8"))["metrics"]["diffusion"]["confusion"]
        reproduced &= all_results["e1"][name]["results"]["ddim_4"]["confusion"] == reference
    frozen_exact = digests["e1"]["frozen_subset"] == digests["e2"]["frozen_subset"]
    adapted_changed = digests["e1"]["adapted_subset"] != digests["e2"]["adapted_subset"]
    summary = {"status": "COMPLETE", "optimizer_attempts": 0, "forward_attempts": budget.attempted,
        "seconds": time.monotonic() - budget.start, "e2_final_used": False,
        "e1_reference_confusions_exact": reproduced, "all_nonwhitelisted_tensors_exact": frozen_exact,
        "adapted_tensors_changed": adapted_changed, "subset_digests": digests,
        "models": all_results, "new_dev_e1_baseline_measured_posthoc_on_preserved_model": True}
    if not reproduced or not frozen_exact or not adapted_changed:
        summary["status"] = "CONTRACT_FAILURE_DO_NOT_PROMOTE"
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "forward_attempts": budget.attempted,
        "seconds": summary["seconds"], "optimizer_attempts": 0, "frozen_exact": frozen_exact, "references_exact": reproduced}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path)
    group.add_argument("--run-plan", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve())
    else:
        run(args.run_plan.resolve())
