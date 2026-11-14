"""Fixed read-only proof: crossed expression families and DDIM discretization.

No optimizer is constructed, no checkpoint is modified, and final E2 rows are
not loaded. The step sweep is development diagnosis, not a new final benchmark.
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
from pathlib import Path
import time

import torch

from fdt_local.data import collate, make_split
from fdt_local.model_masked import Config, FDT100
from fdt_local.objectives import compute_metrics
from fdt_local.runner import configure_determinism, source_digest
from fdt_local.storage import CheckpointStore, ensure_space, sha256
from learning_io_v2 import atomic_json

ROOT = Path(__file__).resolve().parents[1]


class Budget:
    def __init__(self, seconds, forwards):
        self.start = time.monotonic()
        self.deadline = self.start + seconds
        self.limit, self.attempted = forwards, 0

    def reserve(self, count):
        if time.monotonic() >= self.deadline or self.attempted + count > self.limit:
            raise RuntimeError("fixed read-only inference budget exhausted")
        self.attempted += count


def prepare(output):
    if output.exists():
        raise FileExistsError("new proof output required")
    parent = ROOT / "runs/e1_masked_learning_20261026_recovery_v2"
    plan = {"status": "PREPARED_NOT_EXECUTED", "script_sha256": sha256(Path(__file__)),
        "runtime_source_sha256": source_digest(), "io_sha256": sha256(ROOT / "scripts/learning_io_v2.py"),
        "checkpoint_marker": str(parent / "model/complete_001336.json"),
        "checkpoint_sha256": "d57d06ec7cc1ebb4f5ccbf365673554efc8a705043d492dd802fff1b6c7c8edb",
        "crossed_panels": str(ROOT / "research/artifacts/e2_review_20261026_v1/crossed_development_panels.json"),
        "reference_large_eval": str(parent / "evaluation/larger_graphs.json"),
        "wall_seconds": 300, "max_forward_attempts": 700, "optimizer_attempts": 0,
        "steps": [4, 8, 16], "noise_seeds": [17, 29, 43], "batch_size": 4,
        "large_graphs": {"split": "test", "graphs": 16, "seed": 80001, "n_range": [7, 8]},
        "output_dir": str(output), "final_e2_holdout_used": False,
        "interpretation": "Paired development diagnosis; no training or solver-selection final-test claim."}
    for key in ("crossed_panels", "reference_large_eval"):
        plan[key + "_sha256"] = sha256(Path(plan[key]))
    output.mkdir()
    atomic_json(output / "plan.json", plan)
    print(json.dumps({"status": plan["status"], "plan_sha256": sha256(output / "plan.json")}))


@torch.no_grad()
def panel(model, items, seeds, step_counts, budget):
    predictions = {"initializer": []}
    predictions.update({f"ddim_{steps}": [] for steps in step_counts})
    ys, masks, hops, raw = [], [], [], []
    ground_confusion = torch.zeros(4, 4, dtype=torch.long)
    bytes_ = 0
    for start in range(0, len(items), 4):
        batch = items[start:start + 4]
        features, gold, exposure = collate(batch, model.c, "cuda")
        if set(features) != {"tokens", "roles", "fact_mask", "queries", "query_mask"}:
            raise ValueError("unexpected neural input")
        bytes_ += exposure
        budget.reserve(1)
        with torch.autocast("cuda", dtype=torch.float16):
            encoded = model.encode(features)
            guess = encoded["ground"].argmax(-1)
            valid = features["fact_mask"].bool()
            y, p = gold["ground"][valid].cpu(), guess[valid].cpu()
            ground_confusion += torch.bincount(4 * y + p, minlength=16).reshape(4, 4)
            for seed in seeds:
                generator = torch.Generator(device="cuda").manual_seed(seed + start)
                noise = torch.randn(encoded["init"].shape, device="cuda", generator=generator)
                initial = model.readout(encoded["init"]).argmax(-1).cpu()
                predictions["initializer"].append(initial)
                outcomes = {}
                for steps in step_counts:
                    budget.reserve(steps)
                    schedule = tuple(1 - i / steps for i in range(steps + 1))
                    state = model.rollout(encoded, noise, schedule)
                    outcomes[steps] = model.readout(state).argmax(-1).cpu()
                    predictions[f"ddim_{steps}"].append(outcomes[steps])
                ys.append(gold["labels"].cpu())
                masks.append(features["query_mask"].cpu())
                hops.append(gold["hops"].cpu())
                for i, item in enumerate(batch):
                    count = len(item["queries"])
                    raw.append({"graph_key": item["graph_key"], "family_id": item.get("family_id"),
                        "facts": item["facts"], "queries": item["queries"], "gold": item["labels"],
                        "ground_predictions": guess[i, :len(item["facts"])].cpu().tolist(),
                        "initializer": initial[i, :count].tolist(),
                        "diffusion": {str(k): v[i, :count].tolist() for k, v in outcomes.items()}, "seed": seed})
    y, mask, h = torch.cat(ys), torch.cat(masks), torch.cat(hops)
    denominator = ground_confusion.sum(1) + ground_confusion.sum(0)
    f1 = 2 * ground_confusion.diag().double() / denominator.clamp_min(1)
    return {"graphs": len(items), "input_utf8_bytes_once": bytes_, "ground_accuracy": float(ground_confusion.diag().sum() / ground_confusion.sum()),
        "ground_label_order": ["GREATER", "LESS", "NOT_GREATER", "NOT_LESS"],
        "ground_confusion": ground_confusion.tolist(), "ground_class_f1": f1.tolist(),
        "ground_macro_f1": float(f1.mean()), "noise_seeds": seeds,
        "results": {key: compute_metrics(torch.cat(values), y, mask, h) for key, values in predictions.items()},
        "raw": raw, "gold_never_in_neural_input": True,
        "queries_and_seeds_correlated_within_graph": True}


def run(plan_path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output = Path(plan["output_dir"])
    if (output / "clock.json").exists():
        raise FileExistsError("one fixed proof launch only")
    for path, expected in ((Path(__file__), plan["script_sha256"]),
        (ROOT / "scripts/learning_io_v2.py", plan["io_sha256"]),
        (Path(plan["crossed_panels"]), plan["crossed_panels_sha256"]),
        (Path(plan["reference_large_eval"]), plan["reference_large_eval_sha256"])):
        if sha256(path) != expected:
            raise ValueError("sealed proof input changed")
    if source_digest() != plan["runtime_source_sha256"]:
        raise ValueError("runtime changed")
    ensure_space(output, 64 * 1024**2)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("CUDA and 12GiB free required")
    budget = Budget(plan["wall_seconds"], plan["max_forward_attempts"])
    atomic_json(output / "clock.json", {"started_unix": time.time(), "fixed_wall_seconds": 300, "optimizer_attempts": 0})
    configure_determinism()
    marker = Path(plan["checkpoint_marker"])
    meta, payload = CheckpointStore(marker.parent).verify(marker)
    if meta["sha256"] != plan["checkpoint_sha256"] or payload["phase"] != 3:
        raise ValueError("verified final E1 model required")
    model = FDT100(Config(**payload["contract"]["config"]))
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).cuda().eval()
    del payload
    results = {}
    crossed = json.loads(Path(plan["crossed_panels"]).read_text(encoding="utf-8"))
    for name, items in crossed.items():
        results[name] = panel(model, items, plan["noise_seeds"], [4], budget)
        atomic_json(output / (name + ".json"), {"checkpoint_sha256": meta["sha256"], **results[name]})
    values = plan["large_graphs"]
    items = make_split(values["split"], values["graphs"], seed=values["seed"], n_range=tuple(values["n_range"]))
    sweep = panel(model, items, plan["noise_seeds"], plan["steps"], budget)
    reference = json.loads(Path(plan["reference_large_eval"]).read_text(encoding="utf-8"))["metrics"]["diffusion"]
    reproduced = sweep["results"]["ddim_4"]["confusion"] == reference["confusion"]
    atomic_json(output / "large_ddim_sweep.json", {"checkpoint_sha256": meta["sha256"],
                "four_step_reference_confusion_exact": reproduced, **sweep})
    if not reproduced:
        raise ValueError("4-step reference reproduction failed; do not interpret step sweep")
    summary = {"status": "COMPLETE", "checkpoint_sha256": meta["sha256"], "optimizer_attempts": 0,
        "forward_attempts": budget.attempted, "seconds": time.monotonic() - budget.start,
        "four_step_reference_confusion_exact": reproduced, "final_e2_holdout_used": False,
        "crossed": {k: {field: value[field] for field in ("ground_accuracy", "ground_macro_f1", "results")} for k, value in results.items()},
        "large_sweep": sweep["results"], "no_training_or_architecture_superiority_claim": True}
    atomic_json(output / "summary.json", summary)
    print(json.dumps({"status": "COMPLETE", "forward_attempts": budget.attempted, "seconds": summary["seconds"],
                      "reference_exact": reproduced, "optimizer_attempts": 0}))


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
