from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_base_model_strict as strict_v2
import evaluate_base_model_strict_v3 as strict_v3


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_group_sha256(state: dict[str, torch.Tensor], marker: str) -> str:
    digest = hashlib.sha256()
    for name in sorted(key for key in state if marker in key):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.inference_mode()
def trace_checkpoint(
    path: Path,
    payload: dict[str, torch.Tensor],
    ranges: list[tuple[str, int, int]],
    rows_per_category: int,
    device: torch.device,
) -> dict:
    raw = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    anchor_parameter_hash = tensor_group_sha256(raw["model_state_dict"], ".anchor.")
    del raw
    model, _, _, metadata = strict_v2.load_checkpoint_model(path, device)
    for block in model.blocks:
        if block.anchor is not None:
            block.anchor.config.enable_diagnostics = True

    layer_category: dict[str, dict] = {}
    route_vectors: dict[str, list[int]] = {}
    for category, start, end in ranges:
        selected = list(range(start, min(end, start + rows_per_category)))
        for row_id in selected:
            ids = payload["input_ids"][row_id : row_id + 1].to(device=device, dtype=torch.long)
            mask = payload["attention_mask"][row_id : row_id + 1].to(device=device, dtype=torch.long)
            output = model(ids, attention_mask=mask)
            active = mask[0].bool()
            for layer_number, stats in enumerate(output["anchor_stats"]):
                key = f"anchor_layer_{layer_number}"
                top1 = stats.indices[0, active, 0].detach().cpu().tolist()
                top1_weight = stats.membership[0, active, 0].float().detach().cpu().tolist()
                vector_key = f"{category}:{key}"
                route_vectors.setdefault(vector_key, []).extend(int(value) for value in top1)
                slot = layer_category.setdefault(vector_key, {
                    "tokens": 0,
                    "top1_membership_sum": 0.0,
                    "top1_counts": Counter(),
                })
                slot["tokens"] += len(top1)
                slot["top1_membership_sum"] += sum(top1_weight)
                slot["top1_counts"].update(top1)
        print(json.dumps({"event": "routing_category", "checkpoint": path.name, "category": category}), flush=True)

    summaries = {}
    for key, slot in layer_category.items():
        tokens = slot["tokens"]
        probabilities = [count / tokens for count in slot["top1_counts"].values()]
        summaries[key] = {
            "tokens": tokens,
            "unique_top1_anchors": len(slot["top1_counts"]),
            "top1_membership": slot["top1_membership_sum"] / tokens,
            "top_anchor_frequency": max(probabilities) if probabilities else 0.0,
            "top_anchors": slot["top1_counts"].most_common(8),
        }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "metadata": metadata,
        "checkpoint_sha256": file_sha256(path),
        "anchor_parameter_sha256": anchor_parameter_hash,
        "summaries": summaries,
        "route_vectors": route_vectors,
    }


def compare(left: dict, right: dict) -> dict:
    if set(left["route_vectors"]) != set(right["route_vectors"]):
        raise ValueError("Route vector keys differ")
    results = {}
    agreements = []
    for key in sorted(left["route_vectors"]):
        a, b = left["route_vectors"][key], right["route_vectors"][key]
        if len(a) != len(b):
            raise ValueError(f"Route vector length differs: {key}")
        agreement = sum(x == y for x, y in zip(a, b)) / max(len(a), 1)
        results[key] = {"tokens": len(a), "top1_anchor_agreement": agreement}
        agreements.extend(x == y for x, y in zip(a, b))
    return {
        "anchor_parameters_identical": left["anchor_parameter_sha256"] == right["anchor_parameter_sha256"],
        "overall_top1_anchor_agreement": sum(agreements) / max(len(agreements), 1),
        "by_category_layer": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--v15", type=Path, required=True)
    parser.add_argument("--v16", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-category", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    payload = strict_v3.load_capability_payload(args.dataset)
    ranges = strict_v3.capability_ranges(manifest)
    device = torch.device(args.device)
    traces = {
        "base_520m": trace_checkpoint(args.base, payload, ranges, args.rows_per_category, device),
        "v15": trace_checkpoint(args.v15, payload, ranges, args.rows_per_category, device),
        "v16": trace_checkpoint(args.v16, payload, ranges, args.rows_per_category, device),
    }
    report = {
        "status": "ROUTING_TRACE_COMPLETED",
        "training_performed": False,
        "dataset": str(args.dataset.resolve()),
        "manifest_sha256": file_sha256(args.dataset / "manifest.json"),
        "validation_shard_sha256": file_sha256(args.dataset / "shards" / "validation" / "shard_000000.pt"),
        "rows_per_category": args.rows_per_category,
        "checkpoints": {
            name: {key: value for key, value in trace.items() if key != "route_vectors"}
            for name, trace in traces.items()
        },
        "comparisons": {
            "520m_vs_v15": compare(traces["base_520m"], traces["v15"]),
            "v15_vs_v16": compare(traces["v15"], traces["v16"]),
            "520m_vs_v16": compare(traces["base_520m"], traces["v16"]),
        },
        "interpretation": "Anchor parameters can remain identical while token routes change because shared hidden states feeding the frozen router change.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"event": "complete", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
