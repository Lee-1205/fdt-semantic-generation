from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402
from train_fdt_v4_curriculum_bridge import (  # noqa: E402
    atomic_json,
    require_c_path,
    sha256_file,
)


def load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(
        require_c_path(path, "checkpoint"),
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )


def load_fixed52(
    dataset_dir: Path,
    source_path: Path,
    tokenizer_dir: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, Any]]]:
    dataset_dir = require_c_path(dataset_dir, "fixed52 dataset")
    source_path = require_c_path(source_path, "fixed52 source")
    tokenizer_dir = require_c_path(tokenizer_dir, "tokenizer")
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("rows", 0)) != 52:
        raise ValueError("fixed52 manifest must declare exactly 52 rows")
    if sha256_file(source_path) != str(manifest["source_sha256"]).upper():
        raise ValueError("fixed52 source hash mismatch")
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if sha256_file(tokenizer_path) != str(manifest["tokenizer_json_sha256"]).upper():
        raise ValueError("fixed52 tokenizer hash mismatch")
    shard_spec = manifest["shards"]
    if len(shard_spec) != 1:
        raise ValueError("fixed52 requires exactly one shard")
    shard_path = require_c_path(dataset_dir / shard_spec[0]["file"], "fixed52 shard")
    if sha256_file(shard_path) != str(shard_spec[0]["sha256"]).upper():
        raise ValueError("fixed52 shard hash mismatch")
    payload = torch.load(shard_path, map_location="cpu", weights_only=True, mmap=True)
    if int(payload["input_ids"].size(0)) != 52:
        raise ValueError("fixed52 shard must contain exactly 52 rows")
    source_rows = [
        json.loads(line)
        for line in source_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(source_rows) != 52:
        raise ValueError("fixed52 source must contain exactly 52 rows")
    tokenizer = load_tokenizer(str(tokenizer_dir))
    metadata: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows):
        metadata.append(
            {
                "row_index": index,
                "row_identity_sha256": manifest["row_identity_sha256"][index],
                "category": str(row.get("category", "unknown")),
                "prompt_chars": len(str(row["prompt"])),
                "target_chars": len(str(row["target"])),
                "prompt_tokens_separate": len(
                    tokenizer.encode(str(row["prompt"]), add_special_tokens=False)
                ),
                "target_tokens_separate": len(
                    tokenizer.encode(str(row["target"]), add_special_tokens=False)
                ),
            }
        )
    return payload, {
        "directory": str(dataset_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "shard": str(shard_path),
        "shard_sha256": sha256_file(shard_path),
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "tokenizer_json": str(tokenizer_path),
        "tokenizer_json_sha256": sha256_file(tokenizer_path),
        "ordered_row_identity_sha256": manifest["ordered_row_identity_sha256"],
    }, metadata


@torch.inference_mode()
def score_rows(
    model: torch.nn.Module,
    payload: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    model.eval()
    for index, (ids_row, mask_row) in enumerate(
        zip(payload["input_ids"], payload["attention_mask"])
    ):
        active = ids_row[mask_row.bool()].to(device=device, dtype=torch.long)
        tensor = active.unsqueeze(0)
        logits = model(tensor, attention_mask=torch.ones_like(tensor))["logits"][
            :, :-1
        ].float()
        labels = tensor[:, 1:]
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        )
        nll = float(losses.mean().cpu())
        top1 = float(logits.argmax(dim=-1).eq(labels).float().mean().cpu())
        if not math.isfinite(nll):
            raise FloatingPointError(f"nonfinite fixed52 NLL at row {index}")
        rows.append(
            {
                **metadata[index],
                "scored_tokens": int(labels.numel()),
                "nll": nll,
                "top1": top1,
            }
        )
        del active, tensor, logits, labels, losses
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = sum(int(row["scored_tokens"]) for row in rows)
    return {
        "rows": len(rows),
        "tokens": tokens,
        "row_mean_nll": sum(float(row["nll"]) for row in rows) / len(rows),
        "row_mean_top1": sum(float(row["top1"]) for row in rows) / len(rows),
        "token_weighted_nll": sum(
            float(row["nll"]) * int(row["scored_tokens"]) for row in rows
        )
        / tokens,
        "token_weighted_top1": sum(
            float(row["top1"]) * int(row["scored_tokens"]) for row in rows
        )
        / tokens,
    }


def aggregate_categories(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["category"])].append(row)
    return {name: aggregate(grouped[name]) for name in sorted(grouped)}


def score_checkpoint(
    checkpoint: Path,
    alphas: list[float],
    payload: dict[str, torch.Tensor],
    metadata: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_payload = load_checkpoint(checkpoint)
    config = ModelConfig(**checkpoint_payload["model_config"])
    if config.model_type != "fdt_v4":
        raise ValueError(f"expected FDT v4 checkpoint: {checkpoint}")
    model = build_model(config)
    model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    del checkpoint_payload
    model = model.to(device=device, dtype=torch.float32).eval()
    controls: list[dict[str, Any]] = []
    try:
        for alpha in alphas:
            model.set_transition_alpha(float(alpha))
            model.set_legacy_position_scale(1.0)
            model.set_anchor_recency_bias(4.0)
            model.config.routing_logit_quantization = 0.0
            rows = score_rows(model, payload, metadata, device)
            controls.append(
                {
                    "alpha": float(alpha),
                    "controls": {
                        "legacy_position_scale": 1.0,
                        "anchor_recency_bias": 4.0,
                        "routing_logit_quantization": 0.0,
                    },
                    "aggregate": aggregate(rows),
                    "categories": aggregate_categories(rows),
                    "rows": rows,
                }
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "path": str(checkpoint),
        "sha256": sha256_file(checkpoint),
        "controls": controls,
    }


def control_by_alpha(report: dict[str, Any], alpha: float) -> dict[str, Any]:
    for control in report["controls"]:
        if abs(float(control["alpha"]) - float(alpha)) < 1e-12:
            return control
    raise KeyError(alpha)


def build_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    alpha: float,
) -> dict[str, Any]:
    baseline_control = control_by_alpha(baseline, alpha)
    candidate_control = control_by_alpha(candidate, alpha)
    deltas: list[dict[str, Any]] = []
    for before, after in zip(baseline_control["rows"], candidate_control["rows"]):
        if before["row_identity_sha256"] != after["row_identity_sha256"]:
            raise RuntimeError("fixed52 row identity/order changed")
        deltas.append(
            {
                "row_index": before["row_index"],
                "row_identity_sha256": before["row_identity_sha256"],
                "category": before["category"],
                "prompt_chars": before["prompt_chars"],
                "prompt_tokens_separate": before["prompt_tokens_separate"],
                "target_tokens_separate": before["target_tokens_separate"],
                "scored_tokens": before["scored_tokens"],
                "baseline_nll": before["nll"],
                "candidate_nll": after["nll"],
                "nll_delta": float(after["nll"]) - float(before["nll"]),
                "baseline_top1": before["top1"],
                "candidate_top1": after["top1"],
                "top1_delta": float(after["top1"]) - float(before["top1"]),
            }
        )
    category_deltas: dict[str, Any] = {}
    for category in sorted({str(row["category"]) for row in deltas}):
        group = [row for row in deltas if row["category"] == category]
        token_count = sum(int(row["scored_tokens"]) for row in group)
        category_deltas[category] = {
            "rows": len(group),
            "tokens": token_count,
            "row_mean_nll_delta": sum(float(row["nll_delta"]) for row in group)
            / len(group),
            "token_weighted_nll_delta": sum(
                float(row["nll_delta"]) * int(row["scored_tokens"])
                for row in group
            )
            / token_count,
            "row_mean_top1_delta": sum(float(row["top1_delta"]) for row in group)
            / len(group),
        }
    return {
        "alpha": float(alpha),
        "aggregate": {
            "row_mean_nll_delta": candidate_control["aggregate"]["row_mean_nll"]
            - baseline_control["aggregate"]["row_mean_nll"],
            "token_weighted_nll_delta": candidate_control["aggregate"]
            ["token_weighted_nll"]
            - baseline_control["aggregate"]["token_weighted_nll"],
            "row_mean_top1_delta": candidate_control["aggregate"]["row_mean_top1"]
            - baseline_control["aggregate"]["row_mean_top1"],
            "token_weighted_top1_delta": candidate_control["aggregate"]
            ["token_weighted_top1"]
            - baseline_control["aggregate"]["token_weighted_top1"],
        },
        "categories": category_deltas,
        "worst_rows_by_nll_delta": sorted(
            deltas, key=lambda row: float(row["nll_delta"]), reverse=True
        )[:12],
        "best_rows_by_nll_delta": sorted(
            deltas, key=lambda row: float(row["nll_delta"])
        )[:12],
        "rows": deltas,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only fixed52 row/category attribution for FDT v4.1"
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--fixed52-dir", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.23, 0.235, 0.24])
    args = parser.parse_args()

    output = require_c_path(args.output, "output")
    if output.exists() or output.with_name(output.name + ".tmp").exists():
        raise FileExistsError(f"output is not fresh: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    alphas = sorted(set(float(alpha) for alpha in args.alphas))
    if not alphas or any(alpha < 0.0 or alpha > 1.0 for alpha in alphas):
        raise ValueError("alphas must be within [0, 1]")

    payload, dataset, metadata = load_fixed52(
        args.fixed52_dir, args.source, args.tokenizer
    )
    baseline = score_checkpoint(
        require_c_path(args.baseline, "baseline checkpoint"),
        alphas,
        payload,
        metadata,
        device,
    )
    candidate = score_checkpoint(
        require_c_path(args.candidate, "candidate checkpoint"),
        alphas,
        payload,
        metadata,
        device,
    )
    deltas = [build_delta(baseline, candidate, alpha) for alpha in alphas]
    report = {
        "schema": "fdt_v4_1_fixed52_attribution_v1",
        "status": "PASS",
        "operation": "read_only_original_unquantized_fp32",
        "dataset": dataset,
        "alphas": alphas,
        "baseline": baseline,
        "candidate": candidate,
        "candidate_minus_baseline": deltas,
        "training_or_optimizer_update": False,
    }
    atomic_json(output, report)
    digest = sha256_file(output)
    sidecar = output.with_name(output.name + ".sha256")
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    temporary.write_text(f"{digest} *{output.name}\n", encoding="ascii")
    os.replace(temporary, sidecar)
    summary = {
        "output": str(output),
        "sha256": digest,
        "alphas": alphas,
        "delta_at_max_alpha": deltas[-1]["aggregate"],
        "category_delta_at_max_alpha": deltas[-1]["categories"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
