from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prepare_capability_completion_v16 import CATEGORY_NAMES, PROFILES, profile_indices


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), strict=False)


def factual_rows(payload: dict) -> dict[int, dict]:
    rows = payload["evaluation"]["capability_teacher_forced"]["rows"]
    selected = {int(row["row_id"]): row for row in rows if row["category"] == "factual_qa"}
    if len(selected) != 400:
        raise ValueError(f"Expected 400 factual rows, found {len(selected)}")
    return selected


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"rows": 0}
    tokens = sum(row["tokens"] for row in rows)
    return {
        "rows": len(rows),
        "tokens": tokens,
        "mean_tokens": tokens / len(rows),
        "mean_loss_delta": sum(row["loss_delta"] for row in rows) / len(rows),
        "mean_token_loss_delta": sum(row["loss_sum_delta"] for row in rows) / tokens,
        "mean_row_top1_delta": sum(row["top1_rate_delta"] for row in rows) / len(rows),
        "top1_token_delta": sum(row["top1_delta"] for row in rows),
        "improved_top1_rows": sum(row["top1_delta"] > 0 for row in rows),
        "worsened_top1_rows": sum(row["top1_delta"] < 0 for row in rows),
        "unchanged_top1_rows": sum(row["top1_delta"] == 0 for row in rows),
        "improved_loss_rows": sum(row["loss_delta"] < 0 for row in rows),
        "worsened_loss_rows": sum(row["loss_delta"] > 0 for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15", type=Path, required=True)
    parser.add_argument("--v16", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    v15 = load_json(args.v15)
    v16 = load_json(args.v16)
    expected_evaluator = "strict_base_v3_audited_2"
    for label, payload in (("v15", v15), ("v16", v16)):
        if payload["evaluator"]["version"] != expected_evaluator:
            raise ValueError(f"{label} evaluator mismatch")
        if payload["status"] != "EVALUATION_COMPLETED":
            raise ValueError(f"{label} evaluation incomplete")
    if v15["datasets"]["capability_validation_fingerprint"] != v16["datasets"]["capability_validation_fingerprint"]:
        raise ValueError("Capability validation fingerprints differ")

    base = factual_rows(v15)
    candidate = factual_rows(v16)
    if set(base) != set(candidate):
        raise ValueError("Factual row sets differ")

    paired = []
    for row_id in sorted(base):
        left, right = base[row_id], candidate[row_id]
        if int(left["tokens"]) != int(right["tokens"]):
            raise ValueError(f"Token count differs for row {row_id}")
        tokens = int(left["tokens"])
        loss_sum_delta = float(right["loss_sum"]) - float(left["loss_sum"])
        top1_delta = int(right["top1"]) - int(left["top1"])
        paired.append({
            "row_id": row_id,
            "tokens": tokens,
            "loss_sum_delta": loss_sum_delta,
            "loss_delta": loss_sum_delta / tokens,
            "top1_delta": top1_delta,
            "top1_rate_delta": top1_delta / tokens,
            "v15_top1": int(left["top1"]),
            "v16_top1": int(right["top1"]),
        })

    bins = {
        "1_2_tokens": [row for row in paired if row["tokens"] <= 2],
        "3_4_tokens": [row for row in paired if 3 <= row["tokens"] <= 4],
        "5_7_tokens": [row for row in paired if 5 <= row["tokens"] <= 7],
        "8_plus_tokens": [row for row in paired if row["tokens"] >= 8],
    }

    shard = args.dataset / "shards" / "train" / "shard_000000.pt"
    train = torch.load(shard, map_location="cpu", weights_only=False)
    profile = PROFILES["capability_heavy"]
    indices = profile_indices(
        train["category_ids"], profile, args.seed + 100 + len("capability_heavy")
    )
    category_rows = []
    total_completion = 0
    total_prompt_weighted = 0.0
    for category_id, category in enumerate(CATEGORY_NAMES):
        selected = indices[train["category_ids"][indices] == category_id]
        labels = train["labels"].index_select(0, selected) if len(selected) else train["labels"][:0]
        masks = train["attention_mask"].index_select(0, selected) if len(selected) else train["attention_mask"][:0]
        completion = int((labels != -100).sum().item())
        active = int(masks.sum().item())
        prompt = active - completion
        weighted_prompt = prompt * float(profile["prompt_lm_weight"])
        total_completion += completion
        total_prompt_weighted += weighted_prompt
        category_rows.append({
            "category": category,
            "rows": len(selected),
            "active_tokens": active,
            "completion_supervised_tokens": completion,
            "prompt_tokens": prompt,
            "prompt_lm_weight": float(profile["prompt_lm_weight"]),
            "weighted_prompt_token_equivalent": weighted_prompt,
        })
    effective_total = total_completion + total_prompt_weighted
    for row in category_rows:
        row["completion_fraction"] = row["completion_supervised_tokens"] / total_completion
        row["effective_supervision_fraction"] = (
            row["completion_supervised_tokens"] + row["weighted_prompt_token_equivalent"]
        ) / effective_total

    output = {
        "status": "DIAGNOSTIC_COMPLETED",
        "purpose": "Explain V16 factual loss improvement with top1 decline and quantify supervision bias.",
        "inputs": {
            "v15": str(args.v15),
            "v15_sha256": sha256(args.v15),
            "v16": str(args.v16),
            "v16_sha256": sha256(args.v16),
            "dataset": str(args.dataset),
            "dataset_manifest_sha256": sha256(args.dataset / "manifest.json"),
            "train_shard_sha256": sha256(shard),
            "evaluator_version": expected_evaluator,
            "capability_validation_fingerprint": v15["datasets"]["capability_validation_fingerprint"],
        },
        "factual_paired": {
            "overall": aggregate(paired),
            "by_answer_token_length": {name: aggregate(rows) for name, rows in bins.items()},
            "rows": paired,
        },
        "v16_capability_heavy_supervision": {
            "profile": "capability_heavy",
            "rows": len(indices),
            "completion_supervised_tokens": total_completion,
            "weighted_prompt_token_equivalent": total_prompt_weighted,
            "effective_supervision_token_equivalent": effective_total,
            "categories": category_rows,
        },
        "interpretation_limits": [
            "Stored strict artifacts contain row-level loss and top1 counts, not per-position logits or margins.",
            "Confidence calibration, changed token identities, and token-position effects require a real-checkpoint logits diagnostic.",
            "Effective supervision fractions are token-weight equivalents, not measured gradient norms.",
            "Training-source templates cannot be inferred from the fixed validation row aggregates.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "status": output["status"]}, indent=2))


if __name__ == "__main__":
    main()
