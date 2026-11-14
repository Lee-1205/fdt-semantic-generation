from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_base_model_strict as strict_v2
import evaluate_base_model_strict_v3 as strict_v3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint: Path,
    payload: dict[str, torch.Tensor],
    factual_end: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, dict[int, list[dict]]]:
    model, tokenizer, _, metadata = strict_v2.load_checkpoint_model(checkpoint, device)
    eos_id = int(tokenizer.eos_token_id)
    rows: dict[int, list[dict]] = {}
    for start in range(0, factual_end, batch_size):
        end = min(start + batch_size, factual_end)
        ids = payload["input_ids"][start:end].to(device=device, dtype=torch.long)
        labels = payload["labels"][start:end].to(device=device, dtype=torch.long)
        masks = payload["attention_mask"][start:end].to(device=device, dtype=torch.long)
        logits = model(ids, attention_mask=masks)["logits"][:, :-1].float()
        targets = labels[:, 1:]
        valid = (targets != -100) & masks[:, 1:].bool()
        log_probs = F.log_softmax(logits, dim=-1)
        probabilities = log_probs.exp()
        entropy = -(probabilities * log_probs).sum(dim=-1)
        top_values, top_ids = logits.topk(2, dim=-1)
        safe_targets = targets.masked_fill(~valid, 0)
        target_logits = logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        target_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        target_ranks = (logits > target_logits.unsqueeze(-1)).sum(dim=-1) + 1

        for offset in range(end - start):
            row_id = start + offset
            positions = valid[offset].nonzero(as_tuple=False).flatten().tolist()
            records = []
            for answer_position, position in enumerate(positions):
                target_id = int(targets[offset, position])
                top1_id = int(top_ids[offset, position, 0])
                records.append({
                    "answer_position": answer_position,
                    "sequence_position": position + 1,
                    "target_id": target_id,
                    "target_token": tokenizer.decode([target_id], skip_special_tokens=False),
                    "target_is_eos": target_id == eos_id,
                    "target_log_probability": float(target_log_probs[offset, position]),
                    "target_probability": math.exp(float(target_log_probs[offset, position])),
                    "target_rank": int(target_ranks[offset, position]),
                    "top1_id": top1_id,
                    "top1_token": tokenizer.decode([top1_id], skip_special_tokens=False),
                    "top1_probability": math.exp(float(log_probs[offset, position, top1_id])),
                    "top1_correct": top1_id == target_id,
                    "top1_target_logit_margin": float(
                        top_values[offset, position, 0] - target_logits[offset, position]
                    ),
                    "top1_top2_logit_margin": float(
                        top_values[offset, position, 0] - top_values[offset, position, 1]
                    ),
                    "entropy": float(entropy[offset, position]),
                })
            rows[row_id] = records
        print(json.dumps({"event": "factual_logits", "checkpoint": checkpoint.name, "rows": end}), flush=True)
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metadata, rows


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(records: list[dict]) -> dict:
    return {
        "tokens": len(records),
        "target_nll": -mean([row["candidate"]["target_log_probability"] for row in records]),
        "target_log_probability_delta": mean([
            row["candidate"]["target_log_probability"] - row["baseline"]["target_log_probability"]
            for row in records
        ]),
        "top1_accuracy_delta": mean([
            float(row["candidate"]["top1_correct"]) - float(row["baseline"]["top1_correct"])
            for row in records
        ]),
        "mean_target_rank_delta": mean([
            row["candidate"]["target_rank"] - row["baseline"]["target_rank"] for row in records
        ]),
        "mean_top1_target_margin_delta": mean([
            row["candidate"]["top1_target_logit_margin"]
            - row["baseline"]["top1_target_logit_margin"]
            for row in records
        ]),
        "mean_entropy_delta": mean([
            row["candidate"]["entropy"] - row["baseline"]["entropy"] for row in records
        ]),
        "top1_gains": sum(
            row["candidate"]["top1_correct"] and not row["baseline"]["top1_correct"] for row in records
        ),
        "top1_losses": sum(
            row["baseline"]["top1_correct"] and not row["candidate"]["top1_correct"] for row in records
        ),
        "changed_top1_identity": sum(
            row["baseline"]["top1_id"] != row["candidate"]["top1_id"] for row in records
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v15", type=Path, required=True)
    parser.add_argument("--v16", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    manifest_path = args.dataset / "manifest.json"
    shard_path = args.dataset / "shards" / "validation" / "shard_000000.pt"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    categories = manifest["validation"]["categories"]
    if list(categories)[0] != "factual_qa" or int(categories["factual_qa"]) != 400:
        raise ValueError("Unexpected factual validation layout")
    payload = strict_v3.load_capability_payload(args.dataset)
    device = torch.device(args.device)
    v15_meta, v15_rows = evaluate_checkpoint(
        args.v15, payload, 400, device, args.batch_size
    )
    v16_meta, v16_rows = evaluate_checkpoint(
        args.v16, payload, 400, device, args.batch_size
    )
    if set(v15_rows) != set(v16_rows) or len(v15_rows) != 400:
        raise ValueError("Factual row identity mismatch")

    paired = []
    for row_id in sorted(v15_rows):
        left, right = v15_rows[row_id], v16_rows[row_id]
        if len(left) != len(right):
            raise ValueError(f"Token count mismatch for row {row_id}")
        for baseline, candidate in zip(left, right):
            if baseline["target_id"] != candidate["target_id"]:
                raise ValueError(f"Target mismatch for row {row_id}")
            paired.append({
                "row_id": row_id,
                "answer_tokens": len(left),
                "answer_position": baseline["answer_position"],
                "target_id": baseline["target_id"],
                "target_token": baseline["target_token"],
                "target_is_eos": baseline["target_is_eos"],
                "baseline": baseline,
                "candidate": candidate,
            })

    length_bins = {
        "1_2": [row for row in paired if row["answer_tokens"] <= 2],
        "3_4": [row for row in paired if 3 <= row["answer_tokens"] <= 4],
        "5_7": [row for row in paired if 5 <= row["answer_tokens"] <= 7],
        "8_plus": [row for row in paired if row["answer_tokens"] >= 8],
    }
    position_bins = {
        "first": [row for row in paired if row["answer_position"] == 0],
        "second": [row for row in paired if row["answer_position"] == 1],
        "third_fourth": [row for row in paired if 2 <= row["answer_position"] <= 3],
        "fifth_plus": [row for row in paired if row["answer_position"] >= 4],
        "eos": [row for row in paired if row["target_is_eos"]],
        "non_eos": [row for row in paired if not row["target_is_eos"]],
    }
    disagreement = [
        row for row in paired
        if row["candidate"]["target_log_probability"] > row["baseline"]["target_log_probability"]
        and row["baseline"]["top1_correct"] and not row["candidate"]["top1_correct"]
    ]
    payload_out = {
        "status": "REAL_CHECKPOINT_LOGIT_DIAGNOSTIC_COMPLETED",
        "training_performed": False,
        "inputs": {
            "v15": {**v15_meta, "sha256": sha256(args.v15)},
            "v16": {**v16_meta, "sha256": sha256(args.v16)},
            "dataset": str(args.dataset.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "validation_shard_sha256": sha256(shard_path),
            "rows": 400,
        },
        "summary": {
            "overall": summarize(paired),
            "by_answer_length": {key: summarize(value) for key, value in length_bins.items()},
            "by_answer_position": {key: summarize(value) for key, value in position_bins.items()},
            "lower_nll_but_lost_top1": len(disagreement),
        },
        "lower_nll_but_lost_top1_rows": disagreement,
        "tokens": paired,
        "interpretation": {
            "target_rank": "1 plus count of vocabulary logits strictly above the target logit",
            "top1_target_logit_margin": "top1 logit minus target logit; zero when target is top1",
            "entropy": "full-vocabulary next-token entropy in nats",
            "scope": "teacher-forced factual completion tokens including supervised EOS",
        },
    }
    atomic_json(args.output, payload_out)
    print(json.dumps({"event": "complete", "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
