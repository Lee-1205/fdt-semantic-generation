from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_base_model_strict as v2
from evaluate_base_model_strict_v3 import capability_ranges, load_capability_payload


def token_class(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "whitespace"
    if all(character in "{}[],:\"" for character in stripped):
        return "json_punctuation"
    if stripped in {"true", "false", "null"}:
        return "json_literal"
    if stripped.replace(".", "", 1).replace("-", "", 1).isdigit():
        return "number"
    if any(character.isalpha() for character in stripped):
        return "text"
    return "other"


@torch.inference_mode()
def collect_predictions(
    checkpoint: Path,
    payload: dict[str, torch.Tensor],
    start: int,
    end: int,
    device: torch.device,
    batch_size: int,
) -> tuple[dict, object]:
    model, tokenizer, _, metadata = v2.load_checkpoint_model(checkpoint, device)
    rows: dict[int, list[dict]] = defaultdict(list)
    for batch_start in range(start, end, batch_size):
        batch_end = min(batch_start + batch_size, end)
        ids = payload["input_ids"][batch_start:batch_end].to(device=device, dtype=torch.long)
        labels = payload["labels"][batch_start:batch_end].to(device=device, dtype=torch.long)
        masks = payload["attention_mask"][batch_start:batch_end].to(device=device, dtype=torch.long)
        logits = model(ids, attention_mask=masks)["logits"][:, :-1].float()
        targets = labels[:, 1:]
        valid = (targets != -100) & masks[:, 1:].bool()
        safe_targets = targets.masked_fill(~valid, 0)
        predictions = logits.argmax(dim=-1)
        target_nll = F.cross_entropy(
            logits.transpose(1, 2), safe_targets, reduction="none"
        )

        for offset in range(batch_end - batch_start):
            row_id = batch_start + offset
            positions = valid[offset].nonzero(as_tuple=False).flatten().tolist()
            for position in positions:
                rows[row_id].append(
                    {
                        "position": int(position),
                        "target": int(safe_targets[offset, position]),
                        "prediction": int(predictions[offset, position]),
                        "target_nll": float(target_nll[offset, position]),
                    }
                )
        print(json.dumps({"event": "json_diagnostic", "rows": batch_end - start}), flush=True)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"metadata": metadata, "rows": rows}, tokenizer


def compare(base: dict, candidate: dict, tokenizer) -> dict:
    transition_counts = Counter()
    class_transitions: dict[str, Counter] = defaultdict(Counter)
    target_flip_counts = Counter()
    prediction_flip_counts = Counter()
    nll_deltas: list[float] = []
    row_reports = []

    for row_id in sorted(base["rows"]):
        base_tokens = base["rows"][row_id]
        candidate_tokens = candidate["rows"][row_id]
        if len(base_tokens) != len(candidate_tokens):
            raise ValueError(f"Token count mismatch in row {row_id}")
        row_transition = Counter()
        row_nll_delta = 0.0
        for base_token, candidate_token in zip(base_tokens, candidate_tokens):
            if (
                base_token["position"] != candidate_token["position"]
                or base_token["target"] != candidate_token["target"]
            ):
                raise ValueError(f"Token alignment mismatch in row {row_id}")
            target = base_token["target"]
            base_correct = base_token["prediction"] == target
            candidate_correct = candidate_token["prediction"] == target
            transition = f"{int(base_correct)}->{int(candidate_correct)}"
            target_text = tokenizer.decode([target], skip_special_tokens=False)
            category = token_class(target_text)
            transition_counts[transition] += 1
            class_transitions[category][transition] += 1
            row_transition[transition] += 1
            delta = candidate_token["target_nll"] - base_token["target_nll"]
            nll_deltas.append(delta)
            row_nll_delta += delta
            if base_correct and not candidate_correct:
                target_flip_counts[target_text] += 1
                prediction_flip_counts[
                    tokenizer.decode([candidate_token["prediction"]], skip_special_tokens=False)
                ] += 1
        row_reports.append(
            {
                "row_id": row_id,
                "tokens": len(base_tokens),
                "transitions": dict(row_transition),
                "mean_target_nll_delta": row_nll_delta / max(len(base_tokens), 1),
            }
        )

    total = sum(transition_counts.values())
    return {
        "base": base["metadata"],
        "candidate": candidate["metadata"],
        "tokens": total,
        "transitions": dict(transition_counts),
        "base_top1": (transition_counts["1->1"] + transition_counts["1->0"]) / total,
        "candidate_top1": (transition_counts["1->1"] + transition_counts["0->1"]) / total,
        "mean_target_nll_delta": sum(nll_deltas) / max(len(nll_deltas), 1),
        "class_transitions": {key: dict(value) for key, value in class_transitions.items()},
        "top_lost_target_tokens": target_flip_counts.most_common(40),
        "top_wrong_candidate_tokens_on_losses": prediction_flip_counts.most_common(40),
        "rows": row_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=ROOT / "prepared_data" / "capability_curriculum_5k",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    ranges = capability_ranges(manifest)
    json_range = next((start, end) for category, start, end in ranges if category == "json")
    payload = load_capability_payload(args.dataset_dir)
    device = torch.device(args.device)

    base, tokenizer = collect_predictions(
        args.base, payload, *json_range, device=device, batch_size=args.batch_size
    )
    candidate, _ = collect_predictions(
        args.candidate, payload, *json_range, device=device, batch_size=args.batch_size
    )
    report = compare(base, candidate, tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: report[key] for key in (
        "tokens", "transitions", "base_top1", "candidate_top1", "mean_target_nll_delta"
    )}, indent=2))


if __name__ == "__main__":
    main()
