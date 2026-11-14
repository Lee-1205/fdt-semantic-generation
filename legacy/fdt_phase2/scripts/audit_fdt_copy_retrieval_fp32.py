from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.next_tools import generate_ids, load_model_and_tokenizer  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def repeated_ngram_fraction(tokens: list[int], width: int = 4) -> float:
    if len(tokens) < width:
        return 0.0
    grams = [tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def random_code(rng: random.Random, length: int) -> str:
    alphabet = string.ascii_uppercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(length))


def build_cases(seed: int, repeats: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    serial = 0
    words = (
        "harbor", "violet", "circuit", "forest", "silver", "window", "planet", "quiet",
        "lantern", "bridge", "winter", "garden", "signal", "river", "copper", "meadow",
    )
    for length in (8, 16, 32, 64):
        for _ in range(repeats):
            serial += 1
            target = random_code(rng, length)
            rows.append({
                "id": f"literal-{serial:04d}",
                "family": "literal_copy",
                "length_bucket": str(length),
                "distractors": 0,
                "target_position": "single",
                "prompt": f"Copy the target exactly and output nothing else.\nTarget: {target}\nAnswer:",
                "answer": target,
            })
    for count in (4, 8, 16):
        for _ in range(repeats):
            serial += 1
            selected = [rng.choice(words) for _ in range(count)]
            target = " ".join(selected)
            rows.append({
                "id": f"words-{serial:04d}",
                "family": "word_sequence_copy",
                "length_bucket": str(count),
                "distractors": 0,
                "target_position": "single",
                "prompt": f"Repeat the phrase exactly and output nothing else.\nPhrase: {target}\nAnswer:",
                "answer": target,
            })
    for distractors in (4, 16, 32):
        for position_name, target_index in (("front", 0), ("middle", distractors // 2), ("back", distractors - 1)):
            for _ in range(repeats):
                serial += 1
                codes = [random_code(rng, 12) for _ in range(distractors)]
                target = codes[target_index]
                records = [f"Record {index + 1}: key K{index + 1:02d} has value {value}." for index, value in enumerate(codes)]
                rows.append({
                    "id": f"retrieval-{serial:04d}",
                    "family": "distractor_retrieval",
                    "length_bucket": "12",
                    "distractors": distractors,
                    "target_position": position_name,
                    "prompt": "\n".join(records) + f"\nReturn only the value for key K{target_index + 1:02d}.\nAnswer:",
                    "answer": target,
                })
    return rows


@torch.inference_mode()
def score_case(model: Any, tokenizer: Any, config: Any, row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    prompt_ids = list(tokenizer.encode(row["prompt"], add_special_tokens=False))
    answer_ids = list(tokenizer.encode(" " + row["answer"], add_special_tokens=False))
    target_ids = answer_ids + [int(config.eos_token_id)]
    if len(prompt_ids) + len(target_ids) > config.max_seq_len:
        raise ValueError(f"case exceeds context: {row['id']}")

    sequence = torch.tensor([prompt_ids + target_ids], device=device, dtype=torch.long)
    logits = model(sequence, attention_mask=torch.ones_like(sequence))["logits"][0].float()
    target_rows = logits[len(prompt_ids) - 1 : len(prompt_ids) - 1 + len(target_ids)]
    targets = torch.tensor(target_ids, device=device, dtype=torch.long)
    target_logits = target_rows.gather(1, targets[:, None]).squeeze(1)
    log_probs = target_rows.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    ranks = 1 + (target_rows > target_logits[:, None]).sum(-1)
    top1 = target_rows.argmax(-1)

    prompt_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    generated = generate_ids(
        model,
        prompt_tensor,
        max_new_tokens=min(len(target_ids) + 12, config.max_seq_len - len(prompt_ids)),
        greedy=True,
        use_cache=True,
    )
    completion_ids = generated[0, len(prompt_ids) :].tolist()
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    observed = completion.strip()
    expected = row["answer"].strip()
    answer_count = len(answer_ids)
    answer_top1 = top1[:answer_count]
    return {
        **row,
        "prompt_tokens": len(prompt_ids),
        "answer_tokens": answer_count,
        "completion": completion,
        "free_exact": observed == expected,
        "free_prefix": observed.startswith(expected),
        "teacher_answer_top1_exact": bool((answer_top1 == targets[:answer_count]).all().item()),
        "teacher_answer_top1_rate": float((answer_top1 == targets[:answer_count]).float().mean().item()),
        "teacher_answer_mean_log_probability": float(log_probs[:answer_count].mean().item()),
        "teacher_answer_mean_rank": float(ranks[:answer_count].float().mean().item()),
        "teacher_answer_worst_rank": int(ranks[:answer_count].max().item()),
        "eos_top1": bool(top1[-1].item() == config.eos_token_id),
        "eos_log_probability": float(log_probs[-1].item()),
        "generated_tokens": len(completion_ids),
        "repeated_4gram_fraction": repeated_ngram_fraction(completion_ids),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[f"family:{row['family']}"].append(row)
        groups[f"family:{row['family']}|length:{row['length_bucket']}"].append(row)
        if row["family"] == "distractor_retrieval":
            groups[f"retrieval:d{row['distractors']}|{row['target_position']}"] .append(row)
    result = {}
    for name, selected in groups.items():
        count = len(selected)
        result[name] = {
            "rows": count,
            "free_exact": sum(row["free_exact"] for row in selected),
            "free_exact_rate": sum(row["free_exact"] for row in selected) / count,
            "free_prefix_rate": sum(row["free_prefix"] for row in selected) / count,
            "teacher_answer_top1_rate": sum(row["teacher_answer_top1_rate"] for row in selected) / count,
            "teacher_answer_mean_log_probability": sum(row["teacher_answer_mean_log_probability"] for row in selected) / count,
            "teacher_answer_mean_rank": sum(row["teacher_answer_mean_rank"] for row in selected) / count,
            "eos_top1_rate": sum(row["eos_top1"] for row in selected) / count,
            "mean_repeated_4gram_fraction": sum(row["repeated_4gram_fraction"] for row in selected) / count,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact copy and distractor retrieval on original FP32 FDT.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if checkpoint.drive.upper() != "C:" or output.drive.upper() != "C:":
        raise ValueError("audit is C:-only")
    device = torch.device(args.device)
    model, tokenizer, config, payload = load_model_and_tokenizer(checkpoint, device, "fdt_v3", "")
    model = model.to(device=device, dtype=torch.float32).eval()
    del payload
    cases = build_cases(args.seed, args.repeats)
    rows = []
    for index, case in enumerate(cases, 1):
        rows.append(score_case(model, tokenizer, config, case, device))
        print(json.dumps({"event": "case", "completed": index, "total": len(cases)}), flush=True)

    report = {
        "schema_version": 1,
        "status": "AUDIT_COMPLETED",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "precision": "fp32_original",
        "quantization": "none",
        "training_performed": False,
        "construction": {
            "seed": args.seed,
            "repeats": args.repeats,
            "rows": len(rows),
            "families": ["literal_copy", "word_sequence_copy", "distractor_retrieval"],
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    atomic_json(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "overall": report["summary"]["overall"]}, indent=2))


if __name__ == "__main__":
    main()
