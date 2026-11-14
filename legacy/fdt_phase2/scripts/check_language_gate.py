from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.next_tools import generate_ids, load_model_and_tokenizer, safe_torch_load


FACT_CASES = (
    ("The capital of France is", ("paris",)),
    ("The opposite of hot is", ("cold",)),
    ("Two plus two equals", ("four", "4")),
    ("Water freezes at zero degrees", ("celsius",)),
    ("The largest planet in the Solar System is", ("jupiter",)),
    ("A triangle has three", ("sides",)),
    ("The color of healthy grass is usually", ("green",)),
    ("The day after Monday is", ("tuesday",)),
)

COHERENCE_PROMPTS = (
    "A reliable scientific explanation should",
    "When a friend asks for practical advice, a helpful answer",
    "The purpose of testing computer software is",
    "In a small town near the coast, the residents",
)


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def repetition_rate(tokens: list[int], width: int = 3) -> float:
    if len(tokens) < width:
        return 0.0
    grams = [tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)]
    return 1.0 - len(set(grams)) / max(len(grams), 1)


@torch.inference_mode()
def complete(model, tokenizer, config, prompt: str, max_new_tokens: int, device: torch.device):
    ids = list(tokenizer.encode(prompt, add_special_tokens=False))[-config.max_seq_len :]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    budget = min(max_new_tokens, max(config.max_seq_len - len(ids), 0))
    generated = generate_ids(
        model,
        input_ids,
        budget,
        greedy=True,
        repetition_penalty=1.12,
        use_cache=True,
    )
    new_ids = generated[0, len(ids) :].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True), new_ids


def runtime_metrics(
    checkpoint_path: Path, checkpoint_tokens: int, metrics_dir: Path | None = None
) -> dict:
    candidates = []
    directories = [checkpoint_path.parent]
    if metrics_dir is not None and metrics_dir not in directories:
        directories.append(metrics_dir)
    for directory in directories:
        runtime_path = directory / "onk_runtime.json"
        if runtime_path.exists():
            payload = json.loads(runtime_path.read_text(encoding="utf-8"))
            promotion = dict(payload.get("promotion_metrics", {}))
            if promotion:
                candidates.append(promotion)

        training_log_path = directory / "training_log.jsonl"
        if training_log_path.exists():
            with training_log_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    tokens_seen = int(row.get("tokens_seen", -1))
                    if row.get("event") == "validation" and tokens_seen <= checkpoint_tokens:
                        candidates.append(row)

    if not candidates:
        return {}
    return max(candidates, key=lambda row: int(row.get("tokens_seen", -1)))


@torch.inference_mode()
def evaluate_validation_metrics(model, config, checkpoint: dict, device: torch.device, rows: int):
    dataset_dir = checkpoint.get("train_config", {}).get("dataset_dir")
    if not dataset_dir:
        return {}
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = ROOT / dataset_dir
    shards = sorted((dataset_dir / "shards" / "validation").glob("*.pt"))
    if not shards:
        return {}

    ids_rows, mask_rows = [], []
    remaining = rows
    for shard in shards:
        payload = safe_torch_load(shard)
        take = min(remaining, len(payload["input_ids"]))
        ids_rows.append(payload["input_ids"][:take].long())
        mask_rows.append(
            payload.get("attention_mask", torch.ones_like(payload["input_ids"]))[:take].long()
        )
        remaining -= take
        if remaining <= 0:
            break
    if not ids_rows:
        return {}

    model.eval()
    for module in model.modules():
        if module.__class__.__name__.endswith("FuzzyAnchorLayer"):
            module.runtime_anchor_metrics = True
    ids = torch.cat(ids_rows)
    masks = torch.cat(mask_rows)
    loss_sum = 0.0
    token_count = 0
    correct = 0
    stats_rows = []
    batch_size = 2
    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size].to(device)
        batch_mask = masks[start : start + batch_size].to(device)
        output = model(batch_ids, attention_mask=batch_mask)
        logits = output["logits"][:, :-1].float()
        labels = batch_ids[:, 1:]
        valid = batch_mask[:, 1:].bool() & labels.ne(config.pad_token_id)
        losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
        loss_sum += float(losses[valid].sum())
        token_count += int(valid.sum())
        correct += int((logits.argmax(dim=-1).eq(labels) & valid).sum())
        stats_rows.extend(output.get("anchor_stats", []))

    result = {
        "validation_loss": loss_sum / max(token_count, 1),
        "next_token_accuracy": correct / max(token_count, 1),
        "validation_tokens": token_count,
    }
    if stats_rows:
        entropy = torch.stack([row.entropy.detach().float() for row in stats_rows]).mean()
        usage = torch.stack([row.usage_prob.detach().float() for row in stats_rows]).mean(0)
        top1 = torch.stack([row.top1_membership.detach().float() for row in stats_rows]).mean()
        result.update(
            entropy=float(entropy),
            entropy_normalized=float(entropy / max(math.log(max(config.top_k, 2)), 1e-8)),
            effective_k=float(entropy.exp()),
            dead_anchor_ratio=float((usage < 1e-4).float().mean()),
            top1_membership=float(top1),
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate chat/RLM training on actual base-language capability")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-type", default="fdt_optimized")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "language_gate.json")
    parser.add_argument("--max-validation-loss", type=float, default=5.2)
    parser.add_argument("--min-next-token-accuracy", type=float, default=0.20)
    parser.add_argument("--min-fact-rate", type=float, default=0.375)
    parser.add_argument("--min-retrieval-rate", type=float, default=0.30)
    parser.add_argument("--max-repetition-rate", type=float, default=0.20)
    parser.add_argument("--max-dead-anchor-ratio", type=float, default=0.10)
    parser.add_argument("--min-routing-entropy", type=float, default=0.25)
    parser.add_argument("--max-routing-entropy", type=float, default=0.85)
    parser.add_argument("--validation-rows", type=int, default=32)
    parser.add_argument(
        "--allow-accuracy-transition",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow chat SFT after 1B tokens when accuracy and routing are healthy.",
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    device = torch.device(args.device)
    model, tokenizer, config, checkpoint = load_model_and_tokenizer(
        checkpoint_path, device, args.model_type
    )

    fact_rows = []
    for prompt, gold_options in FACT_CASES:
        completion, tokens = complete(model, tokenizer, config, prompt, 12, device)
        normalized = normalize(completion)
        correct = any(normalize(gold) in normalized for gold in gold_options)
        fact_rows.append({"prompt": prompt, "completion": completion, "correct": correct, "tokens": len(tokens)})

    rng = random.Random(20260728)
    retrieval_rows = []
    for index in range(10):
        code = f"QX{rng.randrange(1000, 9999)}"
        if index < 5:
            prompt = f"Memo: The access code is {code}. Therefore the access code is"
        else:
            prompt = f"Context: The access code is {code}.\nQuestion: What is the access code?\nAnswer:"
        completion, tokens = complete(model, tokenizer, config, prompt, 12, device)
        correct = normalize(code) in normalize(completion)
        retrieval_rows.append(
            {"prompt": prompt, "gold": code, "completion": completion, "correct": correct, "tokens": len(tokens)}
        )

    coherence_rows = []
    for prompt in COHERENCE_PROMPTS:
        completion, tokens = complete(model, tokenizer, config, prompt, 48, device)
        coherence_rows.append(
            {
                "prompt": prompt,
                "completion": completion,
                "nonempty": bool(completion.strip()),
                "repetition_rate": repetition_rate(tokens),
            }
        )

    output_dir = checkpoint.get("train_config", {}).get("output_dir")
    metrics_dir = None
    if output_dir:
        metrics_dir = Path(output_dir)
        if not metrics_dir.is_absolute():
            metrics_dir = ROOT / metrics_dir
    promotion = runtime_metrics(
        checkpoint_path, int(checkpoint.get("tokens_seen", 0)), metrics_dir=metrics_dir
    )
    required_metrics = {
        "validation_loss",
        "next_token_accuracy",
        "dead_anchor_ratio",
        "entropy_normalized",
    }
    if not required_metrics.issubset(promotion):
        promotion = {
            **promotion,
            **evaluate_validation_metrics(model, config, checkpoint, device, args.validation_rows),
        }
    validation_loss = float(promotion.get("validation_loss", float("inf")))
    accuracy = float(promotion.get("next_token_accuracy", 0.0))
    fact_rate = sum(row["correct"] for row in fact_rows) / len(fact_rows)
    retrieval_rate = sum(row["correct"] for row in retrieval_rows) / len(retrieval_rows)
    mean_repetition = sum(row["repetition_rate"] for row in coherence_rows) / len(coherence_rows)
    nonempty_rate = sum(row["nonempty"] for row in coherence_rows) / len(coherence_rows)

    checks = {
        "validation_loss": validation_loss <= args.max_validation_loss,
        "next_token_accuracy": accuracy >= args.min_next_token_accuracy,
        "fact_rate": fact_rate >= args.min_fact_rate,
        "retrieval_rate": retrieval_rate >= args.min_retrieval_rate,
        "repetition_rate": mean_repetition <= args.max_repetition_rate,
        "nonempty_rate": nonempty_rate == 1.0,
    }
    routing_checks = {
        "dead_anchor_ratio": float(promotion.get("dead_anchor_ratio", 1.0)) <= args.max_dead_anchor_ratio,
        "entropy_lower_bound": float(promotion.get("entropy_normalized", 0.0)) >= args.min_routing_entropy,
        "entropy_upper_bound": float(promotion.get("entropy_normalized", 1.0)) <= args.max_routing_entropy,
    }
    strict_capability_pass = all(checks.values())
    accuracy_transition_pass = (
        args.allow_accuracy_transition
        and checks["next_token_accuracy"]
        and checks["nonempty_rate"]
        and all(routing_checks.values())
    )
    passed = strict_capability_pass or accuracy_transition_pass
    decision_path = "strict_capability" if strict_capability_pass else (
        "accuracy_transition" if accuracy_transition_pass else "failed"
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "decision_path": decision_path,
        "checkpoint": str(checkpoint_path),
        "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        "metrics": {
            "validation_loss": validation_loss,
            "next_token_accuracy": accuracy,
            "fact_rate": fact_rate,
            "retrieval_rate": retrieval_rate,
            "mean_repetition_rate": mean_repetition,
            "nonempty_rate": nonempty_rate,
        },
        "thresholds": {
            "max_validation_loss": args.max_validation_loss,
            "min_next_token_accuracy": args.min_next_token_accuracy,
            "min_fact_rate": args.min_fact_rate,
            "min_retrieval_rate": args.min_retrieval_rate,
            "max_repetition_rate": args.max_repetition_rate,
            "max_dead_anchor_ratio": args.max_dead_anchor_ratio,
            "min_routing_entropy": args.min_routing_entropy,
            "max_routing_entropy": args.max_routing_entropy,
        },
        "checks": checks,
        "routing_checks": routing_checks,
        "strict_capability_pass": strict_capability_pass,
        "accuracy_transition_pass": accuracy_transition_pass,
        "fact_rows": fact_rows,
        "retrieval_rows": retrieval_rows,
        "coherence_rows": coherence_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision_path": decision_path,
                "metrics": report["metrics"],
                "checks": checks,
                "routing_checks": routing_checks,
            },
            indent=2,
        )
    )
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
