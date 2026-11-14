from __future__ import annotations

import argparse
import ast
import gc
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.config import ModelConfig
from fdt_rlm.data.interleaving import bucket_sequence_counts
from fdt_rlm.models import build_model
from fdt_rlm.next_tools import generate_ids, safe_torch_load
from fdt_rlm.tokenization import load_tokenizer


FACT_CASES = (
    ("The capital of France is", (" Paris", " London", " Berlin", " Madrid"), 0),
    ("The capital of Japan is", (" Tokyo", " Beijing", " Seoul", " Bangkok"), 0),
    ("The largest planet in the Solar System is", (" Jupiter", " Mars", " Earth", " Venus"), 0),
    ("The chemical symbol for gold is", (" Au", " Ag", " Fe", " Cu"), 0),
    ("The chemical symbol for oxygen is", (" O", " H", " N", " C"), 0),
    ("Water freezes at", (" zero degrees Celsius", " fifty degrees Celsius", " one hundred degrees Celsius", " ten degrees Celsius"), 0),
    ("A triangle has", (" three sides", " four sides", " five sides", " six sides"), 0),
    ("The opposite of hot is", (" cold", " tall", " fast", " bright"), 0),
    ("The primary language of Brazil is", (" Portuguese", " Spanish", " French", " English"), 0),
    ("The planet known as the Red Planet is", (" Mars", " Mercury", " Saturn", " Neptune"), 0),
    ("A week contains", (" seven days", " five days", " ten days", " twelve days"), 0),
    ("The square root of nine is", (" three", " four", " five", " six"), 0),
    ("The animal that produces wool is a", (" sheep", " horse", " tiger", " dolphin"), 0),
    ("The gas humans need to breathe is", (" oxygen", " helium", " neon", " argon"), 0),
    ("The first month of the year is", (" January", " March", " June", " December"), 0),
    ("The instrument used to measure temperature is a", (" thermometer", " telescope", " compass", " microscope"), 0),
)

CODE_PROMPTS = (
    "def add(a, b):\n    return ",
    "def is_even(value):\n    return ",
    "def first_item(items):\n    if not items:\n        return None\n    return ",
    "def square(value):\n    return ",
    "def greet(name):\n    return f\"Hello, ",
    "def clamp(value, low, high):\n    return ",
    "def reverse_text(text):\n    return ",
    "def count_items(items):\n    return ",
)

CODE_CLOZE_CASES = (
    ("def add(a, b):\n    return", (" a + b", " a - b", " a * b", " b / a"), 0),
    ("def is_even(value):\n    return", (" value % 2 == 0", " value % 2 == 1", " value > 2", " value == 2"), 0),
    ("def square(value):\n    return", (" value * value", " value + value", " value / value", " value - value"), 0),
    ("def first_item(items):\n    return", (" items[0]", " items[-1]", " len(items)", " items[1:]"), 0),
    ("def reverse_text(text):\n    return", (" text[::-1]", " text[0]", " len(text)", " text.upper()"), 0),
    ("def count_items(items):\n    return", (" len(items)", " items[0]", " sum(items)", " list(items)"), 0),
    ("def clamp(value, low, high):\n    return", (" max(low, min(value, high))", " value + low + high", " min(low, high)", " max(value, high)"), 0),
    ("def greet(name):\n    return", (' f"Hello, {name}"', " name", ' "Hello"', " str()"), 0),
)

JSON_PROMPTS = (
    '{"name":"Ada","active":',
    '{"status":"ok","count":',
    '{"items":[1,2,3],"valid":',
    '{"language":"Python","version":',
    '{"user":{"id":7,"name":"Kim"},"enabled":',
    '{"result":{"score":0.9},"errors":',
)

GENERAL_PROMPTS = (
    "A careful scientific experiment should",
    "The difference between training and inference is",
    "In a quiet town near the mountains,",
    "When a computer program receives input,",
    "A useful way to solve a difficult problem is",
    "The purpose of a public library is",
    "During the winter, the lake",
    "To make the system more reliable, the engineer",
)


def load_checkpoint_model(path: Path, device: torch.device):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        checkpoint = safe_torch_load(path, "cpu")
    config = ModelConfig(**checkpoint["model_config"])
    tokenizer_name = checkpoint.get("train_config", {}).get("tokenizer_name", "") or "gpt2"
    tokenizer = load_tokenizer(tokenizer_name)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metadata = {
        "checkpoint": str(path.resolve()),
        "global_step": int(checkpoint.get("global_step", 0)),
        "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        "model_type": config.model_type,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "tokenizer": str(tokenizer_name),
    }
    del checkpoint
    gc.collect()
    model.to(device).eval()
    return model, tokenizer, config, metadata


def split_rows(dataset_dir: Path, split: str):
    ids_rows, mask_rows = [], []
    for path in sorted((dataset_dir / "shards" / split).glob("*.pt")):
        shard = safe_torch_load(path, "cpu")
        part = shard["input_ids"]
        ids_rows.append(part)
        mask_rows.append(shard.get("attention_mask", torch.ones_like(part)))
    return torch.cat(ids_rows), torch.cat(mask_rows)


def bucket_ranges(manifest: dict, split: str) -> list[tuple[str, int, int]]:
    order = tuple(manifest.get("bucket_order") or manifest.get("plan", {}).keys())
    counts = bucket_sequence_counts(manifest, split=split, bucket_order=order)
    ranges, start = [], 0
    for name, count in counts.items():
        ranges.append((name, start, start + count))
        start += count
    return ranges


def evenly_spaced(start: int, end: int, count: int) -> list[int]:
    available = max(end - start, 0)
    selected = min(count, available)
    if selected <= 0:
        return []
    if selected == 1:
        return [start]
    return [start + index * (available - 1) // (selected - 1) for index in range(selected)]


def routing_summary(stats_rows, top_k: int) -> dict:
    if not stats_rows:
        return {}
    entropies = torch.tensor([float(row.entropy.detach()) for row in stats_rows])
    top1 = torch.tensor([float(row.top1_membership.detach()) for row in stats_rows])
    usage = torch.stack([row.usage_prob.detach().float().cpu() for row in stats_rows]).mean(0)
    entropy = float(entropies.mean())
    return {
        "entropy": entropy,
        "entropy_normalized": entropy / math.log(max(top_k, 2)),
        "effective_k": float(math.exp(entropy)),
        "dead_anchor_ratio": float((usage < 1e-4).float().mean()),
        "top1_membership": float(top1.mean()),
        "top_anchor_frequency": float(usage.max()),
    }


@torch.inference_mode()
def heldout_metrics(model, config, ids, masks, ranges, rows_per_bucket: int, device):
    reports = {}
    overall = defaultdict(float)
    overall_stats = []
    for bucket, start, end in ranges:
        totals = defaultdict(float)
        stats_rows = []
        selected = evenly_spaced(start, end, rows_per_bucket)
        for row_index in selected:
            row_ids = ids[row_index].unsqueeze(0).to(device=device, dtype=torch.long)
            row_mask = masks[row_index].unsqueeze(0).to(device=device, dtype=torch.long)
            output = model(row_ids, attention_mask=row_mask)
            logits = output["logits"][:, :-1].float()
            labels = row_ids[:, 1:]
            valid = row_mask[:, 1:].bool() & (labels != config.pad_token_id)
            losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
            count = int(valid.sum())
            totals["loss_sum"] += float(losses[valid].sum())
            totals["tokens"] += count
            top = logits.topk(5, dim=-1).indices
            totals["top1"] += int(((top[..., 0] == labels) & valid).sum())
            totals["top5"] += int(((top == labels.unsqueeze(-1)).any(-1) & valid).sum())
            stats_rows.extend(output.get("anchor_stats", []))
        count = max(int(totals["tokens"]), 1)
        reports[bucket] = {
            "rows": len(selected),
            "tokens": int(totals["tokens"]),
            "loss": totals["loss_sum"] / count,
            "perplexity": math.exp(totals["loss_sum"] / count),
            "top1_accuracy": totals["top1"] / count,
            "top5_accuracy": totals["top5"] / count,
            "routing": routing_summary(stats_rows, config.top_k),
        }
        for key in ("loss_sum", "tokens", "top1", "top5"):
            overall[key] += totals[key]
        overall_stats.extend(stats_rows)
    count = max(int(overall["tokens"]), 1)
    reports["overall"] = {
        "rows": sum(report["rows"] for report in reports.values()),
        "tokens": int(overall["tokens"]),
        "loss": overall["loss_sum"] / count,
        "perplexity": math.exp(overall["loss_sum"] / count),
        "top1_accuracy": overall["top1"] / count,
        "top5_accuracy": overall["top5"] / count,
        "routing": routing_summary(overall_stats, config.top_k),
    }
    return reports


@torch.inference_mode()
def conditional_score(model, tokenizer, config, prompt: str, completion: str, device) -> dict:
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    completion_ids = list(tokenizer.encode(completion, add_special_tokens=False))
    if not prompt_ids or not completion_ids:
        return {"sum": -float("inf"), "mean": -float("inf"), "tokens": len(completion_ids)}
    overflow = max(len(prompt_ids) + len(completion_ids) - config.max_seq_len, 0)
    if overflow:
        prompt_ids = prompt_ids[overflow:]
    full = torch.tensor([prompt_ids + completion_ids], dtype=torch.long, device=device)
    logits = model(full, attention_mask=torch.ones_like(full))["logits"][0].float()
    start = len(prompt_ids) - 1
    scores = F.log_softmax(logits[start : start + len(completion_ids)], dim=-1)
    target = torch.tensor(completion_ids, dtype=torch.long, device=device)
    values = scores.gather(1, target.unsqueeze(1)).squeeze(1)
    return {"sum": float(values.sum()), "mean": float(values.mean()), "tokens": len(completion_ids)}


def evaluate_facts(model, tokenizer, config, device):
    rows = []
    for prompt, options, gold in FACT_CASES:
        scores = [conditional_score(model, tokenizer, config, prompt, option, device) for option in options]
        predicted = max(range(len(options)), key=lambda index: scores[index]["mean"])
        rows.append({
            "prompt": prompt,
            "options": list(options),
            "gold": gold,
            "predicted": predicted,
            "correct": predicted == gold,
            "scores": scores,
        })
    return {"accuracy": sum(row["correct"] for row in rows) / len(rows), "count": len(rows), "rows": rows}


def evaluate_code_cloze(model, tokenizer, config, device):
    rows = []
    for prompt, options, gold in CODE_CLOZE_CASES:
        scores = [conditional_score(model, tokenizer, config, prompt, option, device) for option in options]
        predicted = max(range(len(options)), key=lambda index: scores[index]["mean"])
        rows.append({
            "prompt": prompt,
            "options": list(options),
            "gold": gold,
            "predicted": predicted,
            "correct": predicted == gold,
            "scores": scores,
        })
    return {"accuracy": sum(row["correct"] for row in rows) / len(rows), "count": len(rows), "rows": rows}


def retrieval_prompt(tokenizer, code: str, distance: int, rng: random.Random) -> str:
    record = f"Archive entry: the verification code is {code}. "
    query = "For this archive entry, the verification code is"
    words = ("method result system sample process value record detail signal context analysis model "
             "data memory output input stable precise ordinary measured reference").split()
    filler = []
    while len(tokenizer.encode(" ".join(filler), add_special_tokens=False)) < distance:
        filler.append(words[rng.randrange(len(words))])
    return record + " ".join(filler) + ". " + query


def normalize_code(text: str) -> str:
    return "".join(character.lower() for character in text if character.isalnum())


@torch.inference_mode()
def greedy_completion(model, tokenizer, config, prompt: str, device, max_new_tokens: int) -> tuple[str, list[int]]:
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))[-(config.max_seq_len - max_new_tokens) :]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = generate_ids(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        greedy=True,
        repetition_penalty=1.05,
        use_cache=True,
    )
    new_ids = generated[0, len(prompt_ids) :].tolist()
    return tokenizer.decode(new_ids, skip_special_tokens=True), new_ids


def evaluate_retrieval(model, tokenizer, config, device, count: int):
    rng = random.Random(20260806)
    distances = (64, 512, 1400)
    rows = []
    for index in range(count):
        code = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8))
        decoys = ["".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(8)) for _ in range(3)]
        distance = distances[index % len(distances)]
        prompt = retrieval_prompt(tokenizer, code, distance, rng)
        options = [" " + code] + [" " + decoy for decoy in decoys]
        scores = [conditional_score(model, tokenizer, config, prompt, option, device) for option in options]
        predicted = max(range(len(options)), key=lambda candidate: scores[candidate]["mean"])
        completion, _ = greedy_completion(model, tokenizer, config, prompt, device, 12)
        greedy_exact = normalize_code(completion).startswith(normalize_code(code))
        rows.append({
            "distance": distance,
            "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
            "code": code,
            "rank_correct": predicted == 0,
            "greedy_exact": greedy_exact,
            "completion": completion,
            "scores": scores,
        })
    by_distance = {}
    for distance in distances:
        selected = [row for row in rows if row["distance"] == distance]
        by_distance[str(distance)] = {
            "count": len(selected),
            "rank_accuracy": sum(row["rank_correct"] for row in selected) / len(selected),
            "greedy_exact_rate": sum(row["greedy_exact"] for row in selected) / len(selected),
        }
    return {
        "count": len(rows),
        "rank_accuracy": sum(row["rank_correct"] for row in rows) / len(rows),
        "greedy_exact_rate": sum(row["greedy_exact"] for row in rows) / len(rows),
        "by_distance": by_distance,
        "rows": rows,
    }


def extract_json_prefix(text: str):
    decoder = json.JSONDecoder()
    try:
        value, end = decoder.raw_decode(text)
        return value, end
    except json.JSONDecodeError:
        return None, 0


def repetition_rate(tokens: list[int], width: int = 3) -> float:
    if len(tokens) < width:
        return 0.0
    grams = [tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def evaluate_generations(model, tokenizer, config, device):
    code_rows = []
    for prompt in CODE_PROMPTS:
        completion, tokens = greedy_completion(model, tokenizer, config, prompt, device, 48)
        combined = prompt + completion
        non_whitespace = bool(completion.strip())
        try:
            ast.parse(combined)
            parseable = True
        except SyntaxError:
            parseable = False
        code_rows.append({
            "prompt": prompt,
            "completion": completion,
            "non_whitespace": non_whitespace,
            "parseable": parseable,
            "meaningful_parseable": parseable and non_whitespace,
            "repetition_rate": repetition_rate(tokens),
        })

    json_rows = []
    for prompt in JSON_PROMPTS:
        completion, tokens = greedy_completion(model, tokenizer, config, prompt, device, 32)
        value, end = extract_json_prefix(prompt + completion)
        json_rows.append({
            "prompt": prompt,
            "completion": completion,
            "valid_json": value is not None,
            "parsed": value,
            "consumed_characters": end,
            "repetition_rate": repetition_rate(tokens),
        })

    general_rows = []
    for prompt in GENERAL_PROMPTS:
        completion, tokens = greedy_completion(model, tokenizer, config, prompt, device, 64)
        general_rows.append({
            "prompt": prompt,
            "completion": completion,
            "tokens": len(tokens),
            "repetition_rate": repetition_rate(tokens),
            "loop_free": repetition_rate(tokens) <= 0.20,
        })
    return {
        "python": {
            "count": len(code_rows),
            "parseable_rate": sum(row["parseable"] for row in code_rows) / len(code_rows),
            "non_whitespace_rate": sum(row["non_whitespace"] for row in code_rows) / len(code_rows),
            "meaningful_parseable_rate": sum(row["meaningful_parseable"] for row in code_rows) / len(code_rows),
            "mean_repetition_rate": sum(row["repetition_rate"] for row in code_rows) / len(code_rows),
            "rows": code_rows,
        },
        "json": {
            "count": len(json_rows),
            "valid_rate": sum(row["valid_json"] for row in json_rows) / len(json_rows),
            "rows": json_rows,
        },
        "general": {
            "count": len(general_rows),
            "loop_free_rate": sum(row["loop_free"] for row in general_rows) / len(general_rows),
            "mean_repetition_rate": sum(row["repetition_rate"] for row in general_rows) / len(general_rows),
            "rows": general_rows,
        },
    }


@torch.inference_mode()
def benchmark_context(model, tokenizer, config, device):
    seed = tokenizer.encode("The system records a stable measurement for later analysis. ", add_special_tokens=False)
    repeated = (seed * math.ceil(1792 / max(len(seed), 1)))[:1792]
    input_ids = torch.tensor([repeated], dtype=torch.long, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.perf_counter()
    output, cache = model.prefill(input_ids, torch.ones_like(input_ids))
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    next_id = output["logits"][:, -1].argmax(-1, keepdim=True)
    started = time.perf_counter()
    decoded = 0
    for _ in range(32):
        output, cache = model.decode_step(next_id, cache)
        next_id = output["logits"][:, -1].argmax(-1, keepdim=True)
        decoded += 1
    if device.type == "cuda":
        torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - started
    return {
        "context_tokens": 1792,
        "prefill_tokens_per_second": 1792 / max(prefill_seconds, 1e-9),
        "decode_tokens_per_second": decoded / max(decode_seconds, 1e-9),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
        "cache_backend": cache.get("backend"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict base-pretraining evaluation without instruction-format assumptions.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "prepared_data" / "v3_broad_additional_300m_combined_v2_24k_ctx2048")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows_per_bucket", type=int, default=12)
    parser.add_argument("--retrieval_cases", type=int, default=40)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer, config, metadata = load_checkpoint_model(args.checkpoint, device)
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    ids, masks = split_rows(args.dataset_dir, "validation")
    ranges = bucket_ranges(manifest, "validation")
    if ranges[-1][2] != ids.size(0):
        raise ValueError(f"Manifest validation rows {ranges[-1][2]} != loaded rows {ids.size(0)}")

    report = {
        "status": "PASS",
        "metadata": metadata,
        "evaluation": {
            "heldout": heldout_metrics(model, config, ids, masks, ranges, args.rows_per_bucket, device),
            "facts": evaluate_facts(model, tokenizer, config, device),
            "code_cloze": evaluate_code_cloze(model, tokenizer, config, device),
            "retrieval": evaluate_retrieval(model, tokenizer, config, device, args.retrieval_cases),
            "generation": evaluate_generations(model, tokenizer, config, device),
            "context_1792": benchmark_context(model, tokenizer, config, device),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "status": "PASS",
        "output": str(args.output),
        "tokens_seen": metadata["tokens_seen"],
        "heldout": report["evaluation"]["heldout"]["overall"],
        "fact_accuracy": report["evaluation"]["facts"]["accuracy"],
        "code_cloze_accuracy": report["evaluation"]["code_cloze"]["accuracy"],
        "retrieval_rank_accuracy": report["evaluation"]["retrieval"]["rank_accuracy"],
        "retrieval_greedy_exact": report["evaluation"]["retrieval"]["greedy_exact_rate"],
        "python_parseable": report["evaluation"]["generation"]["python"]["parseable_rate"],
        "python_meaningful_parseable": report["evaluation"]["generation"]["python"]["meaningful_parseable_rate"],
        "json_valid": report["evaluation"]["generation"]["json"]["valid_rate"],
        "general_loop_free": report["evaluation"]["generation"]["general"]["loop_free_rate"],
        "context_1792": report["evaluation"]["context_1792"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
