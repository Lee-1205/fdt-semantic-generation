from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_base_model_strict as v2


MODE_SETTINGS = {
    "smoke": {
        "rows_per_bucket": 2,
        "retrieval_cases": 12,
        "general_rows_per_bucket": 1,
        "generation": {"factual_qa": 4, "retrieval": 4, "python_code": 4, "json": 4},
    },
    "standard": {
        "rows_per_bucket": 24,
        "retrieval_cases": 120,
        "general_rows_per_bucket": 10,
        "generation": {"factual_qa": 50, "retrieval": 100, "python_code": 25, "json": 50},
    },
    "full": {
        "rows_per_bucket": 64,
        "retrieval_cases": 300,
        "general_rows_per_bucket": 20,
        "generation": {"factual_qa": 400, "retrieval": 200, "python_code": 100, "json": 100},
    },
}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    probability = successes / total
    denominator = 1.0 + z * z / total
    center = (probability + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / total + z * z / (4.0 * total * total)
    ) / denominator
    low = 0.0 if successes == 0 else max(0.0, center - radius)
    high = 1.0 if successes == total else min(1.0, center + radius)
    return [low, high]


def bootstrap_mean_interval(
    values: list[float], samples: int = 2000, seed: int = 20260809
) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(samples)
    )
    low = means[int(0.025 * (samples - 1))]
    high = means[int(0.975 * (samples - 1))]
    return [low, high]


def proportion_summary(successes: int, total: int) -> dict:
    return {
        "successes": int(successes),
        "count": int(total),
        "rate": successes / total if total else 0.0,
        "wilson_95": wilson_interval(successes, total),
    }


def capability_ranges(manifest: dict, split: str = "validation") -> list[tuple[str, int, int]]:
    start = 0
    ranges = []
    for category, count in manifest[split]["categories"].items():
        ranges.append((category, start, start + int(count)))
        start += int(count)
    return ranges


def load_capability_payload(dataset_dir: Path, split: str = "validation") -> dict[str, torch.Tensor]:
    paths = sorted((dataset_dir / "shards" / split).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No {split} shards found under {dataset_dir}")
    parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        for key in ("input_ids", "labels", "attention_mask"):
            parts[key].append(payload[key])
    return {key: torch.cat(tensors) for key, tensors in parts.items()}


def summarize_teacher_forced_rows(rows: list[dict]) -> dict:
    tokens = sum(row["tokens"] for row in rows)
    top1 = sum(row["top1"] for row in rows)
    top5 = sum(row["top5"] for row in rows)
    sequence_exact = sum(row["sequence_exact"] for row in rows)
    loss_sum = sum(row["loss_sum"] for row in rows)
    row_accuracies = [row["top1"] / max(row["tokens"], 1) for row in rows]
    loss = loss_sum / max(tokens, 1)
    return {
        "rows": len(rows),
        "tokens": tokens,
        "loss": loss,
        "perplexity": math.exp(min(loss, 80.0)),
        "top1": proportion_summary(top1, tokens),
        "top5": proportion_summary(top5, tokens),
        "sequence_exact": proportion_summary(sequence_exact, len(rows)),
        "mean_row_top1": sum(row_accuracies) / max(len(row_accuracies), 1),
        "mean_row_top1_bootstrap_95": bootstrap_mean_interval(row_accuracies),
    }


@torch.inference_mode()
def evaluate_capability_teacher_forced(
    model,
    payload: dict[str, torch.Tensor],
    ranges: list[tuple[str, int, int]],
    device: torch.device,
    batch_size: int,
) -> dict:
    category_for_row = {}
    for category, start, end in ranges:
        category_for_row.update({index: category for index in range(start, end)})

    rows = []
    total_rows = payload["input_ids"].size(0)
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        ids = payload["input_ids"][start:end].to(device=device, dtype=torch.long)
        labels = payload["labels"][start:end].to(device=device, dtype=torch.long)
        masks = payload["attention_mask"][start:end].to(device=device, dtype=torch.long)
        output = model(ids, attention_mask=masks)
        logits = output["logits"][:, :-1].float()
        targets = labels[:, 1:]
        valid = (targets != -100) & masks[:, 1:].bool()
        safe_targets = targets.masked_fill(~valid, 0)
        losses = F.cross_entropy(logits.transpose(1, 2), safe_targets, reduction="none")
        top = logits.topk(5, dim=-1).indices
        top1_matches = (top[..., 0] == safe_targets) & valid
        top5_matches = (top == safe_targets.unsqueeze(-1)).any(-1) & valid

        for offset in range(end - start):
            row_index = start + offset
            token_count = int(valid[offset].sum())
            top1 = int(top1_matches[offset].sum())
            rows.append(
                {
                    "row_id": row_index,
                    "category": category_for_row[row_index],
                    "tokens": token_count,
                    "loss_sum": float(losses[offset][valid[offset]].sum()),
                    "top1": top1,
                    "top5": int(top5_matches[offset].sum()),
                    "sequence_exact": bool(token_count and top1 == token_count),
                }
            )
        if end % 40 == 0 or end == total_rows:
            print(json.dumps({"event": "capability_teacher_forced", "rows": end}), flush=True)

    by_category = {}
    for category, _, _ in ranges:
        selected = [row for row in rows if row["category"] == category]
        by_category[category] = summarize_teacher_forced_rows(selected)
    return {"overall": summarize_teacher_forced_rows(rows), "by_category": by_category, "rows": rows}


def decode_capability_row(payload: dict[str, torch.Tensor], tokenizer, row_index: int) -> dict:
    ids = payload["input_ids"][row_index].tolist()
    labels = payload["labels"][row_index].tolist()
    mask = payload["attention_mask"][row_index].bool().tolist()
    prompt_ids = [token for token, label, active in zip(ids, labels, mask) if active and label == -100]
    answer_ids = [label for label, active in zip(labels, mask) if active and label != -100]
    answer_ids = [token for token in answer_ids if token != tokenizer.eos_token_id]
    return {
        "prompt": tokenizer.decode(prompt_ids, skip_special_tokens=True),
        "gold": tokenizer.decode(answer_ids, skip_special_tokens=True),
        "gold_tokens": len(answer_ids),
    }


def normalize_text_tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)


def normalize_text(text: str) -> str:
    return " ".join(normalize_text_tokens(text))


def first_json_value(text: str):
    starts = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not starts:
        return None
    start = min(starts)
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
        return value
    except json.JSONDecodeError:
        return None


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def python_ast_dump(source: str) -> str | None:
    try:
        return ast.dump(ast.parse(source), annotate_fields=True, include_attributes=False)
    except SyntaxError:
        return None


def python_function_signature(tree: ast.AST | None) -> dict | None:
    if tree is None:
        return None
    functions = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if len(functions) != 1:
        return None
    function = functions[0]
    arguments = function.args
    return {
        "name": function.name,
        "positional": [argument.arg for argument in (*arguments.posonlyargs, *arguments.args)],
        "keyword_only": [argument.arg for argument in arguments.kwonlyargs],
        "vararg": arguments.vararg.arg if arguments.vararg else None,
        "kwarg": arguments.kwarg.arg if arguments.kwarg else None,
        "async": isinstance(function, ast.AsyncFunctionDef),
    }


def score_python_structure(completion: str, gold: str | None = None) -> dict:
    source = strip_code_fence(completion)
    try:
        tree = ast.parse(source)
        parseable = True
    except SyntaxError:
        tree = None
        parseable = False

    try:
        compile(source, "<generated>", "exec")
        compilable = True
    except (SyntaxError, TypeError, ValueError):
        compilable = False

    functions = (
        [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        if tree is not None
        else []
    )
    has_function = bool(functions)
    exactly_one_function = len(functions) == 1
    direct_self_recursion = False
    if exactly_one_function:
        function = functions[0]
        if len(function.body) == 1 and isinstance(function.body[0], ast.Return):
            value = function.body[0].value
            direct_self_recursion = bool(
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == function.name
            )

    usable_structure = bool(
        compilable
        and source.strip()
        and exactly_one_function
        and functions[0].body
        and not direct_self_recursion
    )
    gold_source = strip_code_fence(gold) if gold is not None else None
    generated_ast = python_ast_dump(source)
    gold_ast = python_ast_dump(gold_source) if gold_source is not None else None
    ast_exact = bool(gold_ast is not None and generated_ast == gold_ast)
    generated_signature = python_function_signature(tree)
    gold_tree = ast.parse(gold_source) if gold_ast is not None and gold_source is not None else None
    gold_signature = python_function_signature(gold_tree)
    signature_match = bool(gold_signature is not None and generated_signature == gold_signature)
    return {
        # Kept for artifact compatibility. This is structural usability, not semantic exactness.
        "correct": usable_structure,
        "correct_definition": "compilable_single_function_structure_nonsemantic",
        "parseable": parseable,
        "compilable": compilable,
        "has_function": has_function,
        "usable_structure": usable_structure,
        "ast_exact": ast_exact,
        "function_signature_match": signature_match,
        "generated_signature": generated_signature,
        "gold_signature": gold_signature,
    }


def score_capability_generation(category: str, gold: str, completion: str) -> dict:
    if category == "factual_qa":
        expected_tokens = normalize_text_tokens(gold)
        generated_tokens = normalize_text_tokens(completion)
        return {
            "correct": bool(expected_tokens)
            and generated_tokens[: len(expected_tokens)] == expected_tokens,
            "correct_definition": "normalized_whole_word_answer_prefix",
            "normalized_expected_tokens": expected_tokens,
            "normalized_generated_prefix": generated_tokens[: len(expected_tokens)],
        }
    if category == "retrieval":
        expected = v2.normalize_code(gold)
        generated = v2.normalize_code(completion)
        return {
            "correct": bool(expected) and generated[: len(expected)] == expected,
            "correct_definition": "normalized_code_prefix",
            "normalized_expected": expected,
            "normalized_generated_prefix": generated[: len(expected)],
        }
    if category == "json":
        expected = first_json_value(gold)
        generated = first_json_value(completion)
        return {
            "correct": generated == expected and expected is not None,
            "correct_definition": "parsed_json_value_equality",
            "valid": generated is not None,
            "parsed": generated,
            "expected_parsed": expected,
        }
    if category == "python_code":
        return score_python_structure(completion, gold)
    raise ValueError(f"Unsupported category: {category}")


def generated_token_limit(category: str, gold_tokens: int) -> int:
    if category == "retrieval":
        return min(max(gold_tokens + 4, 12), 24)
    if category == "factual_qa":
        return min(max(gold_tokens + 8, 20), 96)
    if category == "json":
        return min(max(gold_tokens + 12, 48), 128)
    if category == "python_code":
        return min(max(gold_tokens + 16, 64), 192)
    raise ValueError(category)


@torch.inference_mode()
def evaluate_capability_generation(
    model,
    tokenizer,
    config,
    payload: dict[str, torch.Tensor],
    ranges: list[tuple[str, int, int]],
    counts: dict[str, int],
    device: torch.device,
) -> dict:
    reports = {}
    for category, start, end in ranges:
        selected = v2.evenly_spaced(start, end, counts.get(category, 0))
        rows = []
        for position, row_index in enumerate(selected, start=1):
            decoded = decode_capability_row(payload, tokenizer, row_index)
            completion, tokens = v2.greedy_completion(
                model,
                tokenizer,
                config,
                decoded["prompt"],
                device,
                generated_token_limit(category, decoded["gold_tokens"]),
            )
            score = score_capability_generation(category, decoded["gold"], completion)
            rows.append(
                {
                    "row_id": row_index,
                    "prompt": decoded["prompt"],
                    "gold": decoded["gold"],
                    "completion": completion,
                    "generated_tokens": len(tokens),
                    "repetition_rate": v2.repetition_rate(tokens),
                    **score,
                }
            )
            if position % 10 == 0 or position == len(selected):
                print(
                    json.dumps(
                        {"event": "capability_generation", "category": category, "rows": position}
                    ),
                    flush=True,
                )
        correct = sum(row["correct"] for row in rows)
        report = {
            "correct": proportion_summary(correct, len(rows)),
            "mean_repetition_rate": sum(row["repetition_rate"] for row in rows) / max(len(rows), 1),
            "mean_repetition_bootstrap_95": bootstrap_mean_interval(
                [row["repetition_rate"] for row in rows]
            ),
            "rows": rows,
        }
        if category == "json":
            report["valid"] = proportion_summary(sum(row["valid"] for row in rows), len(rows))
        if category == "python_code":
            report["parseable"] = proportion_summary(
                sum(row["parseable"] for row in rows), len(rows)
            )
            report["compilable"] = proportion_summary(
                sum(row["compilable"] for row in rows), len(rows)
            )
            report["has_function"] = proportion_summary(
                sum(row["has_function"] for row in rows), len(rows)
            )
            report["usable_structure"] = proportion_summary(
                sum(row["usable_structure"] for row in rows), len(rows)
            )
            report["ast_exact"] = proportion_summary(
                sum(row["ast_exact"] for row in rows), len(rows)
            )
            report["function_signature_match"] = proportion_summary(
                sum(row["function_signature_match"] for row in rows), len(rows)
            )
            report["correct_metric_definition"] = (
                "compilable single-function structural usability; not functional or semantic exactness"
            )
        reports[category] = report
    return reports


@torch.inference_mode()
def evaluate_heldout_generation(
    model,
    tokenizer,
    config,
    ids: torch.Tensor,
    masks: torch.Tensor,
    ranges: list[tuple[str, int, int]],
    rows_per_bucket: int,
    device: torch.device,
    prefix_tokens: int = 96,
    continuation_tokens: int = 64,
) -> dict:
    reports = {}
    all_rows = []
    for bucket, start, end in ranges:
        rows = []
        for row_index in v2.evenly_spaced(start, end, rows_per_bucket):
            active = ids[row_index][masks[row_index].bool()].tolist()
            if len(active) < prefix_tokens + continuation_tokens:
                continue
            prompt_ids = active[:prefix_tokens]
            reference = active[prefix_tokens : prefix_tokens + continuation_tokens]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            completion, generated = v2.greedy_completion(
                model, tokenizer, config, prompt, device, continuation_tokens
            )
            prefix_match = 0
            for generated_id, reference_id in zip(generated, reference):
                if generated_id != reference_id:
                    break
                prefix_match += 1
            rows.append(
                {
                    "row_id": row_index,
                    "prompt": prompt,
                    "reference": tokenizer.decode(reference, skip_special_tokens=True),
                    "completion": completion,
                    "generated_tokens": len(generated),
                    "repetition_rate": v2.repetition_rate(generated),
                    "unique_token_rate": len(set(generated)) / max(len(generated), 1),
                    "reference_prefix_match_tokens": prefix_match,
                    "loop_free": v2.repetition_rate(generated) <= 0.20,
                    "non_whitespace": bool(completion.strip()),
                }
            )
        reports[bucket] = summarize_heldout_generation(rows)
        all_rows.extend({"bucket": bucket, **row} for row in rows)
        print(json.dumps({"event": "heldout_generation", "bucket": bucket, "rows": len(rows)}), flush=True)
    reports["overall"] = summarize_heldout_generation(all_rows)
    reports["rows"] = all_rows
    return reports


def summarize_heldout_generation(rows: list[dict]) -> dict:
    count = len(rows)
    repetitions = [row["repetition_rate"] for row in rows]
    return {
        "count": count,
        "non_whitespace": proportion_summary(sum(row["non_whitespace"] for row in rows), count),
        "loop_free": proportion_summary(sum(row["loop_free"] for row in rows), count),
        "mean_repetition_rate": sum(repetitions) / max(count, 1),
        "mean_repetition_bootstrap_95": bootstrap_mean_interval(repetitions),
        "mean_unique_token_rate": sum(row["unique_token_rate"] for row in rows) / max(count, 1),
        "mean_reference_prefix_match_tokens": sum(
            row["reference_prefix_match_tokens"] for row in rows
        ) / max(count, 1),
    }


def add_retrieval_intervals(report: dict) -> dict:
    report["rank"] = proportion_summary(
        sum(row["rank_correct"] for row in report["rows"]), len(report["rows"])
    )
    report["greedy_exact"] = proportion_summary(
        sum(row["greedy_exact"] for row in report["rows"]), len(report["rows"])
    )
    for distance, summary in report["by_distance"].items():
        rows = [row for row in report["rows"] if str(row["distance"]) == distance]
        summary["rank"] = proportion_summary(sum(row["rank_correct"] for row in rows), len(rows))
        summary["greedy_exact"] = proportion_summary(
            sum(row["greedy_exact"] for row in rows), len(rows)
        )
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_split_fingerprint(dataset_dir: Path, split: str) -> dict:
    manifest_path = dataset_dir / "manifest.json"
    shards = []
    for path in sorted((dataset_dir / "shards" / split).glob("*.pt")):
        shards.append(
            {
                "relative_path": path.relative_to(dataset_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not shards:
        raise FileNotFoundError(f"No {split} shards found under {dataset_dir}")
    aggregate = hashlib.sha256(
        json.dumps(shards, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "split": split,
        "shards": shards,
        "aggregate_sha256": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expanded, statistically reported strict base-model evaluation."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=ROOT / "prepared_data" / "v3_broad_additional_300m_combined_v2_24k_ctx2048",
    )
    parser.add_argument(
        "--capability_dataset_dir",
        type=Path,
        default=ROOT / "prepared_data" / "capability_curriculum_5k",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(MODE_SETTINGS), default="standard")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    started = time.time()
    settings = MODE_SETTINGS[args.mode]
    device = torch.device(args.device)
    model, tokenizer, config, metadata = v2.load_checkpoint_model(args.checkpoint, device)

    broad_manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    broad_ids, broad_masks = v2.split_rows(args.dataset_dir, "validation")
    broad_ranges = v2.bucket_ranges(broad_manifest, "validation")
    if broad_ranges[-1][2] != broad_ids.size(0):
        raise ValueError("Broad validation manifest and shards disagree")

    capability_manifest = json.loads(
        (args.capability_dataset_dir / "manifest.json").read_text(encoding="utf-8")
    )
    capability_payload = load_capability_payload(args.capability_dataset_dir)
    cap_ranges = capability_ranges(capability_manifest)
    if cap_ranges[-1][2] != capability_payload["input_ids"].size(0):
        raise ValueError("Capability validation manifest and shards disagree")

    print(json.dumps({"event": "start", "mode": args.mode, "metadata": metadata}), flush=True)
    heldout = v2.heldout_metrics(
        model,
        config,
        broad_ids,
        broad_masks,
        broad_ranges,
        settings["rows_per_bucket"],
        device,
    )
    print(json.dumps({"event": "heldout_complete"}), flush=True)
    teacher_forced = evaluate_capability_teacher_forced(
        model, capability_payload, cap_ranges, device, args.batch_size
    )
    generation = evaluate_capability_generation(
        model,
        tokenizer,
        config,
        capability_payload,
        cap_ranges,
        settings["generation"],
        device,
    )
    retrieval = add_retrieval_intervals(
        v2.evaluate_retrieval(model, tokenizer, config, device, settings["retrieval_cases"])
    )
    heldout_generation = evaluate_heldout_generation(
        model,
        tokenizer,
        config,
        broad_ids,
        broad_masks,
        broad_ranges,
        settings["general_rows_per_bucket"],
        device,
    )
    legacy = {
        "facts": v2.evaluate_facts(model, tokenizer, config, device),
        "code_cloze": v2.evaluate_code_cloze(model, tokenizer, config, device),
        "generation": v2.evaluate_generations(model, tokenizer, config, device),
        "context_1792": v2.benchmark_context(model, tokenizer, config, device),
    }

    report = {
        "status": "EVALUATION_COMPLETED",
        "capability_verdict": "NOT_ASSIGNED_BY_EVALUATOR",
        "evaluator": {
            "version": "strict_base_v3_audited_2",
            "mode": args.mode,
            "script_sha256": sha256_file(Path(__file__)),
            "settings": settings,
            "elapsed_seconds": time.time() - started,
            "scoring_definitions": {
                "factual_qa": "normalized whole-word answer prefix",
                "retrieval": "normalized code prefix",
                "json": "parsed JSON value equality",
                "python_correct": (
                    "compilable single-function structural usability; not semantic exactness"
                ),
                "python_ast_exact": "Python AST equality to the reference",
                "repetition": "duplicate token trigram fraction; loop_free threshold <= 0.20",
            },
        },
        "metadata": {**metadata, "checkpoint_sha256": sha256_file(args.checkpoint)},
        "datasets": {
            "broad_validation": str(args.dataset_dir.resolve()),
            "broad_validation_fingerprint": dataset_split_fingerprint(
                args.dataset_dir, "validation"
            ),
            "capability_validation": str(args.capability_dataset_dir.resolve()),
            "capability_validation_fingerprint": dataset_split_fingerprint(
                args.capability_dataset_dir, "validation"
            ),
            "capability_validation_rows": int(capability_payload["input_ids"].size(0)),
            "evaluation_performs_training": False,
            "capability_split_role": "reused_lineage_development_gate",
            "unbiased_sealed_final_test": False,
            "interpretation_warning": (
                "This fixed split has informed repeated lineage decisions and must not be treated as "
                "an unbiased one-shot final test. Confirm promotions on an independent sealed holdout."
            ),
        },
        "evaluation": {
            "heldout": heldout,
            "capability_teacher_forced": teacher_forced,
            "capability_generation": generation,
            "retrieval_base_continuation": retrieval,
            "heldout_prefix_generation": heldout_generation,
            "legacy_v2_probes": legacy,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.output)

    summary = {
        "status": "EVALUATION_COMPLETED",
        "capability_verdict": "NOT_ASSIGNED_BY_EVALUATOR",
        "output": str(args.output.resolve()),
        "tokens_seen": metadata["tokens_seen"],
        "heldout": heldout["overall"],
        "capability_teacher_forced": teacher_forced["by_category"],
        "capability_generation": {
            category: {key: value for key, value in metrics.items() if key != "rows"}
            for category, metrics in generation.items()
        },
        "retrieval": {key: retrieval[key] for key in ("rank", "greedy_exact", "by_distance")},
        "heldout_prefix_generation": heldout_generation["overall"],
        "elapsed_seconds": report["evaluator"]["elapsed_seconds"],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
