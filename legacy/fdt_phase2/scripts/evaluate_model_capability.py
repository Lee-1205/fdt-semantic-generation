from __future__ import annotations

import argparse
import ast
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from fdt_rlm.config import ModelConfig
from fdt_rlm.eval.action_metrics import action_metrics
from fdt_rlm.models import build_model
from fdt_rlm.runtime.parser import parse_action
from fdt_rlm.tokenization import load_tokenizer


CHECKPOINTS = {
    "transformer": ROOT / "runs" / "full_transformer" / "best.pt",
    "fdt": ROOT / "runs" / "full_fdt" / "best.pt",
}
PROMPTS = [
    "The main reason neural networks require optimization is",
    "A good scientific experiment should control",
    "In a small village surrounded by mountains,",
    "The difference between training and inference is",
    "When a computer program receives input,",
]
CODE_PROMPTS = [
    ("python", "def calculate_average(values):\n"),
    ("python", "def binary_search(items, target):\n"),
    ("python", "class DatasetLoader:\n    def __init__(self, path):\n"),
    ("javascript", "function parseConfig(config) {\n"),
    ("rust", "fn calculate_sum(values: &[i32]) -> i32 {\n"),
    ("json", '{"name":"experiment","status":'),
]


def safe_load(path: Path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(path: Path, device: torch.device):
    checkpoint = safe_load(path, device)
    config = ModelConfig(**checkpoint["model_config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def validation_data():
    shard = safe_load(ROOT / "prepared_data" / "fdt_rlm_10m" / "shards" / "validation" / "shard_000000.pt")
    ids = shard["input_ids"].long()
    mask = shard.get("attention_mask", torch.ones_like(ids)).long()
    manifest = json.loads((ROOT / "prepared_data" / "fdt_rlm_10m" / "manifest.json").read_text(encoding="utf-8"))
    labels: List[str] = []
    for entry in manifest["shards"]:
        if entry["split"] == "validation":
            for source, tokens in entry["buckets"].items():
                labels.extend([source] * (int(tokens) // int(manifest["seq_len"])))
    labels = (labels + ["unknown"] * len(ids))[: len(ids)]
    return ids, mask, labels


@torch.inference_mode()
def language_metrics(model, config, ids, masks, labels, device):
    totals = defaultdict(lambda: {"loss": 0.0, "tokens": 0, "top1": 0, "top5": 0, "top10": 0})
    anchor_rows = 0
    anchor_entropy = 0.0
    anchor_effective = 0.0
    anchor_top1 = 0.0
    anchor_margin = 0.0
    anchor_usage = None
    for row_ids, row_mask, source in zip(ids, masks, labels):
        batch_ids = row_ids.unsqueeze(0).to(device)
        batch_mask = row_mask.unsqueeze(0).to(device)
        output = model(input_ids=batch_ids, attention_mask=batch_mask)
        logits = output["logits"][:, :-1].float()
        for stats in output.get("anchor_stats", []):
            entropy = float(stats.entropy.detach().float().item())
            anchor_entropy += entropy
            anchor_effective += math.exp(entropy)
            anchor_top1 += float(stats.top1_membership.detach().float().item())
            anchor_margin += float(stats.membership_margin.detach().float().item())
            usage = stats.usage_prob.detach().float().cpu()
            anchor_usage = usage.clone() if anchor_usage is None else anchor_usage + usage
            anchor_rows += 1
        targets = batch_ids[:, 1:]
        valid = batch_mask[:, 1:].bool() & (targets != config.pad_token_id)
        losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        for key in ("overall", source):
            total = totals[key]
            total["loss"] += float(losses[valid].sum().item())
            total["tokens"] += int(valid.sum().item())
            top = logits.topk(10, dim=-1).indices
            truth = targets.unsqueeze(-1)
            total["top1"] += int((top[..., :1] == truth)[valid].any(dim=-1).sum().item())
            total["top5"] += int((top[..., :5] == truth)[valid].any(dim=-1).sum().item())
            total["top10"] += int((top == truth)[valid].any(dim=-1).sum().item())
    report = {}
    for source, total in totals.items():
        count = max(total["tokens"], 1)
        loss = total["loss"] / count
        report[source] = {
            "loss": loss,
            "perplexity": math.exp(loss) if loss < 80 else float("inf"),
            "top1_accuracy": total["top1"] / count,
            "top5_accuracy": total["top5"] / count,
            "top10_accuracy": total["top10"] / count,
            "tokens": total["tokens"],
        }
    anchor_health = None
    if anchor_rows and anchor_usage is not None:
        usage = anchor_usage / anchor_rows
        anchor_health = {
            "entropy": anchor_entropy / anchor_rows,
            "normalized_entropy": (anchor_entropy / anchor_rows) / math.log(max(config.top_k, 2)),
            "effective_k": anchor_effective / anchor_rows,
            "top1_membership": anchor_top1 / anchor_rows,
            "top1_top2_margin": anchor_margin / anchor_rows,
            "dead_anchor_ratio": float((usage < 1e-4).float().mean().item()),
            "top_anchor_frequency": float(usage.max().item()),
            "usage_histogram": usage.tolist(),
        }
    return report, anchor_health


@torch.inference_mode()
def generate(model, config, tokenizer, prompt, device, temperature, max_new_tokens, seed):
    ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    generated = []
    generator = torch.Generator(device=device.type).manual_seed(seed)
    for _ in range(max_new_tokens):
        context = ids[-config.max_seq_len :]
        input_ids = torch.tensor([context], device=device)
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))["logits"][0, -1].float()
        if temperature == 0:
            next_id = int(logits.argmax().item())
        else:
            logits = logits / temperature
            cutoff = torch.topk(logits, min(50, logits.numel())).values[-1]
            logits[logits < cutoff] = -float("inf")
            probs = logits.softmax(dim=-1)
            next_id = int(torch.multinomial(probs, 1, generator=generator).item())
        if next_id in {config.eos_token_id, config.pad_token_id}:
            break
        ids.append(next_id)
        generated.append(next_id)
    return tokenizer.decode(generated, skip_special_tokens=True), generated


def repetition_rate(tokens: List[int], width: int = 3) -> float:
    if len(tokens) < width:
        return 0.0
    grams = [tuple(tokens[i : i + width]) for i in range(len(tokens) - width + 1)]
    return 1.0 - len(set(grams)) / max(len(grams), 1)


def balanced(text: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for char in text:
        if char in "([{":
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack


def code_probe(model, config, tokenizer, device, model_name, max_new_tokens):
    rows = []
    for index, (language, prompt) in enumerate(CODE_PROMPTS):
        completion, tokens = generate(model, config, tokenizer, prompt, device, 0, max_new_tokens, 9000 + index)
        combined = prompt + completion
        parseable = False
        if language == "python":
            try:
                ast.parse(combined)
                parseable = True
            except SyntaxError:
                pass
        elif language == "json":
            try:
                json.loads(combined)
                parseable = True
            except json.JSONDecodeError:
                pass
        else:
            parseable = balanced(combined)
        rows.append({"model": model_name, "language": language, "prompt": prompt, "completion": completion, "parseable": parseable, "balanced_delimiters": balanced(combined), "repetition_rate": repetition_rate(tokens)})
    return rows


def action_cases(count: int):
    templates = [
        ("SEARCH", "An exact phrase must be located.", {"query": "gradient checkpointing"}),
        ("READ", "The relevant chunk index is already known.", {"start": 12, "end": 28}),
        ("COMPARE", "Two result references conflict.", {"result_refs": ["result_0", "result_1"]}),
        ("STOP", "Enough evidence has already been collected.", {"answer": "complete"}),
        ("CALL", "A selected chunk needs isolated sub-analysis.", {"question": "Summarize the memory claim", "context_ref": "result_0"}),
    ]
    for index in range(count):
        action, scenario, arguments = templates[index % len(templates)]
        yield action, arguments, (
            "Return exactly one JSON action with no explanation. Allowed actions: READ, SEARCH, CALL, COMPARE, MERGE, STOP.\n"
            f"Scenario: {scenario}\nAction:"
        )


def action_probe(model, config, tokenizer, device, model_name, count, max_new_tokens):
    rows = []
    for index, (expected, arguments, prompt) in enumerate(action_cases(count)):
        completion, _ = generate(model, config, tokenizer, prompt, device, 0, max_new_tokens, 12000 + index)
        parsed = parse_action(completion)
        valid = parsed.action is not None
        actual = parsed.action.name.value if valid else None
        actual_args = parsed.action.arguments if valid else {}
        rows.append({
            "model": model_name,
            "expected_action": expected,
            "completion": completion,
            "valid_json": valid or parsed.error is not None and "invalid JSON" not in parsed.error,
            "valid_action": valid,
            "action_correct": actual == expected,
            "arguments_correct": actual == expected and all(actual_args.get(key) == value for key, value in arguments.items()),
            "repairable": parsed.repaired,
            "parse_error": parsed.error,
        })
    return {"metrics": action_metrics(rows), "rows": rows}


def retrieval_cases(per_bucket: int):
    rng = random.Random(20260711)
    for chunks in (4, 8, 16):
        for index in range(per_bucket):
            code = f"R{rng.randrange(1000, 9999)}"
            target = rng.randrange(chunks)
            paragraphs = [f"Paragraph {i} contains routine note {rng.randrange(100, 999)}." for i in range(chunks)]
            paragraphs[target] = f"Paragraph {target} contains verification code {code}."
            prompt = "Context:\n" + "\n".join(paragraphs) + f"\nQuestion: What code appears in paragraph {target}?\nAnswer:"
            yield chunks, code, prompt


def retrieval_probe(model, config, tokenizer, device, model_name, per_bucket):
    rows = []
    for index, (chunks, answer, prompt) in enumerate(retrieval_cases(per_bucket)):
        completion, _ = generate(model, config, tokenizer, prompt, device, 0, 12, 16000 + index)
        normalized = re.sub(r"\s+", "", completion).lower()
        rows.append({"model": model_name, "chunks": chunks, "gold": answer, "completion": completion, "exact_match": normalized == answer.lower(), "contains_answer": answer.lower() in normalized})
    total = max(len(rows), 1)
    return {"exact_match": sum(row["exact_match"] for row in rows) / total, "contains_answer": sum(row["contains_answer"] for row in rows) / total, "rows": rows}


def readiness(result):
    action = result["action"]["metrics"]
    retrieval = result["retrieval"]
    parse_rate = sum(row["parseable"] for row in result["code"]) / max(len(result["code"]), 1)
    if action["valid_action_rate"] >= 0.8 and action["action_accuracy"] >= 0.6 and retrieval["contains_answer"] >= 0.5:
        return "READY-A"
    if math.isfinite(result["language"]["overall"]["loss"]) and (parse_rate > 0 or action["valid_action_rate"] > 0.1):
        return "READY-B"
    return "READY-C"


def write_reports(payload, report_prefix=""):
    lines = ["# Model Capability Report", "", "No additional training tokens were used.", "", "## Capability Ladder", ""]
    for name, result in payload["models"].items():
        code_rate = sum(row["parseable"] for row in result["code"]) / max(len(result["code"]), 1)
        lines.extend([
            f"### {name}", "",
            f"- Validation loss: {result['language']['overall']['loss']:.6f}",
            f"- Top-5 accuracy: {result['language']['overall']['top5_accuracy']:.4f}",
            f"- Code/JSON parseability: {code_rate:.4f}",
            f"- Action JSON validity: {result['action']['metrics']['valid_action_rate']:.4f}",
            f"- Action accuracy: {result['action']['metrics']['action_accuracy']:.4f}",
            f"- Short retrieval contains-answer: {result['retrieval']['contains_answer']:.4f}",
            f"- Readiness: {result['readiness']}", "",
        ])
    capability_name = f"{report_prefix}_CAPABILITY_REPORT.md" if report_prefix else "MODEL_CAPABILITY_REPORT.md"
    (ROOT / capability_name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    fdt = payload["models"]["fdt"]
    report = [
        "# RLM Readiness Report", "",
        f"Category: **{fdt['readiness']}**", "",
        "## Gates", "",
        f"- Structured action validity: {fdt['action']['metrics']['valid_action_rate']:.4f}",
        f"- Structured action accuracy: {fdt['action']['metrics']['action_accuracy']:.4f}",
        f"- Short retrieval: {fdt['retrieval']['contains_answer']:.4f}", "",
        "READY-B requires minimal action imitation; READY-C requires improving base/action behavior before deep recursion.",
    ]
    readiness_name = f"{report_prefix}_READINESS_REPORT.md" if report_prefix else "RLM_READINESS_REPORT.md"
    (ROOT / readiness_name).write_text("\n".join(report) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--generation-tokens", type=int, default=40)
    parser.add_argument("--action-examples", type=int, default=100)
    parser.add_argument("--retrieval-per-bucket", type=int, default=10)
    parser.add_argument("--transformer-checkpoint", default=str(CHECKPOINTS["transformer"]))
    parser.add_argument("--fdt-checkpoint", default=str(CHECKPOINTS["fdt"]))
    parser.add_argument("--artifact-prefix", default="model_capability")
    parser.add_argument("--report-prefix", default="")
    args = parser.parse_args()
    device = torch.device(args.device)
    tokenizer = load_tokenizer(str(ROOT / "tokenizers" / "fdt_bpe"))
    ids, masks, labels = validation_data()
    payload = {"fixed_validation_sequences": len(ids), "models": {}}
    checkpoints = {
        "transformer": Path(args.transformer_checkpoint),
        "fdt": Path(args.fdt_checkpoint),
    }
    for model_name, checkpoint in checkpoints.items():
        model, config = load_model(checkpoint, device)
        generations = []
        for temperature in (0.0, 0.7, 1.0):
            for index, prompt in enumerate(PROMPTS):
                completion, tokens = generate(model, config, tokenizer, prompt, device, temperature, args.generation_tokens, 30000 + index + int(temperature * 100))
                generations.append({"prompt": prompt, "temperature": temperature, "completion": completion, "repetition_rate": repetition_rate(tokens)})
        language, anchor_health = language_metrics(model, config, ids, masks, labels, device)
        result = {
            "checkpoint": str(checkpoint),
            "language": language,
            "anchor_health": anchor_health,
            "english_generation": generations,
            "code": code_probe(model, config, tokenizer, device, model_name, args.generation_tokens),
            "action": action_probe(model, config, tokenizer, device, model_name, args.action_examples, 48),
            "retrieval": retrieval_probe(model, config, tokenizer, device, model_name, args.retrieval_per_bucket),
        }
        result["readiness"] = readiness(result)
        payload["models"][model_name] = result
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    write_json(ROOT / "artifacts" / f"{args.artifact_prefix}_results.json", payload)
    readiness_file = "rlm_readiness.json" if args.artifact_prefix == "model_capability" else f"{args.artifact_prefix}_readiness.json"
    write_json(ROOT / "artifacts" / readiness_file, {name: result["readiness"] for name, result in payload["models"].items()})
    write_reports(payload, args.report_prefix)
    print(json.dumps({name: {"loss": result["language"]["overall"]["loss"], "action": result["action"]["metrics"], "retrieval": {key: value for key, value in result["retrieval"].items() if key != "rows"}, "readiness": result["readiness"]} for name, result in payload["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
