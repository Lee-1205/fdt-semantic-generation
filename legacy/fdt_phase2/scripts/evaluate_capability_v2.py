from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.next_tools import NEXT_MODEL_TYPES, choose_onk_cache, generate_ids, load_model_and_tokenizer


CATEGORIES = (
    "pure_continuation",
    "code_prefix_completion",
    "json_prefix_completion",
    "instruction_action_generation",
    "retrieval_continuation",
    "retrieval_instruction",
)

CASES = {
    "pure_continuation": [{"prompt": "The capital of France is", "gold": "Paris"}],
    "code_prefix_completion": [{"prompt": "def add(a, b):\n    return ", "language": "python"}],
    "json_prefix_completion": [{"prompt": '{"name":"Ada","languages":[', "suffix": "]}"}],
    "instruction_action_generation": [{"prompt": 'Return only JSON: {"action":"STOP","answer":"done"}\nJSON:', "gold": "STOP"}],
    "retrieval_continuation": [{"prompt": "Record: verification code is R7319. Therefore the verification code is", "gold": "R7319"}],
    "retrieval_instruction": [{"prompt": "Context: The verification code is R7319.\nQuestion: What is the code?\nAnswer:", "gold": "R7319"}],
}


def extract_json(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def score_case(category: str, case: dict, completion: str) -> dict:
    normalized = re.sub(r"\s+", "", completion).lower()
    result = {"nonempty": bool(completion.strip())}
    if category == "code_prefix_completion":
        try:
            ast.parse(case["prompt"] + completion)
            result["parseable"] = True
        except SyntaxError:
            result["parseable"] = False
    elif category == "json_prefix_completion":
        try:
            json.loads(case["prompt"] + completion + case.get("suffix", ""))
            result["parseable"] = True
        except json.JSONDecodeError:
            result["parseable"] = False
    elif category == "instruction_action_generation":
        parsed = extract_json(completion)
        result["valid_json"] = parsed is not None
        result["correct"] = isinstance(parsed, dict) and str(parsed.get("action", "")).upper() == case["gold"]
    elif "gold" in case:
        result["contains_gold"] = case["gold"].lower() in normalized
    return result


def summarize(rows: list[dict]) -> dict:
    metric_keys = sorted({key for row in rows for key, value in row["score"].items() if isinstance(value, bool)})
    metrics = {key + "_rate": sum(bool(row["score"].get(key)) for row in rows) / max(len(rows), 1) for key in metric_keys}
    return {"count": len(rows), "metrics": metrics, "rows": rows}


@torch.inference_mode()
def evaluate(model, tokenizer, config, max_new_tokens: int, device: torch.device, dry_run: bool):
    report = {}
    for category in CATEGORIES:
        rows = []
        cases = CASES[category][:1] if dry_run else CASES[category]
        for case in cases:
            ids = tokenizer.encode(case["prompt"], add_special_tokens=False)[-config.max_seq_len :]
            input_ids = torch.tensor([ids], device=device)
            budget = min(max_new_tokens, 8 if dry_run else max_new_tokens, config.max_seq_len - len(ids))
            generated = generate_ids(
                model,
                input_ids,
                max(budget, 0),
                greedy=True,
                repetition_penalty=1.15,
                use_cache=choose_onk_cache(config, len(ids), "fp32"),
            )
            completion = tokenizer.decode(generated[0, len(ids) :].tolist(), skip_special_tokens=True)
            rows.append({"prompt": case["prompt"], "completion": completion, "score": score_case(category, case, completion)})
        report[category] = summarize(rows)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Capability evaluator separating continuation and instruction behavior.")
    parser.add_argument("--model_type", choices=NEXT_MODEL_TYPES, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "capability_v2.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    model, tokenizer, config, _ = load_model_and_tokenizer(args.checkpoint, device, args.model_type, args.tokenizer_name)
    report = evaluate(model, tokenizer, config, args.max_new_tokens, device, args.dry_run)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output), "categories": list(report)}, indent=2))


if __name__ == "__main__":
    main()
