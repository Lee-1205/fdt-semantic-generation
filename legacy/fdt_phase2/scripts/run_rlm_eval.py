from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.eval.recursion_metrics import recursion_metrics
from fdt_rlm.models import build_model
from fdt_rlm.runtime.anchor_state import AnchorState
from fdt_rlm.runtime.budget import BudgetConfig
from fdt_rlm.runtime.code_environment import CodeEnvironment
from fdt_rlm.runtime.model_backend import LocalCausalBackend
from fdt_rlm.runtime.recursive_controller import RecursiveController
from fdt_rlm.runtime.text_environment import TextEnvironment
from fdt_rlm.runtime.uncertainty_policy import UncertaintyPolicy
from fdt_rlm.tokenization import load_tokenizer


CHECKPOINTS = {
    "transformer_direct": ROOT / "runs" / "full_transformer" / "best.pt",
    "fdt_direct": ROOT / "runs" / "full_fdt" / "best.pt",
    "transformer_rlm": ROOT / "runs" / "rlm_action_transformer" / "best.pt",
    "fdt_rlm": ROOT / "runs" / "rlm_action_fdt" / "best.pt",
}


def safe_load(path: Path, device):
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


def load_tasks(path: Path, per_family: int) -> list[dict[str, Any]]:
    counts: dict[str, int] = defaultdict(int)
    selected = []
    for line in path.read_text(encoding="utf-8").splitlines():
        task = json.loads(line)
        if counts[task["family"]] < per_family:
            selected.append(task)
            counts[task["family"]] += 1
    return selected


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def scored(answer: str, gold: str) -> tuple[bool, bool]:
    predicted, expected = normalize(answer), normalize(gold)
    return predicted == expected, bool(expected and expected in predicted)


@torch.inference_mode()
def direct_answer(model, config, tokenizer, task, device, max_new_tokens: int) -> tuple[str, int]:
    material = json.dumps(task["files"], ensure_ascii=False) if task.get("files") else task["context"]
    prompt = f"Context:\n{material}\nQuestion: {task['question']}\nAnswer:"
    prompt_budget = max(config.max_seq_len - max_new_tokens, 1)
    ids = list(tokenizer.encode(prompt, add_special_tokens=False))[-prompt_budget:]
    x = torch.tensor([ids], device=device)
    use_cache = hasattr(model, "prefill") and hasattr(model, "decode_step")
    if use_cache:
        output, cache = model.prefill(x, torch.ones_like(x))
    generated = []
    for _ in range(max_new_tokens):
        if use_cache:
            logits = output["logits"][0, -1]
        else:
            context = ids[-config.max_seq_len :]
            x = torch.tensor([context], device=device)
            logits = model(input_ids=x, attention_mask=torch.ones_like(x))["logits"][0, -1]
        token = int(logits.argmax().item())
        if token in {config.eos_token_id, config.pad_token_id}:
            break
        ids.append(token)
        generated.append(token)
        if use_cache:
            if cache["length"] >= config.max_seq_len:
                break
            next_id = torch.tensor([[token]], device=device)
            output, cache = model.decode_step(next_id, cache)
    return tokenizer.decode(generated, skip_special_tokens=True), len(generated)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(rows), 1)
    by_family = {}
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        by_family[family] = sum(row["contains_answer"] for row in family_rows) / max(len(family_rows), 1)
    rec_rows = [
        {
            "call_made": row["call_made"],
            "call_helpful": row["call_made"] and row["contains_answer"],
        }
        for row in rows
        if row["call_required"] or row["call_made"]
    ]
    return {
        "tasks": len(rows),
        "exact_match": sum(row["exact_match"] for row in rows) / total,
        "contains_answer": sum(row["contains_answer"] for row in rows) / total,
        "family_accuracy": by_family,
        "average_steps": sum(row["steps"] for row in rows) / total,
        "average_subcalls": sum(row["subcalls"] for row in rows) / total,
        "average_tokens_read": sum(row["tokens_read"] for row in rows) / total,
        "average_tokens_generated": sum(row["tokens_generated"] for row in rows) / total,
        "average_latency_seconds": sum(row["latency_seconds"] for row in rows) / total,
        "invalid_action_rate": sum(row["invalid_actions"] for row in rows) / max(sum(row["steps"] for row in rows), 1),
        "loop_rate": sum(row["status"] == "loop_detected" for row in rows) / total,
        "budget_exhaustion_rate": sum(row["status"].startswith("budget_exceeded") for row in rows) / total,
        "recursion": recursion_metrics(rec_rows),
    }


def evaluate_direct(name, model, config, tokenizer, tasks, device, max_new_tokens):
    rows = []
    for task in tasks:
        started = time.perf_counter()
        answer, tokens = direct_answer(model, config, tokenizer, task, device, max_new_tokens)
        exact, contains = scored(answer, task["gold_answer"])
        rows.append(
            {
                "mode": name,
                "task_id": task["task_id"],
                "family": task["family"],
                "answer": answer,
                "gold": task["gold_answer"],
                "exact_match": exact,
                "contains_answer": contains,
                "steps": 1,
                "subcalls": 0,
                "tokens_read": 0,
                "tokens_generated": tokens,
                "invalid_actions": 0,
                "call_made": False,
                "call_required": any(action["action"] == "CALL" for action in task["gold_trajectory"]),
                "status": "direct",
                "latency_seconds": time.perf_counter() - started,
                "trace": {"events": []},
            }
        )
    return rows


def evaluate_rlm(name, model, config, tokenizer, tasks, device, budget_config, max_action_tokens, guided):
    rows = []
    uncertainty_rows = []
    for task in tasks:
        environment = CodeEnvironment(task["files"]) if task.get("files") else TextEnvironment(task["context"])
        state = AnchorState(config.num_anchors, config.dim, device=device) if guided else None
        backend = LocalCausalBackend(model, config, tokenizer, device, max_new_tokens=max_action_tokens)
        controller = RecursiveController(
            backend,
            budget_config=budget_config,
            uncertainty_policy=UncertaintyPolicy() if guided else None,
            anchor_state=state,
        )
        started = time.perf_counter()
        result = controller.run(task["question"], environment)
        latency = time.perf_counter() - started
        exact, contains = scored(result.answer, task["gold_answer"])
        actions = [event.action["action"] for event in result.trace.events if event.action]
        invalid = sum(event.action is None for event in result.trace.events)
        remaining = result.budget_remaining
        row = {
            "mode": name,
            "task_id": task["task_id"],
            "family": task["family"],
            "answer": result.answer,
            "gold": task["gold_answer"],
            "exact_match": exact,
            "contains_answer": contains,
            "steps": result.steps,
            "subcalls": budget_config.max_subcalls - remaining["subcalls"],
            "tokens_read": budget_config.max_read_tokens - remaining["read_tokens"],
            "tokens_generated": budget_config.max_generated_tokens - remaining["generated_tokens"],
            "invalid_actions": invalid,
            "call_made": "CALL" in actions,
            "call_required": any(action["action"] == "CALL" for action in task["gold_trajectory"]),
            "status": result.status,
            "latency_seconds": latency,
            "trace": result.trace.to_dict(),
        }
        rows.append(row)
        if guided:
            for index, event in enumerate(result.trace.events):
                uncertainty_rows.append(
                    {
                        "task_id": task["task_id"],
                        "step": event.step,
                        "correct_without_more_context": bool(event.action and event.action["action"] == "STOP" and contains),
                        "additional_read_helpful": bool(index < len(task["gold_trajectory"]) - 1 and contains),
                        "recursive_call_helpful": bool(row["call_required"] and row["call_made"] and contains),
                        "signals": event.signals,
                    }
                )
    return rows, uncertainty_rows


def main():
    parser = argparse.ArgumentParser(description="Matched five-mode FDT-RLM screening evaluation")
    parser.add_argument("--tasks", type=Path, default=ROOT / "artifacts" / "rlm_tasks_500.jsonl")
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizers" / "fdt_bpe"))
    parser.add_argument("--per-family", type=int, default=5)
    parser.add_argument("--direct-tokens", type=int, default=24)
    parser.add_argument("--action-tokens", type=int, default=48)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--current-base-checkpoint", type=Path)
    parser.add_argument("--current-rlm-checkpoint", type=Path)
    parser.add_argument("--output-stem", default="rlm_evaluation_results")
    args = parser.parse_args()
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    tasks = load_tasks(args.tasks, args.per_family)
    budget_config = BudgetConfig(max_steps=8, max_depth=1, max_subcalls=3, max_read_tokens=4096, max_generated_tokens=1024)

    if args.current_base_checkpoint is not None:
        base_path = args.current_base_checkpoint.resolve()
        model, config = load_model(base_path, device)
        direct_rows = evaluate_direct(
            "current_fdt_direct", model, config, tokenizer, tasks, device, args.direct_tokens
        )
        all_rows = list(direct_rows)
        modes = {"current_fdt_direct": aggregate(direct_rows)}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.current_rlm_checkpoint is not None:
            model, config = load_model(args.current_rlm_checkpoint.resolve(), device)
            naive_rows, _ = evaluate_rlm(
                "current_fdt_naive_rlm",
                model,
                config,
                tokenizer,
                tasks,
                device,
                budget_config,
                args.action_tokens,
                False,
            )
            guided_rows, uncertainty_rows = evaluate_rlm(
                "current_fdt_uncertainty_rlm",
                model,
                config,
                tokenizer,
                tasks,
                device,
                budget_config,
                args.action_tokens,
                True,
            )
            all_rows.extend(naive_rows)
            all_rows.extend(guided_rows)
            modes["current_fdt_naive_rlm"] = aggregate(naive_rows)
            modes["current_fdt_uncertainty_rlm"] = aggregate(guided_rows)
        else:
            uncertainty_rows = []

        payload = {
            "scope": f"current FDT screening: {args.per_family} per family",
            "tasks_evaluated_per_mode": len(tasks),
            "same_budget": dict(budget_config.__dict__),
            "base_checkpoint": str(base_path),
            "rlm_checkpoint": str(args.current_rlm_checkpoint.resolve()) if args.current_rlm_checkpoint else None,
            "modes": modes,
            "rows": all_rows,
        }
        artifacts = ROOT / "artifacts"
        result_path = artifacts / f"{args.output_stem}.json"
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with (artifacts / f"{args.output_stem}_uncertainty.jsonl").open("w", encoding="utf-8") as handle:
            for row in uncertainty_rows:
                handle.write(json.dumps(row) + "\n")
        print(json.dumps({"scope": payload["scope"], "modes": modes, "output": str(result_path)}, indent=2))
        return

    for key in ("transformer_rlm", "fdt_rlm"):
        if not CHECKPOINTS[key].exists():
            raise FileNotFoundError(f"Run minimal action training first: {CHECKPOINTS[key]}")
    all_rows = []
    modes = {}

    for mode, checkpoint_key in (("A_transformer_direct", "transformer_direct"), ("B_fdt_direct", "fdt_direct")):
        model, config = load_model(CHECKPOINTS[checkpoint_key], device)
        rows = evaluate_direct(mode, model, config, tokenizer, tasks, device, args.direct_tokens)
        all_rows.extend(rows)
        modes[mode] = aggregate(rows)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model, config = load_model(CHECKPOINTS["transformer_rlm"], device)
    rows, _ = evaluate_rlm("C_transformer_rlm", model, config, tokenizer, tasks, device, budget_config, args.action_tokens, False)
    all_rows.extend(rows)
    modes["C_transformer_rlm"] = aggregate(rows)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, config = load_model(CHECKPOINTS["fdt_rlm"], device)
    rows, _ = evaluate_rlm("D_fdt_naive_rlm", model, config, tokenizer, tasks, device, budget_config, args.action_tokens, False)
    all_rows.extend(rows)
    modes["D_fdt_naive_rlm"] = aggregate(rows)
    guided_rows, uncertainty_rows = evaluate_rlm("E_fdt_uncertainty_rlm", model, config, tokenizer, tasks, device, budget_config, args.action_tokens, True)
    all_rows.extend(guided_rows)
    modes["E_fdt_uncertainty_rlm"] = aggregate(guided_rows)

    payload = {
        "scope": f"screening subset: {args.per_family} per family; expand only if useful",
        "suite_tasks_available": 500,
        "tasks_evaluated_per_mode": len(tasks),
        "same_budget": dict(budget_config.__dict__),
        "modes": modes,
        "rows": all_rows,
    }
    artifacts = ROOT / "artifacts"
    (artifacts / "rlm_evaluation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (artifacts / "rlm_comparison.json").write_text(json.dumps({"modes": modes, "same_budget": payload["same_budget"]}, indent=2), encoding="utf-8")
    with (artifacts / "rlm_uncertainty_dataset.jsonl").open("w", encoding="utf-8") as handle:
        for row in uncertainty_rows:
            handle.write(json.dumps(row) + "\n")
    print(json.dumps({"scope": payload["scope"], "modes": modes}, indent=2))


if __name__ == "__main__":
    main()
