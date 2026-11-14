from __future__ import annotations

import argparse
import ast
import json
import random
import re
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.lexical_pointer import LexicalPointerDecodeState, SparseLexicalPointer
from fdt_rlm.next_tools import load_model_and_tokenizer, safe_torch_load


GENERAL_PROMPTS = (
    "The purpose of scientific testing is",
    "A helpful answer to a practical question should",
    "In a small town near the coast, the residents",
    "The capital of France is",
)


def repetition_rate(token_ids: list[int], width: int = 3) -> float:
    if len(token_ids) < width:
        return 0.0
    grams = [tuple(token_ids[index : index + width]) for index in range(len(token_ids) - width + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def is_boundary(piece: str) -> bool:
    return bool(re.search(r"[.!?;:]\s*$", piece) or "\n" in piece)


@torch.inference_mode()
def generate(
    model,
    pointer,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    device,
    min_gate: float,
    activation_steps: int,
    use_pointer: bool,
    stop_at_period: bool = False,
):
    source_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = torch.tensor([source_ids], device=device)
    state = LexicalPointerDecodeState(
        source_length=len(source_ids), max_activation_steps=activation_steps
    )
    new_ids: list[int] = []
    trace = []
    for _ in range(max_new_tokens):
        mask = torch.ones_like(generated)
        output = model(generated, attention_mask=mask)
        base_logits = output["logits"][:, -1].float()
        if use_pointer:
            logits, diagnostics = state.prepare_logits(
                pointer,
                base_logits,
                output["hidden"],
                generated,
                mask,
                min_gate=min_gate,
            )
        else:
            logits = base_logits
            diagnostics = {"mode": "base", "gate": 0.0, "mix_gate": 0.0}
        selected_id = int(logits.argmax(dim=-1))
        piece = tokenizer.decode([selected_id], skip_special_tokens=True)
        if use_pointer:
            state.commit(selected_id, diagnostics, boundary=is_boundary(piece))
        if len(trace) < 16:
            trace.append(
                {
                    "mode": diagnostics.get("mode", "base"),
                    "gate": float(diagnostics.get("gate", 0.0)),
                    "mix_gate": float(diagnostics.get("mix_gate", 0.0)),
                    "source_positions": diagnostics.get("source_positions", []),
                    "candidate_ids": diagnostics.get("candidate_ids", []),
                    "selected_id": selected_id,
                }
            )
        new_ids.append(selected_id)
        generated = torch.cat(
            (generated, torch.tensor([[selected_id]], device=device)), dim=1
        )
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        if selected_id == tokenizer.eos_token_id or (
            stop_at_period and text.strip().endswith(".")
        ):
            break
    return {
        "completion": tokenizer.decode(new_ids, skip_special_tokens=True),
        "token_ids": new_ids,
        "repetition_3gram": repetition_rate(new_ids),
        "pointer_steps": sum(row["mode"] in {"mixed", "cursor"} for row in trace),
        "cursor_steps": sum(row["mode"] == "cursor" for row in trace),
        "trace": trace,
    }


@torch.inference_mode()
def benchmark_long_context(model, pointer, tokenizer, config, device, length: int, min_gate: float):
    seed_ids = tokenizer.encode(
        "Long-context inference measures stable sparse routing and lexical retrieval. ",
        add_special_tokens=False,
    )
    ids = (seed_ids * (length // len(seed_ids) + 1))[:length]
    input_ids = torch.tensor([ids], device=device)
    mask = torch.ones_like(input_ids)
    model(input_ids[:, : min(length, 256)], attention_mask=mask[:, : min(length, 256)])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = model(input_ids, attention_mask=mask)
    torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    state = LexicalPointerDecodeState(source_length=length)
    started = time.perf_counter()
    _, diagnostics = state.prepare_logits(
        pointer,
        output["logits"][:, -1].float(),
        output["hidden"],
        input_ids,
        mask,
        min_gate=min_gate,
    )
    torch.cuda.synchronize()
    pointer_seconds = time.perf_counter() - started
    return {
        "context_tokens": length,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": length / max(prefill_seconds, 1e-9),
        "pointer_step_ms": pointer_seconds * 1000.0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "pointer_diagnostics": diagnostics,
        "onk_complexity": "O(N(K+W)) with O(1) active cursor continuation",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate an O(NW) lexical pointer adapter")
    parser.add_argument("--pointer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-codes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=202608021041)
    parser.add_argument("--min-gate", type=float, default=0.8)
    parser.add_argument("--activation-steps", type=int, default=4)
    parser.add_argument("--long-context", type=int, default=1792)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = safe_torch_load(args.pointer_checkpoint.resolve())
    model, tokenizer, config, _ = load_model_and_tokenizer(
        Path(checkpoint["base_checkpoint"]), device, "fdt_v3"
    )
    model = model.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    pointer = SparseLexicalPointer(**checkpoint["pointer_config"]).to(device)
    pointer.load_state_dict(checkpoint["pointer_state_dict"], strict=True)
    pointer.eval()

    rng = random.Random(args.seed)
    retrieval_rows = []
    for _ in range(args.random_codes):
        code = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(2))
        code += "".join(rng.choice("0123456789") for _ in range(6))
        prompts = {
            "continuation": f"Record: verification code is {code}. Therefore the verification code is",
            "instruction": f"Context: The verification code is {code}.\nQuestion: What is the code?\nAnswer:",
        }
        for style, prompt in prompts.items():
            result = generate(
                model,
                pointer,
                tokenizer,
                prompt,
                12,
                device,
                args.min_gate,
                args.activation_steps,
                True,
                stop_at_period=True,
            )
            normalized = re.sub(r"[^A-Z0-9]", "", result["completion"].upper())
            retrieval_rows.append(
                {
                    "code": code,
                    "style": style,
                    "correct": normalized.startswith(code),
                    **result,
                }
            )

    general_rows = []
    for prompt in GENERAL_PROMPTS:
        base = generate(model, pointer, tokenizer, prompt, 48, device, args.min_gate, args.activation_steps, False)
        adapted = generate(model, pointer, tokenizer, prompt, 48, device, args.min_gate, args.activation_steps, True)
        general_rows.append(
            {
                "prompt": prompt,
                "identical_to_base": adapted["token_ids"] == base["token_ids"],
                "base": base,
                "pointer": adapted,
            }
        )

    capability_rows = []
    cases = (
        ("python", "def add(a, b):\n    return ", ""),
        ("json", '{"name":"Ada","languages":[', "]}"),
    )
    for name, prompt, suffix in cases:
        result = generate(model, pointer, tokenizer, prompt, 40, device, args.min_gate, args.activation_steps, True)
        try:
            ast.parse(prompt + result["completion"]) if name == "python" else json.loads(prompt + result["completion"] + suffix)
            parseable = True
        except (SyntaxError, json.JSONDecodeError):
            parseable = False
        capability_rows.append({"name": name, "parseable": parseable, **result})

    by_style = {
        style: sum(row["correct"] for row in retrieval_rows if row["style"] == style)
        / max(sum(row["style"] == style for row in retrieval_rows), 1)
        for style in ("continuation", "instruction")
    }
    retrieval_accuracy = sum(row["correct"] for row in retrieval_rows) / max(len(retrieval_rows), 1)
    general_identical_rate = sum(row["identical_to_base"] for row in general_rows) / len(general_rows)
    report = {
        "status": "PASS",
        "pointer_checkpoint": str(args.pointer_checkpoint.resolve()),
        "retrieval": {
            "unique_codes": args.random_codes,
            "generations": len(retrieval_rows),
            "accuracy": retrieval_accuracy,
            "by_style": by_style,
            "rows": retrieval_rows,
        },
        "general": {
            "identical_to_base_rate": general_identical_rate,
            "rows": general_rows,
        },
        "capability": capability_rows,
        "long_context": benchmark_long_context(
            model,
            pointer,
            tokenizer,
            config,
            device,
            min(args.long_context, config.max_seq_len),
            args.min_gate,
        ),
        "promotion": {
            "retrieval_pass": retrieval_accuracy >= 0.9,
            "general_preservation_pass": general_identical_rate >= 0.75,
        },
    }
    report["promotion"]["pass"] = all(report["promotion"].values())
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(args.output.resolve()),
                "retrieval_accuracy": retrieval_accuracy,
                "retrieval_by_style": by_style,
                "general_identical_rate": general_identical_rate,
                "capability": [{"name": row["name"], "parseable": row["parseable"]} for row in capability_rows],
                "long_context": report["long_context"],
                "promotion": report["promotion"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
