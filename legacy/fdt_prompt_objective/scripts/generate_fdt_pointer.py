from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.lexical_pointer import LexicalPointerDecodeState, SparseLexicalPointer
from fdt_rlm.next_tools import load_model_and_tokenizer, safe_torch_load


def apply_repetition_penalty(logits, generated_ids, penalty: float):
    if penalty <= 1.0 or not generated_ids:
        return logits
    token_ids = torch.tensor(sorted(set(generated_ids)), device=logits.device)
    selected = logits.index_select(1, token_ids)
    selected = torch.where(selected < 0, selected * penalty, selected / penalty)
    return logits.scatter(1, token_ids.unsqueeze(0), selected)


def sample_token(logits, temperature: float, top_k: int, top_p: float, generator):
    if temperature <= 0:
        return int(logits.argmax(dim=-1))
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        threshold = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < threshold, -float("inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = logits.sort(descending=True)
        cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf")).scatter(1, sorted_indices, sorted_logits)
    probabilities = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probabilities, 1, generator=generator))


def is_boundary(piece: str) -> bool:
    return bool(re.search(r"[.!?;:]\s*$", piece) or "\n" in piece)


def main():
    parser = argparse.ArgumentParser(description="Generate with FDT and an explainable O(NW) lexical pointer")
    parser.add_argument("--pointer-checkpoint", type=Path, default=ROOT / "runs" / "fdt_v3_lexical_pointer_capability_staged" / "best.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--min-pointer-gate", type=float, default=0.8)
    parser.add_argument("--pointer-activation-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    pointer_payload = safe_torch_load(args.pointer_checkpoint.resolve())
    model, tokenizer, _, _ = load_model_and_tokenizer(
        Path(pointer_payload["base_checkpoint"]), device, "fdt_v3"
    )
    model = model.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    pointer = SparseLexicalPointer(**pointer_payload["pointer_config"]).to(device)
    pointer.load_state_dict(pointer_payload["pointer_state_dict"], strict=True)
    pointer.eval()

    source_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    generated = torch.tensor([source_ids], device=device)
    state = LexicalPointerDecodeState(
        source_length=len(source_ids),
        max_activation_steps=args.pointer_activation_steps,
    )
    rng = torch.Generator(device=device).manual_seed(args.seed)
    new_ids: list[int] = []
    trace = []
    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            mask = torch.ones_like(generated)
            output = model(generated, attention_mask=mask)
            base_logits = apply_repetition_penalty(
                output["logits"][:, -1].float(), new_ids, args.repetition_penalty
            )
            logits, diagnostics = state.prepare_logits(
                pointer,
                base_logits,
                output["hidden"],
                generated,
                mask,
                min_gate=args.min_pointer_gate,
            )
            selected_id = sample_token(
                logits,
                0.0 if diagnostics.get("mode") == "cursor" else args.temperature,
                args.top_k,
                args.top_p,
                rng,
            )
            piece = tokenizer.decode([selected_id], skip_special_tokens=True)
            state.commit(selected_id, diagnostics, boundary=is_boundary(piece))
            diagnostics["selected_id"] = selected_id
            trace.append(diagnostics)
            new_ids.append(selected_id)
            generated = torch.cat(
                (generated, torch.tensor([[selected_id]], device=device)), dim=1
            )
            if selected_id == tokenizer.eos_token_id:
                break

    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(completion)
    if args.trace_output:
        args.trace_output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.trace_output.resolve().write_text(
            json.dumps(
                {
                    "prompt": args.prompt,
                    "completion": completion,
                    "pointer_checkpoint": str(args.pointer_checkpoint.resolve()),
                    "trace": trace,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
