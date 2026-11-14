from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.lexical_pointer import LexicalPointerDecodeState  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402
from fdt_rlm.next_tools import apply_repetition_penalty_  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402


def boundary(text: str) -> bool:
    return bool(re.search(r"[.!?;:]\s*$", text) or "\n" in text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.12)
    parser.add_argument("--min-pointer-gate", type=float, default=0.80)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    payload = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False, mmap=True)
    config = ModelConfig(**payload["model_config"])
    if config.model_type != "fdt_v3_anchor_indexed_dual_memory":
        raise ValueError(f"Unsupported checkpoint model type: {config.model_type}")
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    tokenizer = load_tokenizer(str(args.tokenizer.resolve()))

    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    mask = torch.ones_like(generated)
    state = LexicalPointerDecodeState(source_length=len(prompt_ids), max_activation_steps=16, max_copy_tokens=96)
    new_ids: list[int] = []
    trace = []
    memory = None
    with torch.inference_mode():
        for _ in range(min(args.max_new_tokens, config.max_seq_len - len(prompt_ids))):
            mask = torch.ones_like(generated)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(generated, attention_mask=mask)
                if memory is None:
                    memory = model.build_exact_memory(
                        output["hidden"],
                        generated,
                        mask,
                        source_length=len(prompt_ids),
                    )
                logits = output["logits"][:, -1].float()
                apply_repetition_penalty_(logits, generated[:, len(prompt_ids):], args.repetition_penalty)
                query_anchors = model.exact_route_indices(output["hidden"])[:, -1]
                mixed, diagnostics = state.prepare_logits(
                    model.exact_pointer,
                    logits,
                    output["hidden"],
                    generated,
                    mask,
                    min_gate=args.min_pointer_gate,
                    anchor_memory=memory,
                    query_anchor_ids=query_anchors,
                    max_candidate_chunks=config.exact_pointer_candidate_chunks,
                )
            selected = int(mixed.argmax(dim=-1))
            piece = tokenizer.decode([selected], skip_special_tokens=True)
            state.commit(selected, diagnostics, boundary=boundary(piece))
            diagnostics["selected_id"] = selected
            trace.append(diagnostics)
            new_ids.append(selected)
            generated = torch.cat((generated, torch.tensor([[selected]], device=device)), dim=1)
            if selected == tokenizer.eos_token_id:
                break

    print(json.dumps({
        "completion": tokenizer.decode(new_ids, skip_special_tokens=True),
        "generated_tokens": len(new_ids),
        "trace": trace,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
