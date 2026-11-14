from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model
from fdt_rlm.tokenization import load_tokenizer


def apply_repetition_penalty(logits, token_ids, penalty: float):
    if penalty <= 1.0:
        return logits
    for token_id in set(int(x) for x in token_ids):
        if 0 <= token_id < logits.numel():
            logits[token_id] = logits[token_id] * penalty if logits[token_id] < 0 else logits[token_id] / penalty
    return logits


def filter_logits(logits, top_k: int, top_p: float):
    if top_k and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits = logits.masked_fill(logits < values[-1], -float("inf"))
    if top_p and 0 < top_p < 1:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[sorted_idx[remove]] = -float("inf")
    return logits


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Generate from an FDT-RLM base checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.92)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model_cfg = ModelConfig(**checkpoint["model_config"])
    tokenizer_name = args.tokenizer_name or checkpoint.get("train_config", {}).get("tokenizer_name", "gpt2")
    tokenizer = load_tokenizer(tokenizer_name)
    model = build_model(model_cfg).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    generated = []
    for _ in range(args.max_new_tokens):
        context = ids[-model_cfg.max_seq_len :]
        input_ids = torch.tensor([context], dtype=torch.long, device=args.device)
        mask = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=mask)
        logits = out["logits"][0, -1].float()
        logits = apply_repetition_penalty(logits, generated if generated else context[-64:], args.repetition_penalty)
        if args.temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = filter_logits(logits / max(args.temperature, 1e-5), args.top_k, args.top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
        if next_id == model_cfg.eos_token_id or next_id == model_cfg.pad_token_id:
            break
        generated.append(next_id)
        ids.append(next_id)

    print(tokenizer.decode(generated, skip_special_tokens=True))


if __name__ == "__main__":
    main()

