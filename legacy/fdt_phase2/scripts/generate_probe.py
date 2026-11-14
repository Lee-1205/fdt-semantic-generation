from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.next_tools import NEXT_MODEL_TYPES, choose_onk_cache, generate_ids, load_model_and_tokenizer


def load_prompts(path: str | Path, limit: int = 0) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


@torch.inference_mode()
def run_probes(model, tokenizer, config, prompts, args, device):
    rows = []
    for index, item in enumerate(prompts):
        prompt = str(item["prompt"])
        ids = tokenizer.encode(prompt, add_special_tokens=False)[-config.max_seq_len :]
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        max_new = min(args.max_new_tokens, max(config.max_seq_len - len(ids), 0))
        explicit_cache = getattr(args, "use_cache", None)
        cache_policy = getattr(args, "cache_policy", "auto")
        if explicit_cache is not None:
            use_cache = explicit_cache
        elif cache_policy == "on":
            use_cache = True
        elif cache_policy == "off":
            use_cache = False
        else:
            use_cache = choose_onk_cache(
                config,
                len(ids),
                getattr(args, "effective_precision", "fp32"),
                getattr(args, "memory_priority", False),
            )
        torch.manual_seed(args.seed + index)
        generated = generate_ids(
            model,
            input_ids,
            max_new,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=getattr(args, "top_k", 0),
            repetition_penalty=getattr(args, "repetition_penalty", 1.0),
            greedy=args.greedy,
            use_cache=use_cache,
        )
        completion_ids = generated[0, len(ids) :].tolist()
        rows.append(
            {
                "category": item.get("category", "unknown"),
                "prompt": prompt,
                "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
                "new_tokens": len(completion_ids),
                "cache_backend": (
                    "transformer_kv"
                    if use_cache and config.model_type == "transformer"
                    else "fdt_v3_incremental"
                    if use_cache and config.model_type == "fdt_v3"
                    else "anchor_incremental"
                    if use_cache
                    else "recompute"
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate bounded probes from a next-model checkpoint.")
    parser.add_argument("--model_type", choices=NEXT_MODEL_TYPES, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--prompt_file", default=str(ROOT / "eval_prompts" / "base_probe_prompts.jsonl"))
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--use_cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--cache_policy", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--memory_priority", action="store_true")
    parser.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    parser.add_argument("--num_prompts", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_jsonl", default=str(ROOT / "artifacts" / "next_generation_probes.jsonl"))
    args = parser.parse_args()
    device = torch.device(args.device)
    model, tokenizer, config, _ = load_model_and_tokenizer(
        args.checkpoint, device, args.model_type, args.tokenizer_name
    )
    args.effective_precision = "bf16" if args.precision == "auto" and device.type == "cuda" else args.precision
    if args.effective_precision == "bf16" and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    elif args.effective_precision == "auto":
        args.effective_precision = "fp32"
    rows = run_probes(model, tokenizer, config, load_prompts(args.prompt_file, args.num_prompts), args, device)
    output = Path(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"status": "PASS", "prompts": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
