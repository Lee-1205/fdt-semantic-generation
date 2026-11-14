from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.data.manifest import write_json
from fdt_rlm.models import anchor_regularization_losses, build_model, causal_lm_loss
from fdt_rlm.tokenization import load_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["transformer", "fdt"], required=True)
    p.add_argument("--tokenizer", default=str(ROOT / "tokenizers" / "fdt_bpe"))
    p.add_argument("--output", default="")
    p.add_argument("--dim", type=int, default=384)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=6)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--aggregation-impl",
        choices=["dense_reference", "dense_gather_first"],
        default="dense_reference",
    )
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--include-regularizers", action="store_true")
    p.add_argument("--profile-diversity-weight", type=float, default=0.10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    tok = load_tokenizer(args.tokenizer)
    cfg = ModelConfig(
        vocab_size=len(tok),
        pad_token_id=int(tok.pad_token_id),
        eos_token_id=int(tok.eos_token_id),
        model_type=args.model_type,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len,
        aggregation_impl=args.aggregation_impl,
    )
    model = build_model(cfg).to(device)
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ids = torch.randint(0, len(tok), (args.batch_size, args.max_seq_len), device=device)
    mask = torch.ones_like(ids)
    autocast_enabled = device.type == "cuda" and args.precision == "bf16"

    def run_step():
        model.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            out = model(ids, attention_mask=mask)
            loss = causal_lm_loss(out["logits"], ids, mask, cfg.pad_token_id)
            if args.include_regularizers:
                reg = anchor_regularization_losses(
                    model,
                    out["anchor_stats"],
                    compute_diversity=args.profile_diversity_weight > 0,
                )
                loss = (
                    loss
                    + 0.01 * reg["cluster"]
                    + args.profile_diversity_weight * reg["diversity"]
                    + 0.005 * reg["usage"]
                    - 0.003 * reg["entropy_reward"]
                    + 0.10 * reg["entropy_over"]
                )
        loss.backward()

    for _ in range(args.warmup):
        run_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(args.iterations):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        run_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    elapsed = statistics.mean(timings)
    report = {
        "model_type": args.model_type,
        "parameters": params,
        "trainable_parameters": trainable,
        "device": str(device),
        "batch_size": args.batch_size,
        "context": args.max_seq_len,
        "aggregation_impl": args.aggregation_impl,
        "precision": args.precision,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "include_regularizers": args.include_regularizers,
        "step_time_sec": elapsed,
        "median_step_time_sec": statistics.median(timings),
        "tokens_per_sec": args.batch_size * args.max_seq_len / max(elapsed, 1e-9),
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
    }
    out_path = Path(args.output or ROOT / "artifacts" / f"profile_{args.model_type}.json")
    write_json(out_path, report)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
