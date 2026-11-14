from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.config import ModelConfig
from fdt_rlm.model_growth import unique_parameter_count
from fdt_rlm.models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure one full FDT v3 AdamW training step")
    parser.add_argument("--dim", type=int, required=True)
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--router-dim", type=int, required=True)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-size", type=int, default=24576)
    parser.add_argument("--anchors", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--precision", choices=("bf16", "fp16"), default="bf16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the scale profiler")
    anchor_layers = list(range(0, args.layers, 2))
    config = ModelConfig(
        vocab_size=args.vocab_size,
        pad_token_id=0,
        eos_token_id=2,
        model_type="fdt_v3",
        dim=args.dim,
        n_layers=args.layers,
        n_heads=args.heads,
        mlp_ratio=4,
        max_seq_len=args.seq_len,
        dropout=0.0,
        tie_embeddings=True,
        use_rope=False,
        use_self_attention=False,
        num_anchors=args.anchors,
        top_k=args.top_k,
        routing_type="cosine",
        aggregation_impl="sparse_chunked_scan",
        routing_backend="matmul",
        enable_anchor_metrics=True,
        compute_anchor_metrics_every=4,
        enable_diagnostics=False,
        use_cached_normalized_anchors=True,
        use_incremental_anchor_state=True,
        detach_diagnostics=True,
        anchor_layer_pattern="alternate",
        anchor_layer_indices=anchor_layers,
        local_attention_window=args.window,
        router_dim=args.router_dim,
        anchor_scan_chunk_size=64,
        anchor_recency_bias=4.0,
        anchor_recency_reference_len=512,
    )
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    status = "PASS"
    error = ""
    model = None
    try:
        model = build_model(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.95))
        ids = torch.randint(3, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        mask = torch.ones_like(ids)
        with torch.amp.autocast("cuda", dtype=dtype):
            output = model(ids, attention_mask=mask)
            logits = output["logits"][:, :-1].float()
            loss = F.cross_entropy(logits.transpose(1, 2), ids[:, 1:])
            anchor_terms = [stat.cluster_loss for stat in output["anchor_stats"]]
            if anchor_terms:
                loss = loss + 0.005 * torch.stack(anchor_terms).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
    except torch.OutOfMemoryError as exc:
        status = "OOM"
        error = str(exc)
    elapsed = time.perf_counter() - started
    result = {
        "status": status,
        "error": error,
        "dim": args.dim,
        "layers": args.layers,
        "heads": args.heads,
        "router_dim": args.router_dim,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "parameters": unique_parameter_count(model) if model is not None else None,
        "elapsed_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "complexity": "O(N(K+W))",
    }
    print(json.dumps(result, indent=2))
    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
