from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.models import build_model, causal_lm_loss
from fdt_rlm.next_tools import NEXT_MODEL_TYPES, load_next_config, make_model_config
from fdt_rlm.tokenization import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded context-scaling benchmark.")
    parser.add_argument("--model_type", choices=NEXT_MODEL_TYPES, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer_name", default=str(ROOT / "tokenizers" / "fdt_bpe"))
    parser.add_argument("--seq_lens", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "context_scaling.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.tokenizer_name)
    values = load_next_config(args.config)
    values["model_type"] = args.model_type
    base = make_model_config(values, len(tokenizer), tokenizer.pad_token_id, tokenizer.eos_token_id)
    lengths = args.seq_lens[:1] if args.dry_run else args.seq_lens
    rows = []
    for seq_len in lengths:
        try:
            config = replace(base, max_seq_len=seq_len)
            model = build_model(config).to(device).eval()
            ids = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device=device)
            mask = torch.ones_like(ids)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                output = model(ids, attention_mask=mask)
                loss = causal_lm_loss(output["logits"], ids, mask, config.pad_token_id)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            rows.append({"seq_len": seq_len, "loss": float(loss), "seconds": elapsed, "tokens_per_sec": ids.numel() / max(elapsed, 1e-9), "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except RuntimeError as exc:
            rows.append({"seq_len": seq_len, "error": str(exc)})
            if device.type == "cuda":
                torch.cuda.empty_cache()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(rows), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
