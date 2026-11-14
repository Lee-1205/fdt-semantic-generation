from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.models import build_model, causal_lm_loss
from fdt_rlm.next_tools import load_next_config, make_model_config
from fdt_rlm.tokenization import load_tokenizer


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(config_path: str, model_type: str, seq_len: int, batch_size: int, device: torch.device) -> dict:
    values = load_next_config(config_path)
    values["model_type"] = model_type
    values["max_seq_len"] = seq_len
    tokenizer = load_tokenizer(values.get("tokenizer_name", str(ROOT / "tokenizers" / "fdt_v3_bpe_24k")))
    config = make_model_config(
        values,
        len(tokenizer),
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
    )
    model = build_model(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    ids = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)
    mask = torch.ones_like(ids)

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(ids, attention_mask=mask)
            loss = causal_lm_loss(output["logits"], ids, mask, config.pad_token_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss

    step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    loss = None
    for _ in range(2):
        start = time.perf_counter()
        loss = step()
        synchronize(device)
        times.append(time.perf_counter() - start)
    tokens = batch_size * seq_len * len(times)
    row = {
        "model_type": model_type,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "tokens_per_step": batch_size * seq_len,
        "loss": float(loss.detach()),
        "seconds_mean": sum(times) / len(times),
        "tokens_per_second": tokens / sum(times),
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    del optimizer, model, ids, mask, loss
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FDT v3 and Transformer context scaling.")
    parser.add_argument("--fdt-config", default=str(ROOT / "configs" / "debug_fdt_v3.yaml"))
    parser.add_argument("--transformer-config", default=str(ROOT / "configs" / "debug_transformer_v3_matched.yaml"))
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "v3_context_scaling.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    schedule = ((512, 8), (1024, 4), (2048, 2))
    rows = []
    for model_type, config_path in (
        ("fdt_v3", args.fdt_config),
        ("transformer", args.transformer_config),
    ):
        for seq_len, batch_size in schedule:
            try:
                row = measure(config_path, model_type, seq_len, batch_size, device)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                row = {
                    "model_type": model_type,
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "status": "OOM",
                    "error": str(exc).splitlines()[0],
                }
                gc.collect()
                torch.cuda.empty_cache()
            rows.append(row)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
            print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
