from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.models import build_model, causal_lm_loss
from fdt_rlm.next_tools import load_next_config, make_model_config
from fdt_rlm.tokenization import load_tokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(path)


def measure(
    config_path: Path,
    model_type: str,
    seq_len: int,
    effective_tokens: int,
    measured_steps: int,
    device: torch.device,
) -> dict:
    values = load_next_config(config_path)
    values["model_type"] = model_type
    values["max_seq_len"] = seq_len
    tokenizer = load_tokenizer(values["tokenizer_name"])
    config = make_model_config(values, len(tokenizer), tokenizer.pad_token_id, tokenizer.eos_token_id)
    accumulation = effective_tokens // seq_len
    if accumulation * seq_len != effective_tokens:
        raise ValueError("effective token target must be divisible by sequence length")

    model = build_model(config).to(device).train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.1)
    ids = torch.randint(0, config.vocab_size, (1, seq_len), device=device)
    mask = torch.ones_like(ids)

    def optimizer_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        loss_value = 0.0
        for _ in range(accumulation):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(ids, attention_mask=mask)
                loss = causal_lm_loss(output["logits"], ids, mask, config.pad_token_id)
                scaled_loss = loss / accumulation
            scaled_loss.backward()
            loss_value += float(loss.detach()) / accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss_value

    optimizer_step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    loss_value = 0.0
    for _ in range(measured_steps):
        start = time.perf_counter()
        loss_value = optimizer_step()
        synchronize(device)
        times.append(time.perf_counter() - start)

    total_seconds = sum(times)
    row = {
        "status": "COMPLETED",
        "model_type": model_type,
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "parameters": parameter_count,
        "seq_len": seq_len,
        "microbatch_size": 1,
        "gradient_accumulation_steps": accumulation,
        "effective_tokens_per_optimizer_step": effective_tokens,
        "measured_optimizer_steps": measured_steps,
        "loss": loss_value,
        "seconds_per_optimizer_step_mean": total_seconds / measured_steps,
        "tokens_per_second": effective_tokens * measured_steps / total_seconds,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
    }
    del optimizer, model, ids, mask
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched 426M FDT/Transformer training-cost benchmark.")
    parser.add_argument("--fdt-config", type=Path, required=True)
    parser.add_argument("--transformer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective-tokens", type=int, default=16384)
    parser.add_argument("--measured-steps", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = {
        "status": "IN_PROGRESS",
        "training_performed": False,
        "purpose": "bounded random-initialized forward/backward cost control; not a capability comparison",
        "script_sha256": sha256(Path(__file__)),
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        },
        "rows": [],
    }
    atomic_json(args.output, payload)
    for model_type, config_path in (
        ("fdt_v3", args.fdt_config),
        ("transformer", args.transformer_config),
    ):
        for seq_len in (512, 1024, 2048):
            try:
                row = measure(
                    config_path,
                    model_type,
                    seq_len,
                    args.effective_tokens,
                    args.measured_steps,
                    device,
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                row = {
                    "status": "OOM",
                    "model_type": model_type,
                    "config": str(config_path.resolve()),
                    "config_sha256": sha256(config_path),
                    "seq_len": seq_len,
                    "effective_tokens_per_optimizer_step": args.effective_tokens,
                    "error": str(exc).splitlines()[0],
                }
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            payload["rows"].append(row)
            atomic_json(args.output, payload)
            print(json.dumps(row), flush=True)
    payload["status"] = "BENCHMARK_COMPLETED"
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
