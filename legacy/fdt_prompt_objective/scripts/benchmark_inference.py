from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.models import build_model
from fdt_rlm.next_tools import NEXT_MODEL_TYPES, load_model_and_tokenizer, load_next_config, make_model_config
from fdt_rlm.tokenization import load_tokenizer


def as_bool(value: str) -> bool:
    value = value.lower()
    if value not in {"true", "false", "1", "0", "yes", "no"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return value in {"true", "1", "yes"}


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def tensor_storage_bytes(value, seen: set[int] | None = None) -> int:
    """Count unique tensor storage held by a nested inference cache."""
    seen = set() if seen is None else seen
    if torch.is_tensor(value):
        storage = value.untyped_storage()
        identity = storage.data_ptr()
        if identity in seen:
            return 0
        seen.add(identity)
        return storage.nbytes()
    if isinstance(value, dict):
        return sum(tensor_storage_bytes(item, seen) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_storage_bytes(item, seen) for item in value)
    if hasattr(value, "__dict__"):
        return tensor_storage_bytes(vars(value), seen)
    return 0


def load_target(args, device):
    if args.checkpoint:
        model, tokenizer, config, _ = load_model_and_tokenizer(
            args.checkpoint, device, args.model_type, args.tokenizer_name
        )
        return model, tokenizer, config
    values = load_next_config(args.config)
    values["model_type"] = args.model_type
    tokenizer = load_tokenizer(args.tokenizer_name or str(ROOT / "tokenizers" / "fdt_bpe"))
    config = make_model_config(values, len(tokenizer), tokenizer.pad_token_id, tokenizer.eos_token_id)
    return build_model(config).to(device).eval(), tokenizer, config


@torch.inference_mode()
def benchmark(
    model,
    config,
    prompt_len: int,
    generate_len: int,
    use_cache: bool,
    device: torch.device,
    batch_size: int = 1,
    warmup_steps: int = 2,
):
    if prompt_len + generate_len > config.max_seq_len:
        raise ValueError("prompt_len + generate_len exceeds model max_seq_len")
    ids = torch.randint(0, config.vocab_size, (batch_size, prompt_len), device=device)
    mask = torch.ones_like(ids)

    for _ in range(max(warmup_steps, 0)):
        if use_cache:
            warm_output, warm_cache = model.prefill(ids, mask)
            if prompt_len < config.max_seq_len:
                warm_id = warm_output["logits"][:, -1].argmax(dim=-1, keepdim=True)
                model.decode_step(warm_id, warm_cache)
        else:
            model(ids, attention_mask=mask)
    sync(device)
    if use_cache and warmup_steps > 0:
        del warm_output, warm_cache
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sync(device)
    start = time.perf_counter()
    if use_cache:
        output, cache = model.prefill(ids, mask)
    else:
        output, cache = model(ids, attention_mask=mask), None
    sync(device)
    prefill_seconds = time.perf_counter() - start

    generated = ids
    latencies = []
    for _ in range(generate_len):
        next_id = output["logits"][:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_id), dim=1)
        sync(device)
        start = time.perf_counter()
        if use_cache:
            output, cache = model.decode_step(next_id, cache)
        else:
            output = model(generated, attention_mask=torch.ones_like(generated))
        sync(device)
        latencies.append(time.perf_counter() - start)

    decode_seconds = sum(latencies)
    cache_bytes = tensor_storage_bytes(cache) if cache else 0
    return {
        "model_type": config.model_type,
        "cache_backend": cache.get("backend", "disabled") if cache else "disabled",
        "batch_size": batch_size,
        "prompt_len": prompt_len,
        "generate_len": generate_len,
        "use_cache": use_cache,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_sec": prompt_len / max(prefill_seconds, 1e-9),
        "decode_seconds": decode_seconds,
        "decode_tokens_per_sec": generate_len / max(decode_seconds, 1e-9),
        "first_token_latency_seconds": latencies[0] if latencies else 0.0,
        "mean_inter_token_latency_seconds": statistics.mean(latencies) if latencies else 0.0,
        "cache_bytes": cache_bytes,
        "cache_mebibytes": cache_bytes / (1024**2),
        "peak_vram_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_vram_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark next-model prefill and decode.")
    parser.add_argument("--model_type", choices=NEXT_MODEL_TYPES, required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default=str(ROOT / "configs" / "debug_fdt_optimized.yaml"))
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--prompt_len", type=int, default=32)
    parser.add_argument("--generate_len", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup_steps", type=int, default=2)
    parser.add_argument("--use_cache", type=as_bool, default=True)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, _, config = load_target(args, device)
    if args.precision == "bf16" and device.type == "cuda":
        model = model.to(dtype=torch.bfloat16)
    report = benchmark(
        model,
        config,
        args.prompt_len,
        args.generate_len,
        args.use_cache,
        device,
        args.batch_size,
        args.warmup_steps,
    )
    report["precision"] = args.precision if device.type == "cuda" else "fp32"
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
