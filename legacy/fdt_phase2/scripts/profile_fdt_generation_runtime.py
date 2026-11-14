from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.next_tools import load_model_and_tokenizer  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def repeated_ngram_start(tokens: list[int], width: int = 4) -> int | None:
    seen: dict[tuple[int, ...], int] = {}
    for index in range(max(0, len(tokens) - width + 1)):
        gram = tuple(tokens[index : index + width])
        if gram in seen:
            return index
        seen[gram] = index
    return None


@torch.inference_mode()
def benchmark_case(model: Any, config: Any, prompt_len: int, generate_len: int, device: torch.device) -> dict[str, Any]:
    generator = torch.Generator(device=device).manual_seed(20260820 + prompt_len)
    prompt = torch.randint(0, config.vocab_size, (1, prompt_len), generator=generator, device=device)
    mask = torch.ones_like(prompt)

    for _ in range(3):
        output, cache = model.prefill(prompt, mask)
        next_id = output["logits"][:, -1].argmax(-1, keepdim=True)
        model.decode_step(next_id, cache)
    sync(device)

    start = time.perf_counter()
    output, cache = model.prefill(prompt, mask)
    sync(device)
    prefill_seconds = time.perf_counter() - start

    generated: list[int] = []
    latencies: list[float] = []
    for _ in range(generate_len):
        next_id = output["logits"][:, -1].argmax(-1, keepdim=True)
        generated.append(int(next_id.item()))
        start = time.perf_counter()
        output, cache = model.decode_step(next_id, cache)
        sync(device)
        latencies.append(time.perf_counter() - start)

    return {
        "prompt_len": prompt_len,
        "generate_len": generate_len,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": prompt_len / max(prefill_seconds, 1e-9),
        "decode_seconds": sum(latencies),
        "decode_tokens_per_second": generate_len / max(sum(latencies), 1e-9),
        "first_token_latency_seconds": latencies[0],
        "median_inter_token_latency_seconds": statistics.median(latencies),
        "p95_inter_token_latency_seconds": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        "repeated_4gram_start": repeated_ngram_start(generated),
    }


@torch.inference_mode()
def profile_decode(model: Any, config: Any, device: torch.device) -> list[dict[str, Any]]:
    prompt = torch.randint(0, config.vocab_size, (1, 128), device=device)
    output, cache = model.prefill(prompt, torch.ones_like(prompt))
    for _ in range(4):
        next_id = output["logits"][:, -1].argmax(-1, keepdim=True)
        output, cache = model.decode_step(next_id, cache)
    sync(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as profiler:
        for _ in range(16):
            next_id = output["logits"][:, -1].argmax(-1, keepdim=True)
            output, cache = model.decode_step(next_id, cache)
        sync(device)

    rows = []
    for event in profiler.key_averages().table(sort_by="cuda_time_total", row_limit=30).splitlines():
        if event.strip():
            rows.append({"table_row": event})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile original-FP32 FDT checkpoint generation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if checkpoint.drive.upper() != "C:" or output.drive.upper() != "C:":
        raise ValueError("generation profiling is C:-only")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    model, _, config, payload = load_model_and_tokenizer(checkpoint, device, "fdt_v3", "")
    model = model.to(device=device, dtype=torch.float32).eval()
    del payload

    report = {
        "schema_version": 1,
        "status": "PROFILE_COMPLETED",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "precision": "fp32_original",
        "quantization": "none",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "model": {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "max_seq_len": config.max_seq_len,
            "layers": config.n_layers,
            "dimension": config.dim,
            "local_attention_window": config.local_attention_window,
            "anchors": config.num_anchors,
            "top_k": config.top_k,
        },
        "cases": [
            benchmark_case(model, config, prompt_len, 128, device)
            for prompt_len in (16, 128, 512, 1400)
            if prompt_len + 128 <= config.max_seq_len
        ],
        "profiler_table": profile_decode(model, config, device),
    }
    atomic_json(output, report)
    print(json.dumps({"status": report["status"], "output": str(output), "cases": report["cases"]}, indent=2))


if __name__ == "__main__":
    main()
