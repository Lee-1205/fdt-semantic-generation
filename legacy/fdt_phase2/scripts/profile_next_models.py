from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.models import build_model, causal_lm_loss
from fdt_rlm.models.causal_lm import CausalFuzzyAnchorLayer, CausalSelfAttention, MLP
from fdt_rlm.next_tools import NEXT_MODEL_TYPES, load_model_and_tokenizer, load_next_config, make_model_config
from fdt_rlm.tokenization import load_tokenizer


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def event_seconds(events, names):
    total_us = 0.0
    for event in events:
        if any(name in event.key for name in names):
            total_us += float(getattr(event, "device_time_total", 0.0) or getattr(event, "cuda_time_total", 0.0) or event.cpu_time_total)
    return total_us / 1e6


def load_target(args, device):
    if args.checkpoint:
        model, tokenizer, config, _ = load_model_and_tokenizer(args.checkpoint, device, args.model_type, args.tokenizer_name)
        return model, tokenizer, config
    tokenizer = load_tokenizer(args.tokenizer_name or str(ROOT / "tokenizers" / "fdt_bpe"))
    values = load_next_config(args.config)
    values["model_type"] = args.model_type
    config = make_model_config(values, len(tokenizer), tokenizer.pad_token_id, tokenizer.eos_token_id)
    return build_model(config).to(device), tokenizer, config


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile next FDT model components on a bounded synthetic step.")
    parser.add_argument("--model_type", choices=NEXT_MODEL_TYPES, required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default=str(ROOT / "configs" / "debug_fdt_optimized.yaml"))
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    device = torch.device(args.device)
    model, _, config = load_target(args, device)
    seq_len = min(args.seq_len, config.max_seq_len - 1)
    ids = torch.randint(0, config.vocab_size, (args.batch_size, seq_len), device=device)
    mask = torch.ones_like(ids)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    component_times = {"anchor": 0.0, "attention": 0.0, "mlp": 0.0}
    hook_handles = []

    def add_timer(module, label):
        def before(current, inputs):
            sync(device)
            current._next_profile_start = time.perf_counter()

        def after(current, inputs, output):
            sync(device)
            component_times[label] += time.perf_counter() - current._next_profile_start

        hook_handles.append(module.register_forward_pre_hook(before))
        hook_handles.append(module.register_forward_hook(after))

    for module in model.modules():
        if isinstance(module, CausalFuzzyAnchorLayer):
            add_timer(module, "anchor")
        elif isinstance(module, CausalSelfAttention):
            add_timer(module, "attention")
        elif isinstance(module, MLP):
            add_timer(module, "mlp")

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=activities, record_shapes=False) as profiler:
        sync(device)
        start = time.perf_counter()
        with torch.profiler.record_function("next_forward"):
            output = model(ids, attention_mask=mask)
            loss = causal_lm_loss(output["logits"], ids, mask, config.pad_token_id)
        sync(device)
        forward_time = time.perf_counter() - start
        start = time.perf_counter()
        with torch.profiler.record_function("next_backward"):
            loss.backward()
        sync(device)
        backward_time = time.perf_counter() - start
        start = time.perf_counter()
        with torch.profiler.record_function("next_optimizer_step"):
            optimizer.step()
        sync(device)
        optimizer_time = time.perf_counter() - start
    for handle in hook_handles:
        handle.remove()
    events = profiler.key_averages()

    model.eval()
    prompt = ids[:, : max(1, min(seq_len, 32))]
    sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        prefill_output, cache = model.prefill(prompt, torch.ones_like(prompt))
    sync(device)
    prefill_time = time.perf_counter() - start
    next_id = prefill_output["logits"][:, -1].argmax(-1, keepdim=True)
    sync(device)
    start = time.perf_counter()
    with torch.inference_mode():
        model.decode_step(next_id, cache)
    sync(device)
    decode_time = time.perf_counter() - start

    total_time = forward_time + backward_time + optimizer_time
    report = {
        "model_type": config.model_type,
        "loss": float(loss.detach()),
        "forward_time_seconds": forward_time,
        "backward_time_seconds": backward_time,
        "optimizer_step_time_seconds": optimizer_time,
        "tokens_per_sec": ids.numel() / max(total_time, 1e-9),
        "prefill_time_seconds": prefill_time,
        "decode_step_time_seconds": decode_time,
        "cache_backend": cache.get("backend", "unknown"),
        "peak_allocated_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "peak_reserved_vram_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        "anchor_routing_time_seconds": component_times["anchor"],
        "top_k_time_seconds": event_seconds(events, ["aten::topk"]),
        "causal_aggregation_time_seconds": event_seconds(events, ["aten::scatter_add", "aten::cumsum"]),
        "attention_time_seconds": component_times["attention"],
        "mlp_time_seconds": component_times["mlp"],
    }
    output_path = Path(args.output or ROOT / "artifacts" / f"next_profile_{args.model_type}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
