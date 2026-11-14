from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Any, Iterator

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fdt_rlm.models.fdt_v3 as fdt_v3  # noqa: E402
from fdt_rlm.models.causal_lm import RMSNorm  # noqa: E402
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


@torch.inference_mode()
def decode_run(model: Any, prompt: torch.Tensor, steps: int, device: torch.device) -> dict[str, Any]:
    output, cache = model.prefill(prompt, torch.ones_like(prompt))
    sync(device)
    started = time.perf_counter()
    tokens = []
    for _ in range(steps):
        next_id = output["logits"][:, -1].argmax(-1, keepdim=True)
        tokens.append(next_id)
        output, cache = model.decode_step(next_id, cache)
    sync(device)
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "tokens_per_second": steps / max(elapsed, 1e-9),
        "tokens": torch.cat(tokens, dim=1).cpu().tolist()[0],
    }


@torch.inference_mode()
def trace_logits(model: Any, prompt: torch.Tensor, steps: int) -> tuple[list[int], list[torch.Tensor]]:
    output, cache = model.prefill(prompt, torch.ones_like(prompt))
    tokens = []
    logits = []
    for _ in range(steps):
        current = output["logits"][:, -1].float()
        next_id = current.argmax(-1, keepdim=True)
        tokens.append(int(next_id.item()))
        logits.append(current.cpu())
        output, cache = model.decode_step(next_id, cache)
    return tokens, logits


@contextmanager
def fused_rms_norm(model: Any) -> Iterator[None]:
    modules = [module for module in model.modules() if isinstance(module, RMSNorm)]
    originals = [module.forward for module in modules]

    def fused(self: RMSNorm, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

    try:
        for module in modules:
            module.forward = MethodType(fused, module)
        yield
    finally:
        for module, original in zip(modules, originals):
            module.forward = original


@contextmanager
def bypass_component(model: Any, component: str) -> Iterator[None]:
    restorers = []
    original_anchor_step = None

    def zero_module(self: Any, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    try:
        if component == "anchor":
            original_anchor_step = fdt_v3._single_anchor_step

            def no_anchor(layer, x, state, token_mask, position, compute_stats=True):
                return torch.zeros_like(x), None, state

            fdt_v3._single_anchor_step = no_anchor
        elif component == "local_attention":
            for block in model.blocks:
                original = block.local_attention.step

                def no_local(self, x, state, token_mask):
                    return torch.zeros_like(x), state

                block.local_attention.step = MethodType(no_local, block.local_attention)
                restorers.append((block.local_attention, "step", original))
        elif component == "mlp":
            for block in model.blocks:
                original = block.mlp.forward
                block.mlp.forward = MethodType(zero_module, block.mlp)
                restorers.append((block.mlp, "forward", original))
        elif component == "lm_head":
            original = model.lm_head.forward

            def no_head(self, x: torch.Tensor) -> torch.Tensor:
                return x.new_zeros((*x.shape[:-1], model.config.vocab_size))

            model.lm_head.forward = MethodType(no_head, model.lm_head)
            restorers.append((model.lm_head, "forward", original))
        else:
            raise ValueError(component)
        yield
    finally:
        if original_anchor_step is not None:
            fdt_v3._single_anchor_step = original_anchor_step
        for module, attribute, original in restorers:
            setattr(module, attribute, original)


def median_run(model: Any, prompt: torch.Tensor, steps: int, device: torch.device, repeats: int = 3) -> dict[str, Any]:
    rows = [decode_run(model, prompt, steps, device) for _ in range(repeats)]
    return {
        "tokens_per_second_median": statistics.median(row["tokens_per_second"] for row in rows),
        "tokens_per_second_runs": [row["tokens_per_second"] for row in rows],
        "representative_tokens": rows[-1]["tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate original-FP32 FDT decode components without training.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--steps", type=int, default=96)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if checkpoint.drive.upper() != "C:" or output.drive.upper() != "C:":
        raise ValueError("diagnostic is C:-only")
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    model, _, config, payload = load_model_and_tokenizer(checkpoint, device, "fdt_v3", "")
    model = model.to(device=device, dtype=torch.float32).eval()
    del payload
    prompt = torch.randint(0, config.vocab_size, (1, 128), generator=torch.Generator(device=device).manual_seed(20260820), device=device)

    baseline = median_run(model, prompt, args.steps, device)
    baseline_trace, baseline_logits = trace_logits(model, prompt, 16)
    with fused_rms_norm(model):
        fused = median_run(model, prompt, args.steps, device)
        fused_trace, fused_logits = trace_logits(model, prompt, 16)
    fused["token_trace_equal"] = fused_trace == baseline_trace
    fused["max_logit_error"] = max(
        float((left - right).abs().max().item()) for left, right in zip(baseline_logits, fused_logits)
    )

    ablations = {}
    for component in ("anchor", "local_attention", "mlp", "lm_head"):
        with bypass_component(model, component):
            row = median_run(model, prompt, args.steps, device)
        row["speedup_vs_baseline"] = row["tokens_per_second_median"] / baseline["tokens_per_second_median"]
        ablations[component] = row

    fused["speedup_vs_baseline"] = fused["tokens_per_second_median"] / baseline["tokens_per_second_median"]
    report = {
        "schema_version": 1,
        "status": "DIAGNOSTIC_COMPLETED",
        "purpose": "Separate necessary model component cost from avoidable FP32 runtime dispatch overhead.",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "precision": "fp32_original_weights",
        "quantization": "none",
        "device": str(device),
        "steps_per_run": args.steps,
        "baseline": baseline,
        "fused_rms_norm": fused,
        "component_bypass_upper_bounds": ablations,
        "interpretation": (
            "Component bypasses intentionally change model behavior and are cost upper bounds, not candidate inference modes. "
            "Only fused RMSNorm is a mathematically equivalent optimization candidate and still requires token/logit checks."
        ),
    }
    atomic_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"baseline", "component_bypass_upper_bounds"}}, indent=2))


if __name__ == "__main__":
    main()
