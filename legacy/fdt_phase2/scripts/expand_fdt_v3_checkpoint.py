from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.config import ModelConfig
from fdt_rlm.model_growth import (
    expanded_fdt_v3_config,
    grow_fdt_v3_state_dict,
    unique_parameter_count,
)
from fdt_rlm.models import build_model
from fdt_rlm.next_tools import safe_torch_load


def main() -> None:
    parser = argparse.ArgumentParser(description="Function-preserving FDT v3 width/depth expansion")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-dim", type=int, required=True)
    parser.add_argument("--target-layers", type=int, required=True)
    parser.add_argument("--target-heads", type=int, required=True)
    parser.add_argument("--target-router-dim", type=int, required=True)
    parser.add_argument("--noise-std", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()

    source_path = args.source.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    checkpoint = safe_torch_load(source_path, "cpu")
    source_config = ModelConfig(**checkpoint["model_config"])
    target_config = expanded_fdt_v3_config(
        source_config,
        target_dim=args.target_dim,
        target_layers=args.target_layers,
        target_heads=args.target_heads,
        target_router_dim=args.target_router_dim,
    )
    target_model = build_model(target_config).cpu()
    grown_state = grow_fdt_v3_state_dict(
        checkpoint["model_state_dict"],
        target_model,
        source_layers=source_config.n_layers,
        noise_std=args.noise_std,
        seed=args.seed,
    )
    target_model.load_state_dict(grown_state, strict=True)
    payload = {
        key: value
        for key, value in checkpoint.items()
        if key not in {"model_state_dict", "optimizer_state_dict", "model_config"}
    }
    payload.update(
        model_state_dict=grown_state,
        model_config=asdict(target_config),
        source_checkpoint=str(source_path),
        growth_migration={
            "source_dim": source_config.dim,
            "target_dim": target_config.dim,
            "source_layers": source_config.n_layers,
            "target_layers": target_config.n_layers,
            "source_heads": source_config.n_heads,
            "target_heads": target_config.n_heads,
            "source_router_dim": source_config.router_dim,
            "target_router_dim": target_config.router_dim,
            "noise_std": args.noise_std,
            "seed": args.seed,
            "optimizer_reset": True,
            "complexity": "O(N(K+W))",
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    report = {
        "status": "PASS",
        "source": str(source_path),
        "output": str(output_path),
        "source_parameters": sum(parameter.numel() for parameter in build_model(source_config).parameters()),
        "target_parameters": unique_parameter_count(target_model),
        "model_config": asdict(target_config),
        "tokens_seen": payload.get("tokens_seen", 0),
        "optimizer_step": payload.get("optimizer_step", 0),
    }
    report_path = output_path.with_suffix(".growth.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
