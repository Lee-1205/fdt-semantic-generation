from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model
from fdt_rlm.next_tools import safe_torch_load
import evaluate_base_model_strict as strict
import evaluate_base_model_strict_v3 as strict_v3
from prepare_capability_completion_v15 import require_c_path
from prepare_capability_completion_v19 import CATEGORY_WEIGHTS, file_sha256
from train_capability_completion_v15 import strict_smoke, verify_sha256
from train_capability_completion_v17 import balanced_order
from train_capability_completion_v19_r2 import (
    copy_grounding_loss,
    load_split,
    move_optimizer_state,
    optimizer_state_devices,
)


def gib(value: int) -> float:
    return value / (1024**3)


def memory_snapshot() -> dict:
    return {
        "allocated_gib": gib(torch.cuda.memory_allocated()),
        "reserved_gib": gib(torch.cuda.memory_reserved()),
        "max_allocated_gib": gib(torch.cuda.max_memory_allocated()),
        "max_reserved_gib": gib(torch.cuda.max_memory_reserved()),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-checkpoint V19 R2 memory-safety diagnostic.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--broad-dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for path in (args.checkpoint, args.dataset_dir, args.broad_dataset_dir, args.output):
        require_c_path(path)
    if args.output.exists():
        raise FileExistsError(f"diagnostic output already exists: {args.output}")

    checkpoint_hash = verify_sha256(args.checkpoint, args.expected_checkpoint_sha256, "V18 parent")
    checkpoint = safe_torch_load(args.checkpoint, "cpu")
    config = ModelConfig(**checkpoint["model_config"])
    tokenizer_name = checkpoint.get("train_config", {}).get("tokenizer_name", "") or "gpt2"
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    device = torch.device(args.device)
    model.to(device)
    frozen = []
    for name, parameter in model.named_parameters():
        if ".anchor." in name:
            parameter.requires_grad_(False)
            frozen.append(name)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2.5e-8, weight_decay=0.01)

    train = load_split(args.dataset_dir, "train")
    val = load_split(args.dataset_dir, "validation")
    order = balanced_order(train["category_ids"], 20260819)
    batch_indices = order[:10]
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    logged_loss = 0.0
    for micro_indices in (batch_indices[0::2], batch_indices[1::2]):
        x = train["input_ids"].index_select(0, micro_indices).long().to(device)
        y = train["labels"].index_select(0, micro_indices).long().to(device)
        mask = train["attention_mask"].index_select(0, micro_indices).long().to(device)
        cats = train["category_ids"].index_select(0, micro_indices).long().to(device)
        grounded = train["grounding_mask"].index_select(0, micro_indices).long().to(device)
        signatures = train["signature_mask"].index_select(0, micro_indices).long().to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x, attention_mask=mask)["logits"]
            completion, prompt, _ = copy_grounding_loss(
                logits, x, y, mask, cats, grounded, signatures, config.eos_token_id,
                2.0, 2.0, 3.0, CATEGORY_WEIGHTS,
            )
            loss = completion + 0.25 * prompt
        (loss / 2.0).backward()
        logged_loss += float(loss.detach()) / 2.0
        del x, y, mask, cats, grounded, signatures, logits, completion, prompt, loss
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 0.7)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    train_memory = memory_snapshot()
    no_gradients = all(parameter.grad is None for parameter in trainable)

    move_optimizer_state(optimizer, torch.device("cpu"))
    torch.cuda.empty_cache()
    offloaded_memory = memory_snapshot()
    offloaded_devices = sorted(optimizer_state_devices(optimizer))

    broad_ids, broad_masks = strict.split_rows(args.broad_dataset_dir, "validation")
    broad_manifest = json.loads((args.broad_dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    broad_ranges = strict.bucket_ranges(broad_manifest, "validation")
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    capability_payload = {
        "input_ids": val["input_ids"],
        "labels": val["labels"],
        "attention_mask": val["attention_mask"],
    }
    capability_ranges = strict_v3.capability_ranges(manifest)
    tokenizer = strict.load_tokenizer(tokenizer_name)
    torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.inference_mode():
        smoke = strict_smoke(
            model, tokenizer, config, capability_payload, capability_ranges,
            broad_ids, broad_masks, broad_ranges, device,
        )
    evaluation_memory = memory_snapshot()
    payload = {
        "status": "PASS" if no_gradients and offloaded_devices == ["cpu"] else "FAIL",
        "purpose": "validate memory-safe two-microbatch training and optimizer-offloaded smoke evaluation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_manifest_sha256": file_sha256(args.dataset_dir / "manifest.json"),
        "effective_batch_size": 10,
        "micro_batch_size": 5,
        "gradient_accumulation_steps": 2,
        "diagnostic_step_loss": logged_loss,
        "gradient_norm": float(grad_norm.detach()),
        "all_gradients_released_before_eval": no_gradients,
        "optimizer_state_devices_during_eval": offloaded_devices,
        "training_memory": train_memory,
        "after_optimizer_offload_memory": offloaded_memory,
        "smoke_evaluation_memory": evaluation_memory,
        "smoke_summary": {
            "broad_loss": smoke["heldout"]["loss"],
            "dead_anchor_ratio": smoke["heldout"]["routing"].get("dead_anchor_ratio", 0.0),
            "python_structural": smoke["capability_generation"]["python_code"]["usable_structure"],
            "json_valid": smoke["capability_generation"]["json"]["valid"],
        },
        "persistent_weight_change": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
