from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model
from fdt_rlm.next_tools import safe_torch_load

import evaluate_base_model_strict as strict
import evaluate_base_model_strict_v3 as strict_v3
from prepare_capability_completion_v16 import CATEGORY_NAMES, profile_indices
from train_capability_completion_v15 import strict_smoke, verify_sha256
from train_chat_sft import evaluate_chat, masked_chat_losses


def require_c_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"V16 permits only C: paths: {resolved}")
    return resolved


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_master(dataset_dir: Path):
    paths = sorted((dataset_dir / "shards" / "train").glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No V16 train shards under {dataset_dir}")
    payloads = [safe_torch_load(path, "cpu") for path in paths]
    return tuple(
        torch.cat([payload[key] for payload in payloads], dim=0)
        for key in ("input_ids", "labels", "attention_mask", "category_ids")
    )


def load_validation(dataset_dir: Path):
    payloads = [
        safe_torch_load(path, "cpu")
        for path in sorted((dataset_dir / "shards" / "validation").glob("*.pt"))
    ]
    return tuple(
        torch.cat([payload[key] for payload in payloads], dim=0)
        for key in ("input_ids", "labels", "attention_mask")
    )


def build_parent(checkpoint_path: Path, device: torch.device):
    checkpoint = safe_torch_load(checkpoint_path, "cpu")
    config = ModelConfig(**checkpoint["model_config"])
    tokenizer_name = checkpoint.get("train_config", {}).get("tokenizer_name", "") or "gpt2"
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    model.to(device)
    return model, config, tokenizer_name


def frozen_trainable(model):
    for name, parameter in model.named_parameters():
        if ".anchor." in name:
            parameter.requires_grad_(False)
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def train_smoke_candidate(
    checkpoint_path: Path,
    profile_name: str,
    profile: dict,
    train_payload,
    validation_payload,
    capability_payload,
    capability_ranges,
    broad_ids,
    broad_masks,
    broad_ranges,
    tokenizer,
    baseline: dict,
    args,
    device: torch.device,
) -> dict:
    model, config, tokenizer_name = build_parent(checkpoint_path, device)
    del tokenizer_name
    trainable = frozen_trainable(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    ids, labels, masks, category_ids = train_payload
    indices = profile_indices(category_ids, profile, args.seed + 100 + len(profile_name))
    needed = args.smoke_steps * args.batch_size
    if len(indices) < needed:
        raise ValueError(f"Profile {profile_name} has only {len(indices)} rows for {needed} smoke rows")
    indices = indices[:needed]
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    started = time.perf_counter()
    model.train()
    losses = []
    grad_norms = []
    for start in range(0, needed, args.batch_size):
        batch = indices[start : start + args.batch_size]
        x = ids.index_select(0, batch).long().to(device, non_blocking=True)
        y = labels.index_select(0, batch).long().to(device, non_blocking=True)
        mask = masks.index_select(0, batch).long().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x, attention_mask=mask)["logits"]
            completion_loss, prompt_loss = masked_chat_losses(logits, x, y, mask)
            loss = completion_loss + float(profile["prompt_lm_weight"]) * prompt_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite V16 smoke loss for {profile_name}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite V16 smoke gradient for {profile_name}")
        optimizer.step()
        losses.append(float(loss))
        grad_norms.append(float(grad_norm))

    model.eval()
    smoke = strict_smoke(
        model,
        tokenizer,
        config,
        capability_payload,
        capability_ranges,
        broad_ids,
        broad_masks,
        broad_ranges,
        device,
    )
    val_ids, val_labels, val_masks = validation_payload
    quick_rows = min(args.capability_rows, len(val_ids))
    capability = evaluate_chat(
        model,
        val_ids[:quick_rows],
        val_labels[:quick_rows],
        val_masks[:quick_rows],
        args.batch_size,
        device,
    )
    broad_relative = smoke["heldout"]["loss"] / baseline["heldout"]["loss"] - 1.0
    repetition_delta = (
        smoke["prefix_generation"]["mean_repetition_rate"]
        - baseline["prefix_generation"]["mean_repetition_rate"]
    )
    loop_free_delta = (
        smoke["prefix_generation"]["loop_free"]["rate"]
        - baseline["prefix_generation"]["loop_free"]["rate"]
    )
    capability_delta = capability["loss"] - baseline["quick_capability"]["loss"]
    dead = smoke["heldout"]["routing"].get("dead_anchor_ratio", 0.0)
    eligible = (
        math.isfinite(smoke["heldout"]["loss"])
        and broad_relative <= args.max_broad_relative_regression
        and repetition_delta <= args.max_prefix_repetition_increase
        and loop_free_delta >= -args.max_loop_free_decrease
        and dead <= args.max_dead_anchor_ratio
    )
    score = capability_delta + 2.0 * max(repetition_delta, 0.0) - 0.10 * loop_free_delta
    report = {
        "profile": profile_name,
        "profile_config": profile,
        "steps": args.smoke_steps,
        "rows_seen": needed,
        "mean_train_loss": sum(losses) / len(losses),
        "max_grad_norm": max(grad_norms),
        "elapsed_seconds": time.perf_counter() - started,
        "quick_capability": capability,
        "quick_capability_loss_delta": capability_delta,
        "broad_relative_delta": broad_relative,
        "prefix_repetition_delta": repetition_delta,
        "prefix_loop_free_delta": loop_free_delta,
        "dead_anchor_ratio": dead,
        "eligible": eligible,
        "selection_score": score,
        "smoke": smoke,
    }
    del optimizer, trainable, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Predeclared V16 retention-mix smoke selector.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--broad-dataset-dir",
        type=Path,
        default=ROOT / "prepared_data" / "v3_broad_additional_300m_combined_v2_24k_ctx2048",
    )
    parser.add_argument("--smoke-steps", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--capability-rows", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.7)
    parser.add_argument("--max-broad-relative-regression", type=float, default=0.0015)
    parser.add_argument("--max-prefix-repetition-increase", type=float, default=0.01)
    parser.add_argument("--max-loop-free-decrease", type=float, default=0.10)
    parser.add_argument("--max-dead-anchor-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for path in (args.checkpoint, args.dataset_dir, args.output, args.broad_dataset_dir):
        require_c_path(path)
    if args.output.exists():
        raise FileExistsError(f"V16 selector output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = verify_sha256(
        args.checkpoint, args.expected_checkpoint_sha256, "Promoted 520M checkpoint"
    )
    manifest_hash = verify_sha256(
        args.dataset_dir / "manifest.json", args.expected_manifest_sha256, "V16 manifest"
    )
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    profiles = manifest["profiles"]
    if tuple(manifest["train"]["category_names"]) != CATEGORY_NAMES:
        raise ValueError("V16 category order mismatch")

    device = torch.device(args.device)
    train_payload = load_master(args.dataset_dir)
    validation_payload = load_validation(args.dataset_dir)
    val_ids, val_labels, val_masks = validation_payload
    broad_ids, broad_masks = strict.split_rows(args.broad_dataset_dir, "validation")
    broad_manifest = json.loads((args.broad_dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    broad_ranges = strict.bucket_ranges(broad_manifest, "validation")
    capability_payload = {"input_ids": val_ids, "labels": val_labels, "attention_mask": val_masks}
    capability_ranges = strict_v3.capability_ranges(manifest)

    parent_model, config, tokenizer_name = build_parent(args.checkpoint, device)
    tokenizer = strict.load_tokenizer(tokenizer_name)
    baseline_smoke = strict_smoke(
        parent_model,
        tokenizer,
        config,
        capability_payload,
        capability_ranges,
        broad_ids,
        broad_masks,
        broad_ranges,
        device,
    )
    quick_rows = min(args.capability_rows, len(val_ids))
    baseline_quick = evaluate_chat(
        parent_model,
        val_ids[:quick_rows],
        val_labels[:quick_rows],
        val_masks[:quick_rows],
        args.batch_size,
        device,
    )
    baseline = {**baseline_smoke, "quick_capability": baseline_quick}
    del parent_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    candidates = []
    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    atomic_write_json(
        partial_path,
        {
            "status": "BASELINE_COMPLETE",
            "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": manifest_hash,
            "smoke_steps": args.smoke_steps,
            "baseline": baseline,
            "candidates": candidates,
        },
    )
    for profile_name in ("capability_heavy", "balanced", "retention_heavy"):
        candidates.append(
            train_smoke_candidate(
                args.checkpoint,
                profile_name,
                profiles[profile_name],
                train_payload,
                validation_payload,
                capability_payload,
                capability_ranges,
                broad_ids,
                broad_masks,
                broad_ranges,
                tokenizer,
                baseline,
                args,
                device,
            )
        )
        atomic_write_json(
            partial_path,
            {
                "status": "CANDIDATE_COMPLETE",
                "checkpoint_sha256": checkpoint_hash,
                "manifest_sha256": manifest_hash,
                "smoke_steps": args.smoke_steps,
                "baseline": baseline,
                "candidates": candidates,
            },
        )

    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    selected = min(eligible, key=lambda candidate: candidate["selection_score"]) if eligible else None
    payload = {
        "status": "SELECTED" if selected else "NO_ELIGIBLE_CANDIDATE",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest_sha256": manifest_hash,
        "selection_rule": {
            "predeclared_candidates": ["capability_heavy", "balanced", "retention_heavy"],
            "eligibility": {
                "max_broad_relative_regression": args.max_broad_relative_regression,
                "max_prefix_repetition_increase": args.max_prefix_repetition_increase,
                "max_loop_free_decrease": args.max_loop_free_decrease,
                "max_dead_anchor_ratio": args.max_dead_anchor_ratio,
            },
            "score": "quick_capability_loss_delta + 2*positive_repetition_delta - 0.10*loop_free_delta",
            "tie_break": "lowest_score",
        },
        "smoke_steps": args.smoke_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "baseline": baseline,
        "candidates": candidates,
        "selected_profile": selected["profile"] if selected else None,
        "selected_prompt_lm_weight": selected["profile_config"]["prompt_lm_weight"] if selected else None,
        "model_config": asdict(config),
    }
    atomic_write_json(args.output, payload)
    partial_path.unlink(missing_ok=True)
    print(json.dumps({
        "status": payload["status"],
        "selected_profile": payload["selected_profile"],
        "selected_prompt_lm_weight": payload["selected_prompt_lm_weight"],
        "candidates": [
            {
                "profile": item["profile"],
                "eligible": item["eligible"],
                "score": item["selection_score"],
                "broad_relative_delta": item["broad_relative_delta"],
                "prefix_repetition_delta": item["prefix_repetition_delta"],
                "prefix_loop_free_delta": item["prefix_loop_free_delta"],
                "quick_capability_loss_delta": item["quick_capability_loss_delta"],
            }
            for item in candidates
        ],
    }, indent=2), flush=True)
    raise SystemExit(0 if selected else 2)


if __name__ == "__main__":
    main()
