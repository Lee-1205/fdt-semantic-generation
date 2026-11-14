from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F


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
from prepare_capability_completion_v16 import CATEGORY_NAMES
from prepare_capability_completion_v19 import CATEGORY_WEIGHTS
from train_capability_completion_v15 import append_jsonl, atomic_save, strict_smoke, verify_sha256
from train_capability_completion_v17 import balanced_order
from train_chat_sft import evaluate_chat


STAGE = "capability_completion_v19_copy_grounding"


def load_split(dataset: Path, split: str):
    return safe_torch_load(dataset / "shards" / split / "shard_000000.pt", "cpu")


def validate_category_weights(weights: dict[str, float]) -> None:
    if set(weights) != set(CATEGORY_NAMES):
        raise ValueError("V19 category weights must cover every category exactly once")
    if any(not math.isfinite(value) or value <= 0 for value in weights.values()):
        raise ValueError("V19 category weights must be finite and positive")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("V19 category weights must sum to one")


def validate_resume_checkpoint(
    checkpoint: dict,
    dataset_dir: Path,
    manifest_hash: str,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    prompt_lm_weight: float,
    eos_loss_weight: float,
    grounding_loss_weight: float,
    signature_loss_weight: float,
    seed: int,
    order_size: int,
) -> tuple[int, int, int]:
    settings = checkpoint.get("train_config", {})
    if settings.get("stage") != STAGE or checkpoint.get("stage_status") != "PAUSED":
        raise ValueError("V19 resume requires a PAUSED V19 checkpoint")
    if checkpoint.get("optimizer_state_saved") is not True or not isinstance(checkpoint.get("optimizer_state_dict"), dict):
        raise ValueError("V19 resume checkpoint is missing optimizer state")
    if Path(settings.get("dataset_dir", "")).resolve() != dataset_dir.resolve():
        raise ValueError("V19 resume dataset identity mismatch")
    if str(settings.get("manifest_sha256", "")).upper() != manifest_hash.upper():
        raise ValueError("V19 resume manifest hash mismatch")
    expected = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "prompt_lm_weight": prompt_lm_weight,
        "eos_loss_weight": eos_loss_weight,
        "grounding_loss_weight": grounding_loss_weight,
        "signature_loss_weight": signature_loss_weight,
        "seed": seed,
    }
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"V19 resume {key} mismatch")
    if settings.get("lr_schedule") != "constant" or int(settings.get("lr_warmup_steps", -1)) != 0:
        raise ValueError("V19 resume LR continuity mismatch")
    if settings.get("freeze_routing") is not True:
        raise ValueError("V19 resume requires frozen routing")
    groups = checkpoint["optimizer_state_dict"].get("param_groups", [])
    if not groups or any(float(group.get("weight_decay", -1.0)) != weight_decay for group in groups):
        raise ValueError("V19 resume weight decay mismatch")
    step = int(checkpoint.get("optimizer_step", -1))
    cursor = int(checkpoint.get("sample_cursor", -1))
    tokens_seen = int(checkpoint.get("tokens_seen", -1))
    if step < 0 or cursor != step * batch_size or cursor >= order_size or tokens_seen < 0:
        raise ValueError("V19 resume cursor is inconsistent with deterministic order")
    return step, cursor, tokens_seen


def copy_grounding_loss(
    logits,
    input_ids,
    labels,
    attention_mask,
    category_ids,
    grounding_mask,
    signature_mask,
    eos_id: int,
    eos_loss_weight: float,
    grounding_loss_weight: float,
    signature_loss_weight: float,
    category_weights: dict[str, float],
):
    validate_category_weights(category_weights)
    shifted_logits = logits[:, :-1].float()
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    token_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(targets)
    weights = valid.float()
    weights = weights * torch.where(targets.eq(eos_id), eos_loss_weight, 1.0)
    weights = weights * torch.where(grounding_mask[:, 1:].bool(), grounding_loss_weight, 1.0)
    weights = weights * torch.where(signature_mask[:, 1:].bool(), signature_loss_weight, 1.0)
    per_example = (token_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    category_losses = []
    for category_index in range(len(CATEGORY_NAMES)):
        selected = category_ids.eq(category_index)
        if not bool(selected.any()):
            raise ValueError("balanced V19 batch is missing a category")
        category_losses.append(per_example[selected].mean())
    objective_weights = logits.new_tensor([category_weights[name] for name in CATEGORY_NAMES])
    completion = (torch.stack(category_losses) * objective_weights).sum()

    prompt_mask = targets.eq(-100) & attention_mask[:, 1:].bool()
    prompt_tokens = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view_as(targets)
    prompt_per_example = (prompt_tokens * prompt_mask).sum(dim=1) / prompt_mask.sum(dim=1).clamp_min(1)
    return completion, prompt_per_example.mean(), category_losses


@torch.inference_mode()
def factual_short_eos_metrics(model, ids, labels, masks, factual_range, batch_size, device, eos_id):
    start, end = factual_range
    loss_sum = {"short_non_eos": 0.0, "eos": 0.0}
    correct = {"short_non_eos": 0, "eos": 0}
    tokens = {"short_non_eos": 0, "eos": 0}
    model.eval()
    for offset in range(start, end, batch_size):
        x = ids[offset : min(offset + batch_size, end)].long().to(device)
        y = labels[offset : min(offset + batch_size, end)].long().to(device)
        mask = masks[offset : min(offset + batch_size, end)].long().to(device)
        logits = model(x, attention_mask=mask)["logits"][:, :-1].float()
        targets = y[:, 1:]
        valid = targets.ne(-100)
        losses = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none", ignore_index=-100).view_as(targets)
        predicted = logits.argmax(dim=-1)
        non_eos_counts = (valid & targets.ne(eos_id)).sum(dim=1)
        short = non_eos_counts.le(2).unsqueeze(1)
        selected = {
            "short_non_eos": valid & targets.ne(eos_id) & short,
            "eos": valid & targets.eq(eos_id),
        }
        for name, chosen in selected.items():
            loss_sum[name] += float(losses[chosen].sum())
            correct[name] += int((predicted.eq(targets) & chosen).sum())
            tokens[name] += int(chosen.sum())
    return {
        name: {
            "loss": loss_sum[name] / max(tokens[name], 1),
            "accuracy": correct[name] / max(tokens[name], 1),
            "correct": correct[name],
            "tokens": tokens[name],
        }
        for name in loss_sum
    }


def save_checkpoint(path, model, optimizer, config, settings, step, cursor, tokens_seen, status):
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "train_config": settings,
        "optimizer_step": step,
        "sample_cursor": cursor,
        "tokens_seen": tokens_seen,
        "stage_status": status,
        "optimizer_state_saved": status == "PAUSED",
    }
    if status == "PAUSED":
        payload["optimizer_state_dict"] = optimizer.state_dict()
    atomic_save(path, payload)


def model_from_checkpoint(path: Path, expected_hash: str, device: torch.device, label: str):
    verified = verify_sha256(path, expected_hash, label)
    checkpoint = safe_torch_load(path, "cpu")
    config = ModelConfig(**checkpoint["model_config"])
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    tokenizer_name = checkpoint.get("train_config", {}).get("tokenizer_name", "") or "gpt2"
    return model, config, tokenizer_name, checkpoint, verified


def main() -> None:
    parser = argparse.ArgumentParser(description="V19 copy-and-grounding capability pilot.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--v17-reference-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-v17-reference-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--broad-dataset-dir", type=Path, default=ROOT / "prepared_data" / "v3_broad_additional_300m_combined_v2_24k_ctx2048")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2.5e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--prompt-lm-weight", type=float, default=0.25)
    parser.add_argument("--eos-loss-weight", type=float, default=2.0)
    parser.add_argument("--grounding-loss-weight", type=float, default=2.0)
    parser.add_argument("--signature-loss-weight", type=float, default=3.0)
    parser.add_argument("--grad-clip", type=float, default=0.7)
    parser.add_argument("--smoke-step", type=int, default=100)
    parser.add_argument("--max-broad-relative-regression", type=float, default=0.003)
    parser.add_argument("--max-capability-loss-regression", type=float, default=0.0)
    parser.add_argument("--max-dead-anchor-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for path in (args.checkpoint, args.v17_reference_checkpoint, args.dataset_dir, args.output_dir, args.broad_dataset_dir):
        require_c_path(path)
    output = args.output_dir.resolve()
    launcher_files = {"active_logs.json", "train.pid"}
    if output.exists():
        unexpected = [path.name for path in output.iterdir() if path.name not in launcher_files and not re.fullmatch(r"train\.\d{8}_\d{6}\.(stdout|stderr)\.log", path.name)]
        if unexpected:
            raise FileExistsError(f"fresh V19 output contains files: {unexpected}")
    output.mkdir(parents=True, exist_ok=True)

    manifest_hash = verify_sha256(args.dataset_dir / "manifest.json", args.expected_manifest_sha256, "V19 manifest")
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_weights = {name: float(value) for name, value in manifest.get("category_objective_weights", {}).items()}
    validate_category_weights(manifest_weights)
    expected_manifest = {
        "eos_loss_weight": args.eos_loss_weight,
        "grounding_loss_weight": args.grounding_loss_weight,
        "signature_loss_weight": args.signature_loss_weight,
        "prompt_lm_weight": args.prompt_lm_weight,
    }
    if manifest_weights != CATEGORY_WEIGHTS or manifest.get("post_build_exact_overlap") != 0:
        raise ValueError("V19 manifest objective or overlap invariant failed")
    for key, value in expected_manifest.items():
        if float(manifest.get(key, 0)) != value:
            raise ValueError(f"V19 fixed {key} mismatch")

    device = torch.device(args.device)
    train = load_split(args.dataset_dir, "train")
    val = load_split(args.dataset_dir, "validation")
    required = {"input_ids", "labels", "attention_mask", "category_ids", "grounding_mask", "signature_mask"}
    if not required.issubset(train):
        raise ValueError(f"V19 train shard is missing fields: {sorted(required - set(train))}")
    ids, labels, masks = train["input_ids"], train["labels"], train["attention_mask"]
    categories = train["category_ids"]
    grounding_masks, signature_masks = train["grounding_mask"], train["signature_mask"]
    order = balanced_order(categories, args.seed)
    if args.batch_size != 10 or len(order) % args.batch_size:
        raise ValueError("V19 predeclared balanced batch size must be 10")
    broad_ids, broad_masks = strict.split_rows(args.broad_dataset_dir, "validation")
    broad_manifest = json.loads((args.broad_dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    broad_ranges = strict.bucket_ranges(broad_manifest, "validation")
    capability_payload = {"input_ids": val["input_ids"], "labels": val["labels"], "attention_mask": val["attention_mask"]}
    capability_ranges = strict_v3.capability_ranges(manifest)
    factual = next((start, end) for name, start, end in capability_ranges if name == "factual_qa")

    reference_model, reference_config, reference_tokenizer_name, reference_checkpoint, reference_hash = model_from_checkpoint(
        args.v17_reference_checkpoint, args.expected_v17_reference_sha256, device, "V17 hard reference"
    )
    tokenizer = strict.load_tokenizer(reference_tokenizer_name)
    with torch.inference_mode():
        reference_heldout = strict.heldout_metrics(reference_model, reference_config, broad_ids, broad_masks, broad_ranges, rows_per_bucket=4, device=device)["overall"]
        reference_factual = factual_short_eos_metrics(reference_model, val["input_ids"], val["labels"], val["attention_mask"], factual, args.batch_size, device, reference_config.eos_token_id)
    del reference_model, reference_checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model, config, tokenizer_name, checkpoint, checkpoint_hash = model_from_checkpoint(
        args.checkpoint, args.expected_checkpoint_sha256, device, "V18 parent" if not args.resume else "V19 resume"
    )
    if tokenizer_name != reference_tokenizer_name:
        raise ValueError("V17 and V18 tokenizer identities differ")
    tokens_seen = int(checkpoint.get("tokens_seen", 0))
    frozen = []
    for name, parameter in model.named_parameters():
        if ".anchor." in name:
            parameter.requires_grad_(False)
            frozen.append(name)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    resume_step = resume_cursor = 0
    if args.resume:
        resume_step, resume_cursor, tokens_seen = validate_resume_checkpoint(
            checkpoint, args.dataset_dir, manifest_hash, args.batch_size, args.lr, args.weight_decay,
            args.prompt_lm_weight, args.eos_loss_weight, args.grounding_loss_weight,
            args.signature_loss_weight, args.seed, len(order),
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
            group["weight_decay"] = args.weight_decay
    resume_settings = dict(checkpoint.get("train_config", {})) if args.resume else {}
    del checkpoint

    with torch.inference_mode():
        parent_chat = evaluate_chat(model, val["input_ids"], val["labels"], val["attention_mask"], args.batch_size, device)
        parent_smoke = strict_smoke(model, tokenizer, config, capability_payload, capability_ranges, broad_ids, broad_masks, broad_ranges, device)
        parent_factual = factual_short_eos_metrics(model, val["input_ids"], val["labels"], val["attention_mask"], factual, args.batch_size, device, config.eos_token_id)

    settings = {
        "stage": STAGE,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "v17_reference_checkpoint": str(args.v17_reference_checkpoint.resolve()),
        "v17_reference_sha256": reference_hash,
        "parent_tokens": tokens_seen,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest_sha256": manifest_hash,
        "output_dir": str(output),
        "tokenizer_name": str(tokenizer_name),
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "lr_schedule": "constant",
        "lr_warmup_steps": 0,
        "batch_size": args.batch_size,
        "prompt_lm_weight": args.prompt_lm_weight,
        "eos_loss_weight": args.eos_loss_weight,
        "grounding_loss_weight": args.grounding_loss_weight,
        "signature_loss_weight": args.signature_loss_weight,
        "category_objective_weights": manifest_weights,
        "category_and_example_normalized_loss": True,
        "freeze_routing": True,
        "frozen_parameter_names": len(frozen),
        "conversational_sft": False,
        "seed": args.seed,
    }
    if args.resume:
        settings = resume_settings
        settings.update({"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_hash, "output_dir": str(output), "resume_from_checkpoint": str(args.checkpoint.resolve()), "resume_from_sha256": checkpoint_hash, "resume_step": resume_step, "resume_cursor": resume_cursor})
    (output / "train_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output / "model_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    log_path = output / "training_log.jsonl"
    latest_path = output / "latest.pt"
    stop_path = output / "STOP_REQUESTED"
    append_jsonl(log_path, {
        "event": "resume" if args.resume else "baseline",
        "step": resume_step,
        "sample_cursor": resume_cursor,
        "tokens_seen": tokens_seen,
        "parent_capability": parent_chat,
        "parent_smoke": parent_smoke,
        "parent_factual_short_eos": parent_factual,
        "v17_reference_heldout": reference_heldout,
        "v17_reference_factual_short_eos": reference_factual,
    })

    model.train()
    cursor, step = resume_cursor, resume_step
    status = "COMPLETE"
    started = time.perf_counter()
    token_start = tokens_seen
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    while cursor < len(order):
        batch_indices = order[cursor : cursor + args.batch_size]
        x = ids.index_select(0, batch_indices).long().to(device)
        y = labels.index_select(0, batch_indices).long().to(device)
        mask = masks.index_select(0, batch_indices).long().to(device)
        cats = categories.index_select(0, batch_indices).long().to(device)
        grounded = grounding_masks.index_select(0, batch_indices).long().to(device)
        signatures = signature_masks.index_select(0, batch_indices).long().to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x, attention_mask=mask)["logits"]
            completion, prompt, category_losses = copy_grounding_loss(
                logits, x, y, mask, cats, grounded, signatures, config.eos_token_id,
                args.eos_loss_weight, args.grounding_loss_weight, args.signature_loss_weight,
                manifest_weights,
            )
            loss = completion + args.prompt_lm_weight * prompt
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite V19 loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("nonfinite V19 gradient")
        optimizer.step()
        cursor += len(batch_indices)
        step += 1
        tokens_seen += int(mask.sum())
        if step % 25 == 0:
            append_jsonl(log_path, {"event": "train", "step": step, "sample_cursor": cursor, "tokens_seen": tokens_seen, "loss": float(loss), "completion_loss": float(completion), "prompt_loss": float(prompt), "category_losses": {name: float(value) for name, value in zip(CATEGORY_NAMES, category_losses)}, "grad_norm": float(grad_norm), "tokens_per_sec": (tokens_seen - token_start) / max(time.perf_counter() - started, 1e-9)})
        if stop_path.exists():
            status = "PAUSED"
            break
        if step == args.smoke_step:
            model.eval()
            with torch.inference_mode():
                smoke = strict_smoke(model, tokenizer, config, capability_payload, capability_ranges, broad_ids, broad_masks, broad_ranges, device)
                smoke_chat = evaluate_chat(model, val["input_ids"], val["labels"], val["attention_mask"], args.batch_size, device)
                factual_metrics = factual_short_eos_metrics(model, val["input_ids"], val["labels"], val["attention_mask"], factual, args.batch_size, device, config.eos_token_id)
            broad_limit = min(parent_smoke["heldout"]["loss"], reference_heldout["loss"]) * (1.0 + args.max_broad_relative_regression)
            python_floor = parent_smoke["capability_generation"]["python_code"]["usable_structure"]["successes"]
            retrieval_floor = parent_smoke["capability_generation"]["retrieval"]["correct"]["successes"]
            json_floor = parent_smoke["capability_generation"]["json"]["valid"]["successes"]
            short_floor = min(parent_factual["short_non_eos"]["accuracy"], reference_factual["short_non_eos"]["accuracy"])
            eos_floor = min(parent_factual["eos"]["accuracy"], reference_factual["eos"]["accuracy"])
            dead = smoke["heldout"]["routing"].get("dead_anchor_ratio", 0.0)
            capability_delta = smoke_chat["loss"] - parent_chat["loss"]
            healthy = (
                math.isfinite(smoke["heldout"]["loss"])
                and smoke["heldout"]["loss"] <= broad_limit
                and capability_delta <= args.max_capability_loss_regression
                and dead <= args.max_dead_anchor_ratio
                and smoke["capability_generation"]["python_code"]["usable_structure"]["successes"] >= python_floor
                and smoke["capability_generation"]["retrieval"]["correct"]["successes"] >= retrieval_floor
                and smoke["capability_generation"]["json"]["valid"]["successes"] >= json_floor
                and factual_metrics["short_non_eos"]["accuracy"] >= short_floor
                and factual_metrics["eos"]["accuracy"] >= eos_floor
            )
            append_jsonl(log_path, {"event": "smoke_gate", "step": step, "healthy": healthy, "broad_loss_limit_vs_v17_v18": broad_limit, "capability_loss_delta_vs_v18": capability_delta, "dead_anchor_ratio": dead, "python_structural_floor": python_floor, "retrieval_exact_floor": retrieval_floor, "json_valid_floor": json_floor, "short_fact_accuracy_floor": short_floor, "eos_accuracy_floor": eos_floor, "factual_short_eos": factual_metrics, "capability": smoke_chat, **smoke})
            if not healthy:
                status = "ABORTED_SMOKE_GATE"
                break
            model.train()

    model.eval()
    with torch.inference_mode():
        final_chat = evaluate_chat(model, val["input_ids"], val["labels"], val["attention_mask"], args.batch_size, device)
    save_checkpoint(latest_path, model, optimizer, config, settings, step, cursor, tokens_seen, status)
    event = "complete" if status == "COMPLETE" else "paused" if status == "PAUSED" else "aborted"
    final = {"event": event, "status": status, "step": step, "sample_cursor": cursor, "tokens_seen": tokens_seen, "capability": final_chat}
    append_jsonl(log_path, final)
    print(json.dumps(final, indent=2), flush=True)
    raise SystemExit(0 if status in {"COMPLETE", "PAUSED"} else 2)


if __name__ == "__main__":
    main()
