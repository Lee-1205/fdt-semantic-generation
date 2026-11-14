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
from train_capability_completion_v15 import append_jsonl, atomic_save, strict_smoke, verify_sha256
from train_chat_sft import evaluate_chat


STAGE = "capability_completion_v17_balanced"


def load_split(dataset: Path, split: str):
    payload = safe_torch_load(dataset / "shards" / split / "shard_000000.pt", "cpu")
    return payload


def balanced_order(category_ids: torch.Tensor, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    groups = []
    for category_index in range(len(CATEGORY_NAMES)):
        values = (category_ids == category_index).nonzero(as_tuple=False).flatten()
        groups.append(values.index_select(0, torch.randperm(len(values), generator=generator)))
    if len({len(group) for group in groups}) != 1:
        raise ValueError("V17 requires equal rows per category")
    order = []
    for offset in range(0, len(groups[0]), 2):
        for group in groups:
            order.extend(group[offset : offset + 2].tolist())
    return torch.tensor(order, dtype=torch.long)


def category_normalized_loss(logits, input_ids, labels, attention_mask, category_ids, eos_id: int):
    shifted_logits = logits[:, :-1].float()
    targets = labels[:, 1:]
    valid = targets.ne(-100)
    token_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(targets)
    weights = valid.float() * torch.where(targets.eq(eos_id), 2.0, 1.0)
    per_example = (token_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    category_losses = []
    for category_index in range(len(CATEGORY_NAMES)):
        selected = category_ids.eq(category_index)
        if not bool(selected.any()):
            raise ValueError("balanced V17 batch is missing a category")
        category_losses.append(per_example[selected].mean())
    completion = torch.stack(category_losses).mean()

    prompt_mask = targets.eq(-100) & attention_mask[:, 1:].bool()
    prompt_tokens = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view_as(targets)
    prompt_per_example = (prompt_tokens * prompt_mask).sum(dim=1) / prompt_mask.sum(dim=1).clamp_min(1)
    prompt = prompt_per_example.mean()
    return completion, prompt, category_losses


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Balanced category-normalized V17 capability pilot.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--broad-dataset-dir", type=Path, default=ROOT / "prepared_data" / "v3_broad_additional_300m_combined_v2_24k_ctx2048")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--prompt-lm-weight", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=0.7)
    parser.add_argument("--smoke-step", type=int, default=100)
    parser.add_argument("--max-broad-relative-regression", type=float, default=0.003)
    parser.add_argument("--max-prefix-repetition-increase", type=float, default=0.01)
    parser.add_argument("--max-loop-free-decrease", type=float, default=0.05)
    parser.add_argument("--max-capability-loss-regression", type=float, default=0.0)
    parser.add_argument("--max-dead-anchor-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    for path in (args.checkpoint, args.dataset_dir, args.output_dir, args.broad_dataset_dir):
        require_c_path(path)
    output = args.output_dir.resolve()
    launcher_files = {"active_logs.json", "train.pid"}
    if output.exists():
        unexpected = [p.name for p in output.iterdir() if p.name not in launcher_files and not re.fullmatch(r"train\.\d{8}_\d{6}\.(stdout|stderr)\.log", p.name)]
        if unexpected:
            raise FileExistsError(f"fresh V17 output contains files: {unexpected}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = verify_sha256(args.checkpoint, args.expected_checkpoint_sha256, "V16 parent")
    manifest_hash = verify_sha256(args.dataset_dir / "manifest.json", args.expected_manifest_sha256, "V17 manifest")
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("category_normalized_per_example_loss") or manifest.get("post_build_exact_overlap") != 0:
        raise ValueError("V17 manifest objective or overlap invariant failed")
    if float(manifest.get("eos_loss_weight", 0)) != 2.0 or float(manifest.get("prompt_lm_weight", 0)) != args.prompt_lm_weight:
        raise ValueError("V17 fixed loss weights mismatch")

    checkpoint = safe_torch_load(args.checkpoint, "cpu")
    config = ModelConfig(**checkpoint["model_config"])
    tokenizer_name = checkpoint.get("train_config", {}).get("tokenizer_name", "") or "gpt2"
    tokens_seen = int(checkpoint.get("tokens_seen", 0))
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    frozen = []
    for name, parameter in model.named_parameters():
        if ".anchor." in name:
            parameter.requires_grad_(False)
            frozen.append(name)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    tokenizer = strict.load_tokenizer(tokenizer_name)

    train = load_split(args.dataset_dir, "train")
    val = load_split(args.dataset_dir, "validation")
    ids = train["input_ids"]
    labels = train["labels"]
    masks = train["attention_mask"]
    categories = train["category_ids"]
    order = balanced_order(categories, args.seed)
    if args.batch_size != 10 or len(order) % args.batch_size:
        raise ValueError("V17 predeclared balanced batch size must be 10")
    broad_ids, broad_masks = strict.split_rows(args.broad_dataset_dir, "validation")
    broad_manifest = json.loads((args.broad_dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    broad_ranges = strict.bucket_ranges(broad_manifest, "validation")
    capability_payload = {"input_ids": val["input_ids"], "labels": val["labels"], "attention_mask": val["attention_mask"]}
    capability_ranges = strict_v3.capability_ranges(manifest)

    settings = {
        "stage": STAGE,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "parent_tokens": tokens_seen,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest_sha256": manifest_hash,
        "output_dir": str(output),
        "tokenizer_name": str(tokenizer_name),
        "learning_rate": args.lr,
        "lr_schedule": "constant",
        "lr_warmup_steps": 0,
        "batch_size": args.batch_size,
        "prompt_lm_weight": args.prompt_lm_weight,
        "eos_loss_weight": 2.0,
        "category_normalized_per_example_loss": True,
        "freeze_routing": True,
        "frozen_parameter_names": len(frozen),
        "conversational_sft": False,
        "seed": args.seed,
    }
    (output / "train_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output / "model_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    log_path = output / "training_log.jsonl"
    latest_path = output / "latest.pt"
    stop_path = output / "STOP_REQUESTED"
    baseline_chat = evaluate_chat(model, val["input_ids"], val["labels"], val["attention_mask"], args.batch_size, device)
    baseline_smoke = strict_smoke(model, tokenizer, config, capability_payload, capability_ranges, broad_ids, broad_masks, broad_ranges, device)
    append_jsonl(log_path, {"event": "baseline", "tokens_seen": tokens_seen, "capability": baseline_chat, **baseline_smoke})

    model.train()
    cursor = 0
    step = 0
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
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x, attention_mask=mask)["logits"]
            completion, prompt, category_losses = category_normalized_loss(logits, x, y, mask, cats, config.eos_token_id)
            loss = completion + args.prompt_lm_weight * prompt
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite V17 loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("nonfinite V17 gradient")
        optimizer.step()
        cursor += len(batch_indices)
        step += 1
        tokens_seen += int(mask.sum())
        if step % 25 == 0:
            append_jsonl(log_path, {
                "event": "train", "step": step, "sample_cursor": cursor, "tokens_seen": tokens_seen,
                "loss": float(loss), "completion_loss": float(completion), "prompt_loss": float(prompt),
                "category_losses": {name: float(value) for name, value in zip(CATEGORY_NAMES, category_losses)},
                "grad_norm": float(grad_norm),
                "tokens_per_sec": (tokens_seen - token_start) / max(time.perf_counter() - started, 1e-9),
            })
        if stop_path.exists():
            status = "PAUSED"
            break
        if step == args.smoke_step:
            model.eval()
            smoke = strict_smoke(model, tokenizer, config, capability_payload, capability_ranges, broad_ids, broad_masks, broad_ranges, device)
            smoke_chat = evaluate_chat(model, val["input_ids"], val["labels"], val["attention_mask"], args.batch_size, device)
            broad_relative = smoke["heldout"]["loss"] / baseline_smoke["heldout"]["loss"] - 1.0
            repetition_delta = smoke["prefix_generation"]["mean_repetition_rate"] - baseline_smoke["prefix_generation"]["mean_repetition_rate"]
            loop_free_delta = smoke["prefix_generation"]["loop_free"]["rate"] - baseline_smoke["prefix_generation"]["loop_free"]["rate"]
            dead = smoke["heldout"]["routing"].get("dead_anchor_ratio", 0.0)
            capability_delta = smoke_chat["loss"] - baseline_chat["loss"]
            healthy = (math.isfinite(smoke["heldout"]["loss"]) and broad_relative <= args.max_broad_relative_regression and repetition_delta <= args.max_prefix_repetition_increase and loop_free_delta >= -args.max_loop_free_decrease and capability_delta <= args.max_capability_loss_regression and dead <= args.max_dead_anchor_ratio)
            append_jsonl(log_path, {"event": "smoke_gate", "step": step, "healthy": healthy, "broad_relative_delta": broad_relative, "prefix_repetition_delta": repetition_delta, "prefix_loop_free_delta": loop_free_delta, "capability_loss_delta": capability_delta, "dead_anchor_ratio": dead, "capability": smoke_chat, **smoke})
            if not healthy:
                status = "ABORTED_SMOKE_GATE"
                break
            model.train()

    model.eval()
    final_chat = evaluate_chat(model, val["input_ids"], val["labels"], val["attention_mask"], args.batch_size, device)
    save_checkpoint(latest_path, model, optimizer, config, settings, step, cursor, tokens_seen, status)
    event = "complete" if status == "COMPLETE" else "paused" if status == "PAUSED" else "aborted"
    final = {"event": event, "status": status, "step": step, "sample_cursor": cursor, "tokens_seen": tokens_seen, "capability": final_chat}
    append_jsonl(log_path, final)
    print(json.dumps(final, indent=2), flush=True)
    raise SystemExit(0 if status in {"COMPLETE", "PAUSED"} else 2)


if __name__ == "__main__":
    main()
