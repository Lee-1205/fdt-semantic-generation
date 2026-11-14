from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model
from fdt_rlm.next_tools import safe_torch_load


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_chat_split(dataset_dir: Path, split: str):
    shards = sorted((dataset_dir / "shards" / split).glob("*.pt"))
    if not shards:
        raise FileNotFoundError(f"No chat SFT shards found for {split}")
    payloads = [safe_torch_load(path) for path in shards]
    return tuple(
        torch.cat([payload[key] for payload in payloads], dim=0)
        for key in ("input_ids", "labels", "attention_mask")
    )


def load_broad_validation(checkpoint: dict[str, Any], rows: int):
    dataset_dir = checkpoint.get("train_config", {}).get("dataset_dir")
    if not dataset_dir:
        raise ValueError("Base checkpoint does not identify its broad pretraining dataset")
    directory = (ROOT / dataset_dir / "shards" / "validation").resolve()
    shards = sorted(directory.glob("*.pt"))
    if not shards:
        raise FileNotFoundError(f"No broad validation shards found at {directory}")
    all_ids, all_masks = [], []
    remaining = rows
    for path in shards:
        payload = safe_torch_load(path)
        take = min(remaining, len(payload["input_ids"]))
        all_ids.append(payload["input_ids"][:take].long())
        all_masks.append(payload.get("attention_mask", torch.ones_like(payload["input_ids"]))[:take].long())
        remaining -= take
        if remaining <= 0:
            break
    return torch.cat(all_ids), torch.cat(all_masks)


def masked_chat_losses(logits, input_ids, labels, attention_mask):
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    assistant_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        shifted_labels.reshape(-1),
        ignore_index=-100,
    )
    prompt_mask = shifted_labels.eq(-100) & attention_mask[:, 1:].bool()
    token_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        input_ids[:, 1:].reshape(-1),
        reduction="none",
    ).view_as(prompt_mask)
    prompt_loss = token_loss[prompt_mask].mean() if prompt_mask.any() else assistant_loss.new_zeros(())
    return assistant_loss, prompt_loss


@torch.inference_mode()
def evaluate_chat(model, ids, labels, masks, batch_size: int, device: torch.device):
    model.eval()
    loss_sum = 0.0
    tokens = 0
    correct = 0
    for start in range(0, len(ids), batch_size):
        x = ids[start : start + batch_size].long().to(device)
        y = labels[start : start + batch_size].long().to(device)
        mask = masks[start : start + batch_size].long().to(device)
        logits = model(x, attention_mask=mask)["logits"][:, :-1].float()
        targets = y[:, 1:]
        valid = targets.ne(-100)
        losses = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none", ignore_index=-100
        ).view_as(targets)
        loss_sum += float(losses[valid].sum().item())
        tokens += int(valid.sum().item())
        correct += int((logits.argmax(dim=-1).eq(targets) & valid).sum().item())
    return {"loss": loss_sum / max(tokens, 1), "accuracy": correct / max(tokens, 1), "tokens": tokens}


@torch.inference_mode()
def evaluate_broad(model, config, ids, masks, batch_size: int, device: torch.device):
    model.eval()
    loss_sum = 0.0
    tokens = 0
    for start in range(0, len(ids), batch_size):
        x = ids[start : start + batch_size].to(device)
        mask = masks[start : start + batch_size].to(device)
        logits = model(x, attention_mask=mask)["logits"][:, :-1].float()
        targets = x[:, 1:]
        valid = mask[:, 1:].bool() & targets.ne(config.pad_token_id)
        losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        loss_sum += float(losses[valid].sum().item())
        tokens += int(valid.sum().item())
    return {"loss": loss_sum / max(tokens, 1), "tokens": tokens}


def save_checkpoint(path, model, optimizer, config, settings, base_checkpoint, step, cursor, tokens_seen):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(config),
            "train_config": settings,
            "base_checkpoint": str(base_checkpoint),
            "optimizer_step": step,
            "sample_cursor": cursor,
            "tokens_seen": tokens_seen,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Assistant-only chat SFT with broad-language retention gate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "prepared_data" / "chat_sft_20m")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "fdt_chat_sft")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--prompt-lm-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--broad-validation-rows", type=int, default=128)
    parser.add_argument("--max-broad-loss-increase", type=float, default=0.05)
    parser.add_argument("--min-chat-improvement", type=float, default=0.02)
    parser.add_argument("--freeze-routing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.checkpoint.resolve()
    checkpoint = safe_torch_load(base_path, device)
    config = ModelConfig(**checkpoint["model_config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if args.freeze_routing:
        for name, parameter in model.named_parameters():
            if ".anchor." in name:
                parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    train_ids, train_labels, train_masks = load_chat_split(args.dataset_dir.resolve(), "train")
    val_ids, val_labels, val_masks = load_chat_split(args.dataset_dir.resolve(), "validation")
    broad_ids, broad_masks = load_broad_validation(checkpoint, args.broad_validation_rows)
    settings = vars(args).copy()
    settings.update(
        checkpoint=str(base_path),
        dataset_dir=checkpoint.get("train_config", {}).get("dataset_dir"),
        tokenizer_name=checkpoint.get("train_config", {}).get("tokenizer_name"),
        chat_dataset_dir=str(args.dataset_dir.resolve()),
        output_dir=str(output_dir),
    )
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in settings.items()}
    log_path = output_dir / "training_log.jsonl"
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    step = 0
    cursor = 0
    tokens_seen = int(checkpoint.get("tokens_seen", 0))
    if args.resume and latest_path.exists():
        resumed = safe_torch_load(latest_path, device)
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        step = int(resumed.get("optimizer_step", 0))
        cursor = int(resumed.get("sample_cursor", 0))
        tokens_seen = int(resumed.get("tokens_seen", tokens_seen))

    orders = []
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        orders.append(torch.randperm(len(train_ids), generator=generator))
    order = torch.cat(orders) if orders else torch.empty(0, dtype=torch.long)
    total_samples = len(order)
    baseline_chat = evaluate_chat(model, val_ids, val_labels, val_masks, args.batch_size, device)
    baseline_broad = evaluate_broad(model, config, broad_ids, broad_masks, args.batch_size, device)
    best_chat_loss = float("inf")
    append_jsonl(
        log_path,
        {"event": "baseline", "chat": baseline_chat, "broad": baseline_broad, "tokens_seen": tokens_seen},
    )
    model.train()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    started = time.perf_counter()

    while cursor < total_samples:
        batch_indices = order[cursor : min(cursor + args.batch_size, total_samples)]
        x = train_ids.index_select(0, batch_indices).long().to(device, non_blocking=True)
        y = train_labels.index_select(0, batch_indices).long().to(device, non_blocking=True)
        mask = train_masks.index_select(0, batch_indices).long().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x, attention_mask=mask)["logits"]
            assistant_loss, prompt_loss = masked_chat_losses(logits, x, y, mask)
            loss = assistant_loss + args.prompt_lm_weight * prompt_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite chat SFT loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite chat SFT gradient")
        optimizer.step()
        cursor += len(batch_indices)
        step += 1
        batch_tokens = int(mask.sum().item())
        tokens_seen += batch_tokens
        if step % 25 == 0:
            elapsed = time.perf_counter() - started
            event = {
                "event": "train",
                "step": step,
                "sample_cursor": cursor,
                "tokens_seen": tokens_seen,
                "loss": float(loss.item()),
                "assistant_loss": float(assistant_loss.item()),
                "prompt_loss": float(prompt_loss.item()),
                "grad_norm": float(grad_norm.item()),
                "tokens_per_sec": (tokens_seen - int(checkpoint.get("tokens_seen", 0))) / max(elapsed, 1e-9),
            }
            append_jsonl(log_path, event)
            print(json.dumps(event), flush=True)
        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(latest_path, model, optimizer, config, settings, base_path, step, cursor, tokens_seen)
        if args.eval_every > 0 and step % args.eval_every == 0:
            chat = evaluate_chat(model, val_ids, val_labels, val_masks, args.batch_size, device)
            broad = evaluate_broad(model, config, broad_ids, broad_masks, args.batch_size, device)
            broad_healthy = broad["loss"] <= baseline_broad["loss"] * (1.0 + args.max_broad_loss_increase)
            chat_improvement = baseline_chat["loss"] - chat["loss"]
            promoted = (
                broad_healthy
                and chat_improvement >= args.min_chat_improvement
                and chat["loss"] < best_chat_loss
            )
            event = {
                "event": "validation",
                "step": step,
                "chat": chat,
                "broad": broad,
                "chat_improvement": chat_improvement,
                "broad_healthy": broad_healthy,
                "promoted": promoted,
            }
            append_jsonl(log_path, event)
            print(json.dumps(event), flush=True)
            if promoted:
                save_checkpoint(best_path, model, optimizer, config, settings, base_path, step, cursor, tokens_seen)
                best_chat_loss = chat["loss"]
            model.train()

    save_checkpoint(latest_path, model, optimizer, config, settings, base_path, step, cursor, tokens_seen)
    chat = evaluate_chat(model, val_ids, val_labels, val_masks, args.batch_size, device)
    broad = evaluate_broad(model, config, broad_ids, broad_masks, args.batch_size, device)
    broad_healthy = broad["loss"] <= baseline_broad["loss"] * (1.0 + args.max_broad_loss_increase)
    if broad_healthy and baseline_chat["loss"] - chat["loss"] >= args.min_chat_improvement and chat["loss"] < best_chat_loss:
        save_checkpoint(best_path, model, optimizer, config, settings, base_path, step, cursor, tokens_seen)
    status = "PASS" if best_path.exists() else "FAIL"
    final = {"event": "complete", "status": status, "step": step, "tokens_seen": tokens_seen, "chat": chat, "broad": broad}
    append_jsonl(log_path, final)
    print(json.dumps(final, indent=2))
    raise SystemExit(0 if status == "PASS" else 2)


if __name__ == "__main__":
    main()
