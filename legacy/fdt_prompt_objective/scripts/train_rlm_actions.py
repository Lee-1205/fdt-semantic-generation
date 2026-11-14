from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model
from fdt_rlm.runtime.actions import Action, ActionName
from fdt_rlm.runtime.budget import BudgetConfig, BudgetTracker
from fdt_rlm.runtime.code_environment import CodeEnvironment
from fdt_rlm.runtime.recursive_controller import build_action_prompt
from fdt_rlm.runtime.text_environment import TextEnvironment
from fdt_rlm.tokenization import load_tokenizer


BASE_CHECKPOINTS = {
    "transformer": ROOT / "runs" / "full_transformer" / "best.pt",
    "fdt": ROOT / "runs" / "full_fdt" / "best.pt",
}


def safe_load(path: Path, device: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_tasks(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def balanced_task_order(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_family[str(task.get("family", "unknown"))].append(task)
    families = sorted(by_family)
    ordered: list[dict[str, Any]] = []
    for index in range(max((len(rows) for rows in by_family.values()), default=0)):
        for family in families:
            if index < len(by_family[family]):
                ordered.append(by_family[family][index])
    return ordered


def trajectory_examples(tasks: list[dict[str, Any]], max_examples: int) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    for task in balanced_task_order(tasks):
        environment = CodeEnvironment(task["files"]) if task.get("files") else TextEnvironment(task["context"])
        budget = BudgetTracker(BudgetConfig())
        history: list[dict[str, str]] = []
        for action_dict in task["gold_trajectory"]:
            prompt = build_action_prompt(task["question"], history, budget, 0, environment)
            target = json.dumps(action_dict, ensure_ascii=False, separators=(",", ":"))
            examples.append((prompt, target))
            if len(examples) >= max_examples:
                return examples[:max_examples]
            budget.consume_step()
            action = Action.from_mapping(action_dict)
            if action.name == ActionName.STOP:
                break
            if action.name == ActionName.CALL:
                sub_context = environment.resolve_ref(action.arguments["context_ref"])
                sub_environment = TextEnvironment(sub_context)
                sub_budget = BudgetTracker(BudgetConfig())
                sub_prompt = build_action_prompt(action.arguments["question"], [], sub_budget, 1, sub_environment)
                sub_target = json.dumps(
                    {"action": "STOP", "answer": task["gold_answer"]},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                examples.append((sub_prompt, sub_target))
                result_text = task["gold_answer"]
                history.append({"action": json.dumps(action.to_dict()), "observation": result_text})
                if len(examples) >= max_examples:
                    return examples[:max_examples]
                continue
            result = environment.execute(action)
            budget.consume_read(result.tokens_read)
            history.append(
                {
                    "action": json.dumps(action.to_dict(), ensure_ascii=False),
                    "observation": result.observation,
                }
            )
            if len(examples) >= max_examples:
                return examples
    return examples[:max_examples]


class ActionDataset(Dataset):
    def __init__(self, rows: list[tuple[str, str]], tokenizer, max_length: int, eos_id: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eos_id = eos_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        prompt, target = self.rows[index]
        prompt_ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        target_ids = list(self.tokenizer.encode(target, add_special_tokens=False)) + [self.eos_id]
        prompt_ids = prompt_ids[-max(self.max_length - len(target_ids), 1) :]
        ids = (prompt_ids + target_ids)[-self.max_length :]
        target_start = max(len(ids) - len(target_ids), 0)
        labels = [-100] * target_start + ids[target_start:]
        return ids, labels, len(target_ids)


def collate(rows, pad_id: int):
    length = max(len(row[0]) for row in rows)
    ids, labels, target_tokens = [], [], 0
    for row_ids, row_labels, row_targets in rows:
        padding = length - len(row_ids)
        ids.append(row_ids + [pad_id] * padding)
        labels.append(row_labels + [-100] * padding)
        target_tokens += row_targets
    input_ids = torch.tensor(ids, dtype=torch.long)
    return input_ids, torch.tensor(labels, dtype=torch.long), (input_ids != pad_id).long(), target_tokens


@torch.inference_mode()
def validation_loss(
    model,
    config: ModelConfig,
    device: torch.device,
    shard_path: Path,
    rows: int = 8,
) -> float:
    shard = safe_load(shard_path)
    ids = shard["input_ids"][:rows].long()
    masks = shard.get("attention_mask", torch.ones_like(ids))[:rows].long()
    loss_sum = 0.0
    token_count = 0
    for row_ids, row_mask in zip(ids, masks):
        x = row_ids.unsqueeze(0).to(device)
        mask = row_mask.unsqueeze(0).to(device)
        logits = model(input_ids=x, attention_mask=mask)["logits"][:, :-1].float()
        targets = x[:, 1:]
        valid = mask[:, 1:].bool() & (targets != config.pad_token_id)
        losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        loss_sum += float(losses[valid].sum().item())
        token_count += int(valid.sum().item())
    return loss_sum / max(token_count, 1)


def infer_validation_shard(checkpoint: dict[str, Any], explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    dataset_dir = checkpoint.get("train_config", {}).get("dataset_dir")
    if dataset_dir:
        candidate = (ROOT / dataset_dir / "shards" / "validation" / "shard_000000.pt").resolve()
        if candidate.exists():
            return candidate
    return ROOT / "prepared_data" / "fdt_rlm_10m" / "shards" / "validation" / "shard_000000.pt"


def train_one(
    name: str,
    base_path: Path,
    args,
    tokenizer,
    examples: list[tuple[str, str]],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = safe_load(base_path, device)
    config = ModelConfig(**checkpoint["model_config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if args.freeze_routing and name == "fdt":
        for parameter_name, parameter in model.named_parameters():
            if ".anchor." in parameter_name:
                parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    dataset = ActionDataset(examples, tokenizer, config.max_seq_len, config.eos_token_id)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda rows: collate(rows, config.pad_token_id),
    )
    validation_shard = infer_validation_shard(checkpoint, args.validation_shard)
    before = validation_loss(model, config, device, validation_shard, rows=args.validation_rows)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    action_losses: list[float] = []
    prompt_losses: list[float] = []
    target_tokens = 0
    started = time.perf_counter()
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    for epoch in range(args.epochs):
        for batch_index, (input_ids, labels, attention_mask, batch_targets) in enumerate(loader, start=1):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)["logits"][:, :-1]
                action_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )
                prompt_mask = (labels[:, 1:] == -100) & attention_mask[:, 1:].bool()
                prompt_token_loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1),
                    reduction="none",
                ).view_as(prompt_mask)
                prompt_loss = prompt_token_loss[prompt_mask].mean() if prompt_mask.any() else action_loss.new_zeros(())
                loss = action_loss + args.prompt_lm_weight * prompt_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite action loss for {name}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.item()))
            action_losses.append(float(action_loss.item()))
            prompt_losses.append(float(prompt_loss.item()))
            target_tokens += int(batch_targets)
            if batch_index % 25 == 0:
                print(
                    f"{name} epoch={epoch + 1} batch={batch_index}/{len(loader)} "
                    f"loss={sum(losses[-25:]) / 25:.4f} "
                    f"action={sum(action_losses[-25:]) / 25:.4f} "
                    f"prompt={sum(prompt_losses[-25:]) / 25:.4f}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    model.eval()
    after = validation_loss(model, config, device, validation_shard, rows=args.validation_rows)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else ROOT / "runs" / f"rlm_action_{name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": checkpoint["model_config"],
        "base_checkpoint": str(base_path.relative_to(ROOT)),
        "action_training": {
            "examples": len(examples),
            "epochs": args.epochs,
            "target_tokens_seen": target_tokens,
            "learning_rate": args.lr,
            "freeze_routing": bool(args.freeze_routing and name == "fdt"),
            "mean_loss": sum(losses) / max(len(losses), 1),
            "mean_action_loss": sum(action_losses) / max(len(action_losses), 1),
            "mean_prompt_loss": sum(prompt_losses) / max(len(prompt_losses), 1),
            "prompt_lm_weight": args.prompt_lm_weight,
            "validation_shard": str(validation_shard),
            "validation_rows": args.validation_rows,
            "validation_loss_before": before,
            "validation_loss_after": after,
        },
    }
    latest = output_dir / "latest.pt"
    best = output_dir / "best.pt"
    torch.save(payload, latest)
    shutil.copy2(latest, best)
    return {
        "model": name,
        **payload["action_training"],
        "elapsed_seconds": elapsed,
        "checkpoint": str(best.relative_to(ROOT)),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal output-only RLM action imitation")
    parser.add_argument("--models", nargs="+", choices=sorted(BASE_CHECKPOINTS), default=["transformer", "fdt"])
    parser.add_argument("--checkpoint", type=Path, help="Train one current FDT/ONK checkpoint instead of a legacy model pair")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--validation-shard", type=Path)
    parser.add_argument("--validation-rows", type=int, default=32)
    parser.add_argument("--tasks", type=Path, default=ROOT / "artifacts" / "rlm_tasks_500.jsonl")
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizers" / "fdt_bpe"))
    parser.add_argument("--max-examples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--prompt-lm-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-routing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Count examples/tokens without loading or training a model")
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = load_tokenizer(args.tokenizer)
    examples = trajectory_examples(load_tasks(args.tasks), args.max_examples)
    if len(examples) < args.max_examples:
        raise RuntimeError(f"only {len(examples)} action examples were available")
    if args.dry_run:
        dataset = ActionDataset(examples, tokenizer, 512, int(tokenizer.eos_token_id))
        sequence_tokens = sum(len(dataset[index][0]) for index in range(len(dataset)))
        target_tokens = sum(dataset[index][2] for index in range(len(dataset)))
        print(json.dumps({"examples": len(dataset), "sequence_tokens_per_epoch": sequence_tokens, "target_tokens_per_epoch": target_tokens}, indent=2))
        return
    device = torch.device(args.device)
    if args.checkpoint is not None:
        jobs = [("fdt", args.checkpoint.resolve())]
    else:
        jobs = [(name, BASE_CHECKPOINTS[name]) for name in args.models]
    if args.output_dir is not None and len(jobs) != 1:
        raise ValueError("--output-dir requires exactly one model/checkpoint")
    results = [train_one(name, path, args, tokenizer, examples, device) for name, path in jobs]
    report = {
        "policy": "minimum output-only action imitation; base checkpoints preserved",
        "action_examples": len(examples),
        "results": results,
        "total_target_tokens_seen": sum(item["target_tokens_seen"] for item in results),
    }
    path = ROOT / "artifacts" / "rlm_action_training.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
