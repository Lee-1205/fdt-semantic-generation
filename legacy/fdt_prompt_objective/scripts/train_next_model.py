from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.data.interleaving import bucket_sequence_counts, interleave_remaining_indices
from fdt_rlm.models import anchor_regularization_losses, build_model, causal_lm_loss
from fdt_rlm.next_tools import NEXT_MODEL_TYPES, load_next_config, make_model_config, safe_torch_load
from fdt_rlm.tokenization import load_tokenizer


PROTECTED_OUTPUTS = {
    (ROOT / "runs" / "staged_100m_interleaved" / model).resolve()
    for model in ("transformer", "fdt")
} | {
    (ROOT / "runs" / "staged_50m" / model).resolve()
    for model in ("transformer", "fdt")
}


def load_split(dataset_dir: Path, split: str):
    ids, masks = [], []
    for path in sorted((dataset_dir / "shards" / split).glob("*.pt")):
        shard = safe_torch_load(path)
        part = shard["input_ids"].to(dtype=torch.int32)
        ids.append(part)
        mask = shard.get("attention_mask")
        masks.append(
            torch.ones_like(part, dtype=torch.bool)
            if mask is None
            else mask.to(dtype=torch.bool)
        )
    if not ids:
        raise FileNotFoundError(f"No {split} shards in {dataset_dir}")
    return torch.cat(ids), torch.cat(masks)


def select_training_batch(
    train_ids,
    train_mask,
    sequence_index: int,
    batch_size: int,
    device,
    train_order=None,
):
    total_rows = train_ids.size(0) if train_order is None else train_order.numel()
    if sequence_index + batch_size > total_rows:
        sequence_index = 0
    end_index = sequence_index + batch_size
    if train_order is None:
        ids = train_ids[sequence_index:end_index]
        mask = train_mask[sequence_index:end_index]
    else:
        row_indices = train_order[sequence_index:end_index]
        ids = train_ids.index_select(0, row_indices)
        mask = train_mask.index_select(0, row_indices)
    return (
        ids.to(device=device, dtype=torch.long),
        mask.to(device=device, dtype=torch.long),
        end_index,
    )


def choose(value, configured):
    return configured if value is None else value


def is_local_mixer_parameter(name: str) -> bool:
    return ".local_mixer." in name or ".local_norm." in name


def set_local_mixer_only(model, enabled: bool) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_local_mixer_parameter(name) if enabled else True)


def save_checkpoint(path, model, optimizer, config, settings, step, tokens_seen, sequence_index):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(
            {
                "model_type": config.model_type,
                "model_config": asdict(config),
                "train_config": settings,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "optimizer_step": step,
                "global_step": step,
                "tokens_seen": tokens_seen,
                "sequence_index": sequence_index,
            },
            temporary,
        )
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def link_checkpoint(source: Path, destination: Path) -> None:
    """Create a same-volume numbered checkpoint without duplicating its bytes."""
    if destination.exists():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError as error:
        raise RuntimeError(
            f"Unable to hard-link checkpoint {source} to {destination}"
        ) from error


def prune_step_checkpoints(output_dir: Path, keep: int) -> list[str]:
    """Retain only the newest numbered checkpoints inside one run directory."""
    if keep <= 0:
        return []
    root = output_dir.resolve()
    numbered = []
    for path in output_dir.glob("step_*.pt"):
        match = re.fullmatch(r"step_(\d+)\.pt", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    removed = []
    for _, path in sorted(numbered)[:-keep]:
        resolved = path.resolve()
        if resolved.parent != root:
            raise RuntimeError(f"Refusing to prune checkpoint outside {root}: {resolved}")
        resolved.unlink()
        removed.append(path.name)
    return removed


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def resolve_stop_file(settings: dict, output_dir: Path) -> Path:
    configured = settings.get("stop_file")
    if not configured:
        return output_dir / "STOP_REQUESTED"
    path = Path(configured)
    return path if path.is_absolute() else output_dir / path


def load_resume_state(model, optimizer, checkpoint: dict, config, settings: dict) -> dict:
    """Load a checkpoint while strictly validating intentional migrations."""
    state = checkpoint["model_state_dict"]
    position_key = "position_embedding.weight"
    migrating_position = (
        model.position_embedding is not None
        and position_key not in state
        and bool(settings.get("allow_position_embedding_migration", False))
    )
    model_position = model.state_dict().get(position_key)
    checkpoint_position = state.get(position_key)
    resizing_position = (
        model_position is not None
        and checkpoint_position is not None
        and tuple(model_position.shape) != tuple(checkpoint_position.shape)
        and bool(settings.get("allow_position_embedding_resize", False))
    )
    if resizing_position:
        if model_position.ndim != 2 or checkpoint_position.ndim != 2:
            raise RuntimeError("Position embedding resize requires rank-2 tables")
        if model_position.size(1) != checkpoint_position.size(1):
            raise RuntimeError("Position embedding width cannot change during resize")
        resized_position = model_position.detach().clone()
        copied_rows = min(model_position.size(0), checkpoint_position.size(0))
        resized_position[:copied_rows].copy_(
            checkpoint_position[:copied_rows].to(resized_position)
        )
        resize_init = str(settings.get("position_resize_init", "random")).lower()
        if model_position.size(0) > copied_rows:
            tail_rows = model_position.size(0) - copied_rows
            if resize_init == "repeat_interior":
                source = checkpoint_position[1:-1]
                if source.size(0) == 0:
                    source = checkpoint_position
                indices = torch.arange(tail_rows, device=source.device) % source.size(0)
                resized_position[copied_rows:].copy_(source.index_select(0, indices).to(resized_position))
            elif resize_init == "mean":
                source = checkpoint_position[1:-1]
                if source.size(0) == 0:
                    source = checkpoint_position
                resized_position[copied_rows:].copy_(source.mean(dim=0, keepdim=True).to(resized_position))
            elif resize_init != "random":
                raise ValueError(
                    "position_resize_init must be 'random', 'repeat_interior', or 'mean'"
                )
        state = dict(state)
        state[position_key] = resized_position
    checkpoint_tied = bool(checkpoint.get("model_config", {}).get("tie_embeddings", True))
    migrating_untied_head = (
        checkpoint_tied
        and not config.tie_embeddings
        and bool(settings.get("reset_lm_head_on_untie", False))
    )
    checkpoint_local_mixer = bool(checkpoint.get("model_config", {}).get("use_local_mixer", False))
    migrating_local_mixer = (
        config.use_local_mixer
        and not checkpoint_local_mixer
        and bool(settings.get("allow_local_mixer_migration", False))
    )
    if migrating_position or resizing_position or migrating_local_mixer:
        incompatible = model.load_state_dict(state, strict=False)
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        expected_missing = {"position_embedding.weight"} if migrating_position else set()
        if migrating_local_mixer:
            expected_missing.update(
                key
                for key in model.state_dict()
                if ".local_norm." in key or ".local_mixer." in key
            )
        if missing != expected_missing or unexpected:
            raise RuntimeError(
                f"Unsafe checkpoint migration: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        if migrating_position:
            print(
                "Migrated checkpoint to learned positional embeddings "
                "(new table initialized with std=0.02).",
                flush=True,
            )
        if resizing_position:
            print(
                "Resized learned position embeddings from "
                f"{checkpoint_position.size(0)} to {model_position.size(0)} rows; "
                f"preserved the first {copied_rows} rows and initialized the tail with "
                f"{resize_init!r}.",
                flush=True,
            )
        if migrating_local_mixer:
            print(
                "Added zero-output causal local mixers; checkpoint behavior is preserved.",
                flush=True,
            )
    else:
        model.load_state_dict(state, strict=True)

    if migrating_untied_head:
        divisor = max(float(settings.get("lm_head_migration_divisor", 8.0)), 1.0)
        with torch.no_grad():
            model.lm_head.weight.copy_(state["lm_head.weight"].to(model.lm_head.weight) / divisor)
        print(
            f"Untied saturated LM head and divided its initial weights by {divisor:g}.",
            flush=True,
        )

    optimizer_loaded = False
    reset_optimizer = (
        (migrating_position or resizing_position)
        and bool(settings.get("reset_optimizer_on_migration", True))
    ) or migrating_untied_head or migrating_local_mixer
    if "optimizer_state_dict" in checkpoint and not reset_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        optimizer_loaded = True
    elif reset_optimizer:
        print("Reset optimizer for checkpoint architecture migration.", flush=True)

    if bool(settings.get("override_resume_lr", False)):
        configured_lr = float(settings.get("learning_rate", 3e-5))
        for group in optimizer.param_groups:
            group["lr"] = configured_lr
        print(f"Resume learning rate overridden to {configured_lr:g}.", flush=True)
    if optimizer.param_groups:
        optimizer.param_groups[0]["lr_scale"] = float(settings.get("base_lr_scale", 1.0))
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]["lr_scale"] = float(settings.get("local_mixer_lr_scale", 1.0))
    return {
        "position_embedding_migrated": migrating_position or resizing_position,
        "position_embedding_resized": resizing_position,
        "lm_head_untied": migrating_untied_head,
        "local_mixer_migrated": migrating_local_mixer,
        "optimizer_loaded": optimizer_loaded,
    }


def scheduled_learning_rate(settings: dict, step: int, start_step: int, final_step: int) -> float:
    peak_lr = float(settings.get("learning_rate", 3e-5))
    schedule = str(settings.get("lr_schedule", "constant")).lower()
    if schedule == "constant":
        return peak_lr
    if schedule != "cosine":
        raise ValueError(f"Unknown lr_schedule: {schedule}")

    schedule_start = int(settings.get("lr_schedule_start_step", start_step))
    local_step = max(step - schedule_start, 0)
    warmup_steps = max(int(settings.get("lr_warmup_steps", 0)), 0)
    warmup_start = float(settings.get("lr_warmup_start", peak_lr * 0.1))
    if warmup_steps > 0 and local_step <= warmup_steps:
        progress = local_step / warmup_steps
        return warmup_start + (peak_lr - warmup_start) * progress

    configured_decay = int(settings.get("lr_decay_steps", 0))
    decay_steps = configured_decay if configured_decay > 0 else max(final_step - schedule_start, 1)
    decay_progress = (local_step - warmup_steps) / max(decay_steps - warmup_steps, 1)
    decay_progress = min(max(decay_progress, 0.0), 1.0)
    min_ratio = min(max(float(settings.get("lr_min_ratio", 0.1)), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return peak_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def validate_resume_learning_rate(
    settings: dict,
    previous_lr: float,
    start_step: int,
    final_step: int,
) -> dict:
    """Reject accidental LR jumps when a resumed run changes its token target."""
    next_lr = scheduled_learning_rate(settings, start_step + 1, start_step, final_step)
    denominator = max(min(abs(previous_lr), abs(next_lr)), 1e-12)
    ratio = max(abs(previous_lr), abs(next_lr)) / denominator
    max_ratio = max(float(settings.get("lr_resume_max_ratio", 2.0)), 1.0)
    allowed = bool(settings.get("allow_lr_resume_jump", False))
    if ratio > max_ratio and not allowed:
        raise RuntimeError(
            "Refusing discontinuous resume learning rate: "
            f"checkpoint={previous_lr:.8g}, next={next_lr:.8g}, "
            f"ratio={ratio:.3f} > {max_ratio:.3f}. Set an explicit "
            "lr_decay_steps/stage schedule or allow_lr_resume_jump=true."
        )
    return {
        "event": "resume_lr_check",
        "step": start_step,
        "checkpoint_learning_rate": previous_lr,
        "next_learning_rate": next_lr,
        "ratio": ratio,
        "max_ratio": max_ratio,
        "explicit_decay_steps": int(settings.get("lr_decay_steps", 0)),
        "allowed": allowed,
    }


def anchor_health(anchor_stats, top_k: int) -> dict:
    if not anchor_stats:
        return {
            "entropy": 0.0,
            "entropy_normalized": 0.0,
            "effective_k": 0.0,
            "dead_anchor_ratio": 0.0,
            "top1_membership": 0.0,
        }
    entropy = torch.stack([stats.entropy.detach().float() for stats in anchor_stats]).mean()
    usage = torch.stack([stats.usage_prob.detach().float() for stats in anchor_stats]).mean(0)
    top1 = torch.stack([stats.top1_membership.detach().float() for stats in anchor_stats]).mean()
    return {
        "entropy": float(entropy),
        "entropy_normalized": float(entropy / max(math.log(max(top_k, 2)), 1e-8)),
        "effective_k": float(entropy.exp()),
        "dead_anchor_ratio": float((usage < 1e-4).float().mean()),
        "top1_membership": float(top1),
    }


def validation_batch_indices(
    total_rows: int,
    batch_size: int,
    eval_batches: int,
    sampling: str = "prefix",
) -> list[int]:
    available = math.ceil(total_rows / batch_size)
    count = min(eval_batches, available)
    if count <= 0:
        return []
    if sampling == "prefix":
        return list(range(count))
    if sampling != "evenly_spaced":
        raise ValueError(f"Unknown validation sampling mode: {sampling}")
    if count == 1:
        return [0]
    return [index * (available - 1) // (count - 1) for index in range(count)]


@torch.no_grad()
def evaluate_small(
    model,
    config,
    val_ids,
    val_mask,
    batch_size: int,
    eval_batches: int,
    device,
    sampling: str = "prefix",
):
    model.eval()
    for module in model.modules():
        if module.__class__.__name__.endswith("FuzzyAnchorLayer"):
            module.runtime_anchor_metrics = True
    loss_sum = 0.0
    token_count = 0
    correct_token_count = 0
    clamped_logit_count = 0
    logit_count = 0
    stats_rows = []
    batch_indices = validation_batch_indices(
        val_ids.size(0), batch_size, eval_batches, sampling=sampling
    )
    for batch_index in batch_indices:
        ids = val_ids[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ].to(device=device, dtype=torch.long)
        mask = val_mask[
            batch_index * batch_size : (batch_index + 1) * batch_size
        ].to(device=device, dtype=torch.long)
        output = model(ids, attention_mask=mask)
        loss = causal_lm_loss(output["logits"], ids, mask, config.pad_token_id)
        valid_tokens = int((mask[:, 1:].bool() & (ids[:, 1:] != config.pad_token_id)).sum())
        loss_sum += float(loss) * valid_tokens
        token_count += valid_tokens
        shift_logits = output["logits"][:, :-1].float()
        labels = ids[:, 1:]
        valid_mask = mask[:, 1:].bool() & (labels != config.pad_token_id)
        correct_token_count += int(((shift_logits.argmax(dim=-1) == labels) & valid_mask).sum())
        clamped_logit_count += int((shift_logits.abs() >= config.lm_logit_clip).sum())
        logit_count += shift_logits.numel()
        stats_rows.extend(output["anchor_stats"])
    return {
        "validation_loss": loss_sum / max(token_count, 1),
        "validation_tokens": token_count,
        "validation_sampling": sampling,
        "validation_batches": len(batch_indices),
        "next_token_accuracy": correct_token_count / max(token_count, 1),
        "logit_clamp_fraction": clamped_logit_count / max(logit_count, 1),
        **anchor_health(stats_rows, config.top_k),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a next-generation FDT research mode.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_type", choices=NEXT_MODEL_TYPES)
    parser.add_argument("--dataset_dir")
    parser.add_argument("--tokenizer_name")
    parser.add_argument("--output_dir")
    parser.add_argument("--max_steps", type=int)
    parser.add_argument("--target_tokens", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--grad_accum_steps", type=int)
    parser.add_argument("--dim", type=int)
    parser.add_argument("--n_layers", type=int)
    parser.add_argument("--n_heads", type=int)
    parser.add_argument("--max_seq_len", type=int)
    parser.add_argument("--num_anchors", type=int)
    parser.add_argument("--top_k", type=int)
    parser.add_argument("--anchor_layer_pattern")
    parser.add_argument("--anchor_layer_indices", type=int, nargs="*")
    parser.add_argument("--routing_backend", choices=["cdist", "matmul"])
    parser.add_argument("--resume_from", default="")
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--eval_every", type=int)
    parser.add_argument("--log_every", type=int)
    parser.add_argument("--live_log_path", default="")
    parser.add_argument("--auto_onk", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    settings = load_next_config(args.config)
    override_names = (
        "model_type", "dataset_dir", "tokenizer_name", "output_dir", "max_steps",
        "target_tokens", "batch_size", "grad_accum_steps", "dim", "n_layers",
        "n_heads", "max_seq_len", "num_anchors", "top_k", "anchor_layer_pattern",
        "anchor_layer_indices", "routing_backend", "save_every", "eval_every", "log_every",
    )
    for name in override_names:
        value = getattr(args, name)
        if value is not None:
            settings[name] = value

    output_dir = Path(settings.get("output_dir", ROOT / "runs" / "next_debug" / settings["model_type"]))
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    if output_dir.resolve() in PROTECTED_OUTPUTS:
        raise ValueError("Refusing to write into an official recovered checkpoint directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = Path(settings.get("dataset_dir", ROOT / "prepared_data" / "fdt_rlm_100m"))
    tokenizer_name = str(settings.get("tokenizer_name", ROOT / "tokenizers" / "fdt_bpe"))
    tokenizer = load_tokenizer(tokenizer_name)
    config = make_model_config(
        settings,
        len(tokenizer),
        int(tokenizer.pad_token_id),
        int(tokenizer.eos_token_id),
    )
    model = build_model(config)
    device = torch.device(settings.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    local_parameters = []
    base_parameters = []
    for name, parameter in model.named_parameters():
        target = local_parameters if is_local_mixer_parameter(name) else base_parameters
        target.append(parameter)
    parameter_groups = [
        {
            "params": base_parameters,
            "lr_scale": float(settings.get("base_lr_scale", 1.0)),
        }
    ]
    if local_parameters:
        parameter_groups.append(
            {
                "params": local_parameters,
                "lr_scale": float(settings.get("local_mixer_lr_scale", 1.0)),
            }
        )
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=float(settings.get("learning_rate", 3e-5)),
        weight_decay=float(settings.get("weight_decay", 0.1)),
        betas=(0.9, 0.95),
    )
    start_step = 0
    tokens_seen = 0
    sequence_index = 0
    resumed_dataset_changed = False
    migration_state = {}
    resume_path = Path(args.resume_from) if args.resume_from else None
    if resume_path:
        # Keep the serialized model/optimizer copy off CUDA while parameters
        # are restored. Large grown models otherwise occupy VRAM twice and can
        # fail before the first training batch despite fitting during training.
        checkpoint = safe_torch_load(resume_path, "cpu")
        migration_state = load_resume_state(model, optimizer, checkpoint, config, settings)
        start_step = int(checkpoint.get("optimizer_step", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        sequence_index = int(checkpoint.get("sequence_index", 0))
        previous_dataset = checkpoint.get("train_config", {}).get("dataset_dir", "")
        if previous_dataset and bool(settings.get("reset_data_cursor_on_dataset_change", True)):
            previous_path = Path(previous_dataset)
            if not previous_path.is_absolute():
                previous_path = ROOT / previous_path
            resumed_dataset_changed = previous_path.resolve() != dataset_dir.resolve()
            if resumed_dataset_changed:
                sequence_index = 0
                print(
                    f"Dataset changed from {previous_path} to {dataset_dir}; reset sequence cursor.",
                    flush=True,
                )
        del checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    local_mixer_only_steps = (
        max(int(settings.get("local_mixer_only_steps", 0)), 0)
        if migration_state.get("local_mixer_migrated", False)
        else 0
    )
    local_mixer_only_end = start_step + local_mixer_only_steps
    if local_mixer_only_steps > 0:
        set_local_mixer_only(model, True)
        print(
            f"Training only local mixers for {local_mixer_only_steps} steps, "
            f"through step {local_mixer_only_end}.",
            flush=True,
        )

    batch_size = int(settings.get("batch_size", 8))
    grad_accum = int(settings.get("grad_accum_steps", 1))
    train_order = None
    if args.dry_run:
        # A dry-run verifies the entire optimization/checkpoint path without
        # loading multi-gigabyte prepared shards or consuming dataset tokens.
        rows = max(batch_size * grad_accum * 2, batch_size)
        generator = torch.Generator().manual_seed(20260713)
        train_ids = torch.randint(0, config.vocab_size, (rows, config.max_seq_len), generator=generator)
        train_mask = torch.ones_like(train_ids)
        val_ids, val_mask = train_ids[:batch_size].clone(), train_mask[:batch_size].clone()
    else:
        train_ids, train_mask = load_split(dataset_dir, "train")
        if bool(settings.get("interleave_sources", True)):
            manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
            bucket_order = tuple(
                manifest.get("bucket_order") or manifest.get("plan", {}).keys()
            )
            order = interleave_remaining_indices(
                bucket_sequence_counts(manifest, bucket_order=bucket_order),
                0,
            )
            train_order = order
            if not resume_path or resumed_dataset_changed:
                sequence_index = 0
        val_ids, val_mask = load_split(dataset_dir, "validation")

    target_tokens = int(settings.get("target_tokens", 100_000_000))
    requested_steps = int(settings.get("max_steps", 0))
    early_stop_accuracy = float(settings.get("early_stop_next_token_accuracy", 0.0))
    early_stop_min_tokens = int(settings.get("early_stop_min_tokens", 0))
    tokens_per_step = batch_size * grad_accum * config.max_seq_len
    token_steps = math.ceil(max(target_tokens - tokens_seen, 0) / max(tokens_per_step, 1))
    steps_to_run = token_steps if requested_steps <= 0 else min(token_steps, requested_steps)
    if args.dry_run:
        steps_to_run = min(max(steps_to_run, 1), 2)
    final_step = start_step + steps_to_run
    resume_lr_event = None
    if resume_path and optimizer.param_groups and not bool(settings.get("override_resume_lr", False)):
        previous_lr = float(optimizer.param_groups[0]["lr"])
        previous_lr /= max(float(optimizer.param_groups[0].get("lr_scale", 1.0)), 1e-12)
        resume_lr_event = validate_resume_learning_rate(
            settings, previous_lr, start_step, final_step
        )
    last_completed_step = start_step
    early_stop_triggered = False
    external_stop_triggered = False
    settings.update(
        model_type=config.model_type,
        tokenizer_name=tokenizer_name,
        dataset_dir=str(dataset_dir),
        output_dir=str(output_dir),
        dry_run=bool(args.dry_run),
    )
    (output_dir / "model_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")

    precision = settings.get("precision", "bf16")
    amp_enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    save_every = int(settings.get("save_every", 2500))
    eval_every = int(settings.get("eval_every", 0))
    eval_batches = int(settings.get("eval_batches", 4))
    eval_sampling = str(settings.get("eval_sampling", "prefix"))
    log_every = max(int(settings.get("log_every", 1)), 1)
    loss_ema_beta = min(max(float(settings.get("loss_ema_beta", 0.98)), 0.0), 0.9999)
    grad_clip = float(settings.get("grad_clip", 1.0))
    auto_onk = bool(args.auto_onk or settings.get("auto_onk", False))
    live_log = Path(args.live_log_path) if args.live_log_path else output_dir / "training_log.jsonl"
    if not live_log.is_absolute():
        live_log = ROOT / live_log
    if resume_lr_event:
        append_jsonl(live_log, resume_lr_event)
        print("\n[RESUME LR] " + json.dumps(resume_lr_event), flush=True)
    stop_file = resolve_stop_file(settings, output_dir)
    onk_ready_path = output_dir / "onk_ready.pt"
    onk_runtime_path = output_dir / "onk_runtime.json"
    onk_promoted = onk_ready_path.exists()
    best_onk_validation = float("inf")
    if onk_runtime_path.exists():
        try:
            runtime_state = json.loads(onk_runtime_path.read_text(encoding="utf-8"))
            best_onk_validation = float(runtime_state.get("promotion_metrics", {}).get("validation_loss", float("inf")))
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    is_pure_onk = config.model_type in {"fdt", "fdt_optimized", "fdt_anchor_mixer", "fdt_v3"} and not config.use_self_attention
    complexity_path = "O(n(K+W))" if config.model_type == "fdt_v3" else "O(nK)"
    baseline = None
    if eval_every > 0 or auto_onk:
        baseline = evaluate_small(
            model,
            config,
            val_ids,
            val_mask,
            batch_size,
            eval_batches,
            device,
            sampling=eval_sampling,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        baseline_event = {
            "event": "baseline",
            "step": start_step,
            "tokens_seen": tokens_seen,
            "model_type": config.model_type,
            "complexity_path": complexity_path if is_pure_onk else "hybrid_or_attention",
            **baseline,
        }
        append_jsonl(live_log, baseline_event)
        print("\n[LIVE BASELINE] " + json.dumps(baseline_event, ensure_ascii=False), flush=True)
    model.train()
    loss_ema = None
    lm_ema = None
    progress = tqdm(range(start_step + 1, final_step + 1), desc=f"next:{config.model_type}")
    for step in progress:
        if local_mixer_only_steps > 0 and step == local_mixer_only_end + 1:
            set_local_mixer_only(model, False)
            transition_event = {
                "event": "local_mixer_warmup_complete",
                "step": step - 1,
                "tokens_seen": tokens_seen,
            }
            append_jsonl(live_log, transition_event)
            print("\n[LOCAL MIXER] Unfroze the full model.", flush=True)
        step_started = time.perf_counter()
        current_lr = scheduled_learning_rate(settings, step, start_step, final_step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr * float(group.get("lr_scale", 1.0))
        metrics_interval = int(settings.get("compute_anchor_metrics_every", config.compute_anchor_metrics_every))
        metrics_enabled = bool(settings.get("enable_anchor_metrics", config.enable_anchor_metrics)) and metrics_interval > 0 and step % metrics_interval == 0
        for module in model.modules():
            if module.__class__.__name__.endswith("FuzzyAnchorLayer"):
                module.runtime_anchor_metrics = metrics_enabled
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0
        last_metrics = {}
        for _ in range(grad_accum):
            ids, mask, sequence_index = select_training_batch(
                train_ids,
                train_mask,
                sequence_index,
                batch_size,
                device,
                train_order=train_order,
            )
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(ids, attention_mask=mask)
                lm = causal_lm_loss(output["logits"], ids, mask, config.pad_token_id)
                reg = anchor_regularization_losses(
                    model,
                    output["anchor_stats"],
                    entropy_target_ratio_start=float(settings.get("entropy_target_ratio_start", 0.65)),
                    entropy_target_ratio_end=float(settings.get("entropy_target_ratio_end", 0.45)),
                    compute_diversity=(step % max(int(settings.get("diversity_every", 4)), 1) == 0),
                )
                loss = (
                    lm
                    + float(settings.get("cluster_weight", 0.01)) * reg["cluster"]
                    + float(settings.get("diversity_weight", 0.1)) * reg["diversity"]
                    + float(settings.get("usage_weight", 0.005)) * reg["usage"]
                    - float(settings.get("entropy_weight", 0.003)) * reg["entropy_reward"]
                    + float(settings.get("entropy_over_weight", 0.2)) * reg["entropy_over"]
                ) / grad_accum
            loss.backward()
            accumulated += int(mask.sum().item())
            last_metrics = {
                "loss": float(loss.detach() * grad_accum),
                "lm": float(lm.detach()),
                "cluster": float(reg["cluster"].detach()),
                "diversity": float(reg["diversity"].detach()),
                "usage": float(reg["usage"].detach()),
                "entropy": float(reg["entropy"].detach()),
                "entropy_normalized": float(reg["entropy_normalized"].detach()),
                "effective_k": float(reg["effective_anchors"].detach()),
            }
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient in next-model training")
        optimizer.step()
        loss_ema = (
            last_metrics["loss"]
            if loss_ema is None
            else loss_ema_beta * loss_ema + (1.0 - loss_ema_beta) * last_metrics["loss"]
        )
        lm_ema = (
            last_metrics["lm"]
            if lm_ema is None
            else loss_ema_beta * lm_ema + (1.0 - loss_ema_beta) * last_metrics["lm"]
        )
        tokens_seen += accumulated
        last_completed_step = step
        elapsed = time.perf_counter() - step_started
        step_metrics = {
            "event": "train",
            "step": step,
            "tokens_seen": tokens_seen,
            **last_metrics,
            "loss_ema": loss_ema,
            "lm_ema": lm_ema,
            "grad_norm": float(grad_norm.detach()),
            "tokens_per_sec": accumulated / max(elapsed, 1e-9),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        progress.set_postfix({"loss": f"{last_metrics['loss']:.4f}", "lm": f"{last_metrics['lm']:.4f}", "ent": f"{last_metrics['entropy_normalized']:.3f}", "tok/s": f"{step_metrics['tokens_per_sec']:.0f}"})
        if step % log_every == 0:
            append_jsonl(live_log, step_metrics)

        if stop_file.is_file():
            external_stop_triggered = True
            break

        if save_every > 0 and step % save_every == 0:
            save_checkpoint(output_dir / "latest.pt", model, optimizer, config, settings, step, tokens_seen, sequence_index)
            link_checkpoint(output_dir / "latest.pt", output_dir / f"step_{step}.pt")
            removed = prune_step_checkpoints(
                output_dir,
                int(settings.get("step_checkpoint_retention", 0)),
            )
            if removed:
                append_jsonl(
                    live_log,
                    {"event": "checkpoint_pruned", "step": step, "removed": removed},
                )
        if eval_every > 0 and step % eval_every == 0:
            result = evaluate_small(
                model,
                config,
                val_ids,
                val_mask,
                batch_size,
                eval_batches,
                device,
                sampling=eval_sampling,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            baseline_loss = baseline["validation_loss"] if baseline else result["validation_loss"]
            absolute_improvement = baseline_loss - result["validation_loss"]
            relative_improvement = absolute_improvement / max(abs(baseline_loss), 1e-8)
            healthy_routing = (
                float(settings.get("onk_min_entropy_normalized", 0.25))
                <= result["entropy_normalized"]
                <= float(settings.get("onk_max_entropy_normalized", 0.85))
                and result["dead_anchor_ratio"] <= float(settings.get("onk_max_dead_anchor_ratio", 0.10))
            )
            meaningful = (
                is_pure_onk
                and math.isfinite(result["validation_loss"])
                and absolute_improvement >= float(settings.get("onk_min_absolute_improvement", 0.02))
                and relative_improvement >= float(settings.get("onk_min_relative_improvement", 0.0025))
                and healthy_routing
            )
            eval_event = {
                "event": "validation",
                "step": step,
                "tokens_seen": tokens_seen,
                **result,
                "absolute_improvement": absolute_improvement,
                "relative_improvement": relative_improvement,
                "healthy_routing": healthy_routing,
                "meaningful": meaningful,
                "onk_promoted": onk_promoted,
            }
            append_jsonl(live_log, eval_event)
            print("\n[LIVE VALIDATION] " + json.dumps(eval_event, ensure_ascii=False), flush=True)
            should_update_onk = auto_onk and meaningful and result["validation_loss"] < best_onk_validation
            if should_update_onk:
                promotion_label = "PROMOTED" if not onk_promoted else "UPDATED"
                config.use_incremental_anchor_state = True
                config.use_cached_normalized_anchors = True
                config.enable_diagnostics = False
                settings["onk_promoted_at_step"] = step
                settings["onk_promoted_at_tokens"] = tokens_seen
                settings["onk_best_validation_loss"] = result["validation_loss"]
                save_checkpoint(onk_ready_path, model, optimizer, config, settings, step, tokens_seen, sequence_index)
                onk_runtime_path.write_text(
                    json.dumps(
                        {
                            "status": "READY",
                            "checkpoint": str(onk_ready_path),
                            "training_complexity": complexity_path,
                            "inference_cache_backend": "fdt_v3_incremental" if config.model_type == "fdt_v3" else "anchor_incremental",
                            "use_cache": True,
                            "promotion_metrics": eval_event,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                onk_promoted = True
                best_onk_validation = result["validation_loss"]
                print(f"\n[O(nK) {promotion_label}] {onk_ready_path}", flush=True)
            reached_accuracy_transition = (
                early_stop_accuracy > 0.0
                and tokens_seen >= early_stop_min_tokens
                and result["next_token_accuracy"] >= early_stop_accuracy
                and healthy_routing
            )
            if reached_accuracy_transition:
                # The downstream supervisor consumes onk_ready.pt. Preserve the
                # exact threshold-crossing state even when loss promotion did not fire.
                if not should_update_onk:
                    config.use_incremental_anchor_state = True
                    config.use_cached_normalized_anchors = True
                    config.enable_diagnostics = False
                    settings["onk_promoted_at_step"] = step
                    settings["onk_promoted_at_tokens"] = tokens_seen
                    settings["onk_best_validation_loss"] = result["validation_loss"]
                    save_checkpoint(onk_ready_path, model, optimizer, config, settings, step, tokens_seen, sequence_index)
                    onk_runtime_path.write_text(
                        json.dumps(
                            {
                                "status": "READY",
                                "checkpoint": str(onk_ready_path),
                                "training_complexity": complexity_path,
                                "inference_cache_backend": "fdt_v3_incremental" if config.model_type == "fdt_v3" else "anchor_incremental",
                                "use_cache": True,
                                "promotion_metrics": eval_event,
                            },
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    onk_promoted = True
                early_stop_triggered = True
                append_jsonl(
                    live_log,
                    {
                        "event": "accuracy_transition_reached",
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "next_token_accuracy": result["next_token_accuracy"],
                        "threshold": early_stop_accuracy,
                    },
                )
                print(
                    f"\n[ACCURACY TRANSITION] {result['next_token_accuracy']:.6f} >= {early_stop_accuracy:.6f}",
                    flush=True,
                )
                break
            model.train()

    save_checkpoint(output_dir / "latest.pt", model, optimizer, config, settings, last_completed_step, tokens_seen, sequence_index)
    if external_stop_triggered:
        try:
            stop_file.unlink()
        except FileNotFoundError:
            pass
    final_event = {
        "event": "paused" if external_stop_triggered else "complete",
        "status": "PAUSED" if external_stop_triggered else "PASS",
        "dry_run": args.dry_run,
        "steps": last_completed_step - start_step,
        "tokens_seen": tokens_seen,
        "checkpoint": str(output_dir / "latest.pt"),
        "onk_promoted": onk_promoted,
        "early_stop_triggered": early_stop_triggered,
        "external_stop_triggered": external_stop_triggered,
    }
    append_jsonl(live_log, final_event)
    print(json.dumps(final_event, indent=2))


if __name__ == "__main__":
    main()
