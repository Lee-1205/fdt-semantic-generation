from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from fdt_rlm.config import ModelConfig, TrainConfig, dataclass_to_dict
from fdt_rlm.models import anchor_regularization_losses, build_model
from fdt_rlm.tokenization import load_tokenizer


GENERATION_PROMPTS = [
    "The primary reason why neural networks require optimization is",
    "In a small village surrounded by mountains,",
    "A good experiment should control",
    "def calculate_average(values):\n",
    "class DatasetLoader:\n    def __init__(self, path):\n",
    '{"action":"SEARCH","query":',
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(path: Path) -> str:
    h = hashlib.sha256()
    for file in sorted(p for p in path.glob("*") if p.is_file()):
        if file.name == "tokenizer_validation.json":
            continue
        h.update(file.read_bytes())
    return h.hexdigest()


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def load_manifest(dataset_dir: Path) -> Dict[str, Any]:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_split_tensors(dataset_dir: Path, split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    shard_dir = dataset_dir / "shards" / split
    paths = sorted(shard_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No shards found for split={split}: {shard_dir}")
    ids_parts: List[torch.Tensor] = []
    mask_parts: List[torch.Tensor] = []
    for path in paths:
        data = torch.load(path, map_location="cpu")
        ids = data["input_ids"].long()
        mask = data.get("attention_mask", torch.ones_like(ids)).long()
        ids_parts.append(ids)
        mask_parts.append(mask)
    return torch.cat(ids_parts, dim=0), torch.cat(mask_parts, dim=0)


def split_stats_from_shards(dataset_dir: Path) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "validation", "test"):
        shard_dir = dataset_dir / "shards" / split
        sequences = 0
        tokens = 0
        max_len = 0
        shard_count = 0
        for path in sorted(shard_dir.glob("*.pt")):
            data = torch.load(path, map_location="cpu")
            ids = data["input_ids"]
            mask = data.get("attention_mask", torch.ones_like(ids))
            shard_count += 1
            sequences += int(ids.shape[0])
            max_len = max(max_len, int(ids.shape[1]))
            tokens += int(mask.long().sum().item())
        if sequences:
            stats[split] = {
                "sequences": sequences,
                "tokens": tokens,
                "effective_sequence_length": tokens / sequences,
                "max_sequence_length": max_len,
                "shards": shard_count,
            }
    return stats


def calculate_budget(
    dataset_dir: Path,
    tokenizer_dir: Path,
    batch_size: int,
    grad_accum_steps: int,
    max_seq_len: int,
) -> Dict[str, Any]:
    manifest = load_manifest(dataset_dir)
    split_stats = split_stats_from_shards(dataset_dir)
    train = split_stats["train"]
    train_tokens = int(train["tokens"])
    train_sequences = int(train["sequences"])
    tokens_per_micro = float(train_tokens / max(train_sequences, 1)) * batch_size
    nominal_tokens_per_optimizer_step = batch_size * grad_accum_steps * max_seq_len
    estimated_effective_tokens_per_optimizer_step = tokens_per_micro * grad_accum_steps
    optimizer_steps_for_one_full_pass = math.ceil(train_sequences / (batch_size * grad_accum_steps))
    final_partial_sequences = train_sequences % (batch_size * grad_accum_steps)
    if final_partial_sequences == 0:
        final_partial_sequences = batch_size * grad_accum_steps
    final_partial_microbatches = math.ceil(final_partial_sequences / batch_size)
    source_tokens = int(sum(int(v) for v in manifest.get("tokens_by_bucket", {}).values()))
    packed_tokens = int(manifest.get("packed_tokens", train_tokens))
    discarded_tail = int(manifest.get("discarded_tail_tokens", 0))
    budget = {
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(dataset_dir / "manifest.json"),
        "dataset_manifest_sha256": sha256_file(dataset_dir / "manifest.json"),
        "tokenizer": str(tokenizer_dir),
        "tokenizer_hash": sha256_tree(tokenizer_dir) if tokenizer_dir.exists() else "",
        "split_stats": split_stats,
        "source_tokens": source_tokens,
        "packed_tokens": packed_tokens,
        "discarded_tail_tokens": discarded_tail,
        "packing_efficiency_vs_source": packed_tokens / max(source_tokens, 1),
        "packing_efficiency_vs_source_plus_tail": packed_tokens / max(packed_tokens + discarded_tail, 1),
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum_steps,
        "max_seq_len": max_seq_len,
        "actual_train_tokens": train_tokens,
        "train_sequences": train_sequences,
        "effective_tokens_per_microbatch_mean": tokens_per_micro,
        "nominal_tokens_per_optimizer_step": nominal_tokens_per_optimizer_step,
        "estimated_effective_tokens_per_optimizer_step": estimated_effective_tokens_per_optimizer_step,
        "mean_tokens_per_optimizer_step_over_one_pass": train_tokens / max(optimizer_steps_for_one_full_pass, 1),
        "micro_steps_for_one_full_pass": math.ceil(train_sequences / batch_size),
        "optimizer_steps_for_one_full_pass": optimizer_steps_for_one_full_pass,
        "final_partial_sequences": final_partial_sequences,
        "final_partial_microbatches": final_partial_microbatches,
        "one_pass_policy": "consume every training sequence exactly once; perform a final partial accumulated optimizer step if needed",
    }
    return budget


def make_model_config(
    tokenizer: Any,
    model_type: str,
    dim: int,
    n_layers: int,
    n_heads: int,
    mlp_ratio: int,
    max_seq_len: int,
    dropout: float,
    num_anchors: int,
    top_k: int,
    routing_type: str,
    cosine_temperature: float,
    aggregation_impl: str,
    diversity_margin: float,
    diversity_max_weight: float,
) -> ModelConfig:
    return ModelConfig(
        vocab_size=len(tokenizer),
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        model_type=model_type,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        max_seq_len=max_seq_len,
        dropout=dropout,
        num_anchors=num_anchors,
        top_k=top_k,
        routing_type=routing_type,
        cosine_temperature=cosine_temperature,
        aggregation_impl=aggregation_impl,
        diversity_margin=diversity_margin,
        diversity_max_weight=diversity_max_weight,
    )


def make_train_config(args: argparse.Namespace, output_dir: Path, total_steps: int) -> TrainConfig:
    return TrainConfig(
        tokenizer_name=str(args.tokenizer),
        dataset_name=str(args.dataset_dir),
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=total_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        precision=args.precision,
        lm_weight=args.lm_weight,
        noise_weight=args.noise_weight,
        cluster_weight=args.cluster_weight,
        diversity_weight=args.diversity_weight,
        diversity_every=max(args.diversity_every, 1),
        usage_weight=args.usage_weight,
        entropy_weight=args.entropy_weight,
        entropy_target_ratio_start=args.entropy_target_ratio_start,
        entropy_target_ratio_end=args.entropy_target_ratio_end,
        entropy_over_weight=args.entropy_over_weight,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    mode: str = "cosine",
) -> LambdaLR:
    if mode == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)
    if mode != "cosine":
        raise ValueError(f"unknown scheduler mode: {mode}")
    warmup_steps = max(1, min(warmup_steps, max(total_steps, 1)))
    total_steps = max(total_steps, 1)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(1e-8, float(step + 1) / float(warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return 0.1 + 0.9 * cosine

    return LambdaLR(optimizer, lr_lambda)


def causal_lm_loss_sum(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_token_id: int,
) -> Tuple[torch.Tensor, int]:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    valid = attention_mask[:, 1:].contiguous().bool() & (shift_labels != pad_token_id)
    labels = shift_labels.masked_fill(~valid, -100)
    loss_sum = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )
    return loss_sum, int(valid.long().sum().item())


def compute_loss(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    collect_metrics: bool = True,
    collect_routing_diagnostics: bool = True,
    compute_diversity: bool = True,
    diversity_scale: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    input_ids = batch["input_ids"].to(next(model.parameters()).device)
    attention_mask = batch["attention_mask"].to(input_ids.device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    lm_sum, lm_tokens = causal_lm_loss_sum(out["logits"], input_ids, attention_mask, model_cfg.pad_token_id)
    loss_lm = lm_sum / max(lm_tokens, 1)

    loss_noise = input_ids.new_tensor(0.0, dtype=torch.float32)
    if train_cfg.noise_weight > 0 and model_cfg.diffusion_noise_scale > 0:
        clean = model.token_embedding(input_ids)
        noise = torch.randn_like(clean)
        noisy = clean + model_cfg.diffusion_noise_scale * noise * attention_mask[:, :, None].to(clean.dtype)
        noisy_out = model(inputs_embeds=noisy, attention_mask=attention_mask)
        pred_noise = noisy_out["pred_noise"]
        mse = (pred_noise.float() - noise.float()).pow(2).mean(dim=-1)
        loss_noise = (mse * attention_mask.float()).sum() / attention_mask.float().sum().clamp_min(1.0)

    reg = anchor_regularization_losses(
        model,
        out["anchor_stats"],
        entropy_target_ratio_start=train_cfg.entropy_target_ratio_start,
        entropy_target_ratio_end=train_cfg.entropy_target_ratio_end,
        compute_diversity=compute_diversity,
    )
    total = (
        train_cfg.lm_weight * loss_lm
        + train_cfg.noise_weight * loss_noise
        + train_cfg.cluster_weight * reg["cluster"]
        + train_cfg.diversity_weight * diversity_scale * reg["diversity"]
        + train_cfg.usage_weight * reg["usage"]
        - train_cfg.entropy_weight * reg["entropy_reward"]
        + train_cfg.entropy_over_weight * reg["entropy_over"]
    )
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite loss detected.")

    if not collect_metrics:
        return total, {}

    metric_total = total + (
        train_cfg.diversity_weight
        * (1.0 - diversity_scale)
        * reg["diversity"]
    )

    scalar_names = [
        "loss",
        "lm",
        "noise",
        "cluster",
        "diversity",
        "usage",
        "entropy",
        "entropy_normalized",
        "entropy_target",
        "entropy_over",
        "effective_anchors",
        "mean_top1_membership",
        "mean_top1_top2_margin",
    ]
    scalar_tensors = [
        metric_total,
        loss_lm,
        loss_noise,
        reg["cluster"],
        reg["diversity"],
        reg["usage"],
        reg["entropy"],
        reg["entropy_normalized"],
        reg["entropy_target"],
        reg["entropy_over"],
        reg["effective_anchors"],
        reg["top1_membership"],
        reg["membership_margin"],
    ]
    scalar_values = torch.stack(
        [value.detach().float() for value in scalar_tensors]
    ).cpu().tolist()
    metrics = dict(zip(scalar_names, scalar_values))
    metrics["lm_tokens"] = float(lm_tokens)
    anchor_stats = out["anchor_stats"]
    if anchor_stats and collect_routing_diagnostics:
        usage = torch.stack([s.usage_prob.detach().float() for s in anchor_stats]).mean(dim=0)
        valid = attention_mask.bool()
        num_anchors = usage.numel()
        top1_counts = torch.zeros(num_anchors, device=usage.device)
        topk_counts = torch.zeros(num_anchors, device=usage.device)
        active_per_token = []
        unique_per_sequence = []
        for stats in anchor_stats:
            indices = stats.indices.detach()
            membership = stats.membership.detach()
            top1_counts.scatter_add_(0, indices[..., 0][valid], torch.ones_like(indices[..., 0][valid], dtype=top1_counts.dtype))
            valid_slots = valid.unsqueeze(-1).expand_as(indices)
            topk_counts.scatter_add_(0, indices[valid_slots], torch.ones_like(indices[valid_slots], dtype=topk_counts.dtype))
            active_per_token.append((membership > 1e-8).sum(dim=-1)[valid].float())
            for batch_index in range(indices.size(0)):
                unique_per_sequence.append(torch.unique(indices[batch_index][valid[batch_index]]).numel())
        token_observations = max(int(valid.sum().item()) * len(anchor_stats), 1)
        slot_observations = max(token_observations * anchor_stats[0].membership.size(-1), 1)
        metrics.update(
            {
                "metric_semantics_version": 2.0,
                "mean_active_anchors_per_token": float(torch.cat(active_per_token).mean().item()),
                "mean_unique_anchors_per_sequence_per_layer": float(sum(unique_per_sequence) / max(len(unique_per_sequence), 1)),
                "global_active_anchors_after_layer_average": float((usage >= 1e-4).float().sum().item()),
                "top1_selection_frequency": float((top1_counts / token_observations).max().item()),
                "topk_slot_frequency": float((topk_counts / slot_observations).max().item()),
                "token_coverage_frequency": float((topk_counts / token_observations).max().item()),
                "membership_mass_share": float(usage.max().item()),
                "dead_anchor_ratio": float((usage < 1e-4).float().mean().item()),
                "top_anchor_frequency": float(usage.max().item()),
                "mean_active_anchors": float((usage >= 1e-4).float().sum().item()),
            }
        )
    return total, metrics


@torch.no_grad()
def causal_leakage_check(
    model: torch.nn.Module,
    model_cfg: ModelConfig,
    device: torch.device,
    precision: str,
) -> Dict[str, Any]:
    seq_len = min(32, model_cfg.max_seq_len)
    cut = seq_len // 2
    high = max(32, min(model_cfg.vocab_size - 1, 1000))
    base = (torch.arange(seq_len, device=device) % (high - 16)) + 16
    ids = base.unsqueeze(0).repeat(2, 1).long()
    ids[1, cut:] = torch.flip(ids[1, cut:], dims=[0])
    mask = torch.ones_like(ids)
    was_training = model.training
    model.eval()
    with autocast_context(device, precision):
        logits = model(input_ids=ids, attention_mask=mask)["logits"].float()
    if was_training:
        model.train()
    max_prefix_diff = float((logits[0, :cut] - logits[1, :cut]).abs().max().item())
    return {
        "status": "PASS" if max_prefix_diff < 1e-3 else "FAIL",
        "max_prefix_logit_diff": max_prefix_diff,
        "threshold": 1e-3,
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_size: int,
    model_cfg: ModelConfig,
    device: torch.device,
    precision: str,
) -> Dict[str, float]:
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    token_count = 0
    for start in range(0, input_ids.size(0), batch_size):
        ids = input_ids[start : start + batch_size].to(device)
        mask = attention_mask[start : start + batch_size].to(device)
        with autocast_context(device, precision):
            out = model(input_ids=ids, attention_mask=mask)
        batch_loss_sum, batch_tokens = causal_lm_loss_sum(
            out["logits"].float(), ids, mask, model_cfg.pad_token_id
        )
        loss_sum += batch_loss_sum.float()
        token_count += batch_tokens
    val_loss = float((loss_sum / max(token_count, 1)).item())
    if val_loss < 80.0:
        val_ppl = float(math.exp(val_loss))
    else:
        val_ppl = float("inf")
    model.train()
    return {
        "val_loss": val_loss,
        "val_ppl": val_ppl,
        "val_tokens": float(token_count),
    }


def apply_repetition_penalty(logits: torch.Tensor, token_ids: Iterable[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0:
        return logits
    for token_id in set(int(x) for x in token_ids):
        if 0 <= token_id < logits.numel():
            logits[token_id] = logits[token_id] * penalty if logits[token_id] < 0 else logits[token_id] / penalty
    return logits


def filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.numel()))
        logits = logits.masked_fill(logits < values[-1], -float("inf"))
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        logits[sorted_idx[remove]] = -float("inf")
    return logits


@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    model_cfg: ModelConfig,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.92,
    repetition_penalty: float = 1.1,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated: List[int] = []
    for _ in range(max_new_tokens):
        context = ids[-model_cfg.max_seq_len :]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        mask = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=mask)
        logits = out["logits"][0, -1].float()
        logits = apply_repetition_penalty(logits, generated if generated else context[-64:], repetition_penalty)
        if temperature <= 0:
            next_id = int(torch.argmax(logits).item())
        else:
            logits = filter_logits(logits / max(temperature, 1e-5), top_k, top_p)
            probs = F.softmax(logits, dim=-1)
            next_id = int(torch.multinomial(probs, 1).item())
        if next_id in {model_cfg.eos_token_id, model_cfg.pad_token_id}:
            break
        ids.append(next_id)
        generated.append(next_id)
    model.train()
    return tokenizer.decode(generated, skip_special_tokens=True)


def save_generations(
    path: Path,
    model: torch.nn.Module,
    tokenizer: Any,
    model_cfg: ModelConfig,
    device: torch.device,
    model_type: str,
    phase: str,
    optimizer_step: int,
    tokens_seen: int,
) -> None:
    samples = []
    for prompt in GENERATION_PROMPTS:
        samples.append(
            {
                "prompt": prompt,
                "completion": generate_text(model, tokenizer, model_cfg, prompt, device),
            }
        )
    append_jsonl(
        path,
        {
            "model_type": model_type,
            "phase": phase,
            "optimizer_step": optimizer_step,
            "tokens_seen": tokens_seen,
            "samples": samples,
        },
    )


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    optimizer_step: int,
    micro_step: int,
    sequence_index: int,
    tokens_seen: int,
    best_val_loss: float,
    metrics: Dict[str, Any],
    budget: Dict[str, Any],
    seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": model_cfg.model_type,
            "parameter_count": count_parameters(model),
            "optimizer_step": optimizer_step,
            "global_step": optimizer_step,
            "micro_step": micro_step,
            "sequence_index": sequence_index,
            "tokens_seen": tokens_seen,
            "best_val_loss": best_val_loss,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": dataclass_to_dict(model_cfg),
            "train_config": dataclass_to_dict(train_cfg),
            "tokenizer_hash": budget.get("tokenizer_hash", ""),
            "dataset_manifest_sha256": budget.get("dataset_manifest_sha256", ""),
            "config": {
                "model": dataclass_to_dict(model_cfg),
                "train": dataclass_to_dict(train_cfg),
                "budget": budget,
            },
            "random_seed": seed,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def unique_dir(base: Path, resume: bool) -> Path:
    if resume or not base.exists():
        return base
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return base.with_name(f"{base.name}_{stamp}")


def make_batch(input_ids: torch.Tensor, attention_mask: torch.Tensor, start: int, batch_size: int) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": input_ids[start : start + batch_size],
        "attention_mask": attention_mask[start : start + batch_size],
    }


def train_model(
    *,
    args: argparse.Namespace,
    phase: str,
    model_type: str,
    output_dir: Path,
    budget: Dict[str, Any],
    train_ids: torch.Tensor,
    train_mask: torch.Tensor,
    val_ids: torch.Tensor,
    val_mask: torch.Tensor,
    total_optimizer_steps: int,
    max_sequences: int,
    preflight_resume_probe: bool,
) -> Dict[str, Any]:
    seed_offset = 0
    set_seed(args.seed + seed_offset)
    device = torch.device(args.device)
    tokenizer = load_tokenizer(str(args.tokenizer))
    model_cfg = make_model_config(
        tokenizer=tokenizer,
        model_type=model_type,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        num_anchors=args.num_anchors,
        top_k=args.top_k,
        routing_type=args.routing_type,
        cosine_temperature=args.cosine_temperature,
        aggregation_impl=args.aggregation_impl,
        diversity_margin=args.diversity_margin,
        diversity_max_weight=args.diversity_max_weight,
    )
    train_cfg = make_train_config(args, output_dir, total_optimizer_steps)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "model_config.json", dataclass_to_dict(model_cfg))
    write_json(output_dir / "train_config.json", dataclass_to_dict(train_cfg))
    write_json(output_dir / "budget.json", budget)

    max_token_id = int(train_ids[: min(train_ids.size(0), max_sequences)].max().item())
    if max_token_id >= model_cfg.vocab_size:
        raise ValueError(f"Invalid token id {max_token_id} >= vocab_size {model_cfg.vocab_size}")

    model = build_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = make_scheduler(
        optimizer,
        args.warmup_steps,
        total_optimizer_steps,
        getattr(args, "scheduler_mode", "cosine"),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.precision == "fp16"))

    parameter_count = count_parameters(model)
    leakage = causal_leakage_check(model, model_cfg, device, args.precision)
    append_jsonl(
        output_dir / "metrics.jsonl",
        {
            "event": "causal_leakage_check",
            "phase": phase,
            "model_type": model_type,
            **leakage,
        },
    )
    if leakage["status"] != "PASS":
        raise RuntimeError(f"Causal leakage check failed: {leakage}")

    latest = output_dir / "latest.pt"
    optimizer_step = 0
    micro_step = 0
    sequence_index = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    resume_verified = not preflight_resume_probe
    last_metrics: Dict[str, Any] = {}

    if args.resume and latest.exists():
        checkpoint = load_checkpoint(latest, model, optimizer, scheduler, device)
        optimizer_step = int(checkpoint.get("optimizer_step", 0))
        micro_step = int(checkpoint.get("micro_step", 0))
        sequence_index = int(checkpoint.get("sequence_index", 0))
        tokens_seen = int(checkpoint.get("tokens_seen", 0))
        best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        resume_verified = True
        if phase == "full" and args.additional_optimizer_steps > 0:
            backup = output_dir / f"pre_entropy_repair_step_{optimizer_step}.pt"
            if not backup.exists():
                shutil.copy2(latest, backup)
        if (
            phase == "full"
            and args.additional_optimizer_steps > 0
            and sequence_index >= max_sequences
        ):
            sequence_index = 0
            append_jsonl(
                output_dir / "metrics.jsonl",
                {
                    "event": "dataset_pass_restart",
                    "phase": phase,
                    "model_type": model_type,
                    "optimizer_step": optimizer_step,
                    "tokens_seen": tokens_seen,
                    "reason": "additional_optimizer_steps",
                },
            )

    initial_eval = evaluate(model, val_ids, val_mask, args.eval_batch_size, model_cfg, device, args.precision)
    append_jsonl(
        output_dir / "metrics.jsonl",
        {
            "event": "eval",
            "phase": phase,
            "model_type": model_type,
            "optimizer_step": optimizer_step,
            "tokens_seen": tokens_seen,
            **initial_eval,
        },
    )
    if args.save_generations:
        save_generations(
            output_dir / "generations.jsonl",
            model,
            tokenizer,
            model_cfg,
            device,
            model_type,
            phase,
            optimizer_step,
            tokens_seen,
        )

    generation_milestones = {
        0,
        max(1, int(total_optimizer_steps * 0.25)),
        max(1, int(total_optimizer_steps * 0.50)),
        max(1, int(total_optimizer_steps * 0.75)),
        total_optimizer_steps,
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    optimizer_step_at_start = optimizer_step
    tokens_seen_at_start = tokens_seen
    wall_start = time.perf_counter()
    pbar = tqdm(total=total_optimizer_steps, initial=optimizer_step, desc=f"{phase}:{model_type}")
    train_limit = min(max_sequences, train_ids.size(0))
    failure: Optional[str] = None
    try:
        while optimizer_step < total_optimizer_steps and sequence_index < train_limit:
            remaining_micro = math.ceil((train_limit - sequence_index) / args.batch_size)
            accum_count = min(args.grad_accum_steps, remaining_micro)
            if phase == "preflight":
                remaining_steps = total_optimizer_steps - optimizer_step
                remaining_allowed_micro = max(1, remaining_steps * args.grad_accum_steps)
                accum_count = min(accum_count, remaining_allowed_micro, args.grad_accum_steps)

            optimizer.zero_grad(set_to_none=True)
            accum_metrics: Dict[str, float] = {}
            accum_tokens = 0
            for micro_index in range(accum_count):
                batch_cpu = make_batch(train_ids, train_mask, sequence_index, args.batch_size)
                actual_batch = int(batch_cpu["input_ids"].size(0))
                if actual_batch == 0:
                    break
                sequence_index += actual_batch
                batch_tokens = int(batch_cpu["attention_mask"].sum().item())
                accum_tokens += batch_tokens
                batch = {k: v.to(device) for k, v in batch_cpu.items()}
                collect_metrics = micro_index == accum_count - 1
                collect_routing_diagnostics = collect_metrics and (
                    (optimizer_step + 1) % args.log_every == 0
                    or optimizer_step + 1 == total_optimizer_steps
                )
                diversity_due = collect_metrics and (
                    (optimizer_step + 1) % train_cfg.diversity_every == 0
                    or optimizer_step + 1 == total_optimizer_steps
                )
                with autocast_context(device, args.precision):
                    loss, metrics = compute_loss(
                        model,
                        batch,
                        model_cfg,
                        train_cfg,
                        collect_metrics=collect_metrics,
                        collect_routing_diagnostics=collect_routing_diagnostics,
                        compute_diversity=diversity_due,
                        diversity_scale=(
                            accum_count * train_cfg.diversity_every
                            if diversity_due
                            else 0.0
                        ),
                    )
                    scaled_loss = loss / max(accum_count, 1)
                scaler.scale(scaled_loss).backward()
                micro_step += 1
                for key, value in metrics.items():
                    accum_metrics[key] = float(value)
                del batch

            if accum_count <= 0:
                break
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Non-finite gradient detected.")
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            optimizer_step += 1
            tokens_seen += accum_tokens
            last_metrics = dict(accum_metrics)
            last_metrics.update(
                {
                    "grad_norm": float(grad_norm.detach().item()),
                    "lr": float(scheduler.get_last_lr()[0]),
                    "tokens_seen": float(tokens_seen),
                    "micro_step": float(micro_step),
                }
            )

            append_jsonl(
                output_dir / "metrics.jsonl",
                {
                    "event": "train",
                    "phase": phase,
                    "model_type": model_type,
                    "optimizer_step": optimizer_step,
                    **last_metrics,
                },
            )

            if preflight_resume_probe and not resume_verified and optimizer_step >= args.resume_probe_steps:
                probe = output_dir / "resume_probe.pt"
                save_checkpoint(
                    probe,
                    model,
                    optimizer,
                    scheduler,
                    model_cfg,
                    train_cfg,
                    optimizer_step,
                    micro_step,
                    sequence_index,
                    tokens_seen,
                    best_val_loss,
                    last_metrics,
                    budget,
                    args.seed + seed_offset,
                )
                restored = load_checkpoint(probe, model, optimizer, scheduler, device)
                if int(restored.get("optimizer_step", -1)) != optimizer_step:
                    raise RuntimeError("Resume probe failed: optimizer_step mismatch.")
                if int(restored.get("tokens_seen", -1)) != tokens_seen:
                    raise RuntimeError("Resume probe failed: tokens_seen mismatch.")
                resume_verified = True
                append_jsonl(
                    output_dir / "metrics.jsonl",
                    {
                        "event": "resume_probe",
                        "phase": phase,
                        "model_type": model_type,
                        "optimizer_step": optimizer_step,
                        "tokens_seen": tokens_seen,
                        "status": "PASS",
                    },
                )

            should_eval = (optimizer_step % args.eval_every == 0) or (optimizer_step == total_optimizer_steps)
            if should_eval:
                val_metrics = evaluate(model, val_ids, val_mask, args.eval_batch_size, model_cfg, device, args.precision)
                last_metrics.update(val_metrics)
                append_jsonl(
                    output_dir / "metrics.jsonl",
                    {
                        "event": "eval",
                        "phase": phase,
                        "model_type": model_type,
                        "optimizer_step": optimizer_step,
                        "tokens_seen": tokens_seen,
                        **val_metrics,
                    },
                )
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        optimizer,
                        scheduler,
                        model_cfg,
                        train_cfg,
                        optimizer_step,
                        micro_step,
                        sequence_index,
                        tokens_seen,
                        best_val_loss,
                        last_metrics,
                        budget,
                        args.seed + seed_offset,
                    )

            if args.save_generations and optimizer_step in generation_milestones:
                save_generations(
                    output_dir / "generations.jsonl",
                    model,
                    tokenizer,
                    model_cfg,
                    device,
                    model_type,
                    phase,
                    optimizer_step,
                    tokens_seen,
                )

            should_save = (optimizer_step % args.save_every == 0) or (optimizer_step == total_optimizer_steps)
            if should_save:
                save_checkpoint(
                    output_dir / "latest.pt",
                    model,
                    optimizer,
                    scheduler,
                    model_cfg,
                    train_cfg,
                    optimizer_step,
                    micro_step,
                    sequence_index,
                    tokens_seen,
                    best_val_loss,
                    last_metrics,
                    budget,
                    args.seed + seed_offset,
                )
                save_checkpoint(
                    output_dir / f"step_{optimizer_step}.pt",
                    model,
                    optimizer,
                    scheduler,
                    model_cfg,
                    train_cfg,
                    optimizer_step,
                    micro_step,
                    sequence_index,
                    tokens_seen,
                    best_val_loss,
                    last_metrics,
                    budget,
                    args.seed + seed_offset,
                )

            pbar.update(1)
            pbar.set_postfix(
                {
                    "lm": f"{last_metrics.get('lm', 0.0):.3f}",
                    "val": f"{last_metrics.get('val_loss', initial_eval['val_loss']):.3f}",
                    "tok": tokens_seen,
                }
            )
    except Exception as exc:
        failure = repr(exc)
        append_jsonl(
            output_dir / "metrics.jsonl",
            {
                "event": "failure",
                "phase": phase,
                "model_type": model_type,
                "optimizer_step": optimizer_step,
                "tokens_seen": tokens_seen,
                "error": failure,
            },
        )
    finally:
        pbar.close()

    wall_time = time.perf_counter() - wall_start
    try:
        final_eval = evaluate(model, val_ids, val_mask, args.eval_batch_size, model_cfg, device, args.precision)
    except Exception as exc:
        if failure is None:
            failure = f"final_eval_failed:{exc!r}"
        final_eval = {"val_loss": float("nan"), "val_ppl": float("nan"), "val_tokens": 0.0}
    last_metrics.update(final_eval)
    if final_eval["val_loss"] < best_val_loss:
        best_val_loss = final_eval["val_loss"]
    save_checkpoint(
        output_dir / "latest.pt",
        model,
        optimizer,
        scheduler,
        model_cfg,
        train_cfg,
        optimizer_step,
        micro_step,
        sequence_index,
        tokens_seen,
        best_val_loss,
        last_metrics,
        budget,
        args.seed + seed_offset,
    )
    save_checkpoint(
        output_dir / "final.pt",
        model,
        optimizer,
        scheduler,
        model_cfg,
        train_cfg,
        optimizer_step,
        micro_step,
        sequence_index,
        tokens_seen,
        best_val_loss,
        last_metrics,
        budget,
        args.seed + seed_offset,
    )
    if args.save_generations:
        save_generations(
            output_dir / "generations.jsonl",
            model,
            tokenizer,
            model_cfg,
            device,
            model_type,
            phase,
            optimizer_step,
            tokens_seen,
        )

    peak_allocated = 0
    peak_reserved = 0
    if device.type == "cuda":
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))

    completed_requested_steps = optimizer_step == total_optimizer_steps
    is_repair_stage = phase == "full" and args.additional_optimizer_steps > 0
    consumed_expected_sequences = (
        phase == "preflight" or sequence_index == train_limit or is_repair_stage
    )
    optimizer_steps_this_run = optimizer_step - optimizer_step_at_start
    tokens_this_run = tokens_seen - tokens_seen_at_start
    status = (
        "PASS"
        if failure is None
        and resume_verified
        and completed_requested_steps
        and consumed_expected_sequences
        and math.isfinite(final_eval["val_loss"])
        else "FAIL"
    )
    summary = {
        "status": status,
        "failure": failure,
        "phase": phase,
        "model_type": model_type,
        "output_dir": str(output_dir),
        "parameter_count": parameter_count,
        "initial_val_loss": initial_eval["val_loss"],
        "initial_val_ppl": initial_eval["val_ppl"],
        "final_val_loss": final_eval["val_loss"],
        "final_val_ppl": final_eval["val_ppl"],
        "best_val_loss": best_val_loss,
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "tokens_seen": tokens_seen,
        "optimizer_steps_this_run": optimizer_steps_this_run,
        "tokens_this_run": tokens_this_run,
        "seconds_per_optimizer_step": wall_time / max(optimizer_steps_this_run, 1),
        "tokens_per_second": tokens_this_run / max(wall_time, 1e-8),
        "wall_time_seconds": wall_time,
        "peak_allocated_vram_bytes": peak_allocated,
        "peak_reserved_vram_bytes": peak_reserved,
        "resume_verified": resume_verified,
        "causal_leakage_check": leakage,
        "completed_requested_steps": completed_requested_steps,
        "consumed_expected_sequences": consumed_expected_sequences,
        "entropy_repair_stage": is_repair_stage,
        "last_metrics": last_metrics,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def comparison_entry(summary: Dict[str, Any]) -> Dict[str, Any]:
    last = summary.get("last_metrics", {})
    return {
        "status": summary.get("status"),
        "parameters": summary.get("parameter_count"),
        "tokens_seen": summary.get("tokens_seen"),
        "optimizer_steps": summary.get("optimizer_steps"),
        "micro_steps": summary.get("micro_steps"),
        "initial_val_loss": summary.get("initial_val_loss"),
        "final_val_loss": summary.get("final_val_loss"),
        "best_val_loss": summary.get("best_val_loss"),
        "perplexity": summary.get("final_val_ppl"),
        "peak_vram_gb": summary.get("peak_reserved_vram_bytes", 0) / (1024**3),
        "peak_allocated_vram_gb": summary.get("peak_allocated_vram_bytes", 0) / (1024**3),
        "tokens_per_second": summary.get("tokens_per_second"),
        "seconds_per_optimizer_step": summary.get("seconds_per_optimizer_step"),
        "wall_time_seconds": summary.get("wall_time_seconds"),
        "anchor_entropy": last.get("entropy"),
        "dead_anchor_ratio": last.get("dead_anchor_ratio"),
        "top_anchor_frequency": last.get("top_anchor_frequency"),
        "mean_active_anchors": last.get("mean_active_anchors"),
        "diffusion_auxiliary_loss": last.get("noise"),
        "cluster_loss": last.get("cluster"),
        "diversity_loss": last.get("diversity"),
        "usage_balance_loss": last.get("usage"),
        "normalized_anchor_entropy": last.get("entropy_normalized"),
        "anchor_entropy_target": last.get("entropy_target"),
        "anchor_entropy_over_loss": last.get("entropy_over"),
        "effective_anchors_per_token": last.get("effective_anchors"),
        "mean_top1_membership": last.get("mean_top1_membership"),
        "mean_top1_top2_margin": last.get("mean_top1_top2_margin"),
        "output_dir": summary.get("output_dir"),
    }


def write_report(root: Path, budget: Dict[str, Any], summaries: Dict[str, Dict[str, Any]]) -> None:
    transformer = summaries.get("full_transformer") or summaries.get("preflight_transformer", {})
    fdt = summaries.get("full_fdt") or summaries.get("preflight_fdt", {})
    t_entry = comparison_entry(transformer) if transformer else {}
    f_entry = comparison_entry(fdt) if fdt else {}
    comparison = {
        "dataset": {
            "path": budget["dataset_dir"],
            "tokenizer_hash": budget.get("tokenizer_hash", ""),
            "dataset_manifest_sha256": budget.get("dataset_manifest_sha256", ""),
            "train_tokens": budget["actual_train_tokens"],
            "validation_tokens": budget["split_stats"]["validation"]["tokens"],
            "test_tokens": budget["split_stats"]["test"]["tokens"],
            "packing_efficiency": budget["packing_efficiency_vs_source_plus_tail"],
        },
        "training_budget": budget,
        "transformer": t_entry,
        "fdt": f_entry,
        "comparison": {
            "better_validation_loss": (
                "transformer"
                if t_entry.get("best_val_loss", float("inf")) < f_entry.get("best_val_loss", float("inf"))
                else "fdt"
                if f_entry
                else "unknown"
            ),
            "faster_training": (
                "transformer"
                if t_entry.get("tokens_per_second", 0.0) > f_entry.get("tokens_per_second", 0.0)
                else "fdt"
                if f_entry
                else "unknown"
            ),
            "lower_vram": (
                "transformer"
                if t_entry.get("peak_vram_gb", float("inf")) < f_entry.get("peak_vram_gb", float("inf"))
                else "fdt"
                if f_entry
                else "unknown"
            ),
            "initial_architectural_signal": "single 10M-token matched pass only; not a superiority claim",
        },
        "all_runs": summaries,
    }
    write_json(root / "artifacts" / "first_experiment_comparison.json", comparison)

    lines = [
        "# First FDT vs Transformer Experiment",
        "",
        "This is a matched 10M-token first-pass experiment. It is not evidence of general superiority.",
        "",
        "## Dataset And Budget",
        f"- Dataset: `{budget['dataset_dir']}`",
        f"- Tokenizer hash: `{budget.get('tokenizer_hash', '')}`",
        f"- Train tokens: {budget['actual_train_tokens']}",
        f"- Validation tokens: {budget['split_stats']['validation']['tokens']}",
        f"- Test tokens: {budget['split_stats']['test']['tokens']}",
        f"- Train sequences: {budget['train_sequences']}",
        f"- Context length: {budget['max_seq_len']}",
        f"- Batch size: {budget['batch_size']}",
        f"- Gradient accumulation: {budget['gradient_accumulation_steps']}",
        f"- Optimizer steps for one full pass: {budget['optimizer_steps_for_one_full_pass']}",
        f"- Packing efficiency: {budget['packing_efficiency_vs_source_plus_tail']:.6f}",
        "",
        "## Transformer",
    ]
    if t_entry:
        lines.extend(
            [
                f"- Status: {t_entry.get('status')}",
                f"- Parameters: {t_entry.get('parameters')}",
                f"- Initial val loss: {t_entry.get('initial_val_loss')}",
                f"- Final val loss: {t_entry.get('final_val_loss')}",
                f"- Best val loss: {t_entry.get('best_val_loss')}",
                f"- Final perplexity: {t_entry.get('perplexity')}",
                f"- Tokens/sec: {t_entry.get('tokens_per_second')}",
                f"- Peak reserved VRAM GB: {t_entry.get('peak_vram_gb')}",
                f"- Wall time seconds: {t_entry.get('wall_time_seconds')}",
            ]
        )
    lines.append("")
    lines.append("## FDT")
    if f_entry:
        lines.extend(
            [
                f"- Status: {f_entry.get('status')}",
                f"- Parameters: {f_entry.get('parameters')}",
                f"- Initial val loss: {f_entry.get('initial_val_loss')}",
                f"- Final val loss: {f_entry.get('final_val_loss')}",
                f"- Best val loss: {f_entry.get('best_val_loss')}",
                f"- Final perplexity: {f_entry.get('perplexity')}",
                f"- Tokens/sec: {f_entry.get('tokens_per_second')}",
                f"- Peak reserved VRAM GB: {f_entry.get('peak_vram_gb')}",
                f"- Wall time seconds: {f_entry.get('wall_time_seconds')}",
                f"- Anchor entropy: {f_entry.get('anchor_entropy')}",
                f"- Dead anchor ratio: {f_entry.get('dead_anchor_ratio')}",
                f"- Top anchor frequency: {f_entry.get('top_anchor_frequency')}",
                f"- Mean active anchors: {f_entry.get('mean_active_anchors')}",
                f"- Diffusion auxiliary loss: {f_entry.get('diffusion_auxiliary_loss')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Comparison",
            f"- Better validation loss: {comparison['comparison']['better_validation_loss']}",
            f"- Faster training: {comparison['comparison']['faster_training']}",
            f"- Lower VRAM: {comparison['comparison']['lower_vram']}",
            f"- Initial architectural signal: {comparison['comparison']['initial_architectural_signal']}",
            "",
            "## Files",
            "- `artifacts/first_experiment_budget.json`",
            "- `artifacts/first_experiment_comparison.json`",
            "- per-run `metrics.jsonl`, `generations.jsonl`, `summary.json`, and checkpoints",
        ]
    )
    (root / "FIRST_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    dataset_dir = Path(args.dataset_dir)
    tokenizer_dir = Path(args.tokenizer)
    budget = calculate_budget(dataset_dir, tokenizer_dir, args.batch_size, args.grad_accum_steps, args.max_seq_len)
    write_json(ROOT / "artifacts" / "first_experiment_budget.json", budget)
    print(json.dumps(budget, indent=2, ensure_ascii=False))

    train_ids, train_mask = load_split_tensors(dataset_dir, "train")
    val_ids, val_mask = load_split_tensors(dataset_dir, "validation")
    summaries: Dict[str, Dict[str, Any]] = {}
    for model_type in ("transformer", "fdt"):
        if model_type in args.models:
            continue
        existing_summary = ROOT / "runs" / f"full_{model_type}" / "summary.json"
        if existing_summary.exists():
            summaries[f"full_{model_type}"] = json.loads(
                existing_summary.read_text(encoding="utf-8")
            )

    preflight_sequences = min(train_ids.size(0), args.preflight_optimizer_steps * args.grad_accum_steps * args.batch_size)
    full_steps = int(budget["optimizer_steps_for_one_full_pass"])
    phases: List[str]
    if args.phase == "both":
        phases = ["preflight", "full"]
    else:
        phases = [args.phase]

    for phase in phases:
        if phase == "budget":
            continue
        phase_steps = args.preflight_optimizer_steps if phase == "preflight" else full_steps
        phase_sequences = preflight_sequences if phase == "preflight" else int(budget["train_sequences"])
        for model_type in args.models:
            base = ROOT / "runs" / f"{phase}_{model_type}"
            out_dir = unique_dir(base, args.resume and phase == "full")
            model_phase_steps = phase_steps
            if phase == "full" and args.resume and args.additional_optimizer_steps > 0:
                latest = base / "latest.pt"
                if not latest.exists():
                    raise FileNotFoundError(
                        f"Cannot continue {model_type}: missing checkpoint {latest}"
                    )
                checkpoint_meta = torch.load(latest, map_location="cpu")
                completed_steps = int(checkpoint_meta.get("optimizer_step", 0))
                model_phase_steps = completed_steps + args.additional_optimizer_steps
            summary = train_model(
                args=args,
                phase=phase,
                model_type=model_type,
                output_dir=out_dir,
                budget=budget,
                train_ids=train_ids,
                train_mask=train_mask,
                val_ids=val_ids,
                val_mask=val_mask,
                total_optimizer_steps=model_phase_steps,
                max_sequences=phase_sequences,
                preflight_resume_probe=(phase == "preflight"),
            )
            summaries[f"{phase}_{model_type}"] = summary
            if phase == "preflight" and summary["status"] != "PASS":
                write_report(ROOT, budget, summaries)
                raise RuntimeError(f"{model_type} preflight failed; stopping before full run.")
        if phase == "preflight":
            print(f"Preflight passed for: {', '.join(args.models)}.")

    write_report(ROOT, budget, summaries)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the first matched FDT vs Transformer experiment.")
    parser.add_argument("--dataset_dir", default=str(ROOT / "prepared_data" / "fdt_rlm_10m"))
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizers" / "fdt_bpe"))
    parser.add_argument("--phase", choices=["budget", "preflight", "full", "both"], default="both")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["transformer", "fdt"],
        default=["transformer", "fdt"],
    )
    parser.add_argument("--additional_optimizer_steps", type=int, default=0)

    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_anchors", type=int, default=128)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--routing_type", choices=["cosine", "gaussian"], default="cosine")
    parser.add_argument("--cosine_temperature", type=float, default=0.25)
    parser.add_argument(
        "--aggregation_impl",
        choices=["dense_reference", "dense_gather_first"],
        default="dense_gather_first",
    )
    parser.add_argument("--diversity_margin", type=float, default=0.20)
    parser.add_argument("--diversity_max_weight", type=float, default=2.0)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--preflight_optimizer_steps", type=int, default=100)
    parser.add_argument("--resume_probe_steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=250)
    parser.add_argument("--eval_every", type=int, default=250)
    parser.add_argument(
        "--save_generations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--lm_weight", type=float, default=1.0)
    parser.add_argument("--noise_weight", type=float, default=0.05)
    parser.add_argument("--cluster_weight", type=float, default=0.01)
    parser.add_argument("--diversity_weight", type=float, default=0.10)
    parser.add_argument("--diversity_every", type=int, default=4)
    parser.add_argument("--usage_weight", type=float, default=0.005)
    parser.add_argument("--entropy_weight", type=float, default=0.003)
    parser.add_argument("--entropy_target_ratio_start", type=float, default=0.65)
    parser.add_argument("--entropy_target_ratio_end", type=float, default=0.45)
    parser.add_argument("--entropy_over_weight", type=float, default=0.20)
    args = parser.parse_args()

    if args.additional_optimizer_steps < 0:
        parser.error("--additional_optimizer_steps must be non-negative")
    if args.additional_optimizer_steps > 0 and (not args.resume or args.phase != "full"):
        parser.error("additional optimizer steps require --resume --phase full")

    if args.phase == "budget":
        budget = calculate_budget(
            Path(args.dataset_dir),
            Path(args.tokenizer),
            args.batch_size,
            args.grad_accum_steps,
            args.max_seq_len,
        )
        write_json(ROOT / "artifacts" / "first_experiment_budget.json", budget)
        print(json.dumps(budget, indent=2, ensure_ascii=False))
        return

    summaries = run(args)
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
