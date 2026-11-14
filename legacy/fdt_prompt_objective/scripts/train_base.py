from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fdt_rlm.config import ModelConfig, TrainConfig, dataclass_to_dict
from fdt_rlm.data import PackedTextDataset, ShardedTokenDataset, iter_hf_texts, iter_local_texts
from fdt_rlm.models import anchor_regularization_losses, build_model, causal_lm_loss
from fdt_rlm.tokenization import load_tokenizer


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return torch.amp.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def make_text_iterator(args):
    if args.text_file:
        return lambda: iter_local_texts(args.text_file)
    return lambda: iter_hf_texts(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        streaming=args.streaming,
    )


def compute_loss(
    model,
    batch,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    compute_diversity: bool = True,
    diversity_scale: float = 1.0,
):
    input_ids = batch["input_ids"].to(next(model.parameters()).device)
    attention_mask = batch["attention_mask"].to(input_ids.device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    loss_lm = causal_lm_loss(out["logits"], input_ids, attention_mask, model_cfg.pad_token_id)

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
    metrics = {
        "loss": float(total.detach().item()),
        "lm": float(loss_lm.detach().item()),
        "noise": float(loss_noise.detach().item()),
        "cluster": float(reg["cluster"].detach().item()),
        "div": float(reg["diversity"].detach().item()),
        "usage": float(reg["usage"].detach().item()),
        "entropy": float(reg["entropy"].detach().item()),
        "ent_norm": float(reg["entropy_normalized"].detach().item()),
        "ent_target": float(reg["entropy_target"].detach().item()),
        "ent_over": float(reg["entropy_over"].detach().item()),
        "effective_k": float(reg["effective_anchors"].detach().item()),
        "top1": float(reg["top1_membership"].detach().item()),
        "margin": float(reg["membership_margin"].detach().item()),
    }
    return total, metrics


def save_checkpoint(path: Path, model, optimizer, step: int, model_cfg: ModelConfig, train_cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": dataclass_to_dict(model_cfg),
        "train_config": dataclass_to_dict(train_cfg),
    }
    torch.save(payload, path)


@torch.no_grad()
def evaluate_one_batch(model, dataloader, model_cfg: ModelConfig, train_cfg: TrainConfig):
    model.eval()
    batch = next(iter(dataloader))
    loss, metrics = compute_loss(model, batch, model_cfg, train_cfg)
    model.train()
    return float(loss.detach().item()), metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a micro causal Transformer or FDT-RLM base model.")
    parser.add_argument("--model_type", choices=["transformer", "fdt"], default="fdt")
    parser.add_argument("--tokenizer_name", default="gpt2")
    parser.add_argument("--dataset_name", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text_field", default="text")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--text_file", default="")
    parser.add_argument("--dataset_dir", default="")
    parser.add_argument("--output_dir", default="runs/micro_fdt")

    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--mlp_ratio", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--use_self_attention", action=argparse.BooleanOptionalAction, default=False)
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

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    tokenizer = load_tokenizer(args.tokenizer_name)
    model_cfg = ModelConfig(
        vocab_size=len(tokenizer),
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        model_type=args.model_type,
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        mlp_ratio=args.mlp_ratio,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        use_self_attention=args.use_self_attention,
        num_anchors=args.num_anchors,
        top_k=args.top_k,
        routing_type=args.routing_type,
        cosine_temperature=args.cosine_temperature,
        aggregation_impl=args.aggregation_impl,
        diversity_margin=args.diversity_margin,
        diversity_max_weight=args.diversity_max_weight,
    )
    train_cfg = TrainConfig(
        tokenizer_name=args.tokenizer_name,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        split=args.split,
        text_field=args.text_field,
        streaming=args.streaming,
        text_file=args.text_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "model_config.json").write_text(json.dumps(dataclass_to_dict(model_cfg), indent=2), encoding="utf-8")
    (output_dir / "train_config.json").write_text(json.dumps(dataclass_to_dict(train_cfg), indent=2), encoding="utf-8")

    if args.dataset_dir:
        dataset = ShardedTokenDataset(args.dataset_dir, split="train")
        val_dataset = ShardedTokenDataset(args.dataset_dir, split="validation")
    else:
        texts = make_text_iterator(args)
        dataset = PackedTextDataset(texts, tokenizer, seq_len=args.max_seq_len)
        val_dataset = None
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    iterator = iter(dataloader)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size) if val_dataset is not None else None

    model = build_model(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.precision == "fp16"))
    start_step = 0
    latest = output_dir / "latest.pt"
    if args.resume and latest.exists():
        checkpoint = torch.load(latest, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = int(checkpoint.get("step", 0))

    model.train()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(range(start_step + 1, args.max_steps + 1), desc=f"train:{args.model_type}")
    running = {}
    for step in pbar:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("Training dataset produced no batches.") from exc
        with autocast_context(device, args.precision):
            optimizer_step_number = step // args.grad_accum_steps
            apply_diversity = step % args.grad_accum_steps == 0 and (
                optimizer_step_number % train_cfg.diversity_every == 0
                or step == args.max_steps
            )
            loss, metrics = compute_loss(
                model,
                batch,
                model_cfg,
                train_cfg,
                compute_diversity=apply_diversity,
                diversity_scale=(
                    args.grad_accum_steps * train_cfg.diversity_every
                    if apply_diversity
                    else 0.0
                ),
            )
            loss = loss / args.grad_accum_steps
        scaler.scale(loss).backward()

        if step % args.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if torch.isfinite(grad_norm):
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running = metrics
        if step % args.log_every == 0:
            pbar.set_postfix({k: f"{v:.3f}" for k, v in running.items()})
        if args.eval_every > 0 and val_loader is not None and step % args.eval_every == 0:
            val_loss, val_metrics = evaluate_one_batch(model, val_loader, model_cfg, train_cfg)
            print(f"eval step={step} loss={val_loss:.4f} lm={val_metrics['lm']:.4f}")
        if step % args.save_every == 0:
            save_checkpoint(output_dir / "latest.pt", model, optimizer, step, model_cfg, train_cfg)
            save_checkpoint(output_dir / f"step_{step}.pt", model, optimizer, step, model_cfg, train_cfg)

    save_checkpoint(output_dir / "latest.pt", model, optimizer, args.max_steps, model_cfg, train_cfg)
    print("Training complete.")


if __name__ == "__main__":
    main()
