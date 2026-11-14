from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import torch

from run_first_experiment import calculate_budget, load_split_tensors, train_model
from fdt_rlm.data.interleaving import bucket_sequence_counts, interleave_remaining_indices


ROOT = Path(__file__).resolve().parents[1]
BASE = {
    "transformer": ROOT / "runs" / "full_transformer" / "best.pt",
    "fdt": ROOT / "runs" / "full_fdt" / "best.pt",
}


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def source_checkpoint(model_type: str, milestone: int) -> tuple[Path, bool]:
    if milestone == 25_000_000:
        return BASE[model_type], True
    previous = {50_000_000: 25, 100_000_000: 50}[milestone]
    return ROOT / "runs" / f"staged_{previous}m" / model_type / "latest.pt", False


def seed_stage(
    source: Path,
    destination: Path,
    reset_dataset: bool,
    learning_rate: float,
    save: bool = True,
) -> dict:
    checkpoint = safe_load(source)
    if reset_dataset:
        checkpoint["sequence_index"] = 0
    for group in checkpoint.get("optimizer_state_dict", {}).get("param_groups", []):
        group["lr"] = learning_rate
        group["initial_lr"] = learning_rate
    scheduler = checkpoint.get("scheduler_state_dict", {})
    scheduler["base_lrs"] = [learning_rate for _ in scheduler.get("base_lrs", [learning_rate])]
    scheduler["_last_lr"] = [learning_rate for _ in scheduler.get("_last_lr", [learning_rate])]
    try:
        source_label = str(source.relative_to(ROOT))
    except ValueError:
        source_label = str(source)
    checkpoint["staged_source_checkpoint"] = source_label
    checkpoint["staged_dataset_reset"] = bool(reset_dataset)
    if save:
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, destination)
    return checkpoint


def parse_milestone(value: str) -> int:
    normalized = value.lower().replace("m", "")
    amount = int(normalized)
    if amount not in {25, 50, 100}:
        raise argparse.ArgumentTypeError("milestone must be 25, 50, or 100")
    return amount * 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue untouched base checkpoints to a cumulative token milestone")
    parser.add_argument("--model", choices=["transformer", "fdt"], required=True)
    parser.add_argument("--milestone", type=parse_milestone, required=True)
    parser.add_argument("--dataset_dir", default=str(ROOT / "prepared_data" / "fdt_rlm_100m"))
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizers" / "fdt_bpe"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--source_checkpoint", default="")
    parser.add_argument(
        "--interleave_remaining_sources",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Interleave only source rows not consumed by the source checkpoint.",
    )
    parser.add_argument(
        "--interleave_consumed_prefix",
        type=int,
        default=-1,
        help="Shared original-row prefix to exclude; must cover the checkpoint cursor.",
    )
    parser.add_argument(
        "--target_additional_steps",
        type=int,
        default=0,
        help="Bound this recovery run to N optimizer steps from its source checkpoint.",
    )
    parser.add_argument("--recovery_index", default="")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()

    milestone_m = args.milestone // 1_000_000
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else ROOT / "runs" / f"staged_{milestone_m}m" / args.model
    )
    latest = output_dir / "latest.pt"
    created_from_source = False
    if latest.exists():
        checkpoint = safe_load(latest)
    else:
        if args.source_checkpoint:
            source = Path(args.source_checkpoint)
            reset_dataset = False
        else:
            source, reset_dataset = source_checkpoint(args.model, args.milestone)
        if not source.exists():
            raise FileNotFoundError(source)
        checkpoint = seed_stage(
            source,
            latest,
            reset_dataset,
            args.learning_rate,
            save=not args.plan_only,
        )
        created_from_source = True

    saved_interleave = checkpoint.get("interleave_state")
    if saved_interleave is None:
        saved_interleave = checkpoint.get("config", {}).get("budget", {}).get("interleave")
    if args.interleave_remaining_sources and saved_interleave is None:
        if not created_from_source:
            raise RuntimeError("Cannot change an existing run from sequential to interleaved indexing")
        checkpoint_cursor = int(checkpoint.get("sequence_index", 0))
        consumed_prefix = (
            checkpoint_cursor
            if args.interleave_consumed_prefix < 0
            else args.interleave_consumed_prefix
        )
        if consumed_prefix < checkpoint_cursor:
            raise ValueError("interleave_consumed_prefix would replay rows used by the source checkpoint")
        saved_interleave = {
            "mode": "weighted_deficit_remaining_v1",
            "source_checkpoint_sequence_index": checkpoint_cursor,
            "original_consumed_sequence_index": consumed_prefix,
            "source_optimizer_step": int(checkpoint.get("optimizer_step", 0)),
            "source_tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        }
        checkpoint["sequence_index"] = 0
        checkpoint["interleave_state"] = saved_interleave
        if not args.plan_only:
            torch.save(checkpoint, latest)
    use_interleave = saved_interleave is not None
    if (
        use_interleave
        and args.interleave_consumed_prefix >= 0
        and int(saved_interleave["original_consumed_sequence_index"])
        != args.interleave_consumed_prefix
    ):
        raise ValueError("interleave_consumed_prefix does not match the saved recovery order")

    current_tokens = int(checkpoint.get("tokens_seen", 0))
    if current_tokens >= args.milestone:
        print(json.dumps({"status": "already_complete", "tokens_seen": current_tokens, "output_dir": str(output_dir)}, indent=2))
        return

    dataset_dir = Path(args.dataset_dir)
    tokenizer = Path(args.tokenizer)
    budget = calculate_budget(dataset_dir, tokenizer, args.batch_size, args.grad_accum_steps, 512)
    interleave_order = None
    if use_interleave:
        manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
        source_counts = bucket_sequence_counts(manifest, split="train")
        consumed_prefix = int(saved_interleave["original_consumed_sequence_index"])
        recovery_index_path = (
            Path(args.recovery_index)
            if args.recovery_index
            else Path(saved_interleave["recovery_index"])
            if saved_interleave.get("recovery_index")
            else None
        )
        if recovery_index_path is not None:
            index_payload = safe_load(recovery_index_path)
            interleave_order = index_payload["indices"].long().contiguous()
            if interleave_order.unique().numel() != interleave_order.numel():
                raise ValueError("recovery_index contains duplicate rows")
            if interleave_order.numel() and int(interleave_order.min()) < consumed_prefix:
                raise ValueError("recovery_index replays a consumed row")
            max_dataset_index = sum(source_counts.values()) - 1
            if interleave_order.numel() and int(interleave_order.max()) > max_dataset_index:
                raise ValueError("recovery_index exceeds the prepared training split")
            index_hash = hashlib.sha256(recovery_index_path.read_bytes()).hexdigest()
        else:
            interleave_order = interleave_remaining_indices(source_counts, consumed_prefix)
            index_hash = None
        saved_interleave = {
            **saved_interleave,
            "source_sequence_counts": source_counts,
            "available_sequences": int(interleave_order.numel()),
            "recovery_index": str(recovery_index_path) if recovery_index_path else None,
            "recovery_index_sha256": index_hash,
        }
        budget["interleave"] = saved_interleave
    tokens_per_step = int(args.batch_size * args.grad_accum_steps * 512)
    milestone_steps = math.ceil((args.milestone - current_tokens) / tokens_per_step)
    start_step = int(checkpoint.get("optimizer_step", 0))
    start_sequence = int(checkpoint.get("sequence_index", 0))
    if args.target_additional_steps > 0:
        if not use_interleave:
            raise ValueError("target_additional_steps requires an interleaved recovery run")
        bounded_target_step = (
            int(saved_interleave["source_optimizer_step"])
            + args.target_additional_steps
        )
        additional_steps = min(milestone_steps, max(0, bounded_target_step - start_step))
        if additional_steps == 0:
            print(json.dumps({
                "status": "bounded_run_already_complete",
                "optimizer_step": start_step,
                "tokens_seen": current_tokens,
                "output_dir": str(output_dir),
            }, indent=2))
            return
    else:
        additional_steps = milestone_steps
    target_step = start_step + additional_steps
    max_sequences = start_sequence + additional_steps * args.batch_size * args.grad_accum_steps
    available_sequences = (
        int(interleave_order.numel())
        if interleave_order is not None
        else int(budget["train_sequences"])
    )
    if max_sequences > available_sequences:
        raise RuntimeError("milestone would exceed the prepared training split")
    if args.plan_only:
        groups = checkpoint.get("optimizer_state_dict", {}).get("param_groups", [])
        scheduler = checkpoint.get("scheduler_state_dict", {})
        print(json.dumps({
            "model": args.model,
            "starting_tokens": current_tokens,
            "target_tokens": args.milestone,
            "additional_steps": additional_steps,
            "target_additional_steps": args.target_additional_steps,
            "optimizer_lrs": [group.get("lr") for group in groups],
            "scheduler_base_lrs": scheduler.get("base_lrs"),
            "dataset_start_sequence": start_sequence,
            "dataset_stop_sequence": max_sequences,
            "interleave": saved_interleave,
        }, indent=2))
        return

    # Namespace fields consumed by the established training loop.
    args.resume = True
    args.additional_optimizer_steps = 0
    args.dim = 384
    args.n_layers = 6
    args.n_heads = 6
    args.mlp_ratio = 4
    args.max_seq_len = 512
    args.dropout = 0.1
    args.num_anchors = 128
    args.top_k = 8
    args.routing_type = "cosine"
    args.cosine_temperature = 0.25
    args.aggregation_impl = "dense_gather_first"
    args.diversity_margin = 0.20
    args.diversity_max_weight = 2.0
    args.weight_decay = 0.1
    args.grad_clip = 1.0
    args.warmup_steps = 100
    args.scheduler_mode = "constant"
    args.log_every = 20
    args.save_every = 2500
    args.eval_every = 10**9
    args.save_generations = False
    args.seed = 42
    args.lm_weight = 1.0
    args.noise_weight = 0.0
    args.cluster_weight = 0.01
    args.diversity_weight = 0.10
    args.diversity_every = 4
    args.usage_weight = 0.005
    args.entropy_weight = 0.003
    args.entropy_target_ratio_start = 0.65
    args.entropy_target_ratio_end = 0.45
    args.entropy_over_weight = 0.20

    train_ids, train_mask = load_split_tensors(dataset_dir, "train")
    if interleave_order is not None:
        train_ids = train_ids.index_select(0, interleave_order)
        train_mask = train_mask.index_select(0, interleave_order)
    val_ids, val_mask = load_split_tensors(dataset_dir, "validation")
    summary = train_model(
        args=args,
        phase="staged",
        model_type=args.model,
        output_dir=output_dir,
        budget=budget,
        train_ids=train_ids,
        train_mask=train_mask,
        val_ids=val_ids,
        val_mask=val_mask,
        total_optimizer_steps=target_step,
        max_sequences=max_sequences,
        preflight_resume_probe=False,
    )
    summary["target_cumulative_tokens"] = args.milestone
    summary["starting_tokens"] = current_tokens
    summary["token_overshoot"] = int(summary["tokens_seen"]) - args.milestone
    summary["target_additional_steps"] = args.target_additional_steps
    summary["bounded_recovery_run"] = args.target_additional_steps > 0
    (output_dir / "stage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary.get(key) for key in ("status", "model_type", "starting_tokens", "tokens_seen", "token_overshoot", "final_val_loss", "tokens_per_second", "wall_time_seconds")}, indent=2))


if __name__ == "__main__":
    main()
