from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.data.filters import normalize_code_text, normalize_web_text  # noqa: E402
from fdt_rlm.lexical_pointer import PointerLoss  # noqa: E402
from fdt_rlm.models import build_model, causal_lm_loss  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402
from prepare_balanced_novel_pilot import (  # noqa: E402
    SOURCE_REVISIONS,
    compact_json_documents,
    retrieval_documents,
    stream_hf_texts,
    stream_prompted_python_texts,
)


MIX = {
    "fineweb_edu_dedup": 0.30,
    "cosmopedia": 0.15,
    "tinystories": 0.10,
    "wikipedia": 0.25,
    "retrieval_long": 0.15,
    "json_compact": 0.03,
    "python_prompted": 0.02,
}


def require_c_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"Only C: paths are permitted: {resolved}")
    return resolved


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def document_digest(ids: list[int]) -> str:
    tensor = torch.tensor(ids, dtype=torch.int32)
    return hashlib.blake2b(tensor.numpy().tobytes(), digest_size=16).hexdigest()


def source_factory(name: str, seed: int):
    if name in {"fineweb_edu_dedup", "cosmopedia", "tinystories", "wikipedia"}:
        return stream_hf_texts(name, seed)
    if name == "retrieval_long":
        return retrieval_documents(seed)
    if name == "json_compact":
        return compact_json_documents(seed)
    if name == "python_prompted":
        return stream_prompted_python_texts(seed)
    raise ValueError(name)


class MixedTokenBlocks(IterableDataset):
    def __init__(self, tokenizer, seq_len: int, seed: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        schedule = []
        for name, weight in MIX.items():
            schedule.extend([name] * int(round(weight * 100)))
        random.Random(self.seed).shuffle(schedule)
        epochs = {name: 0 for name in MIX}
        streams = {name: iter(source_factory(name, self.seed + index * 101)) for index, name in enumerate(MIX)}
        seen: set[str] = set()
        cursor = 0
        eos = int(self.tokenizer.eos_token_id)
        while True:
            name = schedule[cursor % len(schedule)]
            cursor += 1
            try:
                raw, _ = next(streams[name])
            except StopIteration:
                epochs[name] += 1
                streams[name] = iter(source_factory(name, self.seed + epochs[name] * 100_003))
                continue
            text = normalize_code_text(raw) if name == "python_prompted" else normalize_web_text(raw)
            if not text or "\x00" in text:
                continue
            ids = [int(value) for value in self.tokenizer.encode(text, add_special_tokens=False)]
            if len(ids) < 32:
                continue
            digest = document_digest(ids)
            if digest in seen:
                continue
            seen.add(digest)
            ids.append(eos)
            for start in range(0, len(ids) - self.seq_len + 1, self.seq_len):
                block = torch.tensor(ids[start : start + self.seq_len], dtype=torch.long)
                yield {"input_ids": block, "attention_mask": torch.ones_like(block)}


def atomic_model_save(path: Path, model, config: ModelConfig, metadata: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": vars(config),
        "train_config": metadata,
        "stage_status": metadata["stage_status"],
        "optimizer_step": metadata["optimizer_step"],
        "tokens_seen": metadata["tokens_seen"],
        "optimizer_state_included": False,
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_recovery_save(path: Path, model, optimizer, config: ModelConfig, metadata: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": vars(config),
        "train_config": metadata,
        "stage_status": metadata["stage_status"],
        "optimizer_step": metadata["optimizer_step"],
        "tokens_seen": metadata["tokens_seen"],
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        "optimizer_state_included": True,
        "data_resume_note": "Streaming sources restart with a new deterministic continuation seed after process loss.",
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_validation(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return payload["input_ids"].long(), payload["attention_mask"].long()


@torch.inference_mode()
def validation_loss(model, ids, masks, device: torch.device, batches: int = 4) -> float:
    model.eval()
    values = []
    for index in range(min(batches, ids.size(0))):
        x = ids[index : index + 1].to(device)
        mask = masks[index : index + 1].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(x, attention_mask=mask)
            value = causal_lm_loss(output["logits"], x, mask, model.config.pad_token_id)
        values.append(float(value))
    model.train()
    return sum(values) / max(len(values), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unattended FDT Dual Memory pretraining")
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--model-type",
        default="fdt_v3_anchor_indexed_dual_memory",
        choices=("fdt_v3_dual_memory", "fdt_v3_anchor_indexed_dual_memory"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--target-additional-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2.5e-9)
    parser.add_argument("--pointer-loss-weight", type=float, default=0.05)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--initial-save-step", type=int, default=25)
    parser.add_argument("--recovery-every", type=int, default=1_000)
    parser.add_argument("--initial-recovery-step", type=int, default=100)
    parser.add_argument("--milestone-tokens", type=int, default=100_000_000)
    parser.add_argument("--validation-shard", type=Path, required=True)
    parser.add_argument("--overfit-check-every", type=int, default=500)
    parser.add_argument("--overfit-relative-regression", type=float, default=0.005)
    parser.add_argument("--overfit-patience", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--min-free-gib", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    parent_path = require_c_path(args.parent)
    resume_path = require_c_path(args.resume) if args.resume else None
    output = require_c_path(args.output_dir)
    tokenizer_path = require_c_path(args.tokenizer)
    validation_path = require_c_path(args.validation_shard)
    launcher_files = {"stdout.log", "stderr.log", "train.pid", "active_logs.json", "monitor.pid"}
    if output.exists():
        unexpected_files = [path.name for path in output.iterdir() if path.name not in launcher_files]
        if unexpected_files:
            raise FileExistsError(f"Output path is not fresh: {output}; found={unexpected_files}")
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    source_checkpoint = torch.load(
        resume_path or parent_path,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    config = ModelConfig(**source_checkpoint["model_config"])
    if resume_path is None:
        if config.model_type != "fdt_v3":
            raise ValueError(f"Expected an fdt_v3 parent, got {config.model_type}")
        config.model_type = args.model_type
        config.exact_pointer_loss_weight = args.pointer_loss_weight
        config.dropout = 0.0
    elif config.model_type != args.model_type:
        raise ValueError(f"Expected a {args.model_type} resume checkpoint, got {config.model_type}")
    tokenizer = load_tokenizer(str(tokenizer_path))
    if len(tokenizer) != config.vocab_size:
        raise ValueError("Tokenizer and parent vocabulary differ")
    model = build_model(config)
    if resume_path is None:
        missing, unexpected = model.load_state_dict(source_checkpoint["model_state_dict"], strict=False)
        if unexpected or not missing or any(not name.startswith("exact_pointer.") for name in missing):
            raise ValueError(f"Unsafe parent migration: missing={missing}, unexpected={unexpected}")
        parent_tokens = int(source_checkpoint.get("tokens_seen", 0))
        start_step = 0
        resumed_additional_tokens = 0
    else:
        if not source_checkpoint.get("optimizer_state_included", False):
            raise ValueError("Resume checkpoint does not contain optimizer state")
        model.load_state_dict(source_checkpoint["model_state_dict"], strict=True)
        prior_metadata = source_checkpoint.get("train_config", {})
        parent_tokens = int(prior_metadata.get("parent_tokens", 0))
        start_step = int(source_checkpoint.get("optimizer_step", prior_metadata.get("optimizer_step", 0)))
        resumed_additional_tokens = int(prior_metadata.get("additional_tokens", 0))

    frozen_routing = []
    for name, parameter in model.named_parameters():
        if ".anchor." in name:
            parameter.requires_grad_(False)
            frozen_routing.append(name)

    device = torch.device(args.device)
    model.to(device=device, dtype=torch.bfloat16).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    if resume_path is not None:
        optimizer.load_state_dict(source_checkpoint["optimizer_state_dict"])
        torch.set_rng_state(source_checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(source_checkpoint["cuda_rng_state_all"])
    del source_checkpoint
    continuation_seed = args.seed + start_step * 100_003
    dataset = MixedTokenBlocks(tokenizer, config.max_seq_len, continuation_seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    iterator = iter(loader)
    tokens_per_step = args.batch_size * args.grad_accum * config.max_seq_len
    total_steps = math.ceil(args.target_additional_tokens / tokens_per_step)
    log_path = output / "training_log.jsonl"
    stop_path = output / "STOP_REQUESTED"
    validation_ids, validation_masks = load_validation(validation_path)
    metadata = {
        "stage": "fdt_dual_memory_v1_unattended_1b",
        "stage_status": "RUNNING",
        "parent_checkpoint": str(parent_path),
        "resume_checkpoint": str(resume_path) if resume_path else None,
        "resume_from_step": start_step,
        "stream_continuation_seed": continuation_seed,
        "parent_tokens": parent_tokens,
        "optimizer_step": start_step,
        "tokens_seen": parent_tokens + resumed_additional_tokens,
        "additional_tokens": resumed_additional_tokens,
        "target_additional_tokens": args.target_additional_tokens,
        "source_mix": MIX,
        "source_revisions": SOURCE_REVISIONS,
        "learning_rate": args.learning_rate,
        "pointer_loss_weight": args.pointer_loss_weight,
        "frozen_routing_parameter_count": len(frozen_routing),
        "precision": "bf16_training_fp32_official_evaluation",
        "evaluation_during_training": False,
        "repetition_policy": "training filters plus output-only generation penalty; exact cursor exempt",
        "checkpoint_policy": "atomic model snapshots plus periodic optimizer-bearing recovery",
        "milestone_tokens": args.milestone_tokens,
        "validation_shard": str(validation_path),
        "overfit_policy": {
            "check_every": args.overfit_check_every,
            "relative_regression": args.overfit_relative_regression,
            "patience": args.overfit_patience,
            "stop_only_when_validation_worsens_while_training_loss_improves": True,
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    append_jsonl(log_path, {"event": "start", **metadata})

    optimizer.zero_grad(set_to_none=True)
    best_validation = validation_loss(model, validation_ids, validation_masks, device)
    overfit_strikes = 0
    best_training_loss = float("inf")
    append_jsonl(log_path, {"event": "validation_baseline", "validation_loss": best_validation})
    started = time.perf_counter()
    last_milestone = resumed_additional_tokens // max(args.milestone_tokens, 1)
    for step in range(start_step + 1, total_steps + 1):
        lm_sum = 0.0
        pointer_sum = 0.0
        entropy_values = []
        active = 0
        for micro_step in range(args.grad_accum):
            batch = next(iterator)
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_row = model(ids, attention_mask=mask)
                lm = causal_lm_loss(output_row["logits"], ids, mask, config.pad_token_id)
                pointer: PointerLoss = model.exact_memory_loss(output_row["hidden"], ids, ids, mask)
                loss = (lm + args.pointer_loss_weight * pointer.loss) / args.grad_accum
            batch_entropy_values = [stats.entropy.detach().float() for stats in output_row["anchor_stats"]]
            entropy_values.extend(batch_entropy_values)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            loss.backward()
            lm_value = float(lm.detach())
            pointer_value = float(pointer.loss.detach())
            batch_entropy = (
                float(torch.stack(batch_entropy_values).mean()) / math.log(max(config.top_k, 2))
                if batch_entropy_values
                else 0.0
            )
            active += int(mask.sum())
            process_tokens = metadata["additional_tokens"] - resumed_additional_tokens + active
            live_throughput = process_tokens / max(time.perf_counter() - started, 1e-9)
            remaining_tokens = max(args.target_additional_tokens - metadata["additional_tokens"] - active, 0)
            atomic_json(
                output / "live_metrics.json",
                {
                    "loss": lm_value + args.pointer_loss_weight * pointer_value,
                    "entropy_normalized": batch_entropy,
                    "step": step,
                    "micro_step": micro_step + 1,
                    "remaining_seconds": remaining_tokens / max(live_throughput, 1e-9),
                },
            )
            lm_sum += lm_value
            pointer_sum += pointer_value
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Non-finite gradient at step {step}")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metadata["optimizer_step"] = step
        metadata["additional_tokens"] += active
        metadata["tokens_seen"] = parent_tokens + metadata["additional_tokens"]
        current_training_loss = lm_sum / args.grad_accum
        current_pointer_loss = pointer_sum / args.grad_accum
        current_total_loss = current_training_loss + args.pointer_loss_weight * current_pointer_loss
        normalized_entropy = (
            float(torch.stack(entropy_values).mean()) / math.log(max(config.top_k, 2))
            if entropy_values
            else 0.0
        )
        training_improved = current_training_loss < best_training_loss
        best_training_loss = min(best_training_loss, current_training_loss)

        process_tokens = metadata["additional_tokens"] - resumed_additional_tokens
        throughput = process_tokens / max(time.perf_counter() - started, 1e-9)
        remaining_seconds = max(args.target_additional_tokens - metadata["additional_tokens"], 0) / max(throughput, 1e-9)
        if step % args.log_every == 0:
            free_gib = shutil.disk_usage("C:\\").free / (1024**3)
            row = {
                "event": "train",
                "step": step,
                "tokens_seen": metadata["tokens_seen"],
                "additional_tokens": metadata["additional_tokens"],
                "loss": current_total_loss,
                "lm_loss": current_training_loss,
                "pointer_loss": current_pointer_loss,
                "entropy_normalized": normalized_entropy,
                "grad_norm": float(grad_norm),
                "tokens_per_second": throughput,
                "remaining_seconds": remaining_seconds,
                "c_free_gib": free_gib,
            }
            append_jsonl(log_path, row)
            print(json.dumps(row), flush=True)
            if free_gib < args.min_free_gib:
                raise OSError(f"C: free space below safety floor: {free_gib:.2f} GiB")

        overfit_detected = False
        if step % args.overfit_check_every == 0:
            heldout = validation_loss(model, validation_ids, validation_masks, device)
            regressed = heldout > best_validation * (1.0 + args.overfit_relative_regression)
            overfit_strikes = overfit_strikes + 1 if regressed and training_improved else 0
            best_validation = min(best_validation, heldout)
            append_jsonl(log_path, {
                "event": "overfit_check",
                "step": step,
                "validation_loss": heldout,
                "best_validation_loss": best_validation,
                "training_loss": current_training_loss,
                "strikes": overfit_strikes,
            })
            overfit_detected = overfit_strikes >= args.overfit_patience

        should_stop = stop_path.exists() or overfit_detected
        milestone_index = metadata["additional_tokens"] // max(args.milestone_tokens, 1)
        milestone_due = milestone_index > last_milestone
        model_checkpoint_due = (
            step == args.initial_save_step
            or step % args.save_every == 0
            or milestone_due
            or should_stop
            or step == total_steps
        )
        if model_checkpoint_due:
            metadata["stage_status"] = "PAUSED" if should_stop else "COMPLETE" if step == total_steps else "RUNNING"
            if overfit_detected:
                metadata["stage_status"] = "PAUSED_OVERFIT"
            atomic_model_save(output / "latest.pt", model, config, metadata)
            append_jsonl(log_path, {"event": "checkpoint", **metadata})
            if milestone_due:
                milestone_path = output / f"checkpoint_{milestone_index * args.milestone_tokens:010d}_tokens.pt"
                atomic_model_save(milestone_path, model, config, metadata)
                append_jsonl(log_path, {"event": "milestone_checkpoint", "path": str(milestone_path), **metadata})
                last_milestone = milestone_index
        recovery_due = step == args.initial_recovery_step or step % args.recovery_every == 0 or milestone_due
        if recovery_due or should_stop or step == total_steps:
            atomic_recovery_save(output / "latest_recovery.pt", model, optimizer, config, metadata)
            append_jsonl(log_path, {"event": "recovery_checkpoint", **metadata})
        if should_stop:
            break

    print(json.dumps({"event": "exit", **metadata}), flush=True)


if __name__ == "__main__":
    main()
