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
from select_capability_completion_v16 import load_master, load_validation, require_c_path
from train_capability_completion_v15 import append_jsonl, atomic_save, strict_smoke, verify_sha256
from train_chat_sft import evaluate_chat, masked_chat_losses


STAGE = "capability_completion_v16_retention"


def unexpected_output_files(output_dir: Path) -> list[str]:
    launcher_files = {"active_logs.json", "train.pid"}
    return [
        path.name
        for path in output_dir.iterdir()
        if path.name not in launcher_files
        and not re.fullmatch(r"train\.\d{8}_\d{6}\.(stdout|stderr)\.log", path.name)
    ]


def validate_resume_checkpoint(
    checkpoint: dict,
    output_dir: Path,
    dataset_dir: Path,
    selection_artifact: Path,
    profile: str,
) -> None:
    settings = checkpoint.get("train_config", {})
    if settings.get("stage") != STAGE:
        raise ValueError("Resume checkpoint is not a V16 retention-completion stage")
    if checkpoint.get("stage_status") != "PAUSED":
        raise ValueError("V16 resume requires a PAUSED checkpoint")
    identities = {
        "output_dir": output_dir,
        "dataset_dir": dataset_dir,
        "selection_artifact": selection_artifact,
    }
    for key, expected in identities.items():
        if Path(settings.get(key, "")).resolve() != expected.resolve():
            raise ValueError(f"V16 resume {key} identity mismatch")
    if settings.get("selected_profile") != profile:
        raise ValueError("V16 resume profile mismatch")
    if int(checkpoint.get("optimizer_step", -1)) < 0 or int(checkpoint.get("sample_cursor", -1)) < 0:
        raise ValueError("V16 resume checkpoint is missing step/cursor state")


def save_checkpoint(path, model, optimizer, config, settings, step, cursor, tokens_seen, status):
    atomic_save(
        path,
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(config),
            "train_config": settings,
            "optimizer_step": step,
            "sample_cursor": cursor,
            "tokens_seen": tokens_seen,
            "stage_status": status,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded non-conversational V16 retention completion pilot.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--selection-artifact", type=Path, required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--broad-dataset-dir",
        type=Path,
        default=ROOT / "prepared_data" / "v3_broad_additional_300m_combined_v2_24k_ctx2048",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.7)
    parser.add_argument("--smoke-step", type=int, default=100)
    parser.add_argument("--max-broad-relative-regression", type=float, default=0.003)
    parser.add_argument("--max-prefix-repetition-increase", type=float, default=0.01)
    parser.add_argument("--max-loop-free-decrease", type=float, default=0.05)
    parser.add_argument("--max-capability-loss-regression", type=float, default=0.0)
    parser.add_argument("--max-dead-anchor-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for path in (
        args.checkpoint,
        args.dataset_dir,
        args.selection_artifact,
        args.output_dir,
        args.broad_dataset_dir,
    ):
        require_c_path(path)
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        unexpected = unexpected_output_files(output_dir)
        if unexpected:
            raise FileExistsError(f"Dedicated V16 output contains non-launcher files: {unexpected}")
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hash = verify_sha256(
        args.checkpoint, args.expected_checkpoint_sha256, "Promoted 520M checkpoint"
    )
    manifest_hash = verify_sha256(
        args.dataset_dir / "manifest.json", args.expected_manifest_sha256, "V16 manifest"
    )
    selection_hash = verify_sha256(
        args.selection_artifact, args.expected_selection_sha256, "V16 selection artifact"
    )
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(args.selection_artifact.read_text(encoding="utf-8"))
    if selection.get("status") != "SELECTED" or not selection.get("selected_profile"):
        raise ValueError("V16 selection artifact has no eligible profile")
    if selection.get("checkpoint_sha256", "").upper() != checkpoint_hash.upper():
        raise ValueError("V16 selection parent hash mismatch")
    if selection.get("manifest_sha256", "").upper() != manifest_hash.upper():
        raise ValueError("V16 selection manifest hash mismatch")
    selected_profile = str(selection["selected_profile"])
    profile = manifest["profiles"][selected_profile]
    prompt_lm_weight = float(selection["selected_prompt_lm_weight"])
    if prompt_lm_weight != float(profile["prompt_lm_weight"]):
        raise ValueError("V16 selected prompt-retention weight mismatch")
    if tuple(manifest["train"]["category_names"]) != CATEGORY_NAMES:
        raise ValueError("V16 category order mismatch")

    checkpoint = safe_torch_load(args.checkpoint, "cpu")
    if args.resume:
        validate_resume_checkpoint(
            checkpoint, output_dir, args.dataset_dir, args.selection_artifact, selected_profile
        )
    config = ModelConfig(**checkpoint["model_config"])
    tokenizer_name = checkpoint.get("train_config", {}).get("tokenizer_name", "") or "gpt2"
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    tokens_seen = int(checkpoint.get("tokens_seen", 0))
    resume_step = int(checkpoint.get("optimizer_step", 0)) if args.resume else 0
    resume_cursor = int(checkpoint.get("sample_cursor", 0)) if args.resume else 0
    resume_settings = dict(checkpoint.get("train_config", {})) if args.resume else {}
    device = torch.device(args.device)
    model.to(device)
    tokenizer = strict.load_tokenizer(tokenizer_name)

    frozen = []
    for name, parameter in model.named_parameters():
        if ".anchor." in name:
            parameter.requires_grad_(False)
            frozen.append(name)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
    del checkpoint

    train_ids, train_labels, train_masks, category_ids = load_master(args.dataset_dir)
    val_ids, val_labels, val_masks = load_validation(args.dataset_dir)
    broad_ids, broad_masks = strict.split_rows(args.broad_dataset_dir, "validation")
    broad_manifest = json.loads((args.broad_dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    broad_ranges = strict.bucket_ranges(broad_manifest, "validation")
    capability_payload = {"input_ids": val_ids, "labels": val_labels, "attention_mask": val_masks}
    capability_ranges = strict_v3.capability_ranges(manifest)
    order = profile_indices(category_ids, profile, args.seed + 100 + len(selected_profile))
    expected_index_hash = profile.get("index_sha256")
    actual_index_hash = __import__("prepare_capability_completion_v16").token_hash(
        order.to(dtype=torch.int32)
    )
    if expected_index_hash != actual_index_hash:
        raise ValueError("V16 selected profile index hash mismatch")

    settings = resume_settings or {
        "stage": STAGE,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "parent_tokens": tokens_seen,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest_sha256": manifest_hash,
        "selection_artifact": str(args.selection_artifact.resolve()),
        "selection_sha256": selection_hash,
        "output_dir": str(output_dir),
        "tokenizer_name": str(tokenizer_name),
        "selected_profile": selected_profile,
        "selected_profile_index_sha256": actual_index_hash,
        "selected_counts": profile["counts"],
        "learning_rate": args.lr,
        "lr_schedule": "constant",
        "lr_warmup_steps": 0,
        "batch_size": args.batch_size,
        "prompt_lm_weight": prompt_lm_weight,
        "freeze_routing": True,
        "frozen_parameter_names": len(frozen),
        "conversational_sft": False,
        "completion_span_supervision": True,
        "seed": args.seed,
    }
    if args.resume:
        if str(settings.get("manifest_sha256", "")).upper() != manifest_hash.upper():
            raise ValueError("V16 resume manifest hash mismatch")
        if str(settings.get("selection_sha256", "")).upper() != selection_hash.upper():
            raise ValueError("V16 resume selection hash mismatch")
        if int(settings.get("batch_size", -1)) != args.batch_size:
            raise ValueError("V16 resume batch size mismatch")
        if float(settings.get("learning_rate", -1.0)) != args.lr:
            raise ValueError("V16 resume learning-rate mismatch")
        if int(settings.get("seed", args.seed)) != args.seed:
            raise ValueError("V16 resume data-order seed mismatch")
        settings["resume_from_checkpoint"] = str(args.checkpoint.resolve())
        settings["resume_from_sha256"] = checkpoint_hash
        settings["resume_step"] = resume_step
        settings["resume_cursor"] = resume_cursor

    (output_dir / "train_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output_dir / "model_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    log_path = output_dir / "training_log.jsonl"
    latest_path = output_dir / "latest.pt"
    stop_path = output_dir / "STOP_REQUESTED"

    baseline_chat = None
    baseline_smoke = None
    if args.resume:
        if resume_step < args.smoke_step:
            raise ValueError("V16 resume before the predeclared smoke gate is unsupported")
        append_jsonl(log_path, {
            "event": "resume",
            "step": resume_step,
            "sample_cursor": resume_cursor,
            "tokens_seen": tokens_seen,
            "checkpoint_sha256": checkpoint_hash,
        })
    else:
        baseline_chat = evaluate_chat(model, val_ids, val_labels, val_masks, args.batch_size, device)
        baseline_smoke = strict_smoke(
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
        append_jsonl(log_path, {
            "event": "baseline",
            "tokens_seen": tokens_seen,
            "selected_profile": selected_profile,
            "prompt_lm_weight": prompt_lm_weight,
            "capability": baseline_chat,
            **baseline_smoke,
        })

    cursor = resume_cursor
    step = resume_step
    status = "COMPLETE"
    started = time.perf_counter()
    token_start = tokens_seen
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    model.train()
    while cursor < len(order):
        indices = order[cursor : cursor + args.batch_size]
        x = train_ids.index_select(0, indices).long().to(device, non_blocking=True)
        labels = train_labels.index_select(0, indices).long().to(device, non_blocking=True)
        mask = train_masks.index_select(0, indices).long().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x, attention_mask=mask)["logits"]
            completion_loss, prompt_loss = masked_chat_losses(logits, x, labels, mask)
            loss = completion_loss + prompt_lm_weight * prompt_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite V16 loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite V16 gradient")
        optimizer.step()
        cursor += len(indices)
        step += 1
        tokens_seen += int(mask.sum().item())
        if step % 25 == 0:
            append_jsonl(log_path, {
                "event": "train",
                "step": step,
                "sample_cursor": cursor,
                "tokens_seen": tokens_seen,
                "loss": float(loss),
                "completion_loss": float(completion_loss),
                "prompt_loss": float(prompt_loss),
                "grad_norm": float(grad_norm),
                "tokens_per_sec": (tokens_seen - token_start) / max(time.perf_counter() - started, 1e-9),
            })
        if stop_path.exists():
            status = "PAUSED"
            break
        if step == args.smoke_step:
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
            smoke_chat = evaluate_chat(model, val_ids, val_labels, val_masks, args.batch_size, device)
            broad_relative = smoke["heldout"]["loss"] / baseline_smoke["heldout"]["loss"] - 1.0
            dead = smoke["heldout"]["routing"].get("dead_anchor_ratio", 0.0)
            repetition_delta = (
                smoke["prefix_generation"]["mean_repetition_rate"]
                - baseline_smoke["prefix_generation"]["mean_repetition_rate"]
            )
            loop_free_delta = (
                smoke["prefix_generation"]["loop_free"]["rate"]
                - baseline_smoke["prefix_generation"]["loop_free"]["rate"]
            )
            capability_delta = smoke_chat["loss"] - baseline_chat["loss"]
            healthy = (
                math.isfinite(smoke["heldout"]["loss"])
                and broad_relative <= args.max_broad_relative_regression
                and repetition_delta <= args.max_prefix_repetition_increase
                and loop_free_delta >= -args.max_loop_free_decrease
                and capability_delta <= args.max_capability_loss_regression
                and dead <= args.max_dead_anchor_ratio
            )
            append_jsonl(log_path, {
                "event": "smoke_gate",
                "step": step,
                "broad_relative_delta": broad_relative,
                "dead_anchor_ratio": dead,
                "prefix_repetition_delta": repetition_delta,
                "prefix_loop_free_delta": loop_free_delta,
                "capability_loss_delta": capability_delta,
                "capability": smoke_chat,
                "healthy": healthy,
                **smoke,
            })
            if not healthy:
                status = "ABORTED_SMOKE_GATE"
                break
            model.train()

    model.eval()
    final_chat = evaluate_chat(model, val_ids, val_labels, val_masks, args.batch_size, device)
    save_checkpoint(latest_path, model, optimizer, config, settings, step, cursor, tokens_seen, status)
    event = "complete" if status == "COMPLETE" else "paused" if status == "PAUSED" else "aborted"
    final = {
        "event": event,
        "status": status,
        "step": step,
        "sample_cursor": cursor,
        "tokens_seen": tokens_seen,
        "selected_profile": selected_profile,
        "prompt_lm_weight": prompt_lm_weight,
        "capability": final_chat,
    }
    append_jsonl(log_path, final)
    print(json.dumps(final, indent=2), flush=True)
    raise SystemExit(0 if status in {"COMPLETE", "PAUSED"} else 2)


if __name__ == "__main__":
    main()
