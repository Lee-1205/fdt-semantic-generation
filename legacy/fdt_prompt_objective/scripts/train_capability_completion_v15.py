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
from train_chat_sft import evaluate_chat, masked_chat_losses


def unexpected_output_files(output_dir: Path) -> list[str]:
    launcher_files = {"active_logs.json", "train.pid"}
    return [
        path.name
        for path in output_dir.iterdir()
        if path.name not in launcher_files
        and not re.fullmatch(r"train\.\d{8}_\d{6}\.(stdout|stderr)\.log", path.name)
    ]


def validate_resume_checkpoint(checkpoint: dict, output_dir: Path, dataset_dir: Path) -> None:
    settings = checkpoint.get("train_config", {})
    if settings.get("stage") != "capability_completion_v15":
        raise ValueError("Resume checkpoint is not a V15 capability-completion stage")
    if checkpoint.get("stage_status") != "PAUSED":
        raise ValueError("V15 resume requires a PAUSED checkpoint")
    if Path(settings.get("output_dir", "")).resolve() != output_dir.resolve():
        raise ValueError("V15 resume output directory identity mismatch")
    if Path(settings.get("dataset_dir", "")).resolve() != dataset_dir.resolve():
        raise ValueError("V15 resume dataset identity mismatch")
    step = int(checkpoint.get("optimizer_step", -1))
    cursor = int(checkpoint.get("sample_cursor", -1))
    if step < 0 or cursor < 0:
        raise ValueError("V15 resume checkpoint is missing step/cursor state")


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def verify_sha256(path: Path, expected: str, label: str) -> str:
    actual = strict_v3.sha256_file(path)
    if actual.upper() != expected.upper():
        raise ValueError(f"{label} hash mismatch: {actual}")
    return actual


def load_completion_split(dataset_dir: Path, split: str):
    payloads = [
        safe_torch_load(path, "cpu")
        for path in sorted((dataset_dir / "shards" / split).glob("*.pt"))
    ]
    if not payloads:
        raise FileNotFoundError(f"No V15 {split} shards under {dataset_dir}")
    return tuple(
        torch.cat([payload[key] for payload in payloads], dim=0)
        for key in ("input_ids", "labels", "attention_mask")
    )


def atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


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


def strict_smoke(
    model,
    tokenizer,
    config,
    capability_payload,
    capability_ranges,
    broad_ids,
    broad_masks,
    broad_ranges,
    device,
) -> dict:
    heldout = strict.heldout_metrics(
        model, config, broad_ids, broad_masks, broad_ranges, rows_per_bucket=4, device=device
    )
    generation = strict_v3.evaluate_capability_generation(
        model,
        tokenizer,
        config,
        capability_payload,
        capability_ranges,
        {"factual_qa": 4, "retrieval": 4, "python_code": 4, "json": 4},
        device,
    )
    prefix = strict_v3.evaluate_heldout_generation(
        model,
        tokenizer,
        config,
        broad_ids,
        broad_masks,
        broad_ranges,
        rows_per_bucket=2,
        device=device,
    )["overall"]
    return {
        "heldout": heldout["overall"],
        "capability_generation": {
            category: {key: value for key, value in report.items() if key != "rows"}
            for category, report in generation.items()
        },
        "prefix_generation": prefix,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded non-conversational completion-span V15 diagnostic.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--broad-dataset-dir",
        type=Path,
        default=ROOT / "prepared_data" / "v3_broad_additional_300m_combined_v2_24k_ctx2048",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--prompt-lm-weight", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=0.7)
    parser.add_argument("--smoke-step", type=int, default=100)
    parser.add_argument("--max-broad-relative-regression", type=float, default=0.003)
    parser.add_argument("--max-prefix-repetition-increase", type=float, default=0.05)
    parser.add_argument("--max-dead-anchor-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    for path in (args.checkpoint, args.dataset_dir, args.output_dir, args.broad_dataset_dir):
        if path.resolve().drive.upper() != "C:":
            raise ValueError(f"V15 permits only C: paths: {path.resolve()}")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and not args.resume:
        unexpected = unexpected_output_files(output_dir)
        if unexpected:
            raise FileExistsError(
                f"Dedicated V15 output contains non-launcher files: {output_dir}: {unexpected}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_hash = verify_sha256(
        args.checkpoint, args.expected_checkpoint_sha256, "Parent checkpoint"
    )
    manifest_hash = verify_sha256(
        args.dataset_dir / "manifest.json", args.expected_manifest_sha256, "V15 manifest"
    )

    device = torch.device(args.device)
    checkpoint = safe_torch_load(args.checkpoint, "cpu")
    if args.resume:
        validate_resume_checkpoint(checkpoint, output_dir, args.dataset_dir)
    config = ModelConfig(**checkpoint["model_config"])
    tokenizer_name = checkpoint.get("train_config", {}).get("tokenizer_name", "") or "gpt2"
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    tokens_seen = int(checkpoint.get("tokens_seen", 0))
    resume_step = int(checkpoint.get("optimizer_step", 0)) if args.resume else 0
    resume_cursor = int(checkpoint.get("sample_cursor", 0)) if args.resume else 0
    resume_settings = dict(checkpoint.get("train_config", {})) if args.resume else {}
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

    train_ids, train_labels, train_masks = load_completion_split(args.dataset_dir, "train")
    val_ids, val_labels, val_masks = load_completion_split(args.dataset_dir, "validation")
    broad_ids, broad_masks = strict.split_rows(args.broad_dataset_dir, "validation")
    broad_manifest = json.loads((args.broad_dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    broad_ranges = strict.bucket_ranges(broad_manifest, "validation")
    capability_payload = {
        "input_ids": val_ids,
        "labels": val_labels,
        "attention_mask": val_masks,
    }
    capability_manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    capability_ranges = strict_v3.capability_ranges(capability_manifest)

    settings = resume_settings or {
        "stage": "capability_completion_v15",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "parent_tokens": tokens_seen,
        "dataset_dir": str(args.dataset_dir.resolve()),
        "manifest_sha256": manifest_hash,
        "output_dir": str(output_dir),
        "tokenizer_name": str(tokenizer_name),
        "learning_rate": args.lr,
        "lr_schedule": "constant",
        "lr_warmup_steps": 0,
        "batch_size": args.batch_size,
        "prompt_lm_weight": args.prompt_lm_weight,
        "freeze_routing": True,
        "frozen_parameter_names": len(frozen),
        "conversational_sft": False,
        "completion_span_supervision": True,
        "seed": args.seed,
    }
    if args.resume:
        expected_manifest = str(settings.get("manifest_sha256", "")).upper()
        if expected_manifest != manifest_hash.upper():
            raise ValueError("V15 resume manifest hash mismatch")
        if int(settings.get("batch_size", -1)) != args.batch_size:
            raise ValueError("V15 resume batch size mismatch")
        if float(settings.get("learning_rate", -1.0)) != args.lr:
            raise ValueError("V15 resume learning rate mismatch")
        if int(settings.get("seed", args.seed)) != args.seed:
            raise ValueError("V15 resume data-order seed mismatch")
        settings["resume_from_checkpoint"] = str(args.checkpoint.resolve())
        settings["resume_from_sha256"] = checkpoint_hash
        settings["resume_step"] = resume_step
        settings["resume_cursor"] = resume_cursor
    (output_dir / "train_config.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    (output_dir / "model_config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    log_path = output_dir / "training_log.jsonl"
    latest_path = output_dir / "latest.pt"
    stop_path = output_dir / "STOP_REQUESTED"

    baseline_smoke = None
    if args.resume:
        if resume_step < args.smoke_step:
            raise ValueError("V15 resume before the predeclared smoke gate is unsupported")
        append_jsonl(
            log_path,
            {
                "event": "resume",
                "step": resume_step,
                "sample_cursor": resume_cursor,
                "tokens_seen": tokens_seen,
                "checkpoint_sha256": checkpoint_hash,
            },
        )
    else:
        baseline_chat = evaluate_chat(
            model, val_ids, val_labels, val_masks, args.batch_size, device
        )
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
        append_jsonl(
            log_path,
            {"event": "baseline", "tokens_seen": tokens_seen, "capability": baseline_chat, **baseline_smoke},
        )

    order = torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(args.seed))
    cursor = resume_cursor
    step = resume_step
    status = "COMPLETE"
    started = time.perf_counter()
    resume_tokens_start = tokens_seen
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
            loss = completion_loss + args.prompt_lm_weight * prompt_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite V15 loss")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite V15 gradient")
        optimizer.step()
        cursor += len(indices)
        step += 1
        batch_tokens = int(mask.sum().item())
        tokens_seen += batch_tokens
        if step % 25 == 0:
            append_jsonl(
                log_path,
                {
                    "event": "train",
                    "step": step,
                    "sample_cursor": cursor,
                    "tokens_seen": tokens_seen,
                    "loss": float(loss),
                    "completion_loss": float(completion_loss),
                    "prompt_loss": float(prompt_loss),
                    "grad_norm": float(grad_norm),
                    "tokens_per_sec": (tokens_seen - resume_tokens_start)
                    / max(time.perf_counter() - started, 1e-9),
                },
            )
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
            broad_relative = (
                smoke["heldout"]["loss"] / baseline_smoke["heldout"]["loss"] - 1.0
            )
            dead = smoke["heldout"]["routing"].get("dead_anchor_ratio", 0.0)
            repetition_increase = (
                smoke["prefix_generation"]["mean_repetition_rate"]
                - baseline_smoke["prefix_generation"]["mean_repetition_rate"]
            )
            healthy = (
                math.isfinite(smoke["heldout"]["loss"])
                and broad_relative <= args.max_broad_relative_regression
                and dead <= args.max_dead_anchor_ratio
                and repetition_increase <= args.max_prefix_repetition_increase
            )
            append_jsonl(
                log_path,
                {
                    "event": "smoke_gate",
                    "step": step,
                    "broad_relative_delta": broad_relative,
                    "dead_anchor_ratio": dead,
                    "prefix_repetition_delta": repetition_increase,
                    "healthy": healthy,
                    **smoke,
                },
            )
            if not healthy:
                status = "ABORTED_SMOKE_GATE"
                break
            model.train()

    model.eval()
    final_chat = evaluate_chat(model, val_ids, val_labels, val_masks, args.batch_size, device)
    save_checkpoint(latest_path, model, optimizer, config, settings, step, cursor, tokens_seen, status)
    final = {
        "event": "complete" if status == "COMPLETE" else "paused",
        "status": status,
        "step": step,
        "sample_cursor": cursor,
        "tokens_seen": tokens_seen,
        "capability": final_chat,
    }
    append_jsonl(log_path, final)
    print(json.dumps(final, indent=2), flush=True)
    raise SystemExit(0 if status in {"COMPLETE", "PAUSED"} else 2)


if __name__ == "__main__":
    main()
