from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.lexical_pointer import LexicalPointerDecodeState, SparseLexicalPointer
from fdt_rlm.next_tools import load_model_and_tokenizer, safe_torch_load


def load_split(dataset_dir: Path, split: str):
    payloads = [safe_torch_load(path) for path in sorted((dataset_dir / "shards" / split).glob("*.pt"))]
    if not payloads:
        raise FileNotFoundError(f"No shards for {split}")
    return tuple(torch.cat([payload[key] for payload in payloads]) for key in ("input_ids", "labels", "attention_mask"))


def append_jsonl(path: Path, payload: dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_c_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"{label} must be on C:, got {resolved}")
    return resolved


def atomic_torch_save(payload: dict, destination: Path):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


@torch.inference_mode()
def evaluate(model, pointer, ids, labels, masks, batch_size, device):
    pointer.eval()
    totals = {"loss": 0.0, "pointer_accuracy": 0.0, "copyable_rate": 0.0, "batches": 0}
    for start in range(0, len(ids), batch_size):
        x = ids[start : start + batch_size].long().to(device)
        y = labels[start : start + batch_size].long().to(device)
        mask = masks[start : start + batch_size].long().to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = model(x, attention_mask=mask)["hidden"]
        result = pointer.training_loss(hidden, x, y, mask)
        totals["loss"] += float(result.loss)
        totals["pointer_accuracy"] += float(result.pointer_accuracy)
        totals["copyable_rate"] += float(result.copyable_rate)
        totals["batches"] += 1
    count = max(totals.pop("batches"), 1)
    return {key: value / count for key, value in totals.items()}


@torch.inference_mode()
def generate(model, pointer, tokenizer, prompt: str, max_new_tokens: int, device):
    ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=device)
    generated = ids
    trace = []
    state = LexicalPointerDecodeState(source_length=ids.size(1))
    for _ in range(max_new_tokens):
        mask = torch.ones_like(generated)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(generated, attention_mask=mask)
        logits, diagnostics = state.prepare_logits(
            pointer,
            output["logits"][:, -1].float(),
            output["hidden"],
            generated,
            mask,
            min_gate=0.8,
        )
        next_id = logits.argmax(dim=-1, keepdim=True)
        diagnostics["selected_id"] = int(next_id)
        trace.append(diagnostics)
        piece = tokenizer.decode([int(next_id)], skip_special_tokens=True)
        state.commit(int(next_id), diagnostics, boundary=bool(piece.rstrip().endswith((".", "!", "?", ";", ":")) or "\n" in piece))
        generated = torch.cat((generated, next_id), dim=1)
        if int(next_id) == tokenizer.eos_token_id:
            break
    return tokenizer.decode(generated[0, ids.size(1) :].tolist(), skip_special_tokens=True), trace


def main():
    parser = argparse.ArgumentParser(description="Train an O(NW) lexical pointer adapter")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers" / "fdt_v3_bpe_24k")
    parser.add_argument("--pointer-checkpoint", type=Path)
    parser.add_argument("--reset-gate", action="store_true")
    parser.add_argument("--train-gate-only", action="store_true")
    parser.add_argument("--train-routing-only", action="store_true")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--pointer-dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.train_gate_only and args.train_routing_only:
        parser.error("--train-gate-only and --train-routing-only are mutually exclusive")

    checkpoint_path = require_c_path(args.checkpoint, "base checkpoint")
    dataset_dir = require_c_path(args.dataset_dir, "dataset")
    output_dir = require_c_path(args.output_dir, "output directory")
    pointer_checkpoint_path = (
        require_c_path(args.pointer_checkpoint, "pointer checkpoint")
        if args.pointer_checkpoint
        else None
    )
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_payload = safe_torch_load(checkpoint_path)
    integrated = checkpoint_payload["model_config"].get("model_type") == "fdt_v3_anchor_indexed_dual_memory"
    model, tokenizer, config, _ = load_model_and_tokenizer(
        checkpoint_path,
        device,
        None if integrated else "fdt_v3",
        tokenizer_name=str(require_c_path(args.tokenizer, "tokenizer")),
    )
    model = model.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if integrated:
        pointer = model.exact_pointer.to(device=device, dtype=torch.float32)
        for parameter in pointer.parameters():
            parameter.requires_grad_(True)
    else:
        pointer = SparseLexicalPointer(config.dim, args.pointer_dim, args.window).to(device)
    if pointer_checkpoint_path:
        resumed_pointer = safe_torch_load(pointer_checkpoint_path)
        pointer.load_state_dict(resumed_pointer["pointer_state_dict"], strict=True)
    if args.reset_gate:
        torch.nn.init.zeros_(pointer.gate_proj.weight)
        torch.nn.init.constant_(pointer.gate_proj.bias, -3.0)
    if args.train_gate_only:
        for name, parameter in pointer.named_parameters():
            parameter.requires_grad_(name.startswith("gate_proj."))
    elif args.train_routing_only:
        for name, parameter in pointer.named_parameters():
            parameter.requires_grad_(not name.startswith("gate_proj."))
    optimizer = torch.optim.AdamW(
        [parameter for parameter in pointer.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )
    train_ids, train_labels, train_masks = load_split(dataset_dir, "train")
    val_ids, val_labels, val_masks = load_split(dataset_dir, "validation")
    base_sha256 = sha256_file(checkpoint_path)
    parent_pointer_sha256 = sha256_file(pointer_checkpoint_path) if pointer_checkpoint_path else None
    if pointer_checkpoint_path:
        expected_parent_base = resumed_pointer.get("base_checkpoint_sha256")
        if expected_parent_base and expected_parent_base.upper() != base_sha256:
            raise ValueError("Pointer checkpoint was trained against a different base SHA-256")
    manifest_path = dataset_dir / "manifest.json"
    dataset_manifest_sha256 = sha256_file(manifest_path) if manifest_path.exists() else None
    order = torch.cat(
        [torch.randperm(len(train_ids), generator=torch.Generator().manual_seed(args.seed + epoch)) for epoch in range(args.epochs)]
    )
    log_path = output_dir / "training_log.jsonl"
    best_metric = float("inf") if args.train_gate_only else -1.0
    best_tie_break_loss = float("inf")
    cursor = 0
    step = 0
    while cursor < len(order):
        indices = order[cursor : cursor + args.batch_size]
        x = train_ids.index_select(0, indices).long().to(device)
        y = train_labels.index_select(0, indices).long().to(device)
        mask = train_masks.index_select(0, indices).long().to(device)
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = model(x, attention_mask=mask)["hidden"]
        pointer.train()
        optimizer.zero_grad(set_to_none=True)
        result = pointer.training_loss(hidden, x, y, mask)
        if not torch.isfinite(result.loss):
            raise FloatingPointError(f"Non-finite pointer loss at step {step + 1}")
        result.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(pointer.parameters(), 1.0)
        if not math.isfinite(float(grad_norm)):
            raise FloatingPointError(f"Non-finite pointer gradient at step {step + 1}")
        optimizer.step()
        cursor += len(indices)
        step += 1
        if step % 10 == 0:
            append_jsonl(log_path, {"event": "train", "step": step, "loss": float(result.loss.detach()), "pointer_accuracy": float(result.pointer_accuracy), "copyable_rate": float(result.copyable_rate), "grad_norm": float(grad_norm)})
        if step % args.eval_every == 0 or cursor >= len(order):
            metrics = evaluate(model, pointer, val_ids, val_labels, val_masks, args.batch_size, device)
            probes = []
            for prompt in (
                "Record: verification code is R7319. Therefore the verification code is",
                "Context: The verification code is R7319.\nQuestion: What is the code?\nAnswer:",
            ):
                completion, trace = generate(model, pointer, tokenizer, prompt, 12, device)
                probes.append({"prompt": prompt, "completion": completion, "trace": trace})
            row = {"event": "validation", "step": step, **metrics, "probes": probes}
            append_jsonl(log_path, row)
            candidate_metric = metrics["loss"] if args.train_gate_only else metrics["pointer_accuracy"]
            improved = (
                candidate_metric < best_metric
                if args.train_gate_only
                else (
                    candidate_metric > best_metric
                    or (
                        math.isclose(candidate_metric, best_metric, rel_tol=0.0, abs_tol=1e-12)
                        and metrics["loss"] < best_tie_break_loss
                    )
                )
            )
            if improved:
                best_metric = candidate_metric
                best_tie_break_loss = metrics["loss"]
                atomic_torch_save(
                    {
                        "pointer_state_dict": pointer.state_dict(),
                        "pointer_config": {
                            "hidden_dim": config.dim,
                            "pointer_dim": args.pointer_dim,
                            "window": args.window,
                        },
                        "base_checkpoint": str(checkpoint_path),
                        "base_checkpoint_sha256": base_sha256,
                        "parent_pointer_checkpoint": (
                            str(pointer_checkpoint_path) if pointer_checkpoint_path else None
                        ),
                        "parent_pointer_checkpoint_sha256": parent_pointer_sha256,
                        "dataset_dir": str(dataset_dir),
                        "dataset_manifest_sha256": dataset_manifest_sha256,
                        "seed": args.seed,
                        "training_mode": (
                            "gate_only"
                            if args.train_gate_only
                            else "routing_only"
                            if args.train_routing_only
                            else "joint"
                        ),
                        "integrated_source": integrated,
                        "validation": row,
                    },
                    output_dir / "best.pt",
                )
    append_jsonl(
        log_path,
        {
            "event": "complete",
            "status": "PASS",
            "step": step,
            "selection_metric": "loss" if args.train_gate_only else "pointer_accuracy",
            "best_metric": best_metric,
            "best_tie_break_loss": best_tie_break_loss,
        },
    )


if __name__ == "__main__":
    main()
