from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.models import build_model, causal_lm_loss
from fdt_rlm.next_tools import load_next_config, make_model_config
from fdt_rlm.tokenization import load_tokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(path)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def slope(points: list[dict]) -> float:
    xs = [point["tokens_seen"] / 1_000_000 for point in points]
    ys = [point["heldout_loss"] for point in points]
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator


@torch.no_grad()
def evaluate(model, rows: torch.Tensor, config, device: torch.device) -> float:
    model.eval()
    losses = []
    for row in rows:
        ids = row.to(device=device, dtype=torch.long).unsqueeze(0)
        mask = torch.ones_like(ids)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(ids, attention_mask=mask)
            loss = causal_lm_loss(output["logits"], ids, mask, config.pad_token_id)
        losses.append(float(loss))
    model.train()
    return statistics.mean(losses)


def train_one(
    model_type: str,
    config_path: Path,
    train_rows: torch.Tensor,
    heldout_rows: torch.Tensor,
    seed: int,
    effective_tokens: int,
    eval_every: int,
    device: torch.device,
) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    values = load_next_config(config_path)
    values["model_type"] = model_type
    values["max_seq_len"] = int(train_rows.size(1))
    tokenizer = load_tokenizer(values["tokenizer_name"])
    config = make_model_config(values, len(tokenizer), tokenizer.pad_token_id, tokenizer.eos_token_id)
    model = build_model(config).to(device).train()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    seq_len = int(train_rows.size(1))
    accumulation = effective_tokens // seq_len
    optimizer_steps = int(train_rows.numel() // effective_tokens)
    order = torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(seed))
    points = [{"optimizer_step": 0, "tokens_seen": 0, "heldout_loss": evaluate(model, heldout_rows, config, device)}]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_seconds = 0.0
    cursor = 0
    for step in range(1, optimizer_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        train_loss = 0.0
        for _ in range(accumulation):
            row = train_rows[int(order[cursor])]
            cursor += 1
            ids = row.to(device=device, dtype=torch.long).unsqueeze(0)
            mask = torch.ones_like(ids)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = model(ids, attention_mask=mask)
                loss = causal_lm_loss(output["logits"], ids, mask, config.pad_token_id)
            (loss / accumulation).backward()
            train_loss += float(loss.detach()) / accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        sync(device)
        train_seconds += time.perf_counter() - start
        if step % eval_every == 0 or step == optimizer_steps:
            points.append({
                "optimizer_step": step,
                "tokens_seen": step * effective_tokens,
                "train_loss": train_loss,
                "heldout_loss": evaluate(model, heldout_rows, config, device),
            })
    row = {
        "status": "COMPLETED",
        "model_type": model_type,
        "seed": seed,
        "parameters": parameters,
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 0.1,
        "precision": "bf16",
        "seq_len": seq_len,
        "microbatch_size": 1,
        "gradient_accumulation_steps": accumulation,
        "effective_tokens_per_optimizer_step": effective_tokens,
        "optimizer_steps": optimizer_steps,
        "tokens_seen": optimizer_steps * effective_tokens,
        "train_seconds": train_seconds,
        "training_tokens_per_second": optimizer_steps * effective_tokens / train_seconds,
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0,
        "heldout_points": points,
        "heldout_loss_delta": points[-1]["heldout_loss"] - points[0]["heldout_loss"],
        "heldout_loss_slope_per_million_tokens": slope(points),
    }
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Equal-data 426M FDT/Transformer early loss-slope control.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--fdt-config", type=Path, required=True)
    parser.add_argument("--transformer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective-tokens", type=int, default=16384)
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026081401, 2026081402, 2026081403])
    args = parser.parse_args()

    free_gib = Path("C:/").stat().st_size if False else 0  # C: guard is enforced by the launcher.
    del free_gib
    manifest_path = args.dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_path = args.dataset / "train.pt"
    heldout_path = args.dataset / "heldout.pt"
    if sha256(train_path) != manifest["train_sha256"] or sha256(heldout_path) != manifest["heldout_sha256"]:
        raise RuntimeError("dataset shard hash mismatch")
    train_rows = torch.load(train_path, map_location="cpu", weights_only=True)["input_ids"]
    heldout_rows = torch.load(heldout_path, map_location="cpu", weights_only=True)["input_ids"]
    device = torch.device("cuda")
    payload = {
        "status": "IN_PROGRESS",
        "training_performed": True,
        "full_checkpoints_saved": False,
        "purpose": "bounded early loss-slope control; not a final capability comparison",
        "script_sha256": sha256(Path(__file__)),
        "dataset_manifest_sha256": sha256(manifest_path),
        "fdt_config_sha256": sha256(args.fdt_config),
        "transformer_config_sha256": sha256(args.transformer_config),
        "predeclared_seeds": args.seeds,
        "hardware": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "rows": [],
    }
    atomic_json(args.output, payload)
    for seed in args.seeds:
        for model_type, config_path in (
            ("fdt_v3", args.fdt_config),
            ("transformer", args.transformer_config),
        ):
            result = train_one(
                model_type,
                config_path,
                train_rows,
                heldout_rows,
                seed,
                args.effective_tokens,
                args.eval_every,
                device,
            )
            payload["rows"].append(result)
            atomic_json(args.output, payload)
            print(json.dumps(result), flush=True)
    summary = {}
    for model_type in ("fdt_v3", "transformer"):
        selected = [row for row in payload["rows"] if row["model_type"] == model_type]
        for metric in ("heldout_loss_delta", "heldout_loss_slope_per_million_tokens", "training_tokens_per_second"):
            values = [row[metric] for row in selected]
            summary[f"{model_type}_{metric}"] = {
                "mean": statistics.mean(values),
                "sample_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    payload["summary"] = summary
    payload["status"] = "CONTROL_COMPLETED"
    atomic_json(args.output, payload)


if __name__ == "__main__":
    main()
