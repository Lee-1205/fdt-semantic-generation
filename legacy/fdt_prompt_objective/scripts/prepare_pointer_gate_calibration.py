from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.next_tools import safe_torch_load


def require_c_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"{label} must be on C:, got {resolved}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_split(dataset_dir: Path, split: str) -> dict[str, torch.Tensor]:
    payloads = [
        safe_torch_load(path, "cpu")
        for path in sorted((dataset_dir / "shards" / split).glob("*.pt"))
    ]
    if not payloads:
        raise FileNotFoundError(f"No {split} shards in {dataset_dir}")
    return {
        key: torch.cat([payload[key] for payload in payloads])
        for key in ("input_ids", "labels", "attention_mask")
    }


def negative_labels(input_ids: torch.Tensor, attention_mask: torch.Tensor, tokens: int) -> torch.Tensor:
    labels = torch.full_like(input_ids, -100)
    for row in range(input_ids.size(0)):
        active = int(attention_mask[row].sum())
        start = max(active - max(int(tokens), 1), 1)
        labels[row, start:active] = input_ids[row, start:active]
    return labels


def shard_group(path: Path) -> str:
    return path.name.split("_shard_", 1)[0]


def sample_broad_windows(
    dataset_dir: Path,
    split: str,
    count: int,
    length: int,
    negative_tokens: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    paths = sorted((dataset_dir / "shards" / split).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No {split} shards in {dataset_dir}")
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[shard_group(path)].append(path)
    rng = random.Random(seed)
    groups = sorted(grouped)
    quotas = {group: count // len(groups) for group in groups}
    for group in groups[: count % len(groups)]:
        quotas[group] += 1

    ids_rows, mask_rows, used_files = [], [], []
    for group_index, group in enumerate(groups):
        remaining = quotas[group]
        group_paths = list(grouped[group])
        rng.shuffle(group_paths)
        for path in group_paths:
            if remaining <= 0:
                break
            payload = safe_torch_load(path, "cpu")
            ids = payload["input_ids"]
            masks = payload.get("attention_mask", torch.ones_like(ids))
            order = list(range(ids.size(0)))
            random.Random(seed + group_index * 1009 + len(used_files)).shuffle(order)
            take = min(remaining, len(order))
            for row_index in order[:take]:
                active = int(masks[row_index].sum())
                if active < 2:
                    continue
                window_length = min(length, active)
                max_start = max(active - window_length, 0)
                start = rng.randrange(max_start + 1) if max_start else 0
                row_ids = ids[row_index, start : start + window_length].to(torch.int32)
                row_mask = masks[row_index, start : start + window_length].bool()
                if window_length < length:
                    row_ids = torch.nn.functional.pad(row_ids, (0, length - window_length))
                    row_mask = torch.nn.functional.pad(row_mask, (0, length - window_length))
                ids_rows.append(row_ids)
                mask_rows.append(row_mask)
            remaining -= take
            used_files.append(str(path.relative_to(dataset_dir)))
        if remaining > 0:
            raise ValueError(f"Insufficient rows for group {group}: {remaining} missing")

    input_ids = torch.stack(ids_rows)
    attention_mask = torch.stack(mask_rows)
    labels = negative_labels(input_ids, attention_mask, negative_tokens)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }, used_files


def combine_and_shuffle(
    positive: dict[str, torch.Tensor], negative: dict[str, torch.Tensor], seed: int
) -> dict[str, torch.Tensor]:
    payload = {
        "input_ids": torch.cat((positive["input_ids"].to(torch.int32), negative["input_ids"])),
        "labels": torch.cat((positive["labels"].to(torch.int32), negative["labels"])),
        "attention_mask": torch.cat((positive["attention_mask"].bool(), negative["attention_mask"])),
    }
    order = torch.randperm(
        payload["input_ids"].size(0), generator=torch.Generator().manual_seed(seed)
    )
    return {key: value.index_select(0, order) for key, value in payload.items()}


def atomic_torch_save(payload: dict, destination: Path):
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced positive/negative lexical-pointer gate calibration")
    parser.add_argument("--copy-dataset", type=Path, default=ROOT / "prepared_data" / "copy_diagnostic_4k")
    parser.add_argument("--broad-dataset", type=Path, default=ROOT / "prepared_data" / "v3_broad_additional_300m_combined_v2_24k_ctx2048")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prepared_data" / "pointer_gate_calibration_500m")
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--train-negatives", type=int, default=2048)
    parser.add_argument("--validation-negatives", type=int, default=256)
    parser.add_argument("--negative-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    copy_dir = require_c_path(args.copy_dataset, "copy dataset")
    broad_dir = require_c_path(args.broad_dataset, "broad dataset")
    output_dir = require_c_path(args.output_dir, "output directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output_dir}")

    split_settings = {
        "train": (args.train_negatives, args.seed),
        "validation": (args.validation_negatives, args.seed + 1),
    }
    manifest_splits = {}
    for split, (negative_count, split_seed) in split_settings.items():
        positive = load_split(copy_dir, split)
        if positive["input_ids"].size(1) != args.length:
            raise ValueError("Copy dataset sequence length does not match calibration length")
        negative, used_files = sample_broad_windows(
            broad_dir,
            split,
            negative_count,
            args.length,
            args.negative_tokens,
            split_seed,
        )
        combined = combine_and_shuffle(positive, negative, split_seed)
        shard_dir = output_dir / "shards" / split
        shard_dir.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(combined, shard_dir / "shard_000000.pt")
        manifest_splits[split] = {
            "rows": int(combined["input_ids"].size(0)),
            "positive_copy_rows": int(positive["input_ids"].size(0)),
            "broad_negative_rows": int(negative["input_ids"].size(0)),
            "broad_negative_supervised_tokens_per_row": args.negative_tokens,
            "broad_source_files": used_files,
        }

    manifest = {
        "format": "lexical_pointer_gate_calibration_v1",
        "purpose": "Calibrate retrieval activation while preserving frozen-base generation",
        "sequence_length": args.length,
        "seed": args.seed,
        "copy_dataset": str(copy_dir),
        "copy_manifest_sha256": sha256_file(copy_dir / "manifest.json"),
        "broad_dataset": str(broad_dir),
        "broad_manifest_sha256": sha256_file(broad_dir / "manifest.json"),
        "splits": manifest_splits,
    }
    temporary = output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(output_dir / "manifest.json")
    print(json.dumps({"status": "PASS", "output": str(output_dir), "splits": manifest_splits}, indent=2))


if __name__ == "__main__":
    main()
