from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.data.interleaving import bucket_sequence_counts
from fdt_rlm.data.manifest import file_sha256
from fdt_rlm.next_tools import safe_torch_load


def load_split(dataset_dir: Path, split: str) -> torch.Tensor:
    rows = []
    for path in sorted((dataset_dir / "shards" / split).glob("*.pt")):
        payload = safe_torch_load(path)
        ids = payload["input_ids"].long()
        mask = payload.get("attention_mask")
        if mask is not None and not bool(mask.bool().all()):
            raise ValueError("Repacking currently requires fully packed source rows")
        rows.append(ids)
    if not rows:
        raise FileNotFoundError(f"No {split} shards in {dataset_dir}")
    return torch.cat(rows, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repack an existing packed-token dataset at a longer context length.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--shard-tokens", type=int, default=10_000_000)
    args = parser.parse_args()

    source = Path(args.input_dir)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    old_seq_len = int(manifest["seq_len"])
    new_seq_len = int(args.seq_len)
    if new_seq_len <= old_seq_len or new_seq_len % old_seq_len:
        raise ValueError("New sequence length must be a larger multiple of the source length")
    bucket_order = tuple(manifest["plan"].keys())
    rows_per_shard = max(int(args.shard_tokens) // new_seq_len, 1)
    shard_rows = []
    packed_by_bucket = defaultdict(int)
    discarded_by_bucket = defaultdict(int)

    for split in ("train", "validation", "test"):
        source_rows = load_split(source, split)
        counts = bucket_sequence_counts(manifest, split=split, bucket_order=bucket_order)
        cursor = 0
        shard_index = 0
        for bucket in bucket_order:
            row_count = int(counts[bucket])
            bucket_rows = source_rows[cursor : cursor + row_count]
            cursor += row_count
            flat = bucket_rows.reshape(-1)
            usable = (flat.numel() // new_seq_len) * new_seq_len
            discarded_by_bucket[f"{split}:{bucket}"] += flat.numel() - usable
            if usable == 0:
                continue
            repacked = flat[:usable].reshape(-1, new_seq_len)
            for start in range(0, repacked.size(0), rows_per_shard):
                chunk = repacked[start : start + rows_per_shard].contiguous()
                split_dir = output / "shards" / split
                split_dir.mkdir(parents=True, exist_ok=True)
                path = split_dir / f"shard_{shard_index:06d}.pt"
                torch.save({"input_ids": chunk, "attention_mask": torch.ones_like(chunk)}, path)
                tokens = int(chunk.numel())
                packed_by_bucket[bucket] += tokens
                shard_rows.append(
                    {
                        "path": str(path),
                        "split": split,
                        "tokens": tokens,
                        "sequences": int(chunk.size(0)),
                        "buckets": {bucket: tokens},
                        "sha256": file_sha256(path),
                    }
                )
                shard_index += 1
        if cursor != source_rows.size(0):
            raise RuntimeError(f"Manifest row counts do not cover {split}: {cursor} != {source_rows.size(0)}")

    output_manifest = dict(manifest)
    output_manifest.update(
        created_at=datetime.now(timezone.utc).isoformat(),
        repacked_from=str(source),
        original_seq_len=old_seq_len,
        seq_len=new_seq_len,
        shard_tokens=int(args.shard_tokens),
        packed_tokens=sum(row["tokens"] for row in shard_rows),
        tokens_by_bucket=dict(packed_by_bucket),
        discarded_tail_tokens=sum(discarded_by_bucket.values()),
        discarded_by_bucket=dict(discarded_by_bucket),
        shards=shard_rows,
    )
    (output / "manifest.json").write_text(json.dumps(output_manifest, indent=2), encoding="utf-8")
    (output / "repack_report.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "input": str(source),
                "output": str(output),
                "old_seq_len": old_seq_len,
                "new_seq_len": new_seq_len,
                "packed_tokens": output_manifest["packed_tokens"],
                "discarded_tail_tokens": output_manifest["discarded_tail_tokens"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(json.loads((output / "repack_report.json").read_text(encoding="utf-8")), indent=2))


if __name__ == "__main__":
    main()
