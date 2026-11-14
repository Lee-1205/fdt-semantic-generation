from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from fdt_rlm.data.manifest import file_sha256, write_json


SPLITS = ("train", "validation", "test")


def load_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_shard(dataset_dir: Path, shard: dict) -> Path:
    path = Path(str(shard["path"]))
    if path.is_absolute():
        return path
    direct = ROOT / path
    if direct.is_file():
        return direct
    return dataset_dir / "shards" / str(shard["split"]) / path.name


def extract_bucket(input_dir: Path, output_dir: Path, bucket: str) -> dict:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir.drive.lower() != output_dir.drive.lower():
        raise ValueError("Input and output must be on the same volume for hard links.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {output_dir}")

    source = load_manifest(input_dir)
    selected = []
    totals = defaultdict(int)
    for shard in source.get("shards", []):
        buckets = {str(name): int(count) for name, count in shard.get("buckets", {}).items()}
        if bucket not in buckets:
            continue
        if set(buckets) != {bucket}:
            raise ValueError(
                f"Cannot extract mixed shard {shard['path']}: {sorted(buckets)}"
            )
        split = str(shard["split"])
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r} in {shard['path']}")
        selected.append((shard, resolve_shard(input_dir, shard)))
        totals[split] += int(shard["tokens"])

    missing = [split for split in SPLITS if totals[split] <= 0]
    if missing:
        raise ValueError(f"Bucket {bucket!r} has no shards for splits: {missing}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_shards = []
    counters = defaultdict(int)
    for shard, source_path in selected:
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing source shard: {source_path}")
        expected_hash = str(shard.get("sha256", ""))
        actual_hash = file_sha256(source_path)
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(f"Source shard hash mismatch: {source_path}")

        split = str(shard["split"])
        destination_dir = output_dir / "shards" / split
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"shard_{counters[split]:06d}.pt"
        counters[split] += 1
        os.link(source_path, destination)
        linked_hash = file_sha256(destination)
        if linked_hash != actual_hash:
            raise RuntimeError(f"Hard-linked shard hash mismatch: {destination}")
        output_shards.append(
            {
                **shard,
                "path": str(destination),
                "sha256": linked_hash,
                "buckets": {bucket: int(shard["tokens"])},
            }
        )

    packed_tokens = sum(int(shard["tokens"]) for shard in output_shards)
    source_manifest_path = input_dir / "manifest.json"
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_tokens": packed_tokens,
        "tokenizer": source["tokenizer"],
        "tokenizer_vocab_size": int(source["tokenizer_vocab_size"]),
        "plan": {bucket: packed_tokens},
        "bucket_order": [bucket],
        "tokens_by_bucket": {bucket: packed_tokens},
        "packed_tokens": packed_tokens,
        "discarded_tail_tokens": 0,
        "seq_len": int(source["seq_len"]),
        "shards": output_shards,
        "extracted_from": {
            "dataset_dir": str(input_dir),
            "manifest_sha256": file_sha256(source_manifest_path),
            "bucket": bucket,
        },
        "storage": "same-volume hard links",
        "split_tokens": dict(totals),
    }
    write_json(output_dir / "manifest.json", manifest)
    write_json(
        output_dir / "dataset_statistics.json",
        {
            "bucket": bucket,
            "packed_tokens": packed_tokens,
            "split_tokens": dict(totals),
            "shards": len(output_shards),
        },
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract an unmixed prepared-data bucket with same-volume hard links."
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bucket", required=True)
    args = parser.parse_args()

    manifest = extract_bucket(Path(args.input_dir), Path(args.output_dir), args.bucket)
    print(
        json.dumps(
            {
                "output": str(Path(args.output_dir).resolve()),
                "bucket": args.bucket,
                "packed_tokens": manifest["packed_tokens"],
                "split_tokens": manifest["split_tokens"],
                "shards": len(manifest["shards"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
