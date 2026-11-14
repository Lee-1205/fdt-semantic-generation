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


def shard_index(manifest: dict) -> dict[tuple[str, str], dict]:
    result = {}
    for shard in manifest.get("shards", []):
        key = (str(shard["split"]), Path(shard["path"]).name)
        if key in result:
            raise ValueError(f"Duplicate shard metadata: {key}")
        result[key] = shard
    return result


def physical_bucket_order(dataset_dir: Path, manifest: dict) -> list[str]:
    explicit = [str(value) for value in manifest.get("bucket_order", [])]
    if explicit:
        return explicit
    metadata = shard_index(manifest)
    order = []
    for path in sorted((dataset_dir / "shards" / "train").glob("*.pt")):
        info = metadata[("train", path.name)]
        for bucket in info.get("buckets", {}):
            if bucket not in order:
                order.append(bucket)
    for bucket in manifest.get("tokens_by_bucket", {}):
        if bucket not in order:
            order.append(bucket)
    return order


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine prepared datasets with same-volume hard links."
    )
    parser.add_argument("--input", action="append", required=True, dest="inputs")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    inputs = [Path(value).resolve() for value in args.inputs]
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to modify non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifests = [load_manifest(path) for path in inputs]
    seq_lens = {int(item["seq_len"]) for item in manifests}
    vocab_sizes = {int(item["tokenizer_vocab_size"]) for item in manifests}
    tokenizers = {str(item["tokenizer"]) for item in manifests}
    if len(seq_lens) != 1 or len(vocab_sizes) != 1 or len(tokenizers) != 1:
        raise ValueError("Inputs must use identical sequence length and tokenizer.")

    combined_shards = []
    tokens_by_bucket = defaultdict(int)
    bucket_order = []
    source_configs = []
    packed_tokens = 0
    requested_tokens = 0
    discarded_tail_tokens = 0

    for source_index, (dataset_dir, manifest) in enumerate(zip(inputs, manifests)):
        metadata = shard_index(manifest)
        discarded_tail_tokens += int(manifest.get("discarded_tail_tokens", 0))
        for bucket, count in manifest.get("tokens_by_bucket", {}).items():
            tokens_by_bucket[bucket] += int(count)
        for bucket in physical_bucket_order(dataset_dir, manifest):
            if bucket not in bucket_order:
                bucket_order.append(bucket)
        source_configs.append(
            {
                "dataset_dir": str(dataset_dir),
                "source_config": manifest.get("source_config", {}),
            }
        )

        for split in SPLITS:
            source_files = sorted((dataset_dir / "shards" / split).glob("*.pt"))
            if not source_files:
                raise FileNotFoundError(f"No {split} shards in {dataset_dir}")
            destination_dir = output / "shards" / split
            destination_dir.mkdir(parents=True, exist_ok=True)
            for shard_index_value, source in enumerate(source_files):
                info = metadata.get((split, source.name))
                if info is None:
                    raise ValueError(f"Manifest has no metadata for {source}")
                destination = destination_dir / (
                    f"source_{source_index:02d}_shard_{shard_index_value:06d}.pt"
                )
                os.link(source, destination)
                digest = file_sha256(destination)
                if digest != info.get("sha256"):
                    raise RuntimeError(f"Hard-linked shard hash mismatch: {destination}")
                tokens = int(info["tokens"])
                packed_tokens += tokens
                combined_shards.append(
                    {
                        **info,
                        "path": str(destination),
                        "sha256": digest,
                    }
                )

    requested_tokens = sum(tokens_by_bucket.values())
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_tokens": requested_tokens,
        "tokenizer": next(iter(tokenizers)),
        "tokenizer_vocab_size": next(iter(vocab_sizes)),
        "plan": {bucket: tokens_by_bucket[bucket] for bucket in bucket_order},
        "bucket_order": bucket_order,
        "tokens_by_bucket": dict(tokens_by_bucket),
        "packed_tokens": packed_tokens,
        "discarded_tail_tokens": discarded_tail_tokens,
        "seq_len": next(iter(seq_lens)),
        "shards": combined_shards,
        "combined_from": source_configs,
        "storage": "same-volume hard links",
    }
    write_json(output / "manifest.json", manifest)
    write_json(
        output / "dataset_statistics.json",
        {
            "tokens_by_bucket": dict(tokens_by_bucket),
            "packed_tokens": packed_tokens,
            "source_count": len(inputs),
        },
    )
    print(json.dumps({
        "output": str(output),
        "packed_tokens": packed_tokens,
        "plan": manifest["plan"],
        "bucket_order": bucket_order,
        "shards": len(combined_shards),
    }, indent=2))


if __name__ == "__main__":
    main()
