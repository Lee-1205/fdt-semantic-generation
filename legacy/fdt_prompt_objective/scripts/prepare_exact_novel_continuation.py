import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.next_tools import safe_torch_load
from fdt_rlm.tokenization import load_tokenizer


def require_c_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"Only C: paths are allowed, got {resolved}")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_hash(tokens: torch.Tensor) -> str:
    compact = tokens.detach().to(device="cpu", dtype=torch.int32).contiguous()
    return hashlib.blake2b(compact.numpy().tobytes(), digest_size=16).hexdigest()


def load_manifest(dataset_dir: Path) -> dict:
    path = require_c_path(dataset_dir) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_shard(dataset_dir: Path, entry: dict) -> Path:
    configured = Path(entry["path"])
    candidates = [configured]
    if not configured.is_absolute():
        candidates.extend([ROOT / configured, dataset_dir / configured])
    candidates.append(
        dataset_dir / "shards" / str(entry["split"]) / configured.name
    )
    for candidate in candidates:
        if candidate.exists():
            return require_c_path(candidate)
    raise FileNotFoundError(f"Unable to resolve shard {entry['path']!r}")


def iter_bucket_token_chunks(dataset_dir: Path, split: str = "train"):
    dataset_dir = require_c_path(dataset_dir)
    manifest = load_manifest(dataset_dir)
    seq_len = int(manifest["seq_len"])
    bucket_order = list(
        manifest.get("bucket_order")
        or manifest.get("plan", {}).keys()
        or manifest.get("tokens_by_bucket", {}).keys()
    )
    for entry in manifest["shards"]:
        if entry["split"] != split:
            continue
        path = resolve_shard(dataset_dir, entry)
        shard = safe_torch_load(path)
        ids = shard["input_ids"]
        mask = shard.get("attention_mask")
        cursor = 0
        entry_buckets = dict(entry.get("buckets", {}))
        if not entry_buckets and len(bucket_order) == 1:
            entry_buckets[bucket_order[0]] = int(ids.size(0)) * seq_len
        unknown = set(entry_buckets) - set(bucket_order)
        if unknown:
            raise ValueError(f"Unknown buckets in {path}: {sorted(unknown)}")
        for bucket in bucket_order:
            token_count = int(entry_buckets.get(bucket, 0))
            if token_count % seq_len:
                raise ValueError(
                    f"Bucket token count is not row aligned in {path}: "
                    f"{bucket}={token_count}"
                )
            row_count = token_count // seq_len
            if row_count == 0:
                continue
            end = cursor + row_count
            if end > ids.size(0):
                raise ValueError(f"Manifest row count exceeds tensor rows in {path}")
            part = ids[cursor:end].to(dtype=torch.int32).reshape(-1)
            if mask is not None:
                valid = mask[cursor:end].to(dtype=torch.bool).reshape(-1)
                part = part[valid]
            yield bucket, part
            cursor = end
        if cursor != ids.size(0):
            raise ValueError(
                f"Manifest accounts for {cursor} rows but {path} has {ids.size(0)}"
            )


def iter_documents(dataset_dir: Path, eos_token_id: int, split: str = "train"):
    carries: dict[str, torch.Tensor] = {}
    for bucket, chunk in iter_bucket_token_chunks(dataset_dir, split=split):
        carry = carries.get(bucket)
        if carry is not None and carry.numel():
            chunk = torch.cat((carry, chunk))
        eos_positions = (chunk == eos_token_id).nonzero(as_tuple=False).flatten().tolist()
        start = 0
        for end in eos_positions:
            yield bucket, chunk[start:end].clone()
            start = end + 1
        carries[bucket] = chunk[start:].clone()


def collect_document_hashes(
    dataset_dir: Path,
    eos_token_id: int,
    minimum_tokens: int = 8,
) -> tuple[dict[str, set[str]], dict[str, int]]:
    hashes: dict[str, set[str]] = defaultdict(set)
    counts: dict[str, int] = defaultdict(int)
    for bucket, document in iter_documents(dataset_dir, eos_token_id):
        if document.numel() < minimum_tokens:
            continue
        hashes[bucket].add(document_hash(document))
        counts[bucket] += 1
    return dict(hashes), dict(counts)


def collect_prefix_hashes(
    dataset_dir: Path,
    bucket: str,
    consumed_rows: int,
    eos_token_id: int,
    minimum_tokens: int = 8,
) -> tuple[set[str], int]:
    if consumed_rows < 0:
        raise ValueError("consumed_rows must be nonnegative")
    seq_len = int(load_manifest(dataset_dir)["seq_len"])
    boundary = consumed_rows * seq_len
    parts = []
    total = 0
    boundary_end = None
    for chunk_bucket, chunk in iter_bucket_token_chunks(dataset_dir, split="train"):
        if chunk_bucket != bucket:
            continue
        previous = total
        parts.append(chunk)
        total += chunk.numel()
        if total < boundary:
            continue
        local_start = max(boundary - previous, 0)
        eos_after = (chunk[local_start:] == eos_token_id).nonzero(as_tuple=False)
        if eos_after.numel():
            boundary_end = previous + local_start + int(eos_after[0]) + 1
            break
    if boundary == 0:
        return set(), 0
    if boundary_end is None:
        boundary_end = total
    stream = torch.cat(parts)[:boundary_end]
    hashes = set()
    start = 0
    for end in (stream == eos_token_id).nonzero(as_tuple=False).flatten().tolist():
        document = stream[start:end]
        if document.numel() >= minimum_tokens:
            hashes.add(document_hash(document))
        start = end + 1
    return hashes, boundary_end


class CompactShardWriter:
    def __init__(self, output_dir: Path, seq_len: int, shard_tokens: int):
        self.output_dir = require_c_path(output_dir)
        self.seq_len = seq_len
        self.rows_per_shard = max(shard_tokens // seq_len, 1)
        self.pending_documents: list[torch.Tensor] = []
        self.pending_tokens = 0
        self.tail = torch.empty(0, dtype=torch.int32)
        self.rows: list[torch.Tensor] = []
        self.shards: list[dict] = []
        self.bucket_rows: dict[str, int] = defaultdict(int)
        self.active_bucket = None

    def add_document(self, bucket: str, document: torch.Tensor, eos_token_id: int):
        if self.active_bucket is not None and bucket != self.active_bucket:
            self._materialize_pending()
            if self.tail.numel():
                self.tail = torch.empty(0, dtype=torch.int32)
        self.active_bucket = bucket
        with_eos = torch.cat(
            (document.to(dtype=torch.int32), torch.tensor([eos_token_id], dtype=torch.int32))
        )
        self.pending_documents.append(with_eos)
        self.pending_tokens += with_eos.numel()
        if self.pending_tokens + self.tail.numel() >= self.seq_len:
            self._materialize_pending()

    def _materialize_pending(self):
        if not self.pending_documents:
            return
        combined = torch.cat([self.tail, *self.pending_documents])
        full_tokens = (combined.numel() // self.seq_len) * self.seq_len
        if full_tokens:
            new_rows = combined[:full_tokens].reshape(-1, self.seq_len)
            for row in new_rows:
                self.rows.append(row.clone())
                self.bucket_rows[self.active_bucket] += 1
                if len(self.rows) >= self.rows_per_shard:
                    self._flush_shard()
        self.tail = combined[full_tokens:].clone()
        self.pending_documents = []
        self.pending_tokens = 0

    def _flush_shard(self):
        if not self.rows:
            return
        shard_dir = self.output_dir / "shards" / "train"
        shard_dir.mkdir(parents=True, exist_ok=True)
        index = len(self.shards)
        path = shard_dir / f"shard_{index:06d}.pt"
        temporary = path.with_name(path.name + ".tmp")
        ids = torch.stack(self.rows).to(dtype=torch.int32)
        mask = torch.ones_like(ids, dtype=torch.bool)
        try:
            torch.save({"input_ids": ids, "attention_mask": mask}, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        buckets = {
            key: value * self.seq_len for key, value in self.bucket_rows.items() if value
        }
        self.shards.append(
            {
                "path": str(path),
                "split": "train",
                "tokens": int(ids.numel()),
                "sequences": int(ids.size(0)),
                "buckets": buckets,
                "sha256": file_sha256(path),
                "storage_dtypes": {"input_ids": "int32", "attention_mask": "bool"},
            }
        )
        self.rows = []
        self.bucket_rows = defaultdict(int)

    def close(self) -> tuple[list[dict], int]:
        self._materialize_pending()
        self._flush_shard()
        discarded = int(self.tail.numel())
        self.tail = torch.empty(0, dtype=torch.int32)
        return self.shards, discarded


def copy_compact_split(
    source_dir: Path,
    output_dir: Path,
    split: str,
) -> list[dict]:
    source_dir = require_c_path(source_dir)
    output_dir = require_c_path(output_dir)
    manifest = load_manifest(source_dir)
    rows = []
    index = 0
    for entry in manifest["shards"]:
        if entry["split"] != split:
            continue
        source = resolve_shard(source_dir, entry)
        shard = safe_torch_load(source)
        ids = shard["input_ids"].to(dtype=torch.int32)
        source_mask = shard.get("attention_mask")
        mask = (
            torch.ones_like(ids, dtype=torch.bool)
            if source_mask is None
            else source_mask.to(dtype=torch.bool)
        )
        destination_dir = output_dir / "shards" / split
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / f"shard_{index:06d}.pt"
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            torch.save({"input_ids": ids, "attention_mask": mask}, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        rows.append(
            {
                "path": str(destination),
                "split": split,
                "tokens": int(ids.numel()),
                "sequences": int(ids.size(0)),
                "buckets": entry.get("buckets", {}),
                "sha256": file_sha256(destination),
                "storage_dtypes": {"input_ids": "int32", "attention_mask": "bool"},
            }
        )
        index += 1
    return rows


def prepare(args) -> dict:
    source = require_c_path(args.source_dataset)
    references = [require_c_path(path) for path in args.reference_dataset]
    validation_source = require_c_path(args.validation_source)
    output = require_c_path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(str(require_c_path(args.tokenizer)))
    eos_token_id = int(tokenizer.eos_token_id)
    source_manifest = load_manifest(source)
    reference_hashes: dict[str, set[str]] = defaultdict(set)
    reference_counts = {}
    for reference in references:
        hashes, counts = collect_document_hashes(reference, eos_token_id)
        reference_counts[str(reference)] = counts
        for bucket, values in hashes.items():
            reference_hashes[bucket].update(values)

    prefix_hashes, prefix_tokens_with_boundary = collect_prefix_hashes(
        source,
        args.consumed_prefix_bucket,
        args.consumed_prefix_rows,
        eos_token_id,
    )
    reference_hashes[args.consumed_prefix_bucket].update(prefix_hashes)

    writer = CompactShardWriter(output, int(source_manifest["seq_len"]), args.shard_tokens)
    kept_documents = defaultdict(int)
    kept_document_tokens = defaultdict(int)
    rejected_documents = defaultdict(int)
    rejected_document_tokens = defaultdict(int)
    seen_hashes: dict[str, set[str]] = defaultdict(set)
    for bucket, document in iter_documents(source, eos_token_id):
        if document.numel() < args.minimum_document_tokens:
            continue
        digest = document_hash(document)
        if digest in reference_hashes[bucket] or digest in seen_hashes[bucket]:
            rejected_documents[bucket] += 1
            rejected_document_tokens[bucket] += document.numel() + 1
            continue
        seen_hashes[bucket].add(digest)
        writer.add_document(bucket, document, eos_token_id)
        kept_documents[bucket] += 1
        kept_document_tokens[bucket] += document.numel() + 1

    train_shards, discarded_tail_tokens = writer.close()
    validation_shards = copy_compact_split(validation_source, output, "validation")
    test_shards = copy_compact_split(validation_source, output, "test")
    packed_train_tokens = sum(row["tokens"] for row in train_shards)
    manifest = {
        "created_by": "scripts/prepare_exact_novel_continuation.py",
        "format": "exact_novel_eos_document_v1",
        "source_dataset": str(source),
        "source_manifest_sha256": file_sha256(source / "manifest.json"),
        "reference_datasets": [str(path) for path in references],
        "reference_manifest_sha256": {
            str(path): file_sha256(path / "manifest.json") for path in references
        },
        "reference_document_counts": reference_counts,
        "consumed_prefix": {
            "bucket": args.consumed_prefix_bucket,
            "rows": args.consumed_prefix_rows,
            "tokens_without_boundary_extension": args.consumed_prefix_rows
            * int(source_manifest["seq_len"]),
            "tokens_with_boundary_extension": prefix_tokens_with_boundary,
            "document_hashes": len(prefix_hashes),
        },
        "method": "EOS-delimited int32 token-document BLAKE2b-128",
        "minimum_document_tokens": args.minimum_document_tokens,
        "tokenizer": str(require_c_path(args.tokenizer)),
        "tokenizer_vocab_size": len(tokenizer),
        "eos_token_id": eos_token_id,
        "seq_len": int(source_manifest["seq_len"]),
        "bucket_order": list(source_manifest.get("bucket_order") or source_manifest["plan"]),
        "kept_documents": dict(kept_documents),
        "kept_document_tokens": dict(kept_document_tokens),
        "rejected_documents": dict(rejected_documents),
        "rejected_document_tokens": dict(rejected_document_tokens),
        "packed_tokens": packed_train_tokens,
        "discarded_tail_tokens": discarded_tail_tokens,
        "storage_dtypes": {"input_ids": "int32", "attention_mask": "bool"},
        "validation_source": str(validation_source),
        "validation_source_manifest_sha256": file_sha256(
            validation_source / "manifest.json"
        ),
        "shards": train_shards + validation_shards + test_shards,
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=ROOT / "prepared_data" / "v3_language_50m_24k_ctx2048",
    )
    parser.add_argument(
        "--reference-dataset",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--validation-source",
        type=Path,
        default=ROOT
        / "prepared_data"
        / "v3_broad_additional_300m_combined_v2_24k_ctx2048",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "prepared_data" / "v3_exact_novel_remaining_after_500m",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "tokenizers" / "fdt_v3_bpe_24k",
    )
    parser.add_argument("--consumed-prefix-bucket", default="cosmopedia")
    parser.add_argument("--consumed-prefix-rows", type=int, default=6864)
    parser.add_argument("--minimum-document-tokens", type=int, default=8)
    parser.add_argument("--shard-tokens", type=int, default=10_000_000)
    args = parser.parse_args()
    if not args.reference_dataset:
        args.reference_dataset = [
            ROOT
            / "prepared_data"
            / "v3_broad_additional_300m_combined_v2_24k_ctx2048",
            ROOT / "prepared_data" / "v3_broad_200m_24k_ctx2048",
        ]
    manifest = prepare(args)
    print(json.dumps({
        "output": str(args.output_dir.resolve()),
        "packed_tokens": manifest["packed_tokens"],
        "kept_documents": manifest["kept_documents"],
        "rejected_documents": manifest["rejected_documents"],
    }, indent=2))


if __name__ == "__main__":
    main()
