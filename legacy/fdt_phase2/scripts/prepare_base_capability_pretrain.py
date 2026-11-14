from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.data.manifest import file_sha256


def ensure_c_workspace_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"Only C: storage is allowed: {resolved}")
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"Path must remain inside {ROOT}: {resolved}") from error
    return resolved


def load_documents(source: Path, split: str) -> tuple[list[list[int]], list[Path]]:
    documents = []
    paths = sorted((source / "shards" / split).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No {split} shards found under {source}")
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        ids = payload["input_ids"]
        masks = payload["attention_mask"].bool()
        for row, mask in zip(ids, masks):
            document = row[mask].to(dtype=torch.int64).tolist()
            if document:
                documents.append(document)
    return documents, paths


def pack_documents(
    documents: list[list[int]], seq_len: int, pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    flattened = [token for document in documents for token in document]
    if not flattened:
        raise ValueError("No active tokens to pack")
    rows = math.ceil(len(flattened) / seq_len)
    padded = flattened + [pad_token_id] * (rows * seq_len - len(flattened))
    mask = [1] * len(flattened) + [0] * (rows * seq_len - len(flattened))
    return (
        torch.tensor(padded, dtype=torch.int32).reshape(rows, seq_len),
        torch.tensor(mask, dtype=torch.uint8).reshape(rows, seq_len),
    )


def document_hash(document: list[int]) -> str:
    tensor = torch.tensor(document, dtype=torch.int32)
    return hashlib.blake2b(tensor.numpy().tobytes(), digest_size=16).hexdigest()


def write_split(
    source: Path,
    output: Path,
    split: str,
    seq_len: int,
    pad_token_id: int,
    seed: int,
) -> dict:
    documents, source_paths = load_documents(source, split)
    random.Random(seed).shuffle(documents)
    ids, masks = pack_documents(documents, seq_len, pad_token_id)
    directory = output / "shards" / split
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "shard_000000.pt"
    temporary = path.with_suffix(".pt.tmp")
    torch.save({"input_ids": ids, "attention_mask": masks}, temporary)
    temporary.replace(path)
    return {
        "split": split,
        "path": str(path.relative_to(ROOT)),
        "sha256": file_sha256(path),
        "source_shards": [
            {"path": str(item.relative_to(ROOT)), "sha256": file_sha256(item)}
            for item in source_paths
        ],
        "documents": len(documents),
        "unique_documents": len({document_hash(document) for document in documents}),
        "sequences": int(ids.size(0)),
        "active_tokens": int(masks.sum()),
        "padded_tokens": int(ids.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert answer-masked capability examples into full-sequence base causal pretraining."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "prepared_data" / "capability_curriculum_5k",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "prepared_data" / "v3_base_capability_pilot_640k_ctx2048",
    )
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--pad_token_id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()

    source = ensure_c_workspace_path(args.source)
    output = ensure_c_workspace_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    split_reports = {
        split: write_split(
            source,
            output,
            split,
            args.seq_len,
            args.pad_token_id,
            args.seed + index,
        )
        for index, split in enumerate(("train", "validation"))
    }
    train_tokens = split_reports["train"]["active_tokens"]
    manifest = {
        "format": "base_causal_pretraining_v1",
        "purpose": "Short base-format pilot for factual QA, retrieval, Python, and JSON continuation",
        "assistant_masking": False,
        "labels": "implicit next-token labels over every active input token",
        "source": str(source.relative_to(ROOT)),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_format": source_manifest.get("format"),
        "tokenizer": source_manifest.get("tokenizer"),
        "tokenizer_vocab_size": 24576,
        "seq_len": args.seq_len,
        "requested_tokens": train_tokens,
        "packed_tokens": train_tokens,
        "plan": {"base_capability": train_tokens},
        "bucket_order": ["base_capability"],
        "tokens_by_bucket": {"base_capability": train_tokens},
        "source_categories": {
            split: source_manifest[split]["categories"] for split in ("train", "validation")
        },
        "seed": args.seed,
        "shards": [split_reports["train"], split_reports["validation"]],
    }
    manifest_path = output / "manifest.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
