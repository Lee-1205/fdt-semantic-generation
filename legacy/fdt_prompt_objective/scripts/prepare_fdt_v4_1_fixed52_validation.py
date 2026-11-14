from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.tokenization import load_tokenizer  # noqa: E402


PINNED_SOURCE_SHA256 = "7F3A28AA049904A4D715ACE2307CDF7A80E7D6E2EB65A707EAAD8764AB908E9E"
PINNED_TOKENIZER_SHA256 = "4FADE0C72E8673BA1E4319C8821593BB5E62A8C27792E2412DE3BDC52D232862"


def require_c_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"C-only contract violated: {resolved}")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def row_identity(row: dict, token_ids: list[int]) -> str:
    payload = json.dumps(
        {
            "category": row.get("category"),
            "prompt": row.get("prompt"),
            "target": row.get("target"),
            "input_ids": token_ids,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare immutable FDT v4.1 fixed52 validation")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = require_c_path(args.source)
    tokenizer_dir = require_c_path(args.tokenizer)
    output = require_c_path(args.output)
    if file_sha256(source) != PINNED_SOURCE_SHA256:
        raise ValueError("fixed52 source SHA-256 mismatch")
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    if file_sha256(tokenizer_json) != PINNED_TOKENIZER_SHA256:
        raise ValueError("tokenizer SHA-256 mismatch")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not fresh: {output}")
    shard_dir = output / "shards" / "validation"
    shard_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 52:
        raise ValueError(f"fixed validation requires 52 rows, found {len(rows)}")
    tokenizer = load_tokenizer(str(tokenizer_dir))
    encoded = [
        list(
            tokenizer.encode(
                str(row["prompt"]) + str(row["target"]),
                add_special_tokens=False,
            )
        )
        for row in rows
    ]
    if any(len(ids) < 2 for ids in encoded):
        raise ValueError("fixed validation row is too short")
    max_length = max(map(len, encoded))
    input_ids = torch.zeros((len(rows), max_length), dtype=torch.int32)
    attention_mask = torch.zeros((len(rows), max_length), dtype=torch.uint8)
    labels = torch.zeros((len(rows), max_length), dtype=torch.int32)
    identities = []
    categories = []
    for index, (row, ids) in enumerate(zip(rows, encoded)):
        length = len(ids)
        values = torch.tensor(ids, dtype=torch.int32)
        input_ids[index, :length] = values
        labels[index, :length] = values
        attention_mask[index, :length] = 1
        identities.append(row_identity(row, ids))
        categories.append(str(row.get("category", "unknown")))
    if len(set(identities)) != len(identities):
        raise RuntimeError("fixed52 row identity collision")

    shard = shard_dir / "shard_00000.pt"
    temporary = shard.with_name(shard.name + ".tmp")
    torch.save(
        {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        },
        temporary,
    )
    os.replace(temporary, shard)
    ordered_identity = hashlib.sha256("".join(identities).encode("ascii")).hexdigest().upper()
    manifest = {
        "schema_version": "fdt_v4_1_fixed52_dual_validation_v1",
        "purpose": "validation_only_never_training",
        "split": "validation",
        "source": str(source),
        "source_sha256": file_sha256(source),
        "tokenizer_json_sha256": file_sha256(tokenizer_json),
        "rows": len(rows),
        "max_sequence_length": max_length,
        "scored_next_tokens": sum(len(ids) - 1 for ids in encoded),
        "categories": categories,
        "row_identity_sha256": identities,
        "ordered_row_identity_sha256": ordered_identity,
        "shards": [
            {
                "file": str(shard.relative_to(output)),
                "rows": len(rows),
                "sha256": file_sha256(shard),
            }
        ],
    }
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
