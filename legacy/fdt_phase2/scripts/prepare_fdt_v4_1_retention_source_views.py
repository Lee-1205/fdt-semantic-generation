from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


CATEGORY_TO_SOURCE = {
    "natural_lm": "natural",
    "factual_qa": "factual",
    "retrieval": "retrieval",
    "python_code": "python",
    "json": "json",
}


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


def row_identity_digest(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for row, mask in zip(input_ids, attention_mask.bool()):
        active = row[mask].to(dtype=torch.int32).contiguous()
        row_digest = hashlib.blake2b(
            active.numpy().tobytes(), digest_size=16
        ).digest()
        digest.update(row_digest)
    return digest.hexdigest().upper()


def select_category(payload: dict[str, torch.Tensor], category_id: int) -> dict[str, torch.Tensor]:
    category_ids = payload.get("category_ids")
    if not isinstance(category_ids, torch.Tensor) or category_ids.ndim != 1:
        raise ValueError("source shard lacks one-dimensional category_ids")
    indices = category_ids.eq(int(category_id)).nonzero().flatten()
    if indices.numel() == 0:
        raise ValueError(f"category {category_id} has no rows")
    rows = int(category_ids.numel())
    selected = {
        name: value.index_select(0, indices).clone()
        for name, value in payload.items()
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == rows
    }
    if set(("input_ids", "labels", "attention_mask")) - set(selected):
        raise ValueError("category view lost required language-model fields")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Build immutable FDT v4.1 category source views")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = require_c_path(args.source)
    output = require_c_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    category_names = list(source_manifest["train"]["category_names"])
    if set(category_names) != set(CATEGORY_TO_SOURCE):
        raise ValueError(f"unexpected source categories: {category_names}")
    source_shard = require_c_path(Path(source_manifest["train"]["shard"]))
    payload = torch.load(source_shard, map_location="cpu", weights_only=True, mmap=True)

    records = {}
    all_identity_digests = set()
    for category_name, source_name in CATEGORY_TO_SOURCE.items():
        category_id = category_names.index(category_name)
        selected = select_category(payload, category_id)
        directory = output / source_name
        shard_dir = directory / "shards" / "train"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard = shard_dir / "shard_00000.pt"
        temporary = shard.with_name(shard.name + ".tmp")
        torch.save(selected, temporary)
        os.replace(temporary, shard)
        identity = row_identity_digest(
            selected["input_ids"], selected["attention_mask"]
        )
        if identity in all_identity_digests:
            raise RuntimeError("category source views share an ordered identity digest")
        all_identity_digests.add(identity)
        record = {
            "source_name": source_name,
            "source_category": category_name,
            "source_category_id": category_id,
            "file": str(shard.relative_to(directory)),
            "rows": int(selected["input_ids"].size(0)),
            "active_tokens": int(selected["attention_mask"].sum()),
            "sha256": file_sha256(shard),
            "row_identity_sha256": identity,
            "fields": sorted(selected),
        }
        records[source_name] = record
        atomic_json(
            directory / "manifest.json",
            {
                "schema_version": "fdt_v4_1_retention_category_view_v1",
                "split": "train",
                "source_dataset": str(source),
                "source_manifest_sha256": file_sha256(source_manifest_path),
                "source_shard_sha256": file_sha256(source_shard),
                "category": record,
            },
        )

    if sum(record["rows"] for record in records.values()) != int(
        source_manifest["train"]["rows"]
    ):
        raise RuntimeError("category views do not cover every source row exactly once")
    report = {
        "schema": "fdt_v4_1_retention_source_views_v1",
        "source_dataset": str(source),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_shard_sha256": file_sha256(source_shard),
        "records": records,
        "rows": sum(record["rows"] for record in records.values()),
        "active_tokens": sum(record["active_tokens"] for record in records.values()),
        "temporary_residue": 0,
    }
    atomic_json(output / "manifest.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
