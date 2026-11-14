from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
CATEGORY_TO_OUTPUT = {
    "natural_lm": "natural",
    "factual_qa": "factual",
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


def token_hash(tokens: torch.Tensor) -> str:
    active = tokens.to(dtype=torch.int32).contiguous()
    return hashlib.blake2b(active.numpy().tobytes(), digest_size=16).hexdigest()


def active_hashes(dataset_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for shard in sorted((dataset_dir / "shards" / "train").glob("*.pt")):
        payload = torch.load(shard, map_location="cpu", weights_only=True, mmap=True)
        ids = payload["input_ids"]
        masks = payload["attention_mask"].bool()
        for row, mask in zip(ids, masks):
            hashes.add(token_hash(row[mask]))
    return hashes


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-natural", type=Path, required=True)
    parser.add_argument("--reference-factual", type=Path, required=True)
    args = parser.parse_args()

    source = require_c_path(args.source)
    output = require_c_path(args.output)
    references = [
        require_c_path(args.reference_natural),
        require_c_path(args.reference_factual),
    ]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_manifest_path = source / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    category_names = list(source_manifest["train"]["category_names"])
    source_shard = require_c_path(Path(source_manifest["train"]["shard"]))
    source_payload = torch.load(
        source_shard, map_location="cpu", weights_only=True, mmap=True
    )
    reference_hashes = set()
    for reference in references:
        reference_hashes.update(active_hashes(reference))

    source_records: dict[str, dict] = {}
    selected_hashes: set[str] = set()
    exact_overlap = 0
    local_duplicates = 0
    for category_name, output_name in CATEGORY_TO_OUTPUT.items():
        category_id = category_names.index(category_name)
        indices = source_payload["category_ids"].eq(category_id).nonzero().flatten()
        selected = {
            name: source_payload[name].index_select(0, indices).clone()
            for name in ("input_ids", "labels", "attention_mask")
        }
        masks = selected["attention_mask"].bool()
        category_hashes: set[str] = set()
        for row, mask in zip(selected["input_ids"], masks):
            digest = token_hash(row[mask])
            exact_overlap += int(digest in reference_hashes)
            local_duplicates += int(digest in selected_hashes or digest in category_hashes)
            category_hashes.add(digest)
        selected_hashes.update(category_hashes)

        directory = output / output_name
        shard_dir = directory / "shards" / "train"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard = shard_dir / "shard_00000.pt"
        temporary = shard.with_name(shard.name + ".tmp")
        torch.save(selected, temporary)
        os.replace(temporary, shard)
        record = {
            "file": str(shard.relative_to(directory)),
            "rows": int(indices.numel()),
            "sha256": file_sha256(shard),
            "tokens": int(selected["attention_mask"].sum()),
            "sequence_length": int(selected["input_ids"].shape[1]),
            "source_category": category_name,
            "source_category_id": category_id,
        }
        source_records[output_name] = record
        atomic_json(
            directory / "manifest.json",
            {
                "name": output_name,
                "schema_version": "fdt_v4_1_stage_b_novel_source_v1",
                "split": "train",
                "shards": [record],
                "source_manifest_sha256": file_sha256(source_manifest_path),
                "source_shard_sha256": file_sha256(source_shard),
                "reference_dataset_sha256": {
                    reference.name: file_sha256(reference / "manifest.json")
                    for reference in references
                },
            },
        )

    if exact_overlap or local_duplicates:
        raise RuntimeError(
            f"novel subset contract failed: overlap={exact_overlap}, duplicates={local_duplicates}"
        )
    report = {
        "schema": "fdt_v4_1_stage_b_novel_subset_v1",
        "source": str(source),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_shard_sha256": file_sha256(source_shard),
        "references": {
            reference.name: file_sha256(reference / "manifest.json")
            for reference in references
        },
        "sources": source_records,
        "selected_rows": sum(record["rows"] for record in source_records.values()),
        "selected_active_tokens": sum(
            record["tokens"] for record in source_records.values()
        ),
        "exact_overlap_with_stage_a": exact_overlap,
        "local_exact_duplicates": local_duplicates,
        "conversational_sft": False,
    }
    atomic_json(output / "manifest.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
