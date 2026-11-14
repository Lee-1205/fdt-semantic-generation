from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_c_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"{label} must stay on C: {resolved}")
    return resolved


def canonical_row_hash(payload: dict[str, Any], index: int) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    fields: list[dict[str, Any]] = []
    for name in sorted(payload):
        value = payload[name]
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value[index].detach().cpu().contiguous()
        metadata = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        raw = tensor.view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        fields.append(metadata)
    return digest.hexdigest().upper(), fields


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset_dir = require_c_path(args.dataset_dir, "validation dataset")
    output = require_c_path(args.output, "identity artifact")
    if output.exists():
        raise FileExistsError(f"identity artifact already exists: {output}")
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = list(manifest.get("shards", []))
    if not shards:
        raise ValueError("validation manifest has no shards")

    epoch = shard_index = cursor = 0
    payload: dict[str, Any] | None = None
    order: torch.Tensor | None = None
    shard_path: Path | None = None
    selected: list[dict[str, Any]] = []

    for ordinal in range(int(args.samples)):
        if payload is None or order is None or cursor >= order.numel():
            if payload is not None:
                shard_index += 1
                cursor = 0
                if shard_index >= len(shards):
                    shard_index = 0
                    epoch += 1
            shard = shards[shard_index]
            shard_path = require_c_path(dataset_dir / shard["file"], "validation shard")
            expected_sha = str(shard["sha256"]).upper()
            actual_sha = sha256_file(shard_path)
            if actual_sha != expected_sha:
                raise ValueError(f"validation shard SHA-256 mismatch: {shard_path}")
            payload = torch.load(shard_path, map_location="cpu", weights_only=True)
            rows = int(payload["input_ids"].size(0))
            generator = torch.Generator(device="cpu").manual_seed(
                int(args.seed) + epoch * 1_000_003 + shard_index * 97
            )
            order = torch.randperm(rows, generator=generator)

        assert payload is not None and order is not None and shard_path is not None
        row_index = int(order[cursor])
        row_sha, fields = canonical_row_hash(payload, row_index)
        selected.append(
            {
                "ordinal": ordinal,
                "epoch": epoch,
                "shard_index": shard_index,
                "shard_file": str(shard_path.relative_to(dataset_dir)),
                "row_index": row_index,
                "row_sha256": row_sha,
                "fields": fields,
            }
        )
        cursor += 1

    identity_material = {
        "seed": int(args.seed),
        "samples": int(args.samples),
        "selected": selected,
    }
    identity_sha = hashlib.sha256(
        json.dumps(identity_material, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    artifact = {
        "schema": "fdt_fixed_validation_identity_v1",
        "status": "PASS",
        "dataset_dir": str(dataset_dir),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_manifest_sha256": manifest.get("source_manifest_sha256"),
        "split": manifest.get("split"),
        **identity_material,
        "sample_identity_sha256": identity_sha,
        "contract": "RowPool epoch-0 deterministic ordering and canonical tensor-row bytes",
    }
    atomic_json(output, artifact)
    print(json.dumps({"status": "PASS", "output": str(output), "sample_identity_sha256": identity_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
