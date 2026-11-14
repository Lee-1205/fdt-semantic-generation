from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.data.interleaving import (  # noqa: E402
    bucket_sequence_counts,
    interleave_remaining_indices,
    source_counts_for_prefix,
)


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def longest_source_run(order: torch.Tensor, counts: dict[str, int]) -> int:
    boundaries = torch.tensor(list(counts.values()), dtype=torch.long).cumsum(0)[:-1]
    labels = torch.bucketize(order, boundaries, right=True)
    if labels.numel() == 0:
        return 0
    changes = torch.cat((torch.tensor([True]), labels[1:] != labels[:-1], torch.tensor([True])))
    positions = changes.nonzero(as_tuple=False).flatten()
    return int((positions[1:] - positions[:-1]).max().item())


def main() -> None:
    dataset_dir = ROOT / "prepared_data" / "fdt_rlm_100m"
    output_dir = ROOT / "prepared_data" / "fdt_rlm_remaining_interleaved_50m"
    source_manifest_path = dataset_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    checkpoints = {
        model: safe_load(ROOT / "runs" / "staged_50m" / model / "latest.pt")
        for model in ("transformer", "fdt")
    }

    cursors = {model: int(checkpoint["sequence_index"]) for model, checkpoint in checkpoints.items()}
    shared_prefix = max(cursors.values())
    counts = bucket_sequence_counts(source_manifest)
    remaining_order = interleave_remaining_indices(counts, shared_prefix)
    required_steps = {
        model: math.ceil((100_000_000 - int(checkpoint["tokens_seen"])) / (8 * 512))
        for model, checkpoint in checkpoints.items()
    }
    if len(set(required_steps.values())) != 1:
        raise RuntimeError(f"Models require different recovery lengths: {required_steps}")
    selected_sequences = next(iter(required_steps.values())) * 8
    indices = remaining_order[:selected_sequences].contiguous()
    if indices.unique().numel() != indices.numel() or int(indices.min()) < shared_prefix:
        raise RuntimeError("Recovery index contains replay or duplicate rows")

    tokenizer_hashes = {checkpoint["tokenizer_hash"] for checkpoint in checkpoints.values()}
    manifest_hashes = {checkpoint["dataset_manifest_sha256"] for checkpoint in checkpoints.values()}
    if len(tokenizer_hashes) != 1 or len(manifest_hashes) != 1:
        raise RuntimeError("Trusted checkpoints do not share tokenizer/dataset identity")
    if next(iter(manifest_hashes)) != file_hash(source_manifest_path):
        raise RuntimeError("Prepared dataset manifest hash differs from the trusted checkpoints")

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "train_indices.pt"
    torch.save({"indices": indices}, index_path)
    selected_counts = source_counts_for_prefix(indices, counts, indices.numel())
    manifest = {
        "format": "virtual_index_v1",
        "source_dataset": str(dataset_dir.relative_to(ROOT)),
        "source_manifest_sha256": file_hash(source_manifest_path),
        "tokenizer_hash": next(iter(tokenizer_hashes)),
        "trusted_checkpoints": {
            model: str((ROOT / "runs" / "staged_50m" / model / "latest.pt").relative_to(ROOT))
            for model in checkpoints
        },
        "checkpoint_sequence_cursors": cursors,
        "shared_excluded_prefix": shared_prefix,
        "optimizer_steps": next(iter(required_steps.values())),
        "sequences": int(indices.numel()),
        "tokens": int(indices.numel() * int(source_manifest["seq_len"])),
        "seq_len": int(source_manifest["seq_len"]),
        "source_sequence_counts": selected_counts,
        "source_ratios": {
            source: count / max(int(indices.numel()), 1)
            for source, count in selected_counts.items()
        },
        "unique_indices": int(indices.unique().numel()),
        "replayed_indices": int((indices < shared_prefix).sum().item()),
        "longest_single_source_run_sequences": longest_source_run(indices, counts),
        "index_file": index_path.name,
        "index_sha256": file_hash(index_path),
        "validation": {
            "status": "PASS",
            "same_stream_for_both_models": True,
            "duplicates": 0,
            "source_manifest_match": True,
            "tokenizer_match": True,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
