from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(item for item in path.glob("*") if item.is_file()):
        if file.name == "tokenizer_validation.json":
            continue
        digest.update(file.read_bytes())
    return digest.hexdigest()


def inspect(name: str, path: Path) -> dict:
    checkpoint = safe_load(path)
    mask_tokens = checkpoint.get("tokens_seen")
    return {
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_origin": name,
        "global_step": int(checkpoint.get("optimizer_step", 0)),
        "optimizer_step": int(checkpoint.get("optimizer_step", 0)),
        "micro_step": int(checkpoint.get("micro_step", 0)),
        "effective_tokens_seen": int(mask_tokens) if mask_tokens is not None else None,
        "valid_non_padding_tokens_seen": int(mask_tokens) if mask_tokens is not None else None,
        "sequence_index": int(checkpoint.get("sequence_index", 0)),
        "training_corpus": "prepared_data/fdt_rlm_10m",
        "best_val_loss": float(checkpoint.get("best_val_loss", float("nan"))),
        "metadata_source": "checkpoint",
    }


def main() -> None:
    tokenizer = ROOT / "tokenizers" / "fdt_bpe"
    payload = {
        "transformer": inspect("untouched Transformer best", ROOT / "runs" / "full_transformer" / "best.pt"),
        "fdt": inspect("corrected FDT best after entropy repair", ROOT / "runs" / "full_fdt" / "best.pt"),
        "tokenizer": {
            "path": str(tokenizer.relative_to(ROOT)),
            "tree_sha256": tree_hash(tokenizer),
            "expected_sha256": "fe9abda889d465b78f41100815ff9b7658d7cd49093664dd6b09b38d165f05b4",
        },
        "fairness_note": "FDT includes the additional entropy-repair continuation; future stages target equal cumulative effective tokens.",
    }
    payload["tokenizer"]["verified"] = payload["tokenizer"]["tree_sha256"] == payload["tokenizer"]["expected_sha256"]
    output = ROOT / "artifacts" / "current_effective_token_state.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "transformer_tokens": payload["transformer"]["effective_tokens_seen"], "fdt_tokens": payload["fdt"]["effective_tokens_seen"], "tokenizer_verified": payload["tokenizer"]["verified"]}, indent=2))


if __name__ == "__main__":
    main()
