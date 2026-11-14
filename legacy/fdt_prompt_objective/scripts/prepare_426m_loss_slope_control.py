from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.tokenization import load_tokenizer


EXPERIMENT_NONCE = "fdt-transformer-control-8f6c4a1e2d9b73c05a41e687bd3209fa"
CATEGORIES = ("natural", "factual", "retrieval", "python", "json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_digest(row: torch.Tensor) -> str:
    return hashlib.sha256(row.numpy().tobytes()).hexdigest()


def fragment(category: str, rng: random.Random, row_index: int, fragment_index: int) -> str:
    entity = f"entity_{rng.randrange(10_000_000):07d}"
    key = f"key_{rng.randrange(10_000_000):07d}"
    value = rng.randrange(1_000_000)
    if category == "natural":
        return (
            f"The field report for {entity} describes a measured value of {value}. "
            "The observer compares the result with an earlier trial, explains the uncertainty, "
            "and records the conclusion in plain language for a later review.\n"
        )
    if category == "factual":
        return f"Fact: the registry code for {entity} is {value}. Answer: {value}.\n"
    if category == "retrieval":
        return (
            f"Context record {fragment_index}: {key} maps to {value}. Other records are distractors. "
            f"Question: what value maps to {key}? Answer: {value}.\n"
        )
    if category == "python":
        offset = rng.randrange(1, 100)
        return (
            f"def transform_{row_index}_{fragment_index}(items):\n"
            f"    return [item + {offset} for item in items if item % 2 == 0]\n"
        )
    if category == "json":
        active = "true" if value % 2 else "false"
        return f'{{"id":"{entity}","key":"{key}","value":{value},"active":{active}}}\n'
    raise ValueError(category)


def build_row(tokenizer, split: str, row_index: int, seq_len: int, seed: int) -> list[int]:
    rng = random.Random(seed + row_index * 1_000_003)
    marker = f"CONTROL_NONCE={EXPERIMENT_NONCE}; split={split}; row={row_index:06d}.\n"
    text = marker
    fragment_index = 0
    while True:
        category = CATEGORIES[(row_index + fragment_index) % len(CATEGORIES)]
        text += fragment(category, rng, row_index, fragment_index)
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) >= seq_len - 1:
            break
        fragment_index += 1
    ids = ids[: seq_len - 1] + [int(tokenizer.eos_token_id)]
    if len(ids) != seq_len:
        raise RuntimeError("packed row length mismatch")
    return [int(value) for value in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare nonce-separated 426M loss-slope control rows.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", default="tokenizers/fdt_v3_bpe_24k")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--train-rows", type=int, default=512)
    parser.add_argument("--heldout-rows", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026081400)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer)
    train = torch.tensor(
        [build_row(tokenizer, "train", i, args.seq_len, args.seed) for i in range(args.train_rows)],
        dtype=torch.int32,
    )
    heldout = torch.tensor(
        [build_row(tokenizer, "heldout", i, args.seq_len, args.seed + 900_000_000) for i in range(args.heldout_rows)],
        dtype=torch.int32,
    )
    train_path = output / "train.pt"
    heldout_path = output / "heldout.pt"
    torch.save({"input_ids": train}, train_path)
    torch.save({"input_ids": heldout}, heldout_path)
    train_hashes = [row_digest(row) for row in train]
    heldout_hashes = [row_digest(row) for row in heldout]
    if len(set(train_hashes + heldout_hashes)) != len(train_hashes) + len(heldout_hashes):
        raise RuntimeError("local exact-row collision")
    manifest = {
        "format": "nonce_separated_loss_slope_control_v1",
        "purpose": "equal-data early loss-slope control, not capability training",
        "experiment_nonce": EXPERIMENT_NONCE,
        "novelty_claim": {
            "available_payload_scan": False,
            "reason": "some historical consumed payloads were pruned after hash journaling",
            "construction": "every row begins with a new experiment nonce plus a unique split/row marker",
            "cryptographic_collision_claim": "not mathematically impossible, treated as negligible for exact full-row equality",
        },
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "seq_len": args.seq_len,
        "train_rows": args.train_rows,
        "heldout_rows": args.heldout_rows,
        "train_tokens": int(train.numel()),
        "heldout_tokens": int(heldout.numel()),
        "seed": args.seed,
        "categories": list(CATEGORIES),
        "train_sha256": sha256(train_path),
        "heldout_sha256": sha256(heldout_path),
        "train_row_hashes_sha256": hashlib.sha256("\n".join(train_hashes).encode()).hexdigest(),
        "heldout_row_hashes_sha256": hashlib.sha256("\n".join(heldout_hashes).encode()).hexdigest(),
        "local_exact_overlap": 0,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({**manifest, "manifest_sha256": sha256(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
