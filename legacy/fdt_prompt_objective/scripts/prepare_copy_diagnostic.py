from __future__ import annotations

import argparse
import json
import random
import string
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.tokenization import load_tokenizer


def random_code(rng: random.Random) -> str:
    prefix = "".join(rng.choices(string.ascii_uppercase, k=2))
    digits = "".join(rng.choices(string.digits, k=5))
    return prefix + digits


def make_example(tokenizer, code: str, instruction_style: bool, max_length: int):
    if instruction_style:
        prompt = (
            f"Context: The verification code is {code}.\n"
            "Question: What is the verification code?\nAnswer:"
        )
    else:
        prompt = f"Record: The verification code is {code}. Therefore the verification code is"
    answer = f" {code}"
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    answer_ids = list(tokenizer.encode(answer, add_special_tokens=False)) + [int(tokenizer.eos_token_id)]
    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_length:
        raise ValueError(f"diagnostic example exceeds max_length={max_length}")
    labels = [-100] * len(prompt_ids) + answer_ids
    padding = max_length - len(input_ids)
    pad_id = int(tokenizer.pad_token_id)
    return (
        input_ids + [pad_id] * padding,
        labels + [-100] * padding,
        [1] * len(input_ids) + [0] * padding,
    )


def write_split(output: Path, split: str, tokenizer, rows: int, max_length: int, seed: int):
    rng = random.Random(seed)
    seen = set()
    ids, labels, masks = [], [], []
    while len(ids) < rows:
        code = random_code(rng)
        if code in seen:
            continue
        seen.add(code)
        example = make_example(tokenizer, code, len(ids) % 2 == 1, max_length)
        input_ids, target_ids, attention_mask = example
        ids.append(input_ids)
        labels.append(target_ids)
        masks.append(attention_mask)
    shard_dir = output / "shards" / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_ids": torch.tensor(ids, dtype=torch.int32),
            "labels": torch.tensor(labels, dtype=torch.int32),
            "attention_mask": torch.tensor(masks, dtype=torch.uint8),
        },
        shard_dir / "shard_000000.pt",
    )
    return {"rows": rows, "unique_codes": len(seen), "seed": seed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a held-out exact-copy diagnostic for FDT")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "prepared_data" / "copy_diagnostic_4k"
    )
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizers" / "fdt_v3_bpe_24k"))
    parser.add_argument("--train-rows", type=int, default=4096)
    parser.add_argument("--validation-rows", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer)
    train = write_split(output, "train", tokenizer, args.train_rows, args.max_length, args.seed)
    validation = write_split(
        output, "validation", tokenizer, args.validation_rows, args.max_length, args.seed + 1
    )
    manifest = {
        "format": "assistant_only_copy_diagnostic_v1",
        "purpose": "Distinguish curriculum scarcity from identity-preserving path failure",
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "train": train,
        "validation": validation,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
