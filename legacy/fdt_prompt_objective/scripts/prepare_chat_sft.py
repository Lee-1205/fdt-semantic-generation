from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.tokenization import load_tokenizer


@dataclass(frozen=True)
class SourceSpec:
    name: str
    config: str
    split: str
    target_tokens: int


TRAIN_SOURCE_PROFILES = {
    "broad": (
        ("HuggingFaceTB/smoltalk", "everyday-conversations", "train", 0.25),
        ("HuggingFaceH4/ultrachat_200k", "default", "train_sft", 0.50),
        ("allenai/tulu-3-sft-mixture", "default", "train", 0.25),
    ),
    "pilot_quality": (
        ("HuggingFaceTB/smoltalk", "everyday-conversations", "train", 0.40),
        ("HuggingFaceH4/ultrachat_200k", "default", "train_sft", 0.60),
    ),
}

VALIDATION_SOURCES = (
    ("HuggingFaceTB/smoltalk", "everyday-conversations", "test"),
    ("HuggingFaceH4/ultrachat_200k", "default", "test_sft"),
)

ROLE_HEADERS = {
    "system": "### System:\n",
    "user": "### User:\n",
    "assistant": "### Assistant:\n",
}


def clean_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    cleaned = []
    for message in row.get("messages") or []:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if role in ROLE_HEADERS and content:
            cleaned.append({"role": role, "content": content})
    return cleaned


def likely_english(text: str) -> bool:
    if not text:
        return False
    printable = sum(character.isascii() and (character.isprintable() or character.isspace()) for character in text)
    return printable / max(len(text), 1) >= 0.85


def conversation_key(messages: list[dict[str, str]]) -> bytes:
    compact = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(compact.encode("utf-8"), digest_size=16).digest()


def format_history(messages: list[dict[str, str]]) -> str:
    pieces = []
    for message in messages:
        pieces.append(f"{ROLE_HEADERS[message['role']]}{message['content']}\n")
    return "\n".join(pieces)


def assistant_examples(
    messages: list[dict[str, str]],
    tokenizer,
    max_length: int,
    max_answer_tokens: int,
) -> Iterable[tuple[list[int], list[int], int]]:
    for index, message in enumerate(messages):
        if message["role"] != "assistant" or index == 0:
            continue
        answer_ids = list(tokenizer.encode(message["content"], add_special_tokens=False))
        if len(answer_ids) < 8:
            continue
        answer_ids = answer_ids[:max_answer_tokens] + [int(tokenizer.eos_token_id)]
        prompt = format_history(messages[:index]) + "\n### Assistant:\n"
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        prompt_budget = max(max_length - len(answer_ids), 1)
        prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        yield input_ids, labels, len(answer_ids)


class ShardWriter:
    def __init__(self, root: Path, split: str, max_length: int, pad_id: int, shard_rows: int):
        self.directory = root / "shards" / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        self.pad_id = pad_id
        self.shard_rows = shard_rows
        self.ids: list[list[int]] = []
        self.labels: list[list[int]] = []
        self.shard_index = 0
        self.rows = 0
        self.sequence_tokens = 0
        self.target_tokens = 0

    def add(self, input_ids: list[int], labels: list[int], target_tokens: int) -> None:
        padding = self.max_length - len(input_ids)
        if padding < 0:
            raise ValueError("chat example exceeds max_length")
        self.ids.append(input_ids + [self.pad_id] * padding)
        self.labels.append(labels + [-100] * padding)
        self.rows += 1
        self.sequence_tokens += len(input_ids)
        self.target_tokens += target_tokens
        if len(self.ids) >= self.shard_rows:
            self.flush()

    def flush(self) -> None:
        if not self.ids:
            return
        input_ids = torch.tensor(self.ids, dtype=torch.int32)
        labels = torch.tensor(self.labels, dtype=torch.int32)
        payload = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.pad_id).to(torch.uint8),
        }
        torch.save(payload, self.directory / f"shard_{self.shard_index:06d}.pt")
        self.shard_index += 1
        self.ids.clear()
        self.labels.clear()

    def summary(self) -> dict[str, int]:
        self.flush()
        return {
            "rows": self.rows,
            "shards": self.shard_index,
            "sequence_tokens": self.sequence_tokens,
            "target_tokens": self.target_tokens,
        }


def stream_rows(name: str, config: str, split: str, seed: int):
    dataset = load_dataset(name, config, split=split, streaming=True)
    return dataset.shuffle(seed=seed, buffer_size=10_000)


def prepare_train(args, tokenizer, writer: ShardWriter, seen: set[bytes]) -> dict[str, Any]:
    source_reports = []
    train_sources = TRAIN_SOURCE_PROFILES[args.source_profile]
    for source_index, (name, config, split, ratio) in enumerate(train_sources):
        target = int(args.target_assistant_tokens * ratio)
        accepted_rows = 0
        accepted_conversations = 0
        rejected = Counter()
        start_tokens = writer.target_tokens
        for row in stream_rows(name, config, split, args.seed + source_index):
            messages = clean_messages(row)
            if len(messages) < 2:
                rejected["short_conversation"] += 1
                continue
            text = "\n".join(message["content"] for message in messages)
            if not likely_english(text):
                rejected["non_english"] += 1
                continue
            key = conversation_key(messages)
            if key in seen:
                rejected["duplicate"] += 1
                continue
            examples = list(
                assistant_examples(messages, tokenizer, args.max_length, args.max_answer_tokens)
            )
            if not examples:
                rejected["no_usable_answer"] += 1
                continue
            seen.add(key)
            accepted_conversations += 1
            for input_ids, labels, target_tokens in examples:
                writer.add(input_ids, labels, target_tokens)
                accepted_rows += 1
                if writer.target_tokens - start_tokens >= target:
                    break
            if writer.target_tokens - start_tokens >= target:
                break
        source_reports.append(
            {
                "dataset": name,
                "config": config,
                "split": split,
                "target_assistant_tokens": target,
                "actual_assistant_tokens": writer.target_tokens - start_tokens,
                "conversations": accepted_conversations,
                "examples": accepted_rows,
                "rejected": dict(rejected),
            }
        )
    return {"sources": source_reports, **writer.summary()}


def prepare_validation(args, tokenizer, writer: ShardWriter, seen: set[bytes]) -> dict[str, Any]:
    per_source = max(args.validation_examples // len(VALIDATION_SOURCES), 1)
    reports = []
    for source_index, (name, config, split) in enumerate(VALIDATION_SOURCES):
        start_rows = writer.rows
        accepted_conversations = 0
        for row in stream_rows(name, config, split, args.seed + 100 + source_index):
            messages = clean_messages(row)
            if len(messages) < 2 or not likely_english("\n".join(m["content"] for m in messages)):
                continue
            key = conversation_key(messages)
            if key in seen:
                continue
            examples = list(
                assistant_examples(messages, tokenizer, args.max_length, args.max_answer_tokens)
            )
            if not examples:
                continue
            seen.add(key)
            accepted_conversations += 1
            for input_ids, labels, target_tokens in examples:
                writer.add(input_ids, labels, target_tokens)
                if writer.rows - start_rows >= per_source:
                    break
            if writer.rows - start_rows >= per_source:
                break
        reports.append(
            {
                "dataset": name,
                "config": config,
                "split": split,
                "conversations": accepted_conversations,
                "examples": writer.rows - start_rows,
            }
        )
    return {"sources": reports, **writer.summary()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare assistant-only English chat SFT shards")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prepared_data" / "chat_sft_20m")
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizers" / "fdt_bpe"))
    parser.add_argument("--target-assistant-tokens", type=int, default=20_000_000)
    parser.add_argument(
        "--source-profile",
        choices=sorted(TRAIN_SOURCE_PROFILES),
        default="broad",
        help="Dataset mixture. pilot_quality excludes the broader Tulu mixture.",
    )
    parser.add_argument("--validation-examples", type=int, default=2_000)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-answer-tokens", type=int, default=384)
    parser.add_argument("--shard-rows", type=int, default=2_048)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer)
    seen: set[bytes] = set()
    train_writer = ShardWriter(output, "train", args.max_length, int(tokenizer.pad_token_id), args.shard_rows)
    validation_writer = ShardWriter(
        output, "validation", args.max_length, int(tokenizer.pad_token_id), args.shard_rows
    )
    train_report = prepare_train(args, tokenizer, train_writer, seen)
    validation_report = prepare_validation(args, tokenizer, validation_writer, seen)
    report = {
        "format": "assistant_only_chat_sft_v1",
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "max_answer_tokens": args.max_answer_tokens,
        "source_profile": args.source_profile,
        "seed": args.seed,
        "train": train_report,
        "validation": validation_report,
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
