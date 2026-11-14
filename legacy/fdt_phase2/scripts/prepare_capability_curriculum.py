from __future__ import annotations

import argparse
import json
import random
import string
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.tokenization import load_tokenizer


class CurriculumWriter:
    def __init__(self, root: Path, split: str, max_length: int, pad_id: int):
        self.directory = root / "shards" / split
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_length = max_length
        self.pad_id = pad_id
        self.ids: list[list[int]] = []
        self.labels: list[list[int]] = []
        self.masks: list[list[int]] = []
        self.categories = Counter()

    def add(self, tokenizer, prompt: str, answer: str, category: str, max_answer_tokens: int = 256):
        answer_ids = list(tokenizer.encode(" " + answer.strip(), add_special_tokens=False))
        answer_ids = answer_ids[:max_answer_tokens] + [int(tokenizer.eos_token_id)]
        prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
        prompt_ids = prompt_ids[-max(self.max_length - len(answer_ids), 1) :]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        padding = self.max_length - len(input_ids)
        self.ids.append(input_ids + [self.pad_id] * padding)
        self.labels.append(labels + [-100] * padding)
        self.masks.append([1] * len(input_ids) + [0] * padding)
        self.categories[category] += 1

    def save(self):
        payload = {
            "input_ids": torch.tensor(self.ids, dtype=torch.int32),
            "labels": torch.tensor(self.labels, dtype=torch.int32),
            "attention_mask": torch.tensor(self.masks, dtype=torch.uint8),
        }
        torch.save(payload, self.directory / "shard_000000.pt")
        return {"rows": len(self.ids), "categories": dict(self.categories)}


def crop_context(context: str, answer: str, radius: int = 1200) -> str:
    position = context.lower().find(answer.lower())
    if position < 0:
        return context[: radius * 2]
    start = max(position - radius, 0)
    end = min(position + len(answer) + radius, len(context))
    return context[start:end]


def add_squad(writer, tokenizer, split: str, rows: int, seed: int):
    dataset = load_dataset("rajpurkar/squad", split=split, streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    added = 0
    for row in dataset:
        answers = row.get("answers", {}).get("text", [])
        if not answers or not answers[0].strip():
            continue
        answer = answers[0].strip()
        context = crop_context(str(row["context"]), answer)
        prompt = f"Context: {context}\nQuestion: {row['question']}\nAnswer:"
        writer.add(tokenizer, prompt, answer, "factual_qa", max_answer_tokens=96)
        added += 1
        if added >= rows:
            break


def random_code(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=2)) + "".join(
        rng.choices(string.digits, k=6)
    )


def add_retrieval(writer, tokenizer, rows: int, seed: int):
    rng = random.Random(seed)
    seen = set()
    templates = (
        lambda code: (f"Record: The access code is {code}. Therefore the access code is", code),
        lambda code: (
            f"Context: The access code is {code}.\nQuestion: What is the access code?\nAnswer:",
            code,
        ),
        lambda code: (
            f"Memo A says ignore old codes. Memo B says the current identifier is {code}.\n"
            "Return the current identifier exactly:",
            code,
        ),
    )
    while len(seen) < rows:
        code = random_code(rng)
        if code in seen:
            continue
        seen.add(code)
        prompt, answer = templates[len(seen) % len(templates)](code)
        writer.add(tokenizer, prompt, answer, "retrieval", max_answer_tokens=32)


def add_mbpp(writer, tokenizer, split: str, rows: int, seed: int):
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split=split, streaming=True)
    examples = list(dataset)
    rng = random.Random(seed)
    rng.shuffle(examples)
    templates = (
        "Write valid Python code for this task:\n{prompt}\nPython:\n",
        "Complete the following programming task. Return only Python code.\n{prompt}\n",
    )
    for index in range(rows):
        row = examples[index % len(examples)]
        prompt = templates[(index // len(examples)) % len(templates)].format(prompt=row["prompt"])
        writer.add(tokenizer, prompt, row["code"], "python_code", max_answer_tokens=256)


def add_json(writer, tokenizer, rows: int, seed: int):
    rng = random.Random(seed)
    names = ("Ada", "Grace", "Linus", "Margaret", "Alan", "Edsger", "Barbara", "Donald")
    cities = ("London", "Paris", "Seoul", "Toronto", "Berlin", "Oslo", "Tokyo", "Dublin")
    for index in range(rows):
        record = {
            "record_id": f"R{seed}-{index}",
            "name": rng.choice(names),
            "age": rng.randrange(18, 80),
            "city": rng.choice(cities),
            "active": bool(rng.randrange(2)),
        }
        prompt = (
            "Convert the record to one valid compact JSON object.\n"
            f"record_id={record['record_id']}; name={record['name']}; age={record['age']}; city={record['city']}; "
            f"active={str(record['active']).lower()}\nJSON:"
        )
        writer.add(
            tokenizer,
            prompt,
            json.dumps(record, separators=(",", ":")),
            "json",
            max_answer_tokens=96,
        )


def prepare_split(output, tokenizer, split: str, max_length: int, counts: dict, seed: int):
    writer = CurriculumWriter(output, split, max_length, int(tokenizer.pad_token_id))
    squad_split = "train" if split == "train" else "validation"
    mbpp_split = "train" if split == "train" else "test"
    add_squad(writer, tokenizer, squad_split, counts["factual_qa"], seed)
    add_retrieval(writer, tokenizer, counts["retrieval"], seed + 1)
    add_mbpp(writer, tokenizer, mbpp_split, counts["python_code"], seed + 2)
    add_json(writer, tokenizer, counts["json"], seed + 3)
    return writer.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare factual/retrieval/code/JSON capability curriculum")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "prepared_data" / "capability_curriculum_5k"
    )
    parser.add_argument("--tokenizer", default=str(ROOT / "tokenizers" / "fdt_v3_bpe_24k"))
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(args.tokenizer)
    train_counts = {"factual_qa": 2500, "retrieval": 1200, "python_code": 500, "json": 800}
    validation_counts = {"factual_qa": 400, "retrieval": 200, "python_code": 100, "json": 100}
    report = {
        "format": "assistant_only_capability_curriculum_v1",
        "purpose": "Teach factual QA, held-out retrieval, Python, and JSON without chat-style SFT",
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "train": prepare_split(output, tokenizer, "train", args.max_length, train_counts, args.seed),
        "validation": prepare_split(
            output, tokenizer, "validation", args.max_length, validation_counts, args.seed + 1000
        ),
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
