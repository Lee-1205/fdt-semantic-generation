from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
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


SOURCE_REVISIONS = {
    "squad": "7b6d24c440a36b6815f21b70d25016731768db1f",
    "code_x_glue_python": "2678080e78a4b20ad79d1f0a514f04d815912564",
}


def require_c_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"V15 permits only C: paths: {resolved}")
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_hash(tokens: list[int] | torch.Tensor) -> str:
    tensor = torch.as_tensor(tokens, dtype=torch.int32).contiguous()
    return hashlib.blake2b(tensor.numpy().tobytes(), digest_size=16).hexdigest()


def active_rows(dataset_dir: Path, split: str = "train"):
    for path in sorted((dataset_dir / "shards" / split).glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        ids = payload["input_ids"]
        masks = payload.get("attention_mask", torch.ones_like(ids)).bool()
        for row, mask in zip(ids, masks):
            yield row[mask].to(dtype=torch.int32)


def reference_hashes(reference_root: Path, excluded: Path, eos_token_id: int) -> tuple[set[str], dict]:
    hashes: set[str] = set()
    report = {}
    for manifest_path in sorted(reference_root.glob("*/manifest.json")):
        directory = manifest_path.parent.resolve()
        if directory == excluded:
            continue
        count = 0
        documents = 0
        for row in active_rows(directory):
            if row.numel() < 2:
                continue
            hashes.add(token_hash(row))
            count += 1
            start = 0
            for end in (row == eos_token_id).nonzero(as_tuple=False).flatten().tolist():
                document = row[start : end + 1]
                if document.numel() >= 2:
                    hashes.add(token_hash(document))
                    documents += 1
                start = end + 1
        report[str(directory)] = {
            "train_rows_scanned": count,
            "eos_documents_scanned": documents,
            "manifest_sha256": file_sha256(manifest_path),
        }
    return hashes, report


class CompletionWriter:
    def __init__(self, output: Path, tokenizer, max_length: int, reference: set[str]):
        self.output = output
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reference = reference
        self.seen: set[str] = set()
        self.ids: list[list[int]] = []
        self.labels: list[list[int]] = []
        self.masks: list[list[int]] = []
        self.categories = Counter()
        self.rejections = Counter()

    def add(self, prompt: str, answer: str, category: str, max_answer_tokens: int) -> bool:
        answer_ids = list(self.tokenizer.encode(" " + answer.strip(), add_special_tokens=False))
        answer_ids = answer_ids[:max_answer_tokens] + [int(self.tokenizer.eos_token_id)]
        prompt_ids = list(self.tokenizer.encode(prompt, add_special_tokens=False))
        prompt_ids = prompt_ids[-max(self.max_length - len(answer_ids), 1) :]
        active = prompt_ids + answer_ids
        digest = token_hash(active)
        if digest in self.reference:
            self.rejections[f"{category}_prior_exact"] += 1
            return False
        if digest in self.seen:
            self.rejections[f"{category}_local_exact"] += 1
            return False
        self.seen.add(digest)
        padding = self.max_length - len(active)
        pad = int(self.tokenizer.pad_token_id)
        self.ids.append(active + [pad] * padding)
        self.labels.append([-100] * len(prompt_ids) + answer_ids + [-100] * padding)
        self.masks.append([1] * len(active) + [0] * padding)
        self.categories[category] += 1
        return True

    def save(self) -> dict:
        directory = self.output / "shards" / "train"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "shard_000000.pt"
        torch.save(
            {
                "input_ids": torch.tensor(self.ids, dtype=torch.int32),
                "labels": torch.tensor(self.labels, dtype=torch.int32),
                "attention_mask": torch.tensor(self.masks, dtype=torch.uint8),
            },
            path,
        )
        active_tokens = sum(sum(mask) for mask in self.masks)
        supervised_tokens = sum(sum(label != -100 for label in labels) for labels in self.labels)
        return {
            "rows": len(self.ids),
            "categories": dict(self.categories),
            "active_tokens": active_tokens,
            "supervised_tokens": supervised_tokens,
            "unique_active_rows": len(self.seen),
            "rejections": dict(self.rejections),
            "shard": str(path),
            "shard_sha256": file_sha256(path),
        }


def crop_context(context: str, answer: str, radius: int = 1200) -> str:
    position = context.lower().find(answer.lower())
    if position < 0:
        return context[: radius * 2]
    return context[max(0, position - radius) : position + len(answer) + radius]


def add_factual(writer: CompletionWriter, rows: int, seed: int) -> None:
    dataset = load_dataset(
        "rajpurkar/squad", split="train", streaming=True, revision=SOURCE_REVISIONS["squad"]
    ).shuffle(seed=seed, buffer_size=20_000)
    for row in dataset:
        answers = row.get("answers", {}).get("text", [])
        if not answers or not str(answers[0]).strip():
            continue
        answer = str(answers[0]).strip()
        context = crop_context(str(row["context"]), answer)
        prompt = f"Context: {context}\nQuestion: {row['question']}\nAnswer:"
        writer.add(prompt, answer, "factual_qa", 96)
        if writer.categories["factual_qa"] >= rows:
            return


def add_python(writer: CompletionWriter, rows: int, seed: int) -> None:
    dataset = load_dataset(
        "google/code_x_glue_ct_code_to_text",
        "python",
        split="train",
        streaming=True,
        revision=SOURCE_REVISIONS["code_x_glue_python"],
    ).shuffle(seed=seed, buffer_size=20_000)
    templates = (
        "Write valid Python code for this task:\n{task}\nPython:\n",
        "Complete this Python programming task. Return only Python code.\n{task}\nPython:\n",
    )
    for index, row in enumerate(dataset):
        task = str(row.get("docstring") or row.get("docstring_tokens") or "").strip()
        code = str(row.get("code") or "").strip()
        if not task or not code:
            continue
        prompt = templates[index % len(templates)].format(task=task)
        writer.add(prompt, code, "python_code", 256)
        if writer.categories["python_code"] >= rows:
            return


def random_code(rng: random.Random) -> str:
    return "".join(rng.choices(string.ascii_uppercase, k=2)) + "".join(
        rng.choices(string.digits, k=6)
    )


def add_retrieval(writer: CompletionWriter, rows: int, seed: int) -> None:
    rng = random.Random(seed)
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
    while writer.categories["retrieval"] < rows:
        code = random_code(rng)
        prompt, answer = templates[writer.categories["retrieval"] % len(templates)](code)
        writer.add(prompt, answer, "retrieval", 32)


def add_json(writer: CompletionWriter, rows: int, seed: int) -> None:
    rng = random.Random(seed)
    names = ("Ada", "Grace", "Linus", "Margaret", "Alan", "Edsger", "Barbara", "Donald")
    cities = ("London", "Paris", "Seoul", "Toronto", "Berlin", "Oslo", "Tokyo", "Dublin")
    index = 0
    while writer.categories["json"] < rows:
        record = {
            "record_id": f"V15-{seed}-{index}",
            "name": rng.choice(names),
            "age": rng.randrange(18, 80),
            "city": rng.choice(cities),
            "active": bool(rng.randrange(2)),
        }
        prompt = (
            "Convert the record to one valid compact JSON object.\n"
            f"record_id={record['record_id']}; name={record['name']}; age={record['age']}; "
            f"city={record['city']}; active={str(record['active']).lower()}\nJSON:"
        )
        writer.add(prompt, json.dumps(record, separators=(",", ":")), "json", 96)
        index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare exact-novel base-style completion data for V15.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=ROOT / "prepared_data")
    parser.add_argument(
        "--validation-source", type=Path, default=ROOT / "prepared_data" / "capability_curriculum_5k"
    )
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers" / "fdt_v3_bpe_24k")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--factual-rows", type=int, default=2000)
    parser.add_argument("--retrieval-rows", type=int, default=1500)
    parser.add_argument("--python-rows", type=int, default=750)
    parser.add_argument("--json-rows", type=int, default=750)
    args = parser.parse_args()

    output = require_c_path(args.output_dir)
    reference_root = require_c_path(args.reference_root)
    validation_source = require_c_path(args.validation_source)
    tokenizer_path = require_c_path(args.tokenizer)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(str(tokenizer_path))
    references, reference_report = reference_hashes(
        reference_root, output, int(tokenizer.eos_token_id)
    )
    writer = CompletionWriter(output, tokenizer, args.max_length, references)
    add_factual(writer, args.factual_rows, args.seed + 1)
    add_retrieval(writer, args.retrieval_rows, args.seed + 2)
    add_python(writer, args.python_rows, args.seed + 3)
    add_json(writer, args.json_rows, args.seed + 4)
    train = writer.save()

    validation_dir = output / "shards" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    validation_source_path = validation_source / "shards" / "validation" / "shard_000000.pt"
    validation_path = validation_dir / "shard_000000.pt"
    shutil.copy2(validation_source_path, validation_path)
    validation_manifest = json.loads((validation_source / "manifest.json").read_text(encoding="utf-8"))

    post_rows = list(active_rows(output))
    overlap = sum(token_hash(row) in references for row in post_rows)
    if overlap:
        raise RuntimeError(f"Post-build exact overlap is nonzero: {overlap}")
    report = {
        "format": "base_style_completion_span_v1",
        "purpose": "Bounded V15 exposure-objective diagnostic without conversational formatting",
        "assistant_masking": False,
        "completion_span_supervision": True,
        "prompt_lm_retention_applied_by_trainer": True,
        "tokenizer": str(tokenizer_path),
        "max_length": args.max_length,
        "seed": args.seed,
        "source_revisions": SOURCE_REVISIONS,
        "train": train,
        "validation": {
            **validation_manifest["validation"],
            "source": str(validation_source),
            "source_manifest_sha256": file_sha256(validation_source / "manifest.json"),
            "shard_sha256": file_sha256(validation_path),
            "unchanged_from_fixed_strict_validation": True,
        },
        "reference_root": str(reference_root),
        "reference_datasets_scanned": len(reference_report),
        "reference_audit": reference_report,
        "post_build_exact_overlap": overlap,
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "reference_audit": {"datasets": len(reference_report)}}), flush=True)


if __name__ == "__main__":
    main()
