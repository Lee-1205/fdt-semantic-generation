from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fdt_rlm.data.filters import normalize_web_text
from fdt_rlm.tokenization import load_tokenizer

from prepare_balanced_novel_pilot import SOURCE_REVISIONS as NATURAL_SOURCE_REVISIONS
from prepare_balanced_novel_pilot import stream_hf_texts
from prepare_capability_completion_v15 import (
    CompletionWriter,
    SOURCE_REVISIONS,
    active_rows,
    add_factual,
    add_python,
    add_retrieval,
    file_sha256,
    reference_hashes,
    require_c_path,
    token_hash,
)


CATEGORY_NAMES = ("factual_qa", "retrieval", "python_code", "json", "natural_lm")
CATEGORY_TO_ID = {name: index for index, name in enumerate(CATEGORY_NAMES)}
PROFILES = {
    "capability_heavy": {
        "counts": {"factual_qa": 2000, "retrieval": 1500, "python_code": 750, "json": 750},
        "prompt_lm_weight": 0.10,
    },
    "balanced": {
        "counts": {
            "factual_qa": 1600,
            "retrieval": 1200,
            "python_code": 600,
            "json": 600,
            "natural_lm": 1000,
        },
        "prompt_lm_weight": 0.15,
    },
    "retention_heavy": {
        "counts": {
            "factual_qa": 1300,
            "retrieval": 975,
            "python_code": 487,
            "json": 488,
            "natural_lm": 1750,
        },
        "prompt_lm_weight": 0.20,
    },
}


class V16Writer(CompletionWriter):
    def __init__(self, output: Path, tokenizer, max_length: int, reference: set[str]):
        super().__init__(output, tokenizer, max_length, reference)
        self.row_categories: list[int] = []

    def add(self, prompt: str, answer: str, category: str, max_answer_tokens: int) -> bool:
        accepted = super().add(prompt, answer, category, max_answer_tokens)
        if accepted:
            self.row_categories.append(CATEGORY_TO_ID[category])
        return accepted

    def add_ids(self, prompt_ids: list[int], answer_ids: list[int], category: str) -> bool:
        answer = list(answer_ids[: self.max_length - 2]) + [int(self.tokenizer.eos_token_id)]
        prompt = list(prompt_ids[-max(self.max_length - len(answer), 1) :])
        active = prompt + answer
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
        self.labels.append([-100] * len(prompt) + answer + [-100] * padding)
        self.masks.append([1] * len(active) + [0] * padding)
        self.categories[category] += 1
        self.row_categories.append(CATEGORY_TO_ID[category])
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
                "category_ids": torch.tensor(self.row_categories, dtype=torch.int16),
            },
            path,
        )
        active_tokens = sum(sum(mask) for mask in self.masks)
        supervised_tokens = sum(sum(label != -100 for label in labels) for labels in self.labels)
        return {
            "rows": len(self.ids),
            "categories": dict(self.categories),
            "category_names": list(CATEGORY_NAMES),
            "active_tokens": active_tokens,
            "supervised_tokens": supervised_tokens,
            "unique_active_rows": len(self.seen),
            "rejections": dict(self.rejections),
            "shard": str(path),
            "shard_sha256": file_sha256(path),
        }


def add_json(writer: V16Writer, rows: int, seed: int) -> None:
    rng = random.Random(seed)
    names = ("Ada", "Grace", "Linus", "Margaret", "Alan", "Edsger", "Barbara", "Donald")
    cities = ("London", "Paris", "Seoul", "Toronto", "Berlin", "Oslo", "Tokyo", "Dublin")
    index = 0
    while writer.categories["json"] < rows:
        record = {
            "record_id": f"V16-{seed}-{index}",
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


def add_natural_retention(writer: V16Writer, rows: int, seed: int) -> dict:
    rng = random.Random(seed)
    streams = {
        "fineweb_edu_dedup": stream_hf_texts("fineweb_edu_dedup", seed + 1),
        "tinystories": stream_hf_texts("tinystories", seed + 2),
    }
    accepted = Counter()
    sources = tuple(streams)
    source_index = 0
    while writer.categories["natural_lm"] < rows:
        source = sources[source_index % len(sources)]
        source_index += 1
        text, _ = next(streams[source])
        text = normalize_web_text(text)
        if not text or "\x00" in text:
            writer.rejections[f"natural_lm_{source}_empty"] += 1
            continue
        ids = [int(value) for value in writer.tokenizer.encode(text, add_special_tokens=False)]
        if len(ids) < 96:
            writer.rejections[f"natural_lm_{source}_short"] += 1
            continue
        span = min(len(ids), writer.max_length - 1)
        offset = rng.randrange(0, len(ids) - span + 1) if len(ids) > span else 0
        active = ids[offset : offset + span]
        split = max(48, min(len(active) - 32, int(len(active) * 0.75)))
        if writer.add_ids(active[:split], active[split:], "natural_lm"):
            accepted[source] += 1
    return dict(accepted)


def profile_indices(category_ids: torch.Tensor, profile: dict, seed: int) -> torch.Tensor:
    chosen = []
    generator = torch.Generator().manual_seed(seed)
    for category, count in profile["counts"].items():
        category_index = CATEGORY_TO_ID[category]
        available = (category_ids == category_index).nonzero(as_tuple=False).flatten()
        if len(available) < count:
            raise ValueError(f"Insufficient rows for {category}: {len(available)} < {count}")
        order = torch.randperm(len(available), generator=generator)
        chosen.append(available.index_select(0, order[:count]))
    merged = torch.cat(chosen)
    return merged.index_select(0, torch.randperm(len(merged), generator=generator))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare exact-novel V16 retention-mix master data.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=ROOT / "prepared_data")
    parser.add_argument(
        "--validation-source", type=Path, default=ROOT / "prepared_data" / "capability_curriculum_5k"
    )
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers" / "fdt_v3_bpe_24k")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260816)
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
    writer = V16Writer(output, tokenizer, args.max_length, references)
    add_factual(writer, 2000, args.seed + 1)
    add_retrieval(writer, 1500, args.seed + 2)
    add_python(writer, 750, args.seed + 3)
    add_json(writer, 750, args.seed + 4)
    natural_sources = add_natural_retention(writer, 1750, args.seed + 5)
    train = writer.save()

    category_ids = torch.tensor(writer.row_categories, dtype=torch.int16)
    for name, profile in PROFILES.items():
        indices = profile_indices(category_ids, profile, args.seed + 100 + len(name))
        profile["rows"] = len(indices)
        profile["index_sha256"] = token_hash(indices.to(dtype=torch.int32))

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
        "format": "base_style_completion_retention_master_v1",
        "purpose": "V16 predeclared retention-mix smoke comparison and bounded completion pilot",
        "conversational_sft": False,
        "completion_span_supervision": True,
        "supervised_eos": True,
        "tokenizer": str(tokenizer_path),
        "max_length": args.max_length,
        "seed": args.seed,
        "source_revisions": {**SOURCE_REVISIONS, **{k: NATURAL_SOURCE_REVISIONS[k] for k in natural_sources}},
        "train": train,
        "natural_sources": natural_sources,
        "profiles": PROFILES,
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
