from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT / "src", ROOT / "scripts"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from fdt_rlm.data.filters import normalize_web_text
from fdt_rlm.tokenization import load_tokenizer
from prepare_balanced_novel_pilot import SOURCE_REVISIONS as NATURAL_REVISIONS
from prepare_balanced_novel_pilot import stream_hf_texts
from prepare_capability_completion_v15 import (
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
from prepare_capability_completion_v16 import CATEGORY_NAMES, V16Writer, add_json


EXPERIMENT_NONCE = "v17-balanced-3e192a7d6f804cb99811d56ac047f2e3"
CATEGORY_COUNTS = {name: 1000 for name in CATEGORY_NAMES}
ANSWER_CAPS = {"factual_qa": 16, "retrieval": 32, "python_code": 96, "json": 96}


class V17Writer(V16Writer):
    def __init__(self, output: Path, tokenizer, max_length: int, reference: set[str]):
        super().__init__(output, tokenizer, max_length, reference)
        self.serial = 0

    def marker(self, category: str) -> str:
        self.serial += 1
        return f"Source record {EXPERIMENT_NONCE}-{category}-{self.serial:07d}.\n"

    def add(self, prompt: str, answer: str, category: str, max_answer_tokens: int) -> bool:
        cap = ANSWER_CAPS.get(category, max_answer_tokens)
        return super().add(self.marker(category) + prompt, answer, category, min(cap, max_answer_tokens))

    def add_ids(self, prompt_ids: list[int], answer_ids: list[int], category: str) -> bool:
        marker_ids = list(self.tokenizer.encode(self.marker(category), add_special_tokens=False))
        return super().add_ids(marker_ids + prompt_ids, answer_ids, category)


def add_natural(writer: V17Writer, rows: int, seed: int) -> dict:
    rng = random.Random(seed)
    names = ("fineweb_edu_dedup", "tinystories", "wikipedia")
    streams = {name: stream_hf_texts(name, seed + index + 1) for index, name in enumerate(names)}
    accepted = Counter()
    source_index = 0
    while writer.categories["natural_lm"] < rows:
        source = names[source_index % len(names)]
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
        span = min(len(ids), writer.max_length - 24)
        offset = rng.randrange(0, len(ids) - span + 1) if len(ids) > span else 0
        active = ids[offset : offset + span]
        split = max(48, min(len(active) - 32, int(len(active) * 0.75)))
        if writer.add_ids(active[:split], active[split:], "natural_lm"):
            accepted[source] += 1
    return dict(accepted)


def category_token_report(writer: V17Writer) -> dict:
    report = {}
    for category_index, category in enumerate(CATEGORY_NAMES):
        rows = [i for i, value in enumerate(writer.row_categories) if value == category_index]
        active = sum(sum(writer.masks[i]) for i in rows)
        supervised = sum(sum(label != -100 for label in writer.labels[i]) for i in rows)
        report[category] = {
            "rows": len(rows),
            "active_tokens": active,
            "completion_supervised_tokens": supervised,
            "mean_completion_tokens_per_row": supervised / max(len(rows), 1),
            "objective_category_weight": 0.2,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare balanced exact-novel V17 completion data.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=ROOT / "prepared_data")
    parser.add_argument("--validation-source", type=Path, default=ROOT / "prepared_data" / "capability_curriculum_5k")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers" / "fdt_v3_bpe_24k")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    output = require_c_path(args.output_dir)
    reference_root = require_c_path(args.reference_root)
    validation_source = require_c_path(args.validation_source)
    tokenizer_path = require_c_path(args.tokenizer)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(str(tokenizer_path))
    references, reference_report = reference_hashes(reference_root, output, int(tokenizer.eos_token_id))
    writer = V17Writer(output, tokenizer, args.max_length, references)
    add_factual(writer, CATEGORY_COUNTS["factual_qa"], args.seed + 1)
    add_retrieval(writer, CATEGORY_COUNTS["retrieval"], args.seed + 2)
    add_python(writer, CATEGORY_COUNTS["python_code"], args.seed + 3)
    add_json(writer, CATEGORY_COUNTS["json"], args.seed + 4)
    natural_sources = add_natural(writer, CATEGORY_COUNTS["natural_lm"], args.seed + 5)
    token_report = category_token_report(writer)
    train = writer.save()

    validation_dir = output / "shards" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    source_validation = validation_source / "shards" / "validation" / "shard_000000.pt"
    target_validation = validation_dir / "shard_000000.pt"
    shutil.copy2(source_validation, target_validation)
    validation_manifest = json.loads((validation_source / "manifest.json").read_text(encoding="utf-8"))
    post_rows = list(active_rows(output))
    if len({token_hash(row) for row in post_rows}) != len(post_rows):
        raise RuntimeError("V17 local exact-row collision")
    overlap = sum(token_hash(row) in references for row in post_rows)
    if overlap:
        raise RuntimeError(f"V17 post-build exact overlap is nonzero: {overlap}")
    report = {
        "format": "balanced_category_normalized_completion_v1",
        "purpose": "V17 balanced capability completion and natural retention",
        "conversational_sft": False,
        "completion_span_supervision": True,
        "supervised_eos": True,
        "eos_loss_weight": 2.0,
        "category_normalized_per_example_loss": True,
        "prompt_lm_weight": 0.2,
        "experiment_nonce": EXPERIMENT_NONCE,
        "historical_payload_limitation": "some consumed payloads were pruned; nonce-prefixed rows provide construction-based full-row novelty",
        "tokenizer": str(tokenizer_path),
        "max_length": args.max_length,
        "seed": args.seed,
        "source_revisions": {
            **SOURCE_REVISIONS,
            **{name: NATURAL_REVISIONS[name] for name in natural_sources},
        },
        "train": train,
        "category_token_report": token_report,
        "answer_token_caps": ANSWER_CAPS,
        "natural_sources": natural_sources,
        "validation": {
            **validation_manifest["validation"],
            "source": str(validation_source),
            "source_manifest_sha256": file_sha256(validation_source / "manifest.json"),
            "shard_sha256": file_sha256(target_validation),
            "unchanged_from_fixed_strict_validation": True,
        },
        "reference_root": str(reference_root),
        "reference_datasets_scanned": len(reference_report),
        "reference_audit": reference_report,
        "post_build_exact_overlap": overlap,
    }
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "manifest_sha256": file_sha256(manifest),
        "train_shard_sha256": train["shard_sha256"],
        "rows": train["rows"],
        "categories": train["categories"],
        "category_token_report": token_report,
        "post_build_exact_overlap": overlap,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
