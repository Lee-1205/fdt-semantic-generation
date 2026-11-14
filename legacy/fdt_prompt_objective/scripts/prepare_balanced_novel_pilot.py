from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.data.dedup import normalized_text_hash
from fdt_rlm.data.filters import normalize_code_text, normalize_web_text
from fdt_rlm.tokenization import load_tokenizer

try:
    from scripts.prepare_exact_novel_continuation import (
        CompactShardWriter,
        collect_document_hashes,
        copy_compact_split,
        document_hash,
        file_sha256,
        iter_documents,
        require_c_path,
    )
except ModuleNotFoundError:
    from prepare_exact_novel_continuation import (
        CompactShardWriter,
        collect_document_hashes,
        copy_compact_split,
        document_hash,
        file_sha256,
        iter_documents,
        require_c_path,
    )


SOURCE_REVISIONS = {
    "fineweb_edu": {
        "dataset": "HuggingFaceFW/fineweb-edu",
        "config": "CC-MAIN-2024-10",
        "revision": "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9",
        "field": "text",
        "license": "ODC-By 1.0",
    },
    "wikipedia": {
        "dataset": "wikimedia/wikipedia",
        "config": "20231101.en",
        "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "field": "text",
        "license": "CC BY-SA 3.0 and GFDL",
    },
    "fineweb_edu_dedup": {
        "dataset": "HuggingFaceTB/smollm-corpus",
        "config": "fineweb-edu-dedup",
        "revision": "3ba9d605774198c5868892d7a8deda78031a781f",
        "field": "text",
        "license": "ODC-By 1.0",
    },
    "cosmopedia": {
        "dataset": "HuggingFaceTB/smollm-corpus",
        "config": "cosmopedia-v2",
        "revision": "3ba9d605774198c5868892d7a8deda78031a781f",
        "field": "text",
        "license": "Apache-2.0",
    },
    "tinystories": {
        "dataset": "roneneldan/TinyStories",
        "config": None,
        "revision": "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64",
        "field": "text",
        "license": "CDLA-Sharing-1.0",
    },
    "code_python": {
        "dataset": "google/code_x_glue_ct_code_to_text",
        "config": "python",
        "revision": "2678080e78a4b20ad79d1f0a514f04d815912564",
        "field": "code",
        "license": "CodeSearchNet source repositories with redistribution-permitting licenses",
    },
    "factual_cloze": {
        "dataset": "rajpurkar/squad",
        "config": None,
        "revision": "7b6d24c440a36b6815f21b70d25016731768db1f",
        "field": "context",
        "license": "CC BY-SA 4.0",
    },
    "code_mbpp": {
        "dataset": "google-research-datasets/mbpp",
        "config": "sanitized",
        "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
        "field": "code",
        "license": "CC BY 4.0",
    },
}

DEFAULT_MIX = {
    "fineweb_edu": 0.30,
    "wikipedia": 0.20,
    "code_python": 0.35,
    "json_structured": 0.075,
    "retrieval_synthetic": 0.075,
}

MIX_PROFILES = {
    "balanced_v1": DEFAULT_MIX,
    "retention_capability_v2": {
        "fineweb_edu": 0.20,
        "wikipedia": 0.10,
        "code_python": 0.29,
        "code_mbpp": 0.01,
        "factual_cloze": 0.10,
        "json_compact": 0.15,
        "retrieval_long": 0.15,
    },
    "retention_capability_v2_scale": {
        "fineweb_edu": 0.25,
        "wikipedia": 0.15,
        "code_python": 0.30,
        "factual_cloze": 0.10,
        "json_compact": 0.10,
        "retrieval_long": 0.10,
    },
    "retention_diversity_v3_pilot": {
        "fineweb_edu": 0.35,
        "wikipedia": 0.20,
        "code_python": 0.35,
        "factual_cloze": 0.04,
        "json_compact": 0.03,
        "retrieval_long": 0.03,
    },
    "retention_alignment_v4_pilot": {
        "fineweb_edu": 0.25,
        "wikipedia": 0.15,
        "code_python": 0.25,
        "factual_cloze": 0.10,
        "json_compact": 0.05,
        "retrieval_prompted": 0.10,
        "retrieval_long": 0.10,
    },
    "code_growth_v5_pilot": {
        "fineweb_edu": 0.25,
        "wikipedia": 0.15,
        "code_python_prompted": 0.15,
        "code_python": 0.20,
        "factual_cloze": 0.08,
        "json_compact": 0.05,
        "retrieval_prompted": 0.06,
        "retrieval_long": 0.06,
    },
    "general_growth_v6_scale": {
        "fineweb_edu": 0.30,
        "wikipedia": 0.20,
        "code_python": 0.30,
        "factual_cloze": 0.08,
        "json_compact": 0.04,
        "retrieval_prompted": 0.04,
        "retrieval_long": 0.04,
    },
    "retrieval_repetition_repair_v7": {
        "fineweb_edu_dedup": 0.15,
        "cosmopedia": 0.20,
        "tinystories": 0.15,
        "wikipedia": 0.10,
        "code_python": 0.10,
        "factual_cloze": 0.05,
        "json_compact": 0.05,
        "retrieval_prompted": 0.10,
        "retrieval_long": 0.10,
    },
    "factual_retrieval_balance_v9": {
        "fineweb_edu": 0.20,
        "wikipedia": 0.15,
        "code_python": 0.15,
        "factual_cloze": 0.30,
        "json_compact": 0.05,
        "retrieval_prompted": 0.075,
        "retrieval_long": 0.075,
    },
    "natural_retrieval_balance_v10": {
        "fineweb_edu_dedup": 0.15,
        "cosmopedia": 0.20,
        "tinystories": 0.10,
        "wikipedia": 0.25,
        "code_python": 0.10,
        "json_compact": 0.05,
        "retrieval_prompted": 0.075,
        "retrieval_long": 0.075,
    },
    "short_fact_repair_v12": {
        "fineweb_edu_dedup": 0.15,
        "cosmopedia": 0.10,
        "tinystories": 0.05,
        "wikipedia": 0.15,
        "wikipedia_short_fact": 0.25,
        "code_python": 0.15,
        "json_compact": 0.05,
        "retrieval_prompted": 0.05,
        "retrieval_long": 0.05,
    },
    "long_horizon_general_v13": {
        "fineweb_edu_dedup": 0.15,
        "cosmopedia": 0.15,
        "tinystories": 0.10,
        "wikipedia": 0.20,
        "code_python": 0.20,
        "json_compact": 0.05,
        "retrieval_prompted": 0.075,
        "retrieval_long": 0.075,
    },
    "capability_retention_v14": {
        "fineweb_edu_dedup": 0.125,
        "cosmopedia": 0.125,
        "tinystories": 0.05,
        "wikipedia": 0.15,
        "code_python": 0.158,
        "code_mbpp": 0.002,
        "factual_cloze": 0.05,
        "json_compact": 0.04,
        "retrieval_prompted": 0.10,
        "retrieval_long": 0.20,
    },
}

TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+|[^\s]", re.UNICODE)
NEAR_DEDUP_EXEMPT_BUCKETS = {
    "json_structured",
    "retrieval_synthetic",
    "json_compact",
    "retrieval_prompted",
    "retrieval_long",
}


def repeated_ngram_fraction(text: str, n: int = 3) -> tuple[float, int]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    count = len(tokens) - n + 1
    if count <= 0:
        return 0.0, 0
    unique = {tuple(tokens[index : index + n]) for index in range(count)}
    return 1.0 - (len(unique) / count), count


def simhash64(text: str) -> int:
    tokens = TOKEN_PATTERN.findall(text.lower())
    if not tokens:
        return 0
    shingles = tokens if len(tokens) < 3 else ["\x1f".join(tokens[i : i + 3]) for i in range(len(tokens) - 2)]
    vector = [0] * 64
    for shingle in shingles[:8192]:
        value = int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "little")
        for bit in range(64):
            vector[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            result |= 1 << bit
    return result


class NearDeduper:
    def __init__(self, max_hamming: int = 3, bands: int = 4):
        self.max_hamming = max_hamming
        self.bands = bands
        self.band_bits = 64 // bands
        self.band_mask = (1 << self.band_bits) - 1
        self.values: list[int] = []
        self.index: dict[tuple[int, int], list[int]] = defaultdict(list)

    def add_if_novel(self, text: str) -> bool:
        value = simhash64(text)
        candidates: set[int] = set()
        for band in range(self.bands):
            key = (band, (value >> (band * self.band_bits)) & self.band_mask)
            candidates.update(self.index.get(key, ()))
        if any((value ^ self.values[index]).bit_count() <= self.max_hamming for index in candidates):
            return False
        index = len(self.values)
        self.values.append(value)
        for band in range(self.bands):
            key = (band, (value >> (band * self.band_bits)) & self.band_mask)
            self.index[key].append(index)
        return True


def stream_hf_texts(bucket: str, seed: int) -> Iterator[tuple[str, dict]]:
    from datasets import load_dataset

    spec = SOURCE_REVISIONS[bucket]
    positional = [spec["dataset"]]
    if spec["config"] is not None:
        positional.append(spec["config"])
    dataset = load_dataset(
        *positional,
        split="train",
        streaming=True,
        revision=spec["revision"],
    ).shuffle(seed=seed, buffer_size=10_000)
    for row in dataset:
        text = row.get(spec["field"], "")
        if not isinstance(text, str) or not text.strip():
            continue
        if bucket == "wikipedia":
            title = str(row.get("title", "")).strip()
            if title:
                text = f"{title}\n\n{text}"
        metadata = {
            "id": str(row.get("id") or row.get("sha") or row.get("url") or ""),
            "url": str(row.get("url") or ""),
            "repo": str(row.get("repo") or ""),
            "path": str(row.get("path") or ""),
        }
        yield text, metadata


def build_wikipedia_short_fact(title: str, text: str) -> str | None:
    title = " ".join(title.split())
    answer = re.sub(r"\s+\([^()]+\)$", "", title).strip()
    if not answer or len(TOKEN_PATTERN.findall(answer)) > 4:
        return None
    if title.lower().startswith(("list of ", "category:", "template:", "portal:")):
        return None

    normalized = " ".join(text.split())
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    excerpt = " ".join(sentences[:3]).strip()
    if len(excerpt) < 120:
        return None
    if len(excerpt) > 900:
        excerpt = excerpt[:900].rsplit(" ", 1)[0]

    masked = re.sub(re.escape(title), "This subject", excerpt, flags=re.IGNORECASE)
    if answer != title:
        masked = re.sub(re.escape(answer), "This subject", masked, flags=re.IGNORECASE)
    if masked == excerpt or answer.lower() in masked.lower():
        return None
    return (
        f"Definition: {masked}\n"
        "Question: Which subject is described?\n"
        f"Answer: {answer}"
    )


def wikipedia_short_fact_documents(seed: int) -> Iterator[tuple[str, dict]]:
    from datasets import load_dataset

    spec = SOURCE_REVISIONS["wikipedia"]
    dataset = load_dataset(
        spec["dataset"],
        spec["config"],
        split="train",
        streaming=True,
        revision=spec["revision"],
    ).shuffle(seed=seed, buffer_size=10_000)
    for row in dataset:
        title = str(row.get("title", "")).strip()
        text = str(row.get("text", "")).strip()
        document = build_wikipedia_short_fact(title, text)
        if document is not None:
            yield document, {
                "id": str(row.get("id", "")),
                "source_title": title,
                "format": "masked_wikipedia_subject_v1",
            }


def stream_prompted_python_texts(seed: int) -> Iterator[tuple[str, dict]]:
    from datasets import load_dataset

    spec = SOURCE_REVISIONS["code_python"]
    dataset = load_dataset(
        spec["dataset"],
        spec["config"],
        split="train",
        streaming=True,
        revision=spec["revision"],
    ).shuffle(seed=seed, buffer_size=10_000)
    templates = (
        "Python task:\n{docstring}\n\nImplementation:\n{code}",
        "Implement a Python function for this description:\n{docstring}\n\nPython:\n{code}",
        "Complete a valid Python implementation.\nRequirement: {docstring}\n\nCode:\n{code}",
    )
    serial = 0
    for row in dataset:
        docstring = str(row.get("docstring", "")).strip()
        code = str(row.get("code", "")).strip()
        if not docstring or not code:
            continue
        serial += 1
        yield templates[(serial - 1) % len(templates)].format(
            docstring=docstring,
            code=code,
        ), {
            "id": str(row.get("id", "")),
            "raw_code": code,
            "template": (serial - 1) % len(templates),
        }


def factual_cloze_documents(seed: int) -> Iterator[tuple[str, dict]]:
    from datasets import load_dataset

    spec = SOURCE_REVISIONS["factual_cloze"]
    dataset = load_dataset(
        spec["dataset"],
        split="train",
        streaming=True,
        revision=spec["revision"],
    ).shuffle(seed=seed, buffer_size=10_000)
    for row in dataset:
        answers = row.get("answers", {}).get("text", [])
        if not answers or not str(answers[0]).strip():
            continue
        answer = str(answers[0]).strip()
        context = str(row.get("context", "")).strip()
        question = str(row.get("question", "")).strip()
        if not context or not question:
            continue
        position = context.lower().find(answer.lower())
        if position >= 0:
            context = context[max(position - 2400, 0) : position + len(answer) + 2400]
        document = f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"
        yield document, {"id": str(row.get("id", ""))}


def mbpp_documents(seed: int) -> Iterator[tuple[str, dict]]:
    from datasets import load_dataset

    spec = SOURCE_REVISIONS["code_mbpp"]
    dataset = list(
        load_dataset(
            spec["dataset"],
            spec["config"],
            split="train",
            revision=spec["revision"],
        )
    )
    rng = random.Random(seed)
    rng.shuffle(dataset)
    for row in dataset:
        prompt = str(row.get("prompt", "")).strip()
        code = str(row.get("code", "")).strip()
        tests = row.get("test_list") or []
        if not prompt or not code:
            continue
        test_text = "\n".join(str(value) for value in tests[:3])
        document = (
            f"Python task:\n{prompt}\n\nReference solution:\n{code}\n\n"
            f"Reference tests:\n{test_text}"
        )
        yield document, {"id": str(row.get("task_id", ""))}


def json_documents(seed: int) -> Iterator[tuple[str, dict]]:
    rng = random.Random(seed)
    categories = ["service", "dataset", "experiment", "module", "release", "sensor"]
    regions = ["north", "south", "east", "west", "central", "remote"]
    while True:
        serial = rng.getrandbits(48)
        records = []
        for index in range(rng.randint(5, 14)):
            record_id = f"r-{serial:012x}-{index:02d}"
            records.append(
                {
                    "id": record_id,
                    "category": rng.choice(categories),
                    "enabled": bool(rng.getrandbits(1)),
                    "region": rng.choice(regions),
                    "thresholds": {
                        "low": round(rng.uniform(0.01, 0.40), 4),
                        "high": round(rng.uniform(0.60, 0.99), 4),
                    },
                    "tags": sorted(rng.sample(categories + regions, rng.randint(2, 5))),
                    "samples": [rng.randint(-5000, 5000) for _ in range(rng.randint(3, 9))],
                }
            )
        payload = {
            "schema_version": f"{rng.randint(1, 7)}.{rng.randint(0, 9)}",
            "batch_id": f"batch-{serial:012x}",
            "generated": False,
            "metadata": {
                "owner": f"team-{rng.randint(100, 999)}",
                "priority": rng.choice(["low", "normal", "high"]),
                "labels": {f"k{n}": f"v{rng.getrandbits(20):05x}" for n in range(rng.randint(2, 6))},
            },
            "records": records,
        }
        yield json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True), {"id": payload["batch_id"]}


def retrieval_documents(seed: int) -> Iterator[tuple[str, dict]]:
    rng = random.Random(seed)
    filler = (
        "analysis archive boundary catalog context detail evidence field history index "
        "ledger method note observation paragraph record reference section summary trace value"
    ).split()
    distances = [96, 448, 1152]
    serial = 0
    while True:
        serial += 1
        target_key = f"K{rng.getrandbits(44):011X}"
        target_value = f"V{rng.getrandbits(52):013X}"
        entries = [(target_key, target_value)]
        for _ in range(rng.randint(12, 28)):
            entries.append((f"K{rng.getrandbits(44):011X}", f"V{rng.getrandbits(52):013X}"))
        rng.shuffle(entries)
        lines = [f"Reference ledger {serial:08d}", "Recorded key-value entries:"]
        lines.extend(f"{key} = {value}" for key, value in entries)
        distance = distances[(serial - 1) % len(distances)]
        paragraph = " ".join(rng.choice(filler) for _ in range(distance))
        lines.extend(
            [
                "",
                paragraph,
                "",
                f"The ledger value associated with {target_key} is {target_value}.",
            ]
        )
        yield "\n".join(lines), {"id": f"ledger-{serial:08d}", "distance_class": distance}


def compact_json_documents(seed: int) -> Iterator[tuple[str, dict]]:
    rng = random.Random(seed)
    names = ("Ada", "Grace", "Linus", "Margaret", "Alan", "Edsger", "Barbara", "Donald")
    cities = ("London", "Paris", "Seoul", "Toronto", "Berlin", "Oslo", "Tokyo", "Dublin")
    serial = 0
    while True:
        serial += 1
        record = {
            "record_id": f"R{seed}-{serial}",
            "name": rng.choice(names),
            "age": rng.randrange(18, 80),
            "city": rng.choice(cities),
            "active": bool(rng.randrange(2)),
        }
        compact = json.dumps(record, separators=(",", ":"), ensure_ascii=True)
        prompt = (
            "Convert the record to one valid compact JSON object.\n"
            f"record_id={record['record_id']}; name={record['name']}; age={record['age']}; "
            f"city={record['city']}; active={str(record['active']).lower()}\nJSON:"
        )
        yield f"{prompt} {compact}", {"id": record["record_id"], "schema": "compact_record_v1"}


def build_filler_to_token_distance(tokenizer, words: list[str], rng: random.Random, distance: int) -> list[str]:
    filler: list[str] = []
    token_count = 0
    while token_count < distance:
        batch_size = min(128, max(16, distance - token_count))
        batch = rng.choices(words, k=batch_size)
        candidate = filler + batch
        candidate_count = len(tokenizer.encode(" ".join(candidate), add_special_tokens=False))
        if candidate_count < distance:
            filler = candidate
            token_count = candidate_count
            continue

        low, high = 1, len(batch)
        while low < high:
            middle = (low + high) // 2
            probe = filler + batch[:middle]
            if len(tokenizer.encode(" ".join(probe), add_special_tokens=False)) >= distance:
                high = middle
            else:
                low = middle + 1
        filler.extend(batch[:low])
        break
    return filler


def long_retrieval_documents(tokenizer, seed: int) -> Iterator[tuple[str, dict]]:
    rng = random.Random(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    words = (
        "method result system sample process value record detail signal context analysis model "
        "data memory output input stable precise ordinary measured reference"
    ).split()
    distances = (64, 512, 1400, 1400)
    templates = (
        (
            "Archive entry: the verification code is {code}. ",
            ". For this archive entry, the verification code is {code}.",
        ),
        (
            "Registry note: keep identifier {code} for the current item. ",
            ". The current item identifier is {code}.",
        ),
        (
            "Evidence record: the retained reference is {code}. ",
            ". The retained reference from the evidence record is {code}.",
        ),
    )
    serial = 0
    while True:
        serial += 1
        code = "".join(rng.choice(alphabet) for _ in range(8))
        distance = distances[(serial - 1) % len(distances)]
        prefix, suffix = templates[(serial - 1) % len(templates)]
        filler = build_filler_to_token_distance(tokenizer, words, rng, distance)
        document = prefix.format(code=code) + " ".join(filler) + suffix.format(code=code)
        yield document, {
            "id": f"long-retrieval-{serial:08d}",
            "code": code,
            "distance_class": distance,
            "template": (serial - 1) % len(templates),
        }


def prompted_retrieval_documents(seed: int) -> Iterator[tuple[str, dict]]:
    rng = random.Random(seed)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    templates = (
        (
            "Maintenance sheet: unit {unit} carries active value {code}. "
            "The older values {old_a} and {old_b} are retired.\n"
            "Requested active value for unit {unit}: {code}"
        ),
        (
            "Inventory record {unit} states that its assigned identifier is {code}. "
            "A neighboring record uses {old_a}, while a closed record used {old_b}.\n"
            "Identifier assigned to record {unit}: {code}"
        ),
        (
            "Operator memo for {unit}: disregard candidate tags {old_a} and {old_b}; "
            "retain reference {code} for the current request.\n"
            "Retained reference: {code}"
        ),
        (
            "Evidence file {unit} contains several labels. {old_a} is historical and {old_b} "
            "belongs to another file. The verified label for this file is {code}.\n"
            "Verified label for evidence file {unit}: {code}"
        ),
    )
    serial = 0
    while True:
        serial += 1
        values = ["".join(rng.choice(alphabet) for _ in range(8)) for _ in range(3)]
        code, old_a, old_b = values
        unit = f"U{rng.randrange(10_000_000):07d}"
        document = templates[(serial - 1) % len(templates)].format(
            unit=unit,
            code=code,
            old_a=old_a,
            old_b=old_b,
        )
        yield document, {
            "id": f"prompted-retrieval-{serial:08d}",
            "code": code,
            "template": (serial - 1) % len(templates),
        }


def split_long_document(ids: list[int], max_document_tokens: int) -> Iterator[list[int]]:
    for start in range(0, len(ids), max_document_tokens):
        part = ids[start : start + max_document_tokens]
        if part:
            yield part


def collect_global_reference_hashes(reference_dirs: Iterable[Path], eos_token_id: int) -> tuple[set[str], dict]:
    hashes: set[str] = set()
    report = {}
    for reference in reference_dirs:
        reference = require_c_path(reference)
        by_bucket, counts = collect_document_hashes(reference, eos_token_id)
        before = len(hashes)
        for values in by_bucket.values():
            hashes.update(values)
        report[str(reference)] = {
            "documents_scanned": int(sum(counts.values())),
            "unique_hashes_added": len(hashes) - before,
            "manifest_sha256": file_sha256(reference / "manifest.json"),
        }
    return hashes, report


def prepare(args: argparse.Namespace) -> dict:
    output = require_c_path(args.output_dir)
    tokenizer_path = require_c_path(args.tokenizer)
    validation_source = require_c_path(args.validation_source)
    references = [require_c_path(path) for path in args.reference_dataset]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    tokenizer = load_tokenizer(str(tokenizer_path))
    eos_token_id = int(tokenizer.eos_token_id)
    reference_hashes, reference_report = collect_global_reference_hashes(references, eos_token_id)

    writer = CompactShardWriter(output, args.seq_len, args.shard_tokens)
    exact_seen: set[str] = set()
    part_seen: set[str] = set()
    prompted_source_code_seen: set[str] = set()
    normalized_seen: set[str] = set()
    near_deduper = NearDeduper(max_hamming=args.near_duplicate_hamming)
    stats: dict[str, Counter] = defaultdict(Counter)
    token_counts: dict[str, int] = defaultdict(int)
    source_rows: dict[str, int] = defaultdict(int)

    mix = dict(MIX_PROFILES[args.profile])
    if abs(sum(mix.values()) - 1.0) > 1e-9:
        raise ValueError(f"Mix profile {args.profile!r} must sum to 1.0")
    quotas = {bucket: int(args.total_tokens * ratio) for bucket, ratio in mix.items()}
    generators = {
        "fineweb_edu": stream_hf_texts("fineweb_edu", args.seed + 11),
        "wikipedia": stream_hf_texts("wikipedia", args.seed + 23),
        "wikipedia_short_fact": wikipedia_short_fact_documents(args.seed + 24),
        "fineweb_edu_dedup": stream_hf_texts("fineweb_edu_dedup", args.seed + 25),
        "cosmopedia": stream_hf_texts("cosmopedia", args.seed + 27),
        "tinystories": stream_hf_texts("tinystories", args.seed + 29),
        "code_python_prompted": stream_prompted_python_texts(args.seed + 31),
        "code_python": stream_hf_texts("code_python", args.seed + 37),
        "json_structured": json_documents(args.seed + 41),
        "retrieval_synthetic": retrieval_documents(args.seed + 53),
        "code_mbpp": mbpp_documents(args.seed + 59),
        "factual_cloze": factual_cloze_documents(args.seed + 61),
        "json_compact": compact_json_documents(args.seed + 67),
        "retrieval_prompted": prompted_retrieval_documents(args.seed + 69),
        "retrieval_long": long_retrieval_documents(tokenizer, args.seed + 71),
    }

    for bucket in mix:
        quota = quotas[bucket]
        for raw_text, metadata in generators[bucket]:
            source_rows[bucket] += 1
            text = (
                normalize_code_text(raw_text)
                if bucket in {"code_python", "code_python_prompted", "code_mbpp", "json_compact"}
                else normalize_web_text(raw_text)
            )
            if not text or "\x00" in text:
                stats[bucket]["empty_or_binary"] += 1
                continue
            if len(text) < args.minimum_characters:
                stats[bucket]["too_short_chars"] += 1
                continue
            if bucket in {
                "fineweb_edu",
                "fineweb_edu_dedup",
                "cosmopedia",
                "tinystories",
                "wikipedia",
                "wikipedia_short_fact",
                "code_python",
                "code_python_prompted",
            }:
                repetition, ngram_count = repeated_ngram_fraction(text)
                threshold = (
                    args.max_code_trigram_repetition
                    if bucket in {"code_python", "code_python_prompted"}
                    else args.max_prose_trigram_repetition
                )
                if ngram_count >= args.minimum_repetition_trigrams and repetition > threshold:
                    stats[bucket]["high_trigram_repetition"] += 1
                    continue
            normalized_digest = normalized_text_hash(text)
            if normalized_digest in normalized_seen:
                stats[bucket]["normalized_duplicate"] += 1
                continue
            full_ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
            if len(full_ids) < args.minimum_document_tokens:
                stats[bucket]["too_short_tokens"] += 1
                continue
            full_digest = document_hash(torch.tensor(full_ids, dtype=torch.int32))
            source_code_digest = None
            if bucket == "code_python_prompted":
                raw_code_ids = [
                    int(value)
                    for value in tokenizer.encode(metadata["raw_code"], add_special_tokens=False)
                ]
                source_code_digest = document_hash(torch.tensor(raw_code_ids, dtype=torch.int32))
                if source_code_digest in reference_hashes:
                    stats[bucket]["consumed_source_code_duplicate"] += 1
                    continue
                if source_code_digest in prompted_source_code_seen:
                    stats[bucket]["source_code_duplicate"] += 1
                    continue
            elif bucket == "code_python" and full_digest in prompted_source_code_seen:
                stats[bucket]["prompted_source_code_duplicate"] += 1
                continue
            if full_digest in reference_hashes:
                stats[bucket]["consumed_exact_duplicate"] += 1
                continue
            if full_digest in exact_seen:
                stats[bucket]["exact_duplicate"] += 1
                continue
            if bucket not in NEAR_DEDUP_EXEMPT_BUCKETS and not near_deduper.add_if_novel(text):
                stats[bucket]["near_duplicate"] += 1
                continue
            normalized_seen.add(normalized_digest)
            exact_seen.add(full_digest)
            if source_code_digest is not None:
                prompted_source_code_seen.add(source_code_digest)

            accepted_any = False
            for part in split_long_document(full_ids, args.maximum_document_tokens):
                if len(part) < args.minimum_document_tokens:
                    continue
                part_digest = document_hash(torch.tensor(part, dtype=torch.int32))
                if part_digest in reference_hashes:
                    stats[bucket]["consumed_exact_chunk_duplicate"] += 1
                    continue
                if part_digest in part_seen:
                    stats[bucket]["exact_chunk_duplicate"] += 1
                    continue
                if token_counts[bucket] + len(part) + 1 > quota + args.quota_tolerance_tokens:
                    continue
                writer.add_document(bucket, torch.tensor(part, dtype=torch.int32), eos_token_id)
                part_seen.add(part_digest)
                token_counts[bucket] += len(part) + 1
                stats[bucket]["accepted_chunks"] += 1
                accepted_any = True
                if token_counts[bucket] >= quota:
                    break
            if accepted_any:
                stats[bucket]["accepted_documents"] += 1
            if token_counts[bucket] >= quota:
                break

    train_shards, discarded_tail_tokens = writer.close()
    validation_shards = copy_compact_split(validation_source, output, "validation")
    test_shards = copy_compact_split(validation_source, output, "test")
    packed_tokens = sum(int(row["tokens"]) for row in train_shards)
    train_sequences = sum(int(row["sequences"]) for row in train_shards)
    usable_sequences = (train_sequences // args.grad_accum_steps) * args.grad_accum_steps
    usable_tokens = usable_sequences * args.seq_len

    manifest = {
        "created_by": "scripts/prepare_balanced_novel_pilot.py",
        "format": "balanced_exact_novel_base_pretraining_v1",
        "purpose": "Bounded mixed-domain non-regression pilot from the frozen 500M base",
        "assistant_masking": False,
        "labels": "implicit next-token labels over every active token",
        "tokenizer": str(tokenizer_path),
        "tokenizer_vocab_size": len(tokenizer),
        "eos_token_id": eos_token_id,
        "seed": args.seed,
        "profile": args.profile,
        "seq_len": args.seq_len,
        "requested_tokens": args.total_tokens,
        "mix": mix,
        "plan": quotas,
        "bucket_order": list(mix),
        "tokens_by_bucket": dict(token_counts),
        "packed_tokens": packed_tokens,
        "train_sequences": train_sequences,
        "usable_sequences_grad_accum_aligned": usable_sequences,
        "usable_tokens_grad_accum_aligned": usable_tokens,
        "discarded_tail_tokens": discarded_tail_tokens,
        "storage_dtypes": {"input_ids": "int32", "attention_mask": "bool"},
        "interleave_policy": "indirect deterministic bucket interleave at training time",
        "source_revisions": SOURCE_REVISIONS,
        "reference_datasets": [str(path) for path in references],
        "reference_audit": reference_report,
        "deduplication": {
            "exact": "global BLAKE2b-128 over canonical int32 token bytes",
            "normalized": "normalized source text hash",
            "near": f"64-bit trigram SimHash, Hamming <= {args.near_duplicate_hamming}",
            "near_exempt_deterministic_synthetic_buckets": sorted(NEAR_DEDUP_EXEMPT_BUCKETS),
            "post_build_reference_overlap": None,
            "post_build_unique_documents": None,
        },
        "quality_filters": {
            "trigram_repetition_definition": "1 - unique token trigrams / total token trigrams",
            "minimum_repetition_trigrams": args.minimum_repetition_trigrams,
            "max_prose_trigram_repetition": args.max_prose_trigram_repetition,
            "max_code_trigram_repetition": args.max_code_trigram_repetition,
            "repetition_filtered_buckets": [
                "fineweb_edu",
                "fineweb_edu_dedup",
                "cosmopedia",
                "tinystories",
                "wikipedia",
                "wikipedia_short_fact",
                "code_python",
                "code_python_prompted",
            ],
        },
        "source_rows_seen": dict(source_rows),
        "filter_stats": {key: dict(value) for key, value in stats.items()},
        "validation_source": str(validation_source),
        "validation_manifest_sha256": file_sha256(validation_source / "manifest.json"),
        "shards": train_shards + validation_shards + test_shards,
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")

    output_hashes: set[str] = set()
    overlap_count = 0
    for _, document in iter_documents(output, eos_token_id):
        digest = document_hash(document)
        output_hashes.add(digest)
        if digest in reference_hashes:
            overlap_count += 1
    if overlap_count:
        raise RuntimeError(f"Post-build exact overlap audit failed: {overlap_count}")
    manifest["deduplication"]["post_build_reference_overlap"] = overlap_count
    manifest["deduplication"]["post_build_unique_documents"] = len(output_hashes)
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, output / "manifest.json")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prepared_data" / "v3_balanced_novel_5m_pilot_ctx2048")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers" / "fdt_v3_bpe_24k")
    parser.add_argument(
        "--validation-source",
        type=Path,
        default=ROOT / "prepared_data" / "v3_broad_additional_300m_combined_v2_24k_ctx2048",
    )
    parser.add_argument("--reference-dataset", type=Path, action="append", default=[])
    parser.add_argument("--total-tokens", type=int, default=5_000_000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--shard-tokens", type=int, default=5_000_000)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--minimum-characters", type=int, default=160)
    parser.add_argument("--minimum-document-tokens", type=int, default=32)
    parser.add_argument("--maximum-document-tokens", type=int, default=4096)
    parser.add_argument("--quota-tolerance-tokens", type=int, default=8192)
    parser.add_argument("--near-duplicate-hamming", type=int, default=3)
    parser.add_argument("--minimum-repetition-trigrams", type=int, default=64)
    parser.add_argument("--max-prose-trigram-repetition", type=float, default=0.20)
    parser.add_argument("--max-code-trigram-repetition", type=float, default=0.32)
    parser.add_argument("--profile", choices=tuple(MIX_PROFILES), default="balanced_v1")
    args = parser.parse_args()
    if not args.reference_dataset:
        args.reference_dataset = [
            ROOT / "prepared_data" / "v3_broad_additional_300m_combined_v2_24k_ctx2048",
            ROOT / "prepared_data" / "v3_broad_200m_24k_ctx2048",
            ROOT / "prepared_data" / "v3_cosmopedia_novel_22m_24k_ctx2048",
            ROOT / "prepared_data" / "v3_exact_novel_remaining_after_500m",
            ROOT / "prepared_data" / "v3_base_capability_pilot_640k_ctx2048",
        ]
    manifest = prepare(args)
    print(json.dumps({
        "output_dir": str(require_c_path(args.output_dir)),
        "packed_tokens": manifest["packed_tokens"],
        "usable_tokens": manifest["usable_tokens_grad_accum_aligned"],
        "tokens_by_bucket": manifest["tokens_by_bucket"],
        "post_build_overlap": manifest["deduplication"]["post_build_reference_overlap"],
    }, indent=2))


if __name__ == "__main__":
    main()
