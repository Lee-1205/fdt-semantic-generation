from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset


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
    file_sha256,
    random_code,
    reference_hashes,
    require_c_path,
    token_hash,
)
from prepare_capability_completion_v16 import CATEGORY_NAMES, CATEGORY_TO_ID, V16Writer
from prepare_capability_completion_v18 import factual_length_bucket, repeated_ngram_rate


EXPERIMENT_NONCE = "v19-copy-grounding-3c1dc84c62be4b2fbf0e681899b3b717"
HOTPOT_DATASET = "hotpotqa/hotpot_qa"
HOTPOT_CONFIG = "distractor"
HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HOTPOT_LICENSE = "cc-by-sa-4.0"
CATEGORY_COUNTS = {name: 1000 for name in CATEGORY_NAMES}
CATEGORY_WEIGHTS = {
    "factual_qa": 0.25,
    "retrieval": 0.20,
    "python_code": 0.15,
    "json": 0.10,
    "natural_lm": 0.30,
}
ANSWER_CAPS = {"factual_qa": 32, "retrieval": 32, "python_code": 96, "json": 96}
FACTUAL_LENGTH_QUOTAS = {"short_1_4": 600, "medium_5_7": 250, "long_8_plus": 150}
NATURAL_SOURCE_QUOTAS = {"wikipedia": 400, "fineweb_edu_dedup": 300, "tinystories": 300}
NATURAL_ANSWER_TOKENS = 64
MAX_NATURAL_REPEATED_4GRAM_RATE = 0.05


def overlap_span(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def token_span_mask(offsets: list[tuple[int, int]], spans: list[tuple[int, int]]) -> list[int]:
    return [int(any(overlap_span(offset, span) for span in spans)) for offset in offsets]


def json_value_spans(serialized: str, value: object) -> list[tuple[int, int]]:
    leaves: list[object] = []

    def collect(current: object) -> None:
        if isinstance(current, dict):
            for child in current.values():
                collect(child)
        elif isinstance(current, list):
            for child in current:
                collect(child)
        else:
            leaves.append(current)

    collect(value)
    spans: list[tuple[int, int]] = []
    for leaf in leaves:
        encoded = json.dumps(leaf, ensure_ascii=True, separators=(",", ":"))
        start = 0
        while True:
            position = serialized.find(encoded, start)
            if position < 0:
                break
            spans.append((position, position + len(encoded)))
            start = position + len(encoded)
    return sorted(set(spans))


def python_signature_span(code: str) -> tuple[int, int] | None:
    match = re.search(r"(?m)^\s*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\([^\n]*\)\s*(?:->\s*[^:\n]+)?\s*:", code)
    return match.span() if match else None


class V19Writer(V16Writer):
    def __init__(self, output: Path, tokenizer, max_length: int, reference: set[str]):
        super().__init__(output, tokenizer, max_length, reference)
        self.serial = 0
        self.grounding_masks: list[list[int]] = []
        self.signature_masks: list[list[int]] = []

    def marker(self, category: str) -> str:
        self.serial += 1
        return f"Source record {EXPERIMENT_NONCE}-{category}-{self.serial:07d}.\n"

    def add_grounded(
        self,
        prompt: str,
        answer: str,
        category: str,
        max_answer_tokens: int,
        grounding_spans: list[tuple[int, int]] | None = None,
        signature_spans: list[tuple[int, int]] | None = None,
    ) -> bool:
        answer_text = " " + answer.strip()
        encoded = self.tokenizer(
            answer_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        answer_ids = [int(value) for value in encoded["input_ids"][:max_answer_tokens]]
        offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"][:max_answer_tokens]]
        grounding = [(start + 1, end + 1) for start, end in (grounding_spans or [])]
        signatures = [(start + 1, end + 1) for start, end in (signature_spans or [])]
        answer_grounding = token_span_mask(offsets, grounding) + [0]
        answer_signature = token_span_mask(offsets, signatures) + [0]
        answer_ids.append(int(self.tokenizer.eos_token_id))

        prompt_ids = list(self.tokenizer.encode(self.marker(category) + prompt, add_special_tokens=False))
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
        self.grounding_masks.append([0] * len(prompt_ids) + answer_grounding + [0] * padding)
        self.signature_masks.append([0] * len(prompt_ids) + answer_signature + [0] * padding)
        self.categories[category] += 1
        self.row_categories.append(CATEGORY_TO_ID[category])
        return True

    def add_ids(self, prompt_ids: list[int], answer_ids: list[int], category: str) -> bool:
        accepted = super().add_ids(
            list(self.tokenizer.encode(self.marker(category), add_special_tokens=False)) + prompt_ids,
            answer_ids,
            category,
        )
        if accepted:
            self.grounding_masks.append([0] * self.max_length)
            self.signature_masks.append([0] * self.max_length)
        return accepted

    def save(self) -> dict:
        directory = self.output / "shards" / "train"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "shard_000000.pt"
        payload = {
            "input_ids": torch.tensor(self.ids, dtype=torch.int32),
            "labels": torch.tensor(self.labels, dtype=torch.int32),
            "attention_mask": torch.tensor(self.masks, dtype=torch.uint8),
            "category_ids": torch.tensor(self.row_categories, dtype=torch.int16),
            "grounding_mask": torch.tensor(self.grounding_masks, dtype=torch.uint8),
            "signature_mask": torch.tensor(self.signature_masks, dtype=torch.uint8),
        }
        torch.save(payload, path)
        return {
            "rows": len(self.ids),
            "categories": dict(self.categories),
            "category_names": list(CATEGORY_NAMES),
            "active_tokens": sum(sum(mask) for mask in self.masks),
            "supervised_tokens": sum(sum(label != -100 for label in labels) for labels in self.labels),
            "grounding_supervised_tokens": sum(sum(mask) for mask in self.grounding_masks),
            "signature_supervised_tokens": sum(sum(mask) for mask in self.signature_masks),
            "unique_active_rows": len(self.seen),
            "rejections": dict(self.rejections),
            "shard": str(path),
            "shard_sha256": file_sha256(path),
        }


def hotpot_support_context(row: dict) -> str:
    context = row.get("context", {})
    titles = [str(value) for value in context.get("title", [])]
    sentences = context.get("sentences", [])
    by_title = {title: index for index, title in enumerate(titles)}
    selected: list[str] = []
    seen: set[tuple[str, int]] = set()
    support = row.get("supporting_facts", {})
    for title, sentence_id in zip(support.get("title", []), support.get("sent_id", [])):
        key = (str(title), int(sentence_id))
        if key in seen:
            continue
        seen.add(key)
        title_index = by_title.get(key[0])
        if title_index is None or title_index >= len(sentences):
            continue
        paragraph = sentences[title_index]
        if key[1] < 0 or key[1] >= len(paragraph):
            continue
        selected.append(f"{key[0]}: {str(paragraph[key[1]]).strip()}")
    return " ".join(selected)


def add_factual_grounded(writer: V19Writer, rows: int, seed: int) -> dict:
    if rows != sum(FACTUAL_LENGTH_QUOTAS.values()):
        raise ValueError("V19 factual row count must match the predeclared quotas")
    dataset = load_dataset(
        HOTPOT_DATASET,
        HOTPOT_CONFIG,
        split="train",
        streaming=True,
        revision=HOTPOT_REVISION,
    ).shuffle(seed=seed, buffer_size=30_000)
    accepted = Counter()
    for row in dataset:
        answer = str(row.get("answer", "")).strip()
        if not answer or answer.lower() in {"yes", "no"}:
            continue
        support = hotpot_support_context(row)
        if not support or answer.lower() not in support.lower():
            writer.rejections["factual_qa_answer_not_in_support"] += 1
            continue
        token_count = len(writer.tokenizer.encode(" " + answer, add_special_tokens=False))
        bucket = factual_length_bucket(token_count)
        if accepted[bucket] >= FACTUAL_LENGTH_QUOTAS[bucket]:
            continue
        prompt = f"Evidence: {support}\nQuestion: {str(row.get('question', '')).strip()}\nAnswer:"
        if writer.add_grounded(
            prompt,
            answer,
            "factual_qa",
            ANSWER_CAPS["factual_qa"],
            grounding_spans=[(0, len(answer))],
        ):
            accepted[bucket] += 1
        if sum(accepted.values()) >= rows:
            break
    if dict(accepted) != FACTUAL_LENGTH_QUOTAS:
        raise RuntimeError(f"could not fill V19 HotpotQA factual quotas: {dict(accepted)}")
    return dict(accepted)


def add_retrieval_grounded(writer: V19Writer, rows: int, seed: int) -> dict:
    rng = random.Random(seed)
    accepted = Counter()
    while writer.categories["retrieval"] < rows:
        index = writer.categories["retrieval"]
        code = random_code(rng)
        if index % 2 == 0:
            prompt = f"Context: The current access code is {code}.\nReturn the current access code exactly:"
            family = "direct"
        else:
            count = (8, 16, 24)[(index // 2) % 3]
            target = max(1, int(count * 0.7))
            records = [
                f"Memo {slot + 1}: identifier {code if slot == target else random_code(rng)} is "
                f"{'CURRENT' if slot == target else 'obsolete'}."
                for slot in range(count)
            ]
            prompt = "\n".join(records) + "\nReturn the CURRENT identifier exactly:"
            family = f"distractor_{count}"
        if writer.add_grounded(
            prompt,
            code,
            "retrieval",
            ANSWER_CAPS["retrieval"],
            grounding_spans=[(0, len(code))],
        ):
            accepted[family] += 1
    return dict(accepted)


def add_python_grounded(writer: V19Writer, rows: int, seed: int) -> dict:
    dataset = load_dataset(
        "google/code_x_glue_ct_code_to_text",
        "python",
        split="train",
        streaming=True,
        revision=SOURCE_REVISIONS["code_x_glue_python"],
    ).shuffle(seed=seed, buffer_size=20_000)
    accepted = Counter()
    for row in dataset:
        task = str(row.get("docstring") or row.get("docstring_tokens") or "").strip()
        code = str(row.get("code") or "").strip()
        signature = python_signature_span(code)
        if not task or not code or signature is None:
            writer.rejections["python_code_missing_signature"] += 1
            continue
        signature_text = code[signature[0] : signature[1]].strip()
        prompt = (
            "Write only valid Python code for the task. Preserve the requested function signature exactly.\n"
            f"Task: {task}\nRequired signature: {signature_text}\nPython:\n"
        )
        if writer.add_grounded(
            prompt,
            code,
            "python_code",
            ANSWER_CAPS["python_code"],
            signature_spans=[signature],
        ):
            accepted["signature_grounded"] += 1
        if writer.categories["python_code"] >= rows:
            break
    if writer.categories["python_code"] != rows:
        raise RuntimeError(f"could not fill V19 Python rows: {writer.categories['python_code']}")
    return dict(accepted)


def add_json_grounded(writer: V19Writer, rows: int, seed: int) -> dict:
    rng = random.Random(seed)
    names = ("Ada", "Grace", "Linus", "Margaret", "Alan", "Edsger", "Barbara", "Donald")
    cities = ("London", "Paris", "Seoul", "Toronto", "Berlin", "Oslo", "Tokyo", "Dublin")
    accepted = Counter()
    index = 0
    while writer.categories["json"] < rows:
        family = index % 4
        if family == 0:
            value = {"record_id": f"V19-{seed}-{index}", "name": rng.choice(names), "age": rng.randrange(18, 80), "city": rng.choice(cities), "active": bool(rng.randrange(2))}
            prompt = f"record_id={value['record_id']}; name={value['name']}; age={value['age']}; city={value['city']}; active={str(value['active']).lower()}\nJSON:"
            name = "record"
        elif family == 1:
            value = {"sku": f"SKU-{rng.randrange(100000, 999999)}", "quantity": rng.randrange(1, 50), "tags": [rng.choice(("new", "sale", "fragile")), rng.choice(("blue", "green", "red"))]}
            prompt = f"sku={value['sku']}; quantity={value['quantity']}; tags={','.join(value['tags'])}\nJSON:"
            name = "inventory"
        elif family == 2:
            value = {"event": "login" if index % 8 else "logout", "user": rng.choice(names), "success": bool(rng.randrange(2)), "attempt": rng.randrange(1, 8)}
            prompt = f"event={value['event']}; user={value['user']}; success={str(value['success']).lower()}; attempt={value['attempt']}\nJSON:"
            name = "event"
        else:
            value = {"name": rng.choice(names), "location": {"city": rng.choice(cities), "country_code": rng.choice(("GB", "FR", "KR", "CA", "DE", "NO", "JP", "IE"))}, "scores": [rng.randrange(0, 101), rng.randrange(0, 101)]}
            prompt = f"name={value['name']}; city={value['location']['city']}; country_code={value['location']['country_code']}; scores={value['scores'][0]},{value['scores'][1]}\nJSON:"
            name = "nested"
        answer = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        if writer.add_grounded(
            "Convert the source fields to one compact JSON object. Copy every value exactly.\n" + prompt,
            answer,
            "json",
            ANSWER_CAPS["json"],
            grounding_spans=json_value_spans(answer, value),
        ):
            accepted[name] += 1
        index += 1
    return dict(accepted)


def add_natural_stable(writer: V19Writer, rows: int, seed: int) -> dict:
    if rows != sum(NATURAL_SOURCE_QUOTAS.values()):
        raise ValueError("V19 natural row count must match the source quotas")
    rng = random.Random(seed)
    streams = {name: stream_hf_texts(name, seed + index + 1) for index, name in enumerate(NATURAL_SOURCE_QUOTAS)}
    accepted = Counter()
    names = tuple(NATURAL_SOURCE_QUOTAS)
    source_index = 0
    for attempts in range(200_000):
        if writer.categories["natural_lm"] >= rows:
            break
        eligible = [name for name in names if accepted[name] < NATURAL_SOURCE_QUOTAS[name]]
        source = eligible[source_index % len(eligible)]
        source_index += 1
        text, _ = next(streams[source])
        text = normalize_web_text(text)
        ids = list(writer.tokenizer.encode(text, add_special_tokens=False)) if text and "\x00" not in text else []
        if len(ids) < 192:
            writer.rejections[f"natural_lm_{source}_short"] += 1
            continue
        prompt_tokens = min(320, len(ids) - NATURAL_ANSWER_TOKENS)
        span = prompt_tokens + NATURAL_ANSWER_TOKENS
        offset = rng.randrange(0, len(ids) - span + 1) if len(ids) > span else 0
        active = ids[offset : offset + span]
        if repeated_ngram_rate(active[-NATURAL_ANSWER_TOKENS:], 4) > MAX_NATURAL_REPEATED_4GRAM_RATE:
            writer.rejections[f"natural_lm_{source}_repetitive"] += 1
            continue
        if writer.add_ids(active[:-NATURAL_ANSWER_TOKENS], active[-NATURAL_ANSWER_TOKENS:], "natural_lm"):
            accepted[source] += 1
    if writer.categories["natural_lm"] != rows:
        raise RuntimeError(f"could not fill V19 natural rows: {dict(accepted)}")
    return dict(accepted)


def category_token_report(writer: V19Writer) -> dict:
    report = {}
    for category_index, category in enumerate(CATEGORY_NAMES):
        rows = [index for index, value in enumerate(writer.row_categories) if value == category_index]
        report[category] = {
            "rows": len(rows),
            "active_tokens": sum(sum(writer.masks[index]) for index in rows),
            "completion_supervised_tokens": sum(sum(value != -100 for value in writer.labels[index]) for index in rows),
            "grounding_supervised_tokens": sum(sum(writer.grounding_masks[index]) for index in rows),
            "signature_supervised_tokens": sum(sum(writer.signature_masks[index]) for index in rows),
            "objective_category_weight": CATEGORY_WEIGHTS[category],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V19 copy-and-grounding completion data.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=ROOT / "prepared_data")
    parser.add_argument("--validation-source", type=Path, default=ROOT / "prepared_data" / "capability_curriculum_5k")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers" / "fdt_v3_bpe_24k")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260819)
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
    writer = V19Writer(output, tokenizer, args.max_length, references)

    factual = add_factual_grounded(writer, CATEGORY_COUNTS["factual_qa"], args.seed + 1)
    retrieval = add_retrieval_grounded(writer, CATEGORY_COUNTS["retrieval"], args.seed + 2)
    python = add_python_grounded(writer, CATEGORY_COUNTS["python_code"], args.seed + 3)
    json_families = add_json_grounded(writer, CATEGORY_COUNTS["json"], args.seed + 4)
    natural = add_natural_stable(writer, CATEGORY_COUNTS["natural_lm"], args.seed + 5)
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
        raise RuntimeError("V19 local exact-row collision")
    overlap = sum(token_hash(row) in references for row in post_rows)
    if overlap:
        raise RuntimeError(f"V19 post-build exact overlap is nonzero: {overlap}")

    report = {
        "format": "copy_grounding_category_weighted_completion_v1",
        "purpose": "V19 source binding, short fact/EOS recovery, signature grounding, and copied JSON values",
        "conversational_sft": False,
        "completion_span_supervision": True,
        "supervised_eos": True,
        "eos_loss_weight": 2.0,
        "grounding_loss_weight": 2.0,
        "signature_loss_weight": 3.0,
        "category_normalized_per_example_loss": True,
        "category_objective_weights": CATEGORY_WEIGHTS,
        "prompt_lm_weight": 0.25,
        "experiment_nonce": EXPERIMENT_NONCE,
        "historical_payload_limitation": "some consumed payloads were pruned after manifest/hash journaling; nonce-prefixed rows provide construction-based full-row novelty",
        "tokenizer": str(tokenizer_path),
        "max_length": args.max_length,
        "seed": args.seed,
        "source_revisions": {
            "hotpot_qa": HOTPOT_REVISION,
            "code_x_glue_python": SOURCE_REVISIONS["code_x_glue_python"],
            **{name: NATURAL_REVISIONS[name] for name in natural},
        },
        "source_provenance": {
            "factual_qa": {"dataset": HOTPOT_DATASET, "config": HOTPOT_CONFIG, "revision": HOTPOT_REVISION, "license": HOTPOT_LICENSE, "construction": "answer must occur in supporting-fact sentences"},
            "python_code": {"dataset": "google/code_x_glue_ct_code_to_text", "config": "python", "revision": SOURCE_REVISIONS["code_x_glue_python"]},
        },
        "train": train,
        "category_token_report": token_report,
        "answer_token_caps": ANSWER_CAPS,
        "factual_length_quotas": factual,
        "retrieval_families": retrieval,
        "python_families": python,
        "json_families": json_families,
        "natural_sources": natural,
        "natural_answer_tokens": NATURAL_ANSWER_TOKENS,
        "max_natural_repeated_4gram_rate": MAX_NATURAL_REPEATED_4GRAM_RATE,
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
        "validation_shard_sha256": file_sha256(target_validation),
        "rows": train["rows"],
        "categories": train["categories"],
        "category_token_report": token_report,
        "post_build_exact_overlap": overlap,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
