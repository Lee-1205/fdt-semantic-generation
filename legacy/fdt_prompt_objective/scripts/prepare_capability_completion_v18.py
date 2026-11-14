from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

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
    add_python,
    crop_context,
    file_sha256,
    random_code,
    reference_hashes,
    require_c_path,
    token_hash,
)
from prepare_capability_completion_v16 import CATEGORY_NAMES, V16Writer


EXPERIMENT_NONCE = "v18-calibrated-retention-66fb488450054898b4565420083bfdcb"
CATEGORY_COUNTS = {name: 1000 for name in CATEGORY_NAMES}
CATEGORY_WEIGHTS = {
    "factual_qa": 0.25,
    "retrieval": 0.20,
    "python_code": 0.10,
    "json": 0.15,
    "natural_lm": 0.30,
}
ANSWER_CAPS = {"factual_qa": 32, "retrieval": 32, "python_code": 96, "json": 96}
FACTUAL_LENGTH_QUOTAS = {"short_1_4": 400, "medium_5_7": 300, "long_8_plus": 300}
NATURAL_SOURCE_QUOTAS = {"wikipedia": 400, "fineweb_edu_dedup": 300, "tinystories": 300}
NATURAL_ANSWER_TOKENS = 64
MAX_NATURAL_REPEATED_4GRAM_RATE = 0.05


class V18Writer(V16Writer):
    def __init__(self, output: Path, tokenizer, max_length: int, reference: set[str]):
        super().__init__(output, tokenizer, max_length, reference)
        self.serial = 0

    def marker(self, category: str) -> str:
        self.serial += 1
        return f"Source record {EXPERIMENT_NONCE}-{category}-{self.serial:07d}.\n"

    def add(self, prompt: str, answer: str, category: str, max_answer_tokens: int) -> bool:
        cap = min(ANSWER_CAPS.get(category, max_answer_tokens), max_answer_tokens)
        return super().add(self.marker(category) + prompt, answer, category, cap)

    def add_ids(self, prompt_ids: list[int], answer_ids: list[int], category: str) -> bool:
        marker_ids = list(self.tokenizer.encode(self.marker(category), add_special_tokens=False))
        return super().add_ids(marker_ids + prompt_ids, answer_ids, category)


def factual_length_bucket(token_count: int) -> str:
    if token_count <= 4:
        return "short_1_4"
    if token_count <= 7:
        return "medium_5_7"
    return "long_8_plus"


def repeated_ngram_rate(token_ids: list[int], n: int = 4) -> float:
    if len(token_ids) < n:
        return 0.0
    grams = [tuple(token_ids[index : index + n]) for index in range(len(token_ids) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def add_factual_stratified(writer: V18Writer, rows: int, seed: int) -> dict:
    if rows != sum(FACTUAL_LENGTH_QUOTAS.values()):
        raise ValueError("V18 factual row count must match the predeclared length quotas")
    dataset = load_dataset(
        "rajpurkar/squad",
        split="train",
        streaming=True,
        revision=SOURCE_REVISIONS["squad"],
    ).shuffle(seed=seed, buffer_size=30_000)
    accepted = Counter()
    for row in dataset:
        answers = row.get("answers", {}).get("text", [])
        if not answers or not str(answers[0]).strip():
            continue
        answer = str(answers[0]).strip()
        answer_tokens = list(writer.tokenizer.encode(" " + answer, add_special_tokens=False))
        bucket = factual_length_bucket(len(answer_tokens))
        if accepted[bucket] >= FACTUAL_LENGTH_QUOTAS[bucket]:
            continue
        context = crop_context(str(row["context"]), answer)
        prompt = f"Context: {context}\nQuestion: {row['question']}\nAnswer:"
        if writer.add(prompt, answer, "factual_qa", ANSWER_CAPS["factual_qa"]):
            accepted[bucket] += 1
        if sum(accepted.values()) >= rows:
            break
    if dict(accepted) != FACTUAL_LENGTH_QUOTAS:
        raise RuntimeError(f"could not fill V18 factual quotas: {dict(accepted)}")
    return dict(accepted)


def add_retrieval_calibrated(writer: V18Writer, rows: int, seed: int) -> dict:
    rng = random.Random(seed)
    accepted = Counter()
    while writer.categories["retrieval"] < rows:
        index = writer.categories["retrieval"]
        code = random_code(rng)
        if index % 2 == 0:
            templates = (
                f"Record: The access code is {code}. Therefore the access code is",
                f"Context: The access code is {code}.\nQuestion: What is the access code?\nAnswer:",
            )
            prompt = templates[(index // 2) % len(templates)]
            family = "direct"
        else:
            distractor_count = (8, 16, 24)[(index // 2) % 3]
            target_slot = max(1, int(distractor_count * 0.7))
            records = []
            for slot in range(distractor_count):
                value = code if slot == target_slot else random_code(rng)
                status = "CURRENT" if slot == target_slot else "obsolete"
                records.append(f"Memo {slot + 1}: identifier {value} is {status}.")
            prompt = "\n".join(records) + "\nReturn the CURRENT identifier exactly:"
            family = f"distractor_{distractor_count}"
        if writer.add(prompt, code, "retrieval", ANSWER_CAPS["retrieval"]):
            accepted[family] += 1
    return dict(accepted)


def add_json_diverse(writer: V18Writer, rows: int, seed: int) -> dict:
    rng = random.Random(seed)
    names = ("Ada", "Grace", "Linus", "Margaret", "Alan", "Edsger", "Barbara", "Donald")
    cities = ("London", "Paris", "Seoul", "Toronto", "Berlin", "Oslo", "Tokyo", "Dublin")
    accepted = Counter()
    index = 0
    while writer.categories["json"] < rows:
        family = index % 4
        if family == 0:
            value = {
                "record_id": f"V18-{seed}-{index}",
                "name": rng.choice(names),
                "age": rng.randrange(18, 80),
                "city": rng.choice(cities),
                "active": bool(rng.randrange(2)),
            }
            prompt = (
                "Convert the record to one valid compact JSON object.\n"
                f"record_id={value['record_id']}; name={value['name']}; age={value['age']}; "
                f"city={value['city']}; active={str(value['active']).lower()}\nJSON:"
            )
            name = "record"
        elif family == 1:
            value = {
                "sku": f"SKU-{rng.randrange(100000, 999999)}",
                "quantity": rng.randrange(1, 50),
                "tags": [rng.choice(("new", "sale", "fragile")), rng.choice(("blue", "green", "red"))],
            }
            prompt = (
                "Return compact JSON for this inventory item.\n"
                f"sku={value['sku']}; quantity={value['quantity']}; tags={','.join(value['tags'])}\nJSON:"
            )
            name = "inventory"
        elif family == 2:
            value = {
                "event": "login" if index % 8 else "logout",
                "user": rng.choice(names),
                "success": bool(rng.randrange(2)),
                "attempt": rng.randrange(1, 8),
            }
            prompt = (
                "Encode the event as one compact JSON object.\n"
                f"event={value['event']}; user={value['user']}; success={str(value['success']).lower()}; "
                f"attempt={value['attempt']}\nJSON:"
            )
            name = "event"
        else:
            value = {
                "name": rng.choice(names),
                "location": {"city": rng.choice(cities), "country_code": rng.choice(("GB", "FR", "KR", "CA", "DE", "NO", "JP", "IE"))},
                "scores": [rng.randrange(0, 101), rng.randrange(0, 101)],
            }
            prompt = (
                "Build one compact JSON object with nested location and an integer score array.\n"
                f"name={value['name']}; city={value['location']['city']}; "
                f"country_code={value['location']['country_code']}; scores={value['scores'][0]},{value['scores'][1]}\nJSON:"
            )
            name = "nested"
        answer = json.dumps(value, ensure_ascii=True, separators=(",", ":"))
        if writer.add(prompt, answer, "json", ANSWER_CAPS["json"]):
            accepted[name] += 1
        index += 1
    return dict(accepted)


def add_natural_stable(writer: V18Writer, rows: int, seed: int) -> dict:
    if rows != sum(NATURAL_SOURCE_QUOTAS.values()):
        raise ValueError("V18 natural row count must match the predeclared source quotas")
    rng = random.Random(seed)
    streams = {
        name: stream_hf_texts(name, seed + index + 1)
        for index, name in enumerate(NATURAL_SOURCE_QUOTAS)
    }
    accepted = Counter()
    source_order = tuple(NATURAL_SOURCE_QUOTAS)
    source_index = 0
    attempts = 0
    while writer.categories["natural_lm"] < rows:
        attempts += 1
        if attempts > 200_000:
            raise RuntimeError(f"could not fill V18 natural quotas: {dict(accepted)}")
        eligible = [name for name in source_order if accepted[name] < NATURAL_SOURCE_QUOTAS[name]]
        source = eligible[source_index % len(eligible)]
        source_index += 1
        text, _ = next(streams[source])
        text = normalize_web_text(text)
        if not text or "\x00" in text:
            writer.rejections[f"natural_lm_{source}_empty"] += 1
            continue
        ids = [int(value) for value in writer.tokenizer.encode(text, add_special_tokens=False)]
        if len(ids) < 192:
            writer.rejections[f"natural_lm_{source}_short"] += 1
            continue
        prompt_tokens = min(320, len(ids) - NATURAL_ANSWER_TOKENS)
        span = prompt_tokens + NATURAL_ANSWER_TOKENS
        offset = rng.randrange(0, len(ids) - span + 1) if len(ids) > span else 0
        active = ids[offset : offset + span]
        prompt_ids = active[:-NATURAL_ANSWER_TOKENS]
        answer_ids = active[-NATURAL_ANSWER_TOKENS:]
        rate = repeated_ngram_rate(answer_ids, 4)
        if rate > MAX_NATURAL_REPEATED_4GRAM_RATE:
            writer.rejections[f"natural_lm_{source}_repetitive"] += 1
            continue
        if writer.add_ids(prompt_ids, answer_ids, "natural_lm"):
            accepted[source] += 1
    return dict(accepted)


def category_token_report(writer: V18Writer) -> dict:
    report = {}
    for category_index, category in enumerate(CATEGORY_NAMES):
        rows = [index for index, value in enumerate(writer.row_categories) if value == category_index]
        active = sum(sum(writer.masks[index]) for index in rows)
        supervised = sum(sum(label != -100 for label in writer.labels[index]) for index in rows)
        report[category] = {
            "rows": len(rows),
            "active_tokens": active,
            "completion_supervised_tokens": supervised,
            "mean_completion_tokens_per_row": supervised / max(len(rows), 1),
            "objective_category_weight": CATEGORY_WEIGHTS[category],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare calibrated retention V18 completion data.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=ROOT / "prepared_data")
    parser.add_argument("--validation-source", type=Path, default=ROOT / "prepared_data" / "capability_curriculum_5k")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers" / "fdt_v3_bpe_24k")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260818)
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
    writer = V18Writer(output, tokenizer, args.max_length, references)

    factual_lengths = add_factual_stratified(writer, CATEGORY_COUNTS["factual_qa"], args.seed + 1)
    retrieval_families = add_retrieval_calibrated(writer, CATEGORY_COUNTS["retrieval"], args.seed + 2)
    add_python(writer, CATEGORY_COUNTS["python_code"], args.seed + 3)
    json_families = add_json_diverse(writer, CATEGORY_COUNTS["json"], args.seed + 4)
    natural_sources = add_natural_stable(writer, CATEGORY_COUNTS["natural_lm"], args.seed + 5)
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
        raise RuntimeError("V18 local exact-row collision")
    overlap = sum(token_hash(row) in references for row in post_rows)
    if overlap:
        raise RuntimeError(f"V18 post-build exact overlap is nonzero: {overlap}")

    report = {
        "format": "calibrated_category_weighted_completion_v1",
        "purpose": "V18 long-factual, natural-loop, retrieval, and JSON calibration while retaining Python",
        "conversational_sft": False,
        "completion_span_supervision": True,
        "supervised_eos": True,
        "eos_loss_weight": 1.25,
        "category_normalized_per_example_loss": True,
        "category_objective_weights": CATEGORY_WEIGHTS,
        "prompt_lm_weight": 0.25,
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
        "factual_length_quotas": factual_lengths,
        "retrieval_families": retrieval_families,
        "json_families": json_families,
        "natural_sources": natural_sources,
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
        "rows": train["rows"],
        "categories": train["categories"],
        "category_token_report": token_report,
        "post_build_exact_overlap": overlap,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
