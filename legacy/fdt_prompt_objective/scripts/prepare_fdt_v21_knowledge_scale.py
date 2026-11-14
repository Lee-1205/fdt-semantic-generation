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

from fdt_rlm.data.filters import normalize_web_text  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402
from prepare_balanced_novel_pilot import SOURCE_REVISIONS as NATURAL_REVISIONS  # noqa: E402
from prepare_balanced_novel_pilot import stream_hf_texts  # noqa: E402
from prepare_capability_completion_v15 import (  # noqa: E402
    SOURCE_REVISIONS,
    active_rows,
    file_sha256,
    random_code,
    reference_hashes,
    require_c_path,
    token_hash,
)
from prepare_capability_completion_v16 import CATEGORY_NAMES, CATEGORY_TO_ID  # noqa: E402
from astra_data_contract import retrieval_contract, python_contract, json_contract  # noqa: E402
from prepare_capability_completion_v18 import factual_length_bucket, repeated_ngram_rate  # noqa: E402
from prepare_capability_completion_v20 import (  # noqa: E402
    V20Writer,
    add_json_grounded,
    add_python_grounded,
    category_token_report,
)


EXPERIMENT_NONCE = "v21-knowledge-scale-t1-20260820-4e77c85d"
HOTPOT_DATASET = "hotpotqa/hotpot_qa"
HOTPOT_CONFIG = "distractor"
HOTPOT_REVISION = "1908d6afbbead072334abe2965f91bd2709910ab"
HOTPOT_LICENSE = "cc-by-sa-4.0"
ROWS_PER_CATEGORY = 20_000
CATEGORY_COUNTS = {name: ROWS_PER_CATEGORY for name in CATEGORY_NAMES}
CATEGORY_WEIGHTS = {
    "factual_qa": 0.25,
    "retrieval": 0.15,
    "python_code": 0.02,
    "json": 0.03,
    "natural_lm": 0.55,
}
PROMPT_LM_WEIGHT = 0.50
EOS_LOSS_WEIGHT = 2.0
GROUNDING_LOSS_WEIGHT = 2.0
SIGNATURE_LOSS_WEIGHT = 1.0
FACTUAL_LENGTH_QUOTAS = {"short_1_4": 14_000, "medium_5_7": 4_000, "long_8_plus": 2_000}
NATURAL_SOURCE_QUOTAS = {"wikipedia": 9_000, "fineweb_edu_dedup": 9_000, "tinystories": 2_000}
NATURAL_ANSWER_TOKENS = 192
MAX_NATURAL_REPEATED_4GRAM_RATE = 0.025


class V21Writer(V20Writer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.natural_terminal_policies = Counter()

    def marker(self, category: str) -> str:
        # Keep the actual language stream clean; the nonce belongs in provenance, not model text.
        self.serial += 1
        return ""

    def add_grounded(self, prompt, answer, category, max_answer_tokens,
                     grounding_spans=None, signature_spans=None):
        # Check the full task before delegating to the historical writer. These
        # guards guarantee its legacy slicing operations cannot remove content.
        answer_ids = self.tokenizer.encode(" " + answer.strip(), add_special_tokens=False)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        reason = None
        if len(answer_ids) > max_answer_tokens:
            reason = "complete_answer_exceeds_cap"
        elif len(prompt_ids) + len(answer_ids) + 1 > self.max_length:
            reason = "complete_task_exceeds_context"
        elif category == "retrieval" and retrieval_contract(prompt, answer) != "PASS":
            reason = "retrieval_source_target_contract"
        elif category == "python_code" and python_contract(answer) != "PASS_SYNTAX_ONLY":
            reason = "invalid_python3_target"
        elif category == "json" and json_contract(answer) != "PASS_SYNTAX_ONLY":
            reason = "invalid_json_target"
        if reason is not None:
            self.rejections[f"{category}_{reason}"] += 1
            return False
        return super().add_grounded(prompt, answer, category, max_answer_tokens,
                                    grounding_spans, signature_spans)

    def add_ids(self, prompt_ids, answer_ids, category, *, terminal_eos=False):
        if category != "natural_lm":
            return super().add_ids(prompt_ids, answer_ids, category)
        prompt, answer = list(prompt_ids), list(answer_ids)
        if terminal_eos:
            answer.append(int(self.tokenizer.eos_token_id))
        active = prompt + answer
        if not prompt or not answer or len(active) > self.max_length:
            self.rejections["natural_lm_complete_span_exceeds_context"] += 1
            return False
        digest = token_hash(active)
        legacy_digest = token_hash(active + [int(self.tokenizer.eos_token_id)]) if not terminal_eos else digest
        if digest in self.reference or legacy_digest in self.reference:
            self.rejections["natural_lm_prior_exact_or_legacy_eos"] += 1
            return False
        if digest in self.seen:
            self.rejections["natural_lm_local_exact"] += 1
            return False
        self.seen.add(digest)
        padding = self.max_length - len(active)
        self.ids.append(active + [int(self.tokenizer.pad_token_id)] * padding)
        self.labels.append([-100] * len(prompt) + answer + [-100] * padding)
        self.masks.append([1] * len(active) + [0] * padding)
        self.grounding_masks.append([0] * self.max_length)
        self.signature_masks.append([0] * self.max_length)
        self.categories[category] += 1
        self.row_categories.append(CATEGORY_TO_ID[category])
        self.natural_terminal_policies["document_end" if terminal_eos else "nonterminal_crop"] += 1
        return True

    def save(self):
        report = super().save()
        report["semantic_contract_version"] = "astra_full_task_no_silent_truncation_v1"
        report["natural_terminal_policies"] = dict(self.natural_terminal_policies)
        return report


def add_factual_knowledge(writer: V21Writer, rows: int, seed: int) -> dict[str, int]:
    dataset = load_dataset(
        HOTPOT_DATASET,
        HOTPOT_CONFIG,
        split="train",
        streaming=True,
        revision=HOTPOT_REVISION,
    ).shuffle(seed=seed, buffer_size=40_000)
    accepted = Counter()
    for row in dataset:
        answer = str(row.get("answer", "")).strip()
        question = str(row.get("question", "")).strip()
        if not answer or not question or answer.lower() in {"yes", "no"}:
            continue
        token_count = len(writer.tokenizer.encode(" " + answer, add_special_tokens=False))
        bucket = factual_length_bucket(token_count)
        if accepted[bucket] >= FACTUAL_LENGTH_QUOTAS[bucket]:
            continue
        prompt = f"Question: {question}\nGive the shortest correct factual answer.\nAnswer:"
        if writer.add_grounded(
            prompt,
            answer,
            "factual_qa",
            max_answer_tokens=48,
            grounding_spans=[(0, len(answer))],
        ):
            accepted[bucket] += 1
        if sum(accepted.values()) >= rows:
            break
    if dict(accepted) != FACTUAL_LENGTH_QUOTAS:
        raise RuntimeError(f"could not fill V21 factual quotas: {dict(accepted)}")
    return dict(accepted)


def add_varied_retrieval(writer: V21Writer, rows: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    accepted = Counter()
    lengths = (8, 16, 32, 64)
    distractor_counts = (4, 8, 16, 32)
    positions = ("front", "middle", "back")
    attempts = 0
    while writer.categories["retrieval"] < rows:
        if attempts >= max(1000, rows * 100):
            raise RuntimeError("retrieval generation cannot fill rows under complete-task context limits")
        index = attempts
        attempts += 1
        length = lengths[index % len(lengths)]
        count = distractor_counts[(index // len(lengths)) % len(distractor_counts)]
        position_name = positions[(index // (len(lengths) * len(distractor_counts))) % len(positions)]
        target_index = 0 if position_name == "front" else count // 2 if position_name == "middle" else count - 1
        target = random_code(rng)
        while len(target) < length:
            target += random_code(rng)
        target = target[:length]
        values = [random_code(rng) * max(1, (length + 6) // 7) for _ in range(count)]
        values = [value[:length] for value in values]
        values[target_index] = target
        records = "\n".join(f"K{slot:02d}={value}" for slot, value in enumerate(values))
        prompt = f"Records:\n{records}\nCopy only the complete value stored at K{target_index:02d}.\nAnswer:"
        if writer.add_grounded(
            prompt,
            target,
            "retrieval",
            max_answer_tokens=96,
            grounding_spans=[(0, len(target))],
        ):
            accepted[f"length_{length}|d{count}|{position_name}"] += 1
    return dict(accepted)


def add_natural_language(writer: V21Writer, rows: int, seed: int) -> dict[str, int]:
    if rows != sum(NATURAL_SOURCE_QUOTAS.values()):
        raise ValueError("V21 natural row count does not match source quotas")
    rng = random.Random(seed)
    streams = {
        name: stream_hf_texts(name, seed + offset + 1)
        for offset, name in enumerate(NATURAL_SOURCE_QUOTAS)
    }
    accepted = Counter()
    source_cursor = 0
    names = tuple(NATURAL_SOURCE_QUOTAS)
    for _ in range(400_000):
        if writer.categories["natural_lm"] >= rows:
            break
        eligible = [name for name in names if accepted[name] < NATURAL_SOURCE_QUOTAS[name]]
        source = eligible[source_cursor % len(eligible)]
        source_cursor += 1
        text, _ = next(streams[source])
        text = normalize_web_text(text)
        ids = list(writer.tokenizer.encode(text, add_special_tokens=False)) if text and "\x00" not in text else []
        if len(ids) < 560:
            writer.rejections[f"natural_lm_{source}_short"] += 1
            continue
        prompt_tokens = min(300, len(ids) - NATURAL_ANSWER_TOKENS)
        span = prompt_tokens + NATURAL_ANSWER_TOKENS
        offset = rng.randrange(0, len(ids) - span + 1)
        active = ids[offset : offset + span]
        completion = active[-NATURAL_ANSWER_TOKENS:]
        if repeated_ngram_rate(completion, 4) > MAX_NATURAL_REPEATED_4GRAM_RATE:
            writer.rejections[f"natural_lm_{source}_repetitive"] += 1
            continue
        if writer.add_ids(active[:-NATURAL_ANSWER_TOKENS], completion, "natural_lm",
                          terminal_eos=offset + span == len(ids)):
            accepted[source] += 1
    if writer.categories["natural_lm"] != rows:
        raise RuntimeError(f"could not fill V21 natural rows: {dict(accepted)}")
    return dict(accepted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a knowledge-first V21 large-scale tranche.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=ROOT / "prepared_data")
    parser.add_argument("--validation-source", type=Path, default=ROOT / "prepared_data" / "capability_curriculum_5k")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers" / "fdt_v3_bpe_24k")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--stage", default="knowledge_scale_v21_t1")
    parser.add_argument("--experiment-nonce", default=EXPERIMENT_NONCE)
    parser.add_argument("--factual-weight", type=float, default=CATEGORY_WEIGHTS["factual_qa"])
    parser.add_argument("--retrieval-weight", type=float, default=CATEGORY_WEIGHTS["retrieval"])
    parser.add_argument("--python-weight", type=float, default=CATEGORY_WEIGHTS["python_code"])
    parser.add_argument("--json-weight", type=float, default=CATEGORY_WEIGHTS["json"])
    parser.add_argument("--natural-weight", type=float, default=CATEGORY_WEIGHTS["natural_lm"])
    parser.add_argument("--prompt-lm-weight", type=float, default=PROMPT_LM_WEIGHT)
    parser.add_argument("--grounding-loss-weight", type=float, default=GROUNDING_LOSS_WEIGHT)
    parser.add_argument("--signature-loss-weight", type=float, default=SIGNATURE_LOSS_WEIGHT)
    parser.add_argument("--max-capability-loss-regression", type=float, default=0.0)
    args = parser.parse_args()

    category_weights = {
        "factual_qa": args.factual_weight,
        "retrieval": args.retrieval_weight,
        "python_code": args.python_weight,
        "json": args.json_weight,
        "natural_lm": args.natural_weight,
    }
    if any(value <= 0 for value in category_weights.values()):
        raise ValueError("all V21 category weights must be positive")
    if abs(sum(category_weights.values()) - 1.0) > 1e-9:
        raise ValueError("V21 category weights must sum to 1.0")
    if args.max_capability_loss_regression < 0:
        raise ValueError("max capability loss regression must be nonnegative")

    output = require_c_path(args.output_dir)
    reference_root = require_c_path(args.reference_root)
    validation_source = require_c_path(args.validation_source)
    tokenizer_path = require_c_path(args.tokenizer)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(str(tokenizer_path))
    references, reference_report = reference_hashes(reference_root, output, int(tokenizer.eos_token_id))
    writer = V21Writer(output, tokenizer, args.max_length, references)

    factual = add_factual_knowledge(writer, CATEGORY_COUNTS["factual_qa"], args.seed + 1)
    retrieval = add_varied_retrieval(writer, CATEGORY_COUNTS["retrieval"], args.seed + 2)
    python = add_python_grounded(writer, CATEGORY_COUNTS["python_code"], args.seed + 3)
    json_families = add_json_grounded(writer, CATEGORY_COUNTS["json"], args.seed + 4)
    natural = add_natural_language(writer, CATEGORY_COUNTS["natural_lm"], args.seed + 5)
    token_report = category_token_report(writer)
    for category, weight in category_weights.items():
        token_report[category]["objective_category_weight"] = weight
    train = writer.save()

    validation_dir = output / "shards" / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    source_validation = validation_source / "shards" / "validation" / "shard_000000.pt"
    target_validation = validation_dir / "shard_000000.pt"
    shutil.copy2(source_validation, target_validation)
    validation_manifest = json.loads((validation_source / "manifest.json").read_text(encoding="utf-8"))
    post_rows = list(active_rows(output))
    if len({token_hash(row) for row in post_rows}) != len(post_rows):
        raise RuntimeError("V21 local exact-row collision")
    overlap = sum(token_hash(row) in references for row in post_rows)
    if overlap:
        raise RuntimeError(f"V21 post-build exact overlap is nonzero: {overlap}")

    report = {
        "stage": args.stage,
        "format": "knowledge_first_large_scale_completion_v1",
        "purpose": "Natural-language and factual-knowledge first scale tranche with explicit long-copy repair",
        "scale_plan": {
            "target_additional_tokens": 2_000_000_000,
            "first_tranche_rows": sum(CATEGORY_COUNTS.values()),
            "next_tranche_requires_audited_gates": True,
            "note": "This first tranche is an evidence gate; it is not claimed to be sufficient final pretraining.",
        },
        "conversational_sft": False,
        "completion_span_supervision": True,
        "supervised_eos": True,
        "eos_loss_weight": EOS_LOSS_WEIGHT,
        "grounding_loss_weight": args.grounding_loss_weight,
        "signature_loss_weight": args.signature_loss_weight,
        "category_normalized_per_example_loss": True,
        "category_objective_weights": category_weights,
        "priority_statement": "Natural language and knowledge are primary; code is auxiliary retention only.",
        "prompt_lm_weight": args.prompt_lm_weight,
        "experiment_nonce": args.experiment_nonce,
        "nonce_inserted_into_model_text": False,
        "historical_payload_limitation": "Some consumed payloads were pruned after manifest/hash journaling; all remaining payloads were scanned, while pruned rows cannot be rescanned.",
        "tokenizer": str(tokenizer_path),
        "max_length": args.max_length,
        "seed": args.seed,
        "source_revisions": {
            "hotpot_qa": HOTPOT_REVISION,
            "code_x_glue_python": SOURCE_REVISIONS["code_x_glue_python"],
            **{name: NATURAL_REVISIONS[name] for name in natural},
        },
        "source_provenance": {
            "factual_qa": {"dataset": HOTPOT_DATASET, "config": HOTPOT_CONFIG, "revision": HOTPOT_REVISION, "license": HOTPOT_LICENSE, "construction": "open-book evidence omitted to train stored factual recall"},
            "natural_lm": {name: NATURAL_REVISIONS[name] for name in natural},
            "python_code": {"dataset": "google/code_x_glue_ct_code_to_text", "config": "python", "revision": SOURCE_REVISIONS["code_x_glue_python"], "role": f"{100 * category_weights['python_code']:.3g}% auxiliary retention objective"},
        },
        "smoke_gate": {
            "max_capability_loss_regression": args.max_capability_loss_regression,
            "interpretation": "Development-only bounded tolerance; unchanged promotion gates still apply to audited final evaluation.",
        },
        "train": train,
        "category_token_report": token_report,
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
        "active_tokens": train["active_tokens"],
        "categories": train["categories"],
        "category_token_report": token_report,
        "post_build_exact_overlap": overlap,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
