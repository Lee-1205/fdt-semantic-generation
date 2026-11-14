from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import evaluate_base_model_strict as strict
from fdt_rlm.lexical_pointer import LexicalPointerDecodeState, SparseLexicalPointer
from fdt_rlm.next_tools import load_model_and_tokenizer, safe_torch_load


MODE_SETTINGS = {
    "smoke": {"codes_per_style": 4, "general_rows_per_bucket": 1},
    "standard": {"codes_per_style": 24, "general_rows_per_bucket": 10},
    "full": {"codes_per_style": 64, "general_rows_per_bucket": 20},
}
RETRIEVAL_STYLES = ("continuation", "distance_64", "distance_512", "distance_1400")
FILLER_TEXT = (
    "This neutral archive note describes ordinary storage, indexing, review, and maintenance. "
    "It contains no access code and introduces no special instruction. "
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def require_c_path(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"{label} must be on C:, got {resolved}")
    return resolved


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def proportion(successes: int, total: int) -> dict:
    return {
        "successes": successes,
        "count": total,
        "rate": successes / total if total else 0.0,
        "wilson_95": wilson_interval(successes, total),
    }


def normalize_code(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def is_boundary(piece: str) -> bool:
    return bool(re.search(r"[.!?;:]\s*$", piece) or "\n" in piece)


def repetition_rate(token_ids: list[int], width: int = 3) -> float:
    if len(token_ids) < width:
        return 0.0
    grams = [tuple(token_ids[index : index + width]) for index in range(len(token_ids) - width + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def random_code(rng: random.Random) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(rng.choice(alphabet) for _ in range(8))


def repeated_filler_ids(tokenizer, count: int) -> list[int]:
    if count <= 0:
        return []
    seed = list(tokenizer.encode(FILLER_TEXT, add_special_tokens=False))
    if not seed:
        raise ValueError("Tokenizer produced no filler tokens")
    return (seed * (count // len(seed) + 1))[:count]


def retrieval_prompt_ids(tokenizer, code: str, style: str) -> list[int]:
    if style == "continuation":
        text = f"Record: verification code is {code}. Therefore the verification code is"
        return list(tokenizer.encode(text, add_special_tokens=False))
    distance = int(style.rsplit("_", 1)[1])
    header = list(
        tokenizer.encode(
            f"Archive entry: the verification code is {code}.\n",
            add_special_tokens=False,
        )
    )
    question = list(
        tokenizer.encode(
            "\nQuestion: What is the verification code?\nAnswer:",
            add_special_tokens=False,
        )
    )
    return header + repeated_filler_ids(tokenizer, distance) + question


@torch.inference_mode()
def generate_ids(
    model,
    pointer,
    tokenizer,
    source_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    min_gate: float,
    activation_steps: int,
    use_pointer: bool,
) -> dict:
    generated = torch.tensor([source_ids], dtype=torch.long, device=device)
    state = LexicalPointerDecodeState(
        source_length=len(source_ids),
        max_activation_steps=activation_steps,
    )
    new_ids: list[int] = []
    trace = []
    for _ in range(max_new_tokens):
        mask = torch.ones_like(generated)
        output = model(generated, attention_mask=mask)
        base_logits = output["logits"][:, -1].float()
        if use_pointer:
            logits, diagnostics = state.prepare_logits(
                pointer,
                base_logits,
                output["hidden"],
                generated,
                mask,
                min_gate=min_gate,
            )
        else:
            logits = base_logits
            diagnostics = {
                "mode": "base",
                "gate": 0.0,
                "mix_gate": 0.0,
                "source_positions": [],
                "candidate_ids": [],
                "search_span": 0,
            }
        selected_id = int(logits.argmax(dim=-1))
        piece = tokenizer.decode([selected_id], skip_special_tokens=True)
        if use_pointer:
            state.commit(selected_id, diagnostics, boundary=is_boundary(piece))
        if len(trace) < 12:
            trace.append(
                {
                    "mode": diagnostics.get("mode", "base"),
                    "gate": float(diagnostics.get("gate", 0.0)),
                    "mix_gate": float(diagnostics.get("mix_gate", 0.0)),
                    "search_span": int(diagnostics.get("search_span", 0)),
                    "source_positions": diagnostics.get("source_positions", []),
                    "candidate_ids": diagnostics.get("candidate_ids", []),
                    "candidate_occurrences": diagnostics.get("candidate_occurrences", []),
                    "selected_id": selected_id,
                }
            )
        new_ids.append(selected_id)
        generated = torch.cat((generated, torch.tensor([[selected_id]], device=device)), dim=1)
        if selected_id == tokenizer.eos_token_id:
            break
    return {
        "completion": tokenizer.decode(new_ids, skip_special_tokens=True),
        "token_ids": new_ids,
        "repetition_3gram": repetition_rate(new_ids),
        "pointer_steps": sum(float(row.get("mix_gate", 0.0)) > 0.0 for row in trace),
        "cursor_steps": sum(row["mode"] == "cursor" for row in trace),
        "trace": trace,
    }


@torch.inference_mode()
def compare_pointer_to_base(
    model,
    pointer,
    tokenizer,
    source_ids: list[int],
    max_new_tokens: int,
    device: torch.device,
    min_gate: float,
    activation_steps: int,
) -> dict:
    generated = torch.tensor([source_ids], dtype=torch.long, device=device)
    state = LexicalPointerDecodeState(
        source_length=len(source_ids),
        max_activation_steps=activation_steps,
    )
    base_ids: list[int] = []
    trace = []
    identical = True
    first_difference = None
    for step in range(max_new_tokens):
        mask = torch.ones_like(generated)
        output = model(generated, attention_mask=mask)
        base_logits = output["logits"][:, -1].float()
        pointer_logits, diagnostics = state.prepare_logits(
            pointer,
            base_logits,
            output["hidden"],
            generated,
            mask,
            min_gate=min_gate,
        )
        base_id = int(base_logits.argmax(dim=-1))
        pointer_id = int(pointer_logits.argmax(dim=-1))
        trace.append(
            {
                "step": step,
                "mode": diagnostics.get("mode", "base"),
                "gate": float(diagnostics.get("gate", 0.0)),
                "mix_gate": float(diagnostics.get("mix_gate", 0.0)),
                "search_span": int(diagnostics.get("search_span", 0)),
                "base_id": base_id,
                "pointer_id": pointer_id,
                "candidate_occurrences": diagnostics.get("candidate_occurrences", []),
            }
        )
        if pointer_id != base_id:
            identical = False
            first_difference = trace[-1]
            break
        piece = tokenizer.decode([base_id], skip_special_tokens=True)
        state.commit(base_id, diagnostics, boundary=is_boundary(piece))
        base_ids.append(base_id)
        generated = torch.cat((generated, torch.tensor([[base_id]], device=device)), dim=1)
        if base_id == tokenizer.eos_token_id:
            break
    return {
        "identical_decisions": identical,
        "checked_tokens": len(trace),
        "base_completion": tokenizer.decode(base_ids, skip_special_tokens=True),
        "pointer_activated": any(row["mix_gate"] > 0.0 for row in trace),
        "first_difference": first_difference,
        "trace": trace,
    }


def summarize_retrieval(rows: list[dict], key: str) -> dict:
    successes = sum(row[key] for row in rows)
    return proportion(successes, len(rows))


@torch.inference_mode()
def evaluate_retrieval(model, pointer, tokenizer, device, settings, seed, min_gate, activation_steps):
    rng = random.Random(seed)
    rows = []
    for style in RETRIEVAL_STYLES:
        for _ in range(settings["codes_per_style"]):
            code = random_code(rng)
            source_ids = retrieval_prompt_ids(tokenizer, code, style)
            base = generate_ids(
                model, pointer, tokenizer, source_ids, 12, device, min_gate, activation_steps, False
            )
            adapted = generate_ids(
                model, pointer, tokenizer, source_ids, 12, device, min_gate, activation_steps, True
            )
            rows.append(
                {
                    "style": style,
                    "code": code,
                    "source_tokens": len(source_ids),
                    "base_correct": normalize_code(base["completion"]).startswith(code),
                    "pointer_correct": normalize_code(adapted["completion"]).startswith(code),
                    "base": base,
                    "pointer": adapted,
                }
            )
        print(json.dumps({"event": "retrieval", "style": style, "rows": settings["codes_per_style"]}), flush=True)
    by_style = {}
    for style in RETRIEVAL_STYLES:
        subset = [row for row in rows if row["style"] == style]
        by_style[style] = {
            "base_exact": summarize_retrieval(subset, "base_correct"),
            "pointer_exact": summarize_retrieval(subset, "pointer_correct"),
        }
    return {
        "base_exact": summarize_retrieval(rows, "base_correct"),
        "pointer_exact": summarize_retrieval(rows, "pointer_correct"),
        "by_style": by_style,
        "rows": rows,
    }


@torch.inference_mode()
def evaluate_general_identity(
    model, pointer, tokenizer, dataset_dir, device, settings, min_gate, activation_steps
):
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    ids, masks = strict.split_rows(dataset_dir, "validation")
    ranges = strict.bucket_ranges(manifest, "validation")
    rows = []
    for bucket, start, end in ranges:
        for row_index in strict.evenly_spaced(start, end, settings["general_rows_per_bucket"]):
            active = int(masks[row_index].sum())
            source_length = min(256, max(active - 64, 32))
            source_ids = ids[row_index, :source_length].tolist()
            comparison = compare_pointer_to_base(
                model,
                pointer,
                tokenizer,
                source_ids,
                32,
                device,
                min_gate,
                activation_steps,
            )
            rows.append(
                {
                    "bucket": bucket,
                    "row_id": row_index,
                    "source_tokens": source_length,
                    "identical_to_base": comparison["identical_decisions"],
                    "pointer_activated": comparison["pointer_activated"],
                    "comparison": comparison,
                }
            )
        print(json.dumps({"event": "general_identity", "bucket": bucket}), flush=True)
    return {
        "identical_to_base": proportion(sum(row["identical_to_base"] for row in rows), len(rows)),
        "pointer_activated": proportion(sum(row["pointer_activated"] for row in rows), len(rows)),
        "rows": rows,
    }


@torch.inference_mode()
def benchmark_long_context(model, pointer, tokenizer, config, device, length: int, min_gate: float):
    ids = repeated_filler_ids(tokenizer, length)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    mask = torch.ones_like(input_ids)
    warm_length = min(length, 256)
    model(input_ids[:, :warm_length], attention_mask=mask[:, :warm_length])
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    output = model(input_ids, attention_mask=mask)
    if device.type == "cuda":
        torch.cuda.synchronize()
    prefill_seconds = time.perf_counter() - started
    state = LexicalPointerDecodeState(source_length=length)
    started = time.perf_counter()
    _, diagnostics = state.prepare_logits(
        pointer,
        output["logits"][:, -1].float(),
        output["hidden"],
        input_ids,
        mask,
        min_gate=min_gate,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    pointer_seconds = time.perf_counter() - started
    return {
        "context_tokens": length,
        "prefill_tokens_per_second": length / max(prefill_seconds, 1e-9),
        "pointer_step_ms": pointer_seconds * 1000.0,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20 if device.type == "cuda" else None,
        "search_span": diagnostics.get("search_span", 0),
        "complexity": "O(N(K+W)) base with O(W) pointer activation and O(1) cursor continuation",
    }


def atomic_json_write(path: Path, payload: dict):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Strict lexical pointer evaluation with long-distance retrieval")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pointer-checkpoint", type=Path)
    source.add_argument("--integrated-checkpoint", type=Path)
    parser.add_argument("--pointer-override", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "tokenizers" / "fdt_v3_bpe_24k")
    parser.add_argument("--general-dataset-dir", type=Path, default=ROOT / "prepared_data" / "v3_broad_additional_300m_combined_v2_24k_ctx2048")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=tuple(MODE_SETTINGS), default="standard")
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--min-gate", type=float, default=0.8)
    parser.add_argument("--activation-steps", type=int, default=4)
    parser.add_argument("--long-context", type=int, default=1792)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dataset_dir = require_c_path(args.general_dataset_dir, "general dataset")
    output_path = require_c_path(args.output, "output")
    device = torch.device(args.device)
    if args.integrated_checkpoint:
        pointer_path = require_c_path(args.integrated_checkpoint, "integrated checkpoint")
        base_path = pointer_path
        actual_base_sha256 = sha256_file(base_path)
        expected_base_sha256 = None
        model, tokenizer, config, _ = load_model_and_tokenizer(
            base_path,
            device,
            tokenizer_name=str(require_c_path(args.tokenizer, "tokenizer")),
        )
        if config.model_type != "fdt_v3_anchor_indexed_dual_memory":
            raise ValueError(f"Unsupported integrated model type: {config.model_type}")
        model = model.to(dtype=torch.float32).eval()
        pointer = model.exact_pointer
        pointer_override = None
        if args.pointer_override:
            override_path = require_c_path(args.pointer_override, "pointer override")
            override = safe_torch_load(override_path)
            incompatible = pointer.load_state_dict(override["pointer_state_dict"], strict=False)
            allowed_missing = {
                "anchor_match_bias",
                "commit_proj.weight",
                "commit_proj.bias",
            }
            if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
                raise ValueError(f"Incompatible pointer override: {incompatible}")
            pointer_override = {
                "path": str(override_path),
                "sha256": sha256_file(override_path),
                "missing_keys": incompatible.missing_keys,
            }
        integrated = True
    else:
        pointer_path = require_c_path(args.pointer_checkpoint, "pointer checkpoint")
        checkpoint = safe_torch_load(pointer_path)
        base_path = require_c_path(Path(checkpoint["base_checkpoint"]), "base checkpoint")
        actual_base_sha256 = sha256_file(base_path)
        expected_base_sha256 = checkpoint.get("base_checkpoint_sha256")
        if expected_base_sha256 and expected_base_sha256.upper() != actual_base_sha256:
            raise ValueError("Base checkpoint SHA-256 does not match pointer metadata")
        model, tokenizer, config, _ = load_model_and_tokenizer(base_path, device, "fdt_v3")
        model = model.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
        pointer = SparseLexicalPointer(**checkpoint["pointer_config"]).to(device)
        pointer.load_state_dict(checkpoint["pointer_state_dict"], strict=True)
        pointer.eval()
        pointer_override = None
        integrated = False

    settings = MODE_SETTINGS[args.mode]
    started = time.perf_counter()
    retrieval = evaluate_retrieval(
        model, pointer, tokenizer, device, settings, args.seed, args.min_gate, args.activation_steps
    )
    general = evaluate_general_identity(
        model,
        pointer,
        tokenizer,
        dataset_dir,
        device,
        settings,
        args.min_gate,
        args.activation_steps,
    )
    long_context = benchmark_long_context(
        model,
        pointer,
        tokenizer,
        config,
        device,
        min(args.long_context, config.max_seq_len),
        args.min_gate,
    )
    style_rates = {
        style: retrieval["by_style"][style]["pointer_exact"]["rate"]
        for style in RETRIEVAL_STYLES
    }
    promotion = {
        "overall_retrieval_at_least_90pct": retrieval["pointer_exact"]["rate"] >= 0.90,
        "each_style_at_least_75pct": min(style_rates.values()) >= 0.75,
        "general_identity_at_least_98pct": general["identical_to_base"]["rate"] >= 0.98,
        "full_1792_search": long_context["search_span"] >= min(args.long_context, config.max_seq_len) - 1,
    }
    promotion["pass"] = all(promotion.values())
    report = {
        "status": "PASS",
        "evaluator": "lexical_pointer_strict_v2",
        "integrated_checkpoint": integrated,
        "pointer_override": pointer_override,
        "mode": args.mode,
        "elapsed_seconds": time.perf_counter() - started,
        "pointer_checkpoint": str(pointer_path),
        "pointer_checkpoint_sha256": sha256_file(pointer_path),
        "base_checkpoint": str(base_path),
        "base_checkpoint_sha256": actual_base_sha256,
        "base_sha256_verified": expected_base_sha256 is None or expected_base_sha256.upper() == actual_base_sha256,
        "settings": settings,
        "retrieval": retrieval,
        "general_identity": general,
        "long_context": long_context,
        "promotion": promotion,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json_write(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output_path),
                "retrieval": retrieval["pointer_exact"],
                "retrieval_by_style": style_rates,
                "base_retrieval": retrieval["base_exact"],
                "general_identity": general["identical_to_base"],
                "long_context": long_context,
                "promotion": promotion,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
