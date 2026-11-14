from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.models import build_model  # noqa: E402
from fdt_rlm.next_tools import generate_ids  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def first_divergence(left: list[Any] | str, right: list[Any] | str) -> int | None:
    for index, (left_value, right_value) in enumerate(zip(left, right)):
        if left_value != right_value:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def fidelity_metrics(completion: str, expected: str, tokenizer: Any) -> dict[str, Any] | None:
    if not expected:
        return None
    observed = completion.strip()
    target = expected.strip()
    common_characters = sum(left == right for left, right in zip(observed, target))
    observed_ids = list(tokenizer.encode(observed, add_special_tokens=False))
    target_ids = list(tokenizer.encode(target, add_special_tokens=False))
    common_tokens = sum(left == right for left, right in zip(observed_ids, target_ids))
    return {
        "expected": expected,
        "exact_match": observed == target,
        "exact_prefix": observed.startswith(target),
        "character_accuracy": common_characters / max(len(observed), len(target), 1),
        "token_accuracy": common_tokens / max(len(observed_ids), len(target_ids), 1),
        "first_character_divergence": first_divergence(observed, target),
        "first_token_divergence": first_divergence(observed_ids, target_ids),
        "observed_characters": len(observed),
        "expected_characters": len(target),
        "observed_tokens": len(observed_ids),
        "expected_tokens": len(target_ids),
    }


@torch.inference_mode()
def test_checkpoint(path: Path, request: dict[str, Any], device: torch.device, index: int) -> dict[str, Any]:
    total_started = time.perf_counter()
    hash_started = time.perf_counter()
    checkpoint_hash = sha256(path)
    hash_seconds = time.perf_counter() - hash_started
    load_started = time.perf_counter()
    checkpoint = load_checkpoint(path)
    config = ModelConfig(**checkpoint["model_config"])
    train_config = checkpoint.get("train_config", {})
    tokenizer_name = train_config.get("tokenizer_name") or "gpt2"
    tokenizer = load_tokenizer(tokenizer_name)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metadata = {
        "stage": train_config.get("stage"),
        "stage_status": checkpoint.get("stage_status"),
        "optimizer_step": int(checkpoint.get("optimizer_step", checkpoint.get("global_step", 0))),
        "tokens_seen": int(checkpoint.get("tokens_seen", 0)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "tokenizer": str(tokenizer_name),
    }
    del checkpoint
    gc.collect()
    # Official checkpoint testing preserves the original FP32 parameter values.
    precision = "fp32"
    model.to(device=device, dtype=torch.float32).eval()
    synchronize(device)
    load_seconds = time.perf_counter() - load_started

    prompt = request["prompt"]
    max_new_tokens = int(request["max_new_tokens"])
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    prompt_ids = prompt_ids[-max(config.max_seq_len - max_new_tokens, 1) :]
    if not prompt_ids:
        raise ValueError("prompt tokenization is empty")
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    torch.manual_seed(int(request["seed"]) + index)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(request["seed"]) + index)
    synchronize(device)
    generation_started = time.perf_counter()
    generated = generate_ids(
        model,
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=float(request["temperature"]),
        top_p=float(request["top_p"]),
        top_k=int(request["top_k"]),
        repetition_penalty=float(request["repetition_penalty"]),
        greedy=request["mode"] == "greedy",
        use_cache=True,
    )
    synchronize(device)
    generation_seconds = time.perf_counter() - generation_started
    new_ids = generated[0, len(prompt_ids) :].tolist()
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    fidelity = fidelity_metrics(completion, str(request.get("expected", "")), tokenizer)
    total_seconds = time.perf_counter() - total_started
    result = {
        "checkpoint": str(path),
        "checkpoint_sha256": checkpoint_hash,
        "metadata": metadata,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(new_ids),
        "completion": completion,
        "elapsed_seconds": generation_seconds,
        "generation_seconds": generation_seconds,
        "total_elapsed_seconds": total_seconds,
        "checkpoint_hash_seconds": hash_seconds,
        "model_load_seconds": load_seconds,
        "precision": precision,
        "generation_tokens_per_second": len(new_ids) / max(generation_seconds, 1e-9),
        "finish_reason": "eos" if new_ids and new_ids[-1] == config.eos_token_id else "length_or_context",
        "fidelity": fidelity,
    }
    del model, tokenizer, input_ids, generated
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare preserved pure-FDT checkpoints on one prompt.")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    request_path = Path(args.request).resolve()
    output_path = Path(args.output).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8-sig"))
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []
    partial = output_path.with_name(output_path.stem + ".partial.json")
    for index, raw_path in enumerate(request["checkpoints"]):
        path = Path(raw_path).resolve()
        try:
            row = test_checkpoint(path, request, device, index)
            row["status"] = "COMPLETE"
        except Exception as error:
            row = {"checkpoint": str(path), "status": "FAILED", "error": str(error)}
            if device.type == "cuda":
                torch.cuda.empty_cache()
        rows.append(row)
        atomic_write_json(partial, {
            "status": "RUNNING",
            "completed": len(rows),
            "total": len(request["checkpoints"]),
            "request": request,
            "results": rows,
        })
        print(json.dumps({"event": "model_test", "completed": len(rows), "total": len(request["checkpoints"])}), flush=True)

    payload = {
        "status": "COMPLETE" if all(row["status"] == "COMPLETE" for row in rows) else "COMPLETE_WITH_ERRORS",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request": request,
        "results": rows,
    }
    atomic_write_json(output_path, payload)
    partial.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
