from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.config import ModelConfig  # noqa: E402
from fdt_rlm.lexical_pointer import LexicalPointerDecodeState  # noqa: E402
from fdt_rlm.models import build_model, causal_lm_loss  # noqa: E402
from fdt_rlm.next_tools import apply_repetition_penalty_  # noqa: E402
from fdt_rlm.tokenization import load_tokenizer  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def generate(model, tokenizer, prompt: str, target_tokens: int, threshold: float, device: torch.device) -> dict:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    state = LexicalPointerDecodeState(
        source_length=len(prompt_ids),
        max_activation_steps=16,
        max_copy_tokens=96,
    )
    memory = None
    new_ids: list[int] = []
    first_diagnostics: dict = {}
    with torch.inference_mode():
        for _ in range(target_tokens):
            mask = torch.ones_like(generated)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(generated, attention_mask=mask)
                if memory is None:
                    memory = model.build_exact_memory(
                        output["hidden"], generated, mask, source_length=len(prompt_ids)
                    )
                logits = output["logits"][:, -1].float()
                apply_repetition_penalty_(logits, generated[:, len(prompt_ids) :], 1.12)
                query_anchors = model.exact_route_indices(output["hidden"])[:, -1]
                mixed, diagnostics = state.prepare_logits(
                    model.exact_pointer,
                    logits,
                    output["hidden"],
                    generated,
                    mask,
                    min_gate=threshold,
                    anchor_memory=memory,
                    query_anchor_ids=query_anchors,
                    max_candidate_chunks=model.config.exact_pointer_candidate_chunks,
                )
            if not first_diagnostics:
                first_diagnostics = diagnostics
            selected = int(mixed.argmax(dim=-1))
            state.commit(selected, diagnostics)
            new_ids.append(selected)
            generated = torch.cat(
                (generated, torch.tensor([[selected]], device=device)), dim=1
            )
    candidate_rows = first_diagnostics.get("candidate_ids") or [[]]
    position_rows = first_diagnostics.get("source_positions") or [[]]
    return {
        "completion": tokenizer.decode(new_ids, skip_special_tokens=True),
        "first_gate": float(first_diagnostics.get("gate", 0.0)),
        "first_mix_gate": float(first_diagnostics.get("mix_gate", 0.0)),
        "first_candidate_ids": candidate_rows[0] if candidate_rows else [],
        "first_source_positions": position_rows[0] if position_rows else [],
        "skipped": bool(first_diagnostics.get("skipped_by_exact_need_gate", False)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused integrated anchor-exact audit")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-rows", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    output_path = args.output.resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = ModelConfig(**payload["model_config"])
    if config.model_type != "fdt_v3_anchor_indexed_dual_memory":
        raise ValueError(f"Unsupported model type: {config.model_type}")
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = torch.device(args.device)
    model.to(device=device, dtype=torch.bfloat16 if device.type == "cuda" else torch.float32).eval()
    tokenizer = load_tokenizer(str(args.tokenizer.resolve()))

    validation = torch.load(args.validation.resolve(), map_location="cpu", weights_only=False, mmap=True)
    validation_rows = []
    with torch.inference_mode():
        for index in range(min(args.validation_rows, validation["input_ids"].size(0))):
            ids = validation["input_ids"][index : index + 1].long().to(device)
            mask = validation["attention_mask"][index : index + 1].long().to(device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                row = model(ids, attention_mask=mask)
                lm = causal_lm_loss(row["logits"], ids, mask, config.pad_token_id)
                pointer = model.exact_memory_loss(row["hidden"], ids, ids, mask)
            validation_rows.append(
                {
                    "lm_loss": float(lm),
                    "total_pointer_loss": float(pointer.loss),
                    "ranking_loss": float(pointer.pointer_loss),
                    "gate_loss": float(pointer.gate_loss),
                    "commit_loss": float(pointer.commit_loss),
                    "copyable_rate": float(pointer.copyable_rate),
                    "pointer_accuracy": float(pointer.pointer_accuracy),
                }
            )

    filler = " Archived reference material remains unchanged for verification."
    cases = []
    for index, repeats in enumerate((1, 8, 24, 48)):
        code = f"ZXQ{7319 + index}KLM{42 + index}"
        target = " " + code
        prompt = (
            f"Record: the unique access value is {code}."
            + filler * repeats
            + "\nQuestion: What is the unique access value?\nAnswer:"
        )
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        forced = generate(model, tokenizer, prompt, len(target_ids) + 2, 0.0, device)
        deployment = generate(model, tokenizer, prompt, len(target_ids) + 2, 0.80, device)
        candidate_ids = forced["first_candidate_ids"]
        cases.append(
            {
                "filler_repeats": repeats,
                "prompt_tokens": len(tokenizer.encode(prompt, add_special_tokens=False)),
                "target": target,
                "target_first_id": target_ids[0],
                "candidate_recall_at_4": target_ids[0] in candidate_ids,
                "forced": forced,
                "forced_exact_prefix": forced["completion"].startswith(target),
                "deployment": deployment,
                "deployment_exact_prefix": deployment["completion"].startswith(target),
            }
        )

    def mean(key: str) -> float:
        return sum(row[key] for row in validation_rows) / max(len(validation_rows), 1)

    result = {
        "status": "COMPLETE",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "model_type": config.model_type,
        "optimizer_step": int(payload.get("optimizer_step", 0)),
        "tokens_seen": int(payload.get("tokens_seen", 0)),
        "validation": {
            "rows": len(validation_rows),
            "mean_lm_loss": mean("lm_loss"),
            "mean_total_pointer_loss": mean("total_pointer_loss"),
            "mean_ranking_loss": mean("ranking_loss"),
            "mean_gate_loss": mean("gate_loss"),
            "mean_commit_loss": mean("commit_loss"),
            "mean_copyable_rate": mean("copyable_rate"),
            "mean_pointer_accuracy": mean("pointer_accuracy"),
            "raw": validation_rows,
        },
        "retrieval": {
            "cases": cases,
            "candidate_recall_at_4": sum(row["candidate_recall_at_4"] for row in cases),
            "forced_exact": sum(row["forced_exact_prefix"] for row in cases),
            "deployment_exact": sum(row["deployment_exact_prefix"] for row in cases),
        },
    }
    atomic_json(output_path, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
