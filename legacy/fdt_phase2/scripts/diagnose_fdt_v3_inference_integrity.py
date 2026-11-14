from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_base_model_strict as strict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_prompt(tokenizer, target_tokens: int, seed: int) -> str:
    if target_tokens <= 64:
        return "Context: Water freezes at zero degrees Celsius. Question: At what temperature does water freeze? Answer:"
    rng = random.Random(seed)
    words = (
        "method result system sample process value record detail signal context analysis model "
        "data memory output input stable precise ordinary measured reference public science history"
    ).split()
    prefix = "Archive entry: the verification code is QV274915. "
    suffix = " For this archive entry, the verification code is"
    filler: list[str] = []
    while True:
        filler.extend(rng.choices(words, k=64))
        text = prefix + " ".join(filler) + suffix
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) >= target_tokens:
            return tokenizer.decode(ids[:target_tokens], skip_special_tokens=True)


@torch.inference_mode()
def compare_prompt(model, tokenizer, prompt: str, device: torch.device, decode_tokens: int) -> dict:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    recompute = model(input_ids, attention_mask=attention_mask)["logits"][:, -1:].float()
    cached, cache = model.prefill(input_ids, attention_mask)
    cached_logits = cached["logits"].float()

    max_abs_errors = [float((cached_logits - recompute).abs().max().item())]
    mean_abs_errors = [float((cached_logits - recompute).abs().mean().item())]
    cached_tokens: list[int] = []
    recompute_tokens: list[int] = []
    generated = input_ids

    for _ in range(decode_tokens):
        cached_next = cached_logits[:, -1].argmax(dim=-1, keepdim=True)
        recompute_next = recompute[:, -1].argmax(dim=-1, keepdim=True)
        cached_tokens.append(int(cached_next.item()))
        recompute_tokens.append(int(recompute_next.item()))
        if not torch.equal(cached_next, recompute_next):
            break
        generated = torch.cat((generated, recompute_next), dim=1)
        cached, cache = model.decode_step(cached_next, cache)
        cached_logits = cached["logits"].float()
        recompute = model(generated, attention_mask=torch.ones_like(generated))["logits"][:, -1:].float()
        max_abs_errors.append(float((cached_logits - recompute).abs().max().item()))
        mean_abs_errors.append(float((cached_logits - recompute).abs().mean().item()))

    return {
        "prompt_tokens": len(prompt_ids),
        "decode_tokens_requested": decode_tokens,
        "decode_tokens_compared": len(cached_tokens),
        "token_identity": cached_tokens == recompute_tokens and len(cached_tokens) == decode_tokens,
        "cached_token_ids": cached_tokens,
        "recompute_token_ids": recompute_tokens,
        "cached_text": tokenizer.decode(cached_tokens, skip_special_tokens=True),
        "recompute_text": tokenizer.decode(recompute_tokens, skip_special_tokens=True),
        "max_abs_logit_error": max(max_abs_errors),
        "mean_abs_logit_error": sum(mean_abs_errors) / len(mean_abs_errors),
        "per_step_max_abs_logit_error": max_abs_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare real FDT v3 cached inference with full recomputation.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--decode-tokens", type=int, default=12)
    parser.add_argument("--atol", type=float, default=3e-4)
    args = parser.parse_args()

    device = torch.device(args.device)
    model, tokenizer, config, metadata = strict.load_checkpoint_model(args.checkpoint, device)
    targets = (48, 512, 1400)
    cases = []
    for index, target in enumerate(targets):
        prompt = build_prompt(tokenizer, target, 20260812 + index)
        case = compare_prompt(model, tokenizer, prompt, device, args.decode_tokens)
        case["target_prompt_tokens"] = target
        cases.append(case)
        print(json.dumps({"event": "case", **case}, ensure_ascii=True), flush=True)

    passed = all(
        case["token_identity"] and case["max_abs_logit_error"] <= args.atol for case in cases
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "diagnostic": "real_checkpoint_fdt_v3_cache_vs_recompute_v1",
        "script_sha256": sha256_file(Path(__file__)),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "metadata": metadata,
        "device": str(device),
        "dtype": str(next(model.parameters()).dtype),
        "tolerance": {"max_abs_logit_error": args.atol, "greedy_token_identity": True},
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": report["status"], "output": str(args.output.resolve())}), flush=True)
    raise SystemExit(0 if passed else 2)


if __name__ == "__main__":
    main()
