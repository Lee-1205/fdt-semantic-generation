from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fdt_rlm.next_tools import NEXT_MODEL_TYPES, load_model_and_tokenizer, safe_torch_load


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming validation/source loss for a next-model checkpoint.")
    parser.add_argument("--model_type", choices=NEXT_MODEL_TYPES, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_dir", default=str(ROOT / "prepared_data" / "fdt_rlm_100m"))
    parser.add_argument("--tokenizer_name", default="")
    parser.add_argument("--eval_set", choices=["validation", "source"], default="validation")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "next_validation.json"))
    args = parser.parse_args()
    device = torch.device(args.device)
    model, _, config, _ = load_model_and_tokenizer(args.checkpoint, device, args.model_type, args.tokenizer_name)
    totals = defaultdict(lambda: [0.0, 0])
    batches = 0
    shard_paths = sorted((Path(args.dataset_dir) / "shards" / "validation").glob("*.pt"))
    if not shard_paths:
        raise FileNotFoundError("No validation shards found")
    for shard_path in shard_paths:
        shard = safe_torch_load(shard_path)
        all_ids = shard["input_ids"].long()
        all_mask = shard.get("attention_mask", torch.ones_like(all_ids)).long()
        for start in range(0, len(all_ids), args.batch_size):
            ids = all_ids[start : start + args.batch_size].to(device)
            mask = all_mask[start : start + args.batch_size].to(device)
            with torch.inference_mode():
                logits = model(ids, attention_mask=mask)["logits"][:, :-1].float()
            labels = ids[:, 1:]
            valid = mask[:, 1:].bool() & (labels != config.pad_token_id)
            losses = F.cross_entropy(logits.transpose(1, 2), labels, reduction="none")
            totals["overall"][0] += float(losses[valid].sum())
            totals["overall"][1] += int(valid.sum())
            batches += 1
            if args.max_batches and batches >= args.max_batches:
                break
        if args.max_batches and batches >= args.max_batches:
            break
    result = {name: {"loss": total / max(tokens, 1), "perplexity": math.exp(total / max(tokens, 1)), "tokens": tokens} for name, (total, tokens) in totals.items()}
    if args.eval_set == "source":
        result["note"] = "Prepared tensor shards do not retain per-row source IDs; this run reports the source-mixed validation loss."
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "batches": batches, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
