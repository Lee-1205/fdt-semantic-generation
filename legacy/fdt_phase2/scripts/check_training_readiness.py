from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.data.manifest import read_json
from fdt_rlm.data.sharded_dataset import ShardedTokenDataset
from fdt_rlm.data.streaming_mix import TokenPlan, assert_within_tolerance
from fdt_rlm.models import build_model, causal_lm_loss
from fdt_rlm.tokenization import load_tokenizer


def check_forward_backward(model_type: str, tokenizer, device, dataset_dir: Path):
    cfg = ModelConfig(vocab_size=len(tokenizer), pad_token_id=int(tokenizer.pad_token_id), eos_token_id=int(tokenizer.eos_token_id), model_type=model_type, dim=64, n_layers=1, n_heads=4, max_seq_len=512, num_anchors=16, top_k=4)
    model = build_model(cfg).to(device)
    batch = next(iter(ShardedTokenDataset(dataset_dir, "train")))
    ids = batch["input_ids"].unsqueeze(0).to(device)
    mask = batch["attention_mask"].unsqueeze(0).to(device)
    out = model(ids, attention_mask=mask)
    loss = causal_lm_loss(out["logits"], ids, mask, cfg.pad_token_id)
    loss.backward()
    return torch.isfinite(loss).item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", default=str(ROOT / "prepared_data" / "fdt_rlm_10m"))
    p.add_argument("--tokenizer", default=str(ROOT / "tokenizers" / "fdt_bpe"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    failures = []
    dataset_dir = Path(args.dataset_dir)
    tokenizer_dir = Path(args.tokenizer)
    if not tokenizer_dir.exists():
        failures.append("tokenizer missing")
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        failures.append("manifest missing")
    else:
        manifest = read_json(manifest_path)
        try:
            plan = TokenPlan(total_tokens=int(manifest["requested_tokens"]), quotas={k: int(v) for k, v in manifest["plan"].items()})
            assert_within_tolerance({k: int(v) for k, v in manifest["tokens_by_bucket"].items()}, plan, 0.001)
        except Exception as exc:
            failures.append(f"ratio validation failed: {exc}")
        if not list((dataset_dir / "shards" / "train").glob("*.pt")):
            failures.append("train shards missing")
        if not list((dataset_dir / "shards" / "validation").glob("*.pt")):
            failures.append("validation shards missing")
    try:
        tokenizer = load_tokenizer(str(tokenizer_dir))
        device = torch.device(args.device)
        for model_type in ("transformer", "fdt"):
            if not check_forward_backward(model_type, tokenizer, device, dataset_dir):
                failures.append(f"{model_type} nonfinite loss")
    except Exception as exc:
        failures.append(f"forward/backward failed: {type(exc).__name__}: {str(exc)[:200]}")
    ckpt_dir = ROOT / "runs"
    try:
        ckpt_dir.mkdir(exist_ok=True)
        test = ckpt_dir / ".write_test"
        test.write_text("ok", encoding="utf-8")
        test.unlink()
    except Exception as exc:
        failures.append(f"checkpoint dir not writable: {exc}")
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    else:
        gpu = "CUDA_NOT_AVAILABLE"
    print(f"GPU: {gpu}")
    if failures:
        print("TRAINING READINESS: FAIL")
        for item in failures:
            print("-", item)
        raise SystemExit(1)
    print("TRAINING READINESS: PASS")


if __name__ == "__main__":
    main()

