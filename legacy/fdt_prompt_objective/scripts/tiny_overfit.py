from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

import torch

from fdt_rlm.config import ModelConfig, TrainConfig
from fdt_rlm.data.manifest import write_json
from fdt_rlm.data.sharded_dataset import ShardedTokenDataset
from fdt_rlm.models import build_model, causal_lm_loss
from fdt_rlm.tokenization import load_tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["transformer", "fdt"], required=True)
    p.add_argument("--dataset-dir", default=str(ROOT / "prepared_data" / "fdt_rlm_10m"))
    p.add_argument("--tokenizer", default=str(ROOT / "tokenizers" / "fdt_bpe"))
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--output", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    tok = load_tokenizer(args.tokenizer)
    cfg = ModelConfig(vocab_size=len(tok), pad_token_id=int(tok.pad_token_id), eos_token_id=int(tok.eos_token_id), model_type=args.model_type, dim=64, n_layers=1, n_heads=4, max_seq_len=512, num_anchors=16, top_k=4)
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    batch = next(iter(ShardedTokenDataset(args.dataset_dir, "train")))
    ids = batch["input_ids"].unsqueeze(0).to(device)
    mask = batch["attention_mask"].unsqueeze(0).to(device)
    losses = []
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        out = model(ids, attention_mask=mask)
        loss = causal_lm_loss(out["logits"], ids, mask, cfg.pad_token_id)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().item()))
    out_path = Path(args.output or ROOT / "artifacts" / f"tiny_overfit_{args.model_type}.json")
    ckpt = out_path.with_suffix(".pt")
    torch.save({"model_state_dict": model.state_dict(), "model_config": cfg.__dict__}, ckpt)
    reloaded = build_model(cfg).to(device)
    reloaded.load_state_dict(torch.load(ckpt, map_location=device)["model_state_dict"], strict=True)
    report = {
        "model_type": args.model_type,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "checkpoint": str(ckpt),
        "checkpoint_reload": True,
    }
    write_json(out_path, report)
    print(report)
    if not report["loss_decreased"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
