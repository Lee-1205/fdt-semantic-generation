import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model, causal_lm_loss


def tiny_config(model_type="fdt"):
    return ModelConfig(
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=1,
        model_type=model_type,
        dim=32,
        n_layers=2,
        n_heads=4,
        mlp_ratio=2,
        max_seq_len=16,
        dropout=0.0,
        tie_embeddings=True,
        use_rope=True,
        use_self_attention=False,
        num_anchors=12,
        top_k=4,
    )


@torch.no_grad()
def assert_no_future_leakage(model_type):
    torch.manual_seed(123)
    model = build_model(tiny_config(model_type))
    model.eval()
    prefix = [2, 3, 4, 5, 6]
    seq_a = torch.tensor([prefix + [7, 8, 9, 10]], dtype=torch.long)
    seq_b = torch.tensor([prefix + [40, 41, 42, 43]], dtype=torch.long)
    mask = torch.ones_like(seq_a)
    logits_a = model(seq_a, attention_mask=mask)["logits"]
    logits_b = model(seq_b, attention_mask=mask)["logits"]
    assert torch.allclose(logits_a[:, : len(prefix), :], logits_b[:, : len(prefix), :], atol=1e-5)


def test_transformer_no_future_leakage():
    assert_no_future_leakage("transformer")


def test_fdt_no_future_leakage():
    assert_no_future_leakage("fdt")


def test_causal_lm_loss_uses_shifted_labels():
    cfg = tiny_config("transformer")
    logits = torch.zeros(1, 4, cfg.vocab_size)
    input_ids = torch.tensor([[10, 11, 12, 0]])
    mask = torch.tensor([[1, 1, 1, 0]])
    logits[0, 0, 11] = 10.0
    logits[0, 1, 12] = 10.0
    loss = causal_lm_loss(logits, input_ids, mask, cfg.pad_token_id)
    assert loss.item() < 0.01


@torch.no_grad()
def test_anchor_membership_sums_to_one():
    torch.manual_seed(456)
    model = build_model(tiny_config("fdt"))
    ids = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)
    out = model(ids, attention_mask=torch.ones_like(ids))
    assert out["anchor_stats"]
    stats = out["anchor_stats"][0]
    sums = stats.membership.sum(dim=-1)
    assert torch.isfinite(stats.membership).all()
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

