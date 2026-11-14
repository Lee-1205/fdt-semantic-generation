import pytest
import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model


@pytest.mark.parametrize("model_type", ["fdt", "fdt_optimized", "fdt_hybrid", "fdt_anchor_mixer", "fdt_v3"])
@torch.no_grad()
def test_no_future_leakage(model_type):
    torch.manual_seed(12)
    cfg = ModelConfig(vocab_size=67, pad_token_id=0, eos_token_id=1, model_type=model_type, dim=32, n_layers=4, n_heads=4, mlp_ratio=2, max_seq_len=16, dropout=0.0, num_anchors=10, top_k=3, anchor_layer_pattern="alternate", use_rope=model_type != "fdt_v3")
    model = build_model(cfg).eval()
    prefix = [2, 3, 4, 5, 6]
    a = torch.tensor([prefix + [7, 8, 9]])
    b = torch.tensor([prefix + [40, 41, 42]])
    logits_a = model(a, attention_mask=torch.ones_like(a))["logits"]
    logits_b = model(b, attention_mask=torch.ones_like(b))["logits"]
    assert torch.allclose(logits_a[:, : len(prefix)], logits_b[:, : len(prefix)], atol=1e-5, rtol=1e-5)
