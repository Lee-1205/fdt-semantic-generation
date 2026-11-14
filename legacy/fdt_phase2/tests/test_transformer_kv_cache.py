import pytest
import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model


@pytest.mark.parametrize("use_rope", [False, True])
@torch.no_grad()
def test_transformer_kv_cache_matches_full_prefix(use_rope):
    torch.manual_seed(53)
    config = ModelConfig(
        vocab_size=79,
        pad_token_id=0,
        eos_token_id=1,
        model_type="transformer",
        dim=32,
        n_layers=2,
        n_heads=4,
        mlp_ratio=2,
        max_seq_len=16,
        dropout=0.0,
        use_rope=use_rope,
    )
    model = build_model(config).eval()
    prefix = torch.tensor([[2, 3, 4, 5, 6]])
    mask = torch.ones_like(prefix)
    expected_prefix = model(prefix, attention_mask=mask)["logits"][:, -1:]
    cached_prefix, cache = model.prefill(prefix, mask)

    assert cache["backend"] == "transformer_kv"
    assert cache["layers"][0]["key"].shape[-2] == config.max_seq_len
    assert torch.allclose(expected_prefix, cached_prefix["logits"], atol=2e-5, rtol=2e-5)

    for token in (7, 8, 9, 10):
        next_id = torch.tensor([[token]])
        decoded, cache = model.decode_step(next_id, cache)
        prefix = torch.cat((prefix, next_id), dim=1)
        expected = model(prefix, attention_mask=torch.ones_like(prefix))["logits"][:, -1:]
        assert torch.allclose(expected, decoded["logits"], atol=2e-5, rtol=2e-5)
