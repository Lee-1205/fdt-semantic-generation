import pytest
import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model


@pytest.mark.parametrize("model_type", ["fdt", "fdt_optimized", "fdt_anchor_mixer"])
@pytest.mark.parametrize("use_rope", [True, False])
@pytest.mark.parametrize("use_local_mixer", [False, True])
@torch.no_grad()
def test_incremental_anchor_cache_matches_full_prefix(model_type, use_rope, use_local_mixer):
    torch.manual_seed(19)
    cfg = ModelConfig(vocab_size=73, pad_token_id=0, eos_token_id=1, model_type=model_type, dim=32, n_layers=2, n_heads=4, mlp_ratio=2, max_seq_len=16, dropout=0.0, num_anchors=11, top_k=3, use_rope=use_rope, use_incremental_anchor_state=True, use_cached_normalized_anchors=True, use_local_mixer=use_local_mixer, local_mixer_kernel_size=3)
    model = build_model(cfg).eval()
    if use_local_mixer:
        for block in model.blocks:
            torch.nn.init.normal_(block.local_mixer.depthwise.weight, std=0.1)
            torch.nn.init.normal_(block.local_mixer.channel_mix.weight, std=0.02)
    prefix = torch.tensor([[2, 3, 4, 5, 6]])
    full_prefix = model(prefix, attention_mask=torch.ones_like(prefix))["logits"][:, -1:]
    cached_prefix, cache = model.prefill(prefix, torch.ones_like(prefix))
    assert cache["backend"] == "anchor_incremental"
    assert torch.allclose(full_prefix, cached_prefix["logits"], atol=2e-5, rtol=2e-5)

    next_id = torch.tensor([[7]])
    decoded, _ = model.decode_step(next_id, cache)
    full = torch.cat((prefix, next_id), dim=1)
    expected = model(full, attention_mask=torch.ones_like(full))["logits"][:, -1:]
    assert torch.allclose(expected, decoded["logits"], atol=2e-5, rtol=2e-5)


@torch.no_grad()
def test_local_mixer_has_no_future_leakage():
    torch.manual_seed(23)
    cfg = ModelConfig(vocab_size=73, pad_token_id=0, eos_token_id=1, model_type="fdt_optimized", dim=32, n_layers=2, n_heads=4, mlp_ratio=2, max_seq_len=16, dropout=0.0, num_anchors=11, top_k=3, use_rope=False, use_local_mixer=True, local_mixer_kernel_size=5)
    model = build_model(cfg).eval()
    for block in model.blocks:
        torch.nn.init.normal_(block.local_mixer.depthwise.weight, std=0.1)
        torch.nn.init.normal_(block.local_mixer.channel_mix.weight, std=0.02)
    left = torch.tensor([[2, 3, 4, 5, 6, 7]])
    right = torch.tensor([[2, 3, 4, 31, 32, 33]])
    left_logits = model(left, attention_mask=torch.ones_like(left))["logits"]
    right_logits = model(right, attention_mask=torch.ones_like(right))["logits"]
    assert torch.allclose(left_logits[:, :3], right_logits[:, :3], atol=2e-5, rtol=2e-5)
