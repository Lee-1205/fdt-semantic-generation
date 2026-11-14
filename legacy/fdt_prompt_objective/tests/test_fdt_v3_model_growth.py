from __future__ import annotations

import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.model_growth import expanded_fdt_v3_config, grow_fdt_v3_state_dict
from fdt_rlm.models import build_model


def test_width_and_depth_growth_preserve_logits_without_growth_noise():
    torch.manual_seed(7)
    source_config = ModelConfig(
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=2,
        model_type="fdt_v3",
        dim=64,
        n_layers=3,
        n_heads=2,
        mlp_ratio=4,
        max_seq_len=32,
        dropout=0.0,
        tie_embeddings=True,
        use_rope=False,
        use_self_attention=False,
        num_anchors=16,
        top_k=4,
        routing_type="cosine",
        aggregation_impl="sparse_chunked_scan",
        routing_backend="matmul",
        enable_diagnostics=False,
        anchor_layer_indices=[0, 2],
        local_attention_window=8,
        router_dim=16,
        anchor_scan_chunk_size=8,
        anchor_recency_reference_len=32,
    )
    target_config = expanded_fdt_v3_config(
        source_config,
        target_dim=96,
        target_layers=5,
        target_heads=3,
        target_router_dim=24,
    )
    source = build_model(source_config).eval()
    target = build_model(target_config).eval()
    state = grow_fdt_v3_state_dict(
        source.state_dict(),
        target,
        source_layers=source_config.n_layers,
        noise_std=0.0,
    )
    target.load_state_dict(state, strict=True)
    ids = torch.randint(3, source_config.vocab_size, (2, 17))
    mask = torch.ones_like(ids)
    with torch.inference_mode():
        source_logits = source(ids, attention_mask=mask)["logits"]
        target_logits = target(ids, attention_mask=mask)["logits"]
    # RMSNorm's epsilon prevents exact scale invariance when zero-padding the
    # normalized width, but the migrated logits remain numerically equivalent.
    torch.testing.assert_close(target_logits, source_logits, atol=2e-4, rtol=2e-4)
    assert target.config.use_self_attention is False
    assert target.anchor_layer_indices == [0, 2, 4]
