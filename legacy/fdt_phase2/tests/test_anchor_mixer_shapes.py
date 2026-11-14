import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model


def test_anchor_mixer_is_attention_free_and_has_expected_shapes():
    cfg = ModelConfig(vocab_size=41, pad_token_id=0, eos_token_id=1, model_type="fdt_anchor_mixer", dim=32, n_layers=3, n_heads=4, max_seq_len=16, num_anchors=12, top_k=3, dropout=0.0)
    model = build_model(cfg)
    assert all(block.attn is None for block in model.blocks)
    ids = torch.randint(2, 41, (2, 9))
    output = model(ids, attention_mask=torch.ones_like(ids))
    assert output["hidden"].shape == (2, 9, 32)
    assert len(output["anchor_stats"]) == 3
    assert all(stats.indices.shape == (2, 9, 3) for stats in output["anchor_stats"])
