import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.models.causal_lm import CausalFuzzyAnchorLayer


def _layer():
    cfg = ModelConfig(vocab_size=64, dim=32, n_layers=1, n_heads=4, max_seq_len=16, num_anchors=12, top_k=4, dropout=0.0)
    layer = CausalFuzzyAnchorLayer(cfg).eval()
    return layer


def test_uniform_membership_counterfactual():
    layer = _layer()
    layer.set_probe_mode("uniform_topk")
    _, stats = layer(torch.randn(2, 7, 32), torch.ones(2, 7))
    assert torch.allclose(stats.membership, torch.full_like(stats.membership, 0.25))


def test_anchor_metric_shapes():
    layer = _layer()
    _, stats = layer(torch.randn(2, 7, 32), torch.ones(2, 7))
    assert stats.indices.shape == (2, 7, 4)
    assert stats.membership.shape == (2, 7, 4)
    assert stats.usage_prob.shape == (12,)


def test_random_probe_reproducibility():
    layer = _layer()
    layer.set_probe_mode("random_topk")
    x = torch.randn(2, 7, 32)
    _, first = layer(x, torch.ones(2, 7))
    _, second = layer(x, torch.ones(2, 7))
    assert torch.equal(first.indices, second.indices)


def test_shuffle_membership_preserves_weights_but_changes_assignment():
    layer = _layer()
    x = torch.randn(2, 7, 32)
    _, learned = layer(x, torch.ones(2, 7))
    layer.set_probe_mode("shuffle_membership")
    _, shuffled = layer(x, torch.ones(2, 7))
    assert torch.equal(shuffled.indices, learned.indices)
    assert torch.allclose(
        shuffled.membership.sort(dim=-1).values,
        learned.membership.sort(dim=-1).values,
    )
    assert not torch.equal(shuffled.membership, learned.membership)


def test_onehot_top1_keeps_selected_indices():
    layer = _layer()
    x = torch.randn(2, 7, 32)
    _, learned = layer(x, torch.ones(2, 7))
    layer.set_probe_mode("onehot_top1")
    _, onehot = layer(x, torch.ones(2, 7))
    assert torch.equal(onehot.indices, learned.indices)
    assert torch.equal(onehot.membership[..., 0], torch.ones_like(onehot.membership[..., 0]))
    assert torch.equal(onehot.membership[..., 1:], torch.zeros_like(onehot.membership[..., 1:]))
