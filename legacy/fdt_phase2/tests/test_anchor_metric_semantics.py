import math

import torch

from fdt_rlm.analysis.routing_metrics import entropy_from_membership, normalized_entropy, usage_metrics


def test_entropy_metric_definition():
    membership = torch.tensor([[[0.5, 0.5], [1.0, 0.0]]])
    entropy = entropy_from_membership(membership)
    assert torch.allclose(entropy, torch.tensor([[math.log(2.0), 0.0]]), atol=1e-6)


def test_normalized_entropy():
    membership = torch.full((2, 3, 8), 1.0 / 8.0)
    normalized = normalized_entropy(entropy_from_membership(membership), top_k=8)
    assert torch.allclose(normalized, torch.ones_like(normalized), atol=1e-6)


def test_padding_excluded_from_anchor_metrics():
    indices = torch.tensor([[[0, 1], [2, 3], [3, 2]]])
    membership = torch.full((1, 3, 2), 0.5)
    mask = torch.tensor([[1, 1, 0]])
    metrics = usage_metrics(indices, membership, mask, num_anchors=4)
    assert metrics["valid_tokens"] == 2
    assert metrics["global_active_anchors"] == 4
    assert metrics["mean_active_anchors_per_token"] == 2.0


def test_metric_semantics_distinguish_global_from_per_token():
    indices = torch.tensor([[[0, 1], [2, 3]]])
    membership = torch.full((1, 2, 2), 0.5)
    metrics = usage_metrics(indices, membership, torch.ones(1, 2), num_anchors=4)
    assert metrics["global_active_anchors"] == 4
    assert metrics["mean_active_anchors_per_token"] == 2.0
