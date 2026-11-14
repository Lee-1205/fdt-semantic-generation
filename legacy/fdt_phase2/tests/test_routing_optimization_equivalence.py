import torch

from fdt_rlm.models.causal_lm import (
    dense_causal_prefix_aggregate,
    dense_causal_prefix_aggregate_reference,
)


def _inputs(dtype=torch.float32):
    torch.manual_seed(17)
    logits = torch.randn(2, 11, 5, dtype=dtype, requires_grad=True)
    membership = logits.softmax(dim=-1)
    indices = torch.stack([torch.randperm(13)[:5] for _ in range(22)]).reshape(2, 11, 5)
    values = torch.randn(2, 11, 7, dtype=dtype, requires_grad=True)
    mask = torch.tensor([[1] * 9 + [0, 0], [1] * 11], dtype=dtype)
    return logits, membership, indices, values, mask


def test_gather_first_matches_reference_output_and_gradients():
    logits, membership, indices, values, mask = _inputs()
    reference = dense_causal_prefix_aggregate_reference(membership, indices, values, mask, 13)
    reference.square().sum().backward()
    reference_logits_grad = logits.grad.detach().clone()
    reference_values_grad = values.grad.detach().clone()

    logits2, membership2, indices2, values2, mask2 = _inputs()
    optimized = dense_causal_prefix_aggregate(membership2, indices2, values2, mask2, 13)
    optimized.square().sum().backward()

    assert torch.equal(optimized, reference.detach())
    assert torch.equal(logits2.grad, reference_logits_grad)
    assert torch.equal(values2.grad, reference_values_grad)


def test_gather_first_preserves_causality():
    _, membership, indices, values, mask = _inputs()
    prefix = dense_causal_prefix_aggregate(membership, indices, values, mask, 13)
    changed = values.detach().clone()
    changed[:, 7:] += 100.0
    changed_prefix = dense_causal_prefix_aggregate(membership.detach(), indices, changed, mask, 13)
    assert torch.equal(prefix[:, :7], changed_prefix[:, :7])
