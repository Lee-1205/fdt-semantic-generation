import math

import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.models.causal_lm import AnchorStats, anchor_regularization_losses, build_model


def _model():
    config = ModelConfig(
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=1,
        dim=32,
        n_layers=1,
        n_heads=4,
        max_seq_len=8,
        num_anchors=4,
        top_k=4,
        dropout=0.0,
    )
    return build_model(config)


def _stats(entropy: torch.Tensor, usage_prob=None, load_prob=None):
    membership = torch.full((1, 1, 4), 0.25)
    usage_prob = usage_prob if usage_prob is not None else torch.full((4,), 0.25)
    load_prob = load_prob if load_prob is not None else torch.full((4,), 0.25)
    return AnchorStats(
        indices=torch.arange(4).view(1, 1, 4),
        membership=membership,
        entropy=entropy,
        usage_prob=usage_prob,
        load_prob=load_prob,
        cluster_loss=entropy.new_zeros(()),
        top1_membership=entropy.new_tensor(0.25),
        membership_margin=entropy.new_zeros(()),
    )


def test_entropy_gradient_pushes_up_below_target():
    entropy = torch.tensor(0.2, requires_grad=True)
    reg = anchor_regularization_losses(_model(), [_stats(entropy)], 0.55, 0.55)
    loss = -0.003 * reg["entropy_reward"] + 0.10 * reg["entropy_over"]
    loss.backward()
    assert entropy.grad.item() < 0.0
    assert math.isclose(reg["entropy_target"].item(), 0.55 * math.log(4.0), rel_tol=1e-6)


def test_entropy_gradient_pushes_down_above_target():
    entropy = torch.tensor(math.log(4.0), requires_grad=True)
    reg = anchor_regularization_losses(_model(), [_stats(entropy)], 0.55, 0.55)
    loss = -0.003 * reg["entropy_reward"] + 0.10 * reg["entropy_over"]
    loss.backward()
    assert entropy.grad.item() > 0.0
    assert reg["entropy_reward"].item() == reg["entropy_target"].item()
    assert reg["entropy_over"].item() > 0.0


def test_switch_usage_penalizes_overloaded_anchor():
    entropy = torch.tensor(0.5)
    balanced = anchor_regularization_losses(_model(), [_stats(entropy)])

    importance = torch.tensor([0.7, 0.1, 0.1, 0.1], requires_grad=True)
    load = torch.tensor([0.7, 0.1, 0.1, 0.1])
    overloaded = anchor_regularization_losses(
        _model(), [_stats(entropy, usage_prob=importance, load_prob=load)]
    )
    assert math.isclose(balanced["usage"].item(), 0.0, abs_tol=1e-6)
    assert overloaded["usage"].item() > balanced["usage"].item()
    overloaded["usage"].backward()
    assert importance.grad[0].item() > importance.grad[1].item()


def test_later_layers_receive_lower_entropy_targets():
    model = _model()
    entropies = [torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5)]
    reg = anchor_regularization_losses(
        model,
        [_stats(value) for value in entropies],
        entropy_target_ratio_start=0.65,
        entropy_target_ratio_end=0.45,
    )
    expected_mean = 0.55 * math.log(4.0)
    assert math.isclose(reg["entropy_target"].item(), expected_mean, rel_tol=1e-6)
