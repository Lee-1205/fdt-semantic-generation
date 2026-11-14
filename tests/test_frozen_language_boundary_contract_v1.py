"""Tiny CPU ownership/gradient checks, not model ability or diffusion training."""
import pytest
import torch
from torch import nn
from research.frozen_language_boundary_contract_v1 import FrozenLanguageBoundary, state_digest


class Codec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4,4)
        self.role_projection = nn.Linear(4,4)
        self.memory_prior = nn.Parameter(torch.randn(2,4))
        self.decoder = nn.Linear(4,4)
        self.register_buffer('memory_epoch',torch.tensor([2]))


def fixture():
    codec, reasoner = Codec(), nn.Linear(4,4)
    contract = FrozenLanguageBoundary(codec,reasoner,('weight','bias'))
    optimizer = torch.optim.AdamW(reasoner.parameters(),lr=.01)
    return codec,reasoner,contract,optimizer


def test_only_reasoner_changes_through_frozen_realizer():
    c,r,g,o = fixture();before=state_digest(c);old=state_digest(r)
    logits=c.decoder(r(c.encoder(torch.ones(2,4)).detach()))
    logits.square().mean().backward()
    assert all(p.grad is not None for p in r.parameters())
    assert all(p.grad is None for p in c.parameters())
    assert g.assert_optimizer(o);o.step();assert g.assert_intact()
    assert state_digest(c)==before and state_digest(r)!=old


def test_no_grad_would_break_language_loss_to_reasoner():
    c,r,g,o=fixture();z=r(torch.ones(2,4))
    with torch.no_grad(): output=c.decoder(z)
    assert not output.requires_grad
    with pytest.raises(RuntimeError):output.square().mean().backward()


def test_stale_endpoint_gradients_are_cleared():
    c,r=Codec(),nn.Linear(4,4)
    for p in c.parameters():p.grad=torch.ones_like(p)
    g=FrozenLanguageBoundary(c,r,('weight','bias'))
    assert g.assert_intact() and all(p.grad is None for p in c.parameters())


def test_even_frozen_foreign_parameter_is_forbidden_in_optimizer():
    c,r,g,o=fixture();o.add_param_group({'params':[c.memory_prior]})
    with pytest.raises(ValueError,match='reasoning weights only'):g.assert_optimizer(o)


def test_shared_parameter_and_storage_view_are_forbidden():
    c=Codec();r=nn.Linear(4,4);r.weight=c.encoder.weight
    with pytest.raises(ValueError,match='share'):FrozenLanguageBoundary(c,r,('weight','bias'))
    r=nn.Linear(4,4);r.weight=nn.Parameter(c.encoder.weight.detach().view(4,4))
    with pytest.raises(ValueError,match='share'):FrozenLanguageBoundary(c,r,('weight','bias'))


def test_mutable_buffer_is_not_frozen_by_requires_grad():
    c,r,g,o=fixture();c.memory_epoch.add_(1)
    with pytest.raises(ValueError,match='buffer changed'):g.assert_intact()


def test_eval_mode_cannot_be_overridden_by_parent_train():
    c,r,g,o=fixture();c.train()
    with pytest.raises(ValueError,match='eval mode'):g.assert_intact()


def test_in_place_endpoint_parameter_change_is_rejected():
    c,r,g,o=fixture()
    with torch.no_grad():c.decoder.bias.add_(1)
    with pytest.raises(ValueError,match='buffer changed'):g.assert_intact()


def test_exact_allowlist_cannot_omit_or_duplicate_parameters():
    for names in ((),('weight',),('weight','bias','bias'),('weight','bias','unknown')):
        with pytest.raises(ValueError,match='exact reasoning'):FrozenLanguageBoundary(Codec(),nn.Linear(4,4),names)


def test_trainability_change_or_structure_change_is_rejected():
    c,r,g,o=fixture();c.encoder.weight.requires_grad_(True)
    with pytest.raises(ValueError,match='trainability'):g.assert_intact()
    c,r,g,o=fixture();r.register_parameter('unexpected',nn.Parameter(torch.ones(1)))
    with pytest.raises(ValueError,match='structure changed'):g.assert_optimizer(o)


def test_new_optimizer_parameter_identity_cannot_replace_owned_tensor():
    c,r,g,o=fixture();r.weight=nn.Parameter(r.weight.detach().clone())
    # Names are insufficient: the optimizer still owns the stale object.
    with pytest.raises(ValueError):g.assert_optimizer(torch.optim.AdamW(r.parameters()))
    with pytest.raises(ValueError,match='identity changed'):g.assert_optimizer(o)
