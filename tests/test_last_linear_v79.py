"""Scope isolation: unchanged V78 objective, frozen features, two-tensor Adam."""
import copy
import pytest
import torch
from torch import nn
from research.commit_patch_readout_v54 import CommitPatchReadout
from research.last_linear_objective_v79 import freeze_to_last_linear,LAST_LINEAR_NAMES
from research import last_linear_objective_v79 as candidate
from research import competing_margin_objective_v78 as previous


class Tiny(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(32,32)
        self.readout=nn.Module();self.readout.role_projection=nn.Linear(16,2,bias=False)
        self.readout.readout=CommitPatchReadout(dimension=32,hidden=64)


def test_loss_math_and_group_weights_are_unchanged():
    assert candidate.decoder_losses is previous.decoder_losses
    assert candidate.gold_margin is previous.gold_margin and candidate.GROUP_ROW_WEIGHTS==previous.GROUP_ROW_WEIGHTS
    assert candidate.MARGIN_COEFFICIENTS==previous.MARGIN_COEFFICIENTS


def test_actual_scope_gradients_and_hidden_features_frozen_after_update():
    torch.manual_seed(79);model=Tiny();initial=copy.deepcopy(model.state_dict());names=freeze_to_last_linear(model)
    assert tuple(names)==LAST_LINEAR_NAMES
    local=torch.randn(2,32);global_views=torch.randn(2,2,32);terminal=torch.tensor([False,True])
    head=model.readout.readout
    def hidden():
        feature=torch.cat((local+terminal[:,None]*head.terminal,global_views.flatten(1)),-1)
        return head.net[:2](head.norm(feature)).detach().clone()
    before=hidden();optimizer=torch.optim.AdamW([dict(model.named_parameters())[n] for n in names],lr=1e-5)
    loss=head(local,global_views,terminal).square().mean();loss.backward()
    assert all(p.grad is not None for n,p in model.named_parameters() if n in names)
    assert all(p.grad is None for n,p in model.named_parameters() if n not in names)
    optimizer.step();assert torch.equal(hidden(),before)
    assert all(torch.equal(t,initial[n]) for n,t in model.state_dict().items() if n not in names)
    assert all(not torch.equal(dict(model.named_parameters())[n],initial[n]) for n in names)
    assert len(optimizer.state)==2


def test_scope_rejects_different_final_layer_name():
    model=Tiny();model.readout.readout.net[2]=nn.Sequential(nn.Linear(64,4128))
    with pytest.raises(ValueError,match='exact last linear'):freeze_to_last_linear(model)
