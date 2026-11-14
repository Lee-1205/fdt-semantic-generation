import pytest
import torch
from research.train_margin_retention_v81 import reference,decision

def scores(gap):
    z=torch.full((1,258),-100.,dtype=torch.float32);z[0,0]=gap;z[0,1]=0;return z

def test_v80_top1_can_accept_margin_collapse_new_criterion_rejects():
    y=torch.tensor([0]);a=scores(1.);b=scores(.01)
    assert int(a.argmax(-1))==int(b.argmax(-1))==0 # old top1 gate passes
    bank=reference(a,y);r=decision(b,bank)
    assert not r.accept and r.measurements['new_top1_flips']==0 and r.measurements['margin_floor_violations']==1
    assert decision(scores(.51),bank).accept

def test_initial_wrong_can_improve_and_PAD_is_not_protected():
    initial=torch.cat([scores(-1.),scores(1.)]);y=torch.tensor([0,257]);bank=reference(initial,y)
    assert bank['protected'].sum()==0 and decision(torch.cat([scores(2.),scores(-2.)]),bank).accept

def test_matching_precision_required_and_unseen_not_certified():
    with pytest.raises(ValueError):reference(scores(1.).bfloat16(),torch.tensor([0]))
    bank=reference(scores(1.),torch.tensor([0]));assert decision(scores(1.),bank).accept
    # A separate unseen feature can still fail: no all-data claim from a bank.
    assert int(scores(-1.).argmax(-1))!=0
