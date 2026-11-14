"""Numerical and autograd contracts; no large model or ability claim."""
import copy
import pytest
import torch
from research.competing_margin_objective_v78 import gold_margin,margin_retention,decoder_losses


def case():
    z=torch.full((1,5,258),-8.)
    y=torch.tensor([[50,43,97,256,257]])
    z[0,0,50]=.7;z[0,0,43]=.5
    z[0,1,43]=.2;z[0,1,57]=.8
    z[0,2,97]=.6;z[0,2,98]=.3
    z[0,3,256]=1.;z[0,4,257]=1.
    correct=z.argmax(-1)==y;m,c=gold_margin(z,y)
    return z,y,torch.tensor([[True,True,False,False,False]]),correct,m,c


def test_initial_margin_is_stationary_and_common_logit_shift_invariant():
    z,y,r,c,m,_=case();z.requires_grad_();loss,parts=margin_retention(z,y,r,c,m)
    assert float(loss.detach())==0.;loss.backward();assert z.grad.count_nonzero()==0
    shifted,_=margin_retention(z.detach()+13.,y,r,c,m)
    assert torch.allclose(loss,shifted,atol=2e-6)


def test_actual_new_competitor_is_constrained_not_only_teacher_competitor():
    z,y,r,c,m,_=case();z[0,0,54]=1.2;z.requires_grad_()
    loss,stats=margin_retention(z,y,r,c,m);loss.backward()
    assert z.grad[0,0,50]<0 and z.grad[0,0,54]>0 and z.grad[0,0,43]==0
    assert stats['critical_margin_violations']==1


def test_incorrect_teacher_EOS_PAD_excluded_and_teacher_detached():
    z,y,r,c,m,_=case();z[0,0,43]=1.2;z.requires_grad_();m.requires_grad_()
    loss,parts=margin_retention(z,y,r,c,m);loss.backward()
    assert m.grad is None and z.grad[0,1].count_nonzero()==0 and z.grad[0,3:].count_nonzero()==0
    assert parts['critical_margin_eligible']==parts['noncritical_margin_eligible']==1


def test_teacher_target_is_capped_and_easy_teacher_is_not_weak():
    z,y,r,c,m,_=case();m[0,0]=2.;m[0,2]=4.
    loss,stats=margin_retention(z,y,r,c,m)
    assert torch.isclose(loss,torch.tensor(.25*(1.-.2))) and stats['noncritical_margin_eligible']==0


def test_empty_mask_has_finite_zero_differentiable_loss():
    z,y,r,c,m,_=case();c.zero_();m=m.abs().neg();z.requires_grad_()
    loss,stats=margin_retention(z,y,r,c,m);loss.backward()
    assert float(loss.detach())==0 and torch.isfinite(z.grad).all() and z.grad.count_nonzero()==0


def test_malformed_teacher_rejected():
    z,y,r,c,m,_=case();m[0,0]=-1
    with pytest.raises(ValueError,match='contradict'):margin_retention(z,y,r,c,m)
    m[0,0]=float('nan')
    with pytest.raises(ValueError):margin_retention(z,y,r,c,m)


def test_complete_loss_finite_with_correction_and_padding():
    z,y,r,c,m,_=case();z.requires_grad_();p=z.detach().softmax(-1).gather(-1,y[...,None]).squeeze(-1);w=torch.ones(258);w[257]=0
    loss,parts=decoder_losses(z,y,w,r,p,c,m);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all() and z.grad[0,-1].count_nonzero()==0
    assert parts['wrong_critical_position_amplification']<=8


def test_five_micro_gradient_matches_one_group_objective_before_single_clip():
    torch.manual_seed(78);x=[torch.randn(i+1,4) for i in range(5)];y=[torch.randn(i+1,3) for i in range(5)]
    a=torch.nn.Linear(4,3);b=copy.deepcopy(a)
    for u,v in zip(x,y):(.2*(a(u)-v).square().mean()).backward()
    sum(.2*(b(u)-v).square().mean() for u,v in zip(x,y)).backward()
    for p,q in zip(a.parameters(),b.parameters()):assert torch.allclose(p.grad,q.grad,atol=1e-7)
    torch.nn.utils.clip_grad_norm_(a.parameters(),1.);torch.nn.utils.clip_grad_norm_(b.parameters(),1.)
    oa=torch.optim.AdamW(a.parameters(),lr=1e-5);ob=torch.optim.AdamW(b.parameters(),lr=1e-5);oa.step();ob.step()
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p,q)


def test_report_diagnostic_does_not_change_after_parent_report_wrapping():
    from research.fuzzy_factual_objective_v70 import training_weights
    from research.update_balance_objective_v77 import frozen_objective_report as v77report
    from research.competing_margin_objective_v78 import frozen_objective_report
    _,current=training_weights([dict(id='t',prompt='A quantity is +2 kg.')])
    parent=copy.deepcopy(current);parent['class_weights']=[7.,8.,9.,10.]
    profile=dict(parent_objective_report=parent,source_SHA='fixture')
    direct=frozen_objective_report(current,profile)
    wrapped=frozen_objective_report(v77report(current,profile),profile)
    assert direct==wrapped and direct['current_unique_pool_class_weights_NOT_used']==current['class_weights']
