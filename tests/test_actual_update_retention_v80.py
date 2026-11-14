"""Tiny CPU actual Adam/AMP transaction semantics, not FDT ability."""
import copy,io,random
import pytest
import torch
from research.actual_update_retention_v80 import bounded_adam_proposal,TrainDecision,_equal
from research.last_linear_objective_v79 import LAST_LINEAR_NAMES


def fixture():
    torch.manual_seed(80);model=torch.nn.Linear(3,2);opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.1);scaler=torch.amp.GradScaler('cpu',init_scale=1024.)
    names=dict(zip(LAST_LINEAR_NAMES,(model.weight,model.bias)))
    return model,opt,scaler,names


def gradients(model,opt,scaler,overflow=False):
    opt.zero_grad(set_to_none=True);loss=model(torch.tensor([[.2,-.5,1.3]])).square().sum();scaler.scale(loss).backward()
    if overflow:model.weight.grad.fill_(float('inf'))
    scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)


def test_full_accept_matches_real_AdamW_with_decay_scaler_and_RNG():
    m,o,s,n=fixture();m2,o2,s2,_=fixture();gradients(m,o,s);gradients(m2,o2,s2)
    rng=torch.get_rng_state().clone();prng=random.getstate();s2.step(o2);s2.update()
    receipt=bounded_adam_proposal(o,n,s,lambda:TrainDecision(True,{'TRAIN_loss':.1}))
    assert receipt['alpha']==1 and _equal(m.state_dict(),m2.state_dict()) and _equal(o.state_dict(),o2.state_dict()) and _equal(s.state_dict(),s2.state_dict())
    assert torch.equal(rng,torch.get_rng_state()) and prng==random.getstate() and all(p.grad is None for p in m.parameters())


def test_half_accept_scales_actual_delta_and_retains_proposal_moments():
    m,o,s,n=fixture();m2,o2,s2,_=fixture();old=copy.deepcopy(m.state_dict());gradients(m,o,s);gradients(m2,o2,s2);s2.step(o2);s2.update();calls=[]
    def accept():calls.append(1);return TrainDecision(len(calls)==2,{'TRAIN_loss':1/len(calls)})
    receipt=bounded_adam_proposal(o,n,s,accept)
    assert receipt['alpha']==.5 and len(calls)==2 and _equal(o.state_dict(),o2.state_dict())
    assert all(torch.equal(v,old[k]+.5*(m2.state_dict()[k]-old[k])) for k,v in m.state_dict().items())


def test_all_rejected_populated_Adam_exact_and_finite_scaler_attempt_consumed():
    m,o,s,n=fixture();gradients(m,o,s);bounded_adam_proposal(o,n,s,lambda:TrainDecision(True,{'TRAIN_loss':0.}))
    old=copy.deepcopy(m.state_dict());adam=copy.deepcopy(o.state_dict());tracker=s.state_dict()['_growth_tracker'];gradients(m,o,s)
    receipt=bounded_adam_proposal(o,n,s,lambda:TrainDecision(False,{'TRAIN_loss':2.}))
    assert not receipt['accepted'] and len(receipt['reference_checks'])==4 and _equal(old,m.state_dict()) and _equal(adam,o.state_dict())
    assert s.state_dict()['_growth_tracker']==tracker+1 and all(p.grad is None for p in m.parameters())


def test_AMP_skip_never_calls_reference_or_mutates_Adam():
    m,o,s,n=fixture();old=copy.deepcopy(m.state_dict());gradients(m,o,s,True)
    def forbidden():raise AssertionError('AMP skipped proposal must not call reference')
    receipt=bounded_adam_proposal(o,n,s,forbidden)
    assert receipt['AMP_skip'] and not o.state and _equal(old,m.state_dict()) and s.get_scale()==512.


@pytest.mark.parametrize('reason',('exception','STOP','mutation','nonfinite'))
def test_any_postproposal_exception_rolls_back_weights_Adam_scaler_and_can_save_restore(reason):
    m,o,s,n=fixture();gradients(m,o,s);old=copy.deepcopy(m.state_dict());adam=copy.deepcopy(o.state_dict());scale=copy.deepcopy(s.state_dict())
    def reference():
        if reason=='exception':raise RuntimeError('bounded reference failure')
        if reason=='STOP':raise TimeoutError('cooperative STOP')
        if reason=='mutation':m.bias.add_(1);return TrainDecision(True,{'TRAIN_loss':0.})
        return TrainDecision(True,{'TRAIN_loss':float('nan')})
    with pytest.raises((RuntimeError,TimeoutError,ValueError,FloatingPointError)):bounded_adam_proposal(o,n,s,reference)
    assert _equal(old,m.state_dict()) and _equal(adam,o.state_dict()) and _equal(scale,s.state_dict()) and all(p.grad is None for p in m.parameters())
    buf=io.BytesIO();torch.save(dict(model=m.state_dict(),Adam=o.state_dict(),scaler=s.state_dict(),rng=torch.get_rng_state()),buf);buf.seek(0);saved=torch.load(buf,weights_only=True)
    m2,o2,s2,n2=fixture();m2.load_state_dict(saved['model']);o2.load_state_dict(saved['Adam']);s2.load_state_dict(saved['scaler']);torch.set_rng_state(saved['rng'])
    gradients(m,o,s);gradients(m2,o2,s2)
    accept=lambda:TrainDecision(True,{'TRAIN_loss':0.})
    bounded_adam_proposal(o,n,s,accept);bounded_adam_proposal(o2,n2,s2,accept)
    assert _equal(m.state_dict(),m2.state_dict()) and _equal(o.state_dict(),o2.state_dict()) and _equal(s.state_dict(),s2.state_dict())


def test_extra_optimizer_tensor_is_rejected_before_step():
    m,o,s,n=fixture();o.add_param_group({'params':[torch.nn.Parameter(torch.ones(1))]});gradients(m,o,s)
    with pytest.raises(ValueError,match='ownership'):bounded_adam_proposal(o,n,s,lambda:TrainDecision(True,{'TRAIN_loss':0.}))


def test_raw_gradient_dot_can_be_safe_while_actual_preconditioned_delta_is_not():
    retention=torch.tensor([1.,1.]);gradient=torch.tensor([2.,-1.]);diagonal=torch.tensor([.1,10.]);delta=-diagonal*gradient
    assert retention@gradient>0 and retention@delta>0
