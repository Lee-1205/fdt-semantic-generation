"""Analytic CPU counterexamples; no actual learned FDT forward or optimizer."""
import copy,pytest,torch
from research.feasible_readout_delta_v89 import LinearConstraint,correct_actual_delta
from research.actual_update_retention_v80 import TrainDecision

ANCHOR='a'*64
def constraint(h=(1.,0.),gold=0,rival=1,floor=.05,role=0,dtype=torch.float64):
    return LinearConstraint(role,gold,rival,torch.tensor(h,dtype=dtype),floor,'CPU_TRAIN:a',0,ANCHOR)
def gap(w,b,h,gold=0,rival=1):return float(torch.dot(w[gold]-w[rival],torch.tensor(h,dtype=w.dtype))+b[gold]-b[rival])
def fixture():
    w=torch.tensor([[.1,-.1],[0.,0.]],dtype=torch.float64);b=torch.zeros(2,dtype=torch.float64);p=torch.tensor([[-.1,.1],[0.,0.]],dtype=torch.float64)
    return w,b,p,b.clone()

def test_changes_direction_when_no_global_scalar_satisfies_both_constraints():
    w,b,p,q=fixture();old=copy.deepcopy((w,b,p,q));rng=torch.get_rng_state().clone();c=constraint()
    # Retaining old h_A margin>=.05 requires alpha<=.25. Correct h_B
    # requires alpha>.5. A single scalar along this proposal cannot do both.
    assert (.05-.1)/(-.2)==.25 and .1/.2==.5
    def full(x,y):return TrainDecision(gap(x,y,(1.,0.))>=.05 and gap(x,y,(0.,1.))>0,dict(A_gap=gap(x,y,(1.,0.)),B_gap=gap(x,y,(0.,1.))))
    out=correct_actual_delta(w,b,p,q,[c],full,classes=2)
    assert out['receipt']['global_reference_pass'] and gap(out['weight'],out['bias'],(0.,1.))>0 and out['receipt']['full_reference_calls']==1
    needed=.05+1e-6-(-.1);tau=needed/4
    assert torch.allclose(out['weight'][0]-p[0],torch.tensor([tau,0.],dtype=w.dtype),atol=1e-12)
    assert abs(out['receipt']['correction_norm']-needed/2)<1e-12
    assert all(torch.equal(a,z) for a,z in zip((w,b,p,q),old)) and torch.equal(rng,torch.get_rng_state())

def test_active_only_repair_cannot_bypass_full_reference():
    w,b,p,q=fixture()
    def full(x,y):return TrainDecision(gap(x,y,(-1.,0.),1,0)>=.05,dict(other_context_gap=gap(x,y,(-1.,0.),1,0)))
    out=correct_actual_delta(w,b,p,q,[constraint()],full,classes=2)
    assert out['weight'] is None and out['receipt']['status']=='full_reference_rejected_active_only_repair'

def test_fixed_radius_and_infeasible_warm_point_fail_closed():
    w,b,p,q=fixture();ok=lambda x,y:TrainDecision(True,dict(test=1))
    out=correct_actual_delta(w,b,p,q,[constraint()],ok,classes=2,relative_radius=.1)
    assert out['weight'] is None and out['receipt']['status']=='fixed_correction_radius_exceeded'
    with pytest.raises(ValueError):correct_actual_delta(w,b,p,q,[constraint(floor=.2)],ok,classes=2)
    with pytest.raises(ValueError):correct_actual_delta(w,b,p,q,[constraint()]*17,ok,classes=2)

def test_bounded_sweeps_are_not_claimed_as_exact_convex_solver():
    w=torch.tensor([[-1.,0.],[0.,0.]],dtype=torch.float64);b=torch.tensor([1.5,0.],dtype=torch.float64);p=w.clone();q=torch.tensor([-1.5,0.],dtype=torch.float64)
    constraints=[constraint(floor=.1),constraint(h=(2.,0.),gold=1,rival=0,floor=.1)]
    out=correct_actual_delta(w,b,p,q,constraints,lambda x,y:TrainDecision(True,dict(test=1)),classes=2)
    assert out['weight'] is None and out['receipt']['status']=='bounded_active_sweeps_not_converged' and out['receipt']['full_reference_calls']==0

def test_real_byte_head_shape_zero_feature_and_no_external_mutation():
    w=torch.zeros(16*258,512,dtype=torch.float32);b=torch.zeros(16*258);g=13*258+43;j=13*258+35;b[g]=.1;p=w.clone();q=b.clone();q[g]=-.1
    c=LinearConstraint(13,43,35,torch.zeros(512),.05,'CPU_TRAIN:byte',13,ANCHOR)
    out=correct_actual_delta(w,b,p,q,[c],lambda x,y:TrainDecision(float(y[g]-y[j])>=.05,dict(gap=float(y[g]-y[j]))))
    assert out['receipt']['global_reference_pass'] and torch.equal(w,p) and q[g]==-.1 and b[g]==.1

def test_reference_mutation_and_boolean_only_success_are_rejected():
    w,b,p,q=fixture()
    def bad(x,y):x.zero_();return TrainDecision(True,dict(test=1))
    with pytest.raises(ValueError):correct_actual_delta(w,b,p,q,[constraint()],bad,classes=2)
    with pytest.raises(ValueError):correct_actual_delta(w,b,p,q,[constraint()],lambda x,y:True,classes=2)
