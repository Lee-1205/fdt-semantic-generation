"""Bounded correction of an actual last-Linear parameter proposal.

CPU candidate primitive, not an optimizer/driver or a learned-model result.
TRAIN-only linear constraints are supplied by an existing full reference check.
All reference floors/anchors stay immutable. No raw-gradient projection, added
SGD step, hidden retry, floor relaxation or inference-time label feature.
Sequential halfspace corrections are not an exact nearest-intersection solver.
The full finite-bank check remains authoritative after the bounded correction.
"""
from dataclasses import dataclass
import math
import torch
from research.actual_update_retention_v80 import TrainDecision

@dataclass(frozen=True)
class LinearConstraint:
    role: int
    gold: int
    rival: int
    hidden: torch.Tensor
    floor: float
    source_id: str
    byte_offset: int
    anchor_SHA: str

def _gap(w,b,c,classes):
    g=c.role*classes+c.gold;j=c.role*classes+c.rival
    return torch.dot((w[g]-w[j]).double(),c.hidden.double())+(b[g]-b[j]).double()

@torch.no_grad()
def correct_actual_delta(warm_w,warm_b,proposal_w,proposal_b,constraints,full_reference,*,classes=258,passes=2,max_constraints=16,relative_radius=1.,padding=1e-6):
    if passes!=2 or max_constraints!=16 or not 0<relative_radius<=1 or padding!=1e-6:raise ValueError('fixed prospective correction policy')
    tensors=(warm_w,warm_b,proposal_w,proposal_b)
    if any(not torch.is_tensor(t) or t.dtype not in (torch.float32,torch.float64) or not torch.isfinite(t).all() for t in tensors):raise ValueError('finite actual parameter tensors')
    if len({t.dtype for t in tensors})!=1 or len({t.device for t in tensors})!=1 or warm_w.ndim!=2 or warm_b.ndim!=1 or warm_w.shape!=proposal_w.shape or warm_b.shape!=proposal_b.shape or warm_w.shape[0]!=len(warm_b) or len(warm_b)%classes:raise ValueError('one last Linear tensor pair')
    if not constraints or len(constraints)>max_constraints:raise ValueError('bounded explicit active TRAIN constraints; no truncation')
    roles=len(warm_b)//classes;anchor={c.anchor_SHA for c in constraints}
    if len(anchor)!=1 or not isinstance(next(iter(anchor)),str) or len(next(iter(anchor)))!=64:raise ValueError('one immutable reference anchor')
    for c in constraints:
        if (not isinstance(c,LinearConstraint) or type(c.role)!=int or not 0<=c.role<roles or type(c.gold)!=int or type(c.rival)!=int or not 0<=c.gold<classes or not 0<=c.rival<classes or c.gold==c.rival
            or c.hidden.shape!=(warm_w.shape[1],) or c.hidden.device!=warm_w.device or c.hidden.dtype!=warm_w.dtype or c.hidden.requires_grad or not torch.isfinite(c.hidden).all()
            or not math.isfinite(c.floor) or c.floor<0 or not c.source_id or type(c.byte_offset)!=int or c.byte_offset<0):raise ValueError('finite TRAIN loss-side constraint')
        if float(_gap(warm_w,warm_b,c,classes))<c.floor:raise ValueError('warm point violates immutable active anchor; no reinitialization/weakening')
    dw=(proposal_w-warm_w).double();db=(proposal_b-warm_b).double();proposal_norm=float(torch.sqrt(dw.square().sum()+db.square().sum()))
    receipt=dict(method='bounded_actual_delta_halfspace_correction',passes_cap=2,active_constraint_cap=16,active_constraints=len(constraints),relative_radius=float(relative_radius),padding=padding,
        proposal_norm=proposal_norm,corrections=0,full_reference_calls=0,anchor_SHA=next(iter(anchor)),optimizer_steps_in_primitive=0,global_reference_pass=False)
    def fail(reason):return dict(weight=None,bias=None,receipt=dict(receipt,status=reason))
    if proposal_norm==0:return fail('zero_actual_proposal_no_fabricated_update')
    w=proposal_w.detach().clone();b=proposal_b.detach().clone()
    for sweep in range(passes):
        for c in constraints:
            gap=float(_gap(w,b,c,classes));needed=c.floor+padding-gap
            if needed<=0:continue
            # Minimum Euclidean correction for THIS single affine inequality:
            # ||a||^2=2*(||h||^2+1), including both independent class biases.
            denominator=2*(float(c.hidden.double().square().sum())+1.);tau=needed/denominator;g=c.role*classes+c.gold;j=c.role*classes+c.rival
            w[g].add_(c.hidden,alpha=tau);w[j].add_(c.hidden,alpha=-tau);b[g].add_(tau);b[j].add_(-tau);receipt['corrections']+=1
            distance=float(torch.sqrt((w-proposal_w).double().square().sum()+(b-proposal_b).double().square().sum()));receipt['correction_norm']=distance
            if distance>relative_radius*proposal_norm:return fail('fixed_correction_radius_exceeded')
        if all(float(_gap(w,b,c,classes))>=c.floor for c in constraints):
            receipt['passes_used']=sweep+1;break
    else:return fail('bounded_active_sweeps_not_converged')
    actual_w=(w-warm_w).double();actual_b=(b-warm_b).double();alignment=float((actual_w*dw).sum()+(actual_b*db).sum());receipt['proposal_alignment']=alignment
    if alignment<=0:return fail('no_forward_component_of_actual_proposal')
    expected_w=w.clone();expected_b=b.clone();decision=full_reference(w,b);receipt['full_reference_calls']=1
    if not torch.equal(w,expected_w) or not torch.equal(b,expected_b):raise ValueError('reference callback mutated corrected candidate')
    if not isinstance(decision,TrainDecision) or type(decision.accept) is not bool or not decision.measurements:raise ValueError('authoritative full TRAIN decision required')
    measurements={k:float(v) for k,v in decision.measurements.items()}
    if any(not isinstance(k,str) or not math.isfinite(v) for k,v in measurements.items()):raise ValueError('finite full-reference measurements')
    receipt['full_reference_measurements']=measurements
    if not decision.accept:return fail('full_reference_rejected_active_only_repair')
    receipt.update(status='bounded_corrected_candidate_full_reference_PASS',global_reference_pass=True)
    return dict(weight=w,bias=b,receipt=receipt)
