"""Bounded transaction around an ACTUAL two-tensor AdamW proposal.

Candidate primitive, not a runnable FDT driver or proof on unseen meaning.
Caller owns finite row/group cursor, exposure, RNG, calls, phase and STOP state.
It unscales/clips once first. References MUST be TRAIN-only, deterministic,
no-grad, and read the current two parameters; no held-out acceptance selector.
Accepted damping keeps the full proposal's Adam moments (defined algorithm).
Full rejection restores weights AND Adam; a finite-attempt scaler update stays.
Exception rolls weights/Adam/scaler back and clears gradients; consumed source
and RNG are NOT replayed. CPU enabled-scaler tests do not prove CUDA recovery.
"""
import copy,math
from dataclasses import dataclass
import torch
from research.last_linear_objective_v79 import LAST_LINEAR_NAMES


@dataclass(frozen=True)
class TrainDecision:
    accept: bool
    measurements: dict


def _equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return a==b


def bounded_adam_proposal(optimizer,named_parameters,scaler,train_reference,*,alphas=(1.,.5,.25,.125),guard=lambda:None):
    """Apply at most one Adam proposal and bounded candidate-reference checks.

    No learning-rate/call/deadline resets, repeated gradient computation, extra
    optimizer steps, or retry after an exception. References consume a fixed
    budget in the eventual driver; this primitive cannot certify that driver.
    """
    if list(named_parameters)!=list(LAST_LINEAR_NAMES) or not isinstance(optimizer,torch.optim.AdamW):raise ValueError('exact lastLinear2 AdamW scope')
    params=list(named_parameters.values());owned=[v for g in optimizer.param_groups for v in g['params']]
    if len(owned)!=2 or {id(p) for p in params}!={id(p) for p in owned} or any(not p.requires_grad for p in params):raise ValueError('optimizer ownership')
    if params[0].ndim!=2 or params[1].ndim!=1 or len(params[0])!=len(params[1]) or len({p.untyped_storage().data_ptr() for p in params})!=2:raise ValueError('independent linear tensors')
    if not alphas or any(type(a) not in (float,int) or not math.isfinite(a) or not 0<a<=1 for a in alphas) or any(a<=b for a,b in zip(alphas,alphas[1:])):raise ValueError('fixed descending positive damping schedule')
    before=[p.detach().clone() for p in params];old_adam=copy.deepcopy(optimizer.state_dict());old_scaler=copy.deepcopy(scaler.state_dict());old_scale=scaler.get_scale();checks=[]

    def reset_weights():
        with torch.no_grad():
            for p,v in zip(params,before):p.copy_(v)

    def rollback(include_scaler):
        reset_weights();optimizer.load_state_dict(old_adam)
        if include_scaler:
            # Public update resets per-optimizer AMP stage before restoring the
            # serialized scale/growth tracker. No private AMP state edit.
            scaler.update(new_scale=float(old_scale));scaler.load_state_dict(old_scaler)
        optimizer.zero_grad(set_to_none=True)

    try:
        guard();scaler.step(optimizer);scaler.update()
        skipped=scaler.get_scale()<old_scale
        if skipped:
            if not all(torch.equal(p,v) for p,v in zip(params,before)) or not _equal(optimizer.state_dict(),old_adam):raise ValueError('AMP skip mutated weights/Adam')
            optimizer.zero_grad(set_to_none=True)
            return dict(accepted=False,AMP_skip=True,alpha=0.,reference_checks=[],Adam_proposals=1,Adam_steps_committed=0)
        proposal=[p.detach().clone() for p in params];proposal_adam=copy.deepcopy(optimizer.state_dict())
        if any(not torch.isfinite(v).all() for v in proposal):raise FloatingPointError('nonfinite actual Adam proposal')
        delta=[v-b for v,b in zip(proposal,before)]
        for alpha in alphas:
            guard()
            with torch.no_grad():
                for p,b,d in zip(params,before,delta):p.copy_(b+alpha*d)
            expected=[p.detach().clone() for p in params]
            with torch.no_grad():decision=train_reference()
            if not all(torch.equal(p,v) for p,v in zip(params,expected)) or not _equal(optimizer.state_dict(),proposal_adam):raise ValueError('reference mutated proposed weights/Adam')
            if not isinstance(decision,TrainDecision) or type(decision.accept) is not bool or not decision.measurements:raise ValueError('explicit TRAIN decision receipt')
            measurements={k:float(v) for k,v in decision.measurements.items()}
            if any(not isinstance(k,str) or not math.isfinite(v) for k,v in measurements.items()):raise FloatingPointError('nonfinite TRAIN reference')
            checks.append(dict(alpha=float(alpha),accept=decision.accept,measurements=measurements));guard()
            if decision.accept:
                optimizer.zero_grad(set_to_none=True)
                return dict(accepted=True,AMP_skip=False,alpha=float(alpha),reference_checks=checks,Adam_proposals=1,Adam_steps_committed=1,
                    Adam_moments='full proposal retained; damping applies to actual total weight delta incl decay/momentum')
        rollback(False)
        return dict(accepted=False,AMP_skip=False,alpha=0.,reference_checks=checks,Adam_proposals=1,Adam_steps_committed=0,
            rejection='all fixed candidates rejected; weights/Adam restored, finite-attempt scaler advanced; source/RNG/cursor remain consumed')
    except BaseException:
        rollback(True)
        raise
