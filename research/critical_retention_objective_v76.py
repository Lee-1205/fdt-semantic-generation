"""Prospective bounded wrong-critical CE; memory/inference paths unchanged.

Only the extra wrong-critical CE denominator changes from V75. Labels and
teacher eligibility are TRAIN loss-only inputs. This module is not a worker.
"""
import torch
from research.critical_retention_objective_v75 import bernoulli_anchor,COEFFICIENTS
from research.fuzzy_cycle_objective_v67 import cycle_losses as base_losses
from research.frozen_verbalizer_objective_v72 import freeze_extraction_and_binding

MAX_POSITION_AMPLIFICATION=8.
OBJECTIVE='V75_wrongCriticalCE_denominator_max(W,C/8);other_terms_unchanged'

def bounded_wrong_mean(ce,critical,wrong):
    if critical.dtype!=torch.bool or wrong.dtype!=torch.bool or critical.shape!=ce.shape or wrong.shape!=ce.shape or (wrong&~critical).any():
        raise ValueError('V76 critical/wrong subset')
    c=int(critical.sum());w=int(wrong.sum())
    denominator=max(float(w),float(c)/MAX_POSITION_AMPLIFICATION)
    value=ce[wrong].sum()/denominator if w else ce.sum()*0.
    return value,dict(wrong_critical_denominator=denominator,wrong_critical_positions=w,
        wrong_critical_position_amplification=(c/denominator if w else 0.))

def decoder_losses(logits,targets,weights,risk_mask,teacher_gold_probability,teacher_correct):
    if (risk_mask.shape!=targets.shape or risk_mask.dtype!=torch.bool or teacher_correct.dtype!=torch.bool or teacher_correct.shape!=targets.shape
        or teacher_gold_probability.shape!=targets.shape or logits.shape!=(*targets.shape,258) or (risk_mask&(targets>=256)).any()):
        raise ValueError('V76 TRAIN loss-only packet shape/mask')
    p0=teacher_gold_probability.detach().float()
    if not torch.isfinite(p0).all() or ((p0<0)|(p0>1)).any():raise ValueError('teacher probability bounds')
    correct=teacher_correct.detach();critical_correct=risk_mask&correct;critical_wrong=risk_mask&~correct
    noncritical_correct=(targets<256)&~risk_mask&correct
    base,parts=base_losses(logits,targets,weights)
    ce=torch.nn.functional.cross_entropy(logits.float().transpose(1,2),targets,ignore_index=257,reduction='none')
    mean=lambda mask:ce[mask].mean() if mask.any() else logits.float().sum()*0.
    wrong,stats=bounded_wrong_mean(ce,risk_mask,critical_wrong)
    terms=dict(correct_critical_CE=mean(critical_correct),wrong_critical_CE=wrong,
        correct_noncritical_KL=bernoulli_anchor(logits,targets,p0,noncritical_correct),
        correct_critical_KL=bernoulli_anchor(logits,targets,p0,critical_correct))
    loss=base+sum(COEFFICIENTS[k]*v for k,v in terms.items())
    parts.update(terms);parts.update(stats);parts.update(correct_critical_positions=int(critical_correct.sum()),
        noncritical_anchor_positions=int(noncritical_correct.sum()),critical_positions=int(risk_mask.sum()),
        critical_errors=int(((logits.argmax(-1)!=targets)&risk_mask).sum()))
    return loss,parts
