"""TRAIN-only separated correction and retention, inference is unchanged.

The saved teacher p(gold) is a Bernoulli summary, not a full-distribution
distillation target or a semantic oracle. Incorrect teacher predictions never
become output labels. Correct critical and noncritical anchors are separately
normalized, so abundant ordinary bytes cannot dilute the critical anchor.
"""
import torch
from research.fuzzy_cycle_objective_v67 import cycle_losses as base_losses
from research.frozen_verbalizer_objective_v72 import freeze_extraction_and_binding

OBJECTIVE='V67+.10_initialCorrectCriticalCE+.25_initialWrongCriticalCE+.10_correctNoncriticalBernoulliKL+.25_correctCriticalBernoulliKL'
COEFFICIENTS=dict(correct_critical_CE=.10,wrong_critical_CE=.25,correct_noncritical_KL=.10,correct_critical_KL=.25)

def bernoulli_anchor(logits,targets,p0,eligible):
    if not eligible.any():return logits.float().sum()*0.
    z=logits.float();normalizer=torch.logsumexp(z,-1)
    gold=z.gather(-1,targets[...,None]).squeeze(-1)-normalizer
    exclude=torch.nn.functional.one_hot(targets,258).bool()
    other=torch.logsumexp(z.masked_fill(exclude,-torch.inf),-1)-normalizer
    entropy=torch.special.xlogy(p0,p0)+torch.special.xlogy(1-p0,1-p0)
    return (entropy-p0*gold-(1-p0)*other)[eligible].mean()

def decoder_losses(logits,targets,weights,risk_mask,teacher_gold_probability,teacher_correct):
    if (risk_mask.shape!=targets.shape or risk_mask.dtype!=torch.bool or teacher_correct.dtype!=torch.bool or teacher_correct.shape!=targets.shape
        or teacher_gold_probability.shape!=targets.shape or logits.shape!=(*targets.shape,258) or (risk_mask&(targets>=256)).any()):raise ValueError('V75 TRAIN loss-only packet shape/mask')
    p0=teacher_gold_probability.detach().float()
    if not torch.isfinite(p0).all() or ((p0<0)|(p0>1)).any():raise ValueError('teacher probability bounds')
    correct=teacher_correct.detach();critical_correct=risk_mask&correct;critical_wrong=risk_mask&~correct
    noncritical_correct=(targets<256)&~risk_mask&correct
    base,parts=base_losses(logits,targets,weights)
    ce=torch.nn.functional.cross_entropy(logits.float().transpose(1,2),targets,ignore_index=257,reduction='none')
    mean=lambda mask:ce[mask].mean() if mask.any() else logits.float().sum()*0.
    terms=dict(correct_critical_CE=mean(critical_correct),wrong_critical_CE=mean(critical_wrong),
        correct_noncritical_KL=bernoulli_anchor(logits,targets,p0,noncritical_correct),correct_critical_KL=bernoulli_anchor(logits,targets,p0,critical_correct))
    loss=base+sum(COEFFICIENTS[k]*v for k,v in terms.items())
    parts.update(terms);parts.update(correct_critical_positions=int(critical_correct.sum()),wrong_critical_positions=int(critical_wrong.sum()),
        noncritical_anchor_positions=int(noncritical_correct.sum()),critical_positions=int(risk_mask.sum()),critical_errors=int(((logits.argmax(-1)!=targets)&risk_mask).sum()))
    return loss,parts
