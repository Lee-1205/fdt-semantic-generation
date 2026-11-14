"""TRAIN-only competitor-margin retention, unchanged Commit-only inference.

New margins are loss targets, never decoder features. One prospective setting,
not a claim of optimal coefficients. Existing capped correction remains.
"""
import torch
from research.critical_retention_objective_v76 import decoder_losses as capped_losses
from research.update_balance_objective_v77 import frozen_objective_report as previous_report

OBJECTIVE='V77+correct_TRAIN_weak_margin_retention+same_update_U_S_O_U_S'
WEAK_TEACHER_MARGIN=3.
RETAIN_MARGIN_CAP=1.
MARGIN_COEFFICIENTS=dict(critical_competitor_margin=.25,noncritical_competitor_margin=.10)
GROUP_ROW_WEIGHTS=(.2,.2,.2,.2,.2)


def gold_margin(logits,targets):
    if logits.shape!=(*targets.shape,258) or targets.dtype!=torch.long or ((targets<0)|(targets>257)).any():
        raise ValueError('byte/EOS/PAD margin shape')
    z=logits.float();gold=z.gather(-1,targets[...,None]).squeeze(-1)
    other=z.masked_fill(torch.nn.functional.one_hot(targets,258).bool(),-torch.inf)
    maximum,competitor=other.max(-1)
    return gold-maximum,competitor


def margin_retention(logits,targets,risk_mask,teacher_correct,teacher_margin):
    if (teacher_margin.shape!=targets.shape or risk_mask.shape!=targets.shape or teacher_correct.shape!=targets.shape
        or teacher_correct.dtype!=torch.bool or risk_mask.dtype!=torch.bool or not torch.isfinite(teacher_margin).all()):
        raise ValueError('TRAIN margin packet shapes/types')
    m0=teacher_margin.detach().float();correct=teacher_correct.detach()
    if ((m0<0)&correct).any() or ((m0>0)&~correct).any():raise ValueError('teacher correctness/margin contradict')
    margin,_=gold_margin(logits,targets)
    eligible=(targets<256)&correct&(m0<=WEAK_TEACHER_MARGIN)
    target=m0.clamp(0,RETAIN_MARGIN_CAP)
    # ReLU has zero derivative at equality; clamp_min would push margins even
    # when the saved teacher boundary is already exactly retained.
    violation=torch.nn.functional.relu(target-margin)
    critical=eligible&risk_mask;other=eligible&~risk_mask
    mean=lambda mask:violation[mask].mean() if mask.any() else logits.float().sum()*0.
    terms=dict(critical_competitor_margin=mean(critical),noncritical_competitor_margin=mean(other))
    stats=dict(critical_margin_eligible=int(critical.sum()),noncritical_margin_eligible=int(other.sum()),
        critical_margin_violations=int((critical&(violation>0)).sum()),noncritical_margin_violations=int((other&(violation>0)).sum()))
    return sum(MARGIN_COEFFICIENTS[k]*v for k,v in terms.items()),dict(**terms,**stats)


def decoder_losses(logits,targets,weights,risk_mask,teacher_gold_probability,teacher_correct,teacher_margin):
    loss,parts=capped_losses(logits,targets,weights,risk_mask,teacher_gold_probability,teacher_correct)
    retention,extra=margin_retention(logits,targets,risk_mask,teacher_correct,teacher_margin)
    parts.update(extra)
    return loss+retention,parts


def frozen_objective_report(current,profile,group_steps=384):
    pool_weights=current.get('current_unique_pool_class_weights_NOT_used',current['class_weights'])
    report=previous_report(current,profile)
    report['current_unique_pool_class_weights_NOT_used']=list(pool_weights)
    report.update(objective=OBJECTIVE,margin_coefficients=MARGIN_COEFFICIENTS,weak_teacher_margin=WEAK_TEACHER_MARGIN,
        retained_margin_cap=RETAIN_MARGIN_CAP,margin_source='TRAIN initial head ONLY, detached strongest-nongold margin and class',
        group_rows=5,group_weights=list(GROUP_ROW_WEIGHTS),group_order='unsigned,signed,OWT,unsigned,signed',
        optimizer_steps=group_steps,source_head_rows=5*group_steps,group_gradient='sum(.2*row_loss gradients) at SAME parameters, unscale/clip1/Adam ONCE per group')
    return report
