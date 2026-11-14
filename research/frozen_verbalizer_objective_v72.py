"""Prospective decoder-only phase: complete expression CE + train-only anchor.

No inference filter, source feature, new memory or trainable module. Numeric
spans below are offline syntactic heuristics, not a semantic extraction system.
"""
import re
import torch
from research.fuzzy_factual_objective_v70 import cycle_losses as factual_losses

OBJECTIVE='V70_complete_expression_CE+.05_Bernoulli_KL_on_initially_correct_noncritical_TRAIN_positions'
NUMERICAL=re.compile(r'(?<![A-Za-z0-9_])(?:[-+]\s*)?(?:\$\s*)?\d+(?:[.,:/-]\d+)*(?:[ ]*(?:degrees [CF]|minutes?|seconds?|hours?|kg|km|cm|mm|ml|am|pm|%|g|m|L|K|M|B))?(?![A-Za-z0-9_])',re.I)

def complete_expression_spans(text,name_spans=()):
    spans=[]
    for match in NUMERICAL.finditer(text):
        a,b=match.span();spans.append(dict(start=len(text[:a].encode()),end=len(text[:b].encode()),kind='number_expression',text=match.group()))
    spans.extend(dict(s) for s in name_spans if s.get('kind')=='name')
    spans.sort(key=lambda s:(s['start'],s['end']))
    for a,b in zip(spans,spans[1:]):
        if a['end']>b['start']:raise ValueError('overlapping offline critical expressions')
    return spans

def anchor_loss(logits,targets,risk_mask,teacher_gold_probability,teacher_correct):
    if teacher_gold_probability.shape!=targets.shape or teacher_correct.shape!=targets.shape or teacher_correct.dtype!=torch.bool:raise ValueError('TRAIN-only cached teacher shape')
    p0=teacher_gold_probability.detach().float()
    if not torch.isfinite(p0).all() or ((p0<0)|(p0>1)).any():raise ValueError('teacher probability')
    eligible=(targets<256)&~risk_mask&teacher_correct
    if not eligible.any():return logits.float().sum()*0.,eligible
    z=logits.float();normalizer=torch.logsumexp(z,-1);gold=z.gather(-1,targets[...,None]).squeeze(-1)-normalizer
    exclude=torch.nn.functional.one_hot(targets,258).bool();rest=torch.logsumexp(z.masked_fill(exclude,-torch.inf),-1)-normalizer
    entropy=torch.special.xlogy(p0,p0)+torch.special.xlogy(1-p0,1-p0)
    kl=entropy-p0*gold-(1-p0)*rest
    return kl[eligible].mean(),eligible

def decoder_losses(logits,targets,weights,risk_mask,teacher_gold_probability,teacher_correct):
    base,parts=factual_losses(logits,targets,weights,risk_mask)
    kl,eligible=anchor_loss(logits,targets,risk_mask,teacher_gold_probability,teacher_correct)
    parts.update(correct_noncritical_anchor_KL=kl,anchor_positions=int(eligible.sum()))
    return base+.05*kl,parts

def freeze_extraction_and_binding(model):
    model.requires_grad_(False);model.readout.readout.requires_grad_(True)
    names=[n for n,p in model.named_parameters() if p.requires_grad]
    if len(names)!=7 or any(not n.startswith('readout.readout.') for n in names) or model.readout.role_projection.weight.requires_grad:
        raise ValueError('exact frozen Commit/binding boundary, seven final-readout tensors')
    return names
