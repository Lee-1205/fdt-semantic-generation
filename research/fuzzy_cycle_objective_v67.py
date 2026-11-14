"""Whole-window risk: rare byte errors and false EOS in the neural tail."""
import math
import torch
from research.fuzzy_cycle_objective_v65 import training_weights as balanced_weights

OBJECTIVE='balanced_CE+.1_true_EOS+.1_top1pct_body_cap64+.1_false_EOS_tail_UL'

def training_weights(rows):
    lookup,report=balanced_weights(rows)
    report.update(objective=OBJECTIVE,hard_body_fraction=.01,hard_body_cap=64,hard_body_coefficient=.1,false_tail_EOS_coefficient=.1)
    return lookup,report

def cycle_losses(logits,targets,weights):
    z=logits.float();active=targets!=257;body=targets<256
    if not active.any() or not body.any() or int((targets==256).sum())!=1:raise ValueError('one full supplied window and one EOS')
    ce=torch.nn.functional.cross_entropy(z.transpose(1,2),targets,ignore_index=257,reduction='none')
    weighted=(ce*weights[targets]).sum()/active.sum();uniform=ce.sum()/active.sum();eos=ce[targets==256].mean()
    body_losses=ce[body];k=min(64,max(1,math.ceil(.01*len(body_losses))))
    hard=torch.topk(body_losses,k).values.mean()
    tail=torch.zeros_like(body);tail[-2:]=True;tail &= body
    # Stable -log(1-p(EOS)), no clipping that would hide high-confidence errors.
    not_eos=torch.cat((z[...,:256],z[...,257:]),-1)
    ul=(torch.logsumexp(z,-1)-torch.logsumexp(not_eos,-1))[tail].mean()
    loss=weighted+.1*eos+.1*hard+.1*ul
    pred=z.argmax(-1)
    return loss,dict(byte_CE=uniform,weighted_byte_CE=weighted,EOS_CE=eos,hard_body_CE=hard,tail_false_EOS_UL=ul,
        hard_k=k,tail_body_positions=int(tail.sum()),body_errors=int(((pred!=targets)&body).sum()),tail_false_EOS=int(((pred==256)&tail).sum()))
