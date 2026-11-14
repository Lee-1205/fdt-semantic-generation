"""Actual full-cycle content fidelity, no auxiliary fact head or runtime labels."""
import torch
from research.fuzzy_cycle_objective_v67 import cycle_losses as base_losses,training_weights as base_weights
OBJECTIVE='V67_cycle_losses+.25_TRAIN_name_numeral_span_CE'

def training_weights(rows):
    weights,report=base_weights(rows);report.update(objective=OBJECTIVE,critical_CE_coefficient=.25,critical_masks_TRAIN_only=True)
    return weights,report

def critical_mask(targets,spans,source_bytes):
    mask=torch.zeros_like(targets,dtype=torch.bool);flat=mask.flatten()
    for span in spans:
        a,b=span['start'],span['end']
        if not 0<=a<b<=source_bytes:raise ValueError('TRAIN span bounds')
        flat[a:b]=True
    if (mask&(targets>=256)).any():raise ValueError('critical loss must exclude EOS/PAD')
    return mask

def cycle_losses(logits,targets,weights,mask):
    if mask.dtype!=torch.bool or mask.shape!=targets.shape or (mask&(targets>=256)).any():raise ValueError('TRAIN critical mask')
    base,parts=base_losses(logits,targets,weights)
    ce=torch.nn.functional.cross_entropy(logits.float().transpose(1,2),targets,ignore_index=257,reduction='none')
    risk=ce[mask].mean() if mask.any() else logits.float().sum()*0.
    parts.update(critical_CE=risk,critical_positions=int(mask.sum()),critical_errors=int(((logits.argmax(-1)!=targets)&mask).sum()))
    return base+.25*risk,parts
