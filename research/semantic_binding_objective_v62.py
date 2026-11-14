"""Align actual bound memory, excluding decoded factor probabilities.

The V61 inference architecture and all three banks are retained unchanged.
Only loss and a newly bounded finite training curriculum change.
"""
import torch
from torch.nn import functional as F
from research.semantic_binding_v61 import training_loss as v61_loss

OBJECTIVE='memory_only_contrast_span2_inside05'


def memory_contrast(role_values, meaning_ids):
    z=F.normalize(F.normalize(role_values.float(),dim=-1,eps=1e-8).flatten(1),dim=-1,eps=1e-8)
    similarity=z@z.T/.2
    eye=torch.eye(len(z),device=z.device,dtype=torch.bool)
    positives=(meaning_ids[:,None]==meaning_ids[None])&~eye
    logp=similarity-similarity.masked_fill(eye,-torch.inf).logsumexp(-1,keepdim=True)
    eligible=positives.any(-1)
    return (-(logp.masked_fill(~positives,0).sum(-1)/positives.sum(-1).clamp_min(1))[eligible].mean()
            if eligible.any() else z.sum()*0)


def training_loss(commit,raw,targets,meaning_ids):
    _,parts=v61_loss(commit,raw,targets,meaning_ids)
    parts=dict(parts,contrast=memory_contrast(commit.role_values,meaning_ids))
    loss=(2*parts['pointer']+parts['relation']+parts['polarity']+2*parts['quantity']
          +.5*parts['interior']+.2*parts['contrast'])
    return loss,parts
