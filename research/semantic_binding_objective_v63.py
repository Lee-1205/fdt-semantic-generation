"""Loss-only role competition, conditional on annotated disjoint source spans.

Inference never receives target spans or a hard no-overlap rule. Nested or
coreferent annotated spans are not penalized by this auxiliary term.
"""
import torch
from torch.nn import functional as F
from research.semantic_binding_objective_v62 import training_loss as previous_loss

OBJECTIVE='memory_contrast_span2_inside05_disjoint_role05'


def role_overlap_loss(role_degree,spans):
    roles=F.normalize(role_degree[:,:3].float(),dim=-1,eps=1e-6)
    terms=[];masks=[]
    for r,s in ((0,1),(0,2),(1,2)):
        a,b=spans[:,r,0],spans[:,r,1];c,d=spans[:,s,0],spans[:,s,1]
        eligible=(a>=0)&(c>=0)&((b<=c)|(d<=a))
        terms.append((roles[:,r]*roles[:,s]).sum(-1).square());masks.append(eligible)
    value=torch.stack(terms,-1);mask=torch.stack(masks,-1)
    return (value*mask).sum()/mask.sum().clamp_min(1)


def training_loss(commit,raw,targets,meaning_ids):
    loss,parts=previous_loss(commit,raw,targets,meaning_ids)
    overlap=role_overlap_loss(raw['role_degree'],targets['spans'])
    return loss+.5*overlap,dict(parts,role_overlap=overlap)
