"""TRAIN-only reference criterion candidate, not an execution/training driver.

V80 protected top1 but allowed all positive margin to vanish. This criterion
also keeps a bounded fraction of the initial FP32 decision margin. References
must be re-extracted under the SAME fixed precision as future deployment,
cover TRAIN data only, and be kept separate from normal neural features.
It cannot certify positions outside its finite reference bank.
"""
import torch
from research.actual_update_retention_v80 import TrainDecision

@torch.no_grad()
def margins(logits,labels):
    if logits.shape[:-1]!=labels.shape or logits.shape[-1]!=258 or labels.dtype!=torch.long:raise ValueError('byte/EOS/PAD label shape')
    if logits.dtype!=torch.float32 or not torch.isfinite(logits).all() or ((labels<0)|(labels>257)).any():raise ValueError('finite FP32 projection and labels required')
    selected=logits.gather(-1,labels[...,None]).squeeze(-1)
    other=logits.clone();other.scatter_(-1,labels[...,None],-torch.inf)
    return selected-other.max(-1).values

@torch.no_grad()
def reference(initial_logits,labels,*,fraction=.5,cap=1.):
    if not 0<fraction<=1 or not 0<cap:raise ValueError('explicit prospective margin policy')
    gap=margins(initial_logits,labels);protected=(initial_logits.argmax(-1)==labels)&(labels!=257)
    return dict(protected=protected,initial_margin=gap,required_margin=gap.clamp(min=0,max=cap)*fraction,labels=labels.clone(),fraction=fraction,cap=cap)

@torch.no_grad()
def decision(candidate_logits,bank):
    labels=bank['labels'];gap=margins(candidate_logits,labels);mask=bank['protected']
    flips=mask&(candidate_logits.argmax(-1)!=labels);erosion=mask&(gap<bank['required_margin'])
    return TrainDecision(not bool((flips|erosion).any()),dict(protected_positions=int(mask.sum()),new_top1_flips=int(flips.sum()),
        margin_floor_violations=int(erosion.sum()),initial_wrong_not_locked=int(((labels!=257)&~mask).sum())))
