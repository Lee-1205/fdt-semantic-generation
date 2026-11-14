"""Cover weak critical TRAIN positions instead of hash-only100 references.

CPU selection contract only, no model/optimizer/decoder feature path. Loss-side
targets select which frozen TRAIN prefix patches a future reference bank keeps.
The stored features and labels must remain separate. Every actual run rebuilds
this selection under its own parent and declared deployment arithmetic policy.
"""
import torch

def select(packet,source_id,train_ids,*,precision_policy,expected_precision_policy,max_margin=1.,max_patches=512):
    if source_id not in train_ids or packet.get('source_id')!=source_id:raise ValueError('TRAIN allowlist/source identity')
    if precision_policy!=expected_precision_policy:raise ValueError('teacher/reference/runtime precision mismatch')
    y=packet['labels'];mask=packet['critical_mask'];correct=packet['teacher_correct'];margin=packet['teacher_margin']
    if y.ndim!=2 or y.shape[-1]!=16 or y.dtype!=torch.long or any(v.shape!=y.shape for v in (mask,correct,margin)):raise ValueError('patch/byte shapes')
    if mask.dtype!=torch.bool or correct.dtype!=torch.bool or not margin.is_floating_point() or not torch.isfinite(margin).all() or ((y<0)|(y>257)).any():raise ValueError('loss-only schema/numeric')
    if not 0<max_margin or type(max_patches)!=int or max_patches<1:raise ValueError('prospective selection caps')
    selected=mask&correct&(y!=257)&(margin<=max_margin)
    patch_indices=selected.any(-1).nonzero().flatten().tolist()
    if len(patch_indices)>max_patches:raise OverflowError('fixed source patch cap; no silent truncation')
    positions=selected.nonzero().tolist()
    return dict(source_id=source_id,precision_policy=precision_policy,max_margin=float(max_margin),patch_indices=patch_indices,
        critical_positions=positions,selected_critical_bytes=len(positions),initial_wrong_critical_bytes=int((mask&~correct&(y!=257)).sum()),
        scope='TRAIN reference only; initial wrong positions remain learnable; no GT roles/spans as neural features')
