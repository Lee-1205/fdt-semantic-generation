"""Immutable, separate reference features/targets; exact bank reconstruction."""
import copy
from pathlib import Path
import torch
from research.critical_reference_bank_v85 import SCHEMA

PART_TENSORS=('labels','critical_mask','teacher_correct','teacher_margin','weak_critical_mask')

def packets(bank):
    features,targets=bank.components();targets=dict(targets,bank_runtime=dict(schema=SCHEMA,config=copy.deepcopy(bank.config),cursor=bank.cursor,
        patch_count=bank.patch_count,weak_count=bank.weak_count,calibration=copy.deepcopy(bank.calibration)))
    if bank.calibration is None:raise ValueError('one calibrated complete bank before immutable storage')
    return features,targets

def reconstruct(features,targets):
    if set(features)!={'schema','hidden','source_ranges','metadata'} or set(targets)!={'schema',*PART_TENSORS,'bank_runtime'} or features['schema']!=SCHEMA+'_features' or targets['schema']!=SCHEMA+'_loss_only':raise ValueError('separate reference schema')
    runtime=copy.deepcopy(targets['bank_runtime']);config=runtime['config'];h=features['hidden'];parts=[];end=0
    metadata={k:v for k,v in config.items() if k!='sources'}
    if metadata!=features['metadata'] or h.dtype!=torch.float32 or h.device.type!='cpu' or not torch.isfinite(h).all():raise ValueError('immutable FP32 epoch/config')
    for i,r in enumerate(features['source_ranges']):
        if r['source_index']!=i or r['start']!=end or not r['start']<=r['end']<=len(h) or r['end']-r['start']!=len(r['source_patch_indices']):raise ValueError('exact source ranges')
        start,end=r['start'],r['end'];identity=config['sources'][i]
        if identity!={'id':r['source_id'],'SHA':r['source_SHA']}:raise ValueError('source allowlist')
        parts.append(dict(source_index=i,source_id=r['source_id'],source_SHA=r['source_SHA'],source_patch_indices=list(r['source_patch_indices']),hidden=h[start:end].clone(),
            **{k:targets[k][start:end].clone() for k in PART_TENSORS}))
    if end!=len(h) or len(parts)!=runtime['cursor'] or runtime['patch_count']!=len(h):raise ValueError('complete bank cursor')
    runtime['parts']=parts;return runtime

def load(reference,allowed_root,sha):
    if set(reference)!={'features','features_SHA','targets','targets_SHA','bank_runtime_SHA'}:raise ValueError('strict immutable bank reference')
    root=Path(allowed_root).resolve()
    for key in ('features','targets'):
        p=Path(reference[key]).resolve()
        if not p.is_relative_to(root) or sha(p)!=reference[key+'_SHA']:raise ValueError('immutable reference path/SHA')
    return reconstruct(torch.load(reference['features'],map_location='cpu',weights_only=True),torch.load(reference['targets'],map_location='cpu',weights_only=True))
