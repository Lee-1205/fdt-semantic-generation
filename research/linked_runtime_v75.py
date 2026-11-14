"""CPU-only exact runtime storage with a mandatory pinned standalone parent.

This is checkpoint serialization, never inference-time Exact/text retrieval.
It saves all runtime state plus active NN tensors and references immutable
parent NN tensors. It is not a standalone fullstate or a V75 training driver.
"""
import copy,hashlib,os
from pathlib import Path
import torch

SCHEMA='fdt_parent_linked_runtime_v75_v1'

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def _path(path,allowed_root):
    p=Path(path).resolve();root=Path(allowed_root).resolve()
    if not p.is_relative_to(root):raise ValueError('linked checkpoint path outside allowed project root')
    return p

def _parent(reference,allowed_root,digest):
    if set(reference)!={'path','SHA','model_key','model_SHA'} or reference['model_key']!='model':raise ValueError('standalone parent identity schema')
    path=_path(reference['path'],allowed_root)
    if file_sha(path)!=reference['SHA']:raise ValueError('parent file SHA mismatch')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(state,dict) or not isinstance(state.get('model'),dict):raise ValueError('standalone model parent required; no linked parent recursion')
    model=state['model']
    if any(not torch.is_tensor(v) or v.device.type!='cpu' for v in model.values()):raise ValueError('CPU tensor model mapping required')
    if digest(model)!=reference['model_SHA']:raise ValueError('parent model digest mismatch')
    return model

def pack(full_state,parent_model,parent_reference,active_names,digest):
    names=list(active_names);model=full_state['model']
    if not names or len(names)!=len(set(names)) or set(model)!=set(parent_model) or set(names)-set(model):raise ValueError('active tensor ownership/model keys')
    for k,v in model.items():
        pv=parent_model[k]
        if not torch.is_tensor(v) or v.device.type!='cpu' or v.shape!=pv.shape or v.dtype!=pv.dtype:raise ValueError('CPU model shape/dtype mismatch')
        if k not in names and not torch.equal(v,pv):raise ValueError('nonactive tensor differs from pinned parent')
    if digest(parent_model)!=parent_reference['model_SHA']:raise ValueError('supplied parent model mismatch')
    frozen=digest({k:v for k,v in model.items() if k not in names})
    # Clone active tensors so a view cannot serialize a larger parent storage.
    return dict(schema=SCHEMA,parent_reference=copy.deepcopy(parent_reference),active_names=names,
        model_keys=list(model),active_weights={k:model[k].detach().clone() for k in names},
        runtime={k:v for k,v in full_state.items() if k!='model'},frozen_model_SHA=frozen,
        reconstructed_fullstate_SHA=digest(full_state),standalone=False)

def unpack(envelope,allowed_root,digest):
    if envelope['schema']!=SCHEMA or envelope['standalone'] is not False or 'model' in envelope['runtime']:raise ValueError('linked runtime schema')
    names=envelope['active_names'];active=envelope['active_weights'];parent=_parent(envelope['parent_reference'],allowed_root,digest)
    if (not names or len(names)!=len(set(names)) or set(active)!=set(names) or set(names)-set(parent)
        or len(envelope['model_keys'])!=len(parent) or set(envelope['model_keys'])!=set(parent)):raise ValueError('linked active model identity')
    if digest({k:v for k,v in parent.items() if k not in names})!=envelope['frozen_model_SHA']:raise ValueError('frozen parent model digest')
    model={}
    for k in envelope['model_keys']:
        v=active[k] if k in active else parent[k]
        if not torch.is_tensor(v) or v.device.type!='cpu' or v.shape!=parent[k].shape or v.dtype!=parent[k].dtype:raise ValueError('linked active tensor shape/dtype')
        model[k]=v
    state=dict(envelope['runtime'],model=model)
    if digest(state)!=envelope['reconstructed_fullstate_SHA']:raise ValueError('reconstructed fullstate digest mismatch')
    return state

def save_verified(path,envelope,allowed_root,digest):
    path=_path(path,allowed_root);partial=path.with_name(path.name+'.part')
    if path.exists() or partial.exists():raise FileExistsError('no checkpoint overwrite or automatic retry')
    path.parent.mkdir(parents=True,exist_ok=True)
    with partial.open('xb') as f:torch.save(envelope,f);f.flush();os.fsync(f.fileno())
    loaded=torch.load(partial,map_location='cpu',weights_only=True)
    unpack(loaded,allowed_root,digest)
    # Windows rename fails if a destination exists. No old file deletion.
    partial.rename(path)
    return dict(path=str(path),SHA=file_sha(path),bytes=path.stat().st_size,
        reconstructed_fullstate_SHA=envelope['reconstructed_fullstate_SHA'],standalone=False,
        mandatory_parent=envelope['parent_reference'])

def load_verified(path,expected_file_SHA,allowed_root,digest):
    path=_path(path,allowed_root)
    if file_sha(path)!=expected_file_SHA:raise ValueError('linked file SHA mismatch')
    return unpack(torch.load(path,map_location='cpu',weights_only=True),allowed_root,digest)
