"""Reuse direct-parent serialization; distinguish 7 stored deltas from 2 Adam tensors.

V75's frozen five prefix tensors differ from the standalone V74 storage base.
They must be stored, even though they are NOT trainable. No recursive parent,
new full model copy, source-text Exact memory, or old source change is needed.
"""
from research import linked_runtime_v75 as storage
from research.last_linear_objective_v79 import LAST_LINEAR_NAMES
file_sha=storage.file_sha


def check(full,digest):
    names=full['optimizer_names']
    if names!=[list(LAST_LINEAR_NAMES)]:raise ValueError('two trainable optimizer tensor names')
    stored=full['storage_delta_names']
    if len(stored)!=7 or len(set(stored))!=7 or not set(LAST_LINEAR_NAMES)<=set(stored) or any(not n.startswith('readout.readout.') for n in stored):
        raise ValueError('seven explicitly stored readout deltas')
    prefix={k:full['model'][k] for k in stored if k not in LAST_LINEAR_NAMES}
    if digest(prefix)!=full['frozen_head_prefix_SHA']:raise ValueError('frozen readout prefix changed')
    if digest({k:v for k,v in full['model'].items() if k not in LAST_LINEAR_NAMES})!=full['frozen_except_last2_SHA']:
        raise ValueError('nontrainable model state changed')
    return stored


def pack(full,parent_model,parent_reference,digest):
    names=check(full,digest)
    envelope=storage.pack(full,parent_model,parent_reference,names,digest)
    envelope['storage_scope']=dict(serialized_override_names=names,trainable_optimizer_names=list(LAST_LINEAR_NAMES),
        frozen_override_names=[n for n in names if n not in LAST_LINEAR_NAMES])
    return envelope


def unpack(envelope,allowed_root,digest):
    full=storage.unpack(envelope,allowed_root,digest);stored=check(full,digest)
    if envelope.get('storage_scope')!=dict(serialized_override_names=stored,trainable_optimizer_names=list(LAST_LINEAR_NAMES),
        frozen_override_names=[n for n in stored if n not in LAST_LINEAR_NAMES]):raise ValueError('storage/optimizer scope metadata')
    return full


def save_verified(path,envelope,allowed_root,digest):
    # The generic writer validates the original complete model/runtime digest.
    # The scope-specific checker validates the extra frozen/optimizer invariants.
    if envelope['runtime']['optimizer_names']!=[list(LAST_LINEAR_NAMES)] or len(envelope['active_names'])!=7:
        raise ValueError('two trained tensors versus seven stored overrides')
    result=storage.save_verified(path,envelope,allowed_root,digest)
    load_verified(path,result['SHA'],allowed_root,digest)
    return result


def load_verified(path,expected_SHA,allowed_root,digest):
    import torch
    if storage.file_sha(path)!=expected_SHA:raise ValueError('V79 runtime file SHA')
    return unpack(torch.load(path,map_location='cpu',weights_only=True),allowed_root,digest)
