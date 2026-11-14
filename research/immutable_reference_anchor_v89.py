"""Reuse the ORIGINAL reference floor; a warmstart is not a new anchor.

Pure metadata/identity contract. Actual worker must verify model tensors and
operator code hashes before supplying these representation signatures.
"""
import hashlib,json

FIELDS={'embedding_SHA','prefix_SHA','core_config_SHA','input_capacity','precision_policy','operator_SHA'}

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def build(origin_source_SHA,representation,bank_reference):
    if not isinstance(origin_source_SHA,str) or len(origin_source_SHA)!=64 or set(representation)!=FIELDS:raise ValueError('explicit source and fixed representation identity')
    keys={'features','features_SHA','targets','targets_SHA','bank_runtime_SHA'}
    if set(bank_reference)!=keys or any(len(bank_reference[k])!=64 for k in ('features_SHA','targets_SHA','bank_runtime_SHA')):raise ValueError('immutable original bank files required')
    body=dict(schema='original_TRAIN_reference_anchor_v89',origin_source_SHA=origin_source_SHA,representation=dict(representation),bank_reference=dict(bank_reference))
    return dict(body,anchor_SHA=digest(body))

def verify(anchor,current_representation,reference):
    body={k:v for k,v in anchor.items() if k!='anchor_SHA'}
    if anchor['anchor_SHA']!=digest(body) or set(current_representation)!=FIELDS or current_representation!=anchor['representation']:raise ValueError('anchor/operator/frozen model mismatch')
    if reference!=anchor['bank_reference']:raise ValueError('reference reset/ratchet or file replacement forbidden')
    return True
