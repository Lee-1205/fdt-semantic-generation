"""Synthetic CPU component contracts, not FDT ability or actual full driver."""
import copy,io
import pytest
import torch
from research.critical_reference_bank_v85 import CriticalReferenceBank
from research.readout_precision_policy_v81 import POLICY

IDS=[dict(id='TRAIN:'+str(i),SHA=str(i+1)*64) for i in range(3)]
IDENTITY=dict(embedding_SHA='a'*64,prefix_SHA='b'*64,parent_SHA='c'*64)

def bank(**kwargs):return CriticalReferenceBank(IDS,[0],**IDENTITY,**kwargs)

def packets(i,weak=True):
    y=torch.full((2,16),257,dtype=torch.long);y[0]=65;y[1,0]=256;valid=y!=257
    crit=torch.zeros_like(valid);crit[0,2]=True;m=torch.full((2,16),3.,dtype=torch.float32)
    if weak and i:m[0,2]=.4
    f=dict(hidden=torch.arange(1024,dtype=torch.float32).reshape(2,512)/1024,source_id=IDS[i]['id'],source_SHA=IDS[i]['SHA'],precision_policy=POLICY,**IDENTITY)
    t=dict(source_id=IDS[i]['id'],source_SHA=IDS[i]['SHA'],source_bytes=16,labels=y,critical_mask=crit,teacher_correct=valid,teacher_margin=m,precision_policy=POLICY)
    return f,t

def logits(t):
    y=t['labels'];m=t['teacher_margin'];z=torch.full((*y.shape,258),-10.)
    z[...,66]=0;z.scatter_(-1,y[...,None],m[...,None]);return z

def populated():
    b=bank()
    for i in range(3):b.ingest(i,*packets(i))
    f,t=b.components();z=logits(t);b.calibrate(z);return b,f,t,z

def equal(a,b):
    if torch.is_tensor(a):return a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,dict):return set(a)==set(b) and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def test_all_weak_sources_added_and_labels_never_feature_fields():
    b,f,t,z=populated();assert b.cursor==3 and b.patch_count==4 and b.weak_count==2
    assert set(f)=={'schema','hidden','source_ranges','metadata'} and 'labels' not in f and 'hidden' not in t
    assert f['source_ranges'][1]['source_patch_indices']==[0] and f['source_ranges'][2]['source_patch_indices']==[0]
    assert b.decision(z).accept

def test_critical_margin_preserved_without_freezing_all_ordinary_margins():
    b,f,t,z=populated();ordinary=z.clone();ordinary[0,5,65]=.01
    assert b.decision(ordinary).accept # generic full-bank margin floor would reject
    critical=z.clone();critical[2,2,65]=.05;d=b.decision(critical)
    assert not d.accept and d.measurements['weak_critical_margin_violations']==1 and d.measurements['new_critical_flips']==0
    flip=z.clone();flip[0,5,65]=-.01;assert not b.decision(flip).accept

def test_exact_partial_and_calibrated_restore_preserve_cursor_without_replay():
    rng=torch.get_rng_state().clone();a=bank();a.ingest(0,*packets(0));stream=io.BytesIO();torch.save(a.export(),stream);stream.seek(0)
    b=bank();b.restore(torch.load(stream,map_location='cpu',weights_only=True));assert equal(a.export(),b.export())
    with pytest.raises(ValueError):b.ingest(0,*packets(0))
    for i in (1,2):a.ingest(i,*packets(i));b.ingest(i,*packets(i))
    assert equal(a.export(),b.export());_,t=a.components();z=logits(t);a.calibrate(z);b.calibrate(z)
    c=bank();c.restore(b.export());assert equal(c.export(),a.export()) and c.decision(z).accept and torch.equal(torch.get_rng_state(),rng)
    corrupted=b.export();corrupted['calibration']['required_margin'].zero_()
    with pytest.raises(ValueError):bank().restore(corrupted)

def test_rejects_source_precision_epoch_and_hidden_grad_features():
    for kind in ('DEV','precision','epoch','grad'):
        b=bank();f,t=packets(0)
        if kind=='DEV':f['source_id']='DEV:0'
        if kind=='precision':f['precision_policy']='historical_BF16'
        if kind=='epoch':f['prefix_SHA']='d'*64
        if kind=='grad':f['hidden'].requires_grad_(True)
        with pytest.raises(ValueError):b.ingest(0,f,t)
        assert b.cursor==0

def test_caps_and_closed_phase_do_not_expand_or_rewind():
    b=bank(max_patches=1)
    with pytest.raises(OverflowError):b.ingest(0,*packets(0))
    assert b.cursor==b.patch_count==0 and b.parts==[]
    b=bank();b.ingest(0,*packets(0))
    with pytest.raises(ValueError):b.components()
    b,f,t,z=populated()
    with pytest.raises(ValueError):b.ingest(3,*packets(0))
    with pytest.raises(ValueError):b.calibrate(z)

def test_precision_and_row_batch_disagreement_fail_closed():
    b=bank()
    for i in range(3):b.ingest(i,*packets(i))
    _,t=b.components();z=logits(t)
    with pytest.raises(ValueError):b.calibrate(z.bfloat16())
    drift=z.clone();drift[0,0,65]+=.01
    with pytest.raises(ValueError):b.calibrate(drift)
    with pytest.raises(ValueError):b.calibrate(z,max_row_batch_margin_difference=1.)
    wrong=z.clone();wrong[0,0,65]=-1
    with pytest.raises(ValueError):b.calibrate(wrong)
