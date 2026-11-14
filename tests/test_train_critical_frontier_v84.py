import json,pytest,torch
from research.train_critical_frontier_v84 import select

def fixture():
    return dict(source_id='TRAIN:a',labels=torch.zeros(3,16,dtype=torch.long),critical_mask=torch.zeros(3,16,dtype=torch.bool),
        teacher_correct=torch.ones(3,16,dtype=torch.bool),teacher_margin=torch.full((3,16),3.))

def call(p,**kwargs):return select(p,p['source_id'],{'TRAIN:a','TRAIN:b'},precision_policy='FP32',expected_precision_policy='FP32',**kwargs)

def test_covers_all_weak_critical_patches_and_excludes_strong_noncritical_PAD_wrong():
    p=fixture();p['critical_mask'][:]=True;p['teacher_margin'][0,4]=.1;p['teacher_margin'][2,9]=.7;p['teacher_margin'][1,1]=.01;p['teacher_correct'][1,1]=False
    p['labels'][1,2]=257;p['teacher_margin'][1,2]=.01;p['critical_mask'][0,5]=False;p['teacher_margin'][0,5]=.01
    r=call(p);assert r['patch_indices']==[0,2] and r['critical_positions']==[[0,4],[2,9]] and r['initial_wrong_critical_bytes']==1
    assert json.loads(json.dumps(r))==r

def test_rejects_DEVELOPMENT_and_precision_mismatch():
    p=fixture();p['source_id']='DEV:a'
    with pytest.raises(ValueError):call(p)
    p['source_id']='TRAIN:a'
    with pytest.raises(ValueError):select(p,'TRAIN:a',{'TRAIN:a'},precision_policy='historical_BF16',expected_precision_policy='FP32')

def test_cap_is_not_silent_truncation_or_unlimited_growth():
    p=fixture();p['critical_mask'][:,0]=True;p['teacher_margin'][:,0]=.1
    with pytest.raises(OverflowError):call(p,max_patches=2)
    assert len(call(p,max_patches=3)['patch_indices'])==3
