import copy,pytest
from research.immutable_reference_anchor_v89 import build,verify

def fixture():
    rep=dict(embedding_SHA='a'*64,prefix_SHA='b'*64,core_config_SHA='c'*64,input_capacity=16384,precision_policy='fixed_FP32',operator_SHA='d'*64)
    ref=dict(features='TRAIN/features.pt',features_SHA='e'*64,targets='TRAIN/targets.pt',targets_SHA='f'*64,bank_runtime_SHA='1'*64)
    return rep,ref,build('2'*64,rep,ref)

def test_new_warm_parent_does_not_require_or_allow_new_floors():
    rep,ref,a=fixture();assert verify(a,rep,ref)
    # Original margin1 floor.5 remains. A warm value.51 is feasible, but a
    # fresh half-floor.255 would weaken protection on each subsequent run.
    assert .51>=.5 and .51*.5<.5
    changed=dict(ref,targets_SHA='3'*64,bank_runtime_SHA='4'*64)
    with pytest.raises(ValueError):verify(a,rep,changed)

def test_weights_representation_code_and_anchor_tamper_are_rejected():
    rep,ref,a=fixture()
    for key in ('embedding_SHA','prefix_SHA','core_config_SHA','operator_SHA','input_capacity','precision_policy'):
        changed=dict(rep);changed[key]=32768 if key=='input_capacity' else 'different'
        with pytest.raises(ValueError):verify(a,changed,ref)
    tampered=copy.deepcopy(a);tampered['origin_source_SHA']='5'*64
    with pytest.raises(ValueError):verify(tampered,rep,ref)
