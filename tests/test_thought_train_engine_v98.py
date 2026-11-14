"""Meaningful CPU exact restore, partial final group, AMP skip and exception tests."""
import io,random,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
import torch
from torch import nn
from research.commit_patch_readout_v54 import CommitPatchReadout
from research.memory_thought_diffusion_v94 import ThoughtConfig,InputMeaning,MemoryThoughtDiffusion,target_meaning
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.thought_train_engine_v98 import ThoughtTrainEngine,cpu_tree
from text_generation_support_v1 import state_sha


class Codec(nn.Module):
    def __init__(self):super().__init__();self.head=CommitPatchReadout(16,16)
    def verbalize(self,commit):return self.head.from_commit(commit)


def fixture(rows=17):
    torch.set_num_threads(2);torch.manual_seed(98113);random.seed(98);np.random.seed(98)
    c=ThoughtConfig(dimension=16,key_dimension=4,value_dimension=8,hidden=16,heads=2,layers=1,answer_local_capacity=5,source_slot_capacity=32,max_steps=4)
    codec=Codec();thought=MemoryThoughtDiffusion(c)
    m=InputMeaning(torch.randn(1,6,16),torch.randn(4,4),torch.randn(4,8),torch.randn(1,4,8),torch.randn(1,4,8),torch.zeros(1,1,4),torch.zeros(1,1,8),torch.zeros(1,1,dtype=torch.bool),'e',(('source','v','SHA'),))
    target,mask=target_meaning(DifferentiableCodecCommit.from_semantic(torch.randn(1,4,16),m.memory_revisions,'e'),codec.head.terminal,5)
    labels=torch.full((6,16),257,dtype=torch.long);labels.flatten()[:3]=torch.tensor([65,66,256])
    p=dict(schema='LOSS_ONLY_answer_meaning_v95',answer_SHA='lossSHA',source_codec_SHA='frozenSHA',semantic_target=target,valid_extent_mask=mask,labels=labels,model_epoch='e',target_condition_features_never_used=True)
    engine=ThoughtTrainEngine(thought,codec,'frozenSHA',[str(i) for i in range(rows)],source_metadata={'cache':'immutable'})
    return engine,m,p


def micro(e,m,p):
    random.random();np.random.random();e.micro(e.orders[e.runtime['cursor']],m,p)


def complete(e,m,p):
    while e.runtime['phase']!='DONE':
        if e.runtime['phase']=='IDLE':e.begin()
        while e.runtime['phase']=='MICROS':micro(e,m,p)
        e.finish()
    return e.payload()


def safeload(p):
    b=io.BytesIO();torch.save(p,b);b.seek(0);return torch.load(b,map_location='cpu',weights_only=True)


@pytest.mark.parametrize('boundary',['MICROS','READY','IDLE'])
def test_complete_fullstate_exact_next_update_allRNG_at_three_boundaries(boundary):
    e,m,p=fixture();e.begin()
    for _ in range(3 if boundary=='MICROS' else 8):micro(e,m,p)
    if boundary=='IDLE':e.finish()
    saved=safeload(e.payload());expected=complete(e,m,p)
    saved_digest=state_sha(saved)
    other,om,op=fixture();other.restore(saved);actual=complete(other,om,op)
    assert state_sha(saved)==saved_digest
    assert state_sha(expected)==state_sha(actual)
    assert actual['runtime']['attempts']==actual['runtime']['updates']==3 and actual['runtime']['cursor']==17
    assert [x['rows'] for x in actual['runtime']['history']]==[8,8,1] and actual['runtime']['exposure_bytes']==34
    assert actual['runtime']['calls']==dict(denoise=17,readout=17)


def test_enabled_scaler_overflow_consumes_rows_without_rewind_then_recovers():
    e,m,p=fixture(rows=9);e.begin()
    for _ in range(8):micro(e,m,p)
    before=state_sha(e.thought.state_dict());e.finish(force_overflow=True)
    assert state_sha(e.thought.state_dict())==before and e.runtime['cursor']==8 and e.runtime['AMP_skips']==1 and e.runtime['updates']==0
    assert e.scaler.get_scale()==512
    saved=safeload(e.payload());expected=complete(e,m,p)
    saved_digest=state_sha(saved)
    other,om,op=fixture(rows=9);other.restore(saved);actual=complete(other,om,op)
    assert state_sha(saved)==saved_digest
    assert state_sha(expected)==state_sha(actual) and actual['runtime']['updates']==1 and actual['runtime']['attempts']==2


def test_partial_micro_exception_restores_previous_gradients_but_consumes_RNG_calls_row(monkeypatch):
    e,m,p=fixture();e.begin();micro(e,m,p);previous=cpu_tree(e._gradients());rng=torch.get_rng_state().clone()
    def broken(commit):raise RuntimeError('injected D failure')
    monkeypatch.setattr(e.codec,'verbalize',broken)
    with pytest.raises(RuntimeError):micro(e,m,p)
    assert state_sha(previous)==state_sha(cpu_tree(e._gradients())) and not torch.equal(torch.get_rng_state(),rng)
    assert e.runtime['phase']=='ABORTED' and e.runtime['cursor']==2 and e.runtime['exposure_bytes']==4
    assert e.runtime['calls']==dict(denoise=2,readout=2) and e.runtime['aborted']['no_graph_replay']


def test_exception_after_partial_Adam_mutation_rolls_back_model_moments_scaler_gradients(monkeypatch):
    e,m,p=fixture();e.begin()
    for _ in range(8):micro(e,m,p)
    before=e.payload()
    def broken(*args,**kwargs):
        with torch.no_grad():next(e.thought.parameters()).add_(100)
        raise RuntimeError('injected partial Adam')
    monkeypatch.setattr(e.optimizer,'step',broken)
    with pytest.raises(RuntimeError):e.finish()
    assert state_sha(before)==state_sha(e.payload()) and e.runtime['phase']=='READY'


def test_restore_rejects_wrong_endpoint_order_and_metadata():
    e,m,p=fixture();payload=e.payload()
    for key,value in [('codec_SHA','other'),('orders',['x']),('source_metadata',{'cache':'different'})]:
        with pytest.raises(ValueError):e.restore(dict(payload,**{key:value}))
