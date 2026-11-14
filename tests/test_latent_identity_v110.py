import io,sys
from pathlib import Path
import pytest
import torch
from torch import nn
from research.memory_thought_diffusion_v94 import ThoughtConfig,InputMeaning
from research.thought_grid_tail_v104 import GridTailThought
from research.latent_identity_thought_v110 import LatentPolicy,LatentIdentityEngine,effective_velocity_rank
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from text_generation_support_v1 import state_sha

def fixture(hidden=24):
    torch.set_num_threads(2);torch.manual_seed(110)
    c=ThoughtConfig(dimension=16,key_dimension=4,value_dimension=8,hidden=hidden,heads=2,layers=1,answer_local_capacity=5,source_slot_capacity=32,max_steps=4)
    thought=GridTailThought(c)
    m=InputMeaning(torch.randn(1,6,16),torch.randn(4,4),torch.randn(4,8),torch.randn(1,4,8),torch.randn(1,4,8),torch.zeros(1,1,4),torch.zeros(1,1,8),torch.zeros(1,1,dtype=torch.bool),'e',(('s','v','SHA'),))
    class Codec(nn.Module):
        def __init__(self):super().__init__();self.frozen=nn.Parameter(torch.randn(3));self.calls=0
        def verbalize(self,*a):self.calls+=1;raise AssertionError('D forbidden in meaning-only train')
    codec=Codec();engine=LatentIdentityEngine(thought,codec,'SHA',['a','b'],times=[1.,.5],batch=2,policy=LatentPolicy(steps=4))
    target=torch.randn(1,7,16);mask=torch.tensor([[1,1,1,1,0,0,0]],dtype=torch.bool);labels=torch.full((6,16),257,dtype=torch.long);labels.flatten()[:3]=torch.tensor([65,66,256])
    packet=dict(schema='LOSS_ONLY_answer_meaning_v95',answer_SHA='x',source_codec_SHA='SHA',semantic_target=target,valid_extent_mask=mask,labels=labels,model_epoch='e',target_condition_features_never_used=True)
    return engine,m,packet

def test_wide_velocity_after_LayerNorm_has_full_affine_span():
    e,_,_=fixture();assert effective_velocity_rank(e.thought)==16
    narrow,_,_=fixture(8);assert effective_velocity_rank(narrow.thought)<=7

def test_meaning_only_micro_NEVER_calls_language_and_only_thought_updates():
    e,m,p=fixture();before=state_sha(e.codec.state_dict());theta=state_sha(e.thought.state_dict());calls=[]
    e.begin();e.micro('a',m,p,calls.append);e.micro('b',m,p,calls.append);e.finish();e.update.audit_endpoint()
    assert calls==['denoise']*10 and e.codec.calls==0 and e.runtime['calls']==dict(denoise=10,readout=0)
    assert state_sha(e.codec.state_dict())==before and state_sha(e.thought.state_dict())!=theta
    assert e.runtime['history'][0]['D_during_training'] is False

def test_exact_READY_restore_preserves_pure_latent_policy_and_input_checkpoint():
    e,m,p=fixture();e.begin();e.micro('a',m,p);e.micro('b',m,p)
    f=io.BytesIO();torch.save(e.payload(),f);f.seek(0);saved=torch.load(f,weights_only=True);old=state_sha(saved)
    e.finish();want=e.payload();e.restore(saved);e.finish();assert state_sha(e.payload())==state_sha(want) and state_sha(saved)==old
    with pytest.raises(ValueError):e.restore(dict(saved,language_losses_used=True))

def test_loss_side_labels_cannot_be_used_in_eval():
    e,m,p=fixture();e.thought.eval()
    with pytest.raises(RuntimeError):e.objective(m,p,torch.ones(1),torch.randn(1,7,16))
    assert e.codec.calls==0
