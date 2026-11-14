import inspect,io,sys
from pathlib import Path
import pytest
import torch
from torch import nn
from research.thought_grid_tail_v104 import GridTailThought
from research.memory_thought_diffusion_v94 import ThoughtConfig,InputMeaning
from research.commit_patch_readout_v54 import CommitPatchReadout
from research.runtime_aligned_thought_v107 import training_rollout,RuntimeLossConfig,RuntimeAlignedEngine
from research.frozen_language_boundary_contract_v1 import state_digest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from text_generation_support_v1 import state_sha


def fixture():
    torch.set_num_threads(2);torch.manual_seed(107)
    c=ThoughtConfig(dimension=16,key_dimension=4,value_dimension=8,hidden=16,heads=2,layers=1,answer_local_capacity=5,source_slot_capacity=32,max_steps=4)
    thought=GridTailThought(c);m=InputMeaning(torch.randn(1,6,16),torch.randn(4,4),torch.randn(4,8),torch.randn(1,4,8),torch.randn(1,4,8),torch.zeros(1,1,4),torch.zeros(1,1,8),torch.zeros(1,1,dtype=torch.bool),'e',(('s','v','SHA'),))
    class Codec(nn.Module):
        def __init__(self):super().__init__();self.head=CommitPatchReadout(16,16);self.calls=0
        def verbalize(self,commit):self.calls+=1;return self.head.from_commit(commit)
    codec=Codec();engine=RuntimeAlignedEngine(thought,codec,'SHA',['a','b'],times=[1.,.5],batch=2,config=RuntimeLossConfig(steps=4))
    target=torch.randn(1,7,16);mask=torch.tensor([[1,1,1,1,0,0,0]],dtype=torch.bool);labels=torch.full((6,16),257,dtype=torch.long);labels.flatten()[:3]=torch.tensor([65,66,256])
    p=dict(schema='LOSS_ONLY_answer_meaning_v95',answer_SHA='loss',source_codec_SHA='SHA',semantic_target=target,valid_extent_mask=mask,labels=labels,model_epoch='e',target_condition_features_never_used=True)
    return engine,m,p


def test_runtime_trajectory_has_no_target_api_and_matches_sampler_arithmetic():
    e,m,p=fixture();noise=torch.randn(1,7,16);calls=[];z=training_rollout(e.thought,m,noise,4,calls.append)
    assert z.requires_grad and calls==['denoise']*4 and e.codec.calls==0
    assert not {'target','mask','length'}&set(inspect.signature(training_rollout).parameters)
    e.thought.eval();y=e.thought.sample(m,noise,steps=4).semantic
    assert torch.allclose(z,y,atol=3e-6,rtol=1e-6)


def test_actual_endpoint_language_gradient_updates_ONLY_thought_and_exact_call_order():
    e,m,p=fixture();before=state_digest(e.codec);prior=state_digest(e.thought);calls=[];seen=[]
    original=e.codec.verbalize
    def read(commit):seen.append(len(calls));return original(commit)
    e.codec.verbalize=read
    loss,metrics=e.objective(m,p,torch.ones(1),torch.randn(1,7,16),calls.append);loss.backward()
    assert calls==['denoise']*4+['readout','denoise'] and seen==[5]
    assert e.thought.velocity.weight.grad.count_nonzero() and e.thought.prior_centre.weight.grad.count_nonzero()
    assert all(x.grad is None and not x.requires_grad for x in e.codec.parameters())
    e.update.step();e.update.audit_endpoint();assert state_digest(e.codec)==before and state_digest(e.thought)!=prior


def test_complete_READY_restore_exact_and_checkpoint_input_not_mutated():
    e,m,p=fixture();e.begin();e.micro('a',m,p);e.micro('b',m,p)
    buf=io.BytesIO();torch.save(e.payload(),buf);buf.seek(0);saved=torch.load(buf,weights_only=True);digest=state_sha(saved)
    e.finish();expected=e.payload();e.restore(saved);e.finish();actual=e.payload()
    assert state_sha(expected)==state_sha(actual) and state_sha(saved)==digest
    assert e.runtime['calls']==dict(denoise=10,readout=2) and 'runtime_loss' in e.runtime['history'][0]
    with pytest.raises(ValueError):e.restore(dict(saved,runtime_loss_config={'steps':1}))


def test_eval_and_violated_codec_boundary_rejected():
    e,m,p=fixture();e.thought.eval()
    with pytest.raises(RuntimeError):e.objective(m,p,torch.ones(1),torch.randn(1,7,16))
    assert e.codec.calls==0;e.thought.train();next(e.codec.parameters()).requires_grad_(True)
    with pytest.raises(ValueError):e.objective(m,p,torch.ones(1),torch.randn(1,7,16))
