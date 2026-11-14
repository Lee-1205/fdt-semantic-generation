"""CPU coverage/endpoint-target-separation/tail-gradient/restore contracts."""
import io,json
from pathlib import Path
import pytest
import torch
from torch import nn
from research.thought_grid_tail_v104 import GridTailThought,GridThoughtTrainEngine,finite_curriculum
from research.memory_thought_diffusion_v94 import ThoughtConfig,InputMeaning,MemoryThoughtDiffusion
from research.commit_patch_readout_v54 import CommitPatchReadout
from research.frozen_language_boundary_contract_v1 import state_digest
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from text_generation_support_v1 import state_sha


def fixture():
    torch.set_num_threads(2);torch.manual_seed(104)
    c=ThoughtConfig(dimension=16,key_dimension=4,value_dimension=8,hidden=16,heads=2,layers=1,answer_local_capacity=5,source_slot_capacity=32,max_steps=4)
    m=InputMeaning(torch.randn(1,6,16),torch.randn(4,4),torch.randn(4,8),torch.randn(1,4,8),torch.randn(1,4,8),torch.zeros(1,1,4),torch.zeros(1,1,8),torch.zeros(1,1,dtype=torch.bool),'e',(('s','v','SHA'),))
    return c,m,GridTailThought(c)


def test_every_existing_TRAIN_row_has_each16_times_and_reverse_pairs_stay_together():
    p=Path(__file__).resolve().parents[1]/'research/artifacts/memory_thought_data_20261113_v95_cpu_v1/train_targets.json'
    rows=json.loads(p.read_text());orders=finite_curriculum(rows);seen={r['id']:[] for r in rows};pair={r['id']:r['pair_key'] for r in rows}
    assert len(orders)==16 and sum(map(len,orders))==6144
    for epoch in orders:
        assert len(epoch)==384 and len({r['id'] for r in epoch})==384
        for i in range(0,384,2):assert pair[epoch[i]['id']]==pair[epoch[i+1]['id']] and epoch[i]['t']==epoch[i+1]['t']
        for r in epoch:seen[r['id']].append(r['t'])
    assert all(sorted(times)==sorted(1-i/16 for i in range(16)) for times in seen.values())


def test_exact_t1_model_input_is_independent_of_target_and_matches_input_conditioned_noise():
    c,m,thought=fixture();noise=torch.randn(1,7,16);mask=torch.tensor([[1,1,1,1,0,0,0]],dtype=torch.bool);inputs=[]
    handle=thought.register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach().clone()))
    thought.training_objective(m,torch.randn_like(noise),mask,torch.ones(1),noise)
    thought.training_objective(m,100*torch.randn_like(noise),mask,torch.ones(1),noise);handle.remove()
    assert torch.equal(inputs[0],inputs[1]) and torch.equal(inputs[0],thought.initial_state(m,noise))


def test_tail_gets_direct_velocity_gradient_without_extra_model_calls():
    c,m,thought=fixture();noise=torch.randn(1,7,16);target=torch.randn_like(noise);mask=torch.tensor([[1,1,1,1,0,0,0]],dtype=torch.bool)
    parameter=nn.Parameter(torch.zeros_like(noise));thought.forward=lambda z,t,m:parameter
    thought.tail_weight=0;loss,_=thought.training_objective(m,target,mask,torch.ones(1),noise);old=torch.autograd.grad(loss,parameter)[0]
    thought.tail_weight=.05;loss,_=thought.training_objective(m,target,mask,torch.ones(1),noise);new=torch.autograd.grad(loss,parameter)[0]
    assert old[~mask].count_nonzero()==0 and new[~mask].count_nonzero()>0 and torch.allclose(old[mask],new[mask])


def test_inference_is_bitwise_same_operator_and_weights_as_parent():
    c,m,thought=fixture();parent=MemoryThoughtDiffusion(c);parent.load_state_dict(thought.state_dict());parent.eval();thought.eval();noise=torch.randn(1,7,16)
    assert state_digest(parent)==state_digest(thought)
    assert torch.equal(parent.sample(m,noise,steps=4).semantic,thought.sample(m,noise,steps=4).semantic)


def test_grid_engine_READY_restore_preserves_checkpoint_and_exact_next_Adam():
    c,m,thought=fixture()
    class Codec(nn.Module):
        def __init__(self):super().__init__();self.head=CommitPatchReadout(16,16)
        def verbalize(self,commit):return self.head.from_commit(commit)
    codec=Codec();engine=GridThoughtTrainEngine(thought,codec,'SHA',['a','b'],times=[1.,.5],batch=2)
    target=torch.randn(1,7,16);mask=torch.tensor([[1,1,1,1,0,0,0]],dtype=torch.bool);labels=torch.full((6,16),257,dtype=torch.long);labels.flatten()[:3]=torch.tensor([65,66,256])
    packet=dict(schema='LOSS_ONLY_answer_meaning_v95',answer_SHA='loss',source_codec_SHA='SHA',semantic_target=target,valid_extent_mask=mask,labels=labels,model_epoch='e',target_condition_features_never_used=True)
    engine.begin();engine.micro('a',m,packet);engine.micro('b',m,packet);p=engine.payload();b=io.BytesIO();torch.save(p,b);b.seek(0);saved=torch.load(b,weights_only=True);digest=state_sha(saved)
    engine.finish();expected=engine.payload();engine.restore(saved);engine.finish();actual=engine.payload()
    assert state_sha(expected)==state_sha(actual) and state_sha(saved)==digest
    with pytest.raises(ValueError):engine.restore(dict(saved,fixed_times=[.5,1.]))
