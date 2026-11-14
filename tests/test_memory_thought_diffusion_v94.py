"""Tiny CPU architecture/ownership checks; not actual FDT training/ability."""
from dataclasses import replace
from types import SimpleNamespace
import inspect
import pytest
import torch
from torch import nn
from research.memory_thought_diffusion_v94 import (
    ThoughtConfig,InputMeaning,FinalMeaning,MemoryThoughtDiffusion,vp,
    target_meaning,prepare_thought_optimizer,verbalize_final,
)
from research.frozen_language_boundary_contract_v1 import state_digest
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.thought_only_update_v94 import ThoughtOnlyUpdate


def fixture():
    c=ThoughtConfig(dimension=16,key_dimension=4,value_dimension=8,hidden=16,heads=2,layers=1,answer_local_capacity=5,source_slot_capacity=32,max_steps=4)
    source=torch.randn(1,6,16)
    m=InputMeaning(source,torch.randn(4,4),torch.randn(4,8),torch.randn(1,4,8),torch.randn(1,4,8),
        torch.zeros(1,1,4),torch.zeros(1,1,8),torch.zeros(1,1,dtype=torch.bool),'CPU_frozen_epoch',(('CPU_source','v1','hash'),))
    return c,m,MemoryThoughtDiffusion(c)


class TinyCodec(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Linear(16,16);self.memory_prior=nn.Parameter(torch.randn(4,8))
        self.binding=nn.Linear(16,16);self.decoder=nn.Linear(16,7)
        self.register_buffer('memory_revision',torch.tensor([3]));self.calls=0
    def verbalize(self,commit):
        self.calls+=1;return self.decoder(commit.reader_tensor())


def test_input_meaning_weights_initial_noise_and_bounds_scale():
    c,m,r=fixture();eps=torch.randn(1,7,16);a,s=r.initial_prior(m)
    assert ((s>.5)&(s<1.5)).all() and torch.allclose(r.initial_state(m,eps),a+s*eps)
    changed=replace(m,source=m.source+1)
    assert not torch.equal(r.initial_state(m,eps),r.initial_state(changed,eps))
    assert torch.equal(r.initial_state(m,eps),r.initial_state(m,eps))


def test_conditional_VP_forward_inverse_and_no_target_in_runtime_API():
    c,m,r=fixture();centre,scale=r.initial_prior(m);target=torch.randn_like(centre);eps=torch.randn_like(centre)
    t=torch.tensor([.37]);a,s=vp(t);residual=(target-centre)/scale
    rt=a*residual+s*eps;v=a*eps-s*residual
    assert torch.allclose(centre+scale*(a*rt-s*v),target,atol=2e-6)
    assert torch.allclose(s*rt+a*v,eps,atol=2e-6)
    assert 'target' not in inspect.signature(r.sample).parameters
    assert 'mask' not in inspect.signature(r.sample).parameters
    a,s=vp(torch.tensor([1.]));assert a.item()==0 and s.item()==1


def test_prior_and_contextual_memories_affect_denoising_and_queries_get_gradient():
    c,m,r=fixture();noise=torch.randn(1,7,16);z=r.initial_state(m,noise);t=torch.tensor([.6])
    base=r(z,t,m)
    without_prior=r(z,t,replace(m,prior=torch.zeros_like(m.prior)))
    without_context=r(z,t,replace(m,contextual_long=torch.zeros_like(m.contextual_long),contextual_recent=torch.zeros_like(m.contextual_recent)))
    assert not torch.allclose(base,without_prior) and not torch.allclose(base,without_context)
    base.square().mean().backward()
    assert r.steps[0].query.weight.grad is not None and r.steps[0].query.weight.grad.count_nonzero()
    assert all(x.grad is None for x in (m.source,m.prior,m.contextual_long,m.contextual_recent))


def test_only_diffusion_weights_change_and_frozen_decoder_passes_gradient_to_thought():
    c,m,r=fixture();codec=TinyCodec();guard,opt=prepare_thought_optimizer(codec,r,lr=1e-3)
    before=state_digest(codec);prior=state_digest(r)
    target=torch.randn(1,7,16);mask=torch.ones(1,7,dtype=torch.bool)
    loss,clean=r.training_objective(m,target,mask,torch.tensor([.5]),torch.randn_like(target))
    logits=codec.verbalize(DifferentiableCodecCommit.from_semantic(clean,m.memory_revisions,m.model_epoch))
    (loss+.1*logits.square().mean()).backward()
    assert r.velocity.weight.grad is not None and r.velocity.weight.grad.count_nonzero()
    assert r.prior_centre.weight.grad is not None and r.prior_scale.weight.grad is not None
    assert all(p.grad is None and not p.requires_grad for p in codec.parameters())
    update=ThoughtOnlyUpdate(guard,opt);update.step();update.audit_endpoint()
    assert state_digest(codec)==before and state_digest(r)!=prior
    opt.add_param_group({'params':[codec.decoder.weight]})
    with pytest.raises(ValueError):guard.assert_optimizer(opt)


def test_all_codec_buffers_and_parameter_changes_are_rejected():
    c,m,r=fixture();codec=TinyCodec();guard,opt=prepare_thought_optimizer(codec,r)
    codec.memory_revision.add_(1)
    with pytest.raises(ValueError):guard.assert_intact()
    codec=TinyCodec();guard,opt=prepare_thought_optimizer(codec,r)
    with torch.no_grad():codec.binding.weight.add_(1e-5)
    with pytest.raises(ValueError):guard.assert_intact()


def test_sampling_reads_memories_without_language_until_final_commit():
    c,m,r=fixture();codec=TinyCodec();guard,opt=prepare_thought_optimizer(codec,r)
    before=tuple(x.clone() for x in (m.source,m.prior,m.contextual_long,m.contextual_recent))
    r.eval();calls=[];noise=torch.randn(1,7,16)
    answer=r.sample(m,noise,steps=3,charge=calls.append)
    assert type(answer)is FinalMeaning and codec.calls==0 and calls==['denoise']*3
    assert torch.equal(answer.semantic,r.sample(m,noise,steps=3).semantic)
    assert all(torch.equal(x,y) for x,y in zip(before,(m.source,m.prior,m.contextual_long,m.contextual_recent)))
    logits=verbalize_final(codec,answer,guard);assert codec.calls==1 and logits.shape==(1,7,7)
    with pytest.raises(TypeError):verbalize_final(codec,answer.semantic,guard)
    with pytest.raises(ValueError):r.sample(m,noise,steps=5)


def test_training_targets_and_gold_extent_never_become_condition_features():
    c,m,r=fixture();codec=TinyCodec();guard,opt=prepare_thought_optimizer(codec,r)
    source=DifferentiableCodecCommit.from_semantic(torch.randn(1,4,16),m.memory_revisions,m.model_epoch)
    terminal=torch.randn(16);target,mask=target_meaning(source,terminal,c.answer_local_capacity)
    assert target.shape==(1,7,16) and mask.sum()==5
    assert torch.equal(target[:,4],terminal[None])
    assert torch.equal(target[:,:4],source.reader_tensor())
    with pytest.raises(ValueError):target_meaning(source,terminal,2)
    r.eval()
    with pytest.raises(RuntimeError):r.training_objective(m,target,mask,torch.tensor([.5]),torch.randn_like(target))


def test_dormant_Exact_and_detached_source_enforced():
    c,m,r=fixture()
    with pytest.raises(ValueError):r.initial_prior(replace(m,exact_valid=torch.ones_like(m.exact_valid)))
    with pytest.raises(ValueError):r.initial_prior(replace(m,exact_values=torch.ones_like(m.exact_values)))
    with pytest.raises(ValueError):r.initial_prior(replace(m,source=m.source.clone().requires_grad_()))
    with pytest.raises(ValueError):r.initial_prior(replace(m,source=torch.randn(1,33,16)))


def test_episode_adapter_clones_readonly_ND_and_NULL_without_raw_source():
    c,m,r=fixture()
    class Context:
        def means(self):return m.contextual_long,m.contextual_recent
    persistent=SimpleNamespace(anchors=m.anchors,prior=m.prior,certified_D=torch.zeros(4,1))
    exact=SimpleNamespace(keys=m.exact_keys,values=m.exact_values,valid=m.exact_valid,spans=((),))
    episode=SimpleNamespace(memories=SimpleNamespace(model_epoch=m.model_epoch,persistent=persistent,contextual=Context(),exact=exact))
    commit=DifferentiableCodecCommit.from_semantic(m.source,m.memory_revisions,m.model_epoch)
    packed=InputMeaning.from_episode(commit,episode)
    assert torch.equal(packed.source,m.source) and packed.prior.data_ptr()!=m.prior.data_ptr()
    assert set(vars(packed))=={'source','anchors','prior','contextual_long','contextual_recent','exact_keys','exact_values','exact_valid','model_epoch','memory_revisions'}


def test_update_rejects_frozen_write_or_foreign_optimizer_before_thought_step():
    c,m,r=fixture();codec=TinyCodec();guard,opt=prepare_thought_optimizer(codec,r);update=ThoughtOnlyUpdate(guard,opt)
    before=state_digest(r);codec.memory_revision.add_(1)
    with pytest.raises(ValueError):update.step()
    assert state_digest(r)==before
    codec=TinyCodec();guard,opt=prepare_thought_optimizer(codec,r);update=ThoughtOnlyUpdate(guard,opt)
    opt.add_param_group({'params':[codec.decoder.weight]})
    with pytest.raises(ValueError):update.step()
    assert state_digest(r)==before


def test_full_checkpoint_audit_catches_data_write_that_can_evade_fast_versions():
    c,m,r=fixture();codec=TinyCodec();guard,opt=prepare_thought_optimizer(codec,r);update=ThoughtOnlyUpdate(guard,opt)
    codec.decoder.bias.data.add_(1.)
    with pytest.raises(ValueError):update.audit_endpoint()
