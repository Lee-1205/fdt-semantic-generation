"""Focused tiny CPU ownership/target separation/EOS objective contracts."""
from dataclasses import replace
import pytest
import torch
from torch import nn
from research.commit_patch_readout_v54 import CommitPatchReadout
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.memory_thought_diffusion_v94 import ThoughtConfig,InputMeaning,MemoryThoughtDiffusion,target_meaning,prepare_thought_optimizer
from research.thought_only_update_v94 import ThoughtOnlyUpdate
from research.thought_language_objective_v96 import ThoughtLanguageObjective
from research.frozen_language_boundary_contract_v1 import state_digest


class Codec(nn.Module):
    def __init__(self):
        super().__init__();self.head=CommitPatchReadout(16,16);self.calls=0
    def verbalize(self,commit):
        self.calls+=1;return self.head.from_commit(commit)


def fixture():
    torch.manual_seed(96);torch.set_num_threads(2)
    c=ThoughtConfig(dimension=16,key_dimension=4,value_dimension=8,hidden=16,heads=2,layers=1,answer_local_capacity=5,source_slot_capacity=32,max_steps=4)
    source=InputMeaning(torch.randn(1,6,16),torch.randn(4,4),torch.randn(4,8),torch.randn(1,4,8),torch.randn(1,4,8),torch.zeros(1,1,4),torch.zeros(1,1,8),torch.zeros(1,1,dtype=torch.bool),'fixture',(('source','v','SHA'),))
    codec=Codec();thought=MemoryThoughtDiffusion(c);boundary,opt=prepare_thought_optimizer(codec,thought)
    update=ThoughtOnlyUpdate(boundary,opt);objective=ThoughtLanguageObjective(thought,codec,update,'frozenSHA')
    commit=DifferentiableCodecCommit.from_semantic(torch.randn(1,4,16),source.memory_revisions,'fixture')
    target,mask=target_meaning(commit,codec.head.terminal,5)
    labels=torch.full((6,16),257,dtype=torch.long);labels.flatten()[:3]=torch.tensor([65,66,256])
    packet=dict(schema='LOSS_ONLY_answer_meaning_v95',answer_SHA='lossSHA',source_codec_SHA='frozenSHA',semantic_target=target,valid_extent_mask=mask,labels=labels,model_epoch='fixture',target_condition_features_never_used=True)
    return source,codec,thought,update,objective,packet


def test_only_thought_receives_and_applies_language_gradient():
    source,codec,thought,update,objective,packet=fixture();frozen=state_digest(codec);before=state_digest(thought);charged=[]
    loss,metrics=objective(source,packet,torch.tensor([.7]),torch.randn(1,7,16),charged.append);loss.backward()
    assert metrics['active_bytes']==2 and metrics['EOS_count']==1 and metrics['PAD_ignored']==93 and charged==['denoise','readout']
    assert thought.velocity.weight.grad.count_nonzero() and all(p.grad is None and not p.requires_grad for p in codec.parameters())
    update.step();update.audit_endpoint();assert state_digest(codec)==frozen and state_digest(thought)!=before


def test_eval_cannot_receive_gold_target_or_EOS_and_does_not_call_language():
    source,codec,thought,update,objective,packet=fixture();thought.eval()
    with pytest.raises(RuntimeError):objective(source,packet,torch.tensor([.7]),torch.randn(1,7,16))
    assert codec.calls==0
    final=thought.sample(source,torch.randn(1,7,16),steps=2);assert final.denoising_steps==2 and codec.calls==0


def test_foreign_or_contaminated_target_and_live_codec_rejected_before_forward():
    source,codec,thought,update,objective,packet=fixture()
    for changed in (dict(packet,source_codec_SHA='other'),dict(packet,answer_role=1),dict(packet,model_epoch='other')):
        with pytest.raises(ValueError):objective(source,changed,torch.tensor([.5]),torch.randn(1,7,16))
    next(codec.parameters()).requires_grad_(True)
    with pytest.raises(ValueError):objective(source,packet,torch.tensor([.5]),torch.randn(1,7,16))
    assert codec.calls==0


def test_PAD_and_EOS_losses_are_distinct_and_invalid_boundaries_rejected():
    source,codec,thought,update,objective,packet=fixture();labels=packet['labels'].clone();labels.flatten()[8]=256
    with pytest.raises(ValueError):objective(source,dict(packet,labels=labels),torch.tensor([.5]),torch.randn(1,7,16))
    assert codec.calls==0
    # Different fixed EOS weight changes the objective by exactly the EOS CE,
    # with same source/noise and frozen D. This is a loss-only choice.
    from research.thought_language_objective_v96 import ThoughtLanguageLossConfig
    other=ThoughtLanguageObjective(thought,codec,update,'frozenSHA',ThoughtLanguageLossConfig(.5,1.5));noise=torch.randn(1,7,16)
    a,ma=objective(source,packet,torch.tensor([.5]),noise);b,mb=other(source,packet,torch.tensor([.5]),noise)
    assert torch.allclose(b-a,a.new_tensor(ma['EOS_CE']),atol=1e-5) and ma['byte_CE']==mb['byte_CE']
