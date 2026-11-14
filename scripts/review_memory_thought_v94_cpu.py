"""Select immutable pre-regression codec and review NEW thought-only CPU path.

Actual123M source files are verified, not instantiated or evaluated again.
The optimizer/denoising fixture uses synthetic tiny tensors only.
"""
import datetime,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
from torch import nn
from research.memory_thought_diffusion_v94 import (
    ThoughtConfig,InputMeaning,MemoryThoughtDiffusion,target_meaning,
    prepare_thought_optimizer,verbalize_final,
)
from research.thought_only_update_v94 import ThoughtOnlyUpdate
from research.frozen_language_boundary_contract_v1 import state_digest
from research.commit_patch_readout_v54 import CommitPatchReadout
from research.three_memory_codec_v33 import DifferentiableCodecCommit
import learning_io_v2 as io
import text_generation_support_v1 as support


def main():
    began=time.monotonic();torch.set_num_threads(2);rng=torch.get_rng_state().clone()
    a=ROOT/'research/artifacts/memory_thought_20261113_v94_cpu_v1';a.mkdir(exist_ok=False)
    source=ROOT/'runs/critical_repair_20261113_v88/model/runtime.pt';meta=io.load_json(source.with_suffix('.json'))
    assert support.sha(source)==meta['SHA']=='ebe939a6c14b205bd92a6cad772b33cf2ee462473bb77daadcc6ba48926c84d7'
    base=Path(meta['mandatory_parent']['path']);assert support.sha(base)==meta['mandatory_parent']['SHA']
    env=torch.load(source,map_location='cpu',weights_only=True)
    assert env['reconstructed_fullstate_SHA']==meta['reconstructed_fullstate_SHA'] and len(env['active_names'])==7
    old=io.load_json(ROOT/'runs/feasible_repair_20261113_v91/root_combined_assessment_review.json')
    cause=io.load_json(ROOT/'research/artifacts/v91_error_cause_20261113_cpu_v1/root_cause_review.json')
    assert old['panels']['long']['before_decoded_errors']==49 and old['panels']['long']['after_decoded_errors']==52
    assert cause['three_new_errors_reproduced_from_only_last2_tensors'] and cause['three_errors_recoved_by_only_two_class_rows_offline']
    # Whole-version selection reverts the unpromoted6updates. No cherry-picked
    # DEV class-row patch, output repair or changed checkpoint file.
    selected=dict(schema='frozen_codec_for_separate_memory_thought_v94',selected_version='V88_frozen_codec_research_endpoint',
        selected_runtime=str(source),selected_SHA=meta['SHA'],source_algorithm_SHA=meta['reconstructed_fullstate_SHA'],
        mandatory_standalone_base=str(base),mandatory_base_SHA=meta['mandatory_parent']['SHA'],
        source_model_SHA='42a2032db3137ffb9f876585cf26979854d2555e19639fa5c2f1e8d61201b96a',
        embedding_SHA=env['frozen_model_SHA'],precision_policy=env['runtime']['precision_policy'],
        freeze='ENTIREcodec123497080parameters+ALLbuffers+allthreeMemoryweights+binding+output+legacybranches',
        trainable='NEWMemoryThoughtDiffusion ONLY; exact optimizer allowlist; no shared storage',
        resolution='DiscardV91asendpoint, reusecompleteV88head, so3newlongregressionsremoved; capitalization-onlyV91gainalsoforegone.',
        V91weights_Adam_RNG_preserved=True,no_checkpoint_rewritten=True,global_model_registry_changed=False,
        known_codec_limits=dict(long_errors=49,long_source_bytes=524071,numeric_value_clear64=52,
            clear_wrong_numeric64=4,nonstandard_prefix_uncertified64=8,whole_semantic_codec_complete=False),
        source_selection_scope='Frozen research endpoint for preparation. Does not certify arbitrary meanings or untrained diffusion ability.')
    selection=ROOT/'configs/frozen_thought_codec_20261113_v94.json'
    if selection.exists():raise RuntimeError('selection already exists; inspect, do not overwrite')
    io.atomic_json(selection,selected)
    class Codec(nn.Module):
        def __init__(self):
            super().__init__();self.encoder=nn.Linear(16,16);self.memory_prior=nn.Parameter(torch.randn(4,8))
            self.readout=CommitPatchReadout(16,32,16);self.register_buffer('memory_epoch',torch.tensor([94]));self.calls=0
        def verbalize(self,c):self.calls+=1;return self.readout.from_commit(c)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(94113)
        config=ThoughtConfig(dimension=16,key_dimension=4,value_dimension=8,hidden=16,heads=2,layers=1,answer_local_capacity=5,source_slot_capacity=32,max_steps=4)
        thought=MemoryThoughtDiffusion(config);codec=Codec()
        boundary,opt=prepare_thought_optimizer(codec,thought);updater=ThoughtOnlyUpdate(boundary,opt)
        codec_before=state_digest(codec);thought_before=state_digest(thought)
        condition=InputMeaning(torch.randn(1,6,16),torch.randn(4,4),codec.memory_prior.detach().clone(),
            torch.randn(1,4,8),torch.randn(1,4,8),torch.zeros(1,1,4),torch.zeros(1,1,8),torch.zeros(1,1,dtype=torch.bool),
            'SYNTHETIC_CPU_frozen_codec',(('source_input','v1','SHA_only'),))
        memories_before=support.state_sha(vars(condition))
        answer=DifferentiableCodecCommit.from_semantic(torch.randn(1,4,16),(('TRAIN_answer_target_only','v1','SHA'),),condition.model_epoch)
        target,mask=target_meaning(answer,codec.readout.terminal,config.answer_local_capacity)
        loss,clean=thought.training_objective(condition,target,mask,torch.tensor([.45]),torch.randn_like(target))
        logits=codec.verbalize(DifferentiableCodecCommit.from_semantic(clean,condition.memory_revisions,condition.model_epoch))
        (loss+.1*logits.square().mean()).backward();grad=thought.velocity.weight.grad
        assert grad is not None and grad.count_nonzero() and all(v.grad is None for v in codec.parameters())
        updater.step();updater.audit_endpoint()
        assert state_digest(codec)==codec_before and state_digest(thought)!=thought_before
        before_calls=codec.calls;thought.eval();charges=[]
        final=thought.sample(condition,torch.randn_like(target),steps=4,charge=charges.append)
        assert codec.calls==before_calls and charges==['denoise']*4
        result=verbalize_final(codec,final,boundary)
        assert codec.calls==before_calls+1 and result.shape==(6,16,258)
        assert support.state_sha(vars(condition))==memories_before and state_digest(codec)==codec_before
        # Verify training EOS latent reproduces the frozen decoder's existing
        # structural terminal calculation; it is not a forced textual template.
        g=condition.source[0,:2].unsqueeze(0)
        x=codec.readout(torch.zeros(1,16),g,torch.ones(1,dtype=torch.bool))
        y=codec.readout(codec.readout.terminal[None],g,torch.zeros(1,dtype=torch.bool))
        assert torch.equal(x,y)
        full_candidate=MemoryThoughtDiffusion()
        parameter_names=[n for n,_ in full_candidate.named_parameters()]
        parameter_count=sum(v.numel() for v in full_candidate.parameters())
        assert len(parameter_names)==len(set(parameter_names))
    assert not torch.cuda.is_initialized() and torch.equal(rng,torch.get_rng_state())
    assert support.sha(source)==selected['selected_SHA']
    report=dict(status='ROOT_V94_FROZEN_V88_CODEC_SELECTION_AND_MEMORY_THOUGHT_CPU_CONTRACT_PASS_NOT_TRAINED',
        selected_codec=selected,focused_CPU_tests=11,candidate_parameter_count=parameter_count,candidate_parameter_names=parameter_names,
        architecture='inputEmeaning→inputconditionedcentre/scaleGaussian→eachstepK_P/K_C/K_E_NULLreads→VPanswermeaning→FINALcommit→frozenD',
        original123Mmodel_instantiated=False,actual_learned_FDT_forwards=0,actual_research_optimizer_updates=0,GPU_calls=0,
        synthetic_thought_only_Adam_steps=1,synthetic_VP_denoising_steps=4,
        all_synthetic_codec_parameters_buffers_bitwise_unchanged=True,all_input_memory_snapshot_values_unchanged=True,
        gradient_reaches_thought_through_frozen_decoder=True,language_not_called_during_sampling=True,EOS_latent_matches_existing_terminal=True,
        all_source_weight_files_unchanged=True,Torch_global_RNG_unchanged=True,CUDA_initialized=False,
        inference_target_or_gold_length_inputs=False,Exact_policy='invalidNULLcontent0copy0; future explicit internalcopy selection not implemented',
        limits='Untrained candidate and synthetic CPU contracts only. Existing V88 data already observed; residual codec errors remain. Source slot cap covers262144-byte geometry, not tested long-context reasoning. First answer capacity32meaninglocal+learnedEOS≤512targetbytes; no long-answer diffusion claim.',
        next='Prepare separate source-only input/memory vs loss-only E(answer) corpus/caches. Measure codec target ceiling separately from thought latent/output accuracy. Actual worker/fullstate/scaler/RNG/newcaps/preflight/ownedOS/resource review before ONE thought-only training; no codec SGD or old budget reuse.',
        actual_new_worker_plan_seal_or_GPU_training=False,CPU_seconds=time.monotonic()-began,Cfree=shutil.disk_usage(ROOT).free)
    io.atomic_json(a/'root_cpu_review.json',report)
    files=[Path(__file__).resolve(),selection,source,a/'root_cpu_review.json',ROOT/'research/memory_thought_diffusion_v94.py',ROOT/'research/thought_only_update_v94.py',
        ROOT/'research/frozen_language_boundary_contract_v1.py',ROOT/'tests/test_memory_thought_diffusion_v94.py',
        ROOT/'research/artifacts/language_reasoning_boundary_20261113_v1_cpu/root_independent_review.json',
        ROOT/'research/artifacts/v91_error_cause_20261113_cpu_v1/root_cause_review.json']
    io.atomic_json(a/'evidence_SHA.json',{str(f):support.sha(f) for f in files})
    print(json.dumps({k:v for k,v in report.items() if k not in ('selected_codec','candidate_parameter_names')}))


if __name__=='__main__':main()
