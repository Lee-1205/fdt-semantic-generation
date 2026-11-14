"""ONE bounded full-size endpoint ownership check. No new text evaluation.

Readout uses random fixture latents to test autograd, not meaning or generation.
One fresh linear probe update is not diffusion training or a research checkpoint.
"""
import argparse, ast, collections, inspect, json, shutil, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from torch import nn
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.frozen_language_boundary_contract_v1 import FrozenLanguageBoundary,state_digest
import learning_io_v2 as io
import text_generation_support_v1 as support


def main(path,output,preflight):
    began=time.monotonic();p=io.load_json(path)
    assert preflight and p['schema']=='fdt_language_reasoning_boundary_CPU_v1' and p['CPU_fixture_only']
    assert not p['authorized'] and not p['diffusion_training'] and p['max_fullsize_readout_forwards']==1
    assert not output.exists() and shutil.disk_usage(ROOT).free>=p['start_free_bytes']
    for f,s in p['files'].items():assert support.sha(f)==s,f
    select=io.load_json(p['selection']);assert select['selected_version']=='V75'
    storage=torch.load(select['mandatory_standalone_base'],map_location='cpu',weights_only=True)
    env=torch.load(select['selected_runtime'],map_location='cpu',weights_only=True)
    assert support.sha(select['mandatory_standalone_base'])==select['mandatory_base_SHA']
    assert support.sha(select['selected_runtime'])==select['selected_SHA']
    assert env['parent_reference']['SHA']==select['mandatory_base_SHA'] and not env['standalone']
    warm=dict(storage['model']);warm.update(env['active_weights'])
    assert set(warm)==set(env['model_keys']) and len(env['active_weights'])==7
    torch.set_num_threads(4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(131113)
        enc=FuzzyCodecV64(Config(**env['runtime']['core']),input_capacity=16384)
        enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks()
        codec=FuzzyRoundTripV64(enc,OrderedFuzzyReadout());codec.load_state_dict(warm,strict=True)
        assert all(torch.equal(t,warm[n]) for n,t in codec.state_dict().items())
        # A one-layer matrix fixture deliberately supplies no denoising model.
        probe=nn.Linear(512,512,bias=False)
        boundary=FrozenLanguageBoundary(codec,probe,('weight',))
        optimizer=torch.optim.AdamW(probe.parameters(),lr=1e-4,weight_decay=.01)
        assert boundary.assert_optimizer(optimizer)
        before=state_digest(codec);probe_before=state_digest(probe)
        x=torch.randn(1,4,512);z=probe(x);z.retain_grad()
        commit=DifferentiableCodecCommit.from_semantic(z,('CPU_gradient_fixture_NO_SOURCE_TEXT',),'boundary_CPU_only')
        logits=codec.verbalize(commit);assert logits.requires_grad
        loss=logits.float().square().mean();loss.backward()
        assert torch.isfinite(probe.weight.grad).all() and probe.weight.grad.count_nonzero()
        assert z.grad is not None and z.grad.count_nonzero()
        assert all(t.grad is None and not t.requires_grad for t in codec.parameters())
        optimizer.step();assert boundary.assert_optimizer(optimizer) and boundary.assert_intact()
        assert state_digest(codec)==before and state_digest(probe)!=probe_before
        parameters=[];groups=collections.Counter();classfiles={}
        for name,t in codec.named_parameters():
            if name.startswith('encoder.core.language.'):group='input_language'
            elif name.startswith('readout.readout.'):group='active_final_language_readout'
            elif name=='readout.role_projection.weight':group='input_fuzzy_order_binding'
            elif name.startswith('encoder.decoder.'):group='inherited_AR_decoder_not_used_by_FuzzyRoundTrip_verbalize'
            elif name.startswith(('encoder.core.slot_router.','encoder.core.rel_','encoder.core.time.','encoder.core.refiner.','encoder.core.refine_norm.','encoder.core.velocity.')):group='inherited_typed_reasoning_NOT_executed_by_current_codec'
            else:group='fuzzy_memory_extraction_and_inherited_other'
            groups[group]+=t.numel();parameters.append(dict(name=name,shape=list(t.shape),parameters=t.numel(),requires_grad=t.requires_grad,group=group))
        assert sum(groups.values())==123497080 and groups['active_final_language_readout']==2908192
        for module in codec.modules():
            cls=type(module)
            if cls.__module__.startswith(('research.','fdt_local.')):
                f=Path(inspect.getfile(cls)).resolve()
                assert f.is_relative_to(ROOT)
                classfiles[str(f)]=dict(SHA=support.sha(f),lines=len(f.read_text(encoding='utf-8-sig').splitlines()))
        mro=[dict(cls=cls.__module__+'.'+cls.__name__,file=inspect.getfile(cls)) for cls in type(codec.encoder).__mro__ if cls.__module__.startswith('research.')]
        # Algebra of the dormant Exact branch: a zero value followed by a biased
        # projection can add a constant; it cannot carry any input text identity.
        with torch.no_grad():
            first,last=codec.encoder.core.condition[0],codec.encoder.core.condition[2]
            null_projection=last.weight@torch.nn.functional.silu(first.bias)
            gains=(128*codec.encoder.memory_codec.local_bank_gains).tanh()
            null_offset=gains[1]*null_projection
        report=dict(status='FULLSIZE_FROZEN_LANGUAGE_BOUNDARY_CPU_PASS',whole_NN_parameters=sum(groups.values()),all_parameter_tensors=len(parameters),
            parameter_groups=dict(groups),parameters=parameters,source_model_all_tensors_exact=True,
            before_after_endpoint_digest=before,all_endpoint_parameters_buffers_unchanged=True,all_endpoint_grad_None=True,endpoint_eval_all_modules=True,
            probe_parameters=probe.weight.numel(),probe_optimizer_steps=1,probe_only_changed=True,gradient_reached_probe_through_frozen_actual_readout=True,
            actual_language_encoder_forwards=0,actual_final_readout_forwards=1,actual_codec_optimizer_updates=0,actual_diffusion_forwards=0,actual_diffusion_updates=0,
            new_prompt_or_held_evaluation_rows=0,CUDA_initialized=torch.cuda.is_initialized(),
            source_SHA={k:support.sha(k) for k in (select['selected_runtime'],select['mandatory_standalone_base'])},
            dormant_Exact_projection=dict(content_entries=0,copy_bytes=0,projected_NULL_norm=float(null_projection.norm()),constant_local_offset_norm=float(null_offset.norm()),
                local_bank_scales=gains.tolist(),interpretation='A potential input-independent projection bias, not Exact retrieval or source copying. No old weights or behavior changed.'),
            inspected_current_module_files=classfiles,encoder_MRO=mro,
            latent_contract=dict(dimension=512,global_views=2,local_slot_source_stride=16,ordered_fuzzy_role_width=32,
                local_Commit_storage_grows_with_input=True,language_layers=9,per_layer_attention_window=48,
                language_local_receptive_previous_bytes_upper_bound=423,global_fuzzy_ND_reads_entire_input=True,
                output_uses_source_H_or_raw_bytes=False,decoder_AR_prefix_used_in_current_path=False,
                original_typed_21field_basis_is_NOT_codec_Commit_schema=True),
            seconds=time.monotonic()-began,limitations='Full-size exact weight restore/freeze and random-latent gradient boundary only; not full worker/scaler/RNG recovery, semantic fidelity, actual thought architecture, or diffusion training.')
        assert not report['CUDA_initialized'] and time.monotonic()-began<p['worker_seconds_cap']
        for f,s in p['files'].items():assert support.sha(f)==s,f
        output.mkdir();io.atomic_json(output/'actual_boundary_audit.json',report)
        assert sum(f.stat().st_size for f in output.rglob('*') if f.is_file())<p['max_output_bytes']
        print('CPU_PREFLIGHT_REPORT '+json.dumps({k:v for k,v in report.items() if k not in ('parameters','inspected_current_module_files','encoder_MRO')}))
        print('CPU_PREFLIGHT_PASS')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',action='store_true')
    a=parser.parse_args();main(a.plan.resolve(),a.output.resolve(),a.preflight)
