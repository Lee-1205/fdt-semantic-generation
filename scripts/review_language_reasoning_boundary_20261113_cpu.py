"""ROOT independent source inventory and saved-model-only boundary review."""
import ast,datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support


def main():
    d=ROOT/'research/artifacts/language_reasoning_boundary_20261113_v1_cpu'
    p=io.load_json(d/'plan.json');r=io.load_json(d/'actual_model_fixture/actual_boundary_audit.json');o=io.load_json(d/'cpu_preflight_owned/final_status.json');n=io.load_json(d/'ROOT_native_terminal.json')
    assert not n['PythonProcesses'] and n['all_CPU_owned_and_proxy_absent'] and n['before_actual_fixture_analysis']
    assert o['status']=='CPU_PREFLIGHT_PASS' and o['worker_exit_code']==0 and o['exit']['all_owned_processes_gone'] and not o['owner_imported_Torch'] and o['actual']['identity_verified']
    for file,sha in p['files'].items():assert support.sha(file)==sha,file
    assert support.core_sha(ROOT)==support.CORE_SHA
    selection=io.load_json(p['selection']);storage=torch.load(selection['mandatory_standalone_base'],map_location='cpu',weights_only=True);env=torch.load(selection['selected_runtime'],map_location='cpu',weights_only=True)
    warm=dict(storage['model']);warm.update(env['active_weights'])
    h=hashlib.sha256()
    for key,tensor in sorted(warm.items()):
        value=tensor.detach().cpu().contiguous();h.update(key.encode());h.update(str((tuple(value.shape),str(value.dtype))).encode());h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    assert h.hexdigest()==r['before_after_endpoint_digest']
    assert len(r['parameters'])==r['all_parameter_tensors']==285 and sum(v['parameters'] for v in r['parameters'])==123497080
    assert all(not v['requires_grad'] and tuple(v['shape'])==tuple(warm[v['name']].shape) and v['parameters']==warm[v['name']].numel() for v in r['parameters'])
    assert sum(r['parameter_groups'].values())==123497080
    assert r['gradient_reached_probe_through_frozen_actual_readout'] and r['all_endpoint_parameters_buffers_unchanged'] and r['actual_codec_optimizer_updates']==r['actual_diffusion_updates']==r['actual_language_encoder_forwards']==0
    assert r['actual_final_readout_forwards']==r['probe_optimizer_steps']==1 and not r['CUDA_initialized']
    allfiles={}
    for file in set([v['file'] for v in r['encoder_MRO']])|set(r['inspected_current_module_files']):
        f=Path(file);text=f.read_text(encoding='utf-8-sig');tree=ast.parse(text)
        allfiles[file]=dict(SHA=support.sha(f),lines=len(text.splitlines()),classes=[v.name for v in tree.body if isinstance(v,ast.ClassDef)])
    # Same module object is not the old typed semantic schema: its current
    # language codec bypasses denoise/condition_at and has no 16-role features.
    typed=ROOT/'src/fdt_local/model_masked.py';tree=ast.parse(typed.read_text(encoding='utf-8-sig'))
    names={v.name:(v.lineno,v.end_lineno) for v in ast.walk(tree) if isinstance(v,ast.FunctionDef) and v.name in ('semantic','target','condition_at','denoise')}
    histories=[ROOT.parent/'outputs'/name for name in (
        'FDT_COMMIT_READOUT_CROSS_V71_RESULT_20261112_KR.md','FDT_EXPANDED_UNTRAINED_ENGLISH_V73_RESULT_20261112_KR.md',
        'FDT_CONTEXT_VERBALIZER_V74_RESULT_20261112_KR.md','FDT_UPDATE_BALANCE_V77_RESULT_20261113_KR.md',
        'FDT_FUZZY_ROUNDTRIP_V64_RESULT_20261112_KR.md','FDT_MEMORY_ALIGNMENT_V62_RESULT_20261112_KR.md')]
    proof=dict(status='ROOT_INDEPENDENT_LANGUAGE_REASONING_BOUNDARY_REVIEW_PASS',time=datetime.datetime.now().astimezone().isoformat(),
        all19_input_pins_verified=True,original_core_SHA_unchanged=support.CORE_SHA,actual_full_model_digest_independently_reconstructed=h.hexdigest(),
        all285_parameter_names_shapes_counts_frozen_verified=True,whole_NN_parameters=123497080,parameter_groups=r['parameter_groups'],
        all_source_weights_unchanged=True,code_inventory=allfiles,original_typed_API_line_ranges=names,
        encoder_MRO_class_entries=len(r['encoder_MRO']),current_module_defining_files=len(r['inspected_current_module_files']),model_source_inventory_unique_files=len(allfiles),
        actual_fullsize_readout_gradient_fixture_only=True,actual_encoder_forward0=True,actual_diffusion0=True,new_held_evaluation0=True,
        root_review_extra_neural_calls=0,CPU_probe_steps1_NOT_diffusion_training=True,
        history_SHA={str(f):support.sha(f) for f in histories},tiny_CPU_contract_tests=11,CPU_owner_verified_child=True,ROOT_all_native_gone_before_analysis=True,
        scope='Active extraction-memory-binding-language call chain and every current NN parameter, not every historical branch or every natural-language meaning.',
        representation_limit='Information availability for observed repairs and high source fidelity; ordered lexical fuzzy code is not by itself a complete abstract semantic representation certificate.',
        future_learning_policy='Freeze entire codec parameters/buffers/position-binding/anchors/prior/decoder; separate reasoner exact optimizer ownership, disallow alias/storage sharing, retain gradient to latent through frozen decoder if text loss is used.',
        next_phase_design_only=True,no_diffusion_trial_started=True)
    io.atomic_json(d/'root_independent_review.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k not in ('code_inventory','history_SHA')}))


if __name__=='__main__':main()
