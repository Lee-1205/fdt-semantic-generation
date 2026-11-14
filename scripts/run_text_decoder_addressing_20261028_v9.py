"""New decoder-alignment learning allocation; original weights/Adam/RNG retained."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,sys,time,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.text_decoder_addressing_v9 import AddressedLanguage,ADDRESS_POLICY
from research.text_language_continuation_v7 import ContinuationConfig
from research.text_corpus_contract_v5 import validate_rows
from text_decoder_addressing_engine_v9 import AddressEngine
from text_decoder_alignment_probes_v8 import interface_probe,causal_codec_probe
from text_language_runtime_support_v6 import RecoveryBudget
from run_text_language_continuation_20261028_deterministic_v2 import evaluate_nll,free_generate,dump_tensors,generation_indices
import text_generation_support_v1 as support
import learning_io_v2 as io

SCHEMA='fdt_text_decoder_addressing_v9'
LIMITS={'attempts':1636,'batch':8,'micro':4,'wall':900,'reserve':90,'max_calls':11000,
    'max_output_bytes':6*1024**3,'start_free_bytes':40*1024**3,'min_gpu_free_mib':12288}
SOURCE_SHA='9b7f5bf310922c6d7261ef75b30091b1252dc51c66bf61d92ed440534dd61258'
SOURCE_PLAN='f52f2c3bcc2f1f0f39d2afc37351f7e77739e14e563bc941abd14892800cd350'
SOURCE_ALGORITHM='8195500995628858c1da6026febdd8c87828f7406ad6098e6836ad4190236515'
MANIFEST_SHA='5b0ff59fd3985ab825085d0f3e4f5f7312baf5a22b137b62ec4b0b23f66f3f6e'


def validate(plan_path,output,worker=None):
    p=io.load_json(plan_path)
    if p.get('schema')!=SCHEMA or type(p.get('authorized')) is not bool or p.get('limits')!=LIMITS:raise ValueError('schema/caps')
    if Path(p['root']).resolve()!=ROOT or Path(p['output']).resolve()!=Path(output).resolve():raise ValueError('root/output')
    if p.get('source_sha')!=SOURCE_SHA or p.get('source_plan_sha')!=SOURCE_PLAN or p.get('source_algorithm')!=SOURCE_ALGORITHM:raise ValueError('source fullstate identity')
    if p.get('E2_final_accessed') is not False or p.get('source_attempts')!=1636 or p.get('global_attempt_cap')!=4499 or p.get('epochs')!=4:raise ValueError('global allocation/holdout')
    if p.get('cost_contract')!={'reconstruction':6544,'NLL':1998,'generation':1100,'held_reconstruction':24,'oracle_causal_generation_max':516,'total_max':10182}:raise ValueError('actual finite cost')
    expected={'epochs':4,'seeds':[81,82,83,84],'reconstruction_attempts':1636,'joint_attempts':0,'training_eos_CE_weight':4.0,'span_UL_beta':0.0,'base_language_LR':1e-5,'base_other_LR':3e-5,'decoder_rawcontext_dropout':0.5,'codec_noise_sigma_max':0.15,'memory_position_init':'retainV8','quality_metrics_stop_training':False,'new_row_exposures':13084,'new_target_bytes':1538700}
    if p.get('curriculum')!=expected or p.get('address_policy')!=ADDRESS_POLICY:raise ValueError('prospective curriculum/address prior')
    if p.get('manifest_sha')!=MANIFEST_SHA or support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('data/core')
    required={str(Path(p['source']).resolve()):SOURCE_SHA,str((Path(p['artifact'])/'manifest.json').resolve()):MANIFEST_SHA}
    if any(p['files'].get(n)!=h for n,h in required.items()):raise ValueError('required binding')
    for path,h in p['files'].items():
        if support.sha(path)!=h:raise ValueError('pinned file changed '+path)
    if worker and str(Path(worker).resolve()) not in p['files']:raise ValueError('worker not pinned')
    if p['deadline'] is not None and (p['deadline']-p['allocation_start']!=900 or p['cutoff']!=p['deadline']-90):raise ValueError('new immutable clock')
    a=Path(p['artifact']);manifest=io.load_json(a/'manifest.json')
    for name,h in manifest['files'].items():
        if p['files'].get(str((a/name).resolve()))!=h:raise ValueError('artifact binding')
    features=io.load_json(a/'features.json');targets=io.load_json(a/'targets.json');docs=io.load_json(a/'documents.json')
    if {k:len(v) for k,v in features.items()}!={'train':3271,'dev':444}:raise ValueError('data pool')
    validate_rows(docs,features,targets)
    if p['generation_indices']!=generation_indices(targets['dev']):raise ValueError('fixed generation panel')
    n=Path(p['negative_artifact']);manifest=io.load_json(n/'manifest.json')
    if support.sha(n/'manifest.json')!=p['negative_manifest_sha'] or support.sha(n/'negatives.json')!=manifest['negative_file_sha256']:raise ValueError('negative artifact')
    negatives=io.load_json(n/'negatives.json')
    if negatives['dev_outputs_used_to_build_negatives'] or negatives['E2_final_accessed']:raise ValueError('train-only negative source')
    if [r['id'] for r in negatives['rows']]!=[r['id'] for r in features['train']]:raise ValueError('negative identities')
    return p,features,targets


def load_import(plan,features,targets,budget,plan_sha,device):
    source=torch.load(plan['source'],map_location='cpu',weights_only=True)
    if support.state_sha(source)!=SOURCE_ALGORITHM:raise ValueError('source algorithm digest')
    model=AddressedLanguage(Config(**source['core']),ContinuationConfig(**source['policy']))
    negative=io.load_json(Path(plan['negative_artifact'])/'negatives.json')['rows']
    engine=AddressEngine(model,features['train'],targets['train'],negative,budget,plan_sha,device)
    digest=engine.import_parent(source,SOURCE_PLAN)
    if (engine.attempts,engine.updates,engine.cursor)!=(0,0,0):raise ValueError('import counters')
    return engine,source['core'],digest


def run(plan_path,output):
    p,features,targets=validate(plan_path,output,__file__)
    if not p['authorized'] or p['deadline'] is None:raise PermissionError('unsealed')
    if (output/'progress.json').exists():raise FileExistsError('ONE invocation only')
    configure_determinism();torch.set_num_threads(4)
    budget=RecoveryBudget(p);engine=None;latest=None;phase='importing';guard=False;started=time.time();core=None
    def progress():
        io.atomic_json(output/'progress.json',{'phase':phase,'attempts':engine.attempts if engine else 0,
            'updates':engine.updates if engine else 0,'cursor':engine.cursor if engine else 0,
            'new_attempts':engine.attempts if engine else 0,'learning_phase':engine.phase if engine else 'import','actual_calls':budget.calls,
            'allocation_calls':budget.allocation_calls,'native_pid':os.getpid(),'parent_pid':os.getppid(),
            'time_unix':time.time(),'elapsed':time.time()-started,'latest':latest})
    def commit(kind='model'):
        nonlocal latest
        if kind=='emergency':
            if time.time()>=p['deadline']:raise TimeoutError('new owner end reached')
        else:budget.check();engine.verify_frozen()
        directory=output/kind;directory.mkdir(exist_ok=True)
        payload=engine.export();payload.update(core=core,policy=engine.model.specification()['policy'],
            source_sha256=SOURCE_SHA,artifact_manifest_sha256=MANIFEST_SHA,
            allocation_calls=dict(budget.allocation_calls),emergency_snapshot=kind=='emergency')
        latest={**support.save_full(directory,payload),'relative_directory':kind};io.atomic_json(output/'latest.json',latest)
    budget.tick=progress;progress()
    try:
        budget.check()
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        engine,core,import_digest=load_import(p,features,targets,budget,support.sha(plan_path),'cuda')
        # Existing baseline is a completed artifact, not a reused forward budget.
        for name in ('parent_reference_NLL.json','parent_reference_NLL_logits.pt','parent_reference_generation.json'):
            source=Path(p['baseline_run'])/name.replace('parent_reference','final')
            with source.open('rb') as src,(output/name).open('xb') as dest:shutil.copyfileobj(src,dest)
        io.atomic_json(output/'import_review.json',{'old_model_Adam_scaler_RNG_component_sha256':import_digest,
            'source_sha256':SOURCE_SHA,'parent_local_attempts':1636,'parent_global_updates':2862,'new_cursor':0,
            'new_allocation_calls':budget.allocation_calls,'baseline_reused_from':p['baseline_run'],
            'source_moments_preserved':True,'old_CUDA_rng_restored':True,
            'curriculum':'same-index soft position prior log(validmemory-1); retain all V8model/145Adam/scaler/RNG including learned positions; beta0/frozenVP/4finiteepochs'})
        commit('import');phase='initial_interface_probe';progress()
        chosen=p['generation_indices'];selected_features=[features['dev'][i] for i in chosen];selected_targets=[targets['dev'][i] for i in chosen]
        initial,tensors=interface_probe(engine.model,selected_features,selected_targets,budget,'cuda')
        dump_tensors(output/'initial_interface_logits.pt',tensors);del tensors
        io.atomic_json(output/'initial_interface.json',initial)
        io.atomic_json(output/'initial_oracle_causal.json',causal_codec_probe(engine.model,budget,'cuda'))
        phase='training';progress()
        while engine.attempts<1636:
            if time.time()+90>=p['cutoff']:raise TimeoutError('final evaluation reserve')
            engine.step()
            if engine.attempts in (409,818,1227):commit()
            progress()
        phase='final_interface_probe';progress()
        final_interface,tensors=interface_probe(engine.model,selected_features,selected_targets,budget,'cuda')
        dump_tensors(output/'final_interface_logits.pt',tensors);del tensors
        io.atomic_json(output/'final_interface.json',final_interface)
        io.atomic_json(output/'final_oracle_causal.json',causal_codec_probe(engine.model,budget,'cuda'))
        phase='final_NLL';progress()
        nll,tensors=evaluate_nll(engine.model,features['dev'],targets['dev'],budget,'cuda')
        ref=dump_tensors(output/'final_NLL_logits.pt',tensors);del tensors
        io.atomic_json(output/'final_NLL.json',nll);engine.evaluations['final_NLL']={'json_sha256':support.sha(output/'final_NLL.json'),**ref}
        phase='final_generation';progress()
        raw=free_generate(engine.model,[features['dev'][i] for i in p['generation_indices']],budget,'cuda')
        io.atomic_json(output/'final_generation.json',raw)
        engine.evaluations['final_generation']={'path':'final_generation.json','sha256':support.sha(output/'final_generation.json')}
        commit();engine.verify_frozen()
        if support.sha(p['source'])!=SOURCE_SHA or support.core_sha(ROOT)!=support.CORE_SHA:raise RuntimeError('source/core changed')
        baseline=io.load_json(output/'parent_reference_NLL.json')
        result={'status':'TEXT_DECODER_ADDRESSING_V9_COMPLETE','global_attempts':2863+engine.attempts,
            'global_updates':2862+engine.updates,'prior_failed_attempts':1,'new_attempts':engine.attempts,
            'new_updates':engine.updates,'AMP_skips':engine.skips,'new_epochs':4,'training_eos_weight':4.0,'span_UL_beta':0.0,'VP_frozen':True,'cursor':engine.cursor,
            'actual_calls':budget.calls,'allocation_calls':budget.allocation_calls,'exposure':engine.exposure,
            'parent_reference_NLL':baseline['NLL'],'final_NLL':nll['NLL'],'NLL_delta':nll['NLL']-baseline['NLL'],
            'latest':latest,'seconds':time.time()-started,'parent_reference_reused':True,'reference_is_parent_architecture':True,'new_architecture_initial_baseline_measured':False,'E2_final_accessed':False,
            'automatic_promotion':False,'interpretation':'Same-index soft attention addressing with unchanged V8weights at import; initial function changes because prior is active. Initial/final interface diagnostics separate from predicted VP final generation. Beta0/VP frozen, alloldstates preserved; early learning.'}
        io.atomic_json(output/'result.json',result);phase='complete';progress();return 0
    except BaseException as exc:
        phase='stopped_or_error'
        if engine:
            try:commit('emergency')
            except BaseException as save_exc:io.atomic_json(output/'save_error.json',{'error':repr(save_exc)})
        io.atomic_json(output/'incident.json',{'error':repr(exc),'traceback':traceback.format_exc(),
            'allocation_calls':budget.allocation_calls,'global_calls':budget.calls});progress();return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    if a.preflight:
        torch.set_num_threads(2);p,f,t=validate(a.plan,a.output,__file__)
        load_import(p,f,t,RecoveryBudget(p),support.sha(a.plan),'cpu')
        if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
        print('CPU_PREFLIGHT_PASS');return 0
    return run(a.plan.resolve(),a.output.resolve())
if __name__=='__main__':raise SystemExit(main())
