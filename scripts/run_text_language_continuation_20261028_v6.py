"""Explicit new bounded continuation of saved model/Adam/RNG at global149."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,sys,time,traceback,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.text_language_continuation_v6 import LanguageContinuation,ContinuationConfig
from research.text_corpus_contract_v5 import validate_rows
from text_language_engine_v6 import RecoveryEngine,finite_order
from text_language_runtime_support_v6 import RecoveryBudget
from run_text_language_continuation_20261028_deterministic_v2 import evaluate_nll,free_generate,dump_tensors,generation_indices
import text_generation_support_v1 as support
import learning_io_v2 as io

SCHEMA='fdt_text_language_continuation_v6'
LIMITS={'attempts':260,'batch':8,'micro':4,'wall':600,'reserve':90,'max_calls':14000,
    'max_output_bytes':6*1024**3,'start_free_bytes':40*1024**3,'min_gpu_free_mib':12288}
SOURCE_SHA='b8d29965fa01430e47db68669667ba20a397bee0877e3e0ec563f653413a31e8'
SOURCE_PLAN='bca6d1f591ab317ff34defe5a142f4d28f41eee0cba342e275372424cbc02d32'
SOURCE_ALGORITHM='51b3ec6193e9d5feb02a89fc55f1e5b0e50f8cb3a127c8cb8df7d94f562766f1'
MANIFEST_SHA='5b0ff59fd3985ab825085d0f3e4f5f7312baf5a22b137b62ec4b0b23f66f3f6e'


def validate(plan_path,output,worker=None):
    p=io.load_json(plan_path)
    if p.get('schema')!=SCHEMA or type(p.get('authorized')) is not bool or p.get('limits')!=LIMITS:raise ValueError('schema/caps')
    if Path(p['root']).resolve()!=ROOT or Path(p['output']).resolve()!=Path(output).resolve():raise ValueError('root/output')
    if p.get('source_sha')!=SOURCE_SHA or p.get('source_plan_sha')!=SOURCE_PLAN or p.get('source_algorithm')!=SOURCE_ALGORITHM:raise ValueError('source fullstate identity')
    if p.get('E2_final_accessed') is not False or p.get('source_attempts')!=149 or p.get('global_attempt_cap')!=409:raise ValueError('global allocation/holdout')
    if p.get('manifest_sha')!=MANIFEST_SHA or support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('data/core')
    required={str(Path(p['source']).resolve()):SOURCE_SHA,str((Path(p['artifact'])/'manifest.json').resolve()):MANIFEST_SHA}
    if any(p['files'].get(n)!=h for n,h in required.items()):raise ValueError('required binding')
    for path,h in p['files'].items():
        if support.sha(path)!=h:raise ValueError('pinned file changed '+path)
    if worker and str(Path(worker).resolve()) not in p['files']:raise ValueError('worker not pinned')
    if p['deadline'] is not None and (p['deadline']-p['allocation_start']!=600 or p['cutoff']!=p['deadline']-90):raise ValueError('new immutable clock')
    a=Path(p['artifact']);manifest=io.load_json(a/'manifest.json')
    for name,h in manifest['files'].items():
        if p['files'].get(str((a/name).resolve()))!=h:raise ValueError('artifact binding')
    features=io.load_json(a/'features.json');targets=io.load_json(a/'targets.json');docs=io.load_json(a/'documents.json')
    if {k:len(v) for k,v in features.items()}!={'train':3271,'dev':444}:raise ValueError('data pool')
    validate_rows(docs,features,targets)
    if p['generation_indices']!=generation_indices(targets['dev']):raise ValueError('fixed generation panel')
    return p,features,targets


def load_import(plan,features,targets,budget,plan_sha,device):
    source=torch.load(plan['source'],map_location='cpu',weights_only=True)
    if support.state_sha(source)!=SOURCE_ALGORITHM:raise ValueError('source algorithm digest')
    model=LanguageContinuation(Config(**source['core']),ContinuationConfig(**source['policy']))
    model.load_state_dict(source['model'],strict=True)
    engine=RecoveryEngine(model,features['train'],targets['train'],finite_order(3271),budget,plan_sha,device)
    digest=engine.import_source(source,SOURCE_PLAN)
    if (engine.attempts,engine.updates,engine.cursor)!=(149,148,1192):raise ValueError('import counters')
    return engine,source['core'],digest


def run(plan_path,output):
    p,features,targets=validate(plan_path,output,__file__)
    if not p['authorized'] or p['deadline'] is None:raise PermissionError('unsealed')
    if (output/'progress.json').exists():raise FileExistsError('ONE invocation only')
    configure_determinism();torch.set_num_threads(4)
    budget=RecoveryBudget(p);engine=None;latest=None;phase='importing';guard=False;started=time.time();core=None
    def progress():
        io.atomic_json(output/'progress.json',{'phase':phase,'attempts':engine.attempts if engine else 149,
            'updates':engine.updates if engine else 148,'cursor':engine.cursor if engine else 1192,
            'new_attempts':engine.attempts-149 if engine else 0,'actual_calls':budget.calls,
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
        for name in ('baseline_NLL.json','baseline_NLL_logits.pt','baseline_generation.json'):
            source=Path(p['baseline_run'])/name
            with source.open('rb') as src,(output/name).open('xb') as dest:shutil.copyfileobj(src,dest)
        io.atomic_json(output/'import_review.json',{'BEFORE_AFTER_algorithm_sha256':import_digest,
            'source_sha256':SOURCE_SHA,'global_attempts':149,'global_updates':148,'cursor':1192,
            'new_allocation_calls':budget.allocation_calls,'baseline_reused_from':p['baseline_run'],
            'source_moments_preserved':True,'old_CUDA_rng_restored':True,
            'cache_correction':'no-grad free proposal autocast cache disabled; required trainable gradients checked'})
        commit('import');phase='training';progress()
        while engine.attempts<409:
            if time.time()+90>=p['cutoff']:raise TimeoutError('final evaluation reserve')
            engine.step()
            if engine.attempts in (256,384):commit()
            progress()
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
        baseline=io.load_json(output/'baseline_NLL.json')
        result={'status':'TEXT_LANGUAGE_V6_BOUNDED_COMPLETE','global_attempts':engine.attempts,
            'global_updates':engine.updates,'prior_failed_attempts':1,'new_attempts':engine.attempts-149,
            'new_updates':engine.updates-148,'AMP_skips':engine.skips,'cursor':engine.cursor,
            'actual_calls':budget.calls,'allocation_calls':budget.allocation_calls,'exposure':engine.exposure,
            'baseline_NLL':baseline['NLL'],'final_NLL':nll['NLL'],'NLL_delta':nll['NLL']-baseline['NLL'],
            'latest':latest,'seconds':time.time()-started,'baseline_reused':True,'E2_final_accessed':False,
            'automatic_promotion':False,'interpretation':'One finite selection epoch with prior failed149 preserved; no replay, broad LM ability unproven.'}
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
