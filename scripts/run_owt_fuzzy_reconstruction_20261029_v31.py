"""One finite focused64 OWT reconstruction curriculum, same fuzzy input meaning path."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,copy,sys,time,traceback
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src'),str(R/'scripts')]
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.owt_extraction_tail_v29 import ExtractionTailFuzzyAE,OBJECTIVE,TRAINABILITY
from owt_fuzzy_engine_v31 import ContinuationEngine,cumulative_count,OWNERSHIP
from owt_fuzzy_engine_v17_base import AMP_STOP_POLICY
from owt_resource_budget_v29 import RecoveryBudget,RESOURCE_POLICY
from run_owt_fuzzy_reconstruction_20261028_v12 import evaluate,finite_orders
from run_owt_fuzzy_reconstruction_20261029_v30 import reconstruct_training_sentences
import text_generation_support_v1 as support
import learning_io_v2 as io

SCHEMA='fdt_owt_fuzzy_reconstruction_v31'
LIMITS={'attempts':2048,'batch':8,'wall':900,'reserve':90,'max_calls':12000,
    'max_output_bytes':6*1024**3,'start_free_bytes':40*1024**3,'min_gpu_free_mib':12288}
COST={'training':6144,'focused_initial_final_NLL_controls':64,'focused_initial_final_BOS_max':4160,
    'held_final_NLL_controls':128,'held_final_BOS_max':520,'total_max':11016}
SOURCE_SHA='96480189836964c58153d7d002a27b25b279057af1555cd38950c754d724e504'
SOURCE_PLAN='fce72e7a7e91773ae3ce8bdfbfd45cbd4f5c194e11287eca32bcf93670a12fbe'
SOURCE_ALGORITHM='1ce01437db667d46adebca638fc295e07d793d9268d8158e76094f4c3112ec81'
MANIFEST='bd887866d4dc46357931a84934e5fb6a1fee1e77b11979fd4a3910490663e7f8'

def validate(path,output,worker=None):
    p=io.load_json(path)
    if p.get('schema')!=SCHEMA or type(p.get('authorized')) is not bool or p.get('limits')!=LIMITS:raise ValueError('actual focused64 schema/caps')
    if Path(p['root']).resolve()!=R or Path(p['output']).resolve()!=Path(output).resolve():raise ValueError('root/output')
    if p.get('epochs')!=256 or p.get('cost_contract')!=COST or p.get('reasoning_diffusion_training') is not False or p.get('E2_final_accessed') is not False:raise ValueError('finite AE-only purpose')
    if (p.get('source_sha'),p.get('source_plan_sha'),p.get('source_algorithm'))!=(SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM):raise ValueError('actual source identity')
    if (p.get('source_global_attempts'),p.get('source_global_updates'))!=(100324,100287):raise ValueError('source counters')
    if p.get('training_objective')!=OBJECTIVE or p.get('trainability')!=TRAINABILITY or p.get('ownership_policy')!=OWNERSHIP or p.get('resource_policy')!=RESOURCE_POLICY or p.get('amp_stop_policy')!=AMP_STOP_POLICY or p.get('loss_reduction')!='elementwise_none_then_sum':raise ValueError('unchanged learning contract')
    if p['deadline'] is not None and (p['deadline']-p['allocation_start']!=900 or p['cutoff']!=p['deadline']-90):raise ValueError('fixed new clock')
    if p.get('external_closed_assessment_calls',{}).get('total')!=1936 or p['external_closed_assessment_calls'].get('included_in_source_ledger') is not False:raise ValueError('external ledger')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('pin changed '+name)
    if worker and str(Path(worker).resolve()) not in p['files']:raise ValueError('actual worker pin')
    if support.core_sha(R)!=support.CORE_SHA or p['files'].get(str(Path(p['source']).resolve()))!=SOURCE_SHA:raise ValueError('source/core')
    a=Path(p['artifact']);q=Path(p['focused_artifact'])
    if support.sha(q/'manifest.json')!=MANIFEST or p.get('focused_manifest_sha')!=MANIFEST:raise ValueError('immutable focused manifest')
    m=io.load_json(q/'manifest.json');broadf=io.load_json(a/'features.json');broady=io.load_json(a/'targets.json');docs=io.load_json(a/'documents.json')
    if support.sha(a/'manifest.json')!=m['source_manifest_sha256'] or len(broadf['train'])!=32768 or len(broadf['dev'])!=256:raise ValueError('broad corpus unchanged')
    f=io.load_json(q/'features.json');y=io.load_json(q/'targets.json');orders=io.load_json(q/'orders.json')
    for name,h in m['file_SHA256'].items():
        if support.sha(q/name)!=h:raise ValueError('focused data hash')
    selected=sorted(docs['train'][4096:],key=lambda d:(d['normalized_sha256'],d['id']))[:64]
    indices=[]
    for doc in selected:
        candidates=[i for i in range(16384,32768) if broady['train'][i]['document_id']==doc['id']]
        if len(candidates)!=4:raise ValueError('four source spans per selected doc')
        indices.append(min(candidates,key=lambda i:(len(broadf['train'][i]['prompt'].encode('utf-8')),broadf['train'][i]['id'])))
    if indices!=m['selected_train_indices'] or f!=[broadf['train'][i] for i in indices] or y!=[broady['train'][i] for i in indices]:raise ValueError('independent selection/source reconstruction')
    if len({t['document_id'] for t in y})!=64 or {r['id'] for r in f}&{r['id'] for r in broadf['dev']}:raise ValueError('train/held split')
    if {d['normalized_sha256'] for d in selected}&{d['normalized_sha256'] for d in docs['dev']}:raise ValueError('document split')
    for feature,target in zip(f,y):
        if set(feature)!={'id','prompt'} or feature['prompt']!=target['target'] or target['target_ids']!=list(feature['prompt'].encode())+[256]:raise ValueError('feature-only input/target separation')
    if orders!=finite_orders(64,range(391,647)):raise ValueError('finite exact ordering')
    exposure={'unique_training_rows':64,'new_rows':16384,'target_UTF8_bytes':256*sum(len(r['prompt'].encode()) for r in f),'finite_epochs':256}
    if exposure!=p.get('training_exposure') or exposure['target_UTF8_bytes']!=1101824 or p.get('generation_indices')!=[4*i for i in range(16)]:raise ValueError('exact exposure/held panel')
    return p,{'train':f,'dev':broadf['dev']},{'train':y,'dev':broady['dev']},orders

def load_import(p,f,y,orders,budget,plan_sha,device):
    s=torch.load(p['source'],map_location='cpu',weights_only=True)
    if support.state_sha(s)!=SOURCE_ALGORITHM or (s['attempts'],s['updates'],s['cursor'])!=(8192,8188,65536):raise ValueError('actual full source algorithm')
    if (cumulative_count(s,'attempts'),cumulative_count(s,'updates'))!=(100324,100287):raise ValueError('source ancestry')
    model=ExtractionTailFuzzyAE(Config(**s['core']))
    e=ContinuationEngine(model,f['train'],y['train'],orders,budget,plan_sha,device,source_parameter_groups=s['optimizer_parameter_names'])
    report=e.import_parent(s,SOURCE_PLAN,SOURCE_SHA)
    if (e.attempts,e.updates,e.cursor)!=(0,0,0):raise ValueError('new finite allocation')
    return e,report

def run(path,output):
    p,f,y,orders=validate(path,output,__file__)
    if not p['authorized'] or p['deadline'] is None:raise PermissionError('unsealed')
    if (output/'progress.json').exists():raise FileExistsError('ONE invocation only')
    configure_determinism();torch.set_num_threads(4);budget=RecoveryBudget(p);engine=None;latest=None;phase='importing';start=time.time();guard=False
    def progress():
        if engine:engine.worker_stage=phase
        io.atomic_json(output/'progress.json',{'phase':phase,'attempts':engine.attempts if engine else 0,'updates':engine.updates if engine else 0,
            'actual_calls':dict(budget.calls),'allocation_calls':dict(budget.allocation_calls),'native_pid':os.getpid(),'parent_pid':os.getppid(),
            'resource_scans':budget.resource_scans,'time_unix':time.time(),'elapsed':time.time()-start,'latest':latest})
    def commit(kind):
        nonlocal latest
        if kind=='emergency':
            if time.time()>=p['deadline']:raise TimeoutError('owner end')
            budget.check_resources(force=True,upcoming_bytes=RESOURCE_POLICY['checkpoint_write_reserve_bytes'])
        else:budget.check(force=True,upcoming_bytes=RESOURCE_POLICY['checkpoint_write_reserve_bytes']);engine.verify_frozen()
        engine.worker_stage=phase;d=output/kind;d.mkdir(exist_ok=True)
        latest={**support.save_full(d,engine.export()),'relative_directory':kind};io.atomic_json(output/'latest.json',latest)
    def assessment(label,features,targets,raw=True):
        nonlocal phase
        phase=label+'_teacher';progress();scores,tensors=evaluate(engine.model,features,targets,budget,'cuda')
        budget.check(force=True,upcoming_bytes=64*1024**2);torch.save(tensors,output/(label+'_logits.pt'));del tensors
        io.atomic_json(output/(label+'_metrics.json'),scores)
        engine.evaluations[label+'_logits']=support.sha(output/(label+'_logits.pt'))
        engine.evaluations[label+'_metrics']=support.sha(output/(label+'_metrics.json'))
        if raw:
            phase=label+'_same_input_BOS';progress();rows=reconstruct_training_sentences(engine.model,features,budget,'cuda')
            io.atomic_json(output/(label+'_reconstruction.json'),rows);engine.evaluations[label+'_reconstruction']=support.sha(output/(label+'_reconstruction.json'))
        return scores
    budget.tick=progress;progress()
    try:
        budget.check(force=True)
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        engine,report=load_import(p,f,y,orders,budget,support.sha(path),'cuda');engine.worker_stage=phase
        io.atomic_json(output/'import_review.json',{'source_sha256':SOURCE_SHA,'exact_continuation_import':report,'source_global_updates':100287,'new_updates':0,'Adam_states':len(engine.optimizer.state),'source_RNG_restored':True,'architecture_changed':False,'old_budget_reused':False})
        baseline=assessment('focused_initial',f['train'],y['train']);commit('import')
        phase='training';progress()
        while engine.attempts<2048:
            if time.time()+90>=p['cutoff']:raise TimeoutError('final assessment reserve')
            engine.step()
            if engine.attempts==1024:commit('model')
            progress()
        final=assessment('focused_final',f['train'],y['train'])
        held=assessment('held_final',f['dev'],y['dev'],raw=False)
        phase='held_final_same_input_BOS';progress()
        held_raw=reconstruct_training_sentences(engine.model,[f['dev'][i] for i in p['generation_indices']],budget,'cuda')
        io.atomic_json(output/'held_final_reconstruction.json',held_raw);engine.evaluations['held_final_reconstruction']=support.sha(output/'held_final_reconstruction.json')
        phase='final_commit';progress();commit('model')
        if support.sha(p['source'])!=SOURCE_SHA or support.core_sha(R)!=support.CORE_SHA:raise RuntimeError('source/core changed')
        result={'status':'OWT_FUZZY_RECONSTRUCTION_V31_COMPLETE','new_attempts':engine.attempts,'new_updates':engine.updates,'global_attempts':100324+engine.attempts,'global_updates':100287+engine.updates,
            'AMP_skips':engine.skips,'epochs':256,'exposure':engine.exposure,'actual_calls':dict(budget.allocation_calls),'cumulative_calls':dict(budget.calls),
            'external_closed_assessment_calls':p['external_closed_assessment_calls'],'resource_scans':budget.resource_scans,'resource_policy':RESOURCE_POLICY,
            'focused_initial':baseline,'focused_final':final,'held_final':held,'latest':latest,'seconds':time.time()-start,
            'interpretation':'64 existing training spans repeated256finiteepochs, same fuzzy input-meaning-input path and all81 Adam retained. Fit includes memorization; no general-LM/unique semantic target claim. State-only decoding uses only own input meaning and BOS/selfprefix. Original reasoning frozen.',
            'reasoning_or_diffusion_training':False,'E2_final_accessed':False}
        io.atomic_json(output/'result.json',result);phase='complete';progress();return 0
    except BaseException as exc:
        phase='stopped_or_error'
        if engine:
            try:commit('emergency')
            except BaseException as save:io.atomic_json(output/'save_error.json',{'error':repr(save)})
        io.atomic_json(output/'incident.json',{'error':repr(exc),'traceback':traceback.format_exc(),'new_calls':budget.allocation_calls,'cumulative_calls':budget.calls});progress();return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    if a.preflight:
        p,f,y,orders=validate(a.plan,a.output,__file__);torch.set_num_threads(2);load_import(p,f,y,orders,RecoveryBudget(p),support.sha(a.plan),'cpu')
        if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
        print('CPU_PREFLIGHT_PASS');return 0
    return run(a.plan.resolve(),a.output.resolve())
if __name__=='__main__':raise SystemExit(main())
