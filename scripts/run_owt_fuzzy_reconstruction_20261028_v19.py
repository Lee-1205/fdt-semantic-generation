"""Deterministic elementwise-CE compatibility fix; actual V18 zero-update fullstate import."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.owt_prefix_reconstruction_v19 import PrefixRobustFuzzyAE as FuzzySentenceAE,OBJECTIVE
from owt_fuzzy_engine_v19 import ContinuationEngine,cumulative_count
from run_owt_fuzzy_reconstruction_20261028_v12 import evaluate,generate,finite_orders
from text_language_runtime_support_v6 import RecoveryBudget
import text_generation_support_v1 as support
import learning_io_v2 as io
from owt_fuzzy_engine_v17_base import AMP_STOP_POLICY

SCHEMA='fdt_owt_fuzzy_reconstruction_v19'
LIMITS={'attempts':2048,'batch':8,'wall':900,'reserve':90,'max_calls':8500,
    'max_output_bytes':6*1024**3,'start_free_bytes':40*1024**3,'min_gpu_free_mib':12288}
SOURCE_SHA='e64642280fa53562d796ae34055e0c46bc69a57f2ad43dbcb0cee1de88f8936d'
SOURCE_PLAN='f02b2e8e023337151e69d93d942ed4c5ca0e65693b6727b21545630772d95613'
SOURCE_ALGORITHM='75701cf038f95088abea169da34a9a67507d20816387f04dca897ec3c4c94a3d'
COST={'training':6144,'initial_NLL_controls':0,'final_NLL_controls':128,'final_free_reconstruction_max':1552,'total_max':7824}

def validate(path,output,worker=None):
    p=io.load_json(path)
    if p.get('schema')!=SCHEMA or type(p.get('authorized')) is not bool or p.get('limits')!=LIMITS:raise ValueError('actual continuation schema/caps')
    if Path(p['root']).resolve()!=ROOT or Path(p['output']).resolve()!=Path(output).resolve():raise ValueError('root/output')
    if p.get('epochs')!=2 or p.get('cost_contract')!=COST or p.get('reasoning_diffusion_training') is not False or p.get('E2_final_accessed') is not False:raise ValueError('finite AE-only contract')
    if p.get('source_sha')!=SOURCE_SHA or p.get('source_plan_sha')!=SOURCE_PLAN or p.get('source_algorithm')!=SOURCE_ALGORITHM:raise ValueError('source identity')
    if p['files'].get(str(Path(p['source']).resolve()))!=SOURCE_SHA or support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('source/core pin')
    for name,h in p['files'].items():
        if support.sha(name)!=h:raise ValueError('pinned file changed '+name)
    if worker and str(Path(worker).resolve()) not in p['files']:raise ValueError('actual worker unpinned')
    if p['deadline'] is not None and (p['deadline']-p['allocation_start']!=900 or p['cutoff']!=p['deadline']-90):raise ValueError('fixed new clock')
    if p.get('amp_stop_policy')!=AMP_STOP_POLICY or p.get('training_objective')!=OBJECTIVE or p.get('loss_reduction')!='elementwise_none_then_sum':raise ValueError('fixed AMP/objective policy')
    a=Path(p['artifact']);base=Path(p['parent_artifact'])
    f=io.load_json(a/'features.json');y=io.load_json(a/'targets.json');d=io.load_json(a/'documents.json');m=io.load_json(a/'manifest.json')
    oldf=io.load_json(base/'features.json');oldy=io.load_json(base/'targets.json');oldd=io.load_json(base/'documents.json')
    if m['revision']!='79d93d786212f7344586290adb811d4ae6a1762c' or {k:len(v) for k,v in f.items()}!={'train':8192,'dev':256}:raise ValueError('fixed source/pool')
    if {k:len(v) for k,v in d.items()}!={'train':2048,'dev':64} or {k:len(v) for k,v in y.items()}!={'train':8192,'dev':256}:raise ValueError('document/target count')
    if f['dev']!=oldf['dev'] or y['dev']!=oldy['dev'] or d['dev']!=oldd['dev']:raise ValueError('unchanged held documents/rows required')
    if f['train'][:4096]!=oldf['train'] or y['train'][:4096]!=oldy['train'] or d['train'][:1024]!=oldd['train']:raise ValueError('old training prefix changed')
    if {r['normalized_sha256'] for r in d['train']}&{r['normalized_sha256'] for r in d['dev']}:raise ValueError('document split leak')
    seen=set()
    for panel in f:
        docs={r['id']:' '.join(r['text'].split()) for r in d[panel]}
        for row,target in zip(f[panel],y[panel]):
            if set(row)!={'id','prompt'} or row['prompt']!=target['target'] or not 48<=len(row['prompt'].encode('utf-8'))<=128:raise ValueError('sentence identity')
            h=support.hashlib.sha256(row['prompt'].encode('utf-8')).hexdigest()
            if h!=row['id'] or h in seen or row['prompt'] not in docs[target['document_id']]:raise ValueError('sentence/source/dedup')
            seen.add(h)
            if target['target_ids']!=list(row['prompt'].encode('utf-8'))+[256] or target['document_EOF_claimed']:raise ValueError('sentence boundary target')
    if p['generation_indices']!=[4*i for i in range(16)]:raise ValueError('same16devdocuments')
    if any(int(row['normalized_sha256'][:8],16)%9==0 for row in d['train']):raise ValueError('prospective document split rule')
    measured={'new_rows':16384,'target_UTF8_bytes':2*sum(len(row['prompt'].encode('utf-8')) for row in f['train']),
        'old_document_rows':8192,'new_document_rows':8192,
        'old_target_UTF8_bytes':2*sum(len(row['prompt'].encode('utf-8')) for row in f['train'][:4096]),
        'new_target_UTF8_bytes':2*sum(len(row['prompt'].encode('utf-8')) for row in f['train'][4096:])}
    if p['training_exposure']!=measured:raise ValueError('actual finite exposure contract')
    return p,f,y,finite_orders(8192,range(271,273))

def load_import(p,f,y,orders,budget,plan_sha,device):
    source=torch.load(p['source'],map_location='cpu',weights_only=True)
    if support.state_sha(source)!=SOURCE_ALGORITHM or (source['attempts'],source['updates'],source['cursor'])!=(1,0,8):raise ValueError('source full algorithm/counters')
    model=FuzzySentenceAE(Config(**source['core']))
    engine=ContinuationEngine(model,f['train'],y['train'],orders,budget,plan_sha,device)
    digest=engine.import_parent(source,SOURCE_PLAN,SOURCE_SHA)
    if cumulative_count(source,'updates')!=21200 or cumulative_count(source,'attempts')!=21204:raise ValueError('source cumulative ancestry')
    if (engine.attempts,engine.updates,engine.cursor)!=(0,0,0):raise ValueError('new allocation counters')
    return engine,digest

def run(path,output):
    p,f,y,orders=validate(path,output,__file__)
    if not p['authorized'] or p['deadline'] is None:raise PermissionError('unsealed')
    if (output/'progress.json').exists():raise FileExistsError('ONE invocation only')
    configure_determinism();torch.set_num_threads(4);budget=RecoveryBudget(p)
    engine=None;latest=None;phase='importing';start=time.time();guard=False
    def progress():
        io.atomic_json(output/'progress.json',{'phase':phase,'attempts':engine.attempts if engine else 0,'updates':engine.updates if engine else 0,
            'actual_calls':dict(budget.calls),'allocation_calls':dict(budget.allocation_calls),'native_pid':os.getpid(),'parent_pid':os.getppid(),
            'time_unix':time.time(),'elapsed':time.time()-start,'latest':latest})
    def commit(kind):
        nonlocal latest
        if kind=='emergency':
            if time.time()>=p['deadline']:raise TimeoutError('new owner end')
        else:budget.check();engine.verify_frozen()
        d=output/kind;d.mkdir(exist_ok=True);latest={**support.save_full(d,engine.export()),'relative_directory':kind};io.atomic_json(output/'latest.json',latest)
    budget.tick=progress;progress()
    try:
        budget.check()
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        engine,digest=load_import(p,f,y,orders,budget,support.sha(path),'cuda');model=engine.model
        io.atomic_json(output/'import_review.json',{'source_sha256':SOURCE_SHA,'all_algorithm_components_sha256':digest,
            'source_local_attempts':1,'source_local_updates':0,'source_global_updates':21200,'new_updates':0,'Adam_states':len(engine.optimizer.state),'source_RNG_restored':True,
            'new_allocation_calls':dict(budget.allocation_calls),'architecture_changed':False,'old_deadline_or_budget_reused':False})
        for name in ('metrics.json','logits.pt','generation.json'):
            source=Path(p['baseline_run'])/('final_'+name)
            with source.open('rb') as src,(output/('parent_reference_'+name)).open('xb') as dst:shutil.copyfileobj(src,dst)
        commit('import');phase='training';progress()
        while engine.attempts<2048:
            if time.time()+90>=p['cutoff']:raise TimeoutError('final evaluation reserve')
            engine.step()
            if engine.attempts in (1024,):commit('model')
            progress()
        phase='final_reconstruction';progress();metrics,tensors=evaluate(model,f['dev'],y['dev'],budget,'cuda')
        torch.save(tensors,output/'final_logits.pt');del tensors;io.atomic_json(output/'final_metrics.json',metrics)
        phase='final_free_reconstruction';progress();raw=generate(model,[f['dev'][i] for i in p['generation_indices']],budget,'cuda')
        io.atomic_json(output/'final_generation.json',raw)
        engine.evaluations={'final_metrics':support.sha(output/'final_metrics.json'),'final_generation':support.sha(output/'final_generation.json')};commit('model')
        if support.sha(p['source'])!=SOURCE_SHA or support.core_sha(ROOT)!=support.CORE_SHA:raise RuntimeError('source/core changed')
        result={'status':'OWT_FUZZY_RECONSTRUCTION_V19_COMPLETE','new_attempts':engine.attempts,'new_updates':engine.updates,
            'global_attempts':21204+engine.attempts,'global_updates':21200+engine.updates,'AMP_skips':engine.skips,'epochs':2,
            'exposure':engine.exposure,'actual_calls':dict(budget.allocation_calls),'cumulative_calls':dict(budget.calls),
            'parent_reference':io.load_json(output/'parent_reference_metrics.json'),'final':metrics,
            'generation':{m:{'rows':len(v),'exact':sum(x['reconstruction_exact'] for x in v),'UTF8':sum(x['generated']['utf8_valid'] for x in v),'EOS':sum(x['generated']['eos'] for x in v)} for m,v in raw.items()},
            'latest':latest,'seconds':time.time()-start,'reasoning_or_diffusion_training':False,'E2_final_accessed':False,
            'interpretation':'Same original fuzzy-AE architecture; all actual V18 zero-update emergency weights/Adam/scaler/RNG and failed history retained; original V17 weights/moments unchanged; deterministic elementwise CE sum fixed, two new finite epochs on unchanged2048training documents with clean0.5+past-prefix-corruption0.5 CE; same inference, separate reasoning remains frozen, old64dev unchanged. No extra baseline forwards or output-meaning reasoning training.'}
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
        p,f,y,orders=validate(a.plan,a.output,__file__);torch.set_num_threads(2)
        load_import(p,f,y,orders,RecoveryBudget(p),support.sha(a.plan),'cpu')
        if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
        print('CPU_PREFLIGHT_PASS');return 0
    return run(a.plan.resolve(),a.output.resolve())
if __name__=='__main__':raise SystemExit(main())
