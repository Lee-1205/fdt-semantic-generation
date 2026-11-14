"""Same-input sentence reconstruction continuation; actual V21 fullstate and same prefix objective/inference."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.owt_prefix_reconstruction_v19 import PrefixRobustFuzzyAE as FuzzySentenceAE,OBJECTIVE
from owt_fuzzy_engine_v22 import ContinuationEngine,cumulative_count
from run_owt_fuzzy_reconstruction_20261028_v12 import evaluate,generate,finite_orders
from text_language_runtime_support_v6 import RecoveryBudget
import text_generation_support_v1 as support
import learning_io_v2 as io
from owt_fuzzy_engine_v17_base import AMP_STOP_POLICY
from research.text_diffusion_v1 import pack_prompts,decode_bytes
from contextlib import nullcontext

TRAIN_RECONSTRUCTION_INDICES=[4*i for i in range(8)]+[8192+4*i for i in range(8)]

SCHEMA='fdt_owt_fuzzy_reconstruction_v22'
LIMITS={'attempts':8192,'batch':8,'wall':900,'reserve':90,'max_calls':28000,
    'max_output_bytes':6*1024**3,'start_free_bytes':40*1024**3,'min_gpu_free_mib':12288}
SOURCE_SHA='32c8ba9af2381a58147eb6db9c9551211c7f218440ce81d7a175634b4aad82d2'
SOURCE_PLAN='01e73a60bb4f01865e07138e57f2bdec07e735b686d0adc65feb3a3ee880b910'
SOURCE_ALGORITHM='9fdd540d19026309b4542a28ff71f64e67ab949812cb920d31cc2a282e86112e'
COST={'training':24576,'initial_NLL_controls':0,'final_NLL_controls':128,'final_state_reconstruction_max':1552,'final_train_state_reconstruction_max':520,'total_max':26776}

def validate(path,output,worker=None):
    p=io.load_json(path)
    if p.get('schema')!=SCHEMA or type(p.get('authorized')) is not bool or p.get('limits')!=LIMITS:raise ValueError('actual continuation schema/caps')
    if Path(p['root']).resolve()!=ROOT or Path(p['output']).resolve()!=Path(output).resolve():raise ValueError('root/output')
    if p.get('epochs')!=4 or p.get('cost_contract')!=COST or p.get('reasoning_diffusion_training') is not False or p.get('E2_final_accessed') is not False:raise ValueError('finite AE-only contract')
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
    if m['revision']!='79d93d786212f7344586290adb811d4ae6a1762c' or {k:len(v) for k,v in f.items()}!={'train':16384,'dev':256}:raise ValueError('fixed source/pool')
    if {k:len(v) for k,v in d.items()}!={'train':4096,'dev':64} or {k:len(v) for k,v in y.items()}!={'train':16384,'dev':256}:raise ValueError('document/target count')
    if f['dev']!=oldf['dev'] or y['dev']!=oldy['dev'] or d['dev']!=oldd['dev']:raise ValueError('unchanged held documents/rows required')
    if f['train'][:8192]!=oldf['train'] or y['train'][:8192]!=oldy['train'] or d['train'][:2048]!=oldd['train']:raise ValueError('old training prefix changed')
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
    measured={'new_rows':65536,'target_UTF8_bytes':4*sum(len(row['prompt'].encode('utf-8')) for row in f['train']),
        'old_document_rows':32768,'new_document_rows':32768,
        'old_target_UTF8_bytes':4*sum(len(row['prompt'].encode('utf-8')) for row in f['train'][:8192]),
        'new_target_UTF8_bytes':4*sum(len(row['prompt'].encode('utf-8')) for row in f['train'][8192:])}
    if p['training_exposure']!=measured:raise ValueError('actual finite exposure contract')
    if p.get('train_reconstruction_indices')!=TRAIN_RECONSTRUCTION_INDICES or p.get('state_conditioned_same_input_only') is not True:raise ValueError('same-input reconstruction contract')
    if len({y['train'][i]['document_id'] for i in TRAIN_RECONSTRUCTION_INDICES})!=16:raise ValueError('sixteen fixed distinct training documents')
    return p,f,y,finite_orders(16384,range(301,305))

def load_import(p,f,y,orders,budget,plan_sha,device):
    source=torch.load(p['source'],map_location='cpu',weights_only=True)
    if support.state_sha(source)!=SOURCE_ALGORITHM or (source['attempts'],source['updates'],source['cursor'])!=(4096,4095,32768):raise ValueError('source full algorithm/counters')
    model=FuzzySentenceAE(Config(**source['core']))
    engine=ContinuationEngine(model,f['train'],y['train'],orders,budget,plan_sha,device)
    digest=engine.import_parent(source,SOURCE_PLAN,SOURCE_SHA)
    if cumulative_count(source,'updates')!=31436 or cumulative_count(source,'attempts')!=31444:raise ValueError('source cumulative ancestry')
    if (engine.attempts,engine.updates,engine.cursor)!=(0,0,0):raise ValueError('new allocation counters')
    return engine,digest

def reconstruct_training_sentences(model,features,budget,device):
    """Only SAMEINPUT state, BOS and self-produced prefix; no target argument."""
    model.eval();rows_out=[]
    amp=torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
    with torch.no_grad(),amp:
        for start in range(0,len(features),4):
            rows=features[start:start+4]
            if any(set(row)!={'id','prompt'} for row in rows):raise ValueError('feature-only reconstruction')
            tokens=pack_prompts([r['prompt'] for r in rows],128,device)
            budget.charge('encode');meaning=model.extract(tokens)
            reconstructed=model.reconstruct(meaning,budget.charge,'normal').cpu()
            for j,row in enumerate(rows):
                decoded=decode_bytes(reconstructed[j].tolist())
                rows_out.append({'id':row['id'],'source':row['prompt'],'generated':decoded,
                    'reconstruction_exact':decoded['well_formed'] and decoded['text']==row['prompt']})
    return rows_out


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
            'source_local_attempts':4096,'source_local_updates':4095,'source_global_updates':31436,'new_updates':0,'Adam_states':len(engine.optimizer.state),'source_RNG_restored':True,
            'new_allocation_calls':dict(budget.allocation_calls),'architecture_changed':False,'old_deadline_or_budget_reused':False})
        for name in ('metrics.json','logits.pt','generation.json'):
            source=Path(p['baseline_run'])/('final_'+name)
            with source.open('rb') as src,(output/('parent_reference_'+name)).open('xb') as dst:shutil.copyfileobj(src,dst)
        commit('import');phase='training';progress()
        while engine.attempts<8192:
            if time.time()+90>=p['cutoff']:raise TimeoutError('final evaluation reserve')
            engine.step()
            if engine.attempts in (4096,):commit('model')
            progress()
        phase='final_reconstruction';progress();metrics,tensors=evaluate(model,f['dev'],y['dev'],budget,'cuda')
        torch.save(tensors,output/'final_logits.pt');del tensors;io.atomic_json(output/'final_metrics.json',metrics)
        phase='final_state_conditioned_dev_reconstruction';progress();raw=generate(model,[f['dev'][i] for i in p['generation_indices']],budget,'cuda')
        io.atomic_json(output/'final_generation.json',raw)
        phase='final_state_conditioned_train_reconstruction';progress()
        train_raw=reconstruct_training_sentences(model,[f['train'][i] for i in p['train_reconstruction_indices']],budget,'cuda')
        io.atomic_json(output/'final_train_reconstruction.json',train_raw)
        engine.evaluations={'final_train_reconstruction':support.sha(output/'final_train_reconstruction.json'),'final_metrics':support.sha(output/'final_metrics.json'),'final_generation':support.sha(output/'final_generation.json')};commit('model')
        if support.sha(p['source'])!=SOURCE_SHA or support.core_sha(ROOT)!=support.CORE_SHA:raise RuntimeError('source/core changed')
        result={'status':'OWT_FUZZY_RECONSTRUCTION_V22_COMPLETE','new_attempts':engine.attempts,'new_updates':engine.updates,
            'global_attempts':31444+engine.attempts,'global_updates':31436+engine.updates,'AMP_skips':engine.skips,'epochs':4,
            'exposure':engine.exposure,'actual_calls':dict(budget.allocation_calls),'cumulative_calls':dict(budget.calls),
            'parent_reference':io.load_json(output/'parent_reference_metrics.json'),'final':metrics,
            'generation':{m:{'rows':len(v),'exact':sum(x['reconstruction_exact'] for x in v),'UTF8':sum(x['generated']['utf8_valid'] for x in v),'EOS':sum(x['generated']['eos'] for x in v)} for m,v in raw.items()},
            'train_reconstruction':{'rows':len(train_raw),'exact':sum(x['reconstruction_exact'] for x in train_raw),'UTF8':sum(x['generated']['utf8_valid'] for x in train_raw),'EOS':sum(x['generated']['eos'] for x in train_raw)},
            'latest':latest,'seconds':time.time()-start,'reasoning_or_diffusion_training':False,'E2_final_accessed':False,
            'interpretation':'Same original fuzzy-AE architecture; all actual V21 weights/Adam/scaler/RNG and complete nested ancestry retained; same deterministic loss and inference, four new finite epochs on unchanged4096training documents; final fixed16training-sentence reconstruction separately from dev with clean0.5+past-prefix-corruption0.5 CE; same inference, separate reasoning remains frozen, old64dev unchanged. No extra baseline forwards or output-meaning reasoning training.'}
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
