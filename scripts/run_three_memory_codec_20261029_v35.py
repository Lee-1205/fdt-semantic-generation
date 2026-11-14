"""ONE bounded unconsumed suffix of the original broad epoch, same language codec."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.three_memory_codec_v33 import ThreeMemoryCodec
from three_memory_codec_engine_v35 import RemainderCodecEngine, CONTINUATION, CONTRACT, OWNERSHIP
from owt_resource_budget_v29 import RecoveryBudget,RESOURCE_POLICY
from owt_fuzzy_engine_v17_base import AMP_STOP_POLICY
from run_owt_fuzzy_reconstruction_20261028_v12 import finite_orders
from run_three_memory_codec_20261029_v33 import teacher_assessment,reconstruct_inputs,START_STORAGE,CORPUS_MANIFEST,FOCUSED_MANIFEST
import text_generation_support_v1 as support
import learning_io_v2 as io

SCHEMA='fdt_three_memory_language_codec_v35'
LIMITS={'attempts':689,'batch':8,'wall':480,'reserve':90,'max_calls':6000,
    'max_output_bytes':6*1024**3,'start_free_bytes':35*1024**3,'min_gpu_free_mib':12288}
COST={'training':2067,'focused_final_teacher':32,'focused_final_BOS_max':2080,
    'held_final_teacher':128,'held_final_BOS_max':520,'held16_three_memory_ablations':12,'total_max':4839}
SOURCE_SHA='f75165174549116d6c731d0bac9a867187120896c64c6aecc628b7910a1c2cfb'
SOURCE_PLAN='7f0014a8dc35a0120f0cc4d663330baa3ea7633e858cee9a7a6fb065b65877e0'
SOURCE_ALGORITHM='e2c596a7ac33d4480fe9a7ba4aa1ec30df92727b81b2d3caf728a297e97cf05e'

def validate(path,output,worker=None):
    p=io.load_json(path)
    if p.get('schema')!=SCHEMA or type(p.get('authorized')) is not bool or p.get('limits')!=LIMITS:raise ValueError('actual broad schema/caps')
    if Path(p['root']).resolve()!=ROOT or Path(p['output']).resolve()!=Path(output).resolve():raise ValueError('root/output')
    for k,v in [('codec_contract',CONTRACT),('ownership_policy',OWNERSHIP),('continuation_policy',CONTINUATION),
        ('resource_policy',RESOURCE_POLICY),('amp_stop_policy',AMP_STOP_POLICY),('cost_contract',COST),('start_storage_contract',START_STORAGE)]:
        if p.get(k)!=v:raise ValueError('actual '+k)
    if p.get('epochs')!=1 or p.get('order_seed_range')!=[34001,34001]:raise ValueError('ONE new finite epoch')
    if any(p.get(k) is not False for k in ('reasoning_diffusion_training','automatic_source_copy','E2_final_accessed')):raise ValueError('codec only')
    if (p.get('source_sha'),p.get('source_plan_sha'),p.get('source_algorithm'))!=(SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM):raise ValueError('partial V34 source')
    if (p.get('source_global_attempts'),p.get('source_global_updates'))!=(107827,107789):raise ValueError('source historical counters')
    if p['deadline'] is not None and (p['deadline']-p['allocation_start']!=480 or p['cutoff']!=p['deadline']-90):raise ValueError('NEW fixed clock')
    for file,sha in p['files'].items():
        if support.sha(file)!=sha:raise ValueError('pin changed '+file)
    if worker and str(Path(worker).resolve()) not in p['files']:raise ValueError('actual worker pin')
    if support.core_sha(ROOT)!=support.CORE_SHA or p['files'].get(str(Path(p['source']).resolve()))!=SOURCE_SHA:raise ValueError('source/core')
    a,q=Path(p['artifact']),Path(p['focused_artifact'])
    if support.sha(a/'manifest.json')!=CORPUS_MANIFEST or support.sha(q/'manifest.json')!=FOCUSED_MANIFEST:raise ValueError('immutable corpus')
    bf,by,docs=[io.load_json(a/(name+'.json')) for name in ('features','targets','documents')]
    ff,fy=io.load_json(q/'features.json'),io.load_json(q/'targets.json')
    if (len(bf['train']),len(bf['dev']),len(docs['train']),len(docs['dev']),len(ff))!=(32768,256,8192,64,64):raise ValueError('actual corpus rows/docs')
    if {d['normalized_sha256'] for d in docs['train']} & {d['normalized_sha256'] for d in docs['dev']}:raise ValueError('split collision')
    for f,y in [(bf['train'],by['train']),(bf['dev'],by['dev']),(ff,fy)]:
        if len(f)!=len(y):raise ValueError('paired rows')
        for x,t in zip(f,y):
            if set(x)!={'id','prompt'} or x['id']!=t['id'] or x['prompt']!=t['target'] or t['target_ids']!=list(x['prompt'].encode())+[256]:raise ValueError('feature-only SAME input')
    orders=finite_orders(32768,[34001])[3407:]
    exposure={'epoch_total_training_docs':8192,'epoch_total_training_rows':32768,'consumed_before_rows':27256,'unique_training_rows':5512,'selected_rows':5512,'target_UTF8_bytes':492660,'origin_finite_epochs':1}
    if p.get('training_exposure')!=exposure or sum(len(bf['train'][i]['prompt'].encode()) for batch in orders for i in batch)!=492660:raise ValueError('exact finite byte exposure')
    if p.get('generation_indices')!=[4*i for i in range(16)]:raise ValueError('same observed held panel')
    source_p=io.load_json(Path(p['source_plan_path']))
    if p['external_closed_assessment_calls']!=source_p['external_closed_assessment_calls']:raise ValueError('preserved external ledger')
    return p,dict(bf,focused=ff),dict(by,focused=fy),orders

def load_import(p,features,targets,orders,budget,plan_sha,device):
    state=torch.load(p['source'],weights_only=True,map_location='cpu')
    if support.state_sha(state)!=SOURCE_ALGORITHM or (state['attempts'],state['updates'],state['cursor'])!=(3407,3406,27256):raise ValueError('actual complete source')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(835);model=ThreeMemoryCodec(Config(**state['core']))
    engine=RemainderCodecEngine(model,features['train'],targets['train'],orders,budget,plan_sha,device,
        source_parameter_groups=state['optimizer_parameter_names'][:3],data_revision=CORPUS_MANIFEST)
    report=engine.import_v34_parent(state,SOURCE_PLAN,SOURCE_SHA)
    if sum(x.numel() for x in model.parameters())!=111732306 or len(engine.optimizer.state)!=91:raise ValueError('unchanged model/ALL91 Adam')
    return engine,report

def run(path,output):
    p,f,y,orders=validate(path,output,__file__)
    if not p['authorized'] or p['deadline'] is None:raise PermissionError('new plan not sealed')
    if (output/'progress.json').exists():raise FileExistsError('ONE invocation only')
    configure_determinism();torch.set_num_threads(4)
    budget=RecoveryBudget(p);engine=None;latest=None;phase='importing';began=time.time();guard=False
    def progress():
        if engine:engine.worker_stage=phase
        io.atomic_json(output/'progress.json',{'phase':phase,'attempts':engine.attempts if engine else 0,'updates':engine.updates if engine else 0,
            'actual_calls':dict(budget.calls),'allocation_calls':dict(budget.allocation_calls),'native_pid':os.getpid(),
            'parent_pid':os.getppid(),'time_unix':time.time(),'elapsed':time.time()-began,'latest':latest,'resource_scans':budget.resource_scans})
    def commit(kind):
        nonlocal latest
        if kind=='emergency':
            if time.time()>=p['deadline']:raise TimeoutError('owner deadline')
            budget.check_resources(force=True,upcoming_bytes=RESOURCE_POLICY['checkpoint_write_reserve_bytes'])
        else:budget.check(force=True,upcoming_bytes=RESOURCE_POLICY['checkpoint_write_reserve_bytes']);engine.verify_frozen()
        engine.worker_stage=phase;directory=output/kind;directory.mkdir(exist_ok=True)
        latest={**support.save_full(directory,engine.export()),'relative_directory':kind};io.atomic_json(output/'latest.json',latest)
    def assess(label,features,targets,raw=False,disabled=(),modes=('normal','swap','zero')):
        nonlocal phase
        phase=label+'_teacher';progress()
        scores,tensors=teacher_assessment(engine,features,targets,budget,'cuda',disabled,modes,p['generation_indices'] if label=='held_final' else ())
        budget.check(force=True,upcoming_bytes=128*1024**2);torch.save(tensors,output/(label+'_logits.pt'));del tensors
        io.atomic_json(output/(label+'_metrics.json'),scores)
        for suffix in ('logits.pt','metrics.json'):engine.evaluations[label+'_'+suffix]=support.sha(output/(label+'_'+suffix))
        if raw:
            phase=label+'_same_input_BOS';progress()
            io.atomic_json(output/(label+'_reconstruction.json'),reconstruct_inputs(engine,features,budget,'cuda'))
            engine.evaluations[label+'_reconstruction']=support.sha(output/(label+'_reconstruction.json'))
        return scores
    budget.tick=progress;progress()
    try:
        budget.check(force=True)
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        engine,report=load_import(p,f,y,orders,budget,support.sha(path),'cuda');io.atomic_json(output/'import_review.json',report)
        phase='initial_commit';progress();commit('import')
        # Pre-epoch V33 saved final is the baseline; V34 performed no new assessment.
        phase='finite_epoch_remainder_training';progress()
        while engine.attempts<LIMITS['attempts']:
            if time.time()+90>=p['cutoff']:raise TimeoutError('final assessment reserve')
            engine.step()
            progress()
        focused=assess('focused_final',f['focused'],y['focused'],raw=True)
        held=assess('held_final',f['dev'],y['dev'])
        sf=[f['dev'][i] for i in p['generation_indices']];sy=[y['dev'][i] for i in p['generation_indices']]
        phase='held_final_same_input_BOS';progress()
        io.atomic_json(output/'held_final_reconstruction.json',reconstruct_inputs(engine,sf,budget,'cuda'))
        engine.evaluations['held_final_reconstruction']=support.sha(output/'held_final_reconstruction.json')
        ablations={m:assess('held16_without_'+m,sf,sy,disabled=(m,),modes=('normal',)) for m in ('persistent','contextual','exact')}
        phase='final_commit';progress();commit('model')
        if support.sha(p['source'])!=SOURCE_SHA or support.core_sha(ROOT)!=support.CORE_SHA:raise RuntimeError('source/core changed')
        io.atomic_json(output/'result.json',{'status':'THREE_MEMORY_CODEC_V35_COMPLETE','new_attempts':engine.attempts,'new_updates':engine.updates,
            'AMP_skips':engine.skips,'global_attempts':107827+engine.attempts,'global_updates':107789+engine.updates,
            'actual_calls':dict(budget.allocation_calls),'cumulative_calls':dict(budget.calls),'external_closed_assessment_calls':p['external_closed_assessment_calls'],
            'exposure':engine.exposure,'focused_final':focused,'held_final':held,'held16_ablations':ablations,'latest':latest,
            'seconds':time.time()-began,'resource_scans':budget.resource_scans,'reasoning_or_diffusion_training':False,'automatic_source_copy':False,
            'E2_final_accessed':False,'interpretation':'Completes only remaining5512rows of original finite TRAIN epoch, same three-memory codec and ALL91Adam; input reconstruction only, no full vNext/general perfection claim.'})
        phase='complete';progress();return 0
    except BaseException as exc:
        phase='stopped_or_error'
        if engine:
            engine.optimizer.zero_grad(set_to_none=True)
            try:commit('emergency')
            except BaseException as save:io.atomic_json(output/'save_error.json',{'error':repr(save)})
        io.atomic_json(output/'incident.json',{'error':repr(exc),'traceback':traceback.format_exc(),'new_calls':budget.allocation_calls,'cumulative_calls':budget.calls})
        progress();return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    if a.preflight:
        p,f,y,orders=validate(a.plan,a.output,__file__);torch.set_num_threads(2)
        load_import(p,f,y,orders,RecoveryBudget(p),support.sha(a.plan),'cpu')
        if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
        print('CPU_PREFLIGHT_PASS');return 0
    return run(a.plan.resolve(),a.output.resolve())
if __name__=='__main__':raise SystemExit(main())
