"""One bounded V77 balanced finite1920 head exposures over1024 unique rows.

Warm V75 linked NN exactly; write active7/runtime against standalone V74 base.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,collections,ctypes,json,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64,EXACT_POLICY
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.update_balance_objective_v77 import OBJECTIVE,COEFFICIENTS,decoder_losses
from research import linked_runtime_v75 as linked
from update_balance_engine_v77 import RetentionEngine,SCHEMA
from update_balance_assess_v77 import assess,source_panel
from factual_cycle_long_eval_v70 import clone_for_long
from run_ordered_fuzzy_20261111_v55 import Budget as BaseBudget
import learning_io_v2 as io
import text_generation_support_v1 as support

PARENT_SHA='f01e851284df0bd971498ee709b81158abb4ce157bf7e19578c146614411c124'
ALGORITHM_SHA='88e1fd3546bdc694e3c2390065c7f6bb586632c26bdd200226ca788fbd4ea623'
STORAGE_PARENT_SHA='734a2eed6526f86526f203d6389f6b1a778b50f9273ae9aaf90d4c801998dab7'
OUTPUT_CAP=5*1024**3//4
START_FREE=108*1024**3+OUTPUT_CAP

def validate(path,output):
    p=io.load_json(path)
    fixed=dict(schema='fdt_update_balance_v77',attempts=1920,epochs=1,batch=1,capacity=16384,amp_dtype='bfloat16',wall_seconds=900,max_calls=7000,cost=6752,
        max_output_bytes=OUTPUT_CAP,start_free_bytes=START_FREE,cycle_sha=PARENT_SHA,cycle_algorithm_sha=ALGORITHM_SHA,objective=OBJECTIVE)
    if any(p.get(k)!=v for k,v in fixed.items()) or Path(p['output']).resolve()!=output.resolve():raise ValueError('actual V77 fixed schema/output')
    if p['authorized'] and (p['deadline']-p['start']!=900 or p['cutoff']!=p['deadline']-120):raise ValueError('new V77 fixed clock')
    for file,sha in p['files'].items():
        if support.sha(file)!=sha:raise ValueError('input pin '+file)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core changed')
    d=Path(p['data']);rows=io.load_json(d/'train_features.json');dev=io.load_json(Path(p['eval_data'])/'features.json');orders=io.load_json(Path(p['balance_data'])/'orders.json')
    if len(rows)!=1024 or len(dev)!=64 or len(orders)!=1920 or any(len(o)!=1 or not 0<=o[0]<1024 for o in orders) or any(set(r)!={'id','prompt'} for r in rows+dev):
        raise ValueError('finite repeated-exposure feature-only data')
    labels=io.load_json(d/'train_targets.json');multiplicity=collections.Counter(o[0] for o in orders)
    if any(multiplicity[i]!={'signed':1,'unsigned':6,'OWT':3}[t.get('mode','OWT')] for i,t in enumerate(labels)):
        raise ValueError('1920 fixed signed1/unsigned6/OWT3 order')
    return p,rows,dev,orders

class Budget(BaseBudget):
    def __init__(self,p):
        super().__init__(p);self.calls=dict(encode=0,readout=0,stream_chunk=0,denoise=0);self.block_work=0;self.data_sha=support.sha(Path(p['data'])/'manifest.json')
        self.critical_targets={t['id']:t['critical_spans'] for t in io.load_json(Path(p['data'])/'train_targets.json')}
        self.frozen_symbol_profile=io.load_json(Path(p['balance_data'])/'symbol_weight_source.json')
        self.parent_reference=dict(cycle_path=p['cycle_source'],cycle_SHA=PARENT_SHA,parent7Adam='V75 preserved; new fresh7Adam LR1e-5',encoder_binding='fully frozen V74/V73',exact_policy=EXACT_POLICY)
    def charge(self,kind):
        self.check()
        if kind not in ('encode','readout','stream_chunk') or sum(self.calls.values())>=7000:raise ValueError('V77 fixed call cap')
        self.calls[kind]+=1

def initialize(p,rows,orders,budget,sha,device):
    parent=linked.load_verified(p['cycle_source'],PARENT_SHA,ROOT,support.state_sha)
    if support.state_sha(parent)!=ALGORITHM_SHA or parent['schema']!='fdt_live_critical_retention_v75_fullstate' or parent['updates']!=640 or len(parent['optimizer']['state'])!=7:
        raise ValueError('immutable complete linked V75 parent')
    if not torch.equal(torch.tensor(budget.frozen_symbol_profile['weights']),parent['symbol_weights']) or budget.frozen_symbol_profile['source_SHA']!=PARENT_SHA:
        raise ValueError('frozen V75 TRAIN symbol profile')
    storage=torch.load(p['storage_parent'],map_location='cpu',weights_only=True)
    if support.sha(p['storage_parent'])!=STORAGE_PARENT_SHA or storage.get('schema')!='fdt_context_verbalizer_v74_fullstate':
        raise ValueError('mandatory standalone V74 storage base')
    active=set(parent['optimizer_names'][0])
    if any(not torch.equal(v,storage['model'][k]) for k,v in parent['model'].items() if k not in active):
        raise ValueError('V75 frozen upstream differs from standalone V74 base')

    torch.manual_seed(77113);random.seed(77113);np.random.seed(77113)
    if device=='cuda':torch.cuda.manual_seed_all(77113)
    enc=FuzzyCodecV64(Config(**parent['core']),input_capacity=16384);enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks();model=FuzzyRoundTripV64(enc,OrderedFuzzyReadout());model.load_state_dict(parent['model'],strict=True)
    if support.state_sha(model.state_dict())!=support.state_sha(parent['model']):raise ValueError('all tensors warm exact')
    engine=RetentionEngine(model,rows,orders,budget,sha,device)
    if engine.objective_report!=p['objective_report'] or engine.embedding_sha!=parent['embedding_sha']:raise ValueError('objective/frozen source epoch signature')
    engine.storage_parent_model=storage['model'];engine.storage_parent_reference=dict(path=str(Path(p['storage_parent']).resolve()),SHA=STORAGE_PARENT_SHA,model_key='model',model_SHA=support.state_sha(storage['model']))
    report=dict(all_parent_tensors_exact=True,all_old_Adam_exact=True,parent7Adam_scaler_ALLRNG_preserved=True,V75_linked_parent_reconstructed=True,V74_standalone_storage_parent=True,fresh7Adam=not bool(engine.optimizer.state),encoder_binding_frozen=True,
        active_readout_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),active_tensors=len(engine.names),engine_schema=SCHEMA,objective_report=engine.objective_report)
    return engine,report

def save_linked(engine,file):
    engine.budget.check(write_bytes=64*1024**2);envelope=linked.pack(engine.export(),engine.storage_parent_model,engine.storage_parent_reference,engine.names,support.state_sha)
    return linked.save_verified(file,envelope,ROOT,support.state_sha)

def run(path,output,preflight=False):
    p,rows,dev,orders=validate(path,output);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    budget=Budget(p);engine,report=initialize(p,rows,orders,budget,support.sha(path),'cpu' if preflight else 'cuda')
    if preflight:
        budget.output=path.parent/'cpu_preflight_owned/runtime';budget.output.mkdir()
        meta=save_linked(engine,budget.output/'initial.pt');state=linked.load_verified(meta['path'],meta['SHA'],ROOT,support.state_sha);engine.restore(state)
        initial_digest=support.state_sha(engine.export());assert initial_digest==meta['reconstructed_fullstate_SHA']
        engine.cache_next();t=engine.packet(0);c=engine.extract_live(rows[0]);budget.charge('readout');z=engine.model.verbalize(c)
        loss,_=decoder_losses(z,t['labels'],engine.symbol_weights,t['critical_mask'],t['teacher_gold_probability'],t['teacher_correct']);loss.backward();named=dict(engine.model.named_parameters())
        if any(named[n].grad is None for n in engine.names) or any(v.grad is not None for n,v in named.items() if n not in engine.names):raise ValueError('actual final7 gradient ownership')
        engine.optimizer.zero_grad(set_to_none=True);engine.verify_frozen();meta=save_linked(engine,budget.output/'partial_teacher.pt');state=linked.load_verified(meta['path'],meta['SHA'],ROOT,support.state_sha);engine.restore(state)
        if support.state_sha(engine.export())!=meta['reconstructed_fullstate_SHA'] or torch.cuda.is_initialized():raise ValueError('actual linked partial-teacher restore')
        from saved_commit_reference_v76_v2 import resolve_saved_commit
        for case in p['CPU_reference_cases']:
            row=io.load_json(case['data'])[case['index']]
            packet=torch.load(case['raw'],map_location='cpu',weights_only=True)
            resolve_saved_commit(packet,row,engine.embedding_sha,p['files'],ROOT,engine.model.encoder.c.z_dim)
        wrapped=clone_for_long(engine.model,'cpu');del wrapped
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(report,fullsize_engine_restore=True,actual_linked_initial_and_partial_teacher_exact=True,actual_only7_gradient=True,live_source_reencoded=True,
            no_feature_cache=True,actual_saved_onelevel_reference_checked=True,long_wrapper_all_tensors_exact=True,CUDA_initialized=False,actual_encoder_forwards=2,actual_readout_forwards=2,optimizer_steps=0,Exact_entries0=True)));print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('one fresh V77 trial')
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()));began=time.time();latest=None;guard=False
    def progress(phase):
        engine.phase=phase;io.atomic_json(output/'progress.json',dict(phase=phase,cache_cursor=engine.cache_cursor,cache_total=1024,head_total=1920,attempts=engine.attempts,updates=engine.updates,cursor=engine.cursor,
            calls=budget.calls,block_work=budget.block_work,time=time.time(),native_pid=os.getpid(),parent_pid=os.getppid(),latest=latest))
    def save(directory,emergency=False):
        nonlocal latest
        budget.check(write_bytes=64*1024**2,emergency=emergency);folder=output/directory;folder.mkdir(exist_ok=True)
        # The outer check permits a bounded STOP save after cutoff. Serialization
        # itself does no model work and its parent and final digest are verified.
        envelope=linked.pack(engine.export(),engine.storage_parent_model,engine.storage_parent_reference,engine.names,support.state_sha)
        m=linked.save_verified(folder/'runtime.pt',envelope,ROOT,support.state_sha)
        latest=dict(file='runtime.pt',directory=directory,sha256=m['SHA'],bytes=m['bytes'],algorithm_sha256=m['reconstructed_fullstate_SHA'],attempts=engine.attempts,updates=engine.updates,
            sampler_cursor=engine.cursor,full_state_verified=True,standalone=False,mandatory_parent=m['mandatory_parent'],exact_policy=EXACT_POLICY)
        io.atomic_json(folder/'runtime.json',latest);io.atomic_json(output/'latest.json',latest)
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        progress('initial_linked_runtime');save('initial')
        while engine.cache_cursor<len(rows):
            if time.time()+180>=p['cutoff']:raise TimeoutError('teacher phase reserve')
            engine.cache_next()
            if engine.cache_cursor%8==0:progress('loss_only_teacher_no_feature_cache')
        source_panel(engine,dev,'new_baseline',progress);progress('teacher_complete_before_live_head');save('teacher_complete')
        while engine.attempts<len(orders):
            if time.time()+120>=p['cutoff']:raise TimeoutError('head final-eval reserve')
            engine.step()
            if engine.attempts%8==0:progress('live_source_critical_retention_head')
        engine.verify_frozen();progress('final_assessment');assess(engine,dev,progress);engine.verify_frozen();progress('final_linked_runtime');save('model')
        if support.sha(p['cycle_source'])!=PARENT_SHA or support.sha(p['storage_parent'])!=STORAGE_PARENT_SHA:raise ValueError('parent changed')
        io.atomic_json(output/'result.json',dict(status='UPDATE_BALANCE_V77_COMPLETE',attempts=engine.attempts,updates=engine.updates,AMP_skips=engine.skips,cache_cursor=engine.cache_cursor,
            exposure=engine.exposure,calls=budget.calls,block_work=budget.block_work,evaluations=engine.evaluations,latest=latest,seconds=time.time()-began,
            frozen_upstream_SHA=engine.embedding_sha,initial_teacher_head_SHA=engine.teacher_sha,trainable_parameters=2908192,active_tensors=7,encoder_binding_updates=0,
            training_GPU_peak_bytes=max(h['GPU_peak_bytes'] for h in engine.history),teacher_GPU_peak_bytes=engine.teacher_GPU_peak_bytes,Exact_entries=0,copy_bytes=0,denoise_calls=0,limitations=p['limitations']))
        progress('complete');return 0
    except BaseException as error:
        engine.optimizer.zero_grad(set_to_none=True);io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc()))
        try:save('emergency',True)
        except BaseException as failure:io.atomic_json(output/'save_error.json',dict(error=repr(failure)))
        progress('stopped');return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--preflight',action='store_true');v=a.parse_args();raise SystemExit(run(v.plan.resolve(),v.output.resolve(),v.preflight))
