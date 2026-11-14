"""Fixed encoder+binding, real Commit caching, then one decoder-only epoch."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,gc,io as memoryio,json,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64,EXACT_POLICY
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.frozen_verbalizer_objective_v72 import decoder_losses
from frozen_verbalizer_engine_v72 import FrozenEngine,SCHEMA
from frozen_verbalizer_assess_v72 import new_dev_baseline,final_assessment,full_path_spotchecks
from run_ordered_fuzzy_20261111_v55 import Budget as BaseBudget
import learning_io_v2 as io
import text_generation_support_v1 as support

PARENT_SHA='809414362bc7ac93b85afbadbb9cba9af67730a74d0ebfe0b744c78ce872f3d4'
ALGORITHM_SHA='f2baf2af63d1e47201206c87e43d53140c856b63022e54084443446b785a6e2f'

def validate(path,output):
    p=io.load_json(path)
    expected=dict(schema='fdt_frozen_verbalizer_v72',output=str(output.resolve()),cycle_sha=PARENT_SHA,cycle_algorithm_sha=ALGORITHM_SHA,
        wall_seconds=900,max_calls=2200,cost=2110,attempts=640,epochs=1,capacity=16384,max_output_bytes=4*1024**3,exact_policy=EXACT_POLICY)
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('actual V72 fixed contract')
    if p['authorized'] and (p['deadline']-p['start']!=900 or p['cutoff']!=p['deadline']-120):raise ValueError('new V72 clock')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('input pin '+name)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core changed')
    data=Path(p['data']);rows=io.load_json(data/'train_features.json');dev=io.load_json(data/'dev_features.json');orders=io.load_json(data/'orders.json')
    if len(rows)!=640 or len(dev)!=32 or len(orders)!=640 or sorted(r[0] for r in orders)!=list(range(640)) or any(set(r)!={'id','prompt'} for r in rows+dev):raise ValueError('finite separated data')
    return p,rows,dev,orders

class Budget(BaseBudget):
    def __init__(self,p):
        super().__init__(p);self.calls=dict(encode=0,readout=0,denoise=0);self.data_sha=support.sha(Path(p['data'])/'manifest.json')
        self.critical_targets={r['id']:r['critical_spans'] for r in io.load_json(Path(p['data'])/'train_targets.json')};self.eval_rows=io.load_json(Path(p['data'])/'dev_features.json')
        self.parent_reference=dict(cycle_path=p['cycle_source'],cycle_SHA=p['cycle_sha'],parent30Adam='immutable preserved; fresh7Adam',
            encoder_binding='all frozen, actual per-input memories still formed',exact_policy=EXACT_POLICY)
    def charge(self,kind):
        self.check()
        if kind not in ('encode','readout') or sum(self.calls.values())>=2200:raise ValueError('V72 fixed model call cap')
        self.calls[kind]+=1

def initialize(p,rows,orders,budget,plan_sha,device):
    parent=torch.load(p['cycle_source'],map_location='cpu',weights_only=True)
    if support.state_sha(parent)!=ALGORITHM_SHA or parent['schema']!='fdt_live_fuzzy_roundtrip_v70_fullstate' or len(parent['optimizer']['state'])!=30 or parent['updates']!=512:raise ValueError('immutable full V70 source')
    torch.manual_seed(72112);random.seed(72112);np.random.seed(72112)
    if device=='cuda':torch.cuda.manual_seed_all(72112)
    enc=FuzzyCodecV64(Config(**parent['core']),input_capacity=16384);enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks()
    enc.load_state_dict({k[8:]:v for k,v in parent['model'].items() if k.startswith('encoder.')},strict=True)
    head=OrderedFuzzyReadout();head.load_state_dict({k[8:]:v for k,v in parent['model'].items() if k.startswith('readout.')},strict=True)
    model=FuzzyRoundTripV64(enc,head)
    if support.state_sha(model.state_dict())!=support.state_sha(parent['model']):raise ValueError('all source tensors warm exact')
    engine=FrozenEngine(model,rows,orders,budget,plan_sha,device)
    if engine.objective_report!=p['objective_report']:raise ValueError('TRAIN objective signature')
    report=dict(all_parent_tensors_exact=True,all_old_Adam_exact=True,parent30Adam_scaler_ALLRNG_preserved=True,
        fresh7Adam=not bool(engine.optimizer.state),active_readout_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        active_tensor_count=len(engine.names),encoder_and_role_projection_frozen=True,source_embedding_SHA=engine.embedding_sha,teacher_readout_SHA=engine.teacher_sha,
        engine_schema=SCHEMA,objective_report=engine.objective_report)
    return engine,report

def run(path,output,preflight=False):
    p,rows,dev,orders=validate(path,output);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    budget=Budget(p);device='cpu' if preflight else 'cuda';engine,report=initialize(p,rows,orders,budget,support.sha(path),device)
    if preflight:
        # One actual TRAIN source: verifies real fullsize cache/gradient/state,
        # not a performance assessment or an actual optimizer trial.
        budget.output=path.parent/'cpu_preflight_owned/runtime_cache';budget.output.mkdir()
        engine.cache_next();f,t=engine.packet(0);c=DifferentiableCodecCommit.from_semantic(f['semantic'],f['memory_revisions'],engine.embedding_sha)
        budget.charge('readout');z=engine.model.verbalize(c);loss,_=decoder_losses(z,t['labels'],engine.symbol_weights,t['critical_mask'],t['teacher_gold_probability'],t['teacher_correct']);loss.backward()
        named=dict(engine.model.named_parameters())
        if any(named[n].grad is None for n in engine.names) or any(v.grad is not None for n,v in named.items() if n not in engine.names):raise ValueError('actual fixed-boundary gradient ownership')
        engine.optimizer.zero_grad(set_to_none=True);engine.verify_frozen();s=engine.export();digest=support.state_sha(s)
        stream=memoryio.BytesIO();torch.save(s,stream);stream.seek(0);engine.restore(torch.load(stream,map_location='cpu',weights_only=True))
        if support.state_sha(engine.export())!=digest or torch.cuda.is_initialized():raise ValueError('actual fullsize partial-cache restore')
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(report,fullsize_engine_restore=True,actual_cache_partial_restore=True,actual_only7_gradient=True,
            CUDA_initialized=False,actual_encoder_forward=1,actual_readout_forward=2,optimizer_steps=0,Exact_entries0=True)))
        print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('ONE new source cache/readout trial')
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()));began=time.time();latest=None;guard=False
    def progress(phase):
        engine.phase=phase;io.atomic_json(output/'progress.json',dict(phase=phase,cache_cursor=engine.cache_cursor,cache_total=len(rows),attempts=engine.attempts,updates=engine.updates,
            cursor=engine.cursor,calls=budget.calls,time=time.time(),native_pid=os.getpid(),parent_pid=os.getppid(),latest=latest))
    def save(directory,emergency=False):
        nonlocal latest
        budget.check(write_bytes=650*1024**2,emergency=emergency);folder=output/directory;folder.mkdir(exist_ok=True)
        latest=dict(support.save_full(folder,engine.export()),directory=directory,exact_policy=EXACT_POLICY);io.atomic_json(output/'latest.json',latest)
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        progress('initial_fixed_boundary');save('initial')
        engine.evaluations['parent_baselines']=io.load_json(Path(p['baseline_run'])/'result.json')
        while engine.cache_cursor<len(rows):
            if time.time()+180>=p['cutoff']:raise TimeoutError('cache phase fixed reserve')
            engine.cache_next()
            if engine.cache_cursor%8==0:progress('frozen_source_teacher_cache')
        engine.new_dev_refs=new_dev_baseline(engine,dev,io.load_json(Path(p['data'])/'dev_targets.json'),progress)
        progress('cache_complete_before_head_updates');save('cache_complete')
        while engine.attempts<len(orders):
            if time.time()+90>=p['cutoff']:raise TimeoutError('head phase final-assessment reserve')
            engine.step()
            if engine.attempts in (1,320):engine.verify_frozen();save('model')
            if engine.attempts%8==0:progress('seven_tensor_verbalizer_training')
        engine.verify_frozen();progress('fixed_upstream_final_readout_assessment');records=final_assessment(engine,dev,engine.new_dev_refs,progress)
        full_path_spotchecks(engine,dev,records,progress);engine.verify_frozen();progress('final_verbalizer_commit');save('model')
        if support.sha(p['cycle_source'])!=PARENT_SHA:raise ValueError('parent changed')
        io.atomic_json(output/'result.json',dict(status='FROZEN_VERBALIZER_V72_COMPLETE',attempts=engine.attempts,updates=engine.updates,AMP_skips=engine.skips,
            cache_cursor=engine.cache_cursor,exposure=engine.exposure,calls=budget.calls,evaluations=engine.evaluations,latest=latest,seconds=time.time()-began,
            embedding_SHA_before_after=engine.embedding_sha,initial_teacher_readout_SHA=engine.teacher_sha,
            trainable_parameters=report['active_readout_parameters'],active_tensors=7,encoder_weight_updates=0,role_projection_updates=0,
            training_GPU_peak_bytes=max(h['GPU_peak_bytes'] for h in engine.history),cache_GPU_peak_bytes=max(r['GPU_peak_bytes'] for r in engine.cache_refs),
            Exact_entries=0,copy_bytes=0,denoise_calls=0,limitations=p['limitations']))
        progress('complete');return 0
    except BaseException as error:
        engine.optimizer.zero_grad(set_to_none=True);io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc()))
        try:save('emergency',True)
        except BaseException as failure:io.atomic_json(output/'save_error.json',dict(error=repr(failure)))
        progress('stopped');return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    raise SystemExit(run(a.plan.resolve(),a.output.resolve(),a.preflight))
