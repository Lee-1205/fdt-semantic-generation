"""One bounded teacher/live extraction trial with a fixed language readout."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,io as memoryio,json,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64,EXACT_POLICY
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from fuzzy_extraction_engine_v73 import ExtractionEngine,SCHEMA,OBJECTIVE
from extraction_assess_v73 import baseline,final_assessment
from factual_cycle_long_eval_v70 import clone_for_long
from run_ordered_fuzzy_20261111_v55 import Budget as BaseBudget
import learning_io_v2 as io
import text_generation_support_v1 as support
PARENT_SHA='74531e0baa91f8f3fd0bfb26aeedc73871c9bc4afbd5a896ad6c25d4158fe864'
ALGORITHM_SHA='d2b59076b393dcfcafef768da2cc377cc95f47bba8ac874247409dabc4eb2e09'

def validate(path,output):
    p=io.load_json(path)
    expected=dict(schema='fdt_live_extraction_v73',output=str(output.resolve()),cycle_sha=PARENT_SHA,cycle_algorithm_sha=ALGORITHM_SHA,wall_seconds=900,max_calls=5600,cost=5346,
        attempts=960,epochs=1,capacity=16384,max_output_bytes=6*1024**3,exact_policy=EXACT_POLICY,objective=OBJECTIVE)
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('actual fixed V73 plan contract')
    if p['authorized'] and (p['deadline']-p['start']!=900 or p['cutoff']!=p['deadline']-120):raise ValueError('new clock')
    for f,sha in p['files'].items():
        if support.sha(f)!=sha:raise ValueError('input pin '+f)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core changed')
    d=Path(p['data']);rows=io.load_json(d/'train_features.json');orders=io.load_json(d/'orders.json');held=io.load_json(Path(p['eval_data'])/'short_features.json');long=io.load_json(Path(p['eval_data'])/'long_features.json')
    if len(rows)!=960 or len(orders)!=960 or sorted(v[0] for v in orders)!=list(range(960)) or len(held)!=128 or len(long)!=10 or any(set(r)!={'id','prompt'} for r in rows+held+long):raise ValueError('finite separated source inputs')
    return p,rows,orders
class Budget(BaseBudget):
    def __init__(self,p):
        super().__init__(p);self.calls=dict(encode=0,readout=0,stream_chunk=0,denoise=0);self.block_work=0;self.data_sha=support.sha(Path(p['data'])/'manifest.json');self.baseline_required=True
        self.critical_targets={t['id']:t['critical_spans'] for t in io.load_json(Path(p['data'])/'train_targets.json')}
        self.parent_reference=dict(cycle_path=p['cycle_source'],cycle_SHA=p['cycle_sha'],parent7Adam='preserved; fresh upstream23Adam',fixed_readout7=True,exact_policy=EXACT_POLICY)
    def charge(self,kind):
        self.check()
        if kind not in ('encode','readout','stream_chunk') or sum(self.calls.values())>=5600:raise ValueError('fixed V73 calls')
        self.calls[kind]+=1
def initialize(p,rows,orders,budget,plan_sha,device):
    parent=torch.load(p['cycle_source'],map_location='cpu',weights_only=True)
    if support.state_sha(parent)!=ALGORITHM_SHA or parent['schema']!='fdt_frozen_verbalizer_v72_fullstate' or parent['updates']!=640 or len(parent['optimizer']['state'])!=7:raise ValueError('immutable complete V72 source')
    torch.manual_seed(73112);random.seed(73112);np.random.seed(73112)
    if device=='cuda':torch.cuda.manual_seed_all(73112)
    enc=FuzzyCodecV64(Config(**parent['core']),input_capacity=16384);enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks();model=FuzzyRoundTripV64(enc,OrderedFuzzyReadout());model.load_state_dict(parent['model'],strict=True)
    if support.state_sha(model.state_dict())!=support.state_sha(parent['model']):raise ValueError('all source tensors exact')
    engine=ExtractionEngine(model,rows,orders,budget,plan_sha,device)
    if engine.objective_report!=p['objective_report']:raise ValueError('actual objective signature')
    report=dict(all_parent_tensors_exact=True,all_old_Adam_exact=True,parent7Adam_scaler_ALLRNG_preserved=True,fresh23Adam=not bool(engine.optimizer.state),
        actual_upstream_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),active_tensors=len(engine.names),fixed_readout7=True,engine_schema=SCHEMA,objective_report=engine.objective_report)
    return engine,report
def run(path,output,preflight=False):
    p,rows,orders=validate(path,output);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    budget=Budget(p);device='cpu' if preflight else 'cuda';engine,report=initialize(p,rows,orders,budget,support.sha(path),device)
    if preflight:
        budget.output=path.parent/'cpu_preflight_owned/runtime_teacher';budget.output.mkdir();engine.teacher_next();loss,parts,t=engine.live_loss(0);loss.backward();named=dict(engine.model.named_parameters())
        if any(named[n].grad is None for n in engine.names) or any(v.grad is not None for n,v in named.items() if n not in engine.names):raise ValueError('actual upstream23 gradient ownership')
        engine.optimizer.zero_grad(set_to_none=True);engine.verify_frozen();s=engine.export();digest=support.state_sha(s);stream=memoryio.BytesIO();torch.save(s,stream);stream.seek(0);engine.restore(torch.load(stream,map_location='cpu',weights_only=True))
        if support.state_sha(engine.export())!=digest or torch.cuda.is_initialized():raise ValueError('actual fullsize phase restore')
        wrapped=clone_for_long(engine.model,'cpu');del wrapped
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(report,fullsize_engine_restore=True,actual_teacher_partial_restore=True,actual_loss_gradient_upstream23=True,long_wrapper_all_tensors_exact=True,
            CUDA_initialized=False,actual_encoder_forwards=2,actual_readout_forwards=2,optimizer_steps=0,Exact_entries0=True)))
        print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('ONE fresh trial')
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()));began=time.time();latest=None;guard=False
    def progress(phase):
        engine.phase=phase;io.atomic_json(output/'progress.json',dict(phase=phase,teacher_cursor=engine.teacher_cursor,teacher_total=960,attempts=engine.attempts,updates=engine.updates,cursor=engine.cursor,
            calls=budget.calls,block_work=budget.block_work,time=time.time(),native_pid=os.getpid(),parent_pid=os.getppid(),latest=latest))
    def save(directory,emergency=False):
        nonlocal latest
        budget.check(write_bytes=700*1024**2,emergency=emergency);folder=output/directory;folder.mkdir(exist_ok=True);latest=dict(support.save_full(folder,engine.export()),directory=directory,exact_policy=EXACT_POLICY);io.atomic_json(output/'latest.json',latest)
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        progress('initial_fixed_readout');save('initial')
        while engine.teacher_cursor<len(rows):
            if time.time()+360>=p['cutoff']:raise TimeoutError('teacher phase live/eval reserve')
            engine.teacher_next()
            if engine.teacher_cursor%8==0:progress('initial_TRAIN_teacher_loss_packets')
        baseline(engine,progress);progress('teacher_complete_before_live_updates');save('teacher_complete')
        while engine.attempts<len(orders):
            if time.time()+180>=p['cutoff']:raise TimeoutError('live phase final-eval reserve')
            engine.step()
            if engine.attempts in (1,480):engine.verify_frozen();save('model')
            if engine.attempts%8==0:progress('live_source_upstream23_training')
        engine.verify_frozen();progress('fresh_final_full_source_assessment');final_assessment(engine,progress);engine.verify_frozen();progress('final_fullstate');save('model')
        if support.sha(p['cycle_source'])!=PARENT_SHA:raise ValueError('parent changed')
        io.atomic_json(output/'result.json',dict(status='LIVE_FUZZY_EXTRACTION_V73_COMPLETE',attempts=engine.attempts,updates=engine.updates,AMP_skips=engine.skips,teacher_cursor=engine.teacher_cursor,
            exposure=engine.exposure,calls=budget.calls,block_work=budget.block_work,evaluations=engine.evaluations,latest=latest,seconds=time.time()-began,
            fixed_tensors_SHA_before_after=engine.frozen_sha,initial_teacher_model_SHA=engine.teacher_model_sha,active_tensors=23,active_upstream_parameters=report['actual_upstream_parameters'],
            final_readout7_updates=0,training_GPU_peak_bytes=max(h['GPU_peak_bytes'] for h in engine.history),teacher_GPU_peak_bytes=max(r['GPU_peak_bytes'] for r in engine.teacher_refs),
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
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--preflight',action='store_true');args=a.parse_args();raise SystemExit(run(args.plan.resolve(),args.output.resolve(),args.preflight))
