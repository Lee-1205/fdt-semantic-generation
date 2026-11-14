"""One new cross-context decoder trial, fixed source and bounded disk footprint."""
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
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.frozen_verbalizer_objective_v72 import decoder_losses
from context_engine_v74 import ContextEngine,SCHEMA
from context_assess_v74 import assess
from frozen_verbalizer_assess_v72 import new_dev_baseline
from factual_cycle_long_eval_v70 import clone_for_long
from run_semantic_binding_20261111_v61 import save_tensor
from run_ordered_fuzzy_20261111_v55 import Budget as BaseBudget
import learning_io_v2 as io
import text_generation_support_v1 as support
PARENT_SHA='81f4bb4bd41f6b3b89131a4aed7f5de6dc34635e1d5dffc8e42eeee22b738300';ALGORITHM_SHA='1fe06c7a0b8d24822e288259c15ce02ee304e3d2ccfef00444a0bf0960e22348'
OUTPUT_CAP=7*1024**3//2
def validate(path,output):
    p=io.load_json(path);expected=dict(schema='fdt_context_verbalizer_v74',output=str(output.resolve()),cycle_sha=PARENT_SHA,cycle_algorithm_sha=ALGORITHM_SHA,
        wall_seconds=900,max_calls=2700,cost=2592,attempts=640,epochs=1,capacity=16384,max_output_bytes=OUTPUT_CAP,exact_policy=EXACT_POLICY)
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('actual fixed V74 plan contract')
    if p['authorized'] and (p['deadline']-p['start']!=900 or p['cutoff']!=p['deadline']-120):raise ValueError('new clock')
    for file,digest in p['files'].items():
        if support.sha(file)!=digest:raise ValueError('input pin '+file)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core changed')
    d=Path(p['data']);rows=io.load_json(d/'train_features.json');dev=io.load_json(d/'dev_features.json');orders=io.load_json(d/'orders.json')
    if len(rows)!=640 or len(dev)!=64 or len(orders)!=640 or sorted(o[0] for o in orders)!=list(range(640)) or any(set(r)!={'id','prompt'} for r in rows+dev):raise ValueError('finite feature-only rows')
    return p,rows,dev,orders
class Budget(BaseBudget):
    def __init__(self,p):
        super().__init__(p);self.calls=dict(encode=0,readout=0,stream_chunk=0,denoise=0);self.block_work=0;self.data_sha=support.sha(Path(p['data'])/'manifest.json');self.eval_rows=io.load_json(Path(p['data'])/'dev_features.json')
        self.critical_targets={t['id']:t['critical_spans'] for t in io.load_json(Path(p['data'])/'train_targets.json')}
        self.parent_reference=dict(cycle_path=p['cycle_source'],cycle_SHA=p['cycle_sha'],parent23Adam='preserved; fresh7Adam',encoder_binding='fully frozen V73',exact_policy=EXACT_POLICY)
    def charge(self,kind):
        self.check()
        if kind not in ('encode','readout','stream_chunk') or sum(self.calls.values())>=2700:raise ValueError('fixed V74 model calls')
        self.calls[kind]+=1
def initialize(p,rows,orders,budget,sha,device):
    parent=torch.load(p['cycle_source'],map_location='cpu',weights_only=True)
    if support.state_sha(parent)!=ALGORITHM_SHA or parent['schema']!='fdt_live_fuzzy_extraction_v73_fullstate' or parent['updates']!=960 or len(parent['optimizer']['state'])!=23:raise ValueError('immutable complete V73 source')
    torch.manual_seed(74112);random.seed(74112);np.random.seed(74112)
    if device=='cuda':torch.cuda.manual_seed_all(74112)
    enc=FuzzyCodecV64(Config(**parent['core']),input_capacity=16384);enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks();model=FuzzyRoundTripV64(enc,OrderedFuzzyReadout());model.load_state_dict(parent['model'],strict=True)
    if support.state_sha(model.state_dict())!=support.state_sha(parent['model']):raise ValueError('all tensors warm exact')
    engine=ContextEngine(model,rows,orders,budget,sha,device)
    if engine.objective_report!=p['objective_report']:raise ValueError('TRAIN objective signature')
    report=dict(all_parent_tensors_exact=True,all_old_Adam_exact=True,parent23Adam_scaler_ALLRNG_preserved=True,fresh7Adam=not bool(engine.optimizer.state),
        active_readout_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),active_tensors=len(engine.names),encoder_binding_frozen=True,engine_schema=SCHEMA,objective_report=engine.objective_report)
    return engine,report
def initial_delta(engine,p):
    state=engine.export();full_digest=support.state_sha(state);model=state.pop('model');model_digest=support.state_sha(model)
    state['model_reference']=dict(path=p['cycle_source'],SHA=PARENT_SHA,key='model',model_digest=model_digest)
    return state,full_digest
def resolve_initial_delta(delta,parent):
    if delta['model_reference']['SHA']!=PARENT_SHA or support.state_sha(parent['model'])!=delta['model_reference']['model_digest']:raise ValueError('initial source model reference')
    state=dict(delta);state.pop('model_reference');state['model']=parent['model'];return state
def run(path,output,preflight=False):
    p,rows,dev,orders=validate(path,output);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    budget=Budget(p);device='cpu' if preflight else 'cuda';engine,report=initialize(p,rows,orders,budget,support.sha(path),device)
    if preflight:
        delta,digest=initial_delta(engine,p);restored=resolve_initial_delta(delta,torch.load(p['cycle_source'],map_location='cpu',weights_only=True));engine.restore(restored)
        if support.state_sha(engine.export())!=digest:raise ValueError('source-linked initial state exact restore')
        budget.output=path.parent/'cpu_preflight_owned/runtime_cache';budget.output.mkdir();engine.cache_next();f,t=engine.packet(0);c=DifferentiableCodecCommit.from_semantic(f['semantic'],f['memory_revisions'],engine.embedding_sha);budget.charge('readout');z=engine.model.verbalize(c)
        loss,_=decoder_losses(z,t['labels'],engine.symbol_weights,t['critical_mask'],t['teacher_gold_probability'],t['teacher_correct']);loss.backward();named=dict(engine.model.named_parameters())
        if any(named[n].grad is None for n in engine.names) or any(v.grad is not None for n,v in named.items() if n not in engine.names):raise ValueError('actual final7 gradient ownership')
        engine.optimizer.zero_grad(set_to_none=True);engine.verify_frozen();s=engine.export();digest=support.state_sha(s);stream=memoryio.BytesIO();torch.save(s,stream);stream.seek(0);engine.restore(torch.load(stream,map_location='cpu',weights_only=True))
        if support.state_sha(engine.export())!=digest or torch.cuda.is_initialized():raise ValueError('actual fullsize cache restore')
        wrapped=clone_for_long(engine.model,'cpu');del wrapped
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(report,fullsize_engine_restore=True,source_linked_initial_exact_restore=True,actual_cache_partial_restore=True,actual_only7_gradient=True,long_wrapper_all_tensors_exact=True,
            CUDA_initialized=False,actual_encoder_forwards=1,actual_readout_forwards=2,optimizer_steps=0,Exact_entries0=True)));print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('ONE source cache/head trial')
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()));began=time.time();latest=None;guard=False
    def progress(phase):
        engine.phase=phase;io.atomic_json(output/'progress.json',dict(phase=phase,cache_cursor=engine.cache_cursor,cache_total=640,attempts=engine.attempts,updates=engine.updates,cursor=engine.cursor,calls=budget.calls,block_work=budget.block_work,
            time=time.time(),native_pid=os.getpid(),parent_pid=os.getppid(),latest=latest))
    def save(directory,emergency=False):
        nonlocal latest
        budget.check(write_bytes=600*1024**2,emergency=emergency);folder=output/directory;folder.mkdir(exist_ok=True);latest=dict(support.save_full(folder,engine.export()),directory=directory,exact_policy=EXACT_POLICY);io.atomic_json(output/'latest.json',latest)
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        progress('source_linked_initial');delta,digest=initial_delta(engine,p);dsha=save_tensor(output/'initial_runtime.pt',delta,budget);io.atomic_json(output/'initial_runtime.json',dict(SHA=dsha,reconstructed_full_algorithm_SHA=digest,model_reference=delta['model_reference'],standalone_model_copy=False));del delta
        while engine.cache_cursor<len(rows):
            if time.time()+180>=p['cutoff']:raise TimeoutError('cache phase reserve')
            engine.cache_next()
            if engine.cache_cursor%8==0:progress('fixed_source_teacher_cache')
        engine.new_dev_refs=new_dev_baseline(engine,dev,io.load_json(Path(p['data'])/'dev_targets.json'),progress);progress('cache_complete_before_head');save('cache_complete')
        while engine.attempts<len(orders):
            if time.time()+120>=p['cutoff']:raise TimeoutError('head phase final-eval reserve')
            engine.step()
            if engine.attempts%8==0:progress('context_cross_final7_training')
        engine.verify_frozen();progress('final_fixed_source_assessment');assess(engine,dev,progress);engine.verify_frozen();progress('final_fullstate');save('model')
        if support.sha(p['cycle_source'])!=PARENT_SHA:raise ValueError('parent changed')
        io.atomic_json(output/'result.json',dict(status='CONTEXT_VERBALIZER_V74_COMPLETE',attempts=engine.attempts,updates=engine.updates,AMP_skips=engine.skips,cache_cursor=engine.cache_cursor,
            exposure=engine.exposure,calls=budget.calls,block_work=budget.block_work,evaluations=engine.evaluations,latest=latest,seconds=time.time()-began,frozen_upstream_SHA=engine.embedding_sha,
            initial_teacher_head_SHA=engine.teacher_sha,trainable_parameters=report['active_readout_parameters'],active_tensors=7,encoder_binding_updates=0,
            training_GPU_peak_bytes=max(h['GPU_peak_bytes'] for h in engine.history),cache_GPU_peak_bytes=max(r['GPU_peak_bytes'] for r in engine.cache_refs),Exact_entries=0,copy_bytes=0,denoise_calls=0,limitations=p['limitations']))
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
