"""NEW readonly73-output purpose from V91 emergency; no SGD or old budget."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,shutil,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.fp32_ordered_readout_v86 import FP32OrderedFuzzyReadout
from research import linked_lastlinear_runtime_v79 as linked
from run_critical_repair_20261113_v88 import assess
from owned_output_size_v52 import owned_output_bytes
import learning_io_v2 as io
import text_generation_support_v1 as support

SCHEMA='fdt_missing73_readonly_assessment_v92'
SOURCE_SHA='31a9cc4e02aabdb834917b208470306612ead06aa0cb786386faf84ab7571069'
ALGORITHM='96e31a299c01f25f748c882d6a0bab4855d6d4f58130798c1922402083b51e89'
OUTPUT=32*1024**2;START=108*1024**3+OUTPUT


def validate(path,out):
    p=io.load_json(path)
    fixed=dict(schema=SCHEMA,source_SHA=SOURCE_SHA,source_algorithm_SHA=ALGORITHM,max_calls=73,expected_rows=73,
        max_output_bytes=OUTPUT,start_free_bytes=START,wall_seconds=180,optimizer_steps=0,encoder_calls=0,denoise_calls=0)
    if any(p.get(k)!=v for k,v in fixed.items()) or Path(p['output']).resolve()!=out:raise ValueError('NEW readonly73 fixed contract')
    if p['authorized'] and (p['deadline']-p['start']!=180 or p['cutoff']!=p['deadline']-45):raise ValueError('new135+45 clock')
    for f,expected in p['files'].items():
        if support.sha(f)!=expected:raise ValueError('input pin '+f)
    items=io.load_json(p['eval_inputs']);full=io.load_json(p['original_eval_inputs'])
    if items!=full[121:] or len(items)!=73:raise ValueError('only never-produced73 outputs; no121repeat')
    reviewed=io.load_json(p['saved_review'])
    if reviewed['partial_outputs_reviewed']!=121 or reviewed['fullstate_SHA']!=SOURCE_SHA:raise ValueError('preserved6update state/partial proof')
    return p,items


class Budget:
    def __init__(self,p,preflight):
        self.p=p;self.output=Path(p['output']);self.preflight=preflight;self.began=time.monotonic();self.calls=dict(readout=0)
    def check(self,write_bytes=0,emergency=False):
        if self.preflight:
            if time.monotonic()-self.began>75:raise TimeoutError('CPU75sec source-only preflight')
        elif time.time()>=self.p['cutoff'] or (self.output/'STOP').exists():raise InterruptedError('NEWreadonlycutoff/STOP')
        if shutil.disk_usage(ROOT).free<100*1024**3+write_bytes:raise OSError('protected100GiB')
        # Reuse the preexisting V52 fix in BOTH worker and owner. A supervisor
        # atomic JSON rename must not fail a model budget check.
        if owned_output_bytes(self.output)+write_bytes>OUTPUT:raise OSError('readonly32MiB output cap')
    def charge(self,kind):
        self.check()
        if kind!='readout' or self.calls['readout']>=73:raise ValueError('readonly73NNcalls only')
        self.calls['readout']+=1


def run(path,out,preflight=False):
    p,items=validate(path,out);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=linked.load_verified(p['source'],SOURCE_SHA,ROOT,support.state_sha)
    if support.state_sha(source)!=ALGORITHM or source['updates']!=6 or source['attempts']!=70:raise ValueError('exact saved6update state')
    encoder=FuzzyCodecV64(Config(**source['core']),input_capacity=source['input_capacity'])
    encoder.enable_position_adapter();encoder.enable_long_capacity();encoder.enable_local_banks()
    model=FuzzyRoundTripV64(encoder,FP32OrderedFuzzyReadout());model.load_state_dict(source['model'],strict=True);model.requires_grad_(False);model.eval()
    model_sha=support.state_sha(model.state_dict())
    if model_sha!=support.state_sha(source['model']):raise ValueError('all actual saved model tensors exact')
    if preflight:
        report=dict(fullsize_engine_restore=True,fullsize_model_restore=True,all_parent_tensors_exact=True,all_old_Adam_exact=True,
            CUDA_initialized=torch.cuda.is_initialized(),source_model_SHA=model_sha,all_stored_Adam_RNG_exact=True,
            source_updates=6,scope='readonly fullstate safeload; no optimizer resumed or new checkpoint',
            actual_encoder_forwards=0,actual_readout_forwards=0,optimizer_steps=0,worker_uses_existing_v52_race_fix=True)
        print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (out/'worker_started.json').exists():raise ValueError('ONE readonlylaunch')
    started=time.time();io.atomic_json(out/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=started,plan_SHA=support.sha(path)))
    budget=Budget(p,False);model.to('cuda');engine=SimpleNamespace(model=model,budget=budget,embedding_sha=source['embedding_sha'],evaluations={})
    def progress(phase):io.atomic_json(out/'progress.json',dict(phase=phase,calls=budget.calls,native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time(),elapsed=time.time()-started,latest_durable=p['source']))
    guard=False
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        progress('readonly_missing73_cached_Commit');assess(engine,items,progress)
        if support.sha(p['source'])!=SOURCE_SHA or support.state_sha(model.state_dict())!=model_sha:raise ValueError('readonly model/source mutated')
        io.atomic_json(out/'result.json',dict(status='READONLY_MISSING73_ASSESSMENT_COMPLETE',rows=73,calls=budget.calls,
            seconds=time.time()-started,source_SHA=SOURCE_SHA,source_algorithm_SHA=ALGORITHM,source_model_SHA=model_sha,
            source_all_stored_Adam_RNG_preserved=True,optimizer_steps=0,encoder_calls=0,denoise_calls=0,checkpoint_writes=0,
            previously_completed121_not_repeated=True,evaluations=engine.evaluations,long_cached_Commit_only=True,Exact_entries=0,copy_bytes=0,
            originalV91status_remains_FAILED=True,new_budget_closed=True));progress('terminal_complete');return 0
    except BaseException as error:
        io.atomic_json(out/'incident.json',dict(error=repr(error),traceback=traceback.format_exc(),calls=budget.calls));return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--preflight',action='store_true');x=a.parse_args()
    raise SystemExit(run(x.plan.resolve(),x.output.resolve(),x.preflight))
