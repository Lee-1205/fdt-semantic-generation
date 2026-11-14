"""ONE frozen-codec cache/target-ceiling purpose, NOT diffusion training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,gc,hashlib,io as memoryio,json,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.fp32_ordered_readout_v86 import FP32OrderedFuzzyReadout
from research.memory_thought_diffusion_v94 import InputMeaning,target_meaning
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.thought_cache_v95 import split_input,assemble_input
from research.text_diffusion_v1 import pack_prompts
from research.readout_decision_trace_v88 import pack as trace_pack,decode
from research import linked_lastlinear_runtime_v79 as linked
import learning_io_v2 as io
import text_generation_support_v1 as support

SCHEMA='fdt_frozen_input_answer_cache_v95';OUTPUT=128*1024**2;START=108*1024**3+OUTPUT
SPLITS=('train','dev_pair','dev_surface','dev_position')


def validate(path,out):
    p=io.load_json(path)
    fixed=dict(schema=SCHEMA,max_calls=542,expected_input_rows=480,expected_answers=31,max_output_bytes=OUTPUT,
        start_free_bytes=START,wall_seconds=300,optimizer_steps=0,denoise_calls=0,answer_capacity=33)
    if any(p.get(k)!=v for k,v in fixed.items()) or Path(p['output']).resolve()!=out:raise ValueError('fixed frozen-cache scope/cost/cap')
    if p['authorized'] and (p['deadline']-p['start']!=300 or p['cutoff']!=p['deadline']-60):raise ValueError('NEW240+60clock')
    for f,sha in p['files'].items():
        if support.sha(f)!=sha:raise ValueError('input pin '+f)
    selection=io.load_json(p['selection'])
    if p['source_codec_SHA']!=selection['selected_SHA']:raise ValueError('actual selected codec identity')
    d=Path(p['data']);m=io.load_json(d/'manifest.json')
    for f,sha in m['files'].items():
        if support.sha(d/f)!=sha:raise ValueError('data immutable')
    rows=[(split,f) for split in SPLITS for f in io.load_json(d/(split+'_features.json'))]
    answers=io.load_json(d/'answers_loss_only.json')
    if len(rows)!=480 or len(answers)!=31 or any(set(f)!={'id','prompt'} for _,f in rows):raise ValueError('source-only rows/target dictionary')
    return p,rows,answers


class Budget:
    def __init__(self,p,preflight):
        self.p=p;self.output=Path(p['output']);self.preflight=preflight;self.began=time.monotonic()
        self.calls=dict(encode=0,readout=0,denoise=0);self.tensor_bytes=0
    def check(self,pending=0):
        if self.preflight:
            if time.monotonic()-self.began>75:raise TimeoutError('CPU75sec work')
        elif time.time()>=self.p['cutoff'] or (self.output/'STOP').exists():raise InterruptedError('newcachecutoff/STOP')
        if shutil.disk_usage(ROOT).free<100*1024**3+pending:raise OSError('100GiB+pending write protected')
        # Exact tensor byte ledger; reserve4MiB for bounded JSON/logs. The
        # independent OS owner ALSO scans all files with the V52 race-safe size.
        # Avoid quadratic directory rescans on every encoder/readout call.
        if self.preflight:
            if self.tensor_bytes+pending>4*1024**2:raise OSError('CPU4MiB payload cap')
        elif self.tensor_bytes+pending+4*1024**2>OUTPUT:raise OSError('whole128MiB incl JSON/log reserve')
    def charge(self,kind):
        self.check()
        if kind not in ('encode','readout') or sum(self.calls.values())>=(3 if self.preflight else 542):raise ValueError('frozen-cache callcap/type')
        self.calls[kind]+=1
    def write_tensor(self,path,value):
        stream=memoryio.BytesIO();torch.save(value,stream);payload=stream.getvalue();self.check(len(payload))
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(payload)
        self.tensor_bytes+=len(payload)
        return dict(path=str(path),SHA=hashlib.sha256(payload).hexdigest(),bytes=len(payload))


def load_codec(p,device):
    selection=io.load_json(p['selection'])
    source=linked.load_verified(selection['selected_runtime'],selection['selected_SHA'],ROOT,support.state_sha)
    if support.state_sha(source)!=selection['source_algorithm_SHA']:raise ValueError('full source including storedAdam/RNG')
    encoder=FuzzyCodecV64(Config(**source['core']),input_capacity=source['input_capacity'])
    encoder.enable_position_adapter();encoder.enable_long_capacity();encoder.enable_local_banks()
    codec=FuzzyRoundTripV64(encoder,FP32OrderedFuzzyReadout());codec.load_state_dict(source['model'],strict=True)
    before=support.state_sha(codec.state_dict())
    if before!=selection['source_model_SHA']:raise ValueError('all actual V88 parameters exact')
    codec.to(device);codec.requires_grad_(False);codec.eval()
    if any(v.requires_grad or v.grad is not None for v in codec.parameters()) or any(m.training for m in codec.modules()):raise ValueError('ALLcodec frozen/eval')
    return codec,source['embedding_sha'],before,selection


@torch.no_grad()
def encode(codec,text,identity,epoch,data_sha,budget,device):
    budget.charge('encode');tokens=pack_prompts([text],len(text.encode()),device)
    with torch.autocast(device,dtype=torch.bfloat16,enabled=device=='cuda'):
        commit,episode=codec.extract(tokens,source_ids=(identity,),source_versions=(data_sha,),model_epoch=epoch)
    return commit,episode


@torch.no_grad()
def cache_source(codec,epoch,data_sha,row,split,budget,device,static):
    commit,episode=encode(codec,row['prompt'],row['id'],epoch,data_sha,budget,device)
    meaning=InputMeaning.from_episode(commit,episode);s,d=split_input(meaning,row['id'],hashlib.sha256(row['prompt'].encode()).hexdigest())
    if static is not None:
        if set(s)!=set(static) or any(not torch.equal(v,static[k]) if torch.is_tensor(v) else v!=static[k] for k,v in s.items()):raise ValueError('canonical prior/NULL changed across source rows')
    else:static=s
    rebuilt=assemble_input(static,d)
    if support.state_sha(vars(rebuilt))!=support.state_sha(vars(InputMeaning(**{k:getattr(meaning,k).cpu() if torch.is_tensor(getattr(meaning,k)) else getattr(meaning,k) for k in vars(meaning)}))):
        raise ValueError('source-memory cache exact roundtrip')
    record=budget.write_tensor(budget.output/'conditions'/split/(row['id']+'.pt'),d)
    return static,dict(id=row['id'],source_SHA=d['source_SHA'],**record)


@torch.no_grad()
def cache_answer(codec,epoch,data_sha,key,text,budget,device):
    if hashlib.sha256(text.encode()).hexdigest()!=key:raise ValueError('loss-only answer identity')
    commit,episode=encode(codec,text,'LOSS_ONLY_answer:'+key,epoch,data_sha,budget,device)
    # Discard the target episode/memories. NEVER insert answer K_C into a
    # source condition. Only the answer Commit supplies a training target.
    del episode
    target,mask=target_meaning(commit,codec.readout.readout.terminal,33)
    packed=DifferentiableCodecCommit.from_semantic(target,commit.memory_revisions,epoch)
    budget.charge('readout');logits=codec.verbalize(packed)
    labels=torch.full(logits.shape[:2],257,dtype=torch.long,device=device);raw=text.encode()
    labels.flatten()[:len(raw)]=torch.tensor(list(raw),device=device);labels.flatten()[len(raw)]=256
    trace=trace_pack(logits,labels,list(dict.fromkeys([0,len(raw)//16])));d=decode(trace)
    loss=dict(schema='LOSS_ONLY_answer_meaning_v95',answer_SHA=key,source_codec_SHA=budget.p['source_codec_SHA'],
        semantic_target=target.cpu(),valid_extent_mask=mask.cpu(),labels=labels.cpu(),model_epoch=epoch,
        target_condition_features_never_used=True)
    target_ref=budget.write_tensor(budget.output/'targets'/(key+'.pt'),loss)
    trace_ref=budget.write_tensor(budget.output/'ceiling'/(key+'.pt'),trace)
    return dict(answer_SHA=key,target=target_ref,ceiling_trace=trace_ref,expected=text,output=d['text'],
        exact=d['text']==text,UTF8=d['UTF8'],EOS=d['EOS'],answer_bytes=len(raw))


def run(path,out,preflight=False):
    p,rows,answers=validate(path,out);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device='cpu' if preflight else 'cuda';budget=Budget(p,preflight)
    if preflight:budget.output=path.parent/'cpu_preflight_owned/runtime';budget.output.mkdir(exist_ok=False)
    codec,epoch,before,selection=load_codec(p,device);data_sha=support.sha(Path(p['data'])/'manifest.json')
    if preflight:
        split,row=rows[0];static,record=cache_source(codec,epoch,data_sha,row,split,budget,device,None)
        key,text=next(iter(sorted(answers.items())));answer=cache_answer(codec,epoch,data_sha,key,text,budget,device)
        static_ref=budget.write_tensor(budget.output/'static_memories.pt',static)
        io.atomic_json(budget.output/'cache_records.json',dict(source=record,static=static_ref,target=answer))
        assert budget.calls==dict(encode=2,readout=1,denoise=0) and support.state_sha(codec.state_dict())==before
        report=dict(fullsize_engine_restore=True,all_parent_tensors_exact=True,all_old_Adam_exact=True,
            CUDA_initialized=torch.cuda.is_initialized(),actual_encoder_forwards=2,actual_readout_forwards=1,optimizer_steps=0,
            all_codec_parameters_buffers_unchanged=True,source_condition_roundtrip_exact=True,
            source_cache_excludes_answer_memory=True,real_target_packed_meaning_readout=True,frozen_model_SHA=before,
            data_source_only_vs_loss_target_separate=True,thought_model_instantiated=False)
        print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (out/'worker_started.json').exists():raise ValueError('ONE new frozen-cache execution')
    started=time.time();io.atomic_json(out/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=started))
    index={split:[] for split in SPLITS};static=None;target_records=[];guard=False
    def progress(phase):io.atomic_json(out/'progress.json',dict(phase=phase,input_rows=sum(map(len,index.values())),targets=len(target_records),calls=budget.calls,
        native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time(),elapsed=time.time()-started,latest_durable=str(out/'conditions')))
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        progress('frozen_SOURCE_only_memories')
        for i,(split,row) in enumerate(rows):
            static,record=cache_source(codec,epoch,data_sha,row,split,budget,device,static);index[split].append(record)
            if i%8==0:progress('frozen_SOURCE_only_memories')
        static_ref=budget.write_tensor(out/'static_memories.pt',static)
        progress('LOSS_ONLY_answer_targets_and_codec_ceiling')
        for key,text in sorted(answers.items()):
            target_records.append(cache_answer(codec,epoch,data_sha,key,text,budget,device));progress('LOSS_ONLY_answer_targets_and_codec_ceiling')
        if budget.calls!=dict(encode=511,readout=31,denoise=0):raise ValueError('actual542call cost')
        if support.state_sha(codec.state_dict())!=before or support.sha(selection['selected_runtime'])!=selection['selected_SHA']:
            raise ValueError('frozen codec changed')
        io.atomic_json(out/'input_index.json',index);io.atomic_json(out/'target_index_loss_only.json',target_records)
        io.atomic_json(out/'result.json',dict(status='FROZEN_INPUT_ANSWER_CACHES_COMPLETE_NOT_THOUGHT_TRAINING',calls=budget.calls,seconds=time.time()-started,
            input_rows480=True,targets31=True,static_memories=static_ref,codec_target_ceiling_exact=sum(x['exact'] for x in target_records),
            codec_target_ceiling_UTF8_EOS=sum(x['UTF8'] and x['EOS'] for x in target_records),
            model_before_after_SHA=before,all_codec_weights_buffers_unchanged=True,optimizer_steps=0,denoise_calls=0,
            source_memory_vs_target_strictly_separate=True,source_codec_SHA=selection['selected_SHA'],tensor_payload_bytes=budget.tensor_bytes,
            first_diffusion_training_started=False,oldbudgets_closed=True));progress('terminal_complete');return 0
    except BaseException as error:
        io.atomic_json(out/'incident.json',dict(error=repr(error),traceback=traceback.format_exc(),calls=budget.calls,
            completed_input_rows=sum(map(len,index.values())),completed_targets=len(target_records)));return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--preflight',action='store_true');x=a.parse_args()
    raise SystemExit(run(x.plan.resolve(),x.output.resolve(),x.preflight))
