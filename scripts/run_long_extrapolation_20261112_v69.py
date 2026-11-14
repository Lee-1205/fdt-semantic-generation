"""ONE frozen inference test beyond trained length. No optimizer or denoising."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,gc,io as memoryio,json,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from fdt_local.model_masked import Config
from research.streaming_fuzzy_codec_v69 import StreamingFuzzyCodecV69
from research.fuzzy_roundtrip_v64 import FuzzyRoundTripV64,EXACT_POLICY
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.fixed_utf8_readout_v69 import decode_long_commit
from research.text_diffusion_v1 import pack_prompts
from owned_output_size_v52 import owned_output_bytes
from run_semantic_binding_20261111_v61 import save_tensor

PARENT_SHA='b47c9bdf920605b153c1ebfaa31a28579eb35c1784de6037ba54f4674cdc18b9'
ALGORITHM_SHA='d8b2ba555605913d38415bf7dc427b16f50245e34b542e5f91c831f059b6cd20'

def validate(path,output):
    p=io.load_json(path)
    expected=dict(schema='fdt_long_extrapolation_v69',output=str(output.resolve()),cycle_sha=PARENT_SHA,
        exact_policy=EXACT_POLICY,wall_seconds=900,max_calls=560,stream_chunk_cap=512,block_work_cap=4700,
        attempts=0,epochs=0,max_output_bytes=2*1024**3,capacity=16384,test_capacity=262144,compute_chunk=2048,amp_dtype='float32')
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('fixed new length test contract')
    if p['cost']!=528 or p['parity_thresholds']!={'commit_relative_L2':.001,'logits_relative_L2':.002,'bank_relative_L2':.001}:raise ValueError('pre-fixed parity/cost')
    if p['authorized'] and (p['deadline']-p['start']!=900 or p['cutoff']!=p['deadline']-120):raise ValueError('fixed new clock')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('input pin '+name)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core changed')
    rows=io.load_json(Path(p['data'])/'features.json');parity=io.load_json(Path(p['data'])/'parity_features.json')
    if len(rows)!=10 or len(parity)!=2 or any(set(r)!={'id','prompt'} for r in rows+parity):raise ValueError('features only')
    if sum((len(r['prompt'].encode())+2047)//2048 for r in rows+parity)!=500:raise ValueError('fixed chunk exposure')
    if max(len(r['prompt'].encode()) for r in rows)>262144 or any(len(r['prompt'].encode())!=4096 for r in parity):raise ValueError('length limits')
    return p,rows,parity

class Budget:
    def __init__(self,p):
        self.p=p;self.output=Path(p['output']);self.calls=dict(encode=0,readout=0,stream_chunk=0,denoise=0)
        self.block_work=0;self.last_size_scan=0;self.last_size=0;self.phase='loading';self.row='';self.last_progress=0
    def check(self,write_bytes=0,emergency=False):
        if not emergency and ((self.output/'STOP').exists() or time.time()>=self.p['cutoff']):raise InterruptedError('fixed clock/STOP')
        if shutil.disk_usage(ROOT).free<101*1024**3+write_bytes:raise OSError('C100GiB protected floor plus reserve')
        if write_bytes or time.monotonic()-self.last_size_scan>2:
            self.last_size=owned_output_bytes(self.output);self.last_size_scan=time.monotonic()
        if self.last_size+write_bytes>self.p['max_output_bytes']:raise OSError('output2GiB cap')
        if torch.cuda.is_initialized() and torch.cuda.memory_allocated()>12*1024**3:raise MemoryError('GPU12GiB cap')
    def charge(self,kind):
        self.check()
        if kind not in self.calls or kind=='denoise' or sum(self.calls.values())>=560:raise ValueError('new fixed model call cap')
        if kind=='stream_chunk' and self.calls[kind]>=512:raise ValueError('chunk cap')
        self.calls[kind]+=1
    def progress(self,**extra):
        io.atomic_json(self.output/'progress.json',dict(phase=self.phase,row=self.row,calls=self.calls,block_work=self.block_work,
            native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time(),latest=None,**extra))
    def stream_guard(self,kind,start,end,layer):
        if kind=='layer':
            self.block_work+=1
            if self.block_work>4700:raise ValueError('block work cap')
        elif kind=='chunk':self.charge('stream_chunk')
        self.check()
        if time.monotonic()-self.last_progress>2:
            self.progress(input_processed=end,layer=layer);self.last_progress=time.monotonic()

def load_model(p,device):
    state=torch.load(p['cycle_source'],weights_only=True,map_location='cpu')
    if support.state_sha(state)!=ALGORITHM_SHA or state['schema']!='fdt_live_fuzzy_roundtrip_v67_fullstate' or len(state['optimizer']['state'])!=30:raise ValueError('frozen V67 fullstate')
    enc=StreamingFuzzyCodecV69(Config(**state['core']),input_capacity=16384,stream_capacity=262144,compute_chunk=2048)
    enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks()
    enc.load_state_dict({k[8:]:v for k,v in state['model'].items() if k.startswith('encoder.')},strict=True)
    head=OrderedFuzzyReadout();head.load_state_dict({k[8:]:v for k,v in state['model'].items() if k.startswith('readout.')},strict=True)
    model=FuzzyRoundTripV64(enc,head).eval().requires_grad_(False);digest=support.state_sha(state['model'])
    if support.state_sha(model.state_dict())!=digest:raise ValueError('all old weights exact/no table resize')
    return model.to(device),state,digest

def relative(a,b):
    a,b=a.double(),b.double();return float((a-b).norm()/a.norm().clamp_min(1e-12))

def snapshot(logits,commit,ep):
    return dict(logits=logits.cpu(),fuzzy_commit=commit.reader_tensor().cpu(),
        banks={k:getattr(ep.memories.contextual,k).cpu() for k in ('long_N','long_D','recent_N','recent_D')})

@torch.no_grad()
def run(path,output,preflight=False):
    p,rows,parity=validate(path,output);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model,parent,digest=load_model(p,'cpu' if preflight else 'cuda')
    if preflight:
        stream=memoryio.BytesIO();torch.save(parent,stream);stream.seek(0);restored=torch.load(stream,weights_only=True,map_location='cpu')
        assert support.state_sha(restored)==ALGORITHM_SHA;model.load_state_dict(restored['model'],strict=True)
        assert support.state_sha(model.state_dict())==digest
        text='Aveline moved seven brass pins before Hector closed the red gate. '
        x=pack_prompts([text],len(text),'cpu');kwargs=dict(source_ids=('preflight69',),source_versions=('v1',),model_epoch=PARENT_SHA)
        model.encoder.compute_chunk=17
        with model.encoder.dense_reference():ld,cd,ed=model(x,**kwargs)
        ls,cs,es=model(x,**kwargs)
        assert relative(cd.reader_tensor(),cs.reader_tensor())<.001 and relative(ld,ls)<.002
        assert not es.memories.exact.valid.any() and support.state_sha(model.state_dict())==digest
        from research.codec_commit_stop_v56 import reconstruct_commit_logits
        fixture=torch.randn(12,16,258);a=decode_long_commit(fixture);b=reconstruct_commit_logits(fixture,max_symbols=192)
        assert a['text']==b['text'] and a['EOS']==b['EOS'] and a['ids']==[v for v in b['ids'] if v!=257]
        report=dict(fullsize_model_restore=True,all_parent_tensors_exact=True,all_old_Adam_exact=True,parent_Adam_states=30,
            parent_scaler_ALLRNG_preserved=True,CUDA_initialized=torch.cuda.is_initialized(),actual_parent_forward=2,optimizer_steps=0,
            model_parameters=sum(v.numel() for v in model.parameters()),trainable_parameters=0,runtime_schema='readonly_long_v69',
            fullsize_CPU_native_stream_small_parity=True,linear_UTF8_FSM_reference_parity=True,Exact_policy=EXACT_POLICY,
            no_preflight_access_to_eval_targets=True)
        print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0
    del parent;gc.collect();budget=Budget(p);model.encoder.stream_guard=budget.stream_guard;began=time.time();records=[];guard=False
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('ONE fresh readonly length assessment')
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        budget.phase='dense_stream_execution_parity';parity_records=[]
        for i,row in enumerate(parity):
            budget.row=row['id'];x=pack_prompts([row['prompt']],len(row['prompt'].encode()),'cuda')
            kwargs=dict(source_ids=(row['id'],),source_versions=(p['cycle_sha'],),model_epoch=p['cycle_sha']);samples=[]
            for dense in (True,False):
                budget.charge('encode');budget.charge('readout')
                if dense:
                    budget.block_work+=len(model.encoder.core.language.blocks)
                    with model.encoder.dense_reference():logits,commit,episode=model(x,**kwargs)
                else:logits,commit,episode=model(x,**kwargs)
                samples.append(snapshot(logits,commit,episode));del logits,commit,episode
            a,b=samples
            metrics=dict(commit_relative_L2=relative(a['fuzzy_commit'],b['fuzzy_commit']),logits_relative_L2=relative(a['logits'],b['logits']),
                bank_relative_L2=max(relative(a['banks'][k],b['banks'][k]) for k in a['banks']),
                argmax_agreement=float((a['logits'].argmax(-1)==b['logits'].argmax(-1)).float().mean()))
            save_tensor(output/f'parity_{i}.pt',dict(native=a,stream=b),budget)
            parity_records.append(dict(id=row['id'],**metrics));io.atomic_json(output/'parity.json',parity_records)
            if not all(metrics[k]<=v for k,v in p['parity_thresholds'].items()):raise ValueError('execution parity failed before extrapolation')
            del samples,a,b,x;gc.collect()
        budget.phase='beyond_trained_length_reconstruction'
        for i,row in enumerate(rows):
            budget.row=row['id'];budget.progress(completed=i,total=10);raw=row['prompt'].encode()
            x=pack_prompts([row['prompt']],len(raw),'cuda');budget.charge('encode');budget.charge('readout')
            logits,commit,ep=model(x,source_ids=(row['id'],),source_versions=(p['cycle_sha'],),model_epoch=p['cycle_sha'])
            if not torch.isfinite(logits).all() or ep.memories.exact.valid.any():raise ValueError('numeric/Exact contract')
            y=torch.full(logits.shape[:2],257,dtype=torch.long,device='cuda');flat=y.flatten();flat[:len(raw)]=torch.tensor(list(raw),device='cuda');flat[len(raw)]=256
            nll=float(torch.nn.functional.cross_entropy(logits.transpose(1,2),y,ignore_index=257,reduction='sum'));pred=logits.argmax(-1)
            decoded=decode_long_commit(logits,budget.check);text=decoded.pop('text');payload=snapshot(logits,commit,ep)
            payload.update(targets=y.cpu(),Exact_entries=0,Exact_policy=EXACT_POLICY)
            raw_sha=save_tensor(output/f'raw_{i:02d}.pt',payload,budget)
            records.append(dict(id=row['id'],source_bytes=len(raw),output=text,raw_sha256=raw_sha,NLL_sum=nll,
                byte_correct=int(((pred==y)&(y<256)).sum()),symbols=len(raw)+1,exact=text==row['prompt'],
                stream_stats=model.encoder.stream_stats,commit_shape=list(payload['fuzzy_commit'].shape),**decoded))
            io.atomic_json(output/'reconstructions.json',records);budget.progress(completed=i+1,total=10)
            del x,logits,commit,ep,y,pred,payload;gc.collect()
        assert support.state_sha(model.state_dict())==digest and support.sha(p['cycle_source'])==PARENT_SHA
        result=dict(status='LONG_EXTRAPOLATION_V69_COMPLETE',rows=10,calls=budget.calls,block_work=budget.block_work,
            optimizer_updates=0,checkpoints_written=0,Exact_entries=0,copy_bytes=0,denoise_calls=0,seconds=time.time()-began,
            source_model_SHA=PARENT_SHA,frozen_before_after_digest=digest,trained_input_max_bytes=16384,
            actual_max_input_bytes=max(r['source_bytes'] for r in records),UTF8=sum(r['UTF8'] for r in records),EOS=sum(r['EOS'] for r in records),
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),limitations=p['limitations'])
        io.atomic_json(output/'result.json',result);return 0
    except BaseException as error:
        io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc(),calls=budget.calls));return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',action='store_true')
    a=parser.parse_args();raise SystemExit(run(a.plan.resolve(),a.output.resolve(),a.preflight))
