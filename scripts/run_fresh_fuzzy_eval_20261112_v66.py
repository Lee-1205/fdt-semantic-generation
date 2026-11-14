"""ONE read-only unseen-authored sentence/window reconstruction assessment."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,gc,io as memoryio,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64,EXACT_POLICY
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.codec_commit_stop_v56 import reconstruct_commit_logits
from research.text_diffusion_v1 import pack_prompts
from run_ordered_fuzzy_20261111_v55 import Budget as BaseBudget
from run_semantic_binding_20261111_v61 import save_tensor

PARENT_SHA='3351bae2358ed660effbb67c642650b3401b9142f6b9b09e19d823310f9cc7a4'
ALGORITHM_SHA='bdc7409c0da8fa43fe8c314cab742e9c50c41ef611706897231883c55485dceb'

def validate(path,output):
    p=io.load_json(path)
    if (p['schema']!='fdt_fresh_fuzzy_eval_v66' or p['output']!=str(output.resolve()) or p['cycle_sha']!=PARENT_SHA
        or p['exact_policy']!=EXACT_POLICY or p['wall_seconds']!=180 or p['max_calls']!=112 or p['cost']!=104
        or p['attempts']!=0 or p['epochs']!=0 or p['max_output_bytes']!=256*1024**2):raise ValueError('new readonly assessment contract')
    if p['authorized'] and (p['deadline']-p['start']!=180 or p['cutoff']!=p['deadline']-45):raise ValueError('fresh fixed assessment clock')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('input pin '+name)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core changed')
    rows=io.load_json(Path(p['data'])/'rows.json')
    if len(rows)!=52 or max(len(r['prompt'].encode()) for r in rows)>16384:raise ValueError('fresh held panel')
    return p,rows

class Budget(BaseBudget):
    def charge(self,kind):
        self.check()
        if kind not in ('encode','readout') or sum(self.calls.values())>=112:raise ValueError('readonly cap/denoise prohibited')
        self.calls[kind]+=1

def load_model(p,device):
    state=torch.load(p['cycle_source'],weights_only=True,map_location='cpu')
    if support.state_sha(state)!=ALGORITHM_SHA or state['schema']!='fdt_live_fuzzy_roundtrip_v65_fullstate' or len(state['optimizer']['state'])!=30:raise ValueError('full frozen V65 state')
    encoder=FuzzyCodecV64(Config(**state['core']),input_capacity=16384);encoder.enable_position_adapter();encoder.enable_long_capacity();encoder.enable_local_banks()
    encoder.load_state_dict({k[8:]:v for k,v in state['model'].items() if k.startswith('encoder.')},strict=True)
    head=OrderedFuzzyReadout();head.load_state_dict({k[8:]:v for k,v in state['model'].items() if k.startswith('readout.')},strict=True)
    model=FuzzyRoundTripV64(encoder,head)
    digest=support.state_sha(state['model'])
    if support.state_sha(model.state_dict())!=digest:raise ValueError('full model warm parity')
    return model.to(device).eval().requires_grad_(False),state,digest

def run(path,output,preflight=False):
    p,rows=validate(path,output);torch.set_num_threads(4);device='cpu' if preflight else 'cuda';model,parent,digest=load_model(p,device)
    if preflight:
        # Read-only runtime needs model restoration and preservation of full
        # parent Adam/scaler/RNG, not a new optimizer or another training trial.
        stream=memoryio.BytesIO();torch.save(parent,stream);stream.seek(0);restored=torch.load(stream,weights_only=True,map_location='cpu')
        assert support.state_sha(restored)==ALGORITHM_SHA
        model.load_state_dict(restored['model'],strict=True);assert support.state_sha(model.state_dict())==digest
        report=dict(fullsize_model_restore=True,all_parent_tensors_exact=True,all_old_Adam_exact=True,parent_Adam_states=30,
            parent_scaler_ALLRNG_preserved=True,CUDA_initialized=torch.cuda.is_initialized(),actual_parent_forward=0,optimizer_steps=0,
            model_parameters=sum(v.numel() for v in model.parameters()),trainable_parameters=0,Exact_policy=EXACT_POLICY,
            runtime_schema='readonly_fuzzy_cycle_v66',no_preflight_access_to_eval_targets=True)
        print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0
    del parent;gc.collect();budget=Budget(p);began=time.time();records=[];guard=False
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('ONE new frozen assessment')
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        with torch.no_grad():
            for i,row in enumerate(rows):
                raw=row['prompt'].encode();x=pack_prompts([row['prompt']],len(raw),'cuda');budget.charge('encode');budget.charge('readout')
                with torch.autocast('cuda',dtype=torch.bfloat16):logits,commit,episode=model(x,source_ids=(row['id'],),source_versions=(support.sha(Path(p['data'])/'rows.json'),),model_epoch=p['cycle_sha'])
                assert not episode.memories.exact.valid.any()
                y=torch.full(logits.shape[:2],257,dtype=torch.long,device='cuda');flat=y.flatten();flat[:len(raw)]=torch.tensor(list(raw),device='cuda');flat[len(raw)]=256
                ce=float(torch.nn.functional.cross_entropy(logits.float().transpose(1,2),y,ignore_index=257,reduction='sum'));pred=logits.argmax(-1);active=y!=257
                decoded=reconstruct_commit_logits(logits.float());text=decoded.pop('text');sealed=commit.seal()
                save_tensor(output/f'raw_{i:02d}.pt',dict(logits=logits.float().cpu(),targets=y.cpu(),fuzzy_commit=sealed.reader_tensor(),
                    Exact_entries=0,Exact_policy=EXACT_POLICY),budget)
                records.append(dict(id=row['id'],language=row['language'],kind=row['kind'],group=row['group'],source=row['prompt'],output=text,
                    source_bytes=len(raw),symbols=len(raw)+1,correct=int(((pred==y)&active).sum()),byte_correct=int(((pred==y)&(y<256)).sum()),
                    NLL_sum=ce,exact=text==row['prompt'],Exact_entries=0,**decoded))
                io.atomic_json(output/'progress.json',dict(phase='readonly_untrained_sentence_reconstruction',rows=i+1,total=52,calls=budget.calls,
                    native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time(),latest=None))
        if support.state_sha(model.state_dict())!=digest or support.sha(p['cycle_source'])!=PARENT_SHA:raise ValueError('frozen model changed')
        io.atomic_json(output/'reconstructions.json',records)
        n=sum(r['symbols'] for r in records);b=sum(r['source_bytes'] for r in records)
        result=dict(status='FRESH_FUZZY_EVAL_V66_COMPLETE',rows=52,calls=budget.calls,optimizer_updates=0,checkpoints_written=0,Exact_entries=0,copy_bytes=0,denoise_calls=0,
            byte_accuracy=sum(r['byte_correct'] for r in records)/b,byte_EOS_accuracy=sum(r['correct'] for r in records)/n,
            exact_windows=sum(r['exact'] for r in records),UTF8=sum(r['UTF8'] for r in records),EOS=sum(r['EOS'] for r in records),seconds=time.time()-began,
            source_model_SHA=PARENT_SHA,frozen_before_after_digest=digest,limitations='fresh authored contrast pairs and templated long contexts, not random corpus or universal semantic evidence')
        io.atomic_json(output/'result.json',result);return 0
    except BaseException as error:
        io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc()));return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',action='store_true')
    a=parser.parse_args();raise SystemExit(run(a.plan.resolve(),a.output.resolve(),a.preflight))
