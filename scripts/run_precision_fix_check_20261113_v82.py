"""Verify one fixed FP32 output patch on the existing 64 cached Commit panel."""
import argparse,gc,io as memoryio,json,os,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_readout_cause_20261113_v81 import load_heads,commit_of
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.readout_precision_policy_v81 import readout,POLICY
from research.commit_patch_readout_v54 import decode_fixed_logits
from owned_output_size_v52 import owned_output_bytes

def validate(path,out):
    p=io.load_json(path)
    if (p['schema']!='fdt_precision_fix_readonly_v82' or p['wall_seconds']!=120 or p['max_calls']!=70 or p['expected_calls']!=64
        or p['max_output_bytes']!=32*1024**2 or p['attempts']!=0 or p['epochs']!=0 or Path(p['output']).resolve()!=out):raise ValueError('fixed precision check schema')
    if p['authorized'] and (p['deadline']-p['start']!=120 or p['cutoff']!=p['deadline']-30):raise ValueError('new fixed clock')
    for f,s in p['files'].items():
        if support.sha(f)!=s:raise ValueError('pin '+f)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('core')
    return p

@torch.no_grad()
def run(path,out,preflight=False):
    p=validate(path,out);torch.set_num_threads(4);heads,proof=load_heads(p);items=io.load_json(p['input_list']);assert len(items)==64
    if preflight:
        z,_,_=commit_of(p,items[0]);c=DifferentiableCodecCommit.from_semantic(z,(('check','v1','metadata'),),'fixed')
        for h in heads.values():
            stream=memoryio.BytesIO();torch.save(h.state_dict(),stream);stream.seek(0);h.load_state_dict(torch.load(stream,map_location='cpu',weights_only=True),strict=True)
            scores=readout(h,c);assert scores.dtype==torch.float32 and torch.isfinite(scores).all()
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(fullsize_engine_restore=True,restored_component='actual fixed 2908192-parameter readout; original fullstate/Adam/RNG reconstructed, encoder unexecuted',
            all_parent_tensors_exact=True,all_old_Adam_exact=True,CUDA_initialized=torch.cuda.is_initialized(),actual_CPU_readout_calls=2,optimizer_steps=0,encoder_calls=0,heads=proof)))
        print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (out/'worker_started.json').exists():raise ValueError('ONE readonly check')
    io.atomic_json(out/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()));began=time.time();calls=0;records=[];head=heads['V80'].to('cuda')
    try:
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        for i,info in enumerate(items):
            if calls>=70 or time.time()>=p['cutoff'] or (out/'STOP').exists():raise InterruptedError('calls/clock/STOP')
            if shutil.disk_usage(ROOT).free<101*1024**3 or owned_output_bytes(out)>p['max_output_bytes']:raise OSError('protected disk/cap')
            z,_,_=commit_of(p,info);c=DifferentiableCodecCommit.from_semantic(z.to('cuda'),(('check','v1','metadata'),),'fixed')
            calls+=1;scores=readout(head,c).cpu()
            if not torch.isfinite(scores).all() or scores.dtype!=torch.float32 or torch.cuda.max_memory_allocated()>12*1024**3:raise ValueError('FP32/numeric/GPU cap')
            f=out/f'raw_{i:02d}.pt'
            with f.open('xb') as stream:torch.save(dict(logits=scores,source_id=info['row']['id'],original_raw_reference=info['raw'],policy=POLICY),stream)
            decoded=decode_fixed_logits(scores,16385)
            records.append(dict(id=info['row']['id'],output=decoded['text'],ids=decoded['ids'],EOS=decoded['EOS'],UTF8=decoded['UTF8'],raw=str(f),SHA=support.sha(f)))
            io.atomic_json(out/'progress.json',dict(phase='fixed_FP32_readout_only_verification',calls=calls,total=64,native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
            del z,c,scores;gc.collect()
        assert calls==64 and support.state_sha(head.state_dict())==proof['V80']['head_SHA']
        io.atomic_json(out/'reconstructions.json',records)
        io.atomic_json(out/'result.json',dict(status='FIXED_FP32_READOUT_V82_CHECK_COMPLETE',calls=calls,seconds=time.time()-began,optimizer_steps=0,encoder_calls=0,denoise_calls=0,checkpoints=0,
            Exact_entries=0,all_weights_unchanged=True,old_Adam_scaler_RNG_preserved=True,policy=POLICY,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
            scope='Same observed 64 correlated source rows. A numerical fix check, not a new benchmark, learned improvement, or abstract semantic-completeness certificate.'))
        return 0
    except BaseException as error:
        io.atomic_json(out/'incident.json',dict(error=repr(error),traceback=traceback.format_exc(),calls=calls));return 2

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--preflight',action='store_true');args=a.parse_args()
    raise SystemExit(run(args.plan.resolve(),args.output.resolve(),args.preflight))
