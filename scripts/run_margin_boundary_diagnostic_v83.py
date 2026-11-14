"""Four readonly projections on the historical TRAIN100 bank, no SGD/encoder."""
import argparse,io as memoryio,json,os,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_readout_cause_20261113_v81 import load_heads
from research.train_margin_retention_v81 import reference,decision
from owned_output_size_v52 import owned_output_bytes

def validate(path,out):
    p=io.load_json(path)
    if p['schema']!='fdt_TRAIN_margin_boundary_readonly_v83' or p['wall_seconds']!=120 or p['max_calls']!=6 or p['expected_calls']!=4 or p['max_output_bytes']!=32*1024**2 or p['attempts']!=0 or p['epochs']!=0 or Path(p['output']).resolve()!=out:raise ValueError('fixed TRAIN-only diagnostic')
    if p['authorized'] and (p['deadline']-p['start']!=120 or p['cutoff']!=p['deadline']-30):raise ValueError('new readonly clock')
    for f,s in p['files'].items():
        if support.sha(f)!=s:raise ValueError('pin '+f)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('core')
    return p

@torch.no_grad()
def run(path,out,preflight=False):
    p=validate(path,out);torch.set_num_threads(4);heads,proof=load_heads(p)
    features=torch.load(p['bank_features'],map_location='cpu',weights_only=True);targets=torch.load(p['bank_targets'],map_location='cpu',weights_only=True)
    h=features['hidden'];y=targets['labels'];assert h.shape==(7487,512) and h.dtype==torch.bfloat16 and y.shape==(7487,16)
    if preflight:
        for head in heads.values():
            layer=head.net[2];s=memoryio.BytesIO();torch.save(layer.state_dict(),s);s.seek(0);layer.load_state_dict(torch.load(s,map_location='cpu',weights_only=True),strict=True)
            z=layer(h[:1].float());assert z.shape==(1,4128) and z.dtype==torch.float32 and torch.isfinite(z).all()
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(fullsize_engine_restore=True,restored_component='actual last Linear2117664 params, full parent model/Adam/RNG digest verified; cached historical BF16prefix ONLY',
            all_parent_tensors_exact=True,all_old_Adam_exact=True,CUDA_initialized=torch.cuda.is_initialized(),actual_CPU_projection_calls=2,optimizer_steps=0,encoder_calls=0,heads=proof)))
        print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (out/'worker_started.json').exists():raise ValueError('ONE diagnostic')
    began=time.time();calls=0;packets={};selected=None;bank=None;receipt=None
    io.atomic_json(out/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=began))
    try:
        torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        h=h.to('cuda');y=y.to('cuda');valid=y!=257
        for head in heads.values():head.to('cuda')
        for version,mode in [('V75','bf16'),('V80','bf16'),('V75','last_fp32'),('V80','last_fp32')]:
            if calls>=6 or time.time()>=p['cutoff'] or (out/'STOP').exists():raise InterruptedError('fixed call/clock/STOP')
            if shutil.disk_usage(ROOT).free<101*1024**3 or owned_output_bytes(out)>p['max_output_bytes']:raise OSError('disk/cap')
            calls+=1
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode=='bf16'):
                z=heads[version].net[2](h if mode=='bf16' else h.float()).reshape(7487,16,258)
            z=z.float()
            if not torch.isfinite(z).all() or torch.cuda.max_memory_allocated()>12*1024**3:raise ValueError('numeric/GPU cap')
            pred=z.argmax(-1);gold=z.gather(-1,y[...,None]).squeeze(-1);other=z.clone();other.scatter_(-1,y[...,None],-torch.inf);other_score,other_index=other.max(-1);margin=gold-other_score;del other
            if selected is None:
                weak=margin.masked_fill(~valid|~targets['initial_correct'].to('cuda'),torch.inf).min(-1).values
                selected=weak.argsort(stable=True)[:128]
            packet=dict(argmax=pred.to(torch.int16).cpu(),gold_score=gold.cpu(),best_other_score=other_score.cpu(),best_other_index=other_index.to(torch.int16).cpu(),
                margin=margin.cpu(),sample_patch_indices=selected.cpu(),sample_full_logits=z[selected].cpu(),prefix='SAME_HISTORICAL_CACHED_BF16_hidden_not_full_FP32_prefix')
            if mode=='bf16':
                native=targets['initial_argmax'] if version=='V75' else torch.load(p['bank_final_predictions'],map_location='cpu',weights_only=True)['argmax']
                packet['all_native_argmax_reproduced']=torch.equal(packet['argmax'].long(),native.long())
                if not packet['all_native_argmax_reproduced']:raise ValueError('entire BF16 bank native diagonal mismatch')
            if mode=='last_fp32' and version=='V75':bank=reference(z,y,fraction=.5,cap=1.)
            if mode=='last_fp32' and version=='V80':receipt=decision(z,bank)
            packets[version+'/'+mode]=packet
            io.atomic_json(out/'progress.json',dict(phase='four_TRAIN_bank_projection_diagnostics',calls=calls,total=4,native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
            del z,pred,gold,other_score,other_index,margin
        assert calls==4
        for v,head in heads.items():assert support.state_sha(head.state_dict())==proof[v]['head_SHA']
        with (out/'raw_projection_statistics.pt').open('xb') as f:torch.save(packets,f)
        io.atomic_json(out/'result.json',dict(status='TRAIN_MARGIN_BOUNDARY_V83_READONLY_COMPLETE',calls=calls,seconds=time.time()-began,encoder_calls=0,optimizer_steps=0,denoise_calls=0,checkpoints=0,
            all_weights_unchanged=True,old_Adam_scaler_RNG_preserved=True,Exact_entries=0,TRAIN_sources=100,hidden_rows=7487,full_native_BF16_diagonal_reproduced=True,
            margin_candidate_endpoint_receipt=dict(accept=receipt.accept,measurements=receipt.measurements),peak_GPU_bytes=torch.cuda.max_memory_allocated(),raw_SHA=support.sha(out/'raw_projection_statistics.pt'),
            caveat='Historical cached prefix remains BF16; last_fp32 is a final-projection component diagnostic, NOT full-FP32 runtime parity. All argmax/margins saved, only128patches full logits saved. No historical per-step logits replay, no optimizer experiment, no unseen meaning guarantee.'))
        return 0
    except BaseException as error:
        io.atomic_json(out/'incident.json',dict(error=repr(error),calls=calls,traceback=traceback.format_exc()));return 2

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--preflight',action='store_true');x=a.parse_args()
    raise SystemExit(run(x.plan.resolve(),x.output.resolve(),x.preflight))
