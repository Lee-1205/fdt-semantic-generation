"""NEW96call read-only TRAIN4 same-input/noise1/4/16-step comparison."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_memory_thought_20261114_v105 import Data
from run_thought_cache_v95 import load_codec
from research.thought_grid_tail_v104 import GridTailThought
from research.memory_thought_diffusion_v94 import ThoughtConfig
from research.readout_decision_trace_v88 import pack as trace_pack,decode
from research.thought_train_engine_v98 import rng_state,restore_rng


def main(path):
    start=time.monotonic();torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');p=io.load_json(path);rng=rng_state()
    assert p['schema']=='readonly_TRAIN_rollout_steps_v106' and p['max_calls']==96 and not p['authorized']
    assert shutil.disk_usage(ROOT).free>=108*1024**3+4*1024**2
    for f,sha in p['files'].items():assert support.sha(f)==sha
    out=path.parent/'cpu_preflight_owned/runtime';out.mkdir(exist_ok=False);calls=dict(denoise=0,readout=0)
    def charge(kind):
        if sum(calls.values())>=96 or time.monotonic()-start>=75:raise TimeoutError('NEW96/75sCPU bounds')
        calls[kind]+=1
    codec,epoch,codec_sha,selection=load_codec(p,'cpu');d=Data(p);saved=torch.load(p['final_thought']['path'],map_location='cpu',weights_only=True)
    assert support.state_sha(saved)==p['final_thought']['algorithm_SHA']
    thought=GridTailThought(ThoughtConfig(**saved['engine']['thought_config']),tail_weight=.05);thought.load_state_dict(saved['engine']['thought'],strict=True);thought.requires_grad_(False);thought.eval();before=support.state_sha(thought.state_dict())
    a=next(x for x in sorted(d.train.values(),key=lambda x:x['id']) if x['label']=='greater');b=next(x for x in d.train.values() if x['pair_key']==a['pair_key'] and x['id']!=a['id'])
    c=next(x for x in sorted(d.train.values(),key=lambda x:x['id']) if x['label']=='unknown');e=next(x for x in d.train.values() if x['pair_key']==c['pair_key'] and x['id']!=c['id'])
    records=[];packets=[]
    for row in (a,b,c,e):
        m=d.source(row['id'],'cpu');seed=int(hashlib.sha256(('V100_fixed17:'+row['id']).encode()).hexdigest()[:15],16)
        noise=torch.randn(1,35,512,generator=torch.Generator().manual_seed(seed))
        for steps in (1,4,16):
            with torch.no_grad():
                final=thought.sample(m,noise,steps=steps,charge=charge);charge('readout');logits=codec.verbalize(final.as_codec_commit())
                labels=d.target(row)['labels'];trace=trace_pack(logits,labels,[0,1]);decoded=decode(trace);n=len(row['answer'].encode())
            records.append(dict(id=row['id'],label=row['label'],expected=row['answer'],steps=steps,noise_seed=seed,output=decoded['text'],
                byte_CE=float(-trace['gold_log_probability'][:n].mean()),EOS_CE=float(-trace['gold_log_probability'][n]),EOS=decoded['EOS'],target_prefix=decoded['text'].startswith(row['answer'])))
            packets.append(dict(id=row['id'],steps=steps,semantic=final.semantic,trace=trace))
    assert calls==dict(denoise=84,readout=12) and support.state_sha(thought.state_dict())==before and support.state_sha(codec.state_dict())==codec_sha
    assert not torch.cuda.is_initialized();restore_rng(rng)
    with (out/'raw.pt').open('xb') as f:torch.save(packets,f)
    assert (out/'raw.pt').stat().st_size<4*1024**2
    io.atomic_json(out/'result.json',dict(status='READONLY_TRAIN_ROLLOUT_COMPARISON_COMPLETE',records=records,calls=calls,optimizer0=True,encoder0=True,
        all_weights_unchanged=True,CUDA_initialized=False,raw_SHA=support.sha(out/'raw.pt'),seconds=time.monotonic()-start,scope_closed=True,not_blind_generalization=True))
    print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(fullsize_engine_restore=True,all_parent_tensors_exact=True,all_old_Adam_exact=True,
        CUDA_initialized=False,diagnostic_only=True,calls=calls,optimizer_steps=0,TRAINonly4=True,all_weights_unchanged=True)))
    print('CPU_PREFLIGHT_PASS')


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--plan',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--preflight',action='store_true');a=q.parse_args();assert a.preflight;main(a.plan.resolve())
