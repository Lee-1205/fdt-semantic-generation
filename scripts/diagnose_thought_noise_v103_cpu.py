"""ONE bounded read-only TRAIN-only noise-gap diagnostic. No optimizer/E/GPU."""
import argparse,hashlib,json,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
from torch.nn import functional as F
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_memory_thought_20261114_v102 import Data,evaluate_one
from run_thought_cache_v95 import load_codec
from research.memory_thought_diffusion_v94 import ThoughtConfig,MemoryThoughtDiffusion
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.thought_train_engine_v98 import rng_state,restore_rng
from research.readout_decision_trace_v88 import pack as trace_pack,decode
from types import SimpleNamespace


def main(path,out):
    began=time.monotonic();torch.set_num_threads(4);p=io.load_json(path);prior_rng=rng_state()
    assert p['schema']=='TRAIN_noise_gap_diagnostic_v103' and p['max_calls']==92 and p['authorized']==False
    assert shutil.disk_usage(ROOT).free>=108*1024**3+4*1024**2
    for name,sha in p['files'].items():assert support.sha(name)==sha
    directory=path.parent/'cpu_preflight_owned/runtime';directory.mkdir(exist_ok=False)
    calls=dict(denoise=0,readout=0)
    def charge(kind):
        if time.monotonic()-began>=75 or sum(calls.values())>=92:raise TimeoutError('CPU75seconds/92calls closed scope')
        calls[kind]+=1
    codec,epoch,old_sha,selection=load_codec(p,'cpu');d=Data(p)
    saved=torch.load(p['thought_final']['path'],map_location='cpu',weights_only=True);initial=torch.load(p['thought_initial']['path'],map_location='cpu',weights_only=True)
    assert support.state_sha(saved)==p['thought_final']['algorithm_SHA']
    thought=MemoryThoughtDiffusion(ThoughtConfig(**saved['engine']['thought_config']));thought.load_state_dict(saved['engine']['thought'],strict=True);thought.requires_grad_(False)
    thought_sha=support.state_sha(thought.state_dict());checkpoint_sha=support.state_sha(saved)
    greater=next(x for x in sorted(d.train.values(),key=lambda x:x['id']) if x['label']=='greater')
    reverse=next(x for x in d.train.values() if x['pair_key']==greater['pair_key'] and x['id']!=greater['id'])
    unknown=next(x for x in sorted(d.train.values(),key=lambda x:x['id']) if x['label']=='unknown')
    unknown_reverse=next(x for x in d.train.values() if x['pair_key']==unknown['pair_key'] and x['id']!=unknown['id'])
    rows=[greater,reverse,unknown,unknown_reverse];records=[];packets=[]
    class ReadonlyGuard:
        def assert_fast(self):
            assert all(not x.requires_grad and x.grad is None for x in codec.parameters())
    engine=SimpleNamespace(thought=thought,codec=codec,update=ReadonlyGuard())
    for row in rows:
        m=d.source(row['id'],'cpu');target=d.target(row);noise=torch.randn(1,35,512,generator=torch.Generator().manual_seed(103))
        for t in (.05,.5,.95):
            thought.train();charge('denoise')
            with torch.no_grad():
                latent,clean=thought.training_objective(m,target['semantic_target'],target['valid_extent_mask'],torch.tensor([t]),noise)
                charge('readout');logits=codec.verbalize(DifferentiableCodecCommit.from_semantic(clean,m.memory_revisions,m.model_epoch))
                labels=target['labels'];trace=trace_pack(logits,labels,[0,1]);decoded=decode(trace);n=len(row['answer'].encode());z=logits.reshape(-1,258)
                byte=float(F.cross_entropy(z[:n],labels.flatten()[:n]));eos=float(F.cross_entropy(z[n:n+1],labels.flatten()[n:n+1]))
            records.append(dict(id=row['id'],label=row['label'],expected=row['answer'],path='teacher_noisy_target_DIAGNOSTIC_ONLY',t=t,
                latent=float(latent),byte_CE=byte,EOS_CE=eos,output=decoded['text'],EOS=decoded['EOS'],target_prefix=decoded['text'].startswith(row['answer'])))
            packets.append(dict(id=row['id'],t=t,semantic=clean,trace=trace))
        thought.eval();raw,decoded,labels=evaluate_one(engine,d,row,charge)
        n=len(row['answer'].encode());records.append(dict(id=row['id'],label=row['label'],expected=row['answer'],path='target_free16steps',t=None,
            byte_CE=float(-raw['trace']['gold_log_probability'][:n].mean()),EOS_CE=float(-raw['trace']['gold_log_probability'][n]),
            output=decoded['text'],EOS=decoded['EOS'],target_prefix=decoded['text'].startswith(row['answer'])))
        packets.append(raw)
    # Count actual high-noise exposure of each TRAIN row, from saved RNG alone.
    generator=torch.Generator();generator.set_state(initial['engine']['ALL_RNG']['torch']);times=[]
    for identity in saved['engine']['orders']:
        t=float(.001+.998*torch.rand(1,generator=generator));torch.randn((1,35,512),generator=generator);times.append((identity,t))
    assert torch.equal(generator.get_state(),saved['engine']['ALL_RNG']['torch'])
    coverage={str(threshold):dict(exposures=sum(t>=threshold for _,t in times),unique_TRAIN_rows=len({i for i,t in times if t>=threshold})) for threshold in (.9,.95,.99,1.)}
    assert calls==dict(denoise=76,readout=16) and support.state_sha(saved)==checkpoint_sha and support.state_sha(thought.state_dict())==thought_sha
    assert support.state_sha(codec.state_dict())==old_sha and not torch.cuda.is_initialized()
    restore_rng(prior_rng)
    packet_path=directory/'raw_train_only_noise_gap.pt'
    with packet_path.open('xb') as f:torch.save(packets,f)
    assert packet_path.stat().st_size<4*1024**2
    report=dict(status='READONLY_TRAIN_ONLY_NOISE_GAP_DIAGNOSTIC_COMPLETE',records=records,actual_high_noise_exposure=coverage,
        TRAIN_rows4=len(rows)==4,teacher_Y_explicitly_diagnostic_not_inference=True,all_full_model_weights_unchanged=True,optimizer_steps=0,encode_calls=0,
        calls=calls,new_budget_closed=True,raw_SHA=support.sha(packet_path),global_RNG_restored=True,CUDA_initialized=False,seconds=time.monotonic()-began)
    io.atomic_json(directory/'result.json',report)
    print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(fullsize_engine_restore=True,all_parent_tensors_exact=True,all_old_Adam_exact=True,CUDA_initialized=False,
        diagnostic_only=True,calls=calls,optimizer_steps=0,TRAIN_only4=True,all_weights_unchanged=True)))
    print('CPU_PREFLIGHT_PASS')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--preflight',action='store_true');x=a.parse_args()
    assert x.preflight;main(x.plan.resolve(),x.output.resolve())
