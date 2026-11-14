"""Separate bounded final TRAIN134 evaluation, paired epsilon, optimizer0.
This completes32positive reversepairs/80positive rows; no budget reuse/SGD.
"""
import argparse,collections,hashlib,json,os,shutil,sys,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P),str(P/'scripts'),str(P/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_thought_cache_v95 import load_codec
from run_answer_meaning_v111_c_cpu import Data,score
from run_memory_thought_20261114_v105 import Budget as SafeSave
from owned_output_size_v52 import owned_output_bytes
from research.thought_grid_tail_v104 import GridTailThought
from research.memory_thought_diffusion_v94 import ThoughtConfig
from research.readout_decision_trace_v88 import pack as trace_pack,decode


class Budget:
    def __init__(self,p):self.p=p;self.out=Path(p['output']);self.calls=dict(denoise=0,readout=0,encode=0)
    def before(self,cost=0,pending=0,emergency=False):
        if (self.out/'STOP').exists() or time.time()>=self.p['cutoff']:raise InterruptedError('fixed readonly cutoff/STOP')
        if sum(self.calls.values())+cost>2300:raise RuntimeError('separate2300readonlycallcap')
        if shutil.disk_usage(P).free<50*1024**3+pending or owned_output_bytes(self.out)+pending>64*1024**2:raise OSError('C50floor/64MiBwhole readonly')
    def charge(self,kind):
        if kind not in ('denoise','readout') or sum(self.calls.values())>=2300:raise RuntimeError('noencode boundedreadonlycalls')
        self.calls[kind]+=1
    save=SafeSave.save


def main(path,out):
    p=io.load_json(path)
    if (p['schema']!='fdt_v111_fullTRAIN_paired_readonly' or p['max_calls']!=2300 or p['expected_calls']!=2278
            or p['wall_seconds']!=120 or p['deadline']-p['start']!=120 or p['cutoff']!=p['deadline']-30
            or Path(p['output']).resolve()!=out or not p['authorized']):raise ValueError('NEWreadonlypurpose/time/callcaps')
    for f,digest in p['files'].items():
        if support.sha(f)!=digest:raise ValueError('readonlyinputpin '+f)
    if (out/'worker_started.json').exists():raise ValueError('ONEreadonlyworker')
    io.atomic_json(out/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');budget=Budget(p);rows=[];began=time.monotonic()
    def progress(phase):io.atomic_json(out/'progress.json',dict(phase=phase,time=time.time(),cursor=len(rows),attempt=0,update=0,calls=budget.calls,native_pid=os.getpid()))
    try:
        progress('readonly_loading');mainp=io.load_json(p['parent_plan']);codec,epoch,before,selection=load_codec(mainp,'cpu');d=Data(mainp)
        ref=p['final_checkpoint'];full=torch.load(ref['path'],map_location='cpu',weights_only=True)
        assert support.state_sha(full)==ref['algorithm_SHA']
        thought=GridTailThought(ThoughtConfig(**full['engine']['thought_config']));thought.load_state_dict(full['engine']['thought'],strict=True)
        thought.requires_grad_(False);thought.eval();theta_before=support.state_sha(thought.state_dict());del full
        for row in sorted(d.train.values(),key=lambda r:r['id']):
            budget.before(17);m=d.source(row['id']);seed=int(hashlib.sha256(('V111_FULLTRAIN_COMMONPAIR:'+row['pair_key']).encode()).hexdigest()[:15],16)
            noise=torch.randn(1,35,512,generator=torch.Generator().manual_seed(seed))
            with torch.no_grad():
                final=thought.sample(m,noise,steps=16,charge=budget.charge);budget.charge('readout');logits=codec.verbalize(final.as_codec_commit())
                packet=d.target(row);trace=trace_pack(logits,packet['labels'],[0,1]);decoded=decode(trace)
                err=(final.semantic-packet['semantic_target']).square().mean(-1);mask=packet['valid_extent_mask']
            raw=dict(schema='V111_FINAL_TRAIN_commonpair_noise_readonly',input_id=row['id'],noise_seed=seed,pair_key=row['pair_key'],
                final_semantic=final.semantic,trace=trace,no_target_in_inference=True,optimizer_steps=0)
            saved=budget.save(out/'raw'/(row['id']+'.pt'),raw)
            active=packet['labels'].flatten()!=257
            rows.append(dict(input_id=row['id'],pair_key=row['pair_key'],label=row['label'],expected=row['answer'],output=decoded['text'],
                EOS=decoded['EOS'],UTF8=decoded['UTF8'],noise_seed=seed,body_MSE=float((err*mask).sum()/mask.sum()),tail_MSE=float((err*(~mask)).sum()/(~mask).sum().clamp_min(1)),
                byte_NLL=float(-trace['gold_log_probability'][active].mean()),**score(decoded['text'],row),raw=saved));progress('readonly_TRAIN134_commonpair_epsilon')
        assert len(rows)==134 and budget.calls==dict(denoise=2144,readout=134,encode=0)
        assert support.state_sha(thought.state_dict())==theta_before and support.state_sha(codec.state_dict())==before
        pairs=collections.defaultdict(list)
        for r in rows:pairs[r['pair_key']].append(r)
        pp=[v for v in pairs.values() if len(v)==2 and all(r['label']!='unknown' for r in v)]
        up=[v for v in pairs.values() if len(v)==2 and all(r['label']=='unknown' for r in v)]
        assert len(pp)==32 and len(up)==26 and all(v[0]['noise_seed']==v[1]['noise_seed'] for v in pp+up)
        io.atomic_json(out/'assessment.json',rows)
        io.atomic_json(out/'result.json',dict(status='V111_FINAL_TRAIN134_PAIRED_READONLY_COMPLETE',rows=134,positive_rows=80,unknown_rows=54,
            positive_reverse_pairs=32,unknown_reverse_pairs=26,singletons=18,optimizer_steps=0,modelcalls=budget.calls,
            common_epsilon_within_every_actualpair=True,all_source_codec_and_thought_tensors_unchanged=True,
            frozen_codec_model_SHA=before,thought_model_SHA=theta_before,seconds=time.monotonic()-began,CUDA_initialized=torch.cuda.is_initialized(),
            scientific_semantic_accuracy_requires_independent_manual_review=True,closed_new_budget_NO_unused_reuse=True))
        progress('readonly_terminal_complete');return 0
    except BaseException as e:
        io.atomic_json(out/'incident.json',dict(error=repr(e),traceback=traceback.format_exc(),calls=budget.calls,rows=len(rows)));return 2


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--plan',type=Path,required=True);q.add_argument('--output',type=Path,required=True);a=q.parse_args();raise SystemExit(main(a.plan.resolve(),a.output.resolve()))
