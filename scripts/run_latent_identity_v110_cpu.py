"""Verified E(X)->D(E(X)) TRAIN meanings only; fresh thought-only identity VP.
All codec/memory/output parameters frozen. This is NOT relation reasoning.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,hashlib,json,random,shutil,sys,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P),str(P/'scripts'),str(P/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_thought_cache_v95 import load_codec
from run_memory_thought_20261114_v105 import Budget as PreviousSafeBudget,seed_all
from owned_output_size_v52 import owned_output_bytes
from research.thought_cache_v95 import assemble_input
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.memory_thought_diffusion_v94 import ThoughtConfig,target_meaning
from research.thought_grid_tail_v104 import GridTailThought
from research.runtime_aligned_thought_v107 import training_rollout
from research.latent_identity_thought_v110 import LatentIdentityEngine,effective_velocity_rank
from research.readout_decision_trace_v88 import pack as trace_pack,decode

SCHEMA='fdt_fullrank_LATENT_ONLY_identity_thought_v110';OUTPUT=640*1024**2

def validate(path,out):
    p=io.load_json(path)
    fixed=dict(schema=SCHEMA,device='cpu',max_calls=13200,expected_calls=12920,max_attempts=67,train_rows=536,
        unique_TRAIN=134,batch=8,epochs=4,eval_rows=112,steps=16,wall_seconds=600,max_output_bytes=OUTPUT,seed=1111410,LR=1e-4)
    if any(p.get(k)!=v for k,v in fixed.items()) or Path(p['output']).resolve()!=out:raise ValueError('new fixed identity-only finite profile')
    if p['authorized'] and (p['deadline']-p['start']!=600 or p['cutoff']!=p['deadline']-120):raise ValueError('new480+120clock')
    for f,sha in p['files'].items():
        if support.sha(f)!=sha:raise ValueError('pinned file '+f)
    return p

class Data:
    def __init__(self,p,codec):
        screen=Path(p['screen']);report=io.load_json(screen/'result.json');rr=io.load_json(screen/'all480.json')
        selected=io.load_json(screen/'selected_TRAIN_only.json')
        if report['selected_TRAIN']!=134 or len(rr)!=480 or report['calls']!=dict(readout=480,encode=0,denoise=0):raise ValueError('actual completed480 screen')
        self.rows={x['id']:dict(x,answer=x['prompt']) for x in rr}
        exact={x['id'] for x in rr if x['split']=='train' and x['strict_exact']}
        if exact!={x['id'] for x in selected} or len(exact)!=134:raise ValueError('strict TRAIN gate exact IDs')
        self.train={k:self.rows[k] for k in sorted(exact)}
        self.evals=[x for x in self.rows.values() if x['split']!='train']
        self.probe=[dict(self.train[k],split='TRAIN_probe') for k in sorted(self.train)[:16]]
        if len(self.evals)!=96 or set(self.train)&{x['id'] for x in self.evals}:raise ValueError('all96 held rows; no DEV fitting')
        cache=Path(p['cache']);cres=io.load_json(cache/'result.json')
        self.static=torch.load(cres['static_memories']['path'],map_location='cpu',weights_only=True)
        self.dynamic={x['id']:torch.load(x['source_condition_ref']['path'],map_location='cpu',weights_only=True) for x in rr}
        self.terminal=codec.readout.readout.terminal.detach().clone()
        curriculum=io.load_json(p['curriculum']);self.orders=[r['id'] for epoch in curriculum for r in epoch];self.times=[r['t'] for epoch in curriculum for r in epoch]
        if len(curriculum)!=4 or len(self.orders)!=536 or any(len(ep)!=134 or {r['id'] for r in ep}!=exact for ep in curriculum):raise ValueError('each selected TRAIN exactlyonce each4epochs')
        self.meta=dict(task='IDENTITY_MEANING_RECOVERY_NOT_RELATION_REASONING',screen_result_SHA=support.sha(screen/'result.json'),
            screen_selected_SHA=support.sha(screen/'selected_TRAIN_only.json'),source_cache_SHA=support.sha(cache/'input_index.json'),
            finite_orders_SHA=support.sha(p['curriculum']),fresh_thought_not_old_relation_trajectory=True,output_on_D=True)
        self.codec_SHA=p['codec_SHA']
    def source(self,identity):return assemble_input(self.static,self.dynamic[identity],'cpu')
    def target(self,row):
        m=self.source(row['id']);commit=DifferentiableCodecCommit.from_semantic(m.source,m.memory_revisions,m.model_epoch)
        target,mask=target_meaning(commit,self.terminal,33)
        labels=torch.full((34,16),257,dtype=torch.long);b=row['prompt'].encode();labels.flatten()[:len(b)]=torch.tensor(list(b));labels.flatten()[len(b)]=256
        return dict(schema='LOSS_ONLY_answer_meaning_v95',answer_SHA=hashlib.sha256(b).hexdigest(),source_codec_SHA=self.codec_SHA,
            semantic_target=target,valid_extent_mask=mask,labels=labels,model_epoch=m.model_epoch,target_condition_features_never_used=True)
    def training(self,identity):
        if identity not in self.train:raise ValueError('only selected134 TRAIN')
        return self.source(identity),self.target(self.train[identity])

class Budget:
    def __init__(self,p):self.p=p;self.out=Path(p['output']);self.calls=dict(denoise=0,readout=0,encode=0)
    def before(self,cost=0,pending=0,emergency=False):
        if not emergency and ((self.out/'STOP').exists() or time.time()>=self.p['cutoff']):raise InterruptedError('fixed cutoff/STOP')
        if sum(self.calls.values())+cost>13200:raise RuntimeError('closed13200callcap')
        if shutil.disk_usage(P).free<100*1024**3 or shutil.disk_usage(self.out).free<1024**3+pending:raise OSError('C100GiB/D1GiB+pending protected')
        if owned_output_bytes(self.out)+pending>OUTPUT:raise OSError('D640MiB whole incl STOPtemp')
    def charge(self,kind):
        if kind not in ('denoise','readout') or sum(self.calls.values())>=13200:raise RuntimeError('only thought/D finite calls')
        self.calls[kind]+=1
    save=PreviousSafeBudget.save

def evaluate_one(engine,d,row,charge):
    m=d.source(row['id']);seed=int(hashlib.sha256(('V110_identity17:'+row['id']).encode()).hexdigest()[:15],16)
    noise=torch.randn(1,35,512,generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        final=engine.thought.sample(m,noise,steps=16,charge=charge);charge('readout');logits=engine.codec.verbalize(final.as_codec_commit())
        # Scoring starts AFTER target-free final meaning and fixedD.
        packet=d.target(row);trace=trace_pack(logits,packet['labels'],[0,1]);decoded=decode(trace)
    raw=dict(schema='FINAL_identity_meaning_then_frozen_D_v109',input_id=row['id'],noise_seed=seed,final_semantic=final.semantic,
        trace=trace,no_target_in_inference=True,source_meaning_condition_is_intentional_identity_task=True)
    return raw,decoded,packet

def preflight(p):
    begin=time.monotonic();directory=Path(p['_plan']).parent/'cpu_preflight_owned/runtime';directory.mkdir(exist_ok=False)
    codec,epoch,before,selection=load_codec(p,'cpu');d=Data(p,codec);seed_all(p['seed']);thought=GridTailThought(ThoughtConfig(hidden=520))
    engine=LatentIdentityEngine(thought,codec,p['codec_SHA'],d.orders[:3],times=[1.,.5,.0625],batch=2,lr=p['LR'],source_metadata=d.meta)
    counts=[];m,packet=d.training(d.orders[0]);noise=torch.randn(1,35,512,generator=torch.Generator().manual_seed(109))
    trajectory=training_rollout(thought,m,noise,16,counts.append);thought.eval();final=thought.sample(m,noise,steps=16,charge=counts.append)
    gap=float((trajectory.detach()-final.semantic).abs().max());assert gap<1e-4
    del trajectory
    with torch.no_grad():
        padded=codec.verbalize(DifferentiableCodecCommit.from_semantic(packet['semantic_target'],m.memory_revisions,m.model_epoch))
        checked=decode(trace_pack(padded,packet['labels'],[0,1]));assert checked['text']==d.train[d.orders[0]]['prompt'] and checked['EOS']
    thought.train();engine.begin();micro_start=time.monotonic()
    for identity in engine.orders[:2]:engine.micro(identity,*d.training(identity))
    engine.finish();engine.begin();engine.micro(engine.orders[2],*d.training(engine.orders[2]))
    micros_seconds=time.monotonic()-micro_start;ready=engine.payload();f=directory/'populated_READY_fullstate.pt'
    with f.open('xb') as o:torch.save(ready,o)
    saved=torch.load(f,map_location='cpu',weights_only=True);saved_hash=support.state_sha(saved);assert saved_hash==support.state_sha(ready)
    engine.finish();expected=engine.payload();engine.restore(saved);engine.finish();actual=engine.payload()
    assert support.state_sha(expected)==support.state_sha(actual) and support.state_sha(saved)==saved_hash
    thought.eval();raw,decoded,_=evaluate_one(engine,d,d.train[d.orders[0]],counts.append)
    with (directory/'identity_eval.pt').open('xb') as o:torch.save(raw,o)
    engine.update.audit_endpoint();assert support.state_sha(codec.state_dict())==before and not torch.cuda.is_initialized()
    report=dict(fullsize_engine_restore=True,all_parent_tensors_exact=True,all_old_Adam_exact=True,CUDA_initialized=False,
        fresh_random_thought_not_parent_model=True,thought_parameters=sum(x.numel() for x in thought.parameters()),
        allcodec_parameters_buffers_frozen=True,only_thought_optimizer=True,actual16step_training_vs_eval_maxgap=gap,
        actual_fullsize_READY_Adam_scaler_RNG_nextupdate_exact=True,actual_padded_selected_target_roundtrip_exact=True,
        saved_READY_SHA=support.sha(f),saved_READY_algorithm_SHA=saved_hash,actual_temporary_updates3=True,
        training_micros3_seconds=micros_seconds,total_seconds=time.monotonic()-begin,source_encoder_calls0=True,
        NN_calls=dict(denoise=99,readout=2),D_calls_during_training=engine.runtime['calls']['readout'],effective_velocity_rank=effective_velocity_rank(thought),identity_not_reasoning=True,language_loss_NOT_used=True,output_and_preflight_on_D=True)
    io.atomic_json(directory/'actual_CPU_report.json',report);print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0

def run(path,out,fixture):
    p=validate(path,out);p['_plan']=str(path);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    if fixture:return preflight(p)
    if not p['authorized'] or (out/'worker_started.json').exists():raise ValueError('ONE new sealed identity trial')
    io.atomic_json(out/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
    budget=Budget(p);driver=dict(phase='loading',started=time.time(),plan_SHA=support.sha(path),eval_refs=[],checkpoints=[]);engine=None;guard=False
    def progress():
        r=engine.runtime if engine else {}
        io.atomic_json(out/'progress.json',dict(phase=driver['phase'],attempt=r.get('attempts',0),update=r.get('updates',0),cursor=r.get('cursor',0),calls=budget.calls,
            time=time.time(),elapsed=time.time()-driver['started'],native_pid=os.getpid(),eval_rows=len(driver['eval_refs']),latest_durable=driver['checkpoints'][-1]['path'] if driver['checkpoints'] else None))
    def checkpoint(name,emergency=False):
        payload=dict(schema='pure_latent_identity_complete_fullstate_v110',engine=engine.payload(),driver=dict(driver),budget=dict(budget.calls),
            frozen_codec_reference=p['selection'],source_cache=p['cache'],selection_screen=p['screen'],plan_SHA=driver['plan_SHA'])
        ref=budget.save(out/'checkpoints'/(name+'.pt'),payload,emergency=emergency);driver['checkpoints'].append(ref);io.atomic_json(out/'latest.json',ref);progress();return ref
    def assess(d,rows,phase):
        driver['phase']=phase;engine.thought.eval();progress()
        for row in rows:
            budget.before(17);raw,decoded,packet=evaluate_one(engine,d,row,budget.charge);ref=budget.save(out/'raw'/(phase+'_'+row['id']+'.pt'),raw)
            active=packet['labels'].flatten()!=257
            driver['eval_refs'].append(dict(input_id=row['id'],split=row['split'],phase=phase,expected=row['prompt'],output=decoded['text'],
                codec_roundtrip_exact=row['strict_exact'],EOS=decoded['EOS'],UTF8=decoded['UTF8'],exact=decoded['text']==row['prompt'],
                byte_NLL=float(-raw['trace']['gold_log_probability'][active].mean()),raw=ref));progress()
        engine.thought.train()
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        budget.before();codec,epoch,before,selection=load_codec(p,'cpu');d=Data(p,codec);seed_all(p['seed']);thought=GridTailThought(ThoughtConfig(hidden=520))
        engine=LatentIdentityEngine(thought,codec,p['codec_SHA'],d.orders,times=d.times,batch=8,lr=p['LR'],source_metadata=d.meta)
        driver['phase']='initialized';checkpoint('state_0000');assess(d,d.probe+d.evals,'baseline')
        driver['phase']='selected_meaning_identity_thought_ONLY_training';progress()
        while engine.runtime['phase']!='DONE':
            budget.before(18);engine.begin()
            while engine.runtime['phase']=='MICROS':
                budget.before(18);identity=engine.orders[engine.runtime['cursor']];engine.micro(identity,*d.training(identity),budget.charge)
            engine.finish();progress()
            if engine.runtime['attempts'] in (34,67):checkpoint('state_'+str(engine.runtime['attempts']).zfill(4))
        assert engine.runtime['cursor']==536 and engine.runtime['attempts']==67
        assert engine.runtime['calls']==dict(denoise=9112,readout=0)
        assess(d,d.probe+d.evals,'final');engine.update.audit_endpoint()
        assert budget.calls==dict(denoise=12696,readout=224,encode=0) and support.state_sha(codec.state_dict())==before
        io.atomic_json(out/'assessment.json',driver['eval_refs'])
        io.atomic_json(out/'result.json',dict(status='FULLRANK_LATENT_ONLY_IDENTITY_TRIAL_COMPLETE_NOT_REASONING',execution_device='cpu',CUDA_initialized=torch.cuda.is_initialized(),
            attempts=67,updates=engine.runtime['updates'],AMP_skips=engine.runtime['AMP_skips'],cursor=536,exposure_target_bytes=engine.runtime['exposure_bytes'],
            calls=budget.calls,seconds=time.time()-driver['started'],selected_TRAIN=134,selected_DEV_for_training=0,source_encode0=True,
            allcodec_parameters_buffers_frozen=True,frozen_codec_SHA_before_after=before,only_new_thought_optimizer=True,
            fresh_thought_and_Adam=True,checkpoints=driver['checkpoints'],assessment_rows224=True,full16step_training_gradient=True,
            identity_recovery_NOT_answer_reasoning=True,D_forward_during_training=0,language_loss_NOT_used=True,hidden520=True,all_oldbudgets_closed=True,no_promotion_or_automatic_repeat=True))
        driver['phase']='terminal_complete';progress();return 0
    except BaseException as e:
        incident=dict(error=repr(e),traceback=traceback.format_exc(),phase=driver['phase'],calls=budget.calls)
        if engine:
            try:incident['emergency']=checkpoint('STOP_current',emergency=True)
            except BaseException as se:incident['emergency_error']=repr(se)
        io.atomic_json(out/'incident.json',incident);return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--plan',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--preflight',action='store_true');a=q.parse_args();raise SystemExit(run(a.plan.resolve(),a.output.resolve(),a.preflight))
