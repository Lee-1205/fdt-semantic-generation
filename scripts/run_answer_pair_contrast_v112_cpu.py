"""ONE finite theta-only pair-contrast repair. Old assets remain immutable."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,collections,ctypes,hashlib,json,shutil,sys,time,traceback
from pathlib import Path
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P),str(P/'scripts'),str(P/'src')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
import run_answer_meaning_v111_c_cpu as previous
from run_thought_cache_v95 import load_codec
from run_memory_thought_20261114_v105 import Budget as SafeSave,seed_all
from owned_output_size_v52 import owned_output_bytes
from research.answer_pair_contrast_v112 import PairContrastEngine
from research.thought_grid_tail_v104 import GridTailThought
from research.memory_thought_diffusion_v94 import ThoughtConfig
from research.runtime_aligned_thought_v107 import training_rollout
from research.readout_decision_trace_v88 import pack as trace_pack,decode

SCHEMA='fdt_answer_pair_contrast_v112';OUTPUT=1024**3


def validate(path,out):
    p=io.load_json(path)
    fixed=dict(schema=SCHEMA,device='cpu',max_calls=41000,expected_calls=40358,max_attempts=268,train_rows=2144,
        unique_TRAIN=134,batch=8,epochs=16,eval_rows=230,steps=16,wall_seconds=1200,max_output_bytes=OUTPUT,
        seed=1111412,LR=1e-4,pair_coefficient=.25)
    if any(p.get(k)!=v for k,v in fixed.items()) or Path(p['output']).resolve()!=out:raise ValueError('new finite paired objective/cost schema')
    if p['authorized'] and (p['deadline']-p['start']!=1200 or p['cutoff']!=p['deadline']-120):raise ValueError('1080+120 new clock')
    for f,digest in p['files'].items():
        if support.sha(f)!=digest:raise ValueError('pinned input '+f)
    return p


class Data(previous.Data):
    def __init__(self,p):
        super().__init__(p);self.units=[]
        order=[r for ep in io.load_json(p['curriculum']) for r in ep]
        for j in range(0,len(order),2):
            a,b=order[j:j+2];ra,rb=self.train[a['id']],self.train[b['id']]
            real=ra['pair_key']==rb['pair_key']
            if real!=a['real_reverse_pair'] or real!=b['real_reverse_pair']:raise ValueError('actual pair ID, not arbitrary co-batch rows')
            if real and not torch.equal(self.target(ra)['valid_extent_mask'],self.target(rb)['valid_extent_mask']):raise ValueError('equal actual-pair body masks')
            self.units.append(dict(ids=[a['id'],b['id']],real_pair=real,key=ra['pair_key'] if real else 'unpaired_schedule:'+str(j)))
        assert sum(u['real_pair'] for u in self.units)==58*16
        self.meta.update(task='ANSWER_PAIR_CONTRAST_EY_v112',warm_thought_SHA=p['warm']['SHA'])
    def unit_inputs(self,unit):
        values=[self.training(i) for i in unit['ids']]
        return unit['ids'],[x[0] for x in values],[x[1] for x in values]


def warm_thought(p):
    ref=p['warm'];full=torch.load(ref['path'],map_location='cpu',weights_only=True)
    if support.sha(ref['path'])!=ref['SHA'] or support.state_sha(full)!=ref['algorithm_SHA']:raise ValueError('preserved V111 parent')
    e=full['engine']
    if e['schema']!='conditional_answer_meaning_fullstate_v111' or e['codec_SHA']!=p['codec_SHA'] or e['thought_config']!=dict(ThoughtConfig(hidden=520).__dict__):raise ValueError('actual V111 conditional parent architecture')
    thought=GridTailThought(ThoughtConfig(hidden=520));thought.load_state_dict(e['thought'],strict=True)
    if support.state_sha(thought.state_dict())!=p['warm_thought_model_SHA']:raise ValueError('ALL theta warm, optimizer NOT resumed')
    return thought


def inherited_baseline(p):
    pairs=io.load_json(p['baseline_TRAIN_assessment']);dev=io.load_json(p['baseline_DEV_assessment'])
    rows=[dict(r,split='TRAIN_all',phase='baseline') for r in pairs]
    rows += [dict(r,phase='baseline') for r in dev if r['phase']=='final' and r['split'].startswith('dev_')]
    if len(rows)!=230 or len({r['input_id'] for r in rows})!=230:raise ValueError('saved final134TRAIN+96DEV baseline, no new forward')
    for r in rows:
        if support.sha(r['raw']['path'])!=r['raw']['SHA']:raise ValueError('inherited baseline raw')
    return rows


def evaluate_one(engine,d,row,charge):
    m=d.source(row['id'])
    key='V111_FULLTRAIN_COMMONPAIR:'+row['pair_key'] if row['split']=='TRAIN_all' else 'V111_ANSWER17:'+row['id']
    seed=int(hashlib.sha256(key.encode()).hexdigest()[:15],16)
    noise=torch.randn(1,35,512,generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        final=engine.thought.sample(m,noise,16,charge);charge('readout');logits=engine.codec.verbalize(final.as_codec_commit())
        packet=d.target(row);trace=trace_pack(logits,packet['labels'],[0,1]);decoded=decode(trace)
        err=(final.semantic-packet['semantic_target']).square().mean(-1);mask=packet['valid_extent_mask']
    raw=dict(schema='V112_final_answer_meaning_frozen_D',input_id=row['id'],noise_seed=seed,
        final_semantic=final.semantic,trace=trace,no_target_in_inference=True)
    return raw,decoded,packet,float(err[mask].mean()),float(err[~mask].mean())


class Budget:
    def __init__(self,p):self.p=p;self.out=Path(p['output']);self.calls=dict(denoise=0,readout=0,encode=0)
    def before(self,cost=0,pending=0,emergency=False):
        if not emergency and ((self.out/'STOP').exists() or time.time()>=self.p['cutoff']):raise InterruptedError('fixed cutoff/STOP')
        if sum(self.calls.values())+cost>41000:raise RuntimeError('new41000callcap')
        if shutil.disk_usage(P).free<50*1024**3+pending or owned_output_bytes(self.out)+pending>OUTPUT:raise OSError('C50GiB/1GiBwholeoutput incl STOP/temp')
    def charge(self,kind):
        if kind not in ('denoise','readout') or sum(self.calls.values())>=41000:raise RuntimeError('bounded thought/D only')
        self.calls[kind]+=1
    save=SafeSave.save


def preflight(p):
    began=time.monotonic();directory=Path(p['_plan']).parent/'cpu_preflight_owned/runtime';directory.mkdir(exist_ok=False)
    if shutil.disk_usage(P).free<100*1024**3+512*1024**2+64*1024**2:raise OSError('C preflight finish reserve')
    codec,epoch,before,selection=load_codec(p,'cpu');d=Data(p);seed_all(p['seed']);thought=warm_thought(p)
    real=[u for u in d.units[:67] if u['real_pair']];single=[u for u in d.units[:67] if not u['real_pair']]
    units=[real[0],single[0],real[1],real[2],real[3],single[1],real[4]]
    where={i:j for j,i in enumerate(d.orders)}
    orders=[i for u in units for i in u['ids']];times=[d.times[where[i]] for i in orders]
    e=PairContrastEngine(thought,codec,p['codec_SHA'],orders,times=times,units=units,batch=8,lr=p['LR'],source_metadata=d.meta)
    assert not e.optimizer.state and e.thought.velocity.weight.shape==(512,520)
    count=collections.Counter()
    def charge(kind):
        count[kind]+=1
        if sum(count.values())>450:raise RuntimeError('preflight450callcap')
    m=d.source(orders[0]);noise=torch.randn(1,35,512,generator=torch.Generator().manual_seed(112))
    z=training_rollout(thought,m,noise,16,charge);thought.eval();sample=thought.sample(m,noise,16,charge)
    gap=float((z.detach()-sample.semantic).abs().max());assert gap<1e-4
    assert torch.equal(sample.semantic,thought.sample(m,noise,16,charge).semantic);del z
    refs=inherited_baseline(p);byid={r['input_id']:r for r in refs}
    for row in [dict(d.train[orders[0]],split='TRAIN_all'),d.evals[0]]:
        raw,decoded,_,_,_=evaluate_one(e,d,row,charge);old=byid[row['id']];stored=torch.load(old['raw']['path'],map_location='cpu',weights_only=True)
        assert torch.equal(raw['final_semantic'],stored['final_semantic']) and decoded['text']==old['output']
    original=codec.verbalize
    def forbidden(*a,**kw):raise AssertionError('D forbidden during theta latent training')
    codec.verbalize=forbidden;thought.train();e.begin()
    while e.runtime['phase']=='MICROS':e.unit(*d.unit_inputs(units[e.runtime['cursor']//2]),charge)
    e.finish();e.begin();assert e.group_size()==6
    e.unit(*d.unit_inputs(units[e.runtime['cursor']//2]),charge);mid=e.payload();midSHA=support.state_sha(mid)
    while e.runtime['phase']=='MICROS':e.unit(*d.unit_inputs(units[e.runtime['cursor']//2]),charge)
    ready=e.payload();readySHA=support.state_sha(ready)
    if owned_output_bytes(directory)+240*1024**2>512*1024**2:raise OSError('preflight outputcap')
    for name,payload in [('MID_after_complete_pair',mid),('READY_partial6',ready)]:
        with (directory/(name+'.pt')).open('xb') as f:torch.save(payload,f)
    e.finish();expected=support.state_sha(e.payload())
    mid=torch.load(directory/'MID_after_complete_pair.pt',map_location='cpu',weights_only=True)
    e.restore(mid)
    while e.runtime['phase']=='MICROS':e.unit(*d.unit_inputs(units[e.runtime['cursor']//2]),charge)
    e.finish();assert support.state_sha(e.payload())==expected and support.state_sha(mid)==midSHA
    ready=torch.load(directory/'READY_partial6.pt',map_location='cpu',weights_only=True);e.restore(ready);e.finish()
    assert support.state_sha(e.payload())==expected and support.state_sha(ready)==readySHA
    codec.verbalize=original;e.update.audit_endpoint()
    assert support.state_sha(codec.state_dict())==before and not torch.cuda.is_initialized()
    assert sum(count.values())==388 and count['readout']==2
    report=dict(fullsize_engine_restore=True,all_parent_tensors_exact=True,all_old_Adam_exact=True,CUDA_initialized=False,
        atomic_pair_MID_and_partial6_READY_exact=True,saved_checkpoint_dictionary_immutable=True,allcodec_frozen=True,
        old_baseline2actual_samples_bitwise=True,training_D0=True,train_eval_maxgap=gap,NN_calls=dict(count),
        temporary_fixture_updates_including_replays=4,final_fixture_unique_updates=2,thought_parameters=sum(x.numel() for x in thought.parameters()),
        actual_real_and_unpaired_units=True,seconds=time.monotonic()-began,
        MID_SHA=support.sha(directory/'MID_after_complete_pair.pt'),READY_SHA=support.sha(directory/'READY_partial6.pt'))
    io.atomic_json(directory/'report.json',report);print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0


def run(path,out,fixture):
    p=validate(path,out);p['_plan']=str(path);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    if fixture:return preflight(p)
    if not p['authorized'] or (out/'worker_started.json').exists():raise ValueError('ONE bounded paired trial')
    io.atomic_json(out/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
    budget=Budget(p);driver=dict(phase='loading',started=time.time(),plan_SHA=support.sha(path),eval_refs=[],checkpoints=[]);e=None;guard=False
    def progress():
        r=e.runtime if e else {}
        io.atomic_json(out/'progress.json',dict(phase=driver['phase'],attempt=r.get('attempts',0),update=r.get('updates',0),cursor=r.get('cursor',0),calls=budget.calls,
            time=time.time(),elapsed=time.time()-driver['started'],native_pid=os.getpid(),eval_rows=len(driver['eval_refs']),latest_durable=driver['checkpoints'][-1]['path'] if driver['checkpoints'] else None))
    def checkpoint(name,emergency=False):
        payload=dict(schema='answer_pair_contrast_complete_v112',engine=e.payload(),driver=dict(driver),budget=dict(budget.calls),
            frozen_codec_reference=p['selection'],source_cache=p['cache'],answer_contract=p['answer_contract'],plan_SHA=driver['plan_SHA'])
        ref=budget.save(out/'checkpoints'/(name+'.pt'),payload,emergency=emergency);driver['checkpoints'].append(ref);io.atomic_json(out/'latest.json',ref);progress()
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        budget.before();codec,epoch,before,selection=load_codec(p,'cpu');d=Data(p);seed_all(p['seed']);thought=warm_thought(p)
        e=PairContrastEngine(thought,codec,p['codec_SHA'],d.orders,times=d.times,units=d.units,batch=8,lr=p['LR'],source_metadata=d.meta)
        driver['eval_refs']=inherited_baseline(p);driver['phase']='initialized_inherited_baseline230_forward0';checkpoint('state_0000')
        driver['phase']='actual_pair_contrast_theta_ONLY_training';progress()
        while e.runtime['phase']!='DONE':
            budget.before(34);e.begin()
            while e.runtime['phase']=='MICROS':
                budget.before(34);e.unit(*d.unit_inputs(d.units[e.runtime['cursor']//2]),budget.charge)
            e.finish();progress()
            if e.runtime['attempts'] in (134,268):checkpoint('state_'+str(e.runtime['attempts']).zfill(4))
        assert e.runtime['cursor']==2144 and e.runtime['attempts']==268 and e.runtime['exposure_bytes']==37088
        assert e.runtime['calls']==dict(denoise=36448,readout=0)
        driver['phase']='final_allTRAIN134_DEV96';e.thought.eval();progress()
        rows=[dict(r,split='TRAIN_all') for r in sorted(d.train.values(),key=lambda r:r['id'])]+d.evals
        for row in rows:
            budget.before(17);raw,decoded,packet,body,tail=evaluate_one(e,d,row,budget.charge)
            ref=budget.save(out/'raw'/('final_'+row['id']+'.pt'),raw);active=packet['labels'].flatten()!=257
            driver['eval_refs'].append(dict(input_id=row['id'],split=row['split'],pair_key=row['pair_key'],phase='final',expected=row['answer'],label=row['label'],
                output=decoded['text'],EOS=decoded['EOS'],UTF8=decoded['UTF8'],**previous.score(decoded['text'],row),
                byte_NLL=float(-raw['trace']['gold_log_probability'][active].mean()),body_MSE=body,tail_MSE=tail,raw=ref));progress()
        e.update.audit_endpoint();assert support.state_sha(codec.state_dict())==before
        assert budget.calls==dict(denoise=40128,readout=230,encode=0)
        io.atomic_json(out/'assessment.json',driver['eval_refs'])
        io.atomic_json(out/'result.json',dict(status='ANSWER_PAIR_CONTRAST_TRIAL_COMPLETE',execution_device='cpu',CUDA_initialized=torch.cuda.is_initialized(),
            attempts=e.runtime['attempts'],updates=e.runtime['updates'],AMP_skips=e.runtime['AMP_skips'],cursor=2144,exposure_target_bytes=37088,
            calls=budget.calls,seconds=time.time()-driver['started'],selected_TRAIN=134,selected_DEV_for_training=0,
            frozen_codec_SHA_before_after=before,only_new_thought_optimizer=True,warm_V111_ONLYtheta=True,fresh_Adam=True,
            checkpoints=driver['checkpoints'],assessment_rows460=True,training_D0=True,full16step_gradient=True,
            baseline230_inherited_forward0=True,old_budget_reuse=False,pair_coefficient=.25,all_old_assets_preserved=True))
        driver['phase']='terminal_complete';progress();return 0
    except BaseException as error:
        incident=dict(error=repr(error),traceback=traceback.format_exc(),phase=driver['phase'],calls=budget.calls)
        if e:
            try:checkpoint('STOP_current',emergency=True)
            except BaseException as other:incident['checkpoint_error']=repr(other)
        io.atomic_json(out/'incident.json',incident);return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--plan',type=Path,required=True);q.add_argument('--output',type=Path,required=True);q.add_argument('--preflight',action='store_true')
    a=q.parse_args();raise SystemExit(run(a.plan.resolve(),a.output.resolve(),a.preflight))
