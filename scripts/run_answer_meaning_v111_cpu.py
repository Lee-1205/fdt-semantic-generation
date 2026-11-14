"""ONE finite conditional answer-meaning trial; E/memories/D completely frozen.
Only cached E(Y) targets, NOT E(X); no D forward/gradient during training.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,collections,ctypes,hashlib,json,shutil,sys,time,traceback,re
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
from research.memory_thought_diffusion_v94 import ThoughtConfig
from research.thought_grid_tail_v104 import GridTailThought
from research.runtime_aligned_thought_v107 import training_rollout
from research.latent_identity_thought_v110 import effective_velocity_rank
from research.answer_meaning_thought_v111 import AnswerMeaningEngine
from research.readout_decision_trace_v88 import pack as trace_pack,decode

SCHEMA='fdt_conditional_ANSWER_meaning_v111';OUTPUT=1024**3


def validate(path,out):
    p=io.load_json(path)
    fixed=dict(schema=SCHEMA,device='cpu',max_calls=41000,expected_calls=40256,max_attempts=268,train_rows=2144,
        unique_TRAIN=134,batch=8,epochs=16,eval_rows=112,steps=16,wall_seconds=1200,max_output_bytes=OUTPUT,seed=1111411,LR=1e-4)
    if any(p.get(k)!=v for k,v in fixed.items()) or Path(p['output']).resolve()!=out:
        raise ValueError('fixed NEW conditional answer purpose/costs')
    if p['authorized'] and (p['deadline']-p['start']!=1200 or p['cutoff']!=p['deadline']-120):raise ValueError('new1080+120clock')
    for f,digest in p['files'].items():
        if support.sha(f)!=digest:raise ValueError('pinned file '+f)
    return p


class Data:
    def __init__(self,p):
        contract=Path(p['answer_contract']);data=Path(p['data']);cache=Path(p['cache'])
        meta=io.load_json(contract/'contract.json');conditions=io.load_json(contract/'train_conditions.json');loss=io.load_json(contract/'train_targets_LOSS_ONLY.json')
        if meta['train_rows']!=134 or meta['DEV_rows_for_training'] or len(conditions)!=134 or len(loss)!=134:raise ValueError('source-selected TRAIN only')
        alltrain={r['id']:r for r in io.load_json(data/'train_targets.json')}
        ff={r['id']:r for r in io.load_json(data/'train_features.json')}
        self.train={r['id']:dict(alltrain[r['id']],split='TRAIN_probe') for r in conditions}
        self.evals=[]
        for split in ('dev_pair','dev_surface','dev_position'):
            self.evals += [dict(r,split=split) for r in io.load_json(data/(split+'_targets.json'))]
        if len(self.evals)!=96 or set(self.train)&{r['id'] for r in self.evals}:raise ValueError('all96DEV excluded from training')
        for condition,target in zip(conditions,loss):
            row=self.train[condition['id']]
            if condition['id']!=target['id'] or ff[row['id']]['prompt']==row['answer'] or target['answer_text_offline_only']!=row['answer']:
                raise ValueError('E(X) condition versus distinct E(Y) target')
            if hashlib.sha256(row['answer'].encode()).hexdigest()!=target['answer_SHA']:raise ValueError('true answer target SHA')
        cres=io.load_json(cache/'result.json');self.static=torch.load(cres['static_memories']['path'],map_location='cpu',weights_only=True)
        index={r['id']:r for rows in io.load_json(cache/'input_index.json').values() for r in rows}
        needed=set(self.train)|{r['id'] for r in self.evals}
        self.dynamic={i:torch.load(index[i]['path'],map_location='cpu',weights_only=True) for i in needed}
        target_index=io.load_json(cache/'target_index_loss_only.json')
        self.targets={r['answer_SHA']:torch.load(r['target']['path'],map_location='cpu',weights_only=True) for r in target_index}
        self.target_records={r['answer_SHA']:r for r in target_index}
        for r in self.train.values():
            packet=self.target(r)
            if packet['semantic_target'].shape!=(1,35,512) or packet['labels'].shape!=(34,16):raise ValueError('same local33/global2 geometry')
            z,mask,labels=packet['semantic_target'],packet['valid_extent_mask'],packet['labels']
            raw=r['answer'].encode();n=2+(len(raw)+15)//16
            if (z.dtype!=torch.float32 or mask.dtype!=torch.bool or mask.shape!=(1,35) or labels.dtype!=torch.long
                    or z.requires_grad or not torch.isfinite(z).all() or not mask[0,:n+1].all() or mask[0,n+1:].any()
                    or packet['source_codec_SHA']!=p['codec_SHA'] or packet['model_epoch']!=self.static['model_epoch']
                    or packet['answer_SHA']!=hashlib.sha256(raw).hexdigest() or not packet['target_condition_features_never_used']):
                raise ValueError('actual answer packet provenance/dtype/body-tail mask')
            if (labels.flatten()[:len(raw)].tolist()!=list(raw) or int(labels.flatten()[len(raw)])!=256
                    or (labels.flatten()[len(raw)+1:]!=257).any() or not torch.equal(z[:,n:],z[:,n:n+1].expand_as(z[:,n:]))):
                raise ValueError('loss-only bytes/EOS/PAD and frozen terminal tail')
        self.probe=[dict(self.train[i],split='TRAIN_probe') for i in p['probe_ids']]
        if len(self.probe)!=16 or sum(r['label']=='unknown' for r in self.probe)!=8:raise ValueError('fixed8positive8unknown probe')
        curriculum=io.load_json(p['curriculum']);selected=set(self.train)
        self.orders=[r['id'] for ep in curriculum for r in ep];self.times=[r['t'] for ep in curriculum for r in ep]
        if len(curriculum)!=16 or len(self.orders)!=2144 or any(len(ep)!=134 or {r['id'] for r in ep}!=selected for ep in curriculum):
            raise ValueError('each134row exactlyonce each16epochs')
        bypair=collections.defaultdict(list)
        for r in self.train.values():bypair[r['pair_key']].append(r['id'])
        for ep in curriculum:
            positions={r['id']:j for j,r in enumerate(ep)}
            for pair in bypair.values():
                if len(pair)==2:
                    a,b=(positions[i] for i in pair)
                    if abs(a-b)!=1 or min(a,b)%2 or ep[a]['t']!=ep[b]['t']:raise ValueError('actual reverse pairs in2rowblocks')
        for i in selected:
            if sorted(self.times[j] for j,x in enumerate(self.orders) if x==i)!=sorted(1-k/16 for k in range(16)):
                raise ValueError('eachrow all16 actualtimepoints')
        self.meta=dict(task='CONDITIONAL_NEW_ANSWER_MEANING_EY_NOT_SOURCE_IDENTITY',
            answer_contract_SHA=support.sha(contract/'contract.json'),data_manifest_SHA=support.sha(data/'manifest.json'),
            finite_orders_SHA=support.sha(p['curriculum']),warm_thought_SHA=p['warm']['SHA'],D_during_training=False)
        self.codec_SHA=p['codec_SHA']

    def source(self,identity):return assemble_input(self.static,self.dynamic[identity],'cpu')
    def target(self,row):return self.targets[hashlib.sha256(row['answer'].encode()).hexdigest()]
    def training(self,identity):
        if identity not in self.train:raise ValueError('no DEV fitting')
        return self.source(identity),self.target(self.train[identity])


def warm_thought(p):
    ref=p['warm'];full=torch.load(ref['path'],map_location='cpu',weights_only=True)
    if support.sha(ref['path'])!=ref['SHA'] or support.state_sha(full)!=ref['algorithm_SHA']:raise ValueError('complete parent preserved')
    e=full['engine']
    if e['schema']!='latent_identity_fullstate_v110' or e['codec_SHA']!=p['codec_SHA'] or e['thought_config']!=dict(ThoughtConfig(hidden=520).__dict__):
        raise ValueError('exact warm corrected architecture/codec')
    thought=GridTailThought(ThoughtConfig(hidden=520));thought.load_state_dict(e['thought'],strict=True)
    if support.state_sha(thought.state_dict())!=p['warm_thought_model_SHA']:raise ValueError('ALLwarmtheta tensors exact')
    del full,e
    return thought


class Budget:
    def __init__(self,p):self.p=p;self.out=Path(p['output']);self.calls=dict(denoise=0,readout=0,encode=0)
    def before(self,cost=0,pending=0,emergency=False):
        if not emergency and ((self.out/'STOP').exists() or time.time()>=self.p['cutoff']):raise InterruptedError('fixed cutoff/STOP')
        if sum(self.calls.values())+cost>41000:raise RuntimeError('closed41000callcap')
        if shutil.disk_usage(P).free<100*1024**3 or shutil.disk_usage(self.out).free<1024**3+pending:raise OSError('C100GiB/D1GiB+pending protected')
        if owned_output_bytes(self.out)+pending>OUTPUT:raise OSError('D1GiB wholeoutput including STOP/temp')
    def charge(self,kind):
        if kind not in ('denoise','readout') or sum(self.calls.values())>=41000:raise RuntimeError('thought/D only boundedcalls')
        self.calls[kind]+=1
    save=PreviousSafeBudget.save


def score(text,row):
    # Offline score only. No parser or labels are available to the generator.
    claims=[]
    for m in re.finditer(r'\b([A-H])\s+is\s+(larger|greater|bigger|smaller|less)\s+than\s+([A-H])\b',text):
        a,word,b=m.groups();a,b=(a,b) if word in ('larger','greater','bigger') else (b,a);claims.append((a,b))
    for m in re.finditer(r'\b([A-H])\s*([<>])\s*([A-H])\b',text):
        a,op,b=m.groups();claims.append((a,b) if op=='>' else (b,a))
    unknown='unknown' in text.casefold() or bool(re.search(r'cannot\s+(?:be\s+)?(?:determined|tell)',text,re.I))
    expected=row['answer'];g=re.search(r'([A-H]) is (larger|smaller) than ([A-H])',expected)
    gold=None if g is None else ((g[1],g[3]) if g[2]=='larger' else (g[3],g[1]))
    present=unknown if gold is None else gold in claims
    contradiction=bool(claims) if gold is None else any(c!=gold for c in claims) or unknown
    return dict(exact=text==expected,normalized_exact=' '.join(text.split()).casefold()==' '.join(expected.split()).casefold(),
        expected_relation_present=present,contradictory_extractable_fact=contradiction,
        partial_grammar_semantic_match=present and not contradiction,partial_grammar_NOT_general_semantic_judge=True)


def evaluate_one(engine,d,row,charge):
    m=d.source(row['id']);seed=int(hashlib.sha256(('V111_ANSWER17:'+row['id']).encode()).hexdigest()[:15],16)
    noise=torch.randn(1,35,512,generator=torch.Generator().manual_seed(seed))
    with torch.no_grad():
        final=engine.thought.sample(m,noise,steps=16,charge=charge);charge('readout');logits=engine.codec.verbalize(final.as_codec_commit())
        # Gold packet is read ONLY AFTER target-free semantic generation/D.
        packet=d.target(row);trace=trace_pack(logits,packet['labels'],[0,1]);decoded=decode(trace)
        error=(final.semantic-packet['semantic_target']).square().mean(-1);mask=packet['valid_extent_mask']
        body=float((error*mask).sum()/mask.sum());tail=float((error*(~mask)).sum()/(~mask).sum().clamp_min(1))
    raw=dict(schema='FINAL_new_answer_meaning_then_frozen_D_v111',input_id=row['id'],noise_seed=seed,
        final_semantic=final.semantic,trace=trace,no_target_in_inference=True,task_is_answer_NOT_source_reconstruction=True)
    return raw,decoded,packet,body,tail


def preflight(p):
    begin=time.monotonic();directory=Path(p['_plan']).parent/'cpu_preflight_owned/runtime';directory.mkdir(exist_ok=False)
    def resource(pending=0):
        if (shutil.disk_usage(P).free<100*1024**3 or shutil.disk_usage(directory).free<1024**3+pending
                or owned_output_bytes(directory)+pending>512*1024**2):raise OSError('fixed CPU-preflight C100/D1GiB/512MiB cap')
    if shutil.disk_usage(P).free<p['resources']['C_start'] or shutil.disk_usage(directory).free<8*1024**3+512*1024**2:
        raise OSError('actual preflight start resources')
    codec,epoch,before,selection=load_codec(p,'cpu');d=Data(p);seed_all(p['seed']);thought=warm_thought(p)
    engine=AnswerMeaningEngine(thought,codec,p['codec_SHA'],d.orders[:14],times=d.times[:14],batch=8,lr=p['LR'],source_metadata=d.meta)
    assert not engine.optimizer.state and effective_velocity_rank(thought)==512
    count=collections.Counter()
    def charge(kind):
        count[kind]+=1
        if sum(count.values())>400:raise RuntimeError('preflight400callcap')
    m,packet=d.training(d.orders[0]);noise=torch.randn(1,35,512,generator=torch.Generator().manual_seed(111))
    trajectory=training_rollout(thought,m,noise,16,charge);thought.eval();final=thought.sample(m,noise,16,charge)
    gap=float((trajectory.detach()-final.semantic).abs().max());assert gap<1e-4;del trajectory
    repeated_noise=torch.randn(1,35,512,generator=torch.Generator().manual_seed(111))
    repeated=thought.sample(m,repeated_noise,16,charge)
    assert torch.equal(noise,repeated_noise) and torch.equal(final.semantic,repeated.semantic)
    initial=thought.initial_state(m,noise).detach();initial_variance=float(initial.var());final_variance=float(final.semantic.var())
    assert torch.isfinite(initial).all() and torch.isfinite(final.semantic).all()
    ceiling=[]
    for key in sorted({hashlib.sha256(r['answer'].encode()).hexdigest() for r in d.train.values()}):
        pkt=d.targets[key];record=d.target_records[key];charge('readout')
        with torch.no_grad():
            logits=codec.verbalize(DifferentiableCodecCommit.from_semantic(pkt['semantic_target'],m.memory_revisions,m.model_epoch))
            decoded=decode(trace_pack(logits,pkt['labels'],[0,1]))
        ceiling.append(dict(answer_SHA=key,expected=record['expected'],output=decoded['text'],prefix=decoded['text'].startswith(record['expected']),
            exact=decoded['text']==record['expected'],EOS=decoded['EOS'],UTF8=decoded['UTF8']))
    assert len(ceiling)==27 and all(r['prefix'] and r['EOS'] and r['UTF8'] for r in ceiling)
    io.atomic_json(directory/'actual27_target_ceiling.json',ceiling)
    original=codec.verbalize
    def forbidden(*a,**kw):raise AssertionError('D forbidden during actualtrainingmicros')
    codec.verbalize=forbidden;thought.train();engine.begin();micro_begin=time.monotonic()
    while engine.runtime['phase']=='MICROS':
        resource()
        identity=engine.orders[engine.runtime['cursor']];engine.micro(identity,*d.training(identity),charge)
    engine.finish();engine.begin();assert engine.group_size()==6
    while engine.runtime['phase']=='MICROS':
        resource()
        identity=engine.orders[engine.runtime['cursor']];engine.micro(identity,*d.training(identity),charge)
    micro_seconds=time.monotonic()-micro_begin;ready=engine.payload();f=directory/'populated_READY_fullstate.pt'
    resource(180*1024**2)
    with f.open('xb') as stream:torch.save(ready,stream)
    saved=torch.load(f,map_location='cpu',weights_only=True);saved_hash=support.state_sha(saved)
    engine.finish();expected=engine.payload();engine.restore(saved);engine.finish();actual=engine.payload()
    assert support.state_sha(expected)==support.state_sha(actual) and support.state_sha(saved)==saved_hash
    codec.verbalize=original;thought.eval();raw,decoded,_,_,_=evaluate_one(engine,d,d.probe[0],charge)
    with (directory/'answer_eval.pt').open('xb') as stream:torch.save(raw,stream)
    engine.update.audit_endpoint();assert support.state_sha(codec.state_dict())==before and not torch.cuda.is_initialized()
    assert engine.runtime['calls']==dict(denoise=14*17,readout=0) and engine.runtime['phase']=='DONE'
    report=dict(fullsize_engine_restore=True,all_parent_tensors_exact=True,all_old_Adam_exact=True,CUDA_initialized=False,
        warm_ONLY_thought_exact=True,fresh_Adam_initially_empty=True,thought_parameters=sum(v.numel() for v in thought.parameters()),
        allcodec_parameters_buffers_frozen=True,only_thought_optimizer=True,actual16step_training_vs_eval_maxgap=gap,
        actual_fullsize_READY_Adam_scaler_RNG_nextupdate_exact=True,last_partial6micros_exact=True,
        saved_READY_SHA=support.sha(f),saved_READY_algorithm_SHA=saved_hash,temporary_optimizer_updates_including_replay=3,
        training_micros14_seconds=micro_seconds,total_seconds=time.monotonic()-begin,NN_calls=dict(count),
        source_encoder_calls0=True,D_calls_during_training=0,effective_velocity_rank=effective_velocity_rank(thought),
        target_is_answer_NOT_source=True,language_loss_NOT_used=True,actual27_target_meaning_prefixes_correct=True,
        target_full_exact=sum(r['exact'] for r in ceiling),target_tail_errors_NOT_gate=True,
        same_seed_actual_sampling_bitwise=True,initial_latent_variance=initial_variance,final_latent_variance=final_variance,
        actual_answer_packet_dtypes_masks_terminal_provenance_verified=True)
    io.atomic_json(directory/'actual_CPU_report.json',report);print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0


def run(path,out,fixture):
    p=validate(path,out);p['_plan']=str(path);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    if fixture:return preflight(p)
    if not p['authorized'] or (out/'worker_started.json').exists():raise ValueError('ONE new sealed answertrial')
    io.atomic_json(out/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
    budget=Budget(p);driver=dict(phase='loading',started=time.time(),plan_SHA=support.sha(path),eval_refs=[],checkpoints=[]);engine=None;guard=False
    def progress():
        r=engine.runtime if engine else {}
        io.atomic_json(out/'progress.json',dict(phase=driver['phase'],attempt=r.get('attempts',0),update=r.get('updates',0),cursor=r.get('cursor',0),calls=budget.calls,
            time=time.time(),elapsed=time.time()-driver['started'],native_pid=os.getpid(),eval_rows=len(driver['eval_refs']),latest_durable=driver['checkpoints'][-1]['path'] if driver['checkpoints'] else None))
    def checkpoint(name,emergency=False):
        payload=dict(schema='conditional_answer_complete_fullstate_v111',engine=engine.payload(),driver=dict(driver),budget=dict(budget.calls),
            frozen_codec_reference=p['selection'],source_cache=p['cache'],answer_contract=p['answer_contract'],plan_SHA=driver['plan_SHA'])
        ref=budget.save(out/'checkpoints'/(name+'.pt'),payload,emergency=emergency);driver['checkpoints'].append(ref);io.atomic_json(out/'latest.json',ref);progress();return ref
    def assess(d,rows,phase):
        driver['phase']=phase;engine.thought.eval();progress()
        for row in rows:
            budget.before(17);raw,decoded,packet,body,tail=evaluate_one(engine,d,row,budget.charge);ref=budget.save(out/'raw'/(phase+'_'+row['id']+'.pt'),raw)
            active=packet['labels'].flatten()!=257
            driver['eval_refs'].append(dict(input_id=row['id'],split=row['split'],phase=phase,expected=row['answer'],label=row['label'],
                output=decoded['text'],EOS=decoded['EOS'],UTF8=decoded['UTF8'],**score(decoded['text'],row),
                byte_NLL=float(-raw['trace']['gold_log_probability'][active].mean()),body_MSE=body,tail_MSE=tail,raw=ref));progress()
        engine.thought.train()
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        budget.before();codec,epoch,before,selection=load_codec(p,'cpu');d=Data(p);seed_all(p['seed']);thought=warm_thought(p)
        engine=AnswerMeaningEngine(thought,codec,p['codec_SHA'],d.orders,times=d.times,batch=8,lr=p['LR'],source_metadata=d.meta)
        driver['phase']='initialized';checkpoint('state_0000');assess(d,d.probe+d.evals,'baseline')
        driver['phase']='conditional_NEW_ANSWER_meaning_thought_ONLY_training';progress()
        while engine.runtime['phase']!='DONE':
            budget.before(18);engine.begin()
            while engine.runtime['phase']=='MICROS':
                budget.before(18);identity=engine.orders[engine.runtime['cursor']];engine.micro(identity,*d.training(identity),budget.charge)
            engine.finish();progress()
            if engine.runtime['attempts'] in (134,268):checkpoint('state_'+str(engine.runtime['attempts']).zfill(4))
        assert engine.runtime['cursor']==2144 and engine.runtime['attempts']==268 and engine.runtime['exposure_bytes']==37088
        assert engine.runtime['calls']==dict(denoise=36448,readout=0)
        assess(d,d.probe+d.evals,'final');engine.update.audit_endpoint()
        assert budget.calls==dict(denoise=40032,readout=224,encode=0) and support.state_sha(codec.state_dict())==before
        io.atomic_json(out/'assessment.json',driver['eval_refs'])
        io.atomic_json(out/'result.json',dict(status='CONDITIONAL_NEW_ANSWER_MEANING_TRIAL_COMPLETE',execution_device='cpu',CUDA_initialized=torch.cuda.is_initialized(),
            attempts=268,updates=engine.runtime['updates'],AMP_skips=engine.runtime['AMP_skips'],cursor=2144,exposure_target_bytes=37088,
            calls=budget.calls,seconds=time.time()-driver['started'],selected_TRAIN=134,selected_DEV_for_training=0,source_encode0=True,
            allcodec_parameters_buffers_frozen=True,frozen_codec_SHA_before_after=before,only_new_thought_optimizer=True,
            warm_V110_thought_ONLY=True,fresh_Adam=True,checkpoints=driver['checkpoints'],assessment_rows224=True,full16step_training_gradient=True,
            target_is_EY_NOT_source_EX=True,D_forward_during_training=0,language_loss_NOT_used=True,hidden520=True,all_oldbudgets_closed=True,no_promotion_or_automatic_repeat=True))
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
