"""ONE finite first memory-thought trial; ONLY new thought weights train.

V88 entire codec frozen, V95 source-only memory caches, V96 byte+EOS alignment,
V98 exact micro/Adam/scaler/RNG state. No old trial or old clock continuation.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,dataclasses,difflib,hashlib,io as memoryio,json,random,re,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from owned_output_size_v52 import owned_output_bytes
from run_thought_cache_v95 import load_codec
from research.thought_cache_v95 import assemble_input
from research.memory_thought_diffusion_v94 import ThoughtConfig,MemoryThoughtDiffusion
from research.thought_train_engine_v98 import ThoughtTrainEngine
from research.readout_decision_trace_v88 import pack as trace_pack,decode

SCHEMA='fdt_first_memory_thought_v100';OUTPUT=256*1024**2;START=108*1024**3+OUTPUT


def validate(path,out):
    p=io.load_json(path)
    fixed=dict(schema=SCHEMA,max_calls=6800,expected_calls=6608,max_attempts=192,train_rows=1536,batch=8,epochs=4,
        eval_rows=96,control_rows=16,steps=16,wall_seconds=600,max_output_bytes=OUTPUT,start_free_bytes=START,seed=11114,LR=1e-4)
    if any(p.get(k)!=v for k,v in fixed.items()) or Path(p['output']).resolve()!=out:raise ValueError('fixed NEW thought-only scope')
    if p['authorized'] and (p['deadline']-p['start']!=600 or p['cutoff']!=p['deadline']-120):raise ValueError('fixed480+120clock')
    for f,sha in p['files'].items():
        if support.sha(f)!=sha:raise ValueError('input pin '+f)
    return p


class Data:
    def __init__(self,p):
        cache=Path(p['cache']);data=Path(p['data']);res=io.load_json(cache/'result.json')
        if res['source_codec_SHA']!=p['codec_SHA'] or res['calls']!=dict(encode=511,readout=31,denoise=0):raise ValueError('completed pinned frozen cache')
        self.static=torch.load(res['static_memories']['path'],map_location='cpu',weights_only=True)
        self.conditions={r['id']:torch.load(r['path'],map_location='cpu',weights_only=True) for rows in io.load_json(cache/'input_index.json').values() for r in rows}
        self.targets={r['answer_SHA']:torch.load(r['target']['path'],map_location='cpu',weights_only=True) for r in io.load_json(cache/'target_index_loss_only.json')}
        self.train={x['id']:x for x in io.load_json(data/'train_targets.json')};self.evals=[]
        for split in ('dev_pair','dev_surface','dev_position'):
            self.evals += [dict(x,split=split) for x in io.load_json(data/(split+'_targets.json'))]
        pairs=io.load_json(data/'four_finite_orders.json');self.orders=[x for epoch in pairs for pair in epoch for x in pair]
        if len(self.orders)!=1536 or len(self.train)!=384 or len(self.evals)!=96 or any(i not in self.train for i in self.orders):raise ValueError('TRAIN-only four finite orders')
        for epoch in pairs:
            flat=[x for pair in epoch for x in pair]
            if len(flat)!=384 or set(flat)!=set(self.train):raise ValueError('one occurrence per epoch, no rewind')
        self.meta=dict(data_manifest_SHA=support.sha(data/'manifest.json'),cache_result_SHA=support.sha(cache/'result.json'),
            cache_input_index_SHA=support.sha(cache/'input_index.json'),codec_SHA=p['codec_SHA'])

    def source(self,identity,device):return assemble_input(self.static,self.conditions[identity],device)
    def target(self,row):return self.targets[hashlib.sha256(row['answer'].encode()).hexdigest()]
    def training(self,identity,device):
        if identity not in self.train:raise ValueError('no DEVtarget fitting')
        return self.source(identity,device),self.target(self.train[identity])


class Budget:
    def __init__(self,p):self.p=p;self.out=Path(p['output']);self.calls=dict(denoise=0,readout=0,encode=0)
    def before(self,cost=0,pending=0,emergency=False):
        if not emergency and ((self.out/'STOP').exists() or time.time()>=self.p['cutoff']):raise InterruptedError('fixed cutoff/STOP')
        if sum(self.calls.values())+cost>self.p['max_calls']:raise RuntimeError('closed6800callcap')
        if shutil.disk_usage(ROOT).free<100*1024**3+pending:raise OSError('100GiB plus pending write protected')
        if owned_output_bytes(self.out)+pending>OUTPUT:raise OSError('whole256MiB output incl emergency/temp')
    def charge(self,kind):
        if kind not in ('denoise','readout') or sum(self.calls.values())>=self.p['max_calls']:raise RuntimeError('thought/D callcap; encode forbidden')
        # Complete the reserved micro/16-step sample atomically. Deadline/disk
        # checked between rows; OS owner remains independent throughout.
        self.calls[kind]+=1
    def save(self,path,payload,*,emergency=False):
        stream=memoryio.BytesIO();torch.save(payload,stream);raw=stream.getvalue();self.before(pending=len(raw),emergency=emergency)
        path.parent.mkdir(exist_ok=True,parents=True)
        if path.exists():raise FileExistsError('preserved state overwrite forbidden')
        temp=path.with_suffix(path.suffix+'.tmp')
        with temp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        io.replace_with_retry(temp,path)
        loaded=torch.load(path,map_location='cpu',weights_only=True);digest=support.state_sha(payload)
        if support.state_sha(loaded)!=digest:raise ValueError('safe-loaded complete payload exact')
        return dict(path=str(path),SHA=hashlib.sha256(raw).hexdigest(),bytes=len(raw),algorithm_SHA=digest)


def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def score(text,expected):
    # Score-only grammar, never an inference feature or output assembler.
    normalized=lambda s:' '.join(s.split()).casefold()
    facts=re.findall(r'\b([A-H]) is (larger|smaller) than ([A-H])\.',text)
    gold=re.findall(r'\b([A-H]) is (larger|smaller) than ([A-H])\.',expected)
    present=bool(gold and gold[0] in facts) or (not gold and 'unknown' in text.casefold())
    contradiction=bool(gold and any(f!=gold[0] for f in facts)) or bool(gold and 'unknown' in text.casefold()) or bool(not gold and facts)
    return dict(exact=text==expected,normalized_exact=normalized(text)==normalized(expected),expected_relation_present=present,
        contradictory_extractable_fact=contradiction,lexical_similarity=difflib.SequenceMatcher(None,text,expected).ratio(),
        expected_sentence_prefix=text.startswith(expected))


def preflight(p,out):
    directory=Path(p['_plan']).parent/'cpu_preflight_owned/runtime';directory.mkdir(exist_ok=False)
    codec,epoch,before,selection=load_codec(p,'cpu');d=Data(p);seed_all(p['seed']);thought=MemoryThoughtDiffusion()
    engine=ThoughtTrainEngine(thought,codec,p['codec_SHA'],d.orders[:3],batch=2,lr=p['LR'],source_metadata=d.meta)
    engine.begin()
    for identity in engine.orders[:2]:engine.micro(identity,*d.training(identity,'cpu'))
    engine.finish();engine.begin();engine.micro(engine.orders[2],*d.training(engine.orders[2],'cpu'))
    ready=engine.payload();path=directory/'populated_READY_fullstate.pt'
    with path.open('xb') as f:torch.save(ready,f)
    restored=torch.load(path,map_location='cpu',weights_only=True)
    if support.state_sha(ready)!=support.state_sha(restored):raise ValueError('actual full thought Adam/scaler/grads/RNG safeload')
    engine.finish();expected=engine.payload();engine.restore(restored);engine.finish();actual=engine.payload()
    if support.state_sha(expected)!=support.state_sha(actual):raise ValueError('actual READY next Adam/scaler/RNG exact')
    if support.state_sha(codec.state_dict())!=before or any(x.grad is not None or x.requires_grad for x in codec.parameters()):raise ValueError('all oldcodec weights frozen')
    report=dict(fullsize_engine_restore=True,all_parent_tensors_exact=True,all_old_Adam_exact=True,CUDA_initialized=torch.cuda.is_initialized(),
        actual_encoder_forwards=0,actual_thought_CPU_forwards=3,actual_readout_CPU_forwards=3,actual_temporary_CPU_thought_updates=3,
        all_old_codec_parameters_buffers_unchanged=True,only_new_thought_optimizer=True,actual_populated_Adam_scaler_RNG_READY_exact=True,
        saved_READY_SHA=support.sha(path),saved_READY_algorithm_SHA=support.state_sha(restored),actual_thought_parameters=sum(x.numel() for x in thought.parameters()),
        fixture_not_GPU_trial=True,TRAIN_only_targets=True,source_conditions_no_answer=True)
    io.atomic_json(directory/'root_actual_CPU_report.json',report);print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0


def run(path,out,is_preflight):
    p=validate(path,out);p['_plan']=str(path);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if is_preflight:return preflight(p,out)
    if not p['authorized'] or (out/'worker_started.json').exists():raise ValueError('ONE fresh authorized trial')
    io.atomic_json(out/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
    budget=Budget(p);driver=dict(phase='loading',eval_refs=[],checkpoints=[],plan_SHA=support.sha(path),started=time.time());guard=False;engine=None
    def progress():
        r=engine.runtime if engine else {}
        io.atomic_json(out/'progress.json',dict(phase=driver['phase'],attempt=r.get('attempts',0),update=r.get('updates',0),cursor=r.get('cursor',0),
            eval_rows=len(driver['eval_refs']),calls=budget.calls,time=time.time(),elapsed=time.time()-driver['started'],native_pid=os.getpid(),
            latest_durable=driver['checkpoints'][-1]['path'] if driver['checkpoints'] else None))
    def checkpoint(name,emergency=False):
        payload=dict(schema='memory_thought_trial_fullstate_v100',engine=engine.payload(),driver=dict(driver),budget=dict(budget.calls),
            frozen_codec_reference=p['selection'],source_cache_reference=p['cache'],plan_SHA=driver['plan_SHA'])
        ref=budget.save(out/'checkpoints'/(name+'.pt'),payload,emergency=emergency);driver['checkpoints'].append(ref)
        io.atomic_json(out/'latest.json',ref);progress();return ref
    def assessment(d,rows,phase,control=None):
        driver['phase']=phase;engine.thought.eval();progress()
        for row in rows:
            budget.before(17);m=d.source(row['id'],'cuda')
            if control=='zero_KC':m=dataclasses.replace(m,contextual_long=torch.zeros_like(m.contextual_long),contextual_recent=torch.zeros_like(m.contextual_recent))
            if control=='zero_KP':m=dataclasses.replace(m,prior=torch.zeros_like(m.prior))
            noise_seed=int(hashlib.sha256(('V100_fixed17:'+row['id']).encode()).hexdigest()[:15],16)
            gen=torch.Generator(device='cuda').manual_seed(noise_seed);noise=torch.randn(1,35,512,generator=gen,device='cuda')
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):meaning=engine.thought.sample(m,noise,steps=16,charge=budget.charge)
            # Only AFTER completed FinalMeaning, with all original D weights frozen.
            engine.update.assert_fast();budget.charge('readout')
            with torch.no_grad():logits=engine.codec.verbalize(meaning.as_codec_commit())
            labels=d.target(row)['labels'].to('cuda');trace=trace_pack(logits,labels,[0,1]);decoded=decode(trace)
            raw=dict(schema='FINAL_meaning_then_frozen_language_v100',final_semantic=meaning.semantic.cpu(),trace=trace,
                input_id=row['id'],control=control,noise_seed=noise_seed,no_target_condition=True,denoise_steps16=True)
            ref=budget.save(out/'raw'/(phase+'_'+row['id']+'.pt'),raw)
            active=labels.cpu().flatten()!=257
            driver['eval_refs'].append(dict(input_id=row['id'],split=row['split'],phase=phase,control=control,expected=row['answer'],label=row['label'],
                output=decoded['text'],EOS=decoded['EOS'],UTF8=decoded['UTF8'],byte_NLL=float(-trace['gold_log_probability'][active].mean()),
                **score(decoded['text'],row['answer']),raw=ref));progress()
        engine.thought.train()
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        budget.before();codec,epoch,before,selection=load_codec(p,'cuda');d=Data(p);seed_all(p['seed']);thought=MemoryThoughtDiffusion().to('cuda')
        engine=ThoughtTrainEngine(thought,codec,p['codec_SHA'],d.orders,batch=8,lr=p['LR'],source_metadata=d.meta)
        driver['phase']='initialized';checkpoint('state_0000')
        assessment(d,d.evals,'baseline')
        driver['phase']='finite_thought_ONLY_training';progress()
        while engine.runtime['phase']!='DONE':
            budget.before(2);engine.begin()
            while engine.runtime['phase']=='MICROS':
                budget.before(2);identity=engine.orders[engine.runtime['cursor']];engine.micro(identity,*d.training(identity,'cuda'),budget.charge)
            engine.finish();progress()
            if engine.runtime['attempts'] in (96,192):checkpoint('state_'+str(engine.runtime['attempts']).zfill(4))
        if engine.runtime['cursor']!=1536 or engine.runtime['attempts']!=192:raise ValueError('finite epoch/attempt exposure')
        assessment(d,d.evals,'final')
        controls=sorted((x for x in d.evals if x['split']=='dev_pair'),key=lambda x:x['id'])[:8]
        assessment(d,controls,'final_zero_KC','zero_KC');assessment(d,controls,'final_zero_KP','zero_KP')
        if budget.calls!=dict(denoise=4864,readout=1744,encode=0):raise ValueError('actual6608cost mismatch')
        engine.update.audit_endpoint()
        if support.state_sha(codec.state_dict())!=before or support.sha(selection['selected_runtime'])!=selection['selected_SHA']:raise ValueError('frozen entire endpoint changed')
        io.atomic_json(out/'assessment.json',driver['eval_refs'])
        io.atomic_json(out/'result.json',dict(status='FIRST_MEMORY_THOUGHT_TRIAL_COMPLETE_NOT_GENERAL_REASONING',attempts=engine.runtime['attempts'],updates=engine.runtime['updates'],
            AMP_skips=engine.runtime['AMP_skips'],cursor=engine.runtime['cursor'],exposure_target_bytes=engine.runtime['exposure_bytes'],calls=budget.calls,
            seconds=time.time()-driver['started'],frozen_codec_SHA_before_after=before,source_codec_file_SHA=p['codec_SHA'],all_input_memory_output_weights_buffers_unchanged=True,
            only_new_thought_optimizer=True,checkpoints=driver['checkpoints'],assessment_rows208=len(driver['eval_refs'])==208,encode_calls0=True,
            four_epochs_not_newfacts=True,all_oldbudgets_closed=True,no_automatic_promotion_or_retry=True))
        driver['phase']='terminal_complete';progress();return 0
    except BaseException as e:
        incident=dict(error=repr(e),traceback=traceback.format_exc(),calls=budget.calls,phase=driver['phase'])
        if engine:
            try:incident['emergency']=checkpoint('STOP_current',emergency=True)
            except BaseException as save_error:incident['emergency_error']=repr(save_error)
        io.atomic_json(out/'incident.json',incident);return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--preflight',action='store_true');x=a.parse_args()
    raise SystemExit(run(x.plan.resolve(),x.output.resolve(),x.preflight))
