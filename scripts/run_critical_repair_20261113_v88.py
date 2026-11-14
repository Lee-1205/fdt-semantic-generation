"""ONE cause-directed FP32/critical-coverage trial; all old budgets closed."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,ctypes,gc,hashlib,json,random,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64,EXACT_POLICY
from research.fp32_ordered_readout_v86 import FP32OrderedFuzzyReadout
from research.readout_precision_policy_v81 import POLICY
from research.readout_decision_trace_v88 import pack as trace_pack,decode as trace_decode
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.lossless_raw_tensor_v80 import unpack as raw_unpack
from research import linked_runtime_v75 as source_storage
from research import linked_lastlinear_runtime_v79 as linked
from research.train_margin_retention_v81 import margins
from compact_critical_engine_v87 import CompactCriticalEngine
from saved_commit_reference_v76_v2 import resolve_saved_commit
import learning_io_v2 as io
import text_generation_support_v1 as support

SCHEMA='fdt_cause_directed_critical_repair_v88';OUTPUT=192*1024**2;START=108*1024**3+OUTPUT
PARENT_SHA='588ceaad4a94dde38ae56e42239497534d69387814a4fa66a4fbcea974db3cd2'
ALGORITHM='1c67b342a0df81aa993f97043f50b57d1d057858ed70c0750f3dfa1bb11b727c'
STORAGE_SHA='734a2eed6526f86526f203d6389f6b1a778b50f9273ae9aaf90d4c801998dab7'
CAPS=dict(initial=13*1024**2,teacher_complete=13*1024**2,model=32*1024**2,emergency=32*1024**2,temporary=32*1024**2,
    teacher_targets=24*1024**2,TRAIN_reference=23*1024**2,assessment=22*1024**2,other=1*1024**2)

def validate(path,out):
    p=io.load_json(path);fixed=dict(schema=SCHEMA,attempts=192,teacher_epochs=1,group_size=5,wall_seconds=900,max_calls=5100,expected_max_calls=5062,
        max_output_bytes=OUTPUT,start_free_bytes=START,precision_policy=POLICY,no_accepted_window=32,source_SHA=PARENT_SHA,source_algorithm_SHA=ALGORITHM,storage_parent_SHA=STORAGE_SHA)
    if any(p.get(k)!=v for k,v in fixed.items()) or Path(p['output']).resolve()!=out or p['component_caps']!=CAPS:raise ValueError('fixed actual repair schema/caps')
    if sum(CAPS.values())!=OUTPUT or p['reference_patch_work_cap']!=8192*(4*192+2):raise ValueError('full output/reference budget')
    if p['authorized'] and (p['deadline']-p['start']!=900 or p['cutoff']!=p['deadline']-120):raise ValueError('new fixed clock')
    for f,s in p['files'].items():
        if support.sha(f)!=s:raise ValueError('input pin '+f)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core')
    rows=io.load_json(Path(p['data'])/'train_features.json');orders=io.load_json(Path(p['data'])/'orders.json')[:192]
    if len(rows)!=1024 or len(orders)!=192 or any(len(o)!=5 for o in orders):raise ValueError('new finite1024teacher/192groups not whole128oldtrial')
    if len(io.load_json(p['eval_inputs']))!=194:raise ValueError('same192short+2long retained panel')
    return p,rows,orders

def size(folder):return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file()) if folder.exists() else 0

class Budget:
    def __init__(self,p,preflight):
        self.p=p;self.output=Path(p['output']);self.preflight=preflight;self.began=time.monotonic();self.calls=dict(encode=0,readout=0,reference_linear=0,denoise=0);self.block_work=0
        self.data_sha=support.sha(Path(p['data'])/'manifest.json');self.parent_SHA=PARENT_SHA;self.parent_reference=dict(source=p['source'],source_SHA=PARENT_SHA,
            old2Adam_scaler_ALLRNG='V80 preserved external; fresh last2Adam under changed FP32/guard',storage_parent=p['storage_parent'],storage_parent_SHA=STORAGE_SHA)
        self.anchor_indices=io.load_json(p['anchor_selection'])['indices'];self.critical_targets={r['id']:r['critical_spans'] for r in io.load_json(Path(p['data'])/'train_targets.json')}
        self.frozen_symbol_profile=io.load_json(p['symbol_profile'])
    def check(self,write_bytes=0,emergency=False):
        if not emergency:
            if self.preflight:
                if time.monotonic()-self.began>75:raise TimeoutError('CPU75sec work, owner90+30 hard')
            elif time.time()>=self.p['cutoff'] or (self.output/'STOP').exists():raise InterruptedError('new model cutoff/cooperative STOP')
        if shutil.disk_usage(ROOT).free<100*1024**3+write_bytes:raise OSError('protected100GiB and pending write')
        cap=64*1024**2 if self.preflight else OUTPUT
        if size(self.output)+write_bytes>cap:raise OSError('fixed whole output bound')
        if not self.preflight:
            for name in ('initial','teacher_complete','model','emergency','teacher_targets','TRAIN_reference','assessment'):
                if size(self.output/name)>CAPS[name]:raise OSError('fixed component cap '+name)
    def charge(self,kind):
        self.check()
        if kind not in self.calls or kind=='denoise' or sum(self.calls.values())>=(16 if self.preflight else 5100):raise ValueError('actual call cap/type')
        self.calls[kind]+=1

def initialize(p,rows,orders,budget,sha,device):
    source=source_storage.load_verified(p['source'],PARENT_SHA,ROOT,support.state_sha)
    if support.state_sha(source)!=ALGORITHM or source['updates']!=247 or len(source['optimizer']['state'])!=2:raise ValueError('actual V80 immutable source fullstate')
    storage=torch.load(p['storage_parent'],map_location='cpu',weights_only=True)
    if support.sha(p['storage_parent'])!=STORAGE_SHA or len(source['model'])!=len(storage['model']):raise ValueError('standalone storage parent')
    torch.manual_seed(88113);random.seed(88113);np.random.seed(88113)
    if device=='cuda':torch.cuda.manual_seed_all(88113)
    encoder=FuzzyCodecV64(Config(**source['core']),input_capacity=16384);encoder.enable_position_adapter();encoder.enable_long_capacity();encoder.enable_local_banks()
    model=FuzzyRoundTripV64(encoder,FP32OrderedFuzzyReadout());model.load_state_dict(source['model'],strict=True)
    if support.state_sha(model.state_dict())!=support.state_sha(source['model']):raise ValueError('all warm tensors including unused branches exact')
    engine=CompactCriticalEngine(model,rows,orders,budget,sha,device)
    if engine.embedding_sha!=source['embedding_sha'] or len(engine.names)!=2 or sum(dict(model.named_parameters())[n].numel() for n in engine.names)!=2117664 or engine.optimizer.state:raise ValueError('fresh2Adam/frozen upstream scope')
    if not torch.equal(engine.symbol_weights.cpu(),source['symbol_weights']):raise ValueError('source symbol weights preserved')
    engine.storage_parent_model=storage['model'];engine.storage_parent_reference=dict(path=p['storage_parent'],SHA=STORAGE_SHA,model_key='model',model_SHA=support.state_sha(storage['model']))
    return engine,dict(all_parent_tensors_exact=True,all_old_Adam_exact=True,all_source_Adam_scaler_ALLRNG_preserved_external=True,source_model_SHA=support.state_sha(source['model']),
        fresh2Adam=True,only2_optimizer_parameters=2117664,seven_stored_overrides=True,all_codec_and_fiveprefix_frozen=True)

def save(engine,folder,emergency=False):
    cap=32*1024**2 if emergency or folder=='model' else 13*1024**2
    engine.budget.check(write_bytes=cap,emergency=emergency);directory=engine.budget.output/folder;directory.mkdir(exist_ok=False)
    envelope=linked.pack(engine.export(),engine.storage_parent_model,engine.storage_parent_reference,support.state_sha)
    meta=linked.save_verified(directory/'runtime.pt',envelope,ROOT,support.state_sha)
    if meta['bytes']>cap:raise OSError('fixed single checkpoint bound')
    io.atomic_json(directory/'runtime.json',meta);return meta

@torch.no_grad()
def final_reference(engine):
    engine.budget.charge('reference_linear');d=engine._device_reference
    if d is None:raise ValueError('initialized reference device required')
    engine.reference_patch_work+=len(d.hidden)
    with torch.autocast(engine.device,dtype=torch.bfloat16,enabled=False):z=engine.model.readout.readout.net[2](d.hidden).reshape(-1,16,258)
    decision=d.decision(z)
    if not decision.accept:raise ValueError('terminal TRAIN reference no longer preserved')
    packet=dict(argmax=z.argmax(-1).to(torch.int16).cpu(),margin=margins(z,d.c['labels']).cpu(),reference=copy.deepcopy(engine.bank_ref),decision=decision.measurements)
    f=engine.budget.output/'TRAIN_reference/final_predictions.pt';engine.budget.check(write_bytes=2*1024**2)
    with f.open('xb') as stream:torch.save(packet,stream)
    engine.reference_final=dict(path=str(f),SHA=support.sha(f),decision=decision.measurements)

@torch.no_grad()
def assess(engine,items,progress,label='final'):
    p=engine.budget.p;directory=engine.budget.output/'assessment'/label;directory.mkdir(exist_ok=False,parents=True);records=[];engine.model.eval()
    for i,info in enumerate(items):
        engine.budget.check();packet=raw_unpack(torch.load(info['raw'],map_location='cpu',weights_only=True));row=info['row']
        tensor,_=resolve_saved_commit(packet,row,engine.embedding_sha,p['files'],ROOT,512)
        commit=DifferentiableCodecCommit.from_semantic(tensor.to(engine.device),((row['id'],'fixed-cached-upstream',packet['source_SHA']),),engine.embedding_sha)
        engine.budget.charge('readout');z=engine.model.verbalize(commit)
        samples=info['sample_patches'] if label=='final' else ([0] if info['index']==0 else [])
        trace=trace_pack(z,packet['targets'].to(engine.device),samples);decoded=trace_decode(trace)
        trace.update(source_id=row['id'],source_SHA=packet['source_SHA'],original_raw_reference=info['raw'],embedding_SHA=engine.embedding_sha,
            Exact_entries=0,precision_policy=POLICY,decoded_ids=decoded['ids'])
        file=directory/f'trace_{i:03d}.pt'
        with file.open('xb') as stream:torch.save(trace,stream)
        labels=packet['targets'].flatten();valid=labels!=257;correct=int(((trace['argmax'].long()==labels)&valid).sum())
        records.append(dict(id=row['id'],panel=info['panel'],index=info['index'],output_storage='trace.decoded_ids, independently UTF8-replayable',exact_auxiliary=decoded['text']==row['prompt'],EOS=decoded['EOS'],UTF8=decoded['UTF8'],source_bytes=len(row['prompt'].encode()),symbols=int(valid.sum()),
            correct=correct,NLL_sum=float(-trace['gold_log_probability'][valid].double().sum()),trace=str(file),SHA=support.sha(file),full258_logits_saved=False))
        del z,tensor,commit,packet,trace,decoded;gc.collect()
        if i%8==0 or info['panel']=='long':progress('cached_FP32_assessment')
    io.atomic_json(directory/'reconstructions.json',records)
    for panel in sorted({x['panel'] for x in items}):
        selected=[r for r in records if r['panel']==panel];n=sum(v['symbols'] for v in selected)
        engine.evaluations[panel if label=='final' else 'baseline_'+panel]=dict(rows=len(selected),symbols=n,byte_EOS_accuracy=sum(v['correct'] for v in selected)/n,NLL=sum(v['NLL_sum'] for v in selected)/n,
            UTF8_EOS=sum(v['UTF8'] and v['EOS'] for v in selected),exact_auxiliary=sum(v['exact_auxiliary'] for v in selected))

def run(path,out,preflight=False):
    p,rows,orders=validate(path,out);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    budget=Budget(p,preflight)
    if preflight:budget.output=path.parent/'cpu_preflight_owned/runtime';budget.output.mkdir(exist_ok=False)
    engine,restore_proof=initialize(p,rows,orders,budget,support.sha(path),'cpu' if preflight else 'cuda')
    if preflight:
        initial=save(engine,'initial');engine.restore(linked.load_verified(initial['path'],initial['SHA'],ROOT,support.state_sha));before=support.state_sha(engine.export())
        for _ in range(3):engine.cache_next()
        c=engine.extract_live(rows[0]);budget.charge('readout');z=engine.model.verbalize(c);t=engine.packet(0)
        from research.competing_margin_objective_v78 import decoder_losses
        loss,_=decoder_losses(z,t['labels'],engine.symbol_weights,t['critical_mask'],t['teacher_gold_probability'],t['teacher_correct'],t['teacher_margin']);loss.backward();named=dict(engine.model.named_parameters())
        if any(named[n].grad is None for n in engine.names) or any(v.grad is not None for n,v in named.items() if n not in engine.names):raise ValueError('actual fullsize only2gradient')
        engine.optimizer.zero_grad(set_to_none=True);engine.verify_frozen();partial=save(engine,'teacher_complete');s=linked.load_verified(partial['path'],partial['SHA'],ROOT,support.state_sha);engine.restore(s)
        if support.state_sha(engine.export())!=partial['reconstructed_fullstate_SHA'] or torch.cuda.is_initialized():raise ValueError('actual fullsize partial-teacher exact restore')
        report=dict(restore_proof,fullsize_engine_restore=True,CUDA_initialized=False,actual_encoder_forwards=4,actual_readout_forwards=4,actual_only2_gradient=True,optimizer_steps=0,
            actual_fullsize_initial_and_partial_restore=True,fullFP32_readout_dtype=str(z.dtype),enabled_CPU_scaler=engine.scaler.is_enabled(),real_fullsize_reference_bank_NOT_built=True,partial_teacher_rows=3,
            initial_bytes=initial['bytes'],partial_bytes=partial['bytes'])
        print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (out/'worker_started.json').exists():raise ValueError('ONE new execution')
    started=time.time();io.atomic_json(out/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=started,plan_SHA=support.sha(path)));guard=False;terminal='CRITICAL_REPAIR_V88_COMPLETE'
    def progress(phase):
        engine.phase=phase;io.atomic_json(out/'progress.json',dict(phase=phase,attempt=engine.attempts,updates=engine.updates,rejections=engine.rejections,teacher_cursor=engine.cache_cursor,
            calls=budget.calls,native_pid=os.getpid(),parent_pid=os.getppid(),time=time.time(),elapsed=time.time()-started,latest_durable=latest))
    latest=None
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        latest=save(engine,'initial');progress('baseline_cached_FP32_observed_panels')
        items=io.load_json(p['eval_inputs']);assess(engine,[x for x in items if x['panel'] in ('observed75','observed74')],progress,'baseline')
        longchecks=[]
        for info in [x for x in items if x['panel']=='long']:
            packet=raw_unpack(torch.load(info['raw'],map_location='cpu',weights_only=True));z,_=resolve_saved_commit(packet,info['row'],engine.embedding_sha,p['files'],ROOT,512)
            c=DifferentiableCodecCommit.from_semantic(z.to(engine.device),((info['row']['id'],'baseline-fixed-cached','metadata'),),engine.embedding_sha);budget.charge('readout')
            with torch.no_grad():current=engine.model.verbalize(c).cpu()
            same=torch.equal(current.argmax(-1),packet['logits'].argmax(-1));difference=float((current-packet['logits']).abs().max())
            if not same or difference>1e-4:raise ValueError('old long FP32 baseline same-head/cachedCommit mismatch')
            longchecks.append(dict(id=info['row']['id'],all_argmax_same=same,logit_max_abs=difference));del current,c,z,packet;gc.collect();progress('baseline_long_FP32_replay_check')
        io.atomic_json(out/'baseline_long_FP32_parity.json',longchecks);progress('finite_fullFP32_teacher')
        while engine.cache_cursor<len(rows):
            engine.cache_next()
            if engine.cache_cursor%8==0:progress('finite_fullFP32_teacher')
        progress('initialize_ALL_TRAIN_weak_critical_reference');engine.initialize_reference();engine.commit_reference_files();latest=save(engine,'teacher_complete');progress('bounded_cause_directed_last2_training')
        while engine.attempts<len(orders):
            if time.time()>=p['cutoff']-120:terminal='CRITICAL_REPAIR_V88_EARLY_STOP_MODEL_RESERVE';break
            engine.step();progress('bounded_cause_directed_last2_training')
            if len(engine.history)>=32 and not any(h['update_receipt']['accepted'] for h in engine.history[-32:]):terminal='CRITICAL_REPAIR_V88_EARLY_STOP_32_NO_ACCEPTED_UPDATES';break
        engine.verify_frozen();progress('terminal_TRAIN_reference');final_reference(engine);progress('terminal_cached_FP32_assessment');assess(engine,io.load_json(p['eval_inputs']),progress)
        engine.verify_frozen();latest=save(engine,'model');progress('terminal_complete')
        io.atomic_json(out/'result.json',dict(status=terminal,attempts=engine.attempts,updates=engine.updates,rejections=engine.rejections,AMP_skips=engine.skips,seconds=time.time()-started,
            calls=budget.calls,reference_patch_work=engine.reference_patch_work,bank_patch_count=engine.bank.patch_count,weak_critical_positions=engine.bank.weak_count,bank_reference=engine.bank_ref,
            final_reference=engine.reference_final,exposure=engine.exposure,evaluations=engine.evaluations,model=latest,restore_proof=restore_proof,Exact_entries=0,copy_bytes=0,denoise=0,
            whole_language_cycle_complete=False,automatic_promotion=False,output_bytes=size(out),fresh_source_encoding_only_TRAIN=True,long_evaluation_cached_Commit_only=True,
            evidence_schema='decision trace with statewise UTF8 winners, goldLP and sampled fullscores; NOT full258 raw storage',old_budgets_closed=True))
        return 0
    except BaseException as error:
        incident=dict(error=repr(error),traceback=traceback.format_exc(),attempts=engine.attempts,calls=budget.calls)
        try:incident['emergency']=save(engine,'emergency',True)
        except BaseException as secondary:incident['emergency_error']=repr(secondary)
        io.atomic_json(out/'incident.json',incident);return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--plan',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--preflight',action='store_true');x=a.parse_args();raise SystemExit(run(x.plan.resolve(),x.output.resolve(),x.preflight))
