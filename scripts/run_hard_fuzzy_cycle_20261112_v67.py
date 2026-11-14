"""Live OWT sentence/window -> fuzzy memory -> sentence, Exact dormant."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,gc,io as memoryio,json,random,sys,time,traceback
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64,EXACT_POLICY
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.three_memory_codec_v33 import DifferentiableCodecCommit
from research.codec_commit_stop_v56 import reconstruct_commit_logits
from research.text_diffusion_v1 import pack_prompts
from hard_fuzzy_cycle_engine_v67 import CycleEngine,SCHEMA
from run_ordered_fuzzy_20261111_v55 import Budget as BaseBudget,SOURCE_SHA,SOURCE_ALGORITHM
from run_semantic_binding_20261111_v61 import save_tensor

HEAD_SHA='a7bbdd7a358bf6beb908a3df0f86c927d89126a1c0d15bc31c54241346b5e2b1'
HEAD_ALGORITHM='4608416d299da2643d2a99766ee4e90982d0119a6deff4056d82c3e2fd9f06c8'


def validate(path,output):
    p=io.load_json(path)
    if (p['schema']!='fdt_hard_fuzzy_cycle_v67' or p['output']!=str(output.resolve()) or p['source_sha']!=SOURCE_SHA or p['head_sha']!=HEAD_SHA
        or p['exact_policy']!=EXACT_POLICY or p['wall_seconds']!=1200 or p['max_calls']!=2200 or p['max_output_bytes']!=4*1024**3
        or p['attempts']!=1024 or p['epochs']!=1 or p['capacity']!=16384 or p['cost']!=2156):raise ValueError('actual live roundtrip contract')
    if p['authorized'] and (p['deadline']-p['start']!=1200 or p['cutoff']!=p['deadline']-180):raise ValueError('new bounded clock')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('pin changed '+name)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core')
    data=Path(p['data']);train=io.load_json(data/'train.json');dev=io.load_json(data/'dev.json');orders=io.load_json(data/'orders.json')
    if len(train)!=1024 or len(dev)!=16 or len(orders)!=1024 or max(len(r['prompt'].encode()) for r in train)!=16384:raise ValueError('declared window pool')
    return p,train,dev,orders


class Budget(BaseBudget):
    def __init__(self,p):
        super().__init__(p);self.calls=dict(encode=0,readout=0,denoise=0);self.data_sha=support.sha(Path(p['data'])/'manifest.json')
        self.parent_reference=dict(cycle_path=p['cycle_source'],cycle_sha256=p['cycle_sha'],path=p['source'],sha256=p['source_sha'],head_path=p['head_source'],head_sha256=p['head_sha'],
            old96_and8_plus30_Adam='preserved unchanged in parent files; new joint optimizer fresh',exact_policy=EXACT_POLICY)
    def charge(self,kind):
        self.check()
        if kind not in ('encode','readout') or sum(self.calls.values())>=2200:raise ValueError('fixed cycle call cap; denoise forbidden')
        self.calls[kind]+=1


def initialize(p,rows,orders,budget,plan_sha,device):
    warm=torch.load(p['cycle_source'],weights_only=True,map_location='cpu')
    if support.state_sha(warm)!=p['cycle_algorithm_sha'] or warm['schema']!='fdt_live_fuzzy_roundtrip_v65_fullstate' or len(warm['optimizer']['state'])!=30 or warm['updates']!=512:raise ValueError('full immutable V65 joint parent')
    torch.manual_seed(67112);random.seed(67112);np.random.seed(67112)
    if device=='cuda':torch.cuda.manual_seed_all(67112)
    encoder=FuzzyCodecV64(replace(Config(**warm['core']),gradient_checkpointing=True),input_capacity=16384)
    encoder.enable_position_adapter();encoder.enable_long_capacity();encoder.enable_local_banks()
    encoder.load_state_dict({k[8:]:v for k,v in warm['model'].items() if k.startswith('encoder.')},strict=True)
    head=OrderedFuzzyReadout();head.load_state_dict({k[8:]:v for k,v in warm['model'].items() if k.startswith('readout.')},strict=True)
    model=FuzzyRoundTripV64(encoder,head)
    if support.state_sha(model.state_dict())!=support.state_sha(warm['model']):raise ValueError('all V65 model tensors warm parity')
    engine=CycleEngine(model,rows,orders,budget,plan_sha,device)
    if engine.objective_report!=p['objective_report']:raise ValueError('TRAIN-only objective weights')
    report=dict(all_parent_tensors_exact=True,all_old_Adam_exact=True,parent_Adam_states=30,parent_Adam_used_for_training=False,
        fresh_joint_optimizer=not bool(engine.optimizer.state),exact_policy=EXACT_POLICY,architecture_unchanged=True,all_V65_tensors_exact=True,
        baseline_reused_not_fresh_assessment=True,objective_report=engine.objective_report,
        actual_trainable_parameters=sum(v.numel() for v in engine.model.parameters() if v.requires_grad),engine_schema=SCHEMA,
        live_encoder_gradient_required=True)
    return engine,report

@torch.no_grad()
def assess(engine,rows,label):
    engine.model.eval();records=[];NLL=correct=count=0;controls=[]
    for i,row in enumerate(rows):
        raw=row['prompt'].encode();x=pack_prompts([row['prompt']],len(raw),engine.device)
        engine.budget.charge('encode');engine.budget.charge('readout')
        with torch.autocast(engine.device,dtype=torch.bfloat16,enabled=engine.device=='cuda'):
            logits,commit,ep=engine.model(x,source_ids=(row['id'],),source_versions=(engine.budget.data_sha,),model_epoch=engine.plan_sha+':'+label)
        y=torch.full(logits.shape[:2],257,device=engine.device,dtype=torch.long);flat=y.flatten();flat[:len(raw)]=torch.tensor(list(raw),device=engine.device);flat[len(raw)]=256
        active=y!=257;ce=torch.nn.functional.cross_entropy(logits.float().transpose(1,2),y,ignore_index=257,reduction='sum')
        NLL+=float(ce);correct+=int(((logits.argmax(-1)==y)&active).sum());count+=int(active.sum())
        decoded=reconstruct_commit_logits(logits.float());output=decoded.pop('text')
        sealed=commit.seal()
        file=engine.budget.output/f'raw_{label}_{i:02d}.pt'
        save_tensor(file,dict(logits=logits.float().cpu(),targets=y.cpu(),fuzzy_commit=sealed.reader_tensor(),memory_revisions=sealed.memory_revisions,
            model_epoch=sealed.model_epoch,Exact_entries=0,Exact_policy=EXACT_POLICY),engine.budget)
        records.append(dict(id=row['id'],source=row['prompt'],output=output,exact=output==row['prompt'],symbols=int(active.sum()),NLL_sum=float(ce),correct=int(((logits.argmax(-1)==y)&active).sum()),**decoded))
        if label=='final' and i in (0,4,8,14):
            empty=DifferentiableCodecCommit.from_semantic(torch.zeros_like(commit.reader_tensor()),commit.memory_revisions,commit.model_epoch)
            engine.budget.charge('readout')
            with torch.autocast(engine.device,dtype=torch.bfloat16,enabled=engine.device=='cuda'):zero=engine.model.verbalize(empty).float()
            controls.append(dict(id=row['id'],zero_content_NLL_sum=float(torch.nn.functional.cross_entropy(zero.transpose(1,2),y,ignore_index=257,reduction='sum')),
                normal_NLL_sum=float(ce),symbols=int(active.sum()),coarse_commit_topology_retained=True))
    io.atomic_json(engine.budget.output/(label+'_reconstructions.json'),records)
    if controls:io.atomic_json(engine.budget.output/'zero_fuzzy_content_controls.json',controls)
    result=dict(NLL=NLL/count,byte_EOS_accuracy=correct/count,exact_sentences=sum(r['exact'] for r in records),rows=len(records),UTF8=sum(r['UTF8'] for r in records),EOS=sum(r['EOS'] for r in records),symbols=count,Exact_entries=0)
    engine.evaluations[label]=result;return result


def run(path,output,preflight=False):
    p,rows,dev,orders=validate(path,output);torch.set_num_threads(4);budget=Budget(p);device='cpu' if preflight else 'cuda'
    engine,report=initialize(p,rows,orders,budget,support.sha(path),device)
    if preflight:
        # Actual warm-size path, distinct from tiny fixture ability scores.
        engine.model.train();text='A fuzzy memory sentence 314159.';x=pack_prompts([text],len(text.encode()),'cpu')
        logits,commit,ep=engine.model(x,source_ids=('actualCPU',),source_versions=('fixture',),model_epoch='preflight')
        y=torch.full(logits.shape[:2],257,dtype=torch.long);flat=y.flatten();raw=text.encode();flat[:len(raw)]=torch.tensor(list(raw));flat[len(raw)]=256
        from research.fuzzy_cycle_objective_v67 import cycle_losses
        cycle_losses(logits,y,engine.symbol_weights)[0].backward()
        named=dict(engine.model.named_parameters())
        if any(named[n].grad is None for n in engine.names) or any(p.grad is not None for p in engine.model.parameters() if not p.requires_grad):raise ValueError('actual fullsize gradient ownership')
        if ep.memories.exact.valid.any():raise ValueError('Exact must be empty')
        engine.optimizer.zero_grad(set_to_none=True);engine.verify_frozen()
        del logits,commit,ep,x,y;gc.collect()
        state=engine.export();digest=support.state_sha(state);stream=memoryio.BytesIO();torch.save(state,stream);stream.seek(0);engine.restore(torch.load(stream,weights_only=True,map_location='cpu'))
        if support.state_sha(engine.export())!=digest or torch.cuda.is_initialized():raise ValueError('fullsize cycle restore')
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(report,fullsize_engine_restore=True,actual_warm_size_roundtrip_gradient=True,actual_new_objective_gradient=True,Exact_entries0=True,CUDA_initialized=False,actual_parent_forward=1,optimizer_steps=0)))
        print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('ONE new cycle only')
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()));latest=None;began=time.time();guard=False
    def progress(phase):
        engine.phase=phase;io.atomic_json(output/'progress.json',dict(phase=phase,attempts=engine.attempts,updates=engine.updates,cursor=engine.cursor,calls=budget.calls,time=time.time(),native_pid=os.getpid(),parent_pid=os.getppid(),latest=latest))
    def save(name,emergency=False):
        nonlocal latest
        budget.check(write_bytes=750*1024**2,emergency=emergency);directory=output/name;directory.mkdir(exist_ok=True)
        latest=dict(support.save_full(directory,engine.export()),directory=name,exact_policy=EXACT_POLICY)
        io.atomic_json(output/'latest.json',latest)
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        progress('initial_fuzzy_cycle');save('initial');progress('reuse_identical_V65_and_V66_baselines');baseline=io.load_json(Path(p['baseline_run'])/'result.json')['final'];engine.evaluations['baseline']=baseline;baseline_English=io.load_json(Path(p['baseline_English_run'])/'root_saved_review.json')['English_all'];engine.evaluations['baseline_English']=baseline_English
        progress('live_language_fuzzy_language_training')
        while engine.attempts<len(orders):
            if time.time()+90>=p['cutoff']:raise TimeoutError('fixed final-cycle reserve')
            engine.step()
            if engine.attempts in (1,512):engine.verify_frozen();save('model')
            if engine.attempts%4==0:progress('live_language_fuzzy_language_training')
        progress('final_roundtrip_assessment');final=assess(engine,dev,'final');retained=assess(engine,io.load_json(Path(p['data'])/'retained_English.json'),'retained_English');engine.verify_frozen();progress('final_fuzzy_commit');save('model')
        io.atomic_json(output/'result.json',dict(status='HARD_FUZZY_CYCLE_V67_COMPLETE',attempts=engine.attempts,updates=engine.updates,AMP_skips=engine.skips,
            exposure=engine.exposure,calls=budget.calls,baseline=baseline,final=final,baseline_English=baseline_English,retained_English=retained,seconds=time.time()-began,latest=latest,
            Exact_entries=0,copy_bytes=0,denoise_calls=0,active_language_encoder=True,active_language_decoder=True,
            limitation='same-input observed OWT windows via fuzzy Commit; Exact dormant; reconstruction is not proof of unique abstract meaning or universal losslessness'))
        progress('complete');return 0
    except BaseException as error:
        engine.optimizer.zero_grad(set_to_none=True);io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc()))
        try:save('emergency',True)
        except BaseException as failure:io.atomic_json(output/'save_error.json',dict(error=repr(failure)))
        progress('stopped');return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args();raise SystemExit(run(args.plan.resolve(),args.output.resolve(),args.preflight))
