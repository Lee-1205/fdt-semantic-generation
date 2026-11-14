"""One new bounded memory-alignment objective trial over immutable caches."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,gc,io as memoryio,json,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.semantic_binding_v61 import SemanticBinderV61
from research.semantic_binding_objective_v63 import OBJECTIVE
from semantic_binding_engine_v63 import SemanticEngine,SCHEMA
from run_semantic_binding_20261111_v61 import evaluate,interchange_events,source_features,target_tensors,save_tensor
from run_ordered_fuzzy_20261111_v55 import Budget as BaseBudget,load_parent,SOURCE_SHA

BINDER_SHA='1c045fe5ead409189c21c0cab0632076adb0d3d9d5f4edd08fc432a66065a25a'
BINDER_ALGORITHM='0dae39e1d6c24762fa0309af9d48f4256aa1739695e97797e8854fa700bf5171'


def validate(path,output):
    p=io.load_json(path)
    if (p['schema']!='fdt_semantic_binding_v63' or p['output']!=str(output.resolve()) or p['source_sha']!=SOURCE_SHA
        or p['wall_seconds']!=900 or p['max_calls']!=2500 or p['max_output_bytes']!=5*1024**3
        or p['attempts']!=2048 or p['epochs']!=8 or p['cost']!=2352 or p['hidden']!=128
        or p['binder_sha']!=BINDER_SHA or p['objective_contract']!=OBJECTIVE):raise ValueError('actual V63 objective/scope')
    if p['authorized'] and (p['deadline']-p['start']!=900 or p['cutoff']!=p['deadline']-120):raise ValueError('new fixed V63 clock')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('pin changed '+name)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core')
    data=Path(p['data']);train=io.load_json(data/'train_features.json');dev=io.load_json(data/'dev_features.json');orders=io.load_json(p['orders'])
    if len(train)!=4096 or len(dev)!=256 or len(orders)!=2048 or any(set(r)!={'id','prompt'} for r in train+dev):raise ValueError('finite unchanged feature contract')
    return p,train,dev,orders


class Budget(BaseBudget):
    def __init__(self,p):
        super().__init__(p);self.calls={'encode':0,'bind':0,'denoise':0};self.baseline_reference=p['baseline_reference']
        self.parent_reference=dict(path=p['source'],sha256=p['source_sha'],codec_path=p['head_source'],codec_sha256=p['head_sha'],
            binder_path=p['binder_source'],binder_sha256=p['binder_sha'],
            old_binder_Adam='preserved closed; new objective fresh AdamW1e-4',old_source96Adam='unchanged in V53',old_codec8Adam='unchanged in V57')
    def charge(self,kind):
        self.check()
        if kind not in ('bind','encode') or sum(self.calls.values())>=2500:raise ValueError('V63 fixed calls; denoise forbidden')
        self.calls[kind]+=1


def run(path,output,preflight=False):
    p,train,dev,orders=validate(path,output);torch.set_num_threads(4);budget=Budget(p);device='cpu' if preflight else 'cuda'
    source,report=load_parent(p,device)  # Verify original source/Adam read-only; NO source forward.
    torch.manual_seed(63112);random.seed(63112);np.random.seed(63112)
    if device=='cuda':torch.cuda.manual_seed_all(63112)
    warm=torch.load(p['binder_source'],weights_only=True,map_location='cpu')
    if support.state_sha(warm)!=BINDER_ALGORITHM or warm['updates']!=2048:raise ValueError('complete closed V62 parent')
    model=SemanticBinderV61(source.core.prior.detach().cpu(),SOURCE_SHA,hidden=128)
    model.load_state_dict(warm['binder'],strict=True)
    if any(not torch.equal(v.cpu(),warm['binder'][k]) for k,v in model.state_dict().items()):raise ValueError('all V61 tensor parity')
    if not torch.equal(model.persistent_prior.cpu(),source.core.prior.detach().cpu().float()):raise ValueError('original K_P prior')
    engine=SemanticEngine(model,None,None,orders,budget,support.sha(path),device)
    report.update(new_semantic_parameters=sum(v.numel() for v in model.parameters()),new_optimizer_fresh=not bool(engine.optimizer.state),
        all_V62_binder_tensors_exact=True,inference_architecture_unchanged=True,objective_contract=OBJECTIVE,
        engine_schema=SCHEMA,graph_contract=engine.export()['graph_contract'],source_encoder_calls=0,new_TRAIN_source_encodes_expected=256,
        old_surface_codec_forward=0,baseline_reused_not_new_assessment=True)
    if preflight:
        state=engine.export();digest=support.state_sha(state);stream=memoryio.BytesIO();torch.save(state,stream);stream.seek(0)
        engine.restore(torch.load(stream,weights_only=True,map_location='cpu'))
        if support.state_sha(engine.export())!=digest or torch.cuda.is_initialized():raise ValueError('actual fullsize V63 restore')
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(report,fullsize_engine_restore=True,CUDA_initialized=False,actual_parent_forward=0)))
        print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('ONE new authorized trial')
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
    del warm;gc.collect();latest=None;began=time.time();guard=False
    def progress(phase,**extra):
        engine.phase=phase;io.atomic_json(output/'progress.json',dict(phase=phase,attempts=engine.attempts,updates=engine.updates,
            cursor=engine.cursor,calls=budget.calls,time=time.time(),native_pid=os.getpid(),parent_pid=os.getppid(),latest=latest,**extra))
    def save(name,emergency=False):
        nonlocal latest
        budget.check(write_bytes=40*1024**2,emergency=emergency);directory=output/name;directory.mkdir(exist_ok=True)
        latest=dict(support.save_full(directory,engine.export()),directory=name,standalone=False,parent_required=SOURCE_SHA)
        io.atomic_json(output/'latest.json',latest)
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        progress('before_reused_memory_load');save('seed');caches={};targets={}
        for label,rows in (('train',train),('dev',dev)):
            if label=='train':
                f=source_features(source,rows,engine,label,progress)
                t=target_tensors(io.load_json(Path(p['data'])/'train_targets.json'))
                for kind,value in (('features',f),('targets',t)):
                    file=output/('train_'+kind+'.pt');engine.cache_refs[str(file)]=save_tensor(file,value,budget)
            else:
                f=torch.load(p['reused_dev_features'],weights_only=True,map_location='cpu')
                t=torch.load(p['reused_dev_targets'],weights_only=True,map_location='cpu')
            if f['source_epoch']!=SOURCE_SHA or [r[0] for r in f['source_revisions']]!=[r['id'] for r in rows]:raise ValueError('cache scope')
            if not torch.equal(t['relation'],torch.tensor([r['meaning']['relation'] for r in io.load_json(Path(p['data'])/(label+'_targets.json'))])):raise ValueError('cached target scope')
            if label=='dev':
                for kind in ('features','targets'):
                    name=p['reused_dev_'+kind];engine.cache_refs[name]=p['files'][name]
                engine.extract_cursor[label]=len(rows);engine.extract_shards[label]=[dict(path=p['reused_dev_features'],reused=True)]
            caches[label]=f;targets[label]=t
        for name,ref in p['baseline_reference'].items():engine.cache_refs[ref['path']]=ref['sha256']
        baseline=io.load_json(p['baseline_reference']['metrics']['path'])
        io.atomic_json(output/'baseline_reference.json',dict(p['baseline_reference'],new_model_calls=0,not_independent_repeat=True))
        io.atomic_json(output/'baseline_metrics.json',baseline)
        if support.state_sha(source.state_dict())!=report['parent_model_digest']:raise ValueError('frozen source changed')
        io.atomic_json(output/'frozen_source_review.json',dict(source_unchanged=True,source_model_digest=report['parent_model_digest'],source_forwards=256,language_encoder_forwards=256,language_decoder_forwards=0,source_optimizer_updates=0))
        del source;gc.collect();torch.cuda.empty_cache()
        engine.features,engine.targets=caches['train'],targets['train'];progress('initial_reused_memory_state');save('initial')
        progress('memory_only_alignment_training')
        while engine.attempts<len(orders):
            if time.time()+60>=p['cutoff']:raise TimeoutError('fixed final assessment reserve')
            engine.step()
            if engine.attempts in (1,1024):save('model')
            if engine.attempts%8==0:progress('memory_only_alignment_training')
        progress('semantic_binding_assessment');dy=io.load_json(Path(p['data'])/'dev_targets.json')
        final=evaluate(engine,caches['dev'],dev,dy,'final')
        controls={bank:evaluate(engine,caches['dev'],dev,dy,'read_off_'+bank,list(range(64)),(bank,)) for bank in ('persistent','contextual','exact')}
        swaps=interchange_events(engine,caches['dev'],dy)
        if support.state_sha({'prior':model.persistent_prior})!=engine.prior_sha:raise ValueError('frozen prior changed')
        progress('final_semantic_commit');save('model')
        io.atomic_json(output/'result.json',dict(status='SEMANTIC_BINDING_V63_COMPLETE',attempts=engine.attempts,updates=engine.updates,AMP_skips=engine.skips,
            exposure=engine.exposure,calls=budget.calls,baseline=baseline,baseline_is_reused_V62=True,final=final,read_channel_controls=controls,event_interchanges=swaps,
            latest=latest,seconds=time.time()-began,source_optimizer_updates=0,language_decoder_updates=0,denoise_calls=0,source_copy_bytes=0,
            objective_contract=OBJECTIVE,limitation='Restricted adaptive observed panel, same inference NN; authored construction coverage and loss-only role competition; broad-family DEV overlap declared; no denoising/general semantics claim'))
        progress('complete');return 0
    except BaseException as error:
        engine.optimizer.zero_grad(set_to_none=True);io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc()))
        try:save('emergency',True)
        except BaseException as failure:io.atomic_json(output/'save_error.json',dict(error=repr(failure)))
        progress('stopped');return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args();raise SystemExit(run(args.plan.resolve(),args.output.resolve(),args.preflight))
