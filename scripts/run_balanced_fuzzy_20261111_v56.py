"""A new finite objective phase: TRAIN-derived byte classes and first32 balance.

Old V53 three banks remain frozen. Continue the eight V55 ordered-codec Adam
states/scaler/RNG, using immutable source-role cache. No source byte reaches head.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,ctypes,io as memoryio,json,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from fdt_local.runner import numpy_state
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.codec_commit_stop_v56 import reconstruct_commit_logits,guard_codec_stop
from balanced_fuzzy_engine_v56 import ReadoutEngine,make_orders
from run_ordered_fuzzy_20261111_v55 import Budget as BaseBudget,load_parent,SOURCE_SHA,save_tensor

PARENT_SHA='99f2842d9505a531773cdb93248f533622365fed7629d6bc4cbce3a22d73502f'
PARENT_ALGORITHM='954d35b78f64d99cabc1a28d6315f462adce19e6007f28e518acf682333248dd'

def validate(path,output):
    p=io.load_json(path)
    if (p['schema']!='fdt_balanced_fuzzy_v56' or p['output']!=str(output.resolve())
        or p['source_sha']!=SOURCE_SHA or p['head_sha']!=PARENT_SHA or p['wall_seconds']!=900
        or p['max_calls']!=3800 or p['max_output_bytes']!=512*1024**2
        or p['attempts']!=3624 or p['epochs']!=6 or p['batch']!=256
        or p['early_patch_multiplier']!=4 or p['cost']!={'encode':0,'readout':3640,'denoise':0}
        or len(p['class_weights'])!=258 or p['class_weights'][256:]!=[1.,0.]):raise ValueError('fixed V56 contract')
    if p['authorized'] and (p['deadline']-p['start']!=900 or p['cutoff']!=p['deadline']-120):raise ValueError('new fixed clock')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('pin changed '+name)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core changed')
    rows=io.load_json(Path(p['data'])/'train.json');dev=io.load_json(Path(p['data'])/'dev.json')
    if len(rows)!=512 or len(dev)!=16:raise ValueError('same immutable split')
    orders,active=make_orders(rows)
    if len(orders)!=3624 or len(active)!=154567:raise ValueError('finite patch order')
    return p,rows,dev,orders

class Budget(BaseBudget):
    def charge(self,kind):
        self.check()
        if kind!='readout' or sum(self.calls.values())>=3800:raise ValueError('no encoder or denoiser in cached phase')
        self.calls[kind]+=1

def initialize(p,budget,orders,plan_sha,device):
    state=torch.load(p['head_source'],weights_only=True,map_location='cpu')
    if support.state_sha(state)!=PARENT_ALGORITHM or state['updates']!=3624 or len(state['optimizer']['state'])!=8:
        raise ValueError('complete immutable V55 active trajectory')
    head=OrderedFuzzyReadout();head.load_state_dict(state['head'],strict=True)
    engine=ReadoutEngine(head,None,None,orders,budget,plan_sha,device)
    engine.optimizer.load_state_dict(state['optimizer']);engine.scaler.load_state_dict(state['scaler'])
    for k in ('cache_refs','extract_cursor','extract_shards'):setattr(engine,k,copy.deepcopy(state[k]))
    engine.phase='new_balanced_objective';engine.inactive_cuda_rng=copy.deepcopy(state['cuda_rng'])
    torch.set_rng_state(state['torch_rng'])
    if device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng']);n=state['numpy_rng']
    np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
    current=engine.export()
    for key in ('head','optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng'):
        if support.state_sha({key:state[key]})!=support.state_sha({key:current[key]}):raise ValueError('V55 active trajectory import '+key)
    return engine

def add_early(features):
    early=torch.zeros(len(features['local']),dtype=torch.bool)
    for start,end in features['offsets']:early[start:min(start+2,end)]=True
    return dict(features,early=early)

@torch.no_grad()
def evaluate(engine,features,targets,rows):
    engine.head.eval();parts=[]
    for start in range(0,len(targets),256):
        ids=torch.arange(start,min(start+256,len(targets)));f=features;engine.budget.charge('readout')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=engine.head(f['local'][ids].cuda(),f['global'][f['doc_index'][ids]].cuda(),
                f['terminal'][ids].cuda(),f['fuzzy_roles'][ids].cuda()).float()
        parts.append(logits.cpu())
    logits=torch.cat(parts);active=targets!=257;byte=targets<256;EOS=targets==256;pred=logits.argmax(-1)
    loss=torch.nn.functional.cross_entropy(logits.transpose(1,2),targets.masked_fill(~active,-100),ignore_index=-100,reduction='sum')
    metrics=dict(NLL=float(loss/active.sum()),accuracy=float(((pred==targets)&active).sum()/active.sum()),
        byte_accuracy=float(((pred==targets)&byte).sum()/byte.sum()),EOS_recall=float(((pred==256)&EOS).sum()/EOS.sum()),
        EOS_false_positives=int(((pred==256)&byte).sum()),symbols=int(active.sum()))
    save_tensor(engine.budget.output/'normal_raw_logits.pt',dict(logits=logits,targets=targets,offsets=features['offsets']),engine.budget)
    raw=[]
    for i in engine.budget.p['raw_indices']:
        start,end=features['offsets'][i];result=reconstruct_commit_logits(logits[start:end]);text=result.pop('text')
        raw.append(dict(id=rows[i]['id'],source=rows[i]['prompt'],output=text,exact=text==rows[i]['prompt'],**result))
    io.atomic_json(engine.budget.output/'same_input_reconstruction.json',raw)
    metrics.update(raw_exact=sum(r['exact'] for r in raw),raw_count=4,codec_stop='Commit structure only; no gold byte count or forced EOS')
    return metrics

def run(path,output,preflight=False):
    p,train,dev,orders=validate(path,output);torch.set_num_threads(4);budget=Budget(p)
    device='cpu' if preflight else 'cuda';engine=initialize(p,budget,orders,support.sha(path),device)
    if preflight:
        model,report=load_parent(p,'cpu');del model
        state=engine.export();digest=support.state_sha(state);stream=memoryio.BytesIO();torch.save(state,stream);stream.seek(0)
        engine.restore(torch.load(stream,weights_only=True,map_location='cpu'))
        if support.state_sha(engine.export())!=digest or torch.cuda.is_initialized():raise ValueError('actual CPU recovery')
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(report,all8_V55_Adam_scaler_RNG_exact=True,
            new_trainable_parameters=2916384,fullsize_engine_restore=True,CUDA_initialized=False,actual_parent_forward=0)))
        print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('one new sealed execution')
    began=time.time();latest=None;guard=False
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=began))
    def progress(phase):
        engine.phase=phase
        io.atomic_json(output/'progress.json',dict(phase=phase,attempts=engine.attempts,updates=engine.updates,
            cursor=engine.cursor,calls=budget.calls,time=time.time(),native_pid=os.getpid(),parent_pid=os.getppid(),latest=latest))
    def save(name,emergency=False):
        nonlocal latest
        budget.check(write_bytes=100*1024**2,emergency=emergency);directory=output/name;directory.mkdir()
        latest=dict(support.save_full(directory,engine.export()),directory=name,standalone=False,parent_required=p['source_sha'])
        io.atomic_json(output/'latest.json',latest)
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        f=torch.load(p['train_features'],weights_only=True,map_location='cpu');y=torch.load(p['train_targets'],weights_only=True,map_location='cpu')
        engine.features,engine.targets=add_early(f),y
        progress('initial_balanced_commit');save('initial');progress('balanced_codec_training')
        while engine.attempts<len(orders):
            if time.time()+60>=p['cutoff']:raise TimeoutError('final assessment reserve')
            engine.step()
            if engine.attempts%8==0:progress('balanced_codec_training')
        progress('same_input_assessment')
        features=torch.load(p['dev_features'],weights_only=True,map_location='cpu')
        targets=torch.load(p['dev_targets'],weights_only=True,map_location='cpu')
        metrics=evaluate(engine,features,targets,dev)
        progress('final_balanced_commit');save('model')
        if support.sha(p['head_source'])!=p['head_sha'] or support.sha(p['source'])!=p['source_sha']:raise ValueError('parents changed')
        io.atomic_json(output/'result.json',dict(status='BALANCED_FUZZY_V56_COMPLETE',attempts=engine.attempts,updates=engine.updates,
            AMP_skips=engine.skips,exposure=engine.exposure,calls=budget.calls,metrics=metrics,latest=latest,seconds=time.time()-began,
            old_source_optimizer_updates=0,active8Adam_inherited_step3624=True,denoise_calls=0,source_copy_bytes=0,
            limitation='TRAIN-derived class and prefix balance on the same observed corpus; not blind or a guarantee of perfect semantics.'))
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
    parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    raise SystemExit(run(a.plan.resolve(),a.output.resolve(),a.preflight))
