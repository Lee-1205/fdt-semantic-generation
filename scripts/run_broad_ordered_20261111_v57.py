"""New broad 8192-document role-codec epoch. No semantic denoiser or raw copy."""
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
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout,fuzzy_role_reads
from research.codec_commit_stop_v56 import reconstruct_commit_logits
from research.text_diffusion_v1 import pack_prompts
from broad_ordered_engine_v57 import BroadOrderedEngine
from run_ordered_fuzzy_20261111_v55 import Budget as BaseBudget,load_parent,SOURCE_SHA,save_tensor

PARENT_SHA='77dadefc0be5ddfde47654e162ec2a8d91e4693e22748a1a9357c68527d8be73'
PARENT_ALGORITHM='b34c094d93a1d3d901e1c063c2000bb98863e222655d82026cb93398952061ad'

def validate(path,output):
    p=io.load_json(path)
    if (p['schema']!='fdt_broad_ordered_v57' or p['output']!=str(output.resolve()) or p['source_sha']!=SOURCE_SHA
        or p['head_sha']!=PARENT_SHA or p['wall_seconds']!=3300 or p['max_calls']!=17000
        or p['max_output_bytes']!=1024**3 or p['attempts']!=8192 or p['epochs']!=1
        or p['capacity']!=16384 or p['cost']!=16416):raise ValueError('actual V57 bounds')
    if p['authorized'] and (p['deadline']-p['start']!=3300 or p['cutoff']!=p['deadline']-300):raise ValueError('new fixed clock')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('pin changed '+name)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core')
    train=io.load_json(Path(p['data'])/'train.json');dev=io.load_json(Path(p['data'])/'dev.json')
    order=list(range(len(train)));random.Random(57111).shuffle(order);orders=[[i] for i in order]
    if len(train)!=8192 or len(dev)!=16 or sum(r['actual_bytes'] for r in train)!=40416999:raise ValueError('actual source pool')
    return p,train,dev,orders

class Budget(BaseBudget):
    def charge(self,kind):
        self.check()
        if kind not in ('encode','readout') or sum(self.calls.values())>=17000:raise ValueError('fixed live codec call cap')
        self.calls[kind]+=1

def initialize(p,budget,orders,plan_sha,device):
    state=torch.load(p['head_source'],weights_only=True,map_location='cpu')
    if support.state_sha(state)!=PARENT_ALGORITHM or state['updates']!=3624 or len(state['optimizer']['state'])!=8:
        raise ValueError('complete V56 active trajectory')
    head=OrderedFuzzyReadout();head.load_state_dict(state['head'],strict=True)
    engine=BroadOrderedEngine(head,None,None,orders,budget,plan_sha,device)
    engine.optimizer.load_state_dict(state['optimizer']);engine.scaler.load_state_dict(state['scaler'])
    engine.inactive_cuda_rng=copy.deepcopy(state['cuda_rng']);torch.set_rng_state(state['torch_rng'])
    if device=='cuda':torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng']);n=state['numpy_rng']
    np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
    current=engine.export()
    for key in ('head','optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng'):
        if support.state_sha({key:state[key]})!=support.state_sha({key:current[key]}):raise ValueError('active trajectory '+key)
    return engine

@torch.no_grad()
def evaluate(model,engine,rows):
    raw=[];scores=[];engine.head.eval()
    for i,row in enumerate(rows):
        source=row['prompt'];data=source.encode();x=pack_prompts([source],len(data),'cuda');engine.budget.charge('encode')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            ep=model.encode_input(x,source_ids=(row['id'],),source_versions=(SOURCE_SHA,),model_epoch=SOURCE_SHA)
            roles=fuzzy_role_reads(ep.diagnostics['token_memberships'],ep.diagnostics['token_values'],x!=256)
            commit=engine.head.bind_commit(ep.commit,roles);engine.budget.charge('readout');logits=engine.head.from_commit(commit).float().cpu()
        y=torch.full(logits.shape[:2],257,dtype=torch.long);flat=y.flatten();flat[:len(data)]=torch.tensor(list(data));flat[len(data)]=256
        active=y!=257;pred=logits.argmax(-1)
        loss=torch.nn.functional.cross_entropy(logits.transpose(1,2),y.masked_fill(~active,-100),ignore_index=-100,reduction='sum')
        scores.append(dict(id=row['id'],NLL_sum=float(loss),correct=int(((pred==y)&active).sum()),symbols=int(active.sum())))
        save_tensor(engine.budget.output/f'raw_dev_{i:02d}.pt',dict(logits=logits,targets=y),engine.budget)
        if i in engine.budget.p['raw_indices']:
            decoded=reconstruct_commit_logits(logits);text=decoded.pop('text');raw.append(dict(id=row['id'],source=source,output=text,exact=text==source,**decoded))
    io.atomic_json(engine.budget.output/'same_input_reconstruction.json',raw)
    return dict(NLL=sum(r['NLL_sum'] for r in scores)/sum(r['symbols'] for r in scores),
        accuracy=sum(r['correct'] for r in scores)/sum(r['symbols'] for r in scores),scores=scores,
        raw_exact=sum(r['exact'] for r in raw),raw_count=4)

def run(path,output,preflight=False):
    p,train,dev,orders=validate(path,output);torch.set_num_threads(4);budget=Budget(p);device='cpu' if preflight else 'cuda'
    model,report=load_parent(p,device);engine=initialize(p,budget,orders,support.sha(path),device)
    if preflight:
        state=engine.export();digest=support.state_sha(state);stream=memoryio.BytesIO();torch.save(state,stream);stream.seek(0)
        engine.restore(torch.load(stream,weights_only=True,map_location='cpu'))
        if support.state_sha(engine.export())!=digest or torch.cuda.is_initialized():raise ValueError('actual whole engine restore')
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(report,all8_V56_Adam_scaler_RNG_exact=True,
            fullsize_engine_restore=True,CUDA_initialized=False,actual_parent_forward=0)))
        print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('one sealed launch')
    latest=None;began=time.time();guard=False
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=began))
    def progress(phase):
        engine.phase=phase;io.atomic_json(output/'progress.json',dict(phase=phase,attempts=engine.attempts,updates=engine.updates,
            cursor=engine.cursor,calls=budget.calls,time=time.time(),native_pid=os.getpid(),parent_pid=os.getppid(),latest=latest))
    def save():
        nonlocal latest
        budget.check(write_bytes=100*1024**2,emergency=engine.phase=='stopped')
        directory=output/('emergency' if engine.phase=='stopped' else 'model');directory.mkdir(exist_ok=True)
        latest=dict(support.save_full(directory,engine.export()),directory='model',standalone=False,parent_required=SOURCE_SHA)
        io.atomic_json(output/'latest.json',latest)
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        progress('initial_broad_commit');save();progress('broad_ordered_codec_training')
        while engine.attempts<len(orders):
            if time.time()+120>=p['cutoff']:raise TimeoutError('final raw assessment reserve')
            engine.step_document(model,train[orders[engine.attempts][0]])
            if engine.attempts in (1,2048,4096,6144):save()
            if engine.attempts%8==0:progress('broad_ordered_codec_training')
        progress('same_input_assessment');metrics=evaluate(model,engine,dev)
        if support.state_sha(model.state_dict())!=report['parent_model_digest']:raise ValueError('frozen source changed')
        progress('final_broad_commit');save()
        io.atomic_json(output/'result.json',dict(status='BROAD_ORDERED_V57_COMPLETE',attempts=engine.attempts,updates=engine.updates,
            AMP_skips=engine.skips,exposure=engine.exposure,calls=budget.calls,metrics=metrics,seconds=time.time()-began,
            latest=latest,source_optimizer_updates=0,denoise_calls=0,source_copy_bytes=0,
            limitation='One original TRAIN epoch, same observed DEV; not perfect semantics or a new blind benchmark.'))
        progress('complete');return 0
    except BaseException as error:
        engine.optimizer.zero_grad(set_to_none=True);io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc()))
        progress('stopped')
        try:save()
        except BaseException as failure:io.atomic_json(output/'save_error.json',dict(error=repr(failure)))
        progress('stopped');return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    raise SystemExit(run(a.plan.resolve(),a.output.resolve(),a.preflight))
