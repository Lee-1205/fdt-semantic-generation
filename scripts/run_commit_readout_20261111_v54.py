"""One frozen-encoder, three-memory Commit-to-language readout learning probe."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,gc,io as memoryio,json,random,shutil,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.prefix_independent_codec_v53 import PrefixIndependentCodec
from research.commit_patch_readout_v54 import CommitPatchReadout,decode_fixed_logits
from research.text_diffusion_v1 import pack_prompts
from commit_readout_engine_v54 import ReadoutEngine,make_orders
from owned_output_size_v52 import owned_output_bytes
import text_generation_support_v1 as support
import learning_io_v2 as io

SOURCE_SHA='7ad721eaeec54ad37f6ba6e5e05a73368674d4517dd05d5a052d72096e271b1f'
SOURCE_ALGORITHM='c61778a5bd7e92bba13501f8e887eb7bcff750ed6ee7d94885e7b0ca901c9531'

def validate(path,output):
    p=io.load_json(path)
    if (p['schema']!='fdt_frozen_commit_readout_v54' or p['output']!=str(output.resolve())
        or p['source_sha']!=SOURCE_SHA or p['wall_seconds']!=900 or p['max_calls']!=5000
        or p['max_output_bytes']!=2*1024**3 or p['attempts']!=3624 or p['epochs']!=6 or p['batch']!=256
        or p['raw_indices']!=[0,4,8,12] or p['source_encoder_optimizer_updates']!=0):raise ValueError('actual V54 plan contract')
    if p['authorized'] and (p['deadline']-p['start']!=900 or p['cutoff']!=p['deadline']-120):raise ValueError('fixed V54 clock')
    for name,digest in p['files'].items():
        if support.sha(name)!=digest:raise ValueError('pin changed '+name)
    if support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('original core changed')
    data=Path(p['data']);train=io.load_json(data/'train.json');dev=io.load_json(data/'dev.json')
    if len(train)!=512 or len(dev)!=16:raise ValueError('finite source document pool')
    orders,active=make_orders(train)
    if len(orders)!=3624 or len(active)!=154567:raise ValueError('finite patch order')
    return p,train,dev,orders,active

class Budget:
    def __init__(self,p):
        self.p=p;self.output=Path(p['output']);self.calls=dict(encode=0,readout=0,denoise=0)
        self.last_size_scan=0;self.last_size=0
        self.parent_reference=dict(path=p['source'],sha256=p['source_sha'],format='V53_standalone_with_ALL96_Adam_and_RNG')
    def check(self,write_bytes=0,emergency=False):
        if not emergency and self.p['authorized'] and ((self.output/'STOP').exists() or time.time()>=self.p['cutoff']):raise InterruptedError('fixed clock/STOP')
        if shutil.disk_usage(ROOT).free<101*1024**3+write_bytes:raise OSError('C100GiB floor+write reserve')
        if self.output.exists() and (write_bytes or time.monotonic()-self.last_size_scan>2):
            self.last_size=owned_output_bytes(self.output);self.last_size_scan=time.monotonic()
        if self.last_size+write_bytes>self.p['max_output_bytes']:raise OSError('owned output cap')
    def charge(self,kind):
        self.check()
        if kind=='denoise' or sum(self.calls.values())>=5000:raise ValueError('V54 fixed model call cap')
        self.calls[kind]+=1

def load_parent(p,device):
    source=torch.load(p['source'],map_location='cpu',weights_only=True)
    if support.state_sha(source)!=SOURCE_ALGORITHM or source['attempts']!=8192 or source['updates']!=8192 or len(source['optimizer']['state'])!=96:
        raise ValueError('complete immutable V53 parent')
    model=PrefixIndependentCodec(Config(**source['core']),input_capacity=16384)
    model.enable_position_adapter();model.enable_long_capacity();model.enable_local_banks()
    model.load_state_dict(source['model'],strict=True);model.requires_grad_(False)
    digest=support.state_sha(source['model'])
    if support.state_sha(model.state_dict())!=digest:raise ValueError('all frozen parent tensors')
    model=model.to(device).eval()
    report=dict(all_parent_tensors_exact=True,all_old_Adam_exact=True,parent_Adam_states=96,
        parent_Adam_used_for_training=False,parent_model_digest=digest,encoder_trainable_parameters=0,
        parent_is_required_for_runtime_and_old_Adam=True)
    del source
    return model,report

def initialize_head(orders,budget,plan_sha,device):
    torch.manual_seed(54111);random.seed(54111);np.random.seed(54111)
    if device=='cuda':torch.cuda.manual_seed_all(54111)
    head=CommitPatchReadout()
    return ReadoutEngine(head,None,None,orders,budget,plan_sha,device)

@torch.no_grad()
def extract(model,rows,engine,progress,label):
    budget=engine.budget;engine.phase='extract_'+label
    locals_,globals_,doc_index,terminal,labels,offsets=[],[],[],[],[],[]
    at=0
    directory=budget.output/'cache_shards';directory.mkdir(exist_ok=True)
    for i,row in enumerate(rows):
        if i<engine.extract_cursor[label]:
            saved=engine.extract_shards[label][i]
            f=torch.load(saved['features'],map_location='cpu',weights_only=True)
            y=torch.load(saved['targets'],map_location='cpu',weights_only=True)
            local=f['local'];count=len(local)
            locals_.append(local);globals_.append(f['global']);doc_index.append(torch.full((count,),i));terminal.append(f['terminal']);labels.append(y)
            offsets.append([at,at+count]);at+=count;continue
        budget.charge('encode');text=row['prompt'];raw=text.encode()
        x=pack_prompts([text],len(raw),'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            episode=model.encode_input(x,source_ids=(row['id'],),source_versions=(budget.p['source_sha'],),model_epoch=budget.p['source_sha'])
        z=episode.commit.reader_tensor().cpu()
        local=torch.cat((z[0,2:],torch.zeros(1,512)),0);count=len(local)
        # An extra structural tail follows the last Commit slot. It is not a
        # gold output length or a literal source span supplied to the head.
        role=torch.zeros(count,dtype=torch.bool);role[-1]=True
        y=torch.full((count*16,),257,dtype=torch.long)
        y[:len(raw)]=torch.tensor(list(raw));y[len(raw)]=256
        feature_path=directory/f'{label}_{i:04d}_features.pt';target_path=directory/f'{label}_{i:04d}_targets.pt'
        f=dict(local=local,global_views=z[0,:2],terminal=role,parent_sha=SOURCE_SHA,id=row['id']);f['global']=f.pop('global_views')
        engine.cache_refs[str(feature_path)]=save_tensor(feature_path,f,budget)
        engine.cache_refs[str(target_path)]=save_tensor(target_path,y.reshape(count,16),budget)
        engine.extract_shards[label].append(dict(features=str(feature_path),targets=str(target_path),id=row['id']))
        engine.extract_cursor[label]=i+1
        io.atomic_json(budget.output/'extraction_journal.json',dict(parent_sha=SOURCE_SHA,phase=engine.phase,
            cursors=engine.extract_cursor,shards=engine.extract_shards,calls=budget.calls))
        locals_.append(local);globals_.append(z[0,:2]);doc_index.append(torch.full((count,),i));terminal.append(role);labels.append(y.reshape(count,16))
        offsets.append([at,at+count]);at+=count
        if i%8==0 or i==len(rows)-1:progress('extract_'+label,completed=i+1,total=len(rows))
        del episode,z,x
    return dict(schema='frozen_three_memory_Commit_features_v54',parent_sha=budget.p['source_sha'],
        local=torch.cat(locals_),global_views=torch.stack(globals_),doc_index=torch.cat(doc_index),terminal=torch.cat(terminal),
        offsets=offsets,ids=[row['id'] for row in rows]),torch.cat(labels)

def save_tensor(path,payload,budget):
    size=sum(t.numel()*t.element_size() for t in payload.values() if isinstance(t,torch.Tensor)) if isinstance(payload,dict) else payload.numel()*payload.element_size()
    budget.check(write_bytes=size+8*1024**2)
    with path.open('xb') as stream:torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    return support.sha(path)

@torch.no_grad()
def evaluate(engine,features,targets,rows,output,condition):
    engine.head.eval();parts=[];n=len(features['local']);d=len(features['global'])
    for start in range(0,n,256):
        ids=torch.arange(start,min(start+256,n));f=features
        local=f['local'][ids];global_views=f['global'][f['doc_index'][ids]]
        if condition=='zero':local=torch.zeros_like(local);global_views=torch.zeros_like(global_views)
        elif condition=='swap':
            local=f['local'][(ids+n//2)%n]
            global_views=f['global'][(f['doc_index'][ids]+d//2)%d]
        engine.budget.charge('readout')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits=engine.head(local.to('cuda'),global_views.to('cuda'),f['terminal'][ids].to('cuda')).float()
        parts.append(logits.cpu())
    logits=torch.cat(parts);active=targets!=257
    ce=torch.nn.functional.cross_entropy(logits.transpose(1,2),targets.masked_fill(~active,-100),ignore_index=-100,reduction='none')
    pred=logits.argmax(-1);byte=targets<256;EOS=targets==256
    metrics=dict(NLL=float(ce.sum()/active.sum()),accuracy=float(((pred==targets)&active).sum()/active.sum()),
        byte_accuracy=float(((pred==targets)&byte).sum()/byte.sum()),EOS_recall=float(((pred==256)&EOS).sum()/EOS.sum()),
        symbols=int(active.sum()),condition=condition)
    save_tensor(output/(condition+'_raw_logits.pt'),dict(logits=logits,targets=targets,offsets=features['offsets']),engine.budget)
    if condition=='normal':
        raw=[]
        for i in engine.budget.p['raw_indices']:
            begin,end=features['offsets'][i]
            decoded=decode_fixed_logits(logits[begin:end])
            raw.append(dict(id=rows[i]['id'],source=rows[i]['prompt'],output=decoded.pop('text'),**decoded))
            raw[-1]['exact']=raw[-1]['output']==raw[-1]['source']
        io.atomic_json(output/'same_input_reconstruction.json',raw)
        metrics.update(raw_exact=sum(row['exact'] for row in raw),raw_count=4)
    io.atomic_json(output/(condition+'_metrics.json'),metrics)
    return metrics

def run(path,output,preflight=False):
    p,train,dev,orders,active=validate(path,output);torch.set_num_threads(4)
    budget=Budget(p);device='cpu' if preflight else 'cuda';model,report=load_parent(p,device)
    engine=initialize_head(orders,budget,support.sha(path),device)
    report.update(new_readout_parameters=sum(x.numel() for x in engine.head.parameters()),head_Adam_fresh=True)
    if preflight:
        payload=engine.export();digest=support.state_sha(payload);stream=memoryio.BytesIO();torch.save(payload,stream);stream.seek(0)
        loaded=torch.load(stream,map_location='cpu',weights_only=True);engine.restore(loaded)
        if support.state_sha(engine.export())!=digest or torch.cuda.is_initialized():raise ValueError('CPU head reference/fullstate recovery')
        print('CPU_PREFLIGHT_REPORT '+json.dumps(dict(report,fullsize_engine_restore=True,CUDA_initialized=False,actual_parent_forward=0)))
        print('CPU_PREFLIGHT_PASS');return 0
    if not p['authorized'] or (output/'worker_started.json').exists():raise ValueError('one newly sealed execution')
    io.atomic_json(output/'worker_started.json',dict(pid=os.getpid(),parent_pid=os.getppid(),time=time.time()))
    began=time.time();guard=False;latest=None
    def progress(phase,**extra):
        engine.phase=phase
        io.atomic_json(output/'progress.json',dict(phase=phase,attempts=engine.attempts,updates=engine.updates,cursor=engine.cursor,
            calls=budget.calls,time=time.time(),native_pid=os.getpid(),parent_pid=os.getppid(),latest=latest,**extra))
    def save(kind,emergency=False):
        nonlocal latest
        budget.check(write_bytes=100*1024**2,emergency=emergency)
        directory=output/kind;directory.mkdir()
        latest=dict(support.save_full(directory,engine.export()),directory=kind,standalone=False,parent_required=SOURCE_SHA)
        io.atomic_json(output/'latest.json',latest)
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        io.atomic_json(output/'import_review.json',report)
        progress('before_extraction');save('seed')
        caches={}
        for name,rows in [('train',train),('dev',dev)]:
            features,targets=extract(model,rows,engine,progress,name)
            features['global']=features.pop('global_views')
            refs={}
            for suffix,value in [('features',features),('targets',targets)]:
                file=output/(name+'_'+suffix+'.pt');refs[str(file)]=save_tensor(file,value,budget)
            engine.cache_refs.update(refs);caches[name]=(features,targets)
        if support.state_sha(model.state_dict())!=report['parent_model_digest']:raise ValueError('frozen encoder/memory changed')
        io.atomic_json(output/'frozen_parent_after_extract.json',dict(parent_model_digest_unchanged=True,source_Adam_updates=0))
        del model;gc.collect();torch.cuda.empty_cache()
        engine.features,engine.targets=caches['train']
        actual=torch.nonzero((engine.targets!=257).any(1)).flatten().tolist()
        if actual!=active:raise ValueError('actual immutable cache row ordering')
        progress('initial_readout_commit');save('initial')
        progress('readout_training')
        while engine.attempts<len(orders):
            if time.time()+60>=p['cutoff']:raise TimeoutError('final assessment reserve')
            engine.step()
            if engine.attempts%8==0:progress('readout_training')
        progress('readout_assessment')
        metrics={name:evaluate(engine,*caches['dev'],dev,output,name) for name in ('normal','zero','swap')}
        progress('final_readout_commit');save('model')
        if support.sha(p['source'])!=SOURCE_SHA:raise ValueError('parent file changed')
        io.atomic_json(output/'result.json',dict(status='FROZEN_COMMIT_READOUT_V54_COMPLETE',attempts=engine.attempts,
            updates=engine.updates,AMP_skips=engine.skips,exposure=engine.exposure,calls=budget.calls,metrics=metrics,
            latest=latest,seconds=time.time()-began,parent_encoder_memory_optimizer_updates=0,
            source_unchanged=True,main_research_model='V53 retained; no automatic promotion',
            limitation='Different learned decoder on the same frozen Commit; not proof of perfect semantics or independent denoising.'))
        progress('complete');return 0
    except BaseException as error:
        engine.optimizer.zero_grad(set_to_none=True)
        io.atomic_json(output/'incident.json',dict(error=repr(error),traceback=traceback.format_exc()))
        try:save('emergency',True)
        except BaseException as saving:io.atomic_json(output/'save_error.json',dict(error=repr(saving)))
        progress('stopped');return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    raise SystemExit(run(a.plan.resolve(),a.output.resolve(),a.preflight))
