"""Bounded first sentence/fuzzy-extraction/reconstruction cycle, no diffusion training."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,random,sys,time,traceback
from contextlib import nullcontext
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.owt_fuzzy_reconstruction_v12 import FuzzySentenceAE
from research.text_diffusion_v1 import pack_prompts,pack_answers,decode_bytes,PAD
from owt_fuzzy_engine_v12 import AEEngine
from text_language_runtime_support_v6 import RecoveryBudget
import text_generation_support_v1 as support
import learning_io_v2 as io

SCHEMA='fdt_owt_fuzzy_reconstruction_v12'
LIMITS={'attempts':2048,'batch':8,'wall':900,'reserve':90,'max_calls':6500,
    'max_output_bytes':6*1024**3,'start_free_bytes':40*1024**3,'min_gpu_free_mib':12288}
SOURCES={
 'language':('runs/text_decoder_addressing_20261028_v9/model/checkpoints/state_1636.pt','daaaaf0b780e93844efd045321a27a0a96769c21014d426858640c32810181b2'),
 'typed':('runs/querycount_alignment_20261027_v1/model/payload_000010.pt','5494ec7c3cf9d8466d799d05fd0b782a101969dedd645678ecc52154dba93675')}
COST={'training':4096,'initial_NLL_controls':128,'final_NLL_controls':128,'final_free_reconstruction_max':1552,'total_max':5904}

def finite_orders(size,seeds=range(201,209)):
    out=[]
    for seed in seeds:
        ids=list(range(size));random.Random(seed).shuffle(ids);out.extend(ids[i:i+8] for i in range(0,size,8))
    return out

def validate(path,output,worker=None):
    p=io.load_json(path)
    if p.get('schema')!=SCHEMA or type(p.get('authorized')) is not bool or p.get('limits')!=LIMITS:raise ValueError('actual AE schema/caps')
    if Path(p['root']).resolve()!=ROOT or Path(p['output']).resolve()!=Path(output).resolve():raise ValueError('root/output')
    if p.get('epochs')!=8 or p.get('cost_contract')!=COST or p.get('reasoning_diffusion_training') is not False:raise ValueError('finite AE-only contract')
    if p.get('E2_final_accessed') is not False or support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('holdout/core')
    for relative,h in SOURCES.values():
        if p['files'].get(str((ROOT/relative).resolve()))!=h:raise ValueError('source checkpoint binding')
    for file,h in p['files'].items():
        if support.sha(file)!=h:raise ValueError('pinned file changed '+file)
    if worker and str(Path(worker).resolve()) not in p['files']:raise ValueError('invoked worker unpinned')
    if p['deadline'] is not None and (p['deadline']-p['allocation_start']!=900 or p['cutoff']!=p['deadline']-90):raise ValueError('fixed new clock')
    a=Path(p['artifact']);f=io.load_json(a/'features.json');y=io.load_json(a/'targets.json');docs=io.load_json(a/'documents.json');m=io.load_json(a/'manifest.json')
    if m['revision']!='79d93d786212f7344586290adb811d4ae6a1762c' or {k:len(v) for k,v in f.items()}!={'train':2048,'dev':256}:raise ValueError('fixed OWT revision/pool')
    if {d['normalized_sha256'] for d in docs['train']}&{d['normalized_sha256'] for d in docs['dev']}:raise ValueError('document split leak')
    seen=set()
    for panel in f:
        source={d['id']:' '.join(d['text'].split()) for d in docs[panel]}
        for x,target in zip(f[panel],y[panel]):
            if set(x)!={'id','prompt'} or x['prompt']!=target['target'] or not 48<=len(x['prompt'].encode('utf-8'))<=128:raise ValueError('sentence identity/capacity')
            h=support.hashlib.sha256(x['prompt'].encode('utf-8')).hexdigest()
            if h!=x['id'] or h in seen or x['prompt'] not in source[target['document_id']]:raise ValueError('sentence/source/split duplicate')
            seen.add(h)
            if target['target_ids']!=list(x['prompt'].encode('utf-8'))+[256] or target['document_EOF_claimed']:raise ValueError('sentence target/EOF')
    if p['generation_indices']!=[4*i for i in range(16)]:raise ValueError('fixed16distinctdevdocuments')
    return p,f,y,finite_orders(2048)

def load_model():
    source=torch.load(ROOT/SOURCES['language'][0],map_location='cpu',weights_only=True)
    c=Config(**source['core']);language=source['model'];del source
    typed=torch.load(ROOT/SOURCES['typed'][0],map_location='cpu',weights_only=True)['model']
    model=FuzzySentenceAE(c);mapping=model.warmstart(typed,language)
    return model,mapping

def evaluate(model,features,targets,budget,device):
    model.eval();modes=('normal','swap','zero');sums={m:[0.,0,0,0] for m in modes}
    stored={m:[] for m in modes};ids=[];memberships=[];views=[]
    amp=torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
    with torch.no_grad(),amp:
        for start in range(0,len(features),8):
            rows=features[start:start+8];ys=targets[start:start+8]
            tokens=pack_prompts([r['prompt'] for r in rows],128,device);target=pack_answers([r['target'] for r in ys],129,device);active=target!=PAD
            budget.charge('encode');meaning=model.extract(tokens)
            ids.append(target.cpu());memberships.append(meaning['memberships'].float().cpu());views.append(meaning['views'].float().cpu())
            for mode in modes:
                budget.charge('decoder');logits=model.teacher_logits(meaning,target,mode)
                nll=logits.logsumexp(-1)-logits.gather(-1,target[...,None]).squeeze(-1)
                a=sums[mode];a[0]+=float(nll[active].double().sum());a[1]+=int((logits.argmax(-1)[active]==target[active]).sum());a[2]+=int(active.sum());a[3]+=int(((target==256)&(logits.argmax(-1)==256)).sum())
                stored[mode].append(logits.half().cpu())
    result={m:{'NLL':v[0]/v[2],'byte_accuracy':v[1]/v[2],'correct':v[1],'symbols':v[2],'sentence_EOS_correct':v[3],'sentence_EOS_total':len(features)} for m,v in sums.items()}
    tensors={m:torch.cat(rows) for m,rows in stored.items()};tensors.update(target_ids=torch.cat(ids),memberships=torch.cat(memberships),views=torch.cat(views))
    return result,tensors

def generate(model,features,budget,device):
    model.eval();result={m:[] for m in ('normal','swap','zero')}
    amp=torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
    with torch.no_grad(),amp:
        for start in range(0,len(features),4):
            rows=features[start:start+4];tokens=pack_prompts([r['prompt'] for r in rows],128,device)
            budget.charge('encode');meaning=model.extract(tokens)
            for mode in result:
                generated=model.reconstruct(meaning,budget.charge,mode).cpu()
                for j,row in enumerate(rows):
                    decoded=decode_bytes(generated[j].tolist())
                    result[mode].append({'id':row['id'],'source':row['prompt'],'generated':decoded,
                        'reconstruction_exact':decoded['well_formed'] and decoded['text']==row['prompt']})
    return result

def run(path,output):
    p,f,y,orders=validate(path,output,__file__)
    if not p['authorized'] or p['deadline'] is None:raise PermissionError('unsealed')
    if (output/'progress.json').exists():raise FileExistsError('ONE invocation only')
    configure_determinism();torch.set_num_threads(4);torch.manual_seed(201);torch.cuda.manual_seed_all(201);random.seed(201);np.random.seed(201)
    budget=RecoveryBudget(p);engine=None;latest=None;phase='loading';start=time.time();guard=False
    def progress():
        io.atomic_json(output/'progress.json',{'phase':phase,'attempts':engine.attempts if engine else 0,'updates':engine.updates if engine else 0,
            'actual_calls':dict(budget.calls),'allocation_calls':dict(budget.allocation_calls),'native_pid':os.getpid(),'parent_pid':os.getppid(),
            'time_unix':time.time(),'elapsed':time.time()-start,'latest':latest})
    def commit(kind):
        nonlocal latest
        if kind=='emergency':
            if time.time()>=p['deadline']:raise TimeoutError('new owner end')
        else:budget.check();engine.verify_frozen()
        d=output/kind;d.mkdir(exist_ok=True);latest={**support.save_full(d,engine.export()),'relative_directory':kind};io.atomic_json(output/'latest.json',latest)
    budget.tick=progress;progress()
    try:
        budget.check()
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        model,mapping=load_model();engine=AEEngine(model,f['train'],y['train'],orders,budget,support.sha(path),'cuda')
        io.atomic_json(output/'specification.json',{'mapping':mapping,'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'trainable_names':[n for n,p in model.named_parameters() if p.requires_grad],
            'fuzzy_membership':'original nonexclusive topk sigmoid mu, probabilistic-sum union; rho normalized retrieval only',
            'decoder_input':'two extracted fuzzy views only, no raw token/context bypass','reasoning_or_diffusion_calls':0})
        commit('initial');phase='initial_reconstruction';progress()
        baseline,tensors=evaluate(model,f['dev'],y['dev'],budget,'cuda');torch.save(tensors,output/'initial_logits.pt');del tensors
        io.atomic_json(output/'initial_metrics.json',baseline);phase='training';progress()
        while engine.attempts<2048:
            if time.time()+90>=p['cutoff']:raise TimeoutError('final evaluation reserve')
            engine.step()
            if engine.attempts in (512,1024,1536):commit('model')
            progress()
        phase='final_reconstruction';progress();metrics,tensors=evaluate(model,f['dev'],y['dev'],budget,'cuda')
        torch.save(tensors,output/'final_logits.pt');del tensors;io.atomic_json(output/'final_metrics.json',metrics)
        phase='final_free_reconstruction';progress();raw=generate(model,[f['dev'][i] for i in p['generation_indices']],budget,'cuda')
        io.atomic_json(output/'final_generation.json',raw);engine.evaluations={'final_metrics':support.sha(output/'final_metrics.json'),'final_generation':support.sha(output/'final_generation.json')}
        commit('model')
        for relative,h in SOURCES.values():
            if support.sha(ROOT/relative)!=h:raise RuntimeError('source changed')
        if support.core_sha(ROOT)!=support.CORE_SHA:raise RuntimeError('core changed')
        result={'status':'OWT_FUZZY_RECONSTRUCTION_V12_COMPLETE','attempts':engine.attempts,'updates':engine.updates,'AMP_skips':engine.skips,
            'epochs':8,'exposure':engine.exposure,'actual_calls':budget.calls,'initial':baseline,'final':metrics,
            'generation':{m:{'rows':len(v),'exact':sum(x['reconstruction_exact'] for x in v),'UTF8':sum(x['generated']['utf8_valid'] for x in v),'EOS':sum(x['generated']['eos'] for x in v)} for m,v in raw.items()},
            'latest':latest,'seconds':time.time()-start,'reasoning_or_diffusion_training':False,'E2_final_accessed':False,
            'interpretation':'First OWT sentence -> original fuzzy-anchor extraction -> neural sentence reconstruction cycle. Reconstruction/content dependence tests information preservation, not universal semantic correctness or novel output-meaning inference.'}
        io.atomic_json(output/'result.json',result);phase='complete';progress();return 0
    except BaseException as exc:
        phase='stopped_or_error'
        if engine:
            try:commit('emergency')
            except BaseException as save:io.atomic_json(output/'save_error.json',{'error':repr(save)})
        io.atomic_json(output/'incident.json',{'error':repr(exc),'traceback':traceback.format_exc(),'actual_calls':budget.calls});progress();return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    if a.preflight:
        validate(a.plan,a.output,__file__);torch.set_num_threads(2);load_model()
        if torch.cuda.is_initialized():raise RuntimeError('CPU preflight CUDA initialized')
        print('CPU_PREFLIGHT_PASS');return 0
    return run(a.plan.resolve(),a.output.resolve())
if __name__=='__main__':raise SystemExit(main())
