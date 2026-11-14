"""NEW bounded hybrid decoding experiment after VP-only exact0/80 failure."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,math,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism,numpy_state
from research.text_diffusion_v1 import TextDiffusion,TextConfig,pack_answers,pack_prompts,decode_bytes,PAD
from research.text_context_decoder_v2 import ContextDecoder
from research.text_generation_data_v1 import check
import text_generation_support_v1 as support
import learning_io_v2 as io

LIMITS={'attempts':512,'batch':8,'wall':900,'reserve':90,'max_calls':5000,
        'max_output_bytes':8*1024**3,'start_free_bytes':36*1024**3,'min_gpu_free_mib':12288}
SCHEMA='fdt_context_decoder_v2'

def validate(plan_path,output,worker=None):
    plan=io.load_json(plan_path)
    if plan.get('schema')!=SCHEMA or type(plan.get('authorized')) is not bool or plan['limits']!=LIMITS:raise ValueError('V2schema/fixed limits')
    if Path(plan['output']).resolve()!=Path(output).resolve() or support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('output/core')
    for path,h in plan['files'].items():
        if support.sha(path)!=h:raise ValueError('pinned V2 dependency '+path)
    if worker and str(Path(worker).resolve()) not in plan['files']:raise ValueError('invoked worker not pinned')
    if plan.get('e2_final_accessed') is not False:raise ValueError('holdout forbidden')
    if plan['deadline'] is not None and plan['cutoff']!=plan['deadline']-90:raise ValueError('fixed reserve')
    data={n:io.load_json(Path(plan['data'])/(n+'.json')) for n in ('train','train_probe','dev_pairs','dev_wording')};check(data)
    selected=torch.randperm(len(data['train']),generator=torch.Generator().manual_seed(29))[:1024].tolist()
    return plan,data,selected

class Budget(support.Budget):
    def __init__(self,plan):super().__init__(plan);self.calls={'encode':0,'denoise':0,'decoder':0}

@torch.no_grad()
def proposal(generator,tokens,noise,charge):
    charge('encode');enc=generator.encode(tokens);z=noise.float()
    schedule=torch.linspace(1,0,17).tolist()
    for t,nxt in zip(schedule,schedule[1:]):
        charge('denoise');v=generator.denoise(z,z.new_full((len(z),),t),enc).float()
        a,s=(0.,1.) if t==1 else (math.cos(t*math.pi/2),math.sin(t*math.pi/2))
        clean=a*z-s*v;eps=s*z+a*v;z=math.cos(nxt*math.pi/2)*clean+math.sin(nxt*math.pi/2)*eps
    return enc,z

@torch.no_grad()
def cache_rows(generator,rows,budget,seed):
    prompts=[];masks=[];latents=[]
    for start in range(0,len(rows),8):
        batch=rows[start:start+8];tokens=pack_prompts([r['prompt'] for r in batch],160,'cuda')
        rng=torch.Generator(device='cuda').manual_seed(seed+start)
        z=torch.randn(len(batch),40,512,device='cuda',generator=rng)
        with torch.autocast('cuda',dtype=torch.float16):
            enc,z=proposal(generator,tokens,z,budget.charge)
        if not torch.isfinite(z).all():raise FloatingPointError('nonfinite cached VP state')
        prompts.append(enc[0].cpu().half());masks.append(enc[1].cpu());latents.append(z.cpu().half())
    return {'prompt':torch.cat(prompts),'mask':torch.cat(masks),'latent':torch.cat(latents)}

def context_slice(decoder,cache,indices,zero=False):
    return decoder.context(cache['prompt'][indices].cuda(),cache['mask'][indices].cuda(),cache['latent'][indices].cuda(),zero)

def evaluate(decoder,panels,caches,budget,zero=False):
    result={};decoder.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        for name,rows in panels.items():
            raw=[];exact=valid=eos=good=correct=total=0
            for start in range(0,len(rows),8):
                batch=rows[start:start+8];context=context_slice(decoder,caches[name],list(range(start,start+len(batch))),zero)
                ids=decoder.generate(context,budget.charge)
                targets=pack_answers([r['answer'] for r in batch],40,'cuda');mask=targets!=PAD
                correct+=int(((ids==targets)&mask).sum());total+=int(mask.sum())
                for j,row in enumerate(batch):
                    decoded=decode_bytes(ids[j].cpu().tolist());hit=decoded['well_formed'] and decoded['text']==row['answer']
                    exact+=int(hit);valid+=int(decoded['utf8_valid']);eos+=int(decoded['eos']);good+=int(decoded['well_formed'])
                    raw.append({**row,'generated':decoded,'exact_match':hit})
            result[name]={'rows':len(rows),'exact':exact,'exact_rate':exact/len(rows),'utf8_valid':valid,'eos':eos,'well_formed':good,
                          'byte_accuracy_including_eos':correct/total,'zero_vp_latent':zero,'raw':raw}
    return result

def run(plan_path,output):
    plan,data,selected=validate(plan_path,output,__file__)
    if not plan['authorized'] or plan['deadline'] is None:raise PermissionError('unsealed V2')
    if (output/'progress.json').exists():raise FileExistsError('V2already launched')
    budget=Budget(plan);budget.check();configure_determinism();torch.set_num_threads(4)
    torch.manual_seed(29);torch.cuda.manual_seed_all(29);random.seed(29);np.random.seed(29)
    attempts=updates=skips=cursor=0;phase='initializing';history=[];latest=None;decoder=optimizer=scaler=generator=None;started=time.time();guard=False
    def progress():
        io.atomic_json(output/'progress.json',{'phase':phase,'attempts':attempts,'updates':updates,'cursor':cursor,
            'actual_calls':budget.calls,'native_pid':os.getpid(),'parent_pid':os.getppid(),'time_unix':time.time(),
            'elapsed':time.time()-started,'latest':latest,'last_loss':history[-1]['loss'] if history else None})
    def payload():
        return {'schema':SCHEMA,'model':{'generator':{k:v.detach().cpu() for k,v in generator.state_dict().items()},
            'decoder':{k:v.detach().cpu() for k,v in decoder.state_dict().items()}},'optimizer':optimizer.state_dict(),
            'scaler':scaler.state_dict(),'attempts':attempts,'updates':updates,'cursor':cursor,'order':order,'selected_train_indices':selected,
            'history':history,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),
            'numpy_rng':numpy_state(),'scheduler':{'horizon':512,'next_attempt':attempts},'actual_calls':dict(budget.calls),
            'plan_sha256':support.sha(plan_path),'parent_generation_checkpoint_sha256':plan['parent_sha'],
            'frozen_generator_sha256':frozen_before,'cache_sha256':cache_sha,'layers':2,'length':40,'core':core_dict}
    progress()
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        parent=torch.load(plan['parent'],map_location='cpu',weights_only=True);core_dict=parent['specification']['core'];core=Config(**core_dict)
        generator=TextDiffusion(core);generator.load_state_dict(parent['model'],strict=True);del parent
        generator.requires_grad_(False);generator=generator.eval().cuda();frozen_before=support.state_sha(generator.state_dict())
        decoder=ContextDecoder(core)
        for i,block in enumerate(decoder.blocks):block.load_state_dict(generator.refiner[i].state_dict(),strict=True)
        decoder=decoder.cuda();optimizer=torch.optim.AdamW(decoder.parameters(),lr=3e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
        train=[data['train'][i] for i in selected];panels={k:data[k] for k in ('train_probe','dev_pairs','dev_wording')}
        phase='frozen_vp_cache';progress()
        caches={'train':cache_rows(generator,train,budget,29)}
        for name,rows in panels.items():caches[name]=cache_rows(generator,rows,budget,17);progress()
        cache_file=output/'frozen_context_cache.pt'
        with cache_file.open('xb') as f:torch.save(caches,f);f.flush();os.fsync(f.fileno())
        cache_sha=support.sha(cache_file)
        order=[];rng=torch.Generator().manual_seed(29)
        for _ in range(4):order+=torch.randperm(1024,generator=rng).tolist()
        io.atomic_json(output/'specification.json',{'frozen_generator_parameters':sum(p.numel() for p in generator.parameters()),
            'trainable_decoder_parameters':sum(p.numel() for p in decoder.parameters()),'architecture':'frozen VP proposals + learned causal byte decoder; raw prompt also provided',
            'training_pool_rows':1024,'prospective_epochs':4,'max_training_exposures':4096,'all_cache_no_gold':True,
            'train_probe_rows_in_decoder_pool':sum(r['id'] in {t['id'] for t in train} for r in panels['train_probe']),
            'controlled_limit':'zero latent retains raw prompt; no independent diffusion gain claimed without observed contribution'})
        latest=support.save_full(output,payload());phase='baseline_decoding';progress()
        baseline=evaluate(decoder,panels,caches,budget);io.atomic_json(output/'baseline_generation.json',baseline)
        phase='training';progress();decoder.train()
        while attempts<512:
            budget.check()
            if time.time()+90>=plan['cutoff']:raise TimeoutError('V2final reserve')
            indices=order[cursor:cursor+8]
            if len(indices)!=8:raise RuntimeError('four finite epochs exhausted')
            cursor+=8;attempts+=1
            context=context_slice(decoder,caches['train'],indices)
            ids=pack_answers([train[i]['answer'] for i in indices],40,'cuda')
            lr=3e-4*(.1+.9*(1+math.cos(math.pi*(attempts-1)/512))/2)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);budget.charge('decoder')
            with torch.autocast('cuda',dtype=torch.float16):loss=decoder.loss(ids,context)
            if not torch.isfinite(loss) or float(loss.detach())>100:raise FloatingPointError('decoder loss nonfinite/explosion')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            if any(p.grad is not None for p in generator.parameters()):raise RuntimeError('frozen VP gradients')
            norm=torch.nn.utils.clip_grad_norm_(decoder.parameters(),1.)
            scale=scaler.get_scale();scaler.step(optimizer);scaler.update();skip=scaler.get_scale()<scale
            skips+=int(skip);updates+=int(not skip)
            history.append({'attempt':attempts,'updates':updates,'loss':float(loss.detach()),'grad_norm':float(norm),'skip':skip,
                'rows':indices,'source_rows':[selected[i] for i in indices],'target_bytes':sum(len(train[i]['answer'].encode()) for i in indices)})
            if skips>=3 or (not skip and not math.isfinite(float(norm))):raise FloatingPointError('AMP/gradient stop')
            if attempts%128==0:
                if support.state_sha(generator.state_dict())!=frozen_before:raise RuntimeError('frozen VP changed')
                latest=support.save_full(output,payload())
            progress()
        phase='final_decoding';progress();final=evaluate(decoder,panels,caches,budget);io.atomic_json(output/'final_generation.json',final)
        zero=evaluate(decoder,panels,caches,budget,zero=True);io.atomic_json(output/'zero_latent_generation.json',zero)
        if support.state_sha(generator.state_dict())!=frozen_before or support.sha(plan['parent'])!=plan['parent_sha']:raise RuntimeError('frozen parent mutation')
        result={'status':'CONTEXT_DECODER_V2_COMPLETE','attempts':attempts,'updates':updates,'amp_skips':skips,'actual_calls':budget.calls,
            'new_training_rows':cursor,'unique_pool_rows':1024,'target_utf8_bytes':sum(r['target_bytes'] for r in history),
            'frozen_generator_unchanged':True,'parent_unchanged':True,'e2_final_accessed':False,'runtime_templates':False,
            'baseline':{k:{a:b for a,b in v.items() if a!='raw'} for k,v in baseline.items()},
            'final':{k:{a:b for a,b in v.items() if a!='raw'} for k,v in final.items()},
            'zero_latent':{k:{a:b for a,b in v.items() if a!='raw'} for k,v in zero.items()},
            'latest':latest,'cache_sha256':cache_sha,'seconds':time.time()-started,
            'interpretation':'Hybrid causal decoder receives raw prompt and fixed VP proposal. General LM ability and a diffusion contribution are separate claims.'}
        io.atomic_json(output/'result.json',result);phase='complete';progress();return 0
    except BaseException as exc:
        phase='stopped_or_error'
        if latest is not None and latest['attempts']!=attempts:
            try:latest=support.save_full(output,payload())
            except BaseException as save_exc:io.atomic_json(output/'save_error.json',{'error':repr(save_exc)})
        io.atomic_json(output/'incident.json',{'error':repr(exc),'traceback':traceback.format_exc(),'attempts':attempts,'updates':updates,'calls':budget.calls});progress();return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    if args.preflight:
        torch.set_num_threads(2);plan,_,_=validate(args.plan,args.output,__file__)
        p=torch.load(plan['parent'],map_location='cpu',weights_only=True)
        generator=TextDiffusion(Config(**p['specification']['core']));generator.load_state_dict(p['model'],strict=True)
        print('CPU_PREFLIGHT_PASS');return 0
    return run(args.plan.resolve(),args.output.resolve())
if __name__=='__main__':raise SystemExit(main())
