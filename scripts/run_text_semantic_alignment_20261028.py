"""NEW semantic binding stage. Frozen VP cache reused as data, not old budget."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,math,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism,numpy_state
from research.text_diffusion_v1 import pack_answers,decode_bytes,PAD
from research.text_semantic_reader_v3 import SemanticGenerator
from research.text_semantic_targets_v3 import build_metadata
from research.text_generation_data_v1 import check
import text_generation_support_v1 as support
import learning_io_v2 as io

SCHEMA='fdt_text_semantic_alignment_v3'
LIMITS={'attempts':512,'batch':8,'wall':900,'reserve':90,'max_calls':2500,
        'max_output_bytes':8*1024**3,'start_free_bytes':36*1024**3,'min_gpu_free_mib':12288}

def validate(plan_path,output,worker=None):
    p=io.load_json(plan_path)
    if p.get('schema')!=SCHEMA or type(p.get('authorized')) is not bool or p['limits']!=LIMITS:raise ValueError('V3schema/caps')
    if Path(p['output']).resolve()!=Path(output).resolve() or support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('output/core')
    for file,h in p['files'].items():
        if support.sha(file)!=h:raise ValueError('V3dependency changed '+file)
    if worker and str(Path(worker).resolve()) not in p['files']:raise ValueError('worker unpinned')
    if p.get('e2_final_accessed') is not False:raise ValueError('E2final forbidden')
    if p['deadline'] is not None and p['cutoff']!=p['deadline']-90:raise ValueError('reserve')
    data={k:io.load_json(Path(p['data'])/(k+'.json')) for k in ('train','train_probe','dev_pairs','dev_wording')};check(data)
    return p,data

class Budget(support.Budget):
    def __init__(self,p):super().__init__(p);self.calls={'reader':0,'decoder':0,'encode':0,'denoise':0}

def context(model,cache,indices,budget,zero=False):
    budget.charge('reader')
    return model.context(cache['prompt'][indices].cuda(),cache['mask'][indices].cuda(),cache['latent'][indices].cuda(),zero)

def evaluate(model,panels,caches,metadata,budget,zero=False):
    model.eval();result={}
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        for name,rows in panels.items():
            raw=[];correct=total=exact=good=0;heads=[0]*4
            for start in range(0,len(rows),8):
                batch=rows[start:start+8];indices=list(range(start,start+len(batch)))
                enc,logits=context(model,caches[name],indices,budget,zero);ids=model.decoder.generate(enc,budget.charge)
                y=torch.tensor(metadata[name][start:start+len(batch)],device='cuda')
                pred=torch.stack([v.argmax(-1) for v in logits],-1)
                for i in range(4):heads[i]+=int((pred[:,i]==y[:,i]).sum())
                target=pack_answers([r['answer'] for r in batch],40,'cuda');mask=target!=PAD
                correct+=int(((ids==target)&mask).sum());total+=int(mask.sum())
                for j,row in enumerate(batch):
                    d=decode_bytes(ids[j].cpu().tolist());hit=d['well_formed'] and d['text']==row['answer']
                    exact+=int(hit);good+=int(d['well_formed'])
                    raw.append({**row,'generated':d,'exact_match':hit,'reader_pred':pred[j].cpu().tolist(),
                        'reader_gold':y[j].cpu().tolist(),'reader_logits':[v[j].float().cpu().tolist() for v in logits]})
            result[name]={'rows':len(rows),'exact':exact,'exact_rate':exact/len(rows),'well_formed':good,
                'byte_accuracy_including_eos':correct/total,'reader_accuracy':[n/len(rows) for n in heads],
                'zero_vp_latent':zero,'raw':raw}
    return result

def run(plan_path,output):
    plan,data=validate(plan_path,output,__file__)
    if not plan['authorized'] or plan['deadline'] is None:raise PermissionError('unsealed V3')
    if (output/'progress.json').exists():raise FileExistsError('V3already executed')
    budget=Budget(plan);budget.check();configure_determinism();torch.set_num_threads(4)
    torch.manual_seed(30);torch.cuda.manual_seed_all(30);random.seed(30);np.random.seed(30)
    attempts=updates=cursor=skips=0;history=[];phase='initializing';latest=None;model=optimizer=scaler=None;started=time.time();guard=False
    def progress():io.atomic_json(output/'progress.json',{'phase':phase,'attempts':attempts,'updates':updates,'cursor':cursor,
        'actual_calls':budget.calls,'native_pid':os.getpid(),'parent_pid':os.getppid(),'time_unix':time.time(),'elapsed':time.time()-started,'latest':latest})
    def payload():return {'schema':SCHEMA,'model':{'generator':frozen,'semantic_generator':{k:v.detach().cpu() for k,v in model.state_dict().items()}},
        'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'attempts':attempts,'updates':updates,'cursor':cursor,
        'order':order,'selected_train_indices':selected,'history':history,'torch_rng':torch.get_rng_state(),
        'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),'numpy_rng':numpy_state(),
        'scheduler':{'horizon':512,'next_attempt':attempts},'actual_calls':dict(budget.calls),'plan_sha256':support.sha(plan_path),
        'parent_sha256':plan['parent_sha'],'cache_sha256':plan['cache_sha'],'frozen_sha256':frozen_sha,'core':core_dict}
    progress()
    try:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        parent=torch.load(plan['parent'],map_location='cpu',weights_only=True);core_dict=parent['core'];core=Config(**core_dict)
        frozen=parent['model']['generator'];frozen_sha=support.state_sha(frozen);selected=parent['selected_train_indices']
        model=SemanticGenerator(core);model.decoder.load_state_dict(parent['model']['decoder'],strict=True)
        for i,block in enumerate(model.reader.blocks):
            prefix=f'refiner.{i}.';block.load_state_dict({k[len(prefix):]:v for k,v in frozen.items() if k.startswith(prefix)},strict=True)
        del parent
        caches=torch.load(plan['cache'],map_location='cpu',weights_only=True)
        train=[data['train'][i] for i in selected];panels={k:data[k] for k in ('train_probe','dev_pairs','dev_wording')}
        metadata=io.load_json(plan['metadata'])
        if metadata!=build_metadata(data,selected):raise ValueError('metadata changed/query target mismatch')
        model=model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01);scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
        order=[];g=torch.Generator().manual_seed(30)
        for _ in range(4):order+=torch.randperm(1024,generator=g).tolist()
        io.atomic_json(output/'specification.json',{'frozen_VP_parameters':sum(v.numel() for k,v in frozen.items() if k not in ('codec','frequencies')),
            'trainable_reader_decoder_parameters':sum(p.numel() for p in model.parameters()),
            'inference_roles':'neural predictions from raw-prompt context; no externally supplied query roles',
            'decoder_inputs':'4soft predicted factors + frozen VP latent; gold restricted to train loss/posthoc scoring',
            'four_targets':['query_subject_AtoH','query_object_AtoH','greater_less_unknown','en_ko'],
            'runtime_regex_templates_or_oracle':False,'finite_epochs':4,'cache_reuse_extra_VP_calls':0})
        latest=support.save_full(output,payload());phase='baseline';progress()
        baseline=evaluate(model,panels,caches,metadata,budget);io.atomic_json(output/'baseline_generation.json',baseline)
        phase='training';progress();model.train()
        while attempts<512:
            budget.check()
            if time.time()+90>=plan['cutoff']:raise TimeoutError('V3final reserve')
            indices=order[cursor:cursor+8]
            if len(indices)!=8:raise RuntimeError('fixed four epochs exhausted')
            cursor+=8;attempts+=1;optimizer.zero_grad(set_to_none=True)
            lr=3e-4*(.1+.9*(1+math.cos(math.pi*(attempts-1)/512))/2)
            for group in optimizer.param_groups:group['lr']=lr
            ids=pack_answers([train[i]['answer'] for i in indices],40,'cuda');gold=torch.tensor([metadata['train'][i] for i in indices],device='cuda')
            with torch.autocast('cuda',dtype=torch.float16):
                enc,logits=context(model,caches['train'],indices,budget);budget.charge('decoder');loss,ar,aux=model.loss(ids,enc,logits,gold)
            if not torch.isfinite(loss) or float(loss.detach())>100:raise FloatingPointError('V3loss numeric/explosion')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            before=scaler.get_scale();scaler.step(optimizer);scaler.update();skip=scaler.get_scale()<before
            skips+=int(skip);updates+=int(not skip)
            history.append({'attempt':attempts,'update':updates,'rows':indices,'loss':float(loss.detach()),'ar_ce':float(ar.detach()),
                'aux_ce':float(aux.detach()),'grad_norm':float(norm),'skip':skip,
                'target_bytes':sum(len(train[i]['answer'].encode()) for i in indices)})
            if skips>=3 or (not skip and not math.isfinite(float(norm))):raise FloatingPointError('AMP/gradient stop')
            if attempts%128==0:latest=support.save_full(output,payload())
            progress()
        phase='final_generation';progress();final=evaluate(model,panels,caches,metadata,budget);io.atomic_json(output/'final_generation.json',final)
        zero=evaluate(model,panels,caches,metadata,budget,True);io.atomic_json(output/'zero_latent_generation.json',zero)
        if support.state_sha(frozen)!=frozen_sha or support.sha(plan['parent'])!=plan['parent_sha'] or support.sha(plan['cache'])!=plan['cache_sha']:raise RuntimeError('frozen source/cache changed')
        result={'status':'TEXT_SEMANTIC_ALIGNMENT_V3_COMPLETE','attempts':attempts,'updates':updates,'amp_skips':skips,
            'actual_calls':budget.calls,'train_exposures':cursor,'unique_pool_rows':1024,'target_utf8_bytes':sum(h['target_bytes'] for h in history),
            'baseline':{k:{a:b for a,b in v.items() if a!='raw'} for k,v in baseline.items()},
            'final':{k:{a:b for a,b in v.items() if a!='raw'} for k,v in final.items()},
            'zero_latent':{k:{a:b for a,b in v.items() if a!='raw'} for k,v in zero.items()},
            'frozen_source_and_cache_unchanged':True,'E2_final_accessed':False,'runtime_templates':False,'latest':latest,
            'seconds':time.time()-started,'interpretation':'Learned structured semantic bottleneck on A-H short synthetic prompts; not broad NER, pure diffusion gain, or general LM.'}
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
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    if a.preflight:
        torch.set_num_threads(2);plan,data=validate(a.plan,a.output,__file__);parent=torch.load(plan['parent'],map_location='cpu',weights_only=True)
        m=SemanticGenerator(Config(**parent['core']));m.decoder.load_state_dict(parent['model']['decoder'],strict=True)
        if build_metadata(data,parent['selected_train_indices'])!=io.load_json(plan['metadata']):raise ValueError('CPU metadata')
        print('CPU_PREFLIGHT_PASS');return 0
    return run(a.plan.resolve(),a.output.resolve())
if __name__=='__main__':raise SystemExit(main())
