"""ONE prospective natural-continuation trial; root-only, no automatic resume."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,hashlib,json,random,sys,time,traceback
from pathlib import Path
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from torch.nn import functional as F
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.text_diffusion_v1 import pack_prompts,decode_bytes,PAD
from research.text_context_decoder_v2 import BOS
from research.text_language_continuation_v5 import LanguageContinuation
from research.text_corpus_contract_v5 import validate_rows
from text_language_engine_v5 import LanguageEngine,finite_order,row_noise
import text_generation_support_v1 as support
import learning_io_v2 as io

SCHEMA='fdt_text_language_continuation_v5'
LIMITS={'attempts':409,'batch':8,'micro':4,'wall':900,'reserve':120,'max_calls':23000,
    'max_output_bytes':8*1024**3,'start_free_bytes':40*1024**3,'min_gpu_free_mib':12288}
PARENT_SHA='5abf9351ddd12509e3622783dc95373f0ac1aeb1fef1a1feb5bb241c44a608df'
MANIFEST_SHA='5b0ff59fd3985ab825085d0f3e4f5f7312baf5a22b137b62ec4b0b23f66f3f6e'


def generation_indices(targets):
    # The first16 document IDs, final window each, fixed before model outputs.
    endings={t['document_id']:i for i,t in enumerate(targets) if t['source_eof']}
    return [endings[key] for key in sorted(endings)[:16]]


def cost_contract():
    return {'training_microbatches':818,'training':818*19,'NLL':2*111*18,
        'generation':2*4*((1+16+129)+129),'total_max':818*19+2*111*18+2*4*(17+258)}


def validate(plan_path,output,worker=None):
    p=io.load_json(plan_path)
    if p.get('schema')!=SCHEMA or type(p.get('authorized')) is not bool or p.get('limits')!=LIMITS:raise ValueError('schema/caps')
    if Path(p['root']).resolve()!=ROOT or Path(p['output']).resolve()!=Path(output).resolve():raise ValueError('root/output')
    if p.get('parent_sha')!=PARENT_SHA or p.get('manifest_sha')!=MANIFEST_SHA or support.core_sha(ROOT)!=support.CORE_SHA:raise ValueError('source/core')
    if p.get('E2_final_accessed') is not False or p.get('epochs')!=1 or p.get('cost_contract')!=cost_contract():raise ValueError('data/holdout/epoch/cost')
    required={str(Path(p['parent']).resolve()):PARENT_SHA,str((Path(p['artifact'])/'manifest.json').resolve()):MANIFEST_SHA}
    if any(p['files'].get(n)!=h for n,h in required.items()):raise ValueError('missing source binding')
    for path,h in p['files'].items():
        if support.sha(path)!=h:raise ValueError('pinned file changed '+path)
    if worker and str(Path(worker).resolve()) not in p['files']:raise ValueError('worker not pinned')
    if p.get('deadline') is not None:
        if p['deadline']-p['allocation_start']!=900 or p['cutoff']!=p['deadline']-120:raise ValueError('immutable clock')
    a=Path(p['artifact']);manifest=io.load_json(a/'manifest.json')
    for name,h in manifest['files'].items():
        if p['files'].get(str((a/name).resolve()))!=h:raise ValueError('artifact binding '+name)
    features=io.load_json(a/'features.json');targets=io.load_json(a/'targets.json');docs=io.load_json(a/'documents.json')
    if {n:len(v) for n,v in features.items()}!={'train':3271,'dev':444}:raise ValueError('exact pool')
    validate_rows(docs,features,targets)
    order=finite_order(3271)
    if p.get('order')!=order or p.get('generation_indices')!=generation_indices(targets['dev']):raise ValueError('prospective order/panel')
    return p,features,targets,order


class Budget(support.Budget):
    def __init__(self,plan):
        super().__init__(plan);self.calls={'encode':0,'denoise':0,'decoder':0};self.tick=None;self.last_tick=0.
    def charge(self,kind):
        super().charge(kind)
        if self.tick and time.time()-self.last_tick>=5:
            self.last_tick=time.time();self.tick()


def load_source(plan):
    parent=torch.load(plan['parent'],map_location='cpu',weights_only=True)
    model=LanguageContinuation(Config(**parent['core']))
    mapping=model.warmstart(parent['model']['generator'],parent['model']['span_generator'])
    return model,parent['core'],mapping


def dump_tensors(path,value):
    with Path(path).open('xb') as f:torch.save(value,f);f.flush();os.fsync(f.fileno())
    return {'path':str(Path(path).name),'sha256':support.sha(path)}


@torch.no_grad()
def evaluate_nll(model,features,targets,budget,device):
    model.eval();all_logits=[];all_ids=[];raw=[];correct=total=eof_correct=eof_count=0;ce=0.
    for start in range(0,len(features),4):
        rows=features[start:start+4];ys=targets[start:start+4]
        tokens=pack_prompts([r['prompt'] for r in rows],model.policy.prompt_bytes,device)
        noise=row_noise(rows,model.policy.target_bytes_with_eos,model.core_config.z_dim,17,device)
        amp=torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
        with amp:
            budget.charge('encode');enc=model.vp.encode(tokens);proposal=model.free_proposal(enc,noise,budget.charge)
            ids=torch.full((len(rows),model.policy.target_bytes_with_eos),PAD,device=device,dtype=torch.long)
            for j,y in enumerate(ys):ids[j,:len(y['target_ids'])]=torch.tensor(y['target_ids'],device=device)
            prefix=torch.cat((torch.full_like(ids[:,:1],BOS),ids[:,:-1]),1)
            context=model.decoder.context(enc[0],enc[1],proposal)
            budget.charge('decoder');logits=model.decoder(prefix,context).float()
        labels=ids.masked_fill(ids==PAD,-100);valid=labels!=-100
        token_loss=F.cross_entropy(logits.transpose(1,2),labels,reduction='none',ignore_index=-100)
        ce+=float(token_loss.double().sum());total+=int(valid.sum());pred=logits.argmax(-1)
        correct+=int(((pred==ids)&valid).sum())
        for j,y in enumerate(ys):
            if y['source_eof']:
                eof_count+=1;eof_correct+=int(pred[j,len(y['target_ids'])-1]==256)
            raw.append({'id':rows[j]['id'],'document_id':y['document_id'],'source_eof':y['source_eof'],
                'active':len(y['target_ids']),'predicted_ids':pred[j].cpu().tolist()})
        all_logits.append(logits.cpu());all_ids.append(ids.cpu())
    return {'rows':len(features),'symbols':total,'ce_sum':ce,'NLL':ce/total,'byte_accuracy':correct/total,
        'true_EOF':eof_count,'true_EOF_correct':eof_correct,'raw':raw}, {'logits':torch.cat(all_logits),'target_ids':torch.cat(all_ids)}


@torch.no_grad()
def free_generate(model,rows,budget,device):
    # Inference helper has no targets, factors, labels or oracle filters.
    model.eval();normal=[];zero=[]
    for start in range(0,len(rows),4):
        batch=rows[start:start+4];tokens=pack_prompts([r['prompt'] for r in batch],model.policy.prompt_bytes,device)
        noise=row_noise(batch,model.policy.target_bytes_with_eos,model.core_config.z_dim,17,device)
        amp=torch.autocast('cuda',dtype=torch.float16) if str(device).startswith('cuda') else nullcontext()
        with amp:
            budget.charge('encode');enc=model.vp.encode(tokens);proposal=model.free_proposal(enc,noise,budget.charge)
            for is_zero,destination in ((False,normal),(True,zero)):
                context=model.decoder.context(enc[0],enc[1],proposal,zero_latent=is_zero)
                generated=model.decoder.generate(context,budget.charge)
                for j,row in enumerate(batch):
                    destination.append({'id':row['id'],'prompt':row['prompt'],'zero_VP':is_zero,
                        'generated':decode_bytes(generated[j].cpu().tolist())})
    return {'normal':normal,'zero':zero}


def run(plan_path,output):
    plan,features,targets,order=validate(plan_path,output,__file__)
    if not plan['authorized'] or plan['deadline'] is None:raise PermissionError('unsealed')
    if (output/'progress.json').exists():raise FileExistsError('ONE launch only')
    configure_determinism();torch.set_num_threads(4);torch.manual_seed(61);torch.cuda.manual_seed_all(61);random.seed(61);np.random.seed(61)
    budget=Budget(plan);engine=None;latest=None;phase='initializing';guard=False;started=time.time();core=None
    def progress():
        io.atomic_json(output/'progress.json',{'phase':phase,'attempts':engine.attempts if engine else 0,
            'updates':engine.updates if engine else 0,'cursor':engine.cursor if engine else 0,
            'actual_calls':budget.calls,'native_pid':os.getpid(),'parent_pid':os.getppid(),
            'time_unix':time.time(),'elapsed':time.time()-started,'latest':latest})
    def commit(emergency=False):
        nonlocal latest
        if emergency:
            if time.time()>=plan['deadline']:raise TimeoutError('owner end reached')
        else:budget.check();engine.verify_frozen()
        directory=output/('emergency' if emergency else 'model');directory.mkdir(exist_ok=True)
        payload=engine.export();payload.update(core=core,policy=model.specification()['policy'],parent_sha256=plan['parent_sha'],
            artifact_manifest_sha256=plan['manifest_sha'],emergency_snapshot=emergency)
        latest={**support.save_full(directory,payload),'relative_directory':directory.name}
        io.atomic_json(output/'latest.json',latest)
    budget.tick=progress;progress()
    try:
        budget.check()
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard=True
        model,core,mapping=load_source(plan)
        engine=LanguageEngine(model,features['train'],targets['train'],order,budget,support.sha(plan_path))
        io.atomic_json(output/'specification.json',{**model.specification(),'mapping':mapping,'cost_contract':cost_contract()})
        commit()
        for label in ('baseline','final'):
            phase=label+'_NLL';progress()
            nll,tensors=evaluate_nll(model,features['dev'],targets['dev'],budget,'cuda')
            ref=dump_tensors(output/(label+'_NLL_logits.pt'),tensors);del tensors
            io.atomic_json(output/(label+'_NLL.json'),nll)
            engine.evaluations[label+'_NLL']={'json_sha256':support.sha(output/(label+'_NLL.json')),**ref}
            phase=label+'_generation';progress()
            generation=free_generate(model,[features['dev'][i] for i in plan['generation_indices']],budget,'cuda')
            io.atomic_json(output/(label+'_generation.json'),generation)
            engine.evaluations[label+'_generation']={'path':label+'_generation.json','sha256':support.sha(output/(label+'_generation.json'))}
            if label=='baseline':
                phase='training';progress()
                while engine.attempts<409:
                    if time.time()+120>=plan['cutoff']:raise TimeoutError('final evaluation reserve')
                    engine.step()
                    if engine.attempts in (1,128,256,384):commit()
                    progress()
            else:final_nll=nll
        commit();engine.verify_frozen()
        if support.sha(plan['parent'])!=PARENT_SHA or support.core_sha(ROOT)!=support.CORE_SHA:raise RuntimeError('source/core changed')
        baseline=io.load_json(output/'baseline_NLL.json')
        result={'status':'TEXT_LANGUAGE_CONTINUATION_V5_COMPLETE','attempts':engine.attempts,'updates':engine.updates,
            'AMP_skips':engine.skips,'actual_calls':budget.calls,'exposure':engine.exposure,'baseline_NLL':baseline['NLL'],
            'final_NLL':final_nll['NLL'],'NLL_delta':final_nll['NLL']-baseline['NLL'],'latest':latest,
            'seconds':time.time()-started,'E2_final_accessed':False,'automatic_promotion':False,
            'interpretation':'Observed English synthetic document continuation; teacher-forced NLL and raw generation are separate.'}
        io.atomic_json(output/'result.json',result);phase='complete';progress();return 0
    except BaseException as exc:
        phase='stopped_or_error'
        if engine:
            try:commit(emergency=True)
            except BaseException as save_exc:io.atomic_json(output/'save_error.json',{'error':repr(save_exc)})
        io.atomic_json(output/'incident.json',{'error':repr(exc),'traceback':traceback.format_exc(),'calls':budget.calls});progress();return 2
    finally:
        if guard:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--preflight',action='store_true');a=parser.parse_args()
    if a.preflight:
        torch.set_num_threads(2);plan,_,_,_=validate(a.plan,a.output,__file__);load_source(plan)
        if torch.cuda.is_initialized():raise RuntimeError('CPU preflight initialized CUDA')
        print('CPU_PREFLIGHT_PASS');return 0
    return run(a.plan.resolve(),a.output.resolve())
if __name__=='__main__':raise SystemExit(main())
