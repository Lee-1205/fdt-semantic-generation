"""Root-only new bounded text VP-diffusion pilot. Old allocations never resumed."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,ctypes,json,math,random,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from torch.nn import functional as F
from fdt_local.model_masked import Config
from fdt_local.runner import configure_determinism
from research.text_diffusion_v1 import TextConfig,TextDiffusion,pack_prompts,pack_answers,decode_bytes,EOS,PAD
import text_generation_support_v1 as support
import learning_io_v2 as io

def evaluate(model,data,budget,seed=17,shuffle=False):
    result={}; device='cuda'; cfg=model.text_config; model.eval()
    with torch.no_grad():
        for name in ('train_probe','dev_pairs','dev_wording'):
            rows=data[name];raw=[]; bytecorrect=bytetotal=exact=valid=eos=0
            # Rotating the complete panel makes a real conditioning control.
            prompts=[r['prompt'] for r in rows]
            if shuffle: prompts=prompts[1:]+prompts[:1]
            for start in range(0,len(rows),8):
                batch=rows[start:start+8]; tokens=pack_prompts(prompts[start:start+8],cfg.prompt_bytes,device)
                g=torch.Generator(device=device).manual_seed(seed+start)
                noise=torch.randn((len(batch),cfg.answer_bytes,model.core_config.z_dim),device=device,generator=g)
                with torch.autocast('cuda',dtype=torch.float16): ids,_=model.generate(tokens,noise,budget.charge)
                target=pack_answers([r['answer'] for r in batch],cfg.answer_bytes,device)
                scoremask=target!=PAD
                bytecorrect+=int(((ids==target)&scoremask).sum());bytetotal+=int(scoremask.sum())
                for j,r in enumerate(batch):
                    decoded=decode_bytes(ids[j].cpu().tolist())
                    match=decoded['well_formed'] and decoded['text']==r['answer']
                    exact+=int(match);valid+=int(decoded['utf8_valid']);eos+=int(decoded['eos'])
                    raw.append({**r,'actual_prompt':prompts[start+j],'generated':decoded,'exact_match':match})
            result[name]={'rows':len(rows),'exact':exact,'exact_rate':exact/len(rows),
                          'byte_accuracy_including_eos':bytecorrect/bytetotal,'utf8_valid':valid,'eos':eos,
                          'seed':seed,'shuffled_prompts':shuffle,'raw':raw}
    return result

def run(plan_path,output):
    plan,data,review=support.validate(plan_path,output,__file__)
    if not plan['authorized'] or plan['deadline'] is None: raise PermissionError('unsealed CPU draft')
    budget=support.Budget(plan);budget.check()
    if (output/'progress.json').exists(): raise FileExistsError('one launch only')
    configure_determinism();torch.set_num_threads(4)
    torch.manual_seed(28);torch.cuda.manual_seed_all(28);random.seed(28);np.random.seed(28)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    attempts=updates=cursor=skips=0;history=[]; latest=None;model=optimizer=scaler=None
    phase='initializing';start=time.time()
    def progress():
        io.atomic_json(output/'progress.json',{'phase':phase,'attempts':attempts,'updates':updates,'cursor':cursor,
           'actual_calls':budget.calls,'native_pid':os.getpid(),'parent_pid':os.getppid(),
           'time_unix':time.time(),'elapsed':time.time()-start,'latest':latest,
           'last_loss':history[-1] if history else None})
    progress()
    def payload():
        return {'schema':support.SCHEMA,'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'attempts':attempts,'updates':updates,
                'cursor':cursor,'order':order,'history':history,'noise_rng':noise_rng.get_state(),
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'python_rng':random.getstate(),
                'numpy_rng':{'name':np.random.get_state()[0],'keys':np.random.get_state()[1].tolist(),
                             'position':np.random.get_state()[2],'gaussian':np.random.get_state()[3],'cached':np.random.get_state()[4]},
                'scheduler':{'horizon':256,'next_attempt':attempts},'actual_calls':dict(budget.calls),
                'plan_sha256':support.sha(plan_path),'specification':model.specification(),'source_sha256':plan['source_sha']}
    guard_set=False
    try:
        if os.name=='nt': ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);guard_set=True
        source=torch.load(plan['source'],map_location='cpu',weights_only=True)
        core=Config(**source['contract']['config']);tc=TextConfig()
        model=TextDiffusion(core,tc);imports=model.import_typed(source['model']);del source
        frozen_before=support.state_sha(model.language.state_dict()); model=model.cuda()
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=1024.)
        noise_rng=torch.Generator(device='cuda').manual_seed(28)
        order=torch.randperm(len(data['train']),generator=torch.Generator().manual_seed(28)).tolist()
        io.atomic_json(output/'specification.json',{**model.specification(),'imported_elements':imports,'data_review':review})
        latest=support.save_full(output,payload());phase='baseline_generation';progress()
        baseline=evaluate(model,data,budget);io.atomic_json(output/'baseline_generation.json',baseline)
        phase='training';progress(); model.train()
        while attempts<256:
            budget.check()
            if time.time()+60>=plan['cutoff']: raise TimeoutError('final checkpoint/evaluation reserve')
            indices=order[cursor:cursor+8]
            if len(indices)!=8: raise RuntimeError('finite epoch exhausted; no reset')
            rows=[data['train'][i] for i in indices];cursor+=8;attempts+=1
            tokens=pack_prompts([r['prompt'] for r in rows],tc.prompt_bytes,'cuda')
            ids=pack_answers([r['answer'] for r in rows],tc.answer_bytes,'cuda')
            t=.01+.98*torch.rand(len(rows),device='cuda',generator=noise_rng)
            noise=torch.randn((len(rows),tc.answer_bytes,core.z_dim),device='cuda',generator=noise_rng)
            lr=1e-4*(.1+.9*(1+math.cos(math.pi*(attempts-1)/256))/2)
            for group in optimizer.param_groups: group['lr']=lr
            optimizer.zero_grad(set_to_none=True);budget.charge('encode')
            with torch.autocast('cuda',dtype=torch.float16):
                enc=model.encode(tokens);budget.charge('denoise');loss,parts=model.objective(enc,ids,t,noise)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            if any(p.grad is not None for p in model.language.parameters()): raise RuntimeError('frozen language gradient')
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            oldscale=scaler.get_scale();scaler.step(optimizer);scaler.update()
            skipped=scaler.get_scale()<oldscale
            if skipped: skips+=1
            else: updates+=1
            history.append({'attempt':attempts,'update':updates,'loss':float(loss),'grad_norm':float(norm),
                'amp_skipped':skipped,'lr':lr,'rows':indices,
                'prompt_bytes':sum(len(r['prompt'].encode()) for r in rows),
                'answer_bytes':sum(len(r['answer'].encode()) for r in rows),
                **{k:float(v) for k,v in parts.items()}})
            if skipped and skips>=3: raise FloatingPointError('three AMP skips; attempts consumed')
            if not skipped and (not math.isfinite(float(norm)) or float(loss)>100): raise FloatingPointError('gradient/loss explosion')
            if attempts%64==0:
                if support.state_sha(model.language.state_dict())!=frozen_before: raise RuntimeError('frozen language changed')
                latest=support.save_full(output,payload())
            progress()
        phase='final_generation';progress()
        final=evaluate(model,data,budget);io.atomic_json(output/'final_generation.json',final)
        shuffled=evaluate(model,data,budget,shuffle=True);io.atomic_json(output/'shuffled_prompt_generation.json',shuffled)
        if support.state_sha(model.language.state_dict())!=frozen_before: raise RuntimeError('frozen language changed')
        if support.sha(plan['source'])!=plan['source_sha'] or support.core_sha(ROOT)!=support.CORE_SHA: raise RuntimeError('source/core changed')
        result={'status':'BOUNDED_TEXT_GENERATION_PILOT_COMPLETE','attempts':attempts,'updates':updates,'amp_skips':skips,
            'actual_calls':budget.calls,'max_calls':1300,'optimizer_new_branch_only':True,'original_trial_resumed':False,
            'training_rows':cursor,'unique_train_rows':len(set(order[:cursor])),
            'prompt_utf8_bytes':sum(x['prompt_bytes'] for x in history),'target_utf8_bytes':sum(x['answer_bytes'] for x in history),
            'frozen_language_unchanged':True,'source_checkpoint_unchanged':True,'e2_final_accessed':False,
            'baseline':{k:{a:b for a,b in v.items() if a!='raw'} for k,v in baseline.items()},
            'final':{k:{a:b for a,b in v.items() if a!='raw'} for k,v in final.items()},
            'shuffled':{k:{a:b for a,b in v.items() if a!='raw'} for k,v in shuffled.items()},
            'checkpoint':latest,'seconds':time.time()-start,'interpretation':'Authored synthetic short-text pilot; not general language competence or an automatic production route.'}
        io.atomic_json(output/'result.json',result);phase='complete';progress();return 0
    except BaseException as exc:
        phase='stopped_or_error'
        if model is not None and optimizer is not None and scaler is not None:
            try:
                if latest is None or latest['attempts']!=attempts: latest=support.save_full(output,payload())
            except BaseException as save_exc: io.atomic_json(output/'save_error.json',{'error':repr(save_exc)})
        io.atomic_json(output/'incident.json',{'error':repr(exc),'traceback':traceback.format_exc(),'attempts':attempts,'updates':updates,'calls':budget.calls})
        progress();return 2
    finally:
        if guard_set: ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--preflight',action='store_true');args=ap.parse_args()
    if args.preflight:
        torch.set_num_threads(2)
        plan,_,_=support.validate(args.plan,args.output,__file__)
        source=torch.load(plan['source'],map_location='cpu',weights_only=True)
        if source['attempts']!=9 or source['updates']!=9: raise ValueError('partial9 provenance')
        model=TextDiffusion(Config(**source['contract']['config'])); model.import_typed(source['model'])
        if any(p.requires_grad for p in model.language.parameters()):raise ValueError('language freeze')
        del model,source
        print('CPU_PREFLIGHT_PASS');return 0
    return run(args.plan.resolve(),args.output.resolve())
if __name__=='__main__': raise SystemExit(main())
