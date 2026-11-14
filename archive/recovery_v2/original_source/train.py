"""Finite, resumable T4 experiments with pre-verifier neural evaluation."""
from dataclasses import asdict,dataclass
import argparse,gc,hashlib,json,math,os,pathlib,time
import torch
from torch.nn import functional as F
from model import Config,FDT100
from data import VERSION,LABELS,collate,make_split,manifest

PHASES=('grounding','semantic','diffusion')


@dataclass(frozen=True)
class Plan:
    seed:int=19
    phase_steps:tuple=(480,320,640)
    micro_batch:int=4
    accumulation:int=2
    save_every:int=100
    log_every:int=25
    train_graphs:int=4096
    dev_graphs:int=96
    test_graphs:int=128
    eval_seed:int=17
    encoder_lr:float=3e-5
    ground_lr:float=1e-3
    semantic_lr:float=3e-4
    diffusion_lr:float=3e-4
    ground_gate:float=.9
    semantic_gate:float=.8


def digest_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def source_digest():
    root=pathlib.Path(__file__).parent
    h=hashlib.sha256()
    for name in ('model.py','data.py','train.py','runtime.py'):
        h.update(name.encode());h.update((root/name).read_bytes())
    return h.hexdigest()


def write_json(path,value):
    p=pathlib.Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    os.replace(temp,p)


def masked_ce(logits,labels,mask,weights=None):
    loss=F.cross_entropy(logits.float().reshape(-1,logits.shape[-1]),labels.flatten(),weight=weights,reduction='none').reshape_as(labels)
    denominator=mask.sum() if weights is None else (mask*weights[labels]).sum()
    return (loss*mask).sum()/denominator.clamp_min(1)


def masked_mse(a,b,mask):
    return ((a.float()-b.float()).square()*mask[...,None]).sum()/(mask.sum()*a.shape[-1]).clamp_min(1)


def diffusion_loss(model,enc,target,labels,mask,generator,weights=None):
    b=len(target)
    t=torch.rand(b,device=target.device,generator=generator)
    t[:max(1,b//2)]=1.
    noise=torch.randn(target.shape,device=target.device,generator=generator)*mask[...,None]
    a=torch.cos(t*math.pi/2)[:,None,None];s=torch.sin(t*math.pi/2)[:,None,None]
    # Exact t=1 boundary, including FP32, avoids tiny signed cosine residue.
    a=torch.where(t[:,None,None]==1,torch.zeros_like(a),a)
    zt=a*target+s*noise;v_target=a*noise-s*target
    v=model.denoise(zt,t,enc).float();clean=a*zt-s*v
    lv=masked_mse(v,v_target,mask);lz=masked_mse(clean,target,mask)
    ce=masked_ce(model.readout(clean),labels,mask,weights)
    return lv+.25*lz+.5*ce,{'v_mse':lv.detach(),'clean_mse':lz.detach(),'diff_ce':ce.detach()}


def compute_metrics(pred,labels,mask,hops):
    valid=mask.bool();p=pred[valid].cpu();y=labels[valid].cpu();h=hops[valid].cpu()
    conf=torch.zeros(4,4,dtype=torch.int64)
    for a,b in zip(y.tolist(),p.tolist()):conf[a,b]+=1
    f1=[]
    for i in range(4):
        tp=conf[i,i].item();den=conf[i].sum().item()+conf[:,i].sum().item()
        f1.append(2*tp/den if den else 0.)
    result={'queries':len(y),'accuracy':float((p==y).float().mean()),'macro_f1':sum(f1)/4,
            'confusion':conf.tolist(),'class_f1':dict(zip(LABELS,f1))}
    for label,select in [('direct',(h==1)),('transitive',(h>=2)),('long_chain',(h>=3)),('unknown',(y==0)),('conflict',(y==3))]:
        result[label]={'count':int(select.sum()),'accuracy':float((p[select]==y[select]).float().mean()) if select.any() else None}
    return result


@torch.no_grad()
def evaluate(model,items,device,batch_size=4,seeds=(17,),full=True):
    model.eval();c=model.c
    predictions={k:[] for k in ('initializer','diffusion','fuzzy_removed','corrupted_initializer','repaired_latent')}
    labels=[];masks=[];hops=[];mse_before=[];mse_after=[];ground_correct=ground_total=0
    for start in range(0,len(items),batch_size):
        feat,gold,_=collate(items[start:start+batch_size],c,device)
        with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
            enc=model.encode(feat);oracle=model.target(feat,gold['labels'])
            ground_pred=enc['ground'].argmax(-1)
            ground_correct+=int(((ground_pred==gold['ground'])*feat['fact_mask']).sum())
            ground_total+=int(feat['fact_mask'].sum())
            ablated=model.encode(feat,True) if full else None
            for seed in seeds:
                g=torch.Generator(device=device).manual_seed(seed+start)
                noise=torch.randn(enc['init'].shape,device=device,generator=g)
                predictions['initializer'].append(model.readout(enc['init']).argmax(-1).cpu())
                if full:
                    refined=model.rollout(enc,noise)
                    removed=model.rollout(ablated,noise,ablate=True)
                    # Separate paired controlled latent corruption; same input conditions.
                    t=.5;corrupt=math.cos(math.pi*t/2)*enc['init'].float()+math.sin(math.pi*t/2)*noise
                    repaired=model.rollout(enc,corrupt,(.5,.375,.25,.125,0.))
                    for key,z in [('diffusion',refined),('fuzzy_removed',removed),('corrupted_initializer',corrupt),('repaired_latent',repaired)]:
                        predictions[key].append(model.readout(z).argmax(-1).cpu())
                    mse_before.append(float(masked_mse(corrupt,oracle,feat['query_mask'])))
                    mse_after.append(float(masked_mse(repaired,oracle,feat['query_mask'])))
                labels.append(gold['labels'].cpu());masks.append(feat['query_mask'].cpu());hops.append(gold['hops'].cpu())
    y,m,h=map(torch.cat,(labels,masks,hops))
    result={'graphs':len(items),'noise_seeds':list(seeds),'ground_accuracy':ground_correct/max(1,ground_total),
            'neural_before_verifier':True,'queries_are_correlated_within_graph_and_seeds':True}
    for key,values in predictions.items():
        if values:result[key]=compute_metrics(torch.cat(values),y,m,h)
    if mse_before:
        result['controlled_latent_repair_mse']={'before':sum(mse_before)/len(mse_before),'after':sum(mse_after)/len(mse_after)}
    return result


class Trainer:
    def __init__(self,out,plan=Plan(),cfg=Config(),device='cuda',resume=False,allow_cpu_smoke=False,warm_start=None):
        if device!='cuda' and not allow_cpu_smoke:raise ValueError('학습은 CUDA 전용입니다. CPU fallback 없음.')
        if device=='cuda':
            if not torch.cuda.is_available():raise ValueError('CUDA 없음. T4 런타임으로 변경하세요.')
            if 'T4' not in torch.cuda.get_device_name():raise ValueError('사용자가 요청한 T4 GPU가 아닙니다.')
        self.out=pathlib.Path(out);self.out.mkdir(parents=True,exist_ok=True);self.plan=plan;self.cfg=cfg;self.device=device
        if not resume and (self.out/'latest.json').exists():raise ValueError('기존 run을 덮어쓸 수 없습니다.')
        torch.manual_seed(plan.seed)
        if device=='cuda':torch.cuda.manual_seed_all(plan.seed)
        self.model=FDT100(cfg).to(device)
        self.train_data=make_split('train',plan.train_graphs);self.dev=make_split('dev',plan.dev_graphs)
        self.data_manifest=manifest({'train':self.train_data,'dev':self.dev})
        self.weights=torch.tensor(self.data_manifest['splits']['train']['query_labels'],device=device,dtype=torch.float32).clamp_min(1).reciprocal()
        self.weights=(self.weights/self.weights.mean()).clamp(.25,4)
        self.ground_weights=torch.tensor(self.data_manifest['splits']['train']['ground_labels'],device=device,dtype=torch.float32).clamp_min(1).reciprocal()
        self.ground_weights=self.ground_weights/self.ground_weights.mean()
        self.parent=None
        self.phase=0;self.phase_step=0;self.updates=0;self.attempts=0;self.exposure=0;self.query_exposure=[0]*4;self.ground_exposure=[0]*4
        self.gates=[];self.history=[];self.gradient_audit={};self.elapsed=0.;self.optimizer=None;self.scaler=None
        self.sampler=torch.Generator().manual_seed(plan.seed+1);self.noise=torch.Generator(device=device).manual_seed(plan.seed+2)
        self.contract={'source':source_digest(),'config':asdict(cfg),'plan':asdict(plan),'data':self.data_manifest['fingerprint'],
                       'torch':str(torch.__version__),'device':device,'gpu':torch.cuda.get_device_name() if device=='cuda' else None}
        write_json(self.out/'data_manifest.json',self.data_manifest)
        self._optimizer()
        if warm_start is not None:
            if resume:raise ValueError('warm start와 exact resume는 동시에 사용할 수 없습니다.')
            self._warm_start(warm_start)
        if resume:self._restore()
        write_json(self.out/'model_manifest.json',self.model.counts())

    def _trainable_names(self,phase):
        if phase==0:prefixes=('language.','phrase_router.','ground_head.','anchors','prior')
        elif phase==1:prefixes=('language.','phrase_router.','ground_head.','anchors','prior','value.','condition.','role_project.','initializer.','initial_out.')
        else:prefixes=('slot_router.','memory_project.','rel_left','rel_right','rel_score.','rel_message.','time.','refiner.','refine_norm.','velocity.')
        return prefixes

    def _warm_start(self,metadata_path):
        meta_path=pathlib.Path(metadata_path);meta=json.loads(meta_path.read_text());name=meta['file']
        if pathlib.Path(name).name!=name:raise ValueError('unsafe parent checkpoint')
        path=meta_path.parent/name
        if digest_file(path)!=meta['sha256']:raise ValueError('parent checksum mismatch')
        old=meta['contract']
        if old['source']!='c144d731b43edf1a375a0289315c8d9ad7490cf83505db249ea361029d140add':raise ValueError('승인된 v0.3 parent source가 아닙니다.')
        if old['data']!=self.contract['data'] or old['config']!={k:v for k,v in asdict(self.cfg).items() if k!='meaning_views'}:raise ValueError('parent architecture/data mismatch')
        saved=torch.load(path,map_location='cpu',weights_only=True);new=self.model.state_dict();old_state=saved['model']
        for key,value in new.items():
            if key=='language.pool':value[0].copy_(old_state[key])
            elif key=='ground_head.weight':
                value[:,:self.cfg.value_dim].copy_(old_state[key]);value[:,self.cfg.value_dim:]*=.1
            else:
                if value.shape!=old_state[key].shape:raise ValueError('unexpected migration shape: '+key)
                value.copy_(old_state[key])
        self.model.load_state_dict(new)
        self.parent={'file':name,'sha256':meta['sha256'],'source':old['source'],'updates':saved['updates'],
                     'migration':'one view -> two learned fuzzy views; shared matching weights','optimizer_reset':True,'exact_resume':False}
        write_json(self.out/'warm_start.json',self.parent);del saved;gc.collect()

    def _optimizer(self):
        self.optimizer=None;gc.collect()
        prefixes=self._trainable_names(min(self.phase,2));encoder=[];rest=[]
        for name,p in self.model.named_parameters():
            p.grad=None;p.requires_grad_(name.startswith(prefixes))
            if p.requires_grad:(encoder if name.startswith('language.') else rest).append(p)
        rate=(self.plan.ground_lr,self.plan.semantic_lr,self.plan.diffusion_lr)[min(self.phase,2)]
        groups=[]
        if encoder:groups.append({'params':encoder,'lr':self.plan.encoder_lr,'base_lr':self.plan.encoder_lr})
        if rest:groups.append({'params':rest,'lr':rate,'base_lr':rate})
        self.optimizer=torch.optim.AdamW(groups,weight_decay=.01,foreach=False)
        self.scaler=torch.amp.GradScaler(self.device,enabled=self.device=='cuda',init_scale=1024.)

    def state(self):
        return {'phase':PHASES[self.phase] if self.phase<3 else 'complete','phase_step':self.phase_step,
                'optimizer_updates':self.updates,'optimizer_attempts':self.attempts,'input_utf8_bytes':self.exposure,
                'query_target_exposure':dict(zip(LABELS,self.query_exposure)),
                'ground_target_exposure':self.ground_exposure,'elapsed_seconds':self.elapsed,'gates':self.gates,
                'parameters':self.model.counts()['parameters'],'device':self.device,'gpu':self.contract['gpu'],
                'peak_gpu_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if self.device=='cuda' else None,
                'history':self.history,'gradient_audit':self.gradient_audit,'contract':self.contract}

    def save(self):
        generation=f'checkpoint_{self.updates:06d}_{self.phase}_{self.attempts:06d}'
        path=self.out/(generation+'.pt');temp=path.with_suffix('.pt.tmp')
        payload={'contract':self.contract,'model':self.model.state_dict(),'optimizer':self.optimizer.state_dict(),
                 'scaler':self.scaler.state_dict(),'phase':self.phase,'phase_step':self.phase_step,'updates':self.updates,
                 'attempts':self.attempts,'exposure':self.exposure,'query_exposure':self.query_exposure,'ground_exposure':self.ground_exposure,
                 'gates':self.gates,'history':self.history,'gradient_audit':self.gradient_audit,'elapsed':self.elapsed,'parent':self.parent,
                 'sampler_rng':self.sampler.get_state(),'noise_rng':self.noise.get_state(),'torch_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if self.device=='cuda' else []}
        torch.save(payload,temp);os.replace(temp,path)
        meta={'file':path.name,'sha256':digest_file(path),'updates':self.updates,'phase':self.phase,'contract':self.contract}
        write_json(path.with_suffix('.json'),meta);write_json(self.out/'latest.json',meta)
        write_json(self.out/'training_status.json',self.state())
        # Keep current and one prior valid generation; no growing checkpoint pile.
        files=sorted(self.out.glob('checkpoint_*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        for old in files[2:]:old.with_suffix('.pt').unlink(missing_ok=True);old.unlink()

    def _restore(self):
        generations=sorted(self.out.glob('checkpoint_*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        newest=self.out/'latest.json'
        candidates=([newest] if newest.exists() else [])+generations
        payload=None;seen=set()
        for meta_path in candidates:
            try:
                meta=json.loads(meta_path.read_text());name=meta['file']
                if pathlib.Path(name).name!=name or not name.startswith('checkpoint_'):raise ValueError('unsafe checkpoint')
                if name in seen:continue
                seen.add(name);path=self.out/name
                if digest_file(path)!=meta['sha256']:continue
                expected=json.loads(json.dumps(self.contract))
                if meta['contract']!=expected:raise ValueError('source/config/plan/data/environment mismatch; 새 run 필요')
                payload=torch.load(path,map_location='cpu',weights_only=True);break
            except (OSError,json.JSONDecodeError):continue
        if payload is None:raise ValueError('유효 checkpoint 없음')
        self.model.load_state_dict(payload['model'])
        self.parent=payload.get('parent')
        for name in ('phase','phase_step','updates','attempts','exposure','query_exposure','ground_exposure','gates','history','gradient_audit','elapsed'):
            setattr(self,name,payload[name])
        self._optimizer();self.optimizer.load_state_dict(payload['optimizer']);self.scaler.load_state_dict(payload['scaler'])
        self.sampler.set_state(payload['sampler_rng']);self.noise.set_state(payload['noise_rng']);torch.set_rng_state(payload['torch_rng'])
        if self.device=='cuda':torch.cuda.set_rng_state_all(payload['cuda_rng'])
        del payload;gc.collect()

    def _batch(self):
        ids=torch.randint(len(self.train_data),(self.plan.micro_batch,),generator=self.sampler).tolist()
        return collate([self.train_data[i] for i in ids],self.cfg,self.device)

    def _step(self):
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        losses={};new_bytes=0;new_query=[0]*4;new_ground=[0]*4
        for _ in range(self.plan.accumulation):
            feat,gold,exposure=self._batch();new_bytes+=exposure
            for i in range(4):
                if self.phase<2:new_ground[i]+=int(((gold['ground']==i)*feat['fact_mask']).sum())
                if self.phase>0:new_query[i]+=int(((gold['labels']==i)*feat['query_mask']).sum())
            with torch.autocast(device_type=self.device,dtype=torch.float16,enabled=self.device=='cuda'):
                if self.phase==2:
                    with torch.no_grad():enc=self.model.encode(feat)
                else:enc=self.model.encode(feat)
                if self.phase==0:
                    loss=masked_ce(enc['ground'],gold['ground'],feat['fact_mask'],self.ground_weights);parts={'ground_ce':loss.detach()}
                elif self.phase==1:
                    target=self.model.target(feat,gold['labels'])
                    ce=masked_ce(self.model.readout(enc['init']),gold['labels'],feat['query_mask'],self.weights)
                    mse=masked_mse(enc['init'],target,feat['query_mask'])
                    ground=masked_ce(enc['ground'],gold['ground'],feat['fact_mask'],self.ground_weights)
                    loss=ce+.25*mse+.2*ground;parts={'semantic_ce':ce.detach(),'semantic_mse':mse.detach(),'ground_ce':ground.detach()}
                else:
                    target=self.model.target(feat,gold['labels'])
                    loss,parts=diffusion_loss(self.model,enc,target,gold['labels'],feat['query_mask'],self.noise,self.weights)
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                self.scaler.scale(loss/self.plan.accumulation).backward()
            for k,v in parts.items():losses[k]=losses.get(k,0.)+float(v)/self.plan.accumulation
            losses['loss']=losses.get('loss',0.)+float(loss.detach())/self.plan.accumulation
        self.scaler.unscale_(self.optimizer)
        if PHASES[self.phase] not in self.gradient_audit:
            grad={name:p.numel() for name,p in self.model.named_parameters() if p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()}
            missing=[name for name,p in self.model.named_parameters() if p.requires_grad and p.grad is None]
            self.gradient_audit[PHASES[self.phase]]={'finite_gradient_parameters':sum(grad.values()),'tensors':len(grad),'missing':missing,'names':list(grad)}
            if missing:raise ValueError('unused trainable tensors: '+str(missing))
        norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.)
        old_scale=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update();self.attempts+=1
        successful=self.scaler.get_scale()>=old_scale
        if not successful:
            print(json.dumps({'amp_step_skipped':True,'attempt':self.attempts,'scale':self.scaler.get_scale()}),flush=True)
            if self.attempts-self.updates>20:raise FloatingPointError('반복 AMP overflow: 학습 중단')
        else:self.updates+=1;self.phase_step+=1
        self.exposure+=new_bytes
        self.query_exposure=[a+b for a,b in zip(self.query_exposure,new_query)]
        self.ground_exposure=[a+b for a,b in zip(self.ground_exposure,new_ground)]
        for group in self.optimizer.param_groups:
            progress=min(1,self.phase_step/self.plan.phase_steps[self.phase])
            group['lr']=group['base_lr']*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        return losses,successful

    def run(self,max_updates=None,wall_seconds=3600):
        start=time.monotonic();starting_updates=self.updates;last_saved=-1
        try:
            while self.phase<3:
                if self.phase_step>=self.plan.phase_steps[self.phase]:
                    result=evaluate(self.model,self.dev,self.device,self.plan.micro_batch,full=False)
                    measure=result['ground_accuracy'] if self.phase==0 else result['initializer']['accuracy']
                    required=(self.plan.ground_gate,self.plan.semantic_gate,0.)[self.phase]
                    gate={'phase':PHASES[self.phase],'measured':measure,'required':required,'passed':measure>=required}
                    self.gates.append(gate);write_json(self.out/(PHASES[self.phase]+'_dev.json'),result)
                    print(json.dumps({'gate':gate},ensure_ascii=False),flush=True)
                    if not gate['passed']:
                        self.elapsed+=time.monotonic()-start;self.save();return self.state()
                    self.phase+=1;self.phase_step=0;self._optimizer()
                    if self.phase==3:break
                if (max_updates is not None and self.updates-starting_updates>=max_updates) or time.monotonic()-start>=wall_seconds:break
                losses,success=self._step()
                if success and (self.updates%self.plan.log_every==0 or self.phase_step==1):
                    log={'update':self.updates,'phase':PHASES[self.phase],'phase_step':self.phase_step,**losses}
                    self.history.append(log);print(json.dumps(log),flush=True)
                    write_json(self.out/'training_status.json',self.state())
                if success and self.updates%self.plan.save_every==0 and last_saved!=self.updates:
                    self.elapsed+=time.monotonic()-start;start=time.monotonic();self.save();last_saved=self.updates
        except BaseException:
            self.elapsed+=time.monotonic()-start;self.save();raise
        self.elapsed+=time.monotonic()-start;self.save()
        if self.phase==3:
            self.export()
            dev=evaluate(self.model,self.dev,self.device,self.plan.micro_batch,(17,29,43))
            write_json(self.out/'development_evaluation.json',dev)
            final={}
            for name,items in [('graph_test',make_split('test',self.plan.test_graphs)),
                               ('heldout_phrase_test',make_split('test',64,seed=20001,heldout=True)),
                               ('larger_graph_test',make_split('test',64,seed=40001,n_range=(7,8)))]:
                final[name]=evaluate(self.model,items,self.device,self.plan.micro_batch,(17,29,43))
            write_json(self.out/'final_evaluation.json',final)
            print(json.dumps({'complete':True,'development':dev,'test':final},ensure_ascii=False),flush=True)
        return self.state()

    def export(self):
        temp=self.out/'runtime.pt.tmp';target=self.out/'runtime.pt'
        torch.save({'model':self.model.state_dict(),'config':asdict(self.cfg),'contract':self.contract,
                    'gates':self.gates,'training_complete':self.phase==3,'updates':self.updates},temp)
        os.replace(temp,target);write_json(self.out/'runtime.json',{'file':'runtime.pt','sha256':digest_file(target),'parameters':self.model.counts()['parameters']})


def preflight(out,device='cuda',cfg=Config()):
    if device=='cuda':
        if not torch.cuda.is_available() or 'T4' not in torch.cuda.get_device_name():raise ValueError('T4 CUDA가 필요합니다.')
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(19);model=FDT100(cfg).to(device);model.train()
    items=make_split('train',4);feat,gold,_=collate(items,cfg,device)
    start=time.monotonic();generator=torch.Generator(device=device).manual_seed(23)
    with torch.autocast(device_type=device,dtype=torch.float16,enabled=device=='cuda'):
        enc=model.encode(feat);target=model.target(feat,gold['labels'])
        loss,_=diffusion_loss(model,enc,target,gold['labels'],feat['query_mask'],generator)
        loss=loss+masked_ce(enc['ground'],gold['ground'],feat['fact_mask'])+masked_mse(enc['init'],target,feat['query_mask'])
    loss.backward()
    if device=='cuda':torch.cuda.synchronize()
    gradients={name:p.numel() for name,p in model.named_parameters() if p.grad is not None and torch.isfinite(p.grad).all()}
    missing=[name for name,p in model.named_parameters() if p.grad is None]
    if missing or len(gradients)!=len(list(model.parameters())):raise ValueError('missing/nonfinite gradients: '+str(missing))
    result={**model.counts(),'device':device,'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name() if device=='cuda' else None,
            'gpu_memory_total_gib':torch.cuda.get_device_properties(0).total_memory/2**30 if device=='cuda' else None,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30 if device=='cuda' else None,
            'finite_gradient_parameters':sum(gradients.values()),'gradient_tensors':len(gradients),
            'forward_backward_seconds':time.monotonic()-start,'loss':float(loss.detach()),'source':source_digest()}
    write_json(out,result);print(json.dumps(result,ensure_ascii=False),flush=True)
    del model;gc.collect()
    if device=='cuda':torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--max-updates',type=int);parser.add_argument('--wall-seconds',type=int,default=3600);parser.add_argument('--warm-start')
    args=parser.parse_args()
    if args.preflight:preflight(pathlib.Path(args.output)/'t4_preflight.json');return
    trainer=Trainer(args.output,resume=args.resume,warm_start=args.warm_start);print(json.dumps(trainer.run(args.max_updates,args.wall_seconds),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
