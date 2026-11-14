"""Finite one-epoch live-language optimization and exact CPU-testable fullstate."""
from __future__ import annotations
import copy, hashlib, math, random
from contextlib import nullcontext
import numpy as np
import torch
from fdt_local.runner import numpy_state
from research.text_diffusion_v1 import pack_prompts, PAD
import text_generation_support_v1 as support


def finite_order(count, seed=61):
    order=list(range(count));random.Random(seed).shuffle(order);return order


def row_noise(rows, length, dim, seed, device):
    values=[]
    for row in rows:
        key=int(hashlib.sha256((str(seed)+'|'+row['id']).encode()).hexdigest()[:15],16)
        generator=torch.Generator(device=device).manual_seed(key)
        values.append(torch.randn(length,dim,generator=generator,device=device))
    return torch.stack(values)


class LanguageEngine:
    def __init__(self,model,features,targets,order,budget,plan_sha,device='cuda'):
        self.model=model.to(device);self.features=features;self.targets=targets;self.order=order
        if sorted(order)!=list(range(len(features))) or len(features)!=len(targets):raise ValueError('finite unique order')
        self.budget=budget;self.plan_sha=plan_sha;self.device=device
        self.limit=math.ceil(len(order)/8);self.attempts=self.updates=self.cursor=self.skips=0
        self.history=[];self.evaluations={};self.exposure={'selected_rows':0,'processed_rows':0,
            'selected_target_bytes':0,'processed_target_bytes':0,'selected_EOF':0,
            'processed_EOF':0,'selected_context_bytes':0,'processed_context_bytes':0}
        self.parameter_names={key:[] for key in ('language','other')};groups={key:[] for key in self.parameter_names}
        for name,param in self.model.named_parameters():
            if param.requires_grad:
                key='language' if name.startswith('vp.language.') else 'other'
                groups[key].append(param);self.parameter_names[key].append(name)
        self.optimizer=torch.optim.AdamW([{'params':groups['language'],'lr':1e-5,'base_lr':1e-5},
            {'params':groups['other'],'lr':3e-5,'base_lr':3e-5}],weight_decay=.01)
        self.scaler=torch.amp.GradScaler('cuda' if str(device).startswith('cuda') else 'cpu',init_scale=1024.)
        self.frozen_sha=support.state_sha(self.frozen_state())

    def frozen_state(self):
        frozen={n:p for n,p in self.model.named_parameters() if not p.requires_grad}
        frozen.update(dict(self.model.named_buffers()));return frozen

    def verify_frozen(self):
        if support.state_sha(self.frozen_state())!=self.frozen_sha:raise RuntimeError('frozen prefix/codec mutated')

    def next_indices(self):
        if self.attempts>=self.limit or self.cursor>=len(self.order):raise StopIteration('one finite epoch closed')
        return self.order[self.cursor:self.cursor+8]

    def step(self):
        self.budget.check();indices=self.next_indices();self.attempts+=1;self.cursor+=len(indices)
        def expose(prefix,ids):
            self.exposure[prefix+'_rows']+=len(ids)
            self.exposure[prefix+'_target_bytes']+=sum(len(self.targets[i]['target'].encode('utf-8')) for i in ids)
            self.exposure[prefix+'_EOF']+=sum(self.targets[i]['source_eof'] for i in ids)
            self.exposure[prefix+'_context_bytes']+=sum(len(self.features[i]['prompt'].encode('utf-8')) for i in ids)
        expose('selected',indices);self.model.train();self.optimizer.zero_grad(set_to_none=True)
        factor=.1+.9*(1+math.cos(math.pi*(self.attempts-1)/self.limit))/2
        for group in self.optimizer.param_groups:group['lr']=group['base_lr']*factor
        total_symbols=sum(len(self.targets[i]['target_ids']) for i in indices);means={};outcome='FAILED_ATTEMPT'
        try:
            for start in range(0,len(indices),4):
                ids=indices[start:start+4];rows=[self.features[i] for i in ids]
                target=torch.full((len(ids),self.model.policy.target_bytes_with_eos),PAD,device=self.device,dtype=torch.long)
                for j,i in enumerate(ids):target[j,:len(self.targets[i]['target_ids'])]=torch.tensor(self.targets[i]['target_ids'],device=self.device)
                tokens=pack_prompts([r['prompt'] for r in rows],self.model.policy.prompt_bytes,self.device)
                t=torch.rand(len(ids),device=self.device)*.998+.001
                noise=torch.randn(len(ids),target.shape[1],self.model.core_config.z_dim,device=self.device)
                proposal=row_noise(rows,target.shape[1],self.model.core_config.z_dim,61,self.device)
                weight=sum(len(self.targets[i]['target_ids']) for i in ids)/total_symbols
                def charge(kind):
                    self.budget.charge(kind)
                    if kind=='encode':expose('processed',ids)
                amp=torch.autocast('cuda',dtype=torch.float16) if str(self.device).startswith('cuda') else nullcontext()
                with amp:loss,aux=self.model.training_loss(tokens,target,t,noise,proposal,charge)
                if not torch.isfinite(loss) or float(loss.detach())>1000:raise FloatingPointError('numeric/loss explosion')
                self.scaler.scale(loss*weight).backward()
                for key in ('velocity_mse','clean_mse','VP_byte_ce','AR_byte_ce'):
                    means[key]=means.get(key,0.)+float(aux[key].detach())*weight
                means['loss']=means.get('loss',0.)+float(loss.detach())*weight
                del aux,loss
            self.scaler.unscale_(self.optimizer)
            for name,param in self.model.named_parameters():
                if not param.requires_grad and param.grad is not None:raise RuntimeError('frozen gradient ownership '+name)
                if param.requires_grad and param.grad is None:raise RuntimeError('missing required trainable gradient '+name)
            norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.)
            before=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update()
            skipped=self.scaler.get_scale()<before;self.skips+=int(skipped);self.updates+=int(not skipped)
            outcome='AMP_SKIP' if skipped else 'UPDATED'
            record={'attempt':self.attempts,'updates':self.updates,'rows':indices,'outcome':outcome,
                **means,'grad_norm':float(norm) if torch.isfinite(norm) else None,
                'AMP_skip':skipped,'scale':self.scaler.get_scale(),'actual_calls':dict(self.budget.calls)}
            self.history.append(record);self.optimizer.zero_grad(set_to_none=True)
            if self.skips>=3 or (not skipped and not torch.isfinite(norm)):raise FloatingPointError('AMP/gradient stop')
            if len(self.history)>=11:
                reference=sorted(r['loss'] for r in self.history[-11:-1] if 'loss' in r)
                if reference and means['loss']>20*reference[len(reference)//2]:raise FloatingPointError('20x loss explosion')
            return record
        except BaseException as exc:
            if len(self.history)<self.attempts:
                self.history.append({'attempt':self.attempts,'updates':self.updates,'rows':indices,
                    'outcome':outcome,'error':repr(exc),'actual_calls':dict(self.budget.calls)})
            raise

    def export(self):
        return {'schema':'fdt_text_language_v5_fullstate','plan_sha256':self.plan_sha,
            'model':{k:v.detach().cpu() for k,v in self.model.state_dict().items()},
            'optimizer':self.optimizer.state_dict(),'optimizer_parameter_names':self.parameter_names,
            'scaler':self.scaler.state_dict(),'order':self.order,'cursor':self.cursor,
            'attempts':self.attempts,'updates':self.updates,'skips':self.skips,'history':self.history,
            'exposure':dict(self.exposure),'actual_calls':dict(self.budget.calls),'evaluations':self.evaluations,
            'frozen_sha':self.frozen_sha,'scheduler':{'horizon':self.limit,'consumed':self.attempts},
            'gradients':{k:p.grad.detach().cpu() for k,p in self.model.named_parameters() if p.grad is not None},
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if str(self.device).startswith('cuda') else [],
            'python_rng':random.getstate(),'numpy_rng':numpy_state()}

    def restore(self,state):
        if state['schema']!='fdt_text_language_v5_fullstate' or state['plan_sha256']!=self.plan_sha:raise ValueError('exact plan required')
        if state['order']!=self.order or state['optimizer_parameter_names']!=self.parameter_names:raise ValueError('order/optimizer ownership')
        if not state['scaler'] or state['scaler'].get('scale',0)<=0:raise ValueError('enabled scaler state')
        attempts=state['attempts'];cursor=min(8*attempts,len(self.order))
        if not 0<=attempts<=self.limit or cursor!=state['cursor'] or state['scheduler']!={'horizon':self.limit,'consumed':attempts}:raise ValueError('finite counter/cursor')
        if [r['attempt'] for r in state['history']]!=list(range(1,attempts+1)):raise ValueError('attempt history')
        if [i for r in state['history'] for i in r['rows']]!=self.order[:cursor]:raise ValueError('row replay/gap')
        self.model.load_state_dict(state['model'],strict=True);self.optimizer.load_state_dict(state['optimizer']);self.scaler.load_state_dict(state['scaler'])
        for key in ('attempts','updates','cursor','skips','history','exposure','evaluations','frozen_sha'):setattr(self,key,copy.deepcopy(state[key]))
        if self.exposure['selected_rows']!=cursor:raise ValueError('selected exposure/cursor')
        self.budget.calls=dict(state['actual_calls']);self.model.zero_grad(set_to_none=True)
        for name,param in self.model.named_parameters():
            if name in state['gradients']:param.grad=state['gradients'][name].to(self.device)
        torch.set_rng_state(state['torch_rng'])
        if str(self.device).startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])
        random.setstate(state['python_rng']);n=state['numpy_rng']
        np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
        self.verify_frozen()


class RecoveryEngine(LanguageEngine):
    def export(self):
        value=super().export()
        if not str(self.device).startswith('cuda') and hasattr(self,'inactive_cuda_rng'):
            value['cuda_rng']=self.inactive_cuda_rng
        return value

    def import_source(self,source,source_plan_sha):
        if source['plan_sha256']!=source_plan_sha:raise ValueError('source plan identity')
        adapted=dict(source);adapted['plan_sha256']=self.plan_sha
        self.inactive_cuda_rng=copy.deepcopy(source['cuda_rng'])
        self.restore(adapted)
        after=self.export()
        keys=set(after)-{'plan_sha256'}
        before_hash=support.state_sha({k:source[k] for k in keys})
        after_hash=support.state_sha({k:after[k] for k in keys})
        if before_hash!=after_hash:raise ValueError('full algorithm import mismatch')
        return before_hash
