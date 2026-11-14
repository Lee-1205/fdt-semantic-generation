"""Finite dynamic-length codec engine; complete RNG/Adam/cursor restoration."""
import copy,random,math
from contextlib import nullcontext
import numpy as np
import torch
from fdt_local.runner import numpy_state
from research.text_diffusion_v1 import pack_prompts,pack_answers
import text_generation_support_v1 as support

SCHEMA='fdt_long_three_memory_codec_v50_fullstate'
NEW=('decoder.positions.extension','decoder.local_read_gain','decoder.local_slot_tag')

def restore_rng(state,device):
    torch.set_rng_state(state['torch_rng'])
    if str(device).startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng']);n=state['numpy_rng']
    np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))

class LongEngine:
    def __init__(self,model,source_names,rows,orders,budget,plan_sha,device='cuda'):
        self.model=model.to(device);self.device=device;self.rows=rows;self.orders=orders
        self.budget=budget;self.plan_sha=plan_sha;self.names=copy.deepcopy(source_names)
        named=dict(model.named_parameters())
        self.optimizer=torch.optim.AdamW([{'params':[named[n] for n in group]} for group in self.names])
        self.scaler=torch.amp.GradScaler('cuda' if str(device).startswith('cuda') else 'cpu',init_scale=1024.)
        self.attempts=self.updates=self.cursor=self.skips=0;self.history=[];self.evaluations={}
        self.exposure=dict(selected_rows=0,target_UTF8_bytes=0,active_symbols_per_pass=0)
        self.inactive_cuda_rng=[];self.ancestry={}
        self.frozen_sha=support.state_sha({n:p for n,p in model.named_parameters() if not p.requires_grad})

    def import_parent(self,state):
        if state['optimizer_parameter_names']!=self.names:raise ValueError('old92 Adam ownership')
        self.optimizer.load_state_dict(state['optimizer']);self.scaler.load_state_dict(state['scaler'])
        self.inactive_cuda_rng=copy.deepcopy(state['cuda_rng'])
        if support.state_sha(self.optimizer.state_dict())!=support.state_sha(state['optimizer']):raise ValueError('old Adam changed')
        self.ancestry={k:copy.deepcopy(state[k]) for k in ('plan_sha256','attempts','updates','cursor','history','actual_calls','scaler','evaluations')}
        self.ancestry['source_optimizer_digest']=support.state_sha(state['optimizer'])
        named=dict(self.model.named_parameters())
        if set(n for n,p in named.items() if p.requires_grad)!={n for group in self.names for n in group}|set(NEW):
            raise ValueError('complete trainable ownership')
        self.optimizer.add_param_group(dict(params=[named[n] for n in NEW],lr=1e-4,base_lr=1e-4,weight_decay=.01))
        self.names.append(list(NEW))
        restore_rng(state,self.device)
        self.verify_frozen()

    def verify_frozen(self):
        if support.state_sha({n:p for n,p in self.model.named_parameters() if not p.requires_grad})!=self.frozen_sha:
            raise ValueError('frozen original core changed')

    def step(self,force_overflow=False):
        self.budget.check()
        if self.attempts>=len(self.orders):raise StopIteration('closed finite epoch')
        indices=self.orders[self.attempts];rows=[self.rows[i] for i in indices]
        texts=[r['prompt'] for r in rows];length=max(len(t.encode()) for t in texts)
        self.attempts+=1;self.cursor+=len(rows);self.exposure['selected_rows']+=len(rows)
        self.exposure['target_UTF8_bytes']+=sum(len(t.encode()) for t in texts)
        self.exposure['active_symbols_per_pass']+=sum(len(t.encode())+1 for t in texts)
        self.model.train();self.model.set_training_attempt(self.attempts);self.optimizer.zero_grad(set_to_none=True)
        x=pack_prompts(texts,length,self.device);y=pack_answers(texts,length+1,self.device)
        amp=torch.autocast('cuda',dtype=torch.bfloat16) if str(self.device).startswith('cuda') else nullcontext()
        with amp:
            loss,episode=self.model.reconstruction_loss(x,y,source_ids=tuple(r['id'] for r in rows),
                source_versions=(self.budget.data_sha,)*len(rows),model_epoch=str(self.attempts),charge=self.budget.charge)
        if not torch.isfinite(loss) or float(loss.detach())>1000:raise FloatingPointError('loss numeric stop')
        self.scaler.scale(loss).backward()
        if force_overflow:
            next(p for p in self.model.parameters() if p.grad is not None).grad.fill_(float('inf'))
        self.scaler.unscale_(self.optimizer)
        for name,parameter in self.model.named_parameters():
            if parameter.requires_grad and parameter.grad is None:raise ValueError('missing gradient '+name)
            if not parameter.requires_grad and parameter.grad is not None:raise ValueError('frozen gradient '+name)
        norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.)
        before=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update()
        skip=self.scaler.get_scale()<before;self.skips+=int(skip);self.updates+=int(not skip)
        record=dict(attempt=self.attempts,updates=self.updates,rows=indices,length=length,
            loss=float(loss.detach()),AMP_skip=skip,grad_norm=float(norm) if torch.isfinite(norm) else None,
            scale=self.scaler.get_scale(),optimizer_group_lrs=[g['lr'] for g in self.optimizer.param_groups])
        self.history.append(record);self.optimizer.zero_grad(set_to_none=True)
        flags=[h['AMP_skip'] for h in self.history]
        if sum(flags[-16:])>=3 or self.skips>=8:raise FloatingPointError('persistent AMP skip stop')
        if not skip and not torch.isfinite(norm):raise FloatingPointError('nonfinite applied update')
        return record

    def export(self):
        return dict(schema=SCHEMA,plan_sha256=self.plan_sha,core=self.model.c.__dict__,input_capacity=self.model.input_capacity,
            model={k:v.detach().cpu() for k,v in self.model.state_dict().items()},optimizer=self.optimizer.state_dict(),
            optimizer_parameter_names=self.names,scaler=self.scaler.state_dict(),orders=self.orders,
            attempts=self.attempts,updates=self.updates,cursor=self.cursor,skips=self.skips,history=self.history,
            exposure=self.exposure,evaluations=self.evaluations,actual_calls=dict(self.budget.calls),frozen_sha=self.frozen_sha,
            ancestry=self.ancestry,gradients={},torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if str(self.device).startswith('cuda') else self.inactive_cuda_rng,
            python_rng=random.getstate(),numpy_rng=numpy_state(),denoise_calls=0,copy_bytes=0)

    def restore(self,state):
        if (state['schema']!=SCHEMA or state['plan_sha256']!=self.plan_sha or state['orders']!=self.orders
            or state['optimizer_parameter_names']!=self.names or state['input_capacity']!=self.model.input_capacity
            or state['gradients'] or state['cursor']!=sum(len(b) for b in self.orders[:state['attempts']])
            or [h['rows'] for h in state['history']]!=self.orders[:state['attempts']]):
            raise ValueError('exact full recovery contract')
        self.model.load_state_dict(state['model'],strict=True);self.optimizer.load_state_dict(state['optimizer'])
        self.scaler.load_state_dict(state['scaler'])
        for k in ('attempts','updates','cursor','skips','history','exposure','evaluations','ancestry','frozen_sha'):
            setattr(self,k,copy.deepcopy(state[k]))
        self.budget.calls=dict(state['actual_calls']);self.inactive_cuda_rng=copy.deepcopy(state['cuda_rng'])
        restore_rng(state,self.device);self.verify_frozen()
