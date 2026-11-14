"""Finite sentence AE with bounded persistent-overflow detection; source state preserved."""
import copy,math,random
from contextlib import nullcontext
import numpy as np
import torch
from fdt_local.runner import numpy_state
from research.text_diffusion_v1 import pack_prompts,pack_answers
import text_generation_support_v1 as support

AMP_STOP_POLICY={'consecutive':3,'window':16,'in_window':3,'total':8}

def amp_stop_reason(history,prefix,total_skips,skip,norm_finite):
    if not skip and not norm_finite:return 'successful update has nonfinite gradient'
    flags=[bool(h.get('AMP_skip',False)) for h in prefix[-16:]+history]
    if len(flags)>=3 and all(flags[-3:]):return '3 consecutive AMP skips'
    if sum(flags[-16:])>=3:return '3 AMP skips within16attempts'
    if total_skips>=8:return '8 AMP skips in new finite allocation'
    return None

class AEEngine:
    def __init__(self,model,features,targets,orders,budget,plan_sha,device='cuda'):
        self.model=model.to(device);self.features=features;self.targets=targets;self.orders=orders
        self.budget=budget;self.plan_sha=plan_sha;self.device=device
        named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
        lang=[(n,p) for n,p in named if n.startswith('core.language.')];other=[(n,p) for n,p in named if not n.startswith('core.language.')]
        self.names=[[n for n,p in group] for group in (lang,other)]
        self.optimizer=torch.optim.AdamW([{'params':[p for n,p in lang],'lr':1e-5,'base_lr':1e-5},
            {'params':[p for n,p in other],'lr':1e-4,'base_lr':1e-4}],weight_decay=.01)
        self.scaler=torch.amp.GradScaler('cuda' if str(device).startswith('cuda') else 'cpu',init_scale=1024.)
        self.attempts=self.updates=self.cursor=self.skips=0;self.history=[];self.evaluations={}
        self.exposure={'selected_rows':0,'processed_rows':0,'selected_target_UTF8_bytes':0,'target_UTF8_bytes':0,'source_sentence_boundaries':0}
        self.frozen_sha=support.state_sha({n:p for n,p in model.named_parameters() if not p.requires_grad})
        self.inactive_cuda_rng=[]

    def verify_frozen(self):
        if support.state_sha({n:p for n,p in self.model.named_parameters() if not p.requires_grad})!=self.frozen_sha:raise RuntimeError('frozen original reasoning/prefix changed')

    def next_indices(self):
        if self.attempts>=len(self.orders):raise StopIteration('finite epochs closed')
        return self.orders[self.attempts]

    def step(self):
        self.budget.check();indices=self.next_indices();self.attempts+=1;self.cursor+=len(indices)
        self.exposure['selected_rows']+=len(indices);self.model.train();self.optimizer.zero_grad(set_to_none=True)
        self.exposure['selected_target_UTF8_bytes']+=sum(len(self.targets[i]['target'].encode('utf-8')) for i in indices)
        factor=.1+.9*(1+math.cos(math.pi*(self.attempts-1)/len(self.orders)))/2
        for group in self.optimizer.param_groups:group['lr']=group['base_lr']*factor
        x=pack_prompts([self.features[i]['prompt'] for i in indices],128,self.device)
        y=pack_answers([self.targets[i]['target'] for i in indices],129,self.device)
        def charge(kind):
            self.budget.charge(kind)
            if kind=='encode':
                self.exposure['processed_rows']+=len(indices);self.exposure['target_UTF8_bytes']+=sum(len(self.targets[i]['target'].encode('utf-8')) for i in indices)
                self.exposure['source_sentence_boundaries']+=len(indices)
        amp=torch.autocast('cuda',dtype=torch.float16) if str(self.device).startswith('cuda') else nullcontext()
        try:
            with amp:loss,meaning=self.model.reconstruction_loss(x,y,charge)
            if not torch.isfinite(loss) or float(loss.detach())>1000:raise FloatingPointError('AE nonfinite/loss explosion')
            self.scaler.scale(loss).backward();self.scaler.unscale_(self.optimizer)
            for n,p in self.model.named_parameters():
                if p.requires_grad and p.grad is None:raise RuntimeError('missing required AE gradient '+n)
                if not p.requires_grad and p.grad is not None:raise RuntimeError('frozen gradient '+n)
            norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.)
            before=self.scaler.get_scale();self.scaler.step(self.optimizer);self.scaler.update();skip=self.scaler.get_scale()<before
            self.skips+=int(skip);self.updates+=int(not skip)
            rec={'attempt':self.attempts,'updates':self.updates,'rows':indices,'loss':float(loss.detach()),'AMP_skip':skip,
                'grad_norm':float(norm) if torch.isfinite(norm) else None,'scale':self.scaler.get_scale(),
                'mean_fuzzy_degree':float(meaning['memberships'].detach().mean()),'outcome':'AMP_SKIP' if skip else 'UPDATED'}
            self.history.append(rec);self.optimizer.zero_grad(set_to_none=True)
            reason=amp_stop_reason(self.history,getattr(self,'ancestry',{}).get('history',[]),self.skips,skip,bool(torch.isfinite(norm)))
            if reason:raise FloatingPointError('AE persistent AMP/gradient stop: '+reason)
            if len(self.history)>10 and rec['loss']>20*sorted(r['loss'] for r in self.history[-11:-1] if 'loss' in r)[5]:raise FloatingPointError('20x loss explosion')
            return rec
        except BaseException as exc:
            if len(self.history)<self.attempts:self.history.append({'attempt':self.attempts,'updates':self.updates,'rows':indices,'outcome':'FAILED_ATTEMPT','error':repr(exc)})
            raise

    def export(self):
        return {'schema':'fdt_owt_fuzzy_AE_v12_fullstate','plan_sha256':self.plan_sha,'core':self.model.c.__dict__,
            'model':{k:v.detach().cpu() for k,v in self.model.state_dict().items()},'optimizer':self.optimizer.state_dict(),
            'optimizer_parameter_names':self.names,'scaler':self.scaler.state_dict(),'orders':self.orders,
            'attempts':self.attempts,'updates':self.updates,'cursor':self.cursor,'skips':self.skips,'history':self.history,
            'exposure':self.exposure,'evaluations':self.evaluations,'actual_calls':dict(self.budget.calls),'frozen_sha':self.frozen_sha,
            'gradients':{n:p.grad.detach().cpu() for n,p in self.model.named_parameters() if p.grad is not None},
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if str(self.device).startswith('cuda') else self.inactive_cuda_rng,
            'python_rng':random.getstate(),'numpy_rng':numpy_state()}

    def restore(self,s):
        if s['schema']!='fdt_owt_fuzzy_AE_v12_fullstate' or s['plan_sha256']!=self.plan_sha or s['orders']!=self.orders or s['optimizer_parameter_names']!=self.names:raise ValueError('exact AE contract/order')
        if not 0<=s['attempts']<=len(self.orders) or sum(map(len,self.orders[:s['attempts']]))!=s['cursor']:raise ValueError('finite cursor')
        if [r['rows'] for r in s['history']]!=self.orders[:s['attempts']] or [r['attempt'] for r in s['history']]!=list(range(1,s['attempts']+1)) or s['gradients']:raise ValueError('complete history')
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(s['optimizer']);self.scaler.load_state_dict(s['scaler'])
        for k in ('attempts','updates','cursor','skips','history','exposure','evaluations','frozen_sha'):setattr(self,k,copy.deepcopy(s[k]))
        self.budget.calls=dict(s['actual_calls'])
        if hasattr(self.budget,'allocation_calls'):self.budget.allocation_calls=dict(s['actual_calls'])
        torch.set_rng_state(s['torch_rng'])
        if str(self.device).startswith('cuda'):torch.cuda.set_rng_state_all(s['cuda_rng'])
        else:self.inactive_cuda_rng=copy.deepcopy(s['cuda_rng'])
        random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
        self.model.zero_grad(set_to_none=True);self.verify_frozen()
