"""Finite three-phase training; exact state/optimizer/scaler/RNG recovery."""
import copy,math,random
from contextlib import nullcontext
import numpy as np
import torch
from fdt_local.runner import numpy_state
from research.text_diffusion_v1 import pack_prompts,pack_answers
from research.semantic_output_pipeline_v11 import PHASES
import text_generation_support_v1 as support

class SemanticEngine:
    def __init__(self,model,features,targets,orders,budget,plan_sha,device='cuda'):
        self.model=model.to(device);self.features=features;self.targets=targets;self.orders=orders
        self.budget=budget;self.plan_sha=plan_sha;self.device=device;self.phase='input'
        self.attempts_by_phase={p:0 for p in PHASES};self.updates_by_phase=dict(self.attempts_by_phase);self.cursors=dict(self.attempts_by_phase)
        self.history=[];self.exposure={p:0 for p in PHASES};self.exposure.update(teacher_rows=0,predicted_rows=0,fallback_rows=0,language_target_bytes=0)
        self.evaluations={};self.cache_reference=None;self.cache=None;self.skips=0
        self.optimizers={};self.parameter_names={};self.inactive_cuda_rng=[]
        self.scaler=torch.amp.GradScaler('cuda' if str(device).startswith('cuda') else 'cpu',init_scale=1024.)
        self.permanent_names=[n for n,p in self.model.named_parameters() if n not in self.model.potential_trainable]
        self.permanent_sha=support.state_sha({n:p for n,p in self.model.named_parameters() if n in self.permanent_names})
        self.phase_frozen_sha=None;self.set_phase('input',initial=True)

    @property
    def attempts(self):return sum(self.attempts_by_phase.values())
    @property
    def updates(self):return sum(self.updates_by_phase.values())
    @property
    def cursor(self):return sum(self.cursors.values())

    def set_phase(self,phase,initial=False):
        if not initial:
            self.verify_frozen()
            old=PHASES.index(self.phase)
            if PHASES.index(phase)!=old+1 or self.attempts_by_phase[self.phase]!=len(self.orders[self.phase]):raise ValueError('phase boundary')
        self.model.zero_grad(set_to_none=True);self.model.set_phase(phase);self.phase=phase
        selected=[(n,p) for n,p in self.model.named_parameters() if p.requires_grad]
        if phase not in self.optimizers:
            language=[(n,p) for n,p in selected if n.startswith('encoder.language.')]
            other=[(n,p) for n,p in selected if not n.startswith('encoder.language.')]
            groups=[]
            for rows,lr in ((language,1e-5),(other,3e-4 if phase!='meaning' else 3e-5)):
                if rows:groups.append({'params':[p for n,p in rows],'lr':lr,'base_lr':lr})
            self.optimizers[phase]=torch.optim.AdamW(groups,weight_decay=.01)
            self.parameter_names[phase]=[n for n,p in language+other]
        self.phase_frozen_sha=support.state_sha({n:p for n,p in self.model.named_parameters() if not p.requires_grad})

    def verify_frozen(self):
        permanent={n:p for n,p in self.model.named_parameters() if n in self.permanent_names}
        if support.state_sha(permanent)!=self.permanent_sha:raise RuntimeError('permanent language prefix changed')
        if self.phase_frozen_sha and support.state_sha({n:p for n,p in self.model.named_parameters() if not p.requires_grad})!=self.phase_frozen_sha:raise RuntimeError('inactive phase weights changed')

    def next_indices(self):
        i=self.attempts_by_phase[self.phase]
        if i>=len(self.orders[self.phase]):raise StopIteration('finite phase ended')
        return self.orders[self.phase][i]

    def step(self):
        self.budget.check();indices=self.next_indices();phase=self.phase
        self.attempts_by_phase[phase]+=1;self.cursors[phase]+=len(indices);self.exposure[phase]+=len(indices)
        attempt=self.attempts_by_phase[phase];optim=self.optimizers[phase];optim.zero_grad(set_to_none=True)
        self.model.train();lr_factor=.1+.9*(1+math.cos(math.pi*(attempt-1)/len(self.orders[phase])))/2
        for group in optim.param_groups:group['lr']=group['base_lr']*lr_factor
        ys=[self.targets['train'][i] for i in indices];gold=torch.tensor([y['output_labels'] for y in ys],device=self.device)
        amp=torch.autocast('cuda',dtype=torch.float16) if str(self.device).startswith('cuda') else nullcontext()
        teacher=predicted=fallback=0
        try:
            with amp:
                if phase in ('input','meaning'):
                    tokens=pack_prompts([self.features['train'][i]['prompt'] for i in indices],160,self.device)
                    if phase=='input':read=self.model.extract(tokens,self.budget.charge)
                    else:
                        with torch.no_grad():read=self.model.extract(tokens,self.budget.charge)
                    if phase=='input':
                        labels=torch.tensor([y['input_labels'] for y in ys],device=self.device)
                        spans=torch.tensor([y['input_spans'] for y in ys],device=self.device)
                        loss=self.model.input.loss(read,spans,labels);aux={}
                    else:
                        noise=torch.randn(len(indices),1,self.model.c.z_dim,device=self.device)
                        t=torch.rand(len(indices),device=self.device)*.998+.001
                        self.budget.charge('denoise');loss,aux=self.model.meaning.loss(read,gold,t,noise)
                else:
                    if self.cache is None:raise ValueError('frozen-stage inference cache required')
                    committed=self.cache['train']['labels'][indices].to(self.device)
                    matched=(committed==gold).all(-1)
                    mixed=attempt>len(self.orders[phase])//2
                    use=matched if mixed else torch.zeros_like(matched)
                    committed=torch.where(use[:,None],committed,gold)
                    predicted=int(use.sum());teacher=len(indices)-predicted;fallback=int((~matched).sum()) if mixed else 0
                    self.exposure['teacher_rows']+=teacher;self.exposure['predicted_rows']+=predicted;self.exposure['fallback_rows']+=fallback
                    self.exposure['language_target_bytes']+=sum(len(y['answer'].encode('utf-8')) for y in ys)
                    answer=pack_answers([y['answer'] for y in ys],40,self.device)
                    self.budget.charge('decoder');loss=self.model.realizer.loss(committed,answer);aux={}
            if not torch.isfinite(loss) or float(loss.detach())>1000:raise FloatingPointError('numeric/loss explosion')
            self.scaler.scale(loss).backward();self.scaler.unscale_(optim)
            for n,p in self.model.named_parameters():
                if not p.requires_grad and p.grad is not None:raise RuntimeError('frozen gradient '+n)
                if p.requires_grad and p.grad is None:raise RuntimeError('missing trainable gradient '+n)
            norm=torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad],1.)
            before=self.scaler.get_scale();self.scaler.step(optim);self.scaler.update();skip=self.scaler.get_scale()<before
            self.skips+=int(skip);self.updates_by_phase[phase]+=int(not skip)
            rec={'attempt':self.attempts,'phase':phase,'phase_attempt':attempt,'rows':indices,'updates':self.updates,
                'loss':float(loss.detach()),'grad_norm':float(norm) if torch.isfinite(norm) else None,'AMP_skip':skip,
                'outcome':'AMP_SKIP' if skip else 'UPDATED','teacher_rows':teacher,'predicted_rows':predicted,'fallback_rows':fallback,
                'aux':{k:float(v.detach()) for k,v in aux.items()}}
            self.history.append(rec);optim.zero_grad(set_to_none=True)
            if self.skips>=3 or (not skip and not torch.isfinite(norm)):raise FloatingPointError('AMP/gradient stop')
            recent=[r['loss'] for r in self.history[-11:-1] if r['phase']==phase and 'loss' in r]
            if len(recent)==10 and rec['loss']>20*sorted(recent)[5]:raise FloatingPointError('20x loss explosion')
            return rec
        except BaseException as exc:
            if len(self.history)<self.attempts:self.history.append({'attempt':self.attempts,'phase':phase,'phase_attempt':attempt,'rows':indices,'outcome':'FAILED_ATTEMPT','error':repr(exc)})
            raise

    def export(self):
        return {'schema':'fdt_semantic_output_v11_fullstate','plan_sha256':self.plan_sha,
            'model':{k:v.detach().cpu() for k,v in self.model.state_dict().items()},'core':self.model.c.__dict__,
            'optimizers':{k:v.state_dict() for k,v in self.optimizers.items()},'optimizer_parameter_names':self.parameter_names,
            'potential_trainable':self.model.potential_trainable,'scaler':self.scaler.state_dict(),
            'phase':self.phase,'phase_frozen_sha':self.phase_frozen_sha,'permanent_sha':self.permanent_sha,'permanent_names':self.permanent_names,
            'attempts':self.attempts,'updates':self.updates,'cursor':self.cursor,'attempts_by_phase':self.attempts_by_phase,
            'updates_by_phase':self.updates_by_phase,'cursors':self.cursors,'orders':self.orders,'history':self.history,
            'exposure':self.exposure,'skips':self.skips,'actual_calls':dict(self.budget.calls),'evaluations':self.evaluations,
            'cache_reference':self.cache_reference,'gradients':{},
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if str(self.device).startswith('cuda') else self.inactive_cuda_rng,
            'python_rng':random.getstate(),'numpy_rng':numpy_state()}

    def restore(self,state):
        if state['schema']!='fdt_semantic_output_v11_fullstate' or state['plan_sha256']!=self.plan_sha or state['orders']!=self.orders:raise ValueError('exact state/plan/order')
        if state['potential_trainable']!=self.model.potential_trainable or state['gradients']:raise ValueError('ownership/unfinished gradients')
        if len(state['history'])!=state['attempts'] or [r['attempt'] for r in state['history']]!=list(range(1,state['attempts']+1)):raise ValueError('complete history')
        for phase in PHASES:
            count=state['attempts_by_phase'][phase]
            if not 0<=count<=len(self.orders[phase]) or sum(map(len,self.orders[phase][:count]))!=state['cursors'][phase]:raise ValueError('finite cursor')
            if [r['rows'] for r in state['history'] if r['phase']==phase]!=self.orders[phase][:count]:raise ValueError('row replay/gap')
        self.model.load_state_dict(state['model'],strict=True)
        self.optimizers={};self.parameter_names={}
        for phase in PHASES:
            if phase in state['optimizers']:
                self.set_phase(phase,initial=True);self.optimizers[phase].load_state_dict(state['optimizers'][phase])
        if self.parameter_names!=state['optimizer_parameter_names']:raise ValueError('optimizer ownership')
        self.phase=state['phase'];self.model.set_phase(self.phase);self.scaler.load_state_dict(state['scaler'])
        for key in ('attempts_by_phase','updates_by_phase','cursors','history','exposure','skips','evaluations','cache_reference','permanent_sha','phase_frozen_sha'):
            setattr(self,key,copy.deepcopy(state[key]))
        if (self.attempts,self.updates,self.cursor)!=(state['attempts'],state['updates'],state['cursor']):raise ValueError('counter sums')
        self.budget.calls=dict(state['actual_calls'])
        if hasattr(self.budget,'allocation_calls'):self.budget.allocation_calls=dict(state['actual_calls'])
        torch.set_rng_state(state['torch_rng'])
        if str(self.device).startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])
        else:self.inactive_cuda_rng=copy.deepcopy(state['cuda_rng'])
        random.setstate(state['python_rng']);n=state['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
        if self.cache_reference:
            path=self.budget.output/self.cache_reference['path']
            if support.sha(path)!=self.cache_reference['sha256']:raise ValueError('frozen prediction cache changed')
            self.cache=torch.load(path,map_location='cpu',weights_only=True)
        self.model.zero_grad(set_to_none=True);self.verify_frozen()
