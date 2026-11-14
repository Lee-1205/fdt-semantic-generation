"""Two new finite epochs; retained Adam/scaler/RNG, explicit new LR curriculum."""
from __future__ import annotations
import copy,random
import numpy as np
import torch
from text_language_engine_v6 import LanguageEngine,finite_order
import text_generation_support_v1 as support


class CurriculumEngine(LanguageEngine):
    def __init__(self,model,features,targets,budget,plan_sha,device='cuda',seeds=(62,63)):
        super().__init__(model,features,targets,finite_order(len(features),seeds[0]),budget,plan_sha,device)
        self.seeds=list(seeds);self.batches=[]
        for seed in seeds:
            order=finite_order(len(features),seed)
            self.batches.extend(order[i:i+8] for i in range(0,len(order),8))
        self.order=[i for batch in self.batches for i in batch];self.limit=len(self.batches)
        self.ancestry={}

    def next_indices(self):
        if self.attempts>=self.limit:raise StopIteration('two finite epochs closed')
        return self.batches[self.attempts]

    def step(self):
        # A prospective lower-base LR schedule, not a reset of any old deadline.
        for i,group in enumerate(self.optimizer.param_groups):
            group['base_lr']=5e-6 if i==0 else 1.5e-5
        return super().step()

    def export(self):
        state=super().export();state.update(schema='fdt_text_language_v7_fullstate',
            curriculum={'seeds':self.seeds,'batches':self.batches,'epochs':len(self.seeds)},ancestry=self.ancestry)
        if not str(self.device).startswith('cuda') and hasattr(self,'inactive_cuda_rng'):
            state['cuda_rng']=self.inactive_cuda_rng
        return state

    def restore(self,state):
        if state['schema']!='fdt_text_language_v7_fullstate' or state['plan_sha256']!=self.plan_sha:raise ValueError('exact plan')
        if state['curriculum']!={'seeds':self.seeds,'batches':self.batches,'epochs':len(self.seeds)}:raise ValueError('curriculum')
        if state['order']!=self.order or state['optimizer_parameter_names']!=self.parameter_names:raise ValueError('order/ownership')
        attempts=state['attempts'];cursor=sum(map(len,self.batches[:attempts]))
        if not 0<=attempts<=self.limit or cursor!=state['cursor'] or state['scheduler']!={'horizon':self.limit,'consumed':attempts}:raise ValueError('finite counter')
        if [r['attempt'] for r in state['history']]!=list(range(1,attempts+1)):raise ValueError('history')
        if [r['rows'] for r in state['history']]!=self.batches[:attempts]:raise ValueError('replay/gap')
        if not state['scaler'] or state['scaler'].get('scale',0)<=0:raise ValueError('enabled scaler')
        self.model.load_state_dict(state['model'],strict=True);self.optimizer.load_state_dict(state['optimizer']);self.scaler.load_state_dict(state['scaler'])
        for key in ('attempts','updates','cursor','skips','history','exposure','evaluations','frozen_sha','ancestry'):setattr(self,key,copy.deepcopy(state[key]))
        if self.exposure['selected_rows']!=cursor:raise ValueError('exposure')
        self.budget.calls=dict(state['actual_calls']);self.model.zero_grad(set_to_none=True)
        for name,param in self.model.named_parameters():
            if name in state['gradients']:param.grad=state['gradients'][name].to(self.device)
        self.restore_rng(state);self.verify_frozen()

    def restore_rng(self,state):
        torch.set_rng_state(state['torch_rng'])
        if str(self.device).startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])
        else:self.inactive_cuda_rng=copy.deepcopy(state['cuda_rng'])
        random.setstate(state['python_rng']);n=state['numpy_rng']
        np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))

    def import_parent(self,state,source_plan):
        if state['plan_sha256']!=source_plan or state['gradients']:raise ValueError('parent identity/unfinished gradient')
        if state['optimizer_parameter_names']!=self.parameter_names:raise ValueError('parent optimizer ownership')
        self.model.load_state_dict(state['model'],strict=True);self.optimizer.load_state_dict(state['optimizer']);self.scaler.load_state_dict(state['scaler'])
        self.frozen_sha=state['frozen_sha'];self.budget.calls=dict(state['actual_calls']);self.restore_rng(state);self.verify_frozen()
        keys=('model','optimizer','scaler','optimizer_parameter_names','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')
        before=support.state_sha({k:state[k] for k in keys});after=self.export()
        if before!=support.state_sha({k:after[k] for k in keys}):raise ValueError('parent full algorithm components not retained')
        self.ancestry={k:copy.deepcopy(state[k]) for k in ('plan_sha256','attempts','updates','cursor','history','exposure','order','evaluations','actual_calls','scheduler')}
        self.ancestry['imported_component_sha256']=before
        return before
