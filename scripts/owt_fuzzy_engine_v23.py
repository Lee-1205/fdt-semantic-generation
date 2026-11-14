"""Exact V22 import; train-only prefix objective and complete masked-RNG recovery."""
import copy,random
import numpy as np
import torch
from owt_fuzzy_engine_v17_base import AEEngine,AMP_STOP_POLICY
from owt_fuzzy_engine_v13 import COMPONENTS
import text_generation_support_v1 as support
from research.owt_prefix_reconstruction_v19 import OBJECTIVE

def cumulative_count(state,key):
    total=0;node=state
    while key in node:
        total+=node[key];node=node.get('ancestry',{})
    return total

class ContinuationEngine(AEEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.ancestry={}

    def import_parent(self,s,source_plan,source_sha):
        if s['schema']!='fdt_owt_fuzzy_AE_v22_fullstate' or s['plan_sha256']!=source_plan or s['gradients']:raise ValueError('actual V22 source/unfinished gradient')
        if s['optimizer_parameter_names']!=self.names or self.attempts or self.cursor or self.history:raise ValueError('optimizer/new allocation ownership')
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(copy.deepcopy(s['optimizer']));self.scaler.load_state_dict(s['scaler'])
        self.frozen_sha=s['frozen_sha'];self.budget.calls=dict(s['actual_calls'])
        torch.set_rng_state(s['torch_rng'])
        if str(self.device).startswith('cuda'):torch.cuda.set_rng_state_all(s['cuda_rng'])
        else:self.inactive_cuda_rng=copy.deepcopy(s['cuda_rng'])
        random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
        self.verify_frozen();before=support.state_sha({k:s[k] for k in COMPONENTS});out=self.export()
        if before!=support.state_sha({k:out[k] for k in COMPONENTS}):raise ValueError('all imported algorithm components changed')
        self.ancestry={k:copy.deepcopy(s[k]) for k in ('plan_sha256','attempts','updates','cursor','orders','history','exposure','actual_calls','evaluations','ancestry')}
        self.ancestry.update(source_sha256=source_sha,imported_component_sha256=before,
            source_global_attempts=cumulative_count(s,'attempts'),source_global_updates=cumulative_count(s,'updates'))
        return before

    def step(self):
        self.model.last_objective_stats=None
        try:return super().step()
        finally:
            stats=self.model.last_objective_stats
            if stats is not None and self.history and self.history[-1]['attempt']==self.attempts:
                self.history[-1].update(copy.deepcopy(stats))
                for key in ('masked_prefix_bytes','eligible_prefix_bytes','decoder_supervised_symbol_occurrences'):
                    self.exposure[key]=self.exposure.get(key,0)+stats[key]

    def export(self):
        state=super().export();state.update(schema='fdt_owt_fuzzy_AE_v23_fullstate',ancestry=self.ancestry,amp_guard_policy=dict(AMP_STOP_POLICY),training_objective=dict(OBJECTIVE),loss_reduction='elementwise_none_then_sum',
            allocation_calls=dict(getattr(self.budget,'allocation_calls',self.budget.calls)))
        return state

    def restore(self,s):
        if s['schema']!='fdt_owt_fuzzy_AE_v23_fullstate' or s.get('amp_guard_policy')!=AMP_STOP_POLICY or s.get('training_objective')!=OBJECTIVE or s.get('loss_reduction')!='elementwise_none_then_sum':raise ValueError('V22 fullstate/guard/objective schema')
        adapted=dict(s);adapted['schema']='fdt_owt_fuzzy_AE_v12_fullstate';super().restore(adapted)
        self.ancestry=copy.deepcopy(s['ancestry'])
        if hasattr(self.budget,'allocation_calls'):self.budget.allocation_calls=dict(s['allocation_calls'])
