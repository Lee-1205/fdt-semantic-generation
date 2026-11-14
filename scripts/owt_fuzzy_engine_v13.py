"""Exact source-algorithm import into a NEW finite expanded-corpus allocation."""
import copy,random
import numpy as np
import torch
from owt_fuzzy_engine_v12 import AEEngine
import text_generation_support_v1 as support

COMPONENTS=('model','optimizer','optimizer_parameter_names','scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')
class ContinuationEngine(AEEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.ancestry={}

    def import_parent(self,s,source_plan,source_sha):
        if s['schema']!='fdt_owt_fuzzy_AE_v12_fullstate' or s['plan_sha256']!=source_plan or s['gradients']:raise ValueError('source identity/unfinished gradient')
        if s['optimizer_parameter_names']!=self.names:raise ValueError('source optimizer ownership')
        if self.attempts or self.cursor or self.history:raise ValueError('new allocation counters')
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(copy.deepcopy(s['optimizer']));self.scaler.load_state_dict(s['scaler'])
        self.frozen_sha=s['frozen_sha'];self.budget.calls=dict(s['actual_calls'])
        torch.set_rng_state(s['torch_rng'])
        if str(self.device).startswith('cuda'):torch.cuda.set_rng_state_all(s['cuda_rng'])
        else:self.inactive_cuda_rng=copy.deepcopy(s['cuda_rng'])
        random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
        self.verify_frozen();before=support.state_sha({k:s[k] for k in COMPONENTS});after=self.export()
        if before!=support.state_sha({k:after[k] for k in COMPONENTS}):raise ValueError('model/allAdam/scaler/allRNG changed on import')
        self.ancestry={k:copy.deepcopy(s[k]) for k in ('plan_sha256','attempts','updates','cursor','orders','history','exposure','actual_calls','evaluations')}
        self.ancestry.update(source_sha256=source_sha,imported_component_sha256=before)
        return before

    def export(self):
        out=super().export();out.update(schema='fdt_owt_fuzzy_AE_v13_fullstate',ancestry=self.ancestry,
            allocation_calls=dict(getattr(self.budget,'allocation_calls',self.budget.calls)))
        return out

    def restore(self,s):
        if s['schema']!='fdt_owt_fuzzy_AE_v13_fullstate':raise ValueError('new fullstate schema')
        adapted=dict(s);adapted['schema']='fdt_owt_fuzzy_AE_v12_fullstate';super().restore(adapted)
        self.ancestry=copy.deepcopy(s['ancestry'])
        if hasattr(self.budget,'allocation_calls'):self.budget.allocation_calls=dict(s['allocation_calls'])
