"""Continue the first four-block adaptation, preserving all THREE populated Adam groups."""
import copy,random
import numpy as np
import torch
from owt_fuzzy_engine_v29 import ContinuationEngine as TailEngine,COMPONENTS,cumulative_count,OWNERSHIP
from owt_fuzzy_engine_v17_base import AMP_STOP_POLICY
from owt_resource_budget_v29 import RESOURCE_POLICY
from research.owt_extraction_tail_v29 import OBJECTIVE,TRAINABILITY
import text_generation_support_v1 as support

class ContinuationEngine(TailEngine):
    def __init__(self,*args,source_parameter_groups,**kwargs):
        if len(source_parameter_groups)!=3:raise ValueError('all three source groups required')
        super().__init__(*args,source_parameter_groups=source_parameter_groups[:2],**kwargs)
        if self.names!=source_parameter_groups:raise ValueError('unchanged four-block optimizer ownership')

    def import_parent(self,s,source_plan,source_sha):
        if s['schema']!='fdt_owt_fuzzy_AE_v29_fullstate' or s['plan_sha256']!=source_plan or s['gradients']:raise ValueError('actual V29 source')
        if s.get('constant_mask_probability')!=.5 or s.get('trainability')!=TRAINABILITY or s.get('ownership_policy')!=OWNERSHIP or s.get('resource_policy')!=RESOURCE_POLICY or s.get('training_objective')!=OBJECTIVE or s.get('amp_guard_policy')!=AMP_STOP_POLICY or s.get('loss_reduction')!='elementwise_none_then_sum':raise ValueError('unchanged source learning contract')
        if s['optimizer_parameter_names']!=self.names or self.attempts or self.cursor or self.history:raise ValueError('all81 source ownership/new allocation')
        if s.get('external_closed_assessment_calls')!=self.budget.plan.get('external_closed_assessment_calls'):raise ValueError('external closed ledger')
        if len(s['optimizer']['state'])!=sum(map(len,self.names)):raise ValueError('all three groups must be populated')
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(copy.deepcopy(s['optimizer']));self.scaler.load_state_dict(s['scaler'])
        self.frozen_sha=s['frozen_sha'];self.migration=copy.deepcopy(s['migration']);self.budget.calls=dict(s['actual_calls'])
        torch.set_rng_state(s['torch_rng'])
        if str(self.device).startswith('cuda'):torch.cuda.set_rng_state_all(s['cuda_rng'])
        else:self.inactive_cuda_rng=copy.deepcopy(s['cuda_rng'])
        random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
        before=support.state_sha({k:s[k] for k in COMPONENTS});out=self.export()
        if before!=support.state_sha({k:out[k] for k in COMPONENTS}):raise ValueError('model/ALL81Adam/scaler/frozen/allRNG changed')
        self.ancestry={k:copy.deepcopy(s[k]) for k in ('plan_sha256','attempts','updates','cursor','orders','history','exposure','actual_calls','evaluations','ancestry')}
        self.ancestry.update(source_sha256=source_sha,source_training_objective=copy.deepcopy(s['training_objective']),
            source_constant_mask_probability=.5,source_first_thaw_migration=copy.deepcopy(s['migration']),
            imported_component_sha256=before,external_closed_assessment_calls=copy.deepcopy(s['external_closed_assessment_calls']),
            source_global_attempts=cumulative_count(s,'attempts'),source_global_updates=cumulative_count(s,'updates'))
        self.verify_frozen();self.budget.invalidate_resources();return {'all81_component_sha256':before,'all_source_components_exact':True,'fresh_moments_added':False}

    def export(self):
        s=super().export();s['schema']='fdt_owt_fuzzy_AE_v30_fullstate';return s

    def restore(self,s):
        if s['schema']!='fdt_owt_fuzzy_AE_v30_fullstate':raise ValueError('V30 whole driver schema')
        adapted=dict(s);adapted['schema']='fdt_owt_fuzzy_AE_v29_fullstate';super().restore(adapted)
