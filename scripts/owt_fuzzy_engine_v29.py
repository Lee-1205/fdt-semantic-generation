"""V28 exact old Adam substate plus fresh ownership for two existing language blocks."""
import copy,random
import numpy as np
import torch
from owt_fuzzy_engine_v17_base import AEEngine,AMP_STOP_POLICY
from owt_fuzzy_engine_v13 import COMPONENTS
from owt_fuzzy_engine_v28 import cumulative_count
from owt_resource_budget_v24 import RESOURCE_POLICY as SOURCE_RESOURCE_POLICY
from owt_resource_budget_v29 import RESOURCE_POLICY
from research.owt_extraction_tail_v29 import OBJECTIVE,TRAINABILITY,new_trainable_names
import text_generation_support_v1 as support

OWNERSHIP={'old_groups':2,'append_groups':1,'new_group_base_lr':3e-6,
           'new_group_weight_decay':.01,'old_moments_preserved':True,'new_states_initially_empty':True}
UNCHANGED=tuple(k for k in COMPONENTS if k not in ('optimizer','optimizer_parameter_names','frozen_sha'))

def old_optimizer_view(optimizer,source_optimizer):
    ids={i for g in source_optimizer['param_groups'] for i in g['params']}
    return {'state':{i:optimizer['state'][i] for i in ids if i in optimizer['state']},
            'param_groups':optimizer['param_groups'][:2]}

class ContinuationEngine(AEEngine):
    def __init__(self,*args,source_parameter_groups,**kwargs):
        super().__init__(*args,**kwargs)
        if len(source_parameter_groups)!=2:raise ValueError('exact two source groups')
        added=new_trainable_names(self.model,source_parameter_groups)
        expected=[n for n,p in self.model.named_parameters() if any(n.startswith('core.language.blocks.'+str(i)+'.') for i in range(max(0,len(self.model.core.language.blocks)-4),max(0,len(self.model.core.language.blocks)-2)))]
        if added!=expected or not added:raise ValueError('only two existing tail blocks may join optimizer')
        self.names=copy.deepcopy(source_parameter_groups)+[added]
        named=dict(self.model.named_parameters())
        self.optimizer=torch.optim.AdamW([{'params':[named[n] for n in names],'lr':lr,'base_lr':lr}
            for names,lr in zip(self.names,(1e-5,1e-4,3e-6))],weight_decay=.01)
        self.ancestry={};self.migration={}

    def import_parent(self,s,source_plan,source_sha):
        if s['schema']!='fdt_owt_fuzzy_AE_v28_fullstate' or s.get('constant_mask_probability')!=.5 or s['plan_sha256']!=source_plan or s['gradients']:raise ValueError('actual V28 source identity')
        if s.get('resource_policy')!=SOURCE_RESOURCE_POLICY or s.get('training_objective')!=OBJECTIVE or s.get('amp_guard_policy')!=AMP_STOP_POLICY or s.get('loss_reduction')!='elementwise_none_then_sum':raise ValueError('source training/resource contract')
        if s.get('external_closed_assessment_calls')!=self.budget.plan.get('external_closed_assessment_calls'):raise ValueError('source external ledger')
        if s['optimizer_parameter_names']!=self.names[:2] or self.attempts or self.cursor or self.history:raise ValueError('source optimizer/new allocation ownership')
        self.model.load_state_dict(s['model'],strict=True)
        old_names={n for g in self.names[:2] for n in g}
        source_frozen=support.state_sha({n:p for n,p in self.model.named_parameters() if n not in old_names})
        if source_frozen!=s['frozen_sha']:raise ValueError('source frozen scope changed')
        opt=copy.deepcopy(s['optimizer'])
        ids=[i for g in opt['param_groups'] for i in g['params']]
        if ids!=list(range(len(ids))) or len(ids)!=len(old_names) or set(opt['state'])!=set(ids):raise ValueError('complete contiguous old Adam ownership')
        fresh=copy.deepcopy(self.optimizer.state_dict()['param_groups'][2]);fresh['params']=list(range(len(ids),len(ids)+len(self.names[2])))
        opt['param_groups'].append(fresh);self.optimizer.load_state_dict(opt)
        self.scaler.load_state_dict(s['scaler']);self.budget.calls=dict(s['actual_calls'])
        self.frozen_sha=support.state_sha({n:p for n,p in self.model.named_parameters() if not p.requires_grad})
        torch.set_rng_state(s['torch_rng'])
        if str(self.device).startswith('cuda'):torch.cuda.set_rng_state_all(s['cuda_rng'])
        else:self.inactive_cuda_rng=copy.deepcopy(s['cuda_rng'])
        random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
        out=self.export()
        if support.state_sha({k:s[k] for k in UNCHANGED})!=support.state_sha({k:out[k] for k in UNCHANGED}):raise ValueError('model/scaler/allRNG changed')
        if support.state_sha(old_optimizer_view(out['optimizer'],s['optimizer']))!=support.state_sha(s['optimizer']):raise ValueError('old Adam subcomponents changed')
        if any(i in out['optimizer']['state'] for i in fresh['params']):raise ValueError('fresh Adam moments must be empty')
        self.migration={'source_all_component_sha256':support.state_sha({k:s[k] for k in COMPONENTS}),
            'unchanged_model_scaler_rng_sha256':support.state_sha({k:s[k] for k in UNCHANGED}),
            'old_optimizer_sha256':support.state_sha(s['optimizer']),'source_frozen_sha':source_frozen,
            'phase_frozen_sha':self.frozen_sha,'source_parameter_groups':copy.deepcopy(self.names[:2]),
            'new_parameter_names':list(self.names[2]),'old_state_count':len(ids),'new_state_count_at_import':0,
            'source_global_updates':cumulative_count(s,'updates')}
        self.ancestry={k:copy.deepcopy(s[k]) for k in ('plan_sha256','attempts','updates','cursor','orders','history','exposure','actual_calls','evaluations','ancestry')}
        self.ancestry.update(source_sha256=source_sha,source_training_objective=copy.deepcopy(s['training_objective']),
            source_constant_mask_probability=s['constant_mask_probability'],source_frozen_sha=source_frozen,
            imported_component_sha256=self.migration['source_all_component_sha256'],
            external_closed_assessment_calls=copy.deepcopy(s['external_closed_assessment_calls']),
            source_global_attempts=cumulative_count(s,'attempts'),source_global_updates=cumulative_count(s,'updates'))
        self.verify_frozen();self.budget.invalidate_resources();return copy.deepcopy(self.migration)

    def step(self):
        if self.attempts<len(self.orders):self.model.set_training_attempt(self.attempts+1)
        self.model.last_objective_stats=None
        try:return super().step()
        finally:
            stats=self.model.last_objective_stats
            if stats is not None and self.history and self.history[-1]['attempt']==self.attempts:
                self.history[-1].update(copy.deepcopy(stats))
                for k in ('masked_prefix_bytes','eligible_prefix_bytes','decoder_supervised_symbol_occurrences'):self.exposure[k]=self.exposure.get(k,0)+stats[k]

    def export(self):
        state=super().export();state.update(schema='fdt_owt_fuzzy_AE_v29_fullstate',constant_mask_probability=.5,
            training_objective=copy.deepcopy(OBJECTIVE),trainability=copy.deepcopy(TRAINABILITY),ownership_policy=dict(OWNERSHIP),
            migration=copy.deepcopy(self.migration),resource_policy=dict(RESOURCE_POLICY),amp_guard_policy=dict(AMP_STOP_POLICY),
            loss_reduction='elementwise_none_then_sum',ancestry=self.ancestry,
            external_closed_assessment_calls=copy.deepcopy(self.budget.plan.get('external_closed_assessment_calls')),
            allocation_calls=dict(self.budget.allocation_calls))
        return state

    def restore(self,s):
        if s['schema']!='fdt_owt_fuzzy_AE_v29_fullstate' or s.get('constant_mask_probability')!=.5 or s.get('trainability')!=TRAINABILITY or s.get('ownership_policy')!=OWNERSHIP or s.get('resource_policy')!=RESOURCE_POLICY or s.get('training_objective')!=OBJECTIVE or s.get('amp_guard_policy')!=AMP_STOP_POLICY or s.get('loss_reduction')!='elementwise_none_then_sum' or s.get('external_closed_assessment_calls')!=self.budget.plan.get('external_closed_assessment_calls'):raise ValueError('V29 fullstate contract')
        if s['migration'].get('new_parameter_names')!=self.names[2] or s['migration'].get('source_parameter_groups')!=self.names[:2]:raise ValueError('migration ownership')
        phase_frozen=support.state_sha({n:s['model'][n] for n,p in self.model.named_parameters() if not p.requires_grad})
        if phase_frozen!=s['migration']['phase_frozen_sha'] or phase_frozen!=s['frozen_sha']:raise ValueError('phase frozen scope')
        adapted=dict(s);adapted['schema']='fdt_owt_fuzzy_AE_v12_fullstate';super().restore(adapted)
        self.ancestry=copy.deepcopy(s['ancestry']);self.migration=copy.deepcopy(s['migration'])
        self.budget.allocation_calls=dict(s['allocation_calls']);self.budget.invalidate_resources()
        self.model.set_training_attempt(max(1,self.attempts))
