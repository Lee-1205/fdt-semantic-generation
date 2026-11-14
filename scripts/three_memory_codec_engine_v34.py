"""Unchanged V33 neural function, all91 populated Adam states continued.

New one-epoch broad-data allocation; old clocks/cursors are ancestry, never
reopened. No new module, no denoising, no raw-input decoder bypass.
"""
import copy,random
import numpy as np
import torch
from three_memory_codec_engine_v33 import CodecEngine, CONTRACT, OWNERSHIP
import text_generation_support_v1 as support

SCHEMA='fdt_three_memory_language_codec_v34_fullstate'
CONTINUATION={'neural_function':'unchanged_V33_three_memory_codec','all_populated_Adam_states':91,
    'new_parameters':0,'new_finite_data_epoch':True,'old_allocation_reopened':False,
    'learning_rate_schedule':'same_group_base_LRs_new_finite_cosine_no_moment_reset'}

class BroadCodecEngine(CodecEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.continuation={}

    def import_codec_parent(self,state,source_plan,source_sha):
        if state['schema']!='fdt_three_memory_language_codec_v33_fullstate' or state['plan_sha256']!=source_plan:
            raise ValueError('completed V33 source schema/plan')
        if state['gradients'] or state['optimizer_parameter_names']!=self.names or self.attempts or self.cursor or self.history:
            raise ValueError('same91 ownership and empty new allocation')
        if state['codec_contract']!=CONTRACT or state['ownership_policy']!=OWNERSHIP or state['data_revision']!=self.data_revision:
            raise ValueError('unchanged source codec/data contract')
        if state['external_closed_assessment_calls']!=self.budget.plan.get('external_closed_assessment_calls'):
            raise ValueError('external closed ledger')
        all_ids={i for g in state['optimizer']['param_groups'] for i in g['params']}
        if len(all_ids)!=91 or set(state['optimizer']['state'])!=all_ids:raise ValueError('ALL91 populated Adam')
        self.model.load_state_dict(state['model'],strict=True)
        self.optimizer.load_state_dict(state['optimizer']);self.scaler.load_state_dict(state['scaler'])
        self.frozen_sha=state['frozen_sha'];self.budget.calls=dict(state['actual_calls'])
        torch.set_rng_state(state['torch_rng'])
        if str(self.device).startswith('cuda'):torch.cuda.set_rng_state_all(state['cuda_rng'])
        else:self.inactive_cuda_rng=copy.deepcopy(state['cuda_rng'])
        random.setstate(state['python_rng']);n=state['numpy_rng']
        np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
        self.migration=copy.deepcopy(state['migration'])
        self.ancestry={k:copy.deepcopy(state[k]) for k in ('plan_sha256','attempts','updates','cursor','orders','history',
            'exposure','actual_calls','evaluations','ancestry','external_closed_assessment_calls')}
        self.ancestry.update(source_sha256=source_sha,source_schema=state['schema'],migration=copy.deepcopy(state['migration']))
        parts=('model','optimizer','optimizer_parameter_names','scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')
        source_digest=support.state_sha({k:state[k] for k in parts})
        after=super().export()
        if support.state_sha({k:after[k] for k in parts})!=source_digest:raise ValueError('exact model91Adam/scaler/ALLstoredRNG import')
        self.continuation={'policy':copy.deepcopy(CONTINUATION),'source_sha256':source_sha,'source_plan_sha256':source_plan,
            'imported_component_sha256':source_digest,'source_allocation_calls':copy.deepcopy(state['allocation_calls']),
            'source_worker_stage':state['worker_stage'],'source_memory_epoch':state['memory_epoch']}
        self.verify_frozen();self.budget.invalidate_resources()
        return copy.deepcopy(self.continuation)

    def export(self):
        state=super().export();state.update(schema=SCHEMA,continuation=copy.deepcopy(self.continuation))
        return state

    def restore(self,state):
        if state['schema']!=SCHEMA or state['continuation']['policy']!=CONTINUATION:
            raise ValueError('broad continuation fullstate contract')
        adapted=dict(state);adapted['schema']='fdt_three_memory_language_codec_v33_fullstate'
        super().restore(adapted);self.continuation=copy.deepcopy(state['continuation'])
