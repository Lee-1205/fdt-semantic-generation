"""Import V45 exactly, then add one initially inert position parameter."""
import copy
import random
import numpy as np

import torch

from codec_resume_v47_contract import POLICY, PROTECTED
from owt_fuzzy_engine_v17_base import AEEngine
from research.position_sensitive_codec_v46 import ADAPTER_NAME
from research.three_memory_contract_v32 import assert_codec_training_only
import text_generation_support_v1 as support
from three_memory_codec_engine_v33 import CodecEngine, CONTRACT, OWNERSHIP
from three_memory_codec_engine_v43 import CONTINUATION as RESTORE_POLICY
from three_memory_codec_engine_v43 import SCHEMA as RESTORE_SCHEMA, TwoEpochCodecEngine
from three_memory_codec_engine_v45 import CodecBudget
from three_memory_codec_engine_v46 import SCHEMA as SOURCE_SCHEMA
from codec_resume_v46_contract import POLICY as SOURCE_POLICY, TRAIN_DATA as OLD_DATA

SCHEMA = 'fdt_three_memory_language_codec_v47_fullstate'


class PositionCodecEngine(TwoEpochCodecEngine):
    def __init__(self, model, features, targets, orders, budget, plan_sha, device, *,
                 source_parameter_groups, data_revision):
        AEEngine.__init__(self, model, features, targets, orders, budget, plan_sha, device)
        self.names = copy.deepcopy(source_parameter_groups)
        named = {name: param for name,param in self.model.named_parameters() if param.requires_grad}
        if len(self.names) != 4 or len(named) != 92 or set(named) != {n for group in self.names for n in group}:
            raise ValueError('unchanged V45 optimizer ownership before migration')
        lrs = (1e-5, 1e-4, 3e-6, 1e-4)
        self.optimizer = torch.optim.AdamW([
            {'params': [named[n] for n in group], 'lr': lr, 'base_lr': lr}
            for group, lr in zip(self.names, lrs)
        ], weight_decay=.01)
        self.data_revision = data_revision
        self.migration = {}
        self.ancestry = {}
        self.continuation = {}
        self.worker_stage = 'new_allocation'
        assert_codec_training_only(model.named_parameters())

    def import_unchanged_parameters(self,state,source_plan,source_sha):
        if state['schema']!='fdt_three_memory_language_codec_v33_fullstate' or state['plan_sha256']!=source_plan:
            raise ValueError('completed V33 source schema/plan')
        if state['gradients'] or state['optimizer_parameter_names']!=self.names or self.attempts or self.cursor or self.history:
            raise ValueError('same92 ownership and empty new allocation')
        if state['codec_contract']!=CONTRACT or state['ownership_policy']!=OWNERSHIP or state['data_revision']!=self.data_revision:
            raise ValueError('unchanged source codec/data contract')
        if state['external_closed_assessment_calls']!=self.budget.plan.get('external_closed_assessment_calls'):
            raise ValueError('external closed ledger')
        all_ids={i for g in state['optimizer']['param_groups'] for i in g['params']}
        if len(all_ids)!=92 or set(state['optimizer']['state'])!=all_ids:raise ValueError('ALL92 populated Adam')
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
        after=CodecEngine.export(self)
        if support.state_sha({k:after[k] for k in parts})!=source_digest:raise ValueError('exact model92Adam/scaler/ALLstoredRNG import')
        self.continuation={'policy':copy.deepcopy(POLICY),'source_sha256':source_sha,'source_plan_sha256':source_plan,
            'imported_component_sha256':source_digest,'source_allocation_calls':copy.deepcopy(state['allocation_calls']),
            'source_worker_stage':state['worker_stage'],'source_memory_epoch':state['memory_epoch']}
        self.verify_frozen();self.budget.invalidate_resources()
        return copy.deepcopy(self.continuation)

    def import_completed_v46(self, state, source_plan, source_sha):
        if (state['schema'] != SOURCE_SCHEMA or state['continuation']['policy'] != SOURCE_POLICY
                or state['worker_stage'] != 'final_commit' or state['data_revision'] != support.sha(OLD_DATA/'manifest.json')
                or state['attempts'] != len(state['orders'])
                or state['cursor'] != sum(map(len, state['orders']))):
            raise ValueError('complete V45 source contract')
        adapted = dict(state, schema='fdt_three_memory_language_codec_v33_fullstate',data_revision=self.data_revision)
        self.import_unchanged_parameters(adapted, source_plan, source_sha)
        self.ancestry.update(source_schema=state['schema'], source_data_revision=state['data_revision'],
                             continuation=copy.deepcopy(state['continuation']))
        self.continuation.update(policy=copy.deepcopy(POLICY),
            source_terminal_group_lrs=[float(group['lr']) for group in state['optimizer']['param_groups']])
        self._terminal_lrs()
        return copy.deepcopy(self.continuation)

    def enable_position_adapter(self):
        parameter = self.model.enable_position_adapter()
        self.optimizer.param_groups[-1]['params'].append(parameter)
        self.names[-1].append(ADAPTER_NAME)
        if 'new_parameter_names' in self.migration:
            self.migration['new_parameter_names'].append(ADAPTER_NAME)
        return parameter

    def export(self):
        return dict(super().export(), schema=SCHEMA, protected_free_bytes=PROTECTED)

    def restore(self, state):
        if (state['schema'] != SCHEMA or state['continuation']['policy'] != POLICY
                or state['protected_free_bytes'] != PROTECTED):
            raise ValueError('V47 recovery contract')
        if ADAPTER_NAME in state['model'] and self.model.memory_codec.position_adapter is None:
            self.enable_position_adapter()
        continuation = copy.deepcopy(state['continuation'])
        adapted = dict(state, schema=RESTORE_SCHEMA, continuation=dict(continuation, policy=RESTORE_POLICY))
        super().restore(adapted)
        self.continuation = continuation
