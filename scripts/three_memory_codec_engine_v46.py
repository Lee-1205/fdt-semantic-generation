"""Import V45 exactly, then add one initially inert position parameter."""
import copy

import torch

from codec_resume_v46_contract import POLICY, PROTECTED
from owt_fuzzy_engine_v17_base import AEEngine
from research.position_sensitive_codec_v46 import ADAPTER_NAME
from research.three_memory_contract_v32 import assert_codec_training_only
from three_memory_codec_engine_v34 import BroadCodecEngine
from three_memory_codec_engine_v43 import CONTINUATION as RESTORE_POLICY
from three_memory_codec_engine_v43 import SCHEMA as RESTORE_SCHEMA, TwoEpochCodecEngine
from three_memory_codec_engine_v45 import CodecBudget, SCHEMA as SOURCE_SCHEMA
from codec_resume_v45_contract import POLICY as SOURCE_POLICY

SCHEMA = 'fdt_three_memory_language_codec_v46_fullstate'


class PositionCodecEngine(TwoEpochCodecEngine):
    def __init__(self, model, features, targets, orders, budget, plan_sha, device, *,
                 source_parameter_groups, data_revision):
        AEEngine.__init__(self, model, features, targets, orders, budget, plan_sha, device)
        self.names = copy.deepcopy(source_parameter_groups)
        named = {name: param for name,param in self.model.named_parameters() if param.requires_grad}
        if len(self.names) != 4 or len(named) != 91 or set(named) != {n for group in self.names for n in group}:
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

    def import_completed_v45(self, state, source_plan, source_sha):
        if (state['schema'] != SOURCE_SCHEMA or state['continuation']['policy'] != SOURCE_POLICY
                or state['worker_stage'] != 'final_commit' or state['data_revision'] != self.data_revision
                or state['attempts'] != len(state['orders'])
                or state['cursor'] != sum(map(len, state['orders']))):
            raise ValueError('complete V45 source contract')
        adapted = dict(state, schema='fdt_three_memory_language_codec_v33_fullstate')
        BroadCodecEngine.import_codec_parent(self, adapted, source_plan, source_sha)
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
            raise ValueError('V46 recovery contract')
        if ADAPTER_NAME in state['model'] and self.model.memory_codec.position_adapter is None:
            self.enable_position_adapter()
        continuation = copy.deepcopy(state['continuation'])
        adapted = dict(state, schema=RESTORE_SCHEMA, continuation=dict(continuation, policy=RESTORE_POLICY))
        super().restore(adapted)
        self.continuation = continuation
