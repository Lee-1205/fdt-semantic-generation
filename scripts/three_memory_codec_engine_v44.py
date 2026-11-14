"""Continue all 91 Adam states on the V44 TRAIN schedule."""
import copy
import shutil
import time

from codec_resume_v44_contract import GIB, POLICY, PROTECTED, STOP_FREE
from owt_resource_budget_v29 import RecoveryBudget, RESOURCE_POLICY
from text_language_runtime_support_v6 import owned_bytes
from three_memory_codec_engine_v34 import BroadCodecEngine
from three_memory_codec_engine_v43 import CONTINUATION as PARENT_POLICY
from three_memory_codec_engine_v43 import SCHEMA as PARENT_SCHEMA, TwoEpochCodecEngine

SCHEMA = 'fdt_three_memory_language_codec_v44_fullstate'


class CodecBudget(RecoveryBudget):
    def __init__(self, plan):
        super().__init__(dict(plan, resource_policy=RESOURCE_POLICY))
        self.plan = plan
        self.minimum_free_bytes = None

    def check_resources(self, force=False, upcoming_bytes=0, *, emergency=False):
        if upcoming_bytes < 0:
            raise ValueError('negative write estimate')
        now = time.monotonic()
        if not force and not upcoming_bytes and self.last_resource_check is not None:
            if now-self.last_resource_check < 1:
                return
        free = shutil.disk_usage(self.output).free
        self.minimum_free_bytes = min(self.minimum_free_bytes or free, free)
        self.resource_scans += 1
        if free < (PROTECTED if emergency else STOP_FREE) + upcoming_bytes:
            raise OSError('C: protected 100GiB floor plus upcoming write')
        if owned_bytes(self.output) + upcoming_bytes >= self.plan['limits']['max_output_bytes']:
            raise OSError('V44 owned output cap')
        self.last_resource_check = now

    def check_write(self, size, emergency=False):
        if emergency:
            self.check_resources(True, size, emergency=True)
        else:
            self.check(True, size)


class FreshSpanCodecEngine(TwoEpochCodecEngine):
    def import_completed_v43(self, state, source_plan, source_sha, parent_data_revision):
        if (state['schema'] != PARENT_SCHEMA or state['continuation']['policy'] != PARENT_POLICY
                or state['worker_stage'] != 'final_commit' or state['data_revision'] != parent_data_revision
                or state['attempts'] != len(state['orders'])
                or state['cursor'] != sum(map(len, state['orders']))):
            raise ValueError('complete V43 source contract')
        # The parent importer verifies the neural/optimizer/RNG state. V44
        # changes only the finite data selection and the storage guard.
        adapted = dict(state, schema='fdt_three_memory_language_codec_v33_fullstate',
                       data_revision=self.data_revision)
        BroadCodecEngine.import_codec_parent(self, adapted, source_plan, source_sha)
        self.ancestry.update(source_schema=state['schema'], source_data_revision=state['data_revision'],
                             continuation=copy.deepcopy(state['continuation']))
        self.continuation.update(policy=copy.deepcopy(POLICY),
            source_terminal_group_lrs=[float(group['lr']) for group in state['optimizer']['param_groups']])
        self._terminal_lrs()
        return copy.deepcopy(self.continuation)

    def export(self):
        return dict(super().export(), schema=SCHEMA, protected_free_bytes=PROTECTED)

    def restore(self, state):
        if (state['schema'] != SCHEMA or state['continuation']['policy'] != POLICY
                or state['protected_free_bytes'] != PROTECTED):
            raise ValueError('V44 recovery contract')
        continuation = copy.deepcopy(state['continuation'])
        adapted = dict(state, schema=PARENT_SCHEMA,
                       continuation=dict(continuation, policy=PARENT_POLICY))
        super().restore(adapted)
        self.continuation = continuation
