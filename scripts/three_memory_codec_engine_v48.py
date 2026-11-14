"""Continue V47's 92 Adam states and add two zero-initialized local-read tensors."""
import copy

from codec_resume_v47_contract import POLICY as SOURCE_POLICY
from codec_resume_v48_contract import POLICY, PROTECTED
from research.wide_context_codec_v48 import LOCAL_GAIN, LOCAL_TAG
from three_memory_codec_engine_v47 import CodecBudget, PositionCodecEngine, SCHEMA as SOURCE_SCHEMA

SCHEMA = 'fdt_three_memory_language_codec_v48_fullstate'


class WideCodecEngine(PositionCodecEngine):
    def import_completed_v47(self, state, source_plan, source_sha):
        if (state['schema'] != SOURCE_SCHEMA or state['continuation']['policy'] != SOURCE_POLICY
                or state['worker_stage'] != 'final_commit' or state['data_revision'] != self.data_revision
                or state['attempts'] != len(state['orders'])
                or state['cursor'] != sum(map(len, state['orders']))):
            raise ValueError('complete V47 source contract')
        adapted = dict(state, schema='fdt_three_memory_language_codec_v33_fullstate')
        self.import_unchanged_parameters(adapted, source_plan, source_sha)
        self.ancestry.update(source_schema=state['schema'], source_data_revision=state['data_revision'],
                             continuation=copy.deepcopy(state['continuation']))
        self.continuation.update(policy=copy.deepcopy(POLICY),
            source_terminal_group_lrs=[float(group['lr']) for group in state['optimizer']['param_groups']])
        self._terminal_lrs()
        return copy.deepcopy(self.continuation)

    def enable_local_read(self):
        parameters = self.model.decoder.enable_local_read()
        names = [LOCAL_GAIN, LOCAL_TAG]
        self.optimizer.param_groups[-1]['params'].extend(parameters)
        self.names[-1].extend(names)
        if 'new_parameter_names' in self.migration:
            self.migration['new_parameter_names'].extend(names)

    def export(self):
        return dict(super().export(), schema=SCHEMA, protected_free_bytes=PROTECTED)

    def restore(self, state):
        if (state['schema'] != SCHEMA or state['continuation']['policy'] != POLICY
                or state['protected_free_bytes'] != PROTECTED):
            raise ValueError('V48 recovery contract')
        if LOCAL_GAIN in state['model'] and self.model.decoder.local_read_gain is None:
            self.enable_local_read()
        continuation = copy.deepcopy(state['continuation'])
        adapted = dict(state, schema=SOURCE_SCHEMA,
                       continuation=dict(continuation, policy=SOURCE_POLICY))
        super().restore(adapted)
        self.continuation = continuation
