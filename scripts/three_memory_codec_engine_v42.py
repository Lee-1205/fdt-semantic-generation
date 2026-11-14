"""Two broad finite epochs; unchanged neural codec and ALL91 Adam states."""
import copy
from three_memory_codec_engine_v33 import CodecEngine,CONTRACT,OWNERSHIP
from three_memory_codec_engine_v34 import BroadCodecEngine

SCHEMA='fdt_three_memory_language_codec_v42_fullstate'
CONTINUATION={'neural_function':'unchanged_V33_three_memory_codec','all_populated_Adam_states':91,
    'new_parameters':0,'old_allocation_reopened':False,'purpose':'two_broad_finite_TRAIN_epochs',
    'learning_rate_schedule':'same_group_base_LRs_new_finite_cosine_no_moment_or_step_reset',
    'source_epoch_completed':True,'new_finite_epochs':2}

class TwoEpochCodecEngine(BroadCodecEngine):
    def import_completed_v41(self,state,source_plan,source_sha):
        if state['schema']!='fdt_three_memory_language_codec_v41_fullstate':raise ValueError('completed V41 source schema')
        if state['worker_stage']!='final_commit' or state['attempts']!=len(state['orders']) or state['cursor']!=sum(map(len,state['orders'])):
            raise ValueError('source must be terminal and all its finite suffix consumed')
        adapted=dict(state);adapted['schema']='fdt_three_memory_language_codec_v33_fullstate'
        super().import_codec_parent(adapted,source_plan,source_sha)
        self.ancestry['source_schema']=state['schema']
        self.ancestry['continuation']=copy.deepcopy(state['continuation'])
        self.continuation['policy']=copy.deepcopy(CONTINUATION)
        return copy.deepcopy(self.continuation)

    def export(self):
        state=CodecEngine.export(self)
        state.update(schema=SCHEMA,continuation=copy.deepcopy(self.continuation))
        return state

    def restore(self,state):
        if state['schema']!=SCHEMA or state['continuation']['policy']!=CONTINUATION:raise ValueError('two_epoch finite epoch recovery contract')
        adapted=dict(state);adapted['schema']='fdt_three_memory_language_codec_v33_fullstate'
        CodecEngine.restore(self,adapted);self.continuation=copy.deepcopy(state['continuation'])
