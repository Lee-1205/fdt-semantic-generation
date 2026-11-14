"""Unchanged96-parameter ownership; all populated Adam states continue exactly."""
import copy
from long_codec_engine_v50 import LongEngine,restore_rng,SCHEMA as OLD_SCHEMA
import text_generation_support_v1 as support

SCHEMA='fdt_prefix_independent_codec_v53_fullstate'
class PrefixEngine(LongEngine):
    def import_parent(self,state):
        if state['optimizer_parameter_names']!=self.names:raise ValueError('unchanged96 ownership')
        self.optimizer.load_state_dict(state['optimizer']);self.scaler.load_state_dict(state['scaler'])
        if support.state_sha(self.optimizer.state_dict())!=support.state_sha(state['optimizer']):raise ValueError('all96 Adam changed')
        self.inactive_cuda_rng=copy.deepcopy(state['cuda_rng'])
        self.ancestry={k:copy.deepcopy(state[k]) for k in ('plan_sha256','attempts','updates','cursor','history','actual_calls','scaler','evaluations','ancestry')}
        restore_rng(state,self.device);self.verify_frozen()

    def step(self,*args,**kwargs):
        record=super().step(*args,**kwargs);record['objective']='half_clean_half_all_previous_bytes_masked';return record

    def export(self):return dict(super().export(),schema=SCHEMA,objective='half_clean_half_all_previous_bytes_masked')

    def restore(self,state):
        if state['schema']!=SCHEMA or state['objective']!='half_clean_half_all_previous_bytes_masked':raise ValueError('objective recovery identity')
        super().restore(dict(state,schema=OLD_SCHEMA))
