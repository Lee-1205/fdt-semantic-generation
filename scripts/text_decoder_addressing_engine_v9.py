"""Exact V8 model/moments/RNG import into an independent addressed-memory arm."""
import copy
from text_decoder_alignment_engine_v8 import AlignmentEngine
from text_language_engine_v7 import CurriculumEngine
from research.text_decoder_addressing_v9 import ADDRESS_POLICY
import text_generation_support_v1 as support


class AddressEngine(AlignmentEngine):
    def __init__(self,model,features,targets,negatives,budget,plan_sha,device='cuda',seeds=(81,82,83,84),reconstruction_epochs=4):
        super().__init__(model,features,targets,negatives,budget,plan_sha,device,seeds,reconstruction_epochs)

    def import_parent(self,state,source_plan):
        self.phase_vp_sha=support.state_sha({k[3:]:v for k,v in state['model'].items() if k.startswith('vp.')})
        digest=CurriculumEngine.import_parent(self,state,source_plan)
        self.ancestry['ancestry']=copy.deepcopy(state.get('ancestry',{}))
        self.ancestry['old_component_sha256']=digest
        return digest

    def export(self):
        state=super().export();state.update(schema='fdt_text_decoder_addressing_v9_fullstate',address_policy=ADDRESS_POLICY)
        return state

    def restore(self,state):
        if state['schema']!='fdt_text_decoder_addressing_v9_fullstate' or state['address_policy']!=ADDRESS_POLICY:raise ValueError('addressed fullstate policy')
        adapted=dict(state);adapted['schema']='fdt_text_decoder_alignment_v8_fullstate';super().restore(adapted)
