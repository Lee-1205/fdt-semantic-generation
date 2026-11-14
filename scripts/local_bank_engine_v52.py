"""Continue all95 populated V50 Adam states; only2 local bank gates are new."""
import copy
import torch
from long_codec_engine_v50 import LongEngine,restore_rng,SCHEMA as OLD_SCHEMA
import text_generation_support_v1 as support

SCHEMA='fdt_local_three_bank_codec_v52_fullstate'
class BankEngine(LongEngine):
    def import_parent(self,state):
        if state['optimizer_parameter_names']!=self.names:raise ValueError('95 source ownership')
        self.optimizer.load_state_dict(state['optimizer']);self.scaler.load_state_dict(state['scaler'])
        self.inactive_cuda_rng=copy.deepcopy(state['cuda_rng'])
        if support.state_sha(self.optimizer.state_dict())!=support.state_sha(state['optimizer']):raise ValueError('old95 Adam changed')
        self.ancestry={k:copy.deepcopy(state[k]) for k in ('plan_sha256','attempts','updates','cursor','history','actual_calls','scaler','evaluations','ancestry')}
        self.ancestry['source_optimizer_digest']=support.state_sha(state['optimizer'])
        named=dict(self.model.named_parameters());new=['memory_codec.local_bank_gains']
        if set(n for n,p in named.items() if p.requires_grad)!={n for g in self.names for n in g}|set(new):
            raise ValueError('complete bank ownership')
        self.optimizer.add_param_group(dict(params=[named[n] for n in new],lr=1e-4,base_lr=1e-4,weight_decay=.01))
        self.names.append(new);restore_rng(state,self.device);self.verify_frozen()

    def export(self):return dict(super().export(),schema=SCHEMA)

    def restore(self,state):
        if state['schema']!=SCHEMA:raise ValueError('V52 recovery schema')
        super().restore(dict(state,schema=OLD_SCHEMA))
