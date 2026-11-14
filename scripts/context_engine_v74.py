"""V72 fixed-upstream machinery, explicit V74 state and interrupted-attempt ledger."""
import copy,random
import numpy as np
import torch
from frozen_verbalizer_engine_v72 import FrozenEngine
from research.fuzzy_roundtrip_v64 import EXACT_POLICY
from research.frozen_verbalizer_objective_v72 import OBJECTIVE
import text_generation_support_v1 as support
SCHEMA='fdt_context_verbalizer_v74_fullstate'
class ContextEngine(FrozenEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.inflight=None
    def step(self,force_overflow=False):
        if self.inflight is not None:raise RuntimeError('interrupted V74 attempt preserved; no automatic replay')
        self.inflight=dict(requested_attempt=self.attempts+1,stage='base_step_requested')
        try:
            record=super().step(force_overflow);self.inflight=None;return record
        except BaseException:
            if self.attempts==len(self.history):self.inflight=None
            else:self.inflight.update(stage='interrupted_base_step',consumed_attempt=self.attempts,rows=self.orders[self.attempts-1])
            raise
    def export(self):
        s=super().export();s.update(schema=SCHEMA,inflight=copy.deepcopy(self.inflight),long_block_work=getattr(self.budget,'block_work',0));return s
    def restore(self,s):
        if (s['schema']!=SCHEMA or s['objective']!=OBJECTIVE or s['objective_report']!=self.objective_report or not torch.equal(s['symbol_weights'],self.symbol_weights.cpu())
            or s['plan_sha256']!=self.plan_sha or s['exact_policy']!=EXACT_POLICY or s['orders']!=self.orders or s['optimizer_names']!=self.optimizer_names or s['parent_reference']!=self.budget.parent_reference
            or s['input_capacity']!=self.model.encoder.input_capacity or s['embedding_sha']!=self.embedding_sha or s['teacher_sha']!=self.teacher_sha or s['gradients']
            or s['cache_cursor']!=len(s['cache_refs']) or s['cache_cursor']>len(self.rows) or s['cursor']!=s['attempts'] or [h['rows'] for h in s['history']]!=self.orders[:len(s['history'])]
            or len(s['history'])!=s['attempts']-int(s['inflight'] is not None) or (s['attempts'] and s['cache_cursor']!=len(self.rows))):raise ValueError('V74 full phase recovery identity')
        self.model.load_state_dict(s['model'],strict=True);self.optimizer.load_state_dict(s['optimizer']);self.scaler.load_state_dict(s['scaler'])
        for k in ('cache_cursor','cache_refs','new_dev_refs','attempts','updates','cursor','skips','phase','history','exposure','evaluations','inflight'):setattr(self,k,copy.deepcopy(s[k]))
        self.budget.calls=copy.deepcopy(s['calls']);self.inactive_cuda_rng=copy.deepcopy(s['cuda_rng']);self.budget.block_work=s.get('long_block_work',0);self.verify_frozen()
        if not self.attempts and support.state_sha(self.model.readout.readout.state_dict())!=self.teacher_sha:raise ValueError('restored teacher mismatch')
        for i in range(self.cache_cursor):self.packet(i)
        for i,ref in enumerate(self.new_dev_refs):
            if support.sha(ref['features'])!=ref['features_SHA']:raise ValueError('DEV cache changed')
            f=torch.load(ref['features'],map_location='cpu',weights_only=True)
            if f['embedding_sha']!=self.embedding_sha or f['cache_key']!=self.source_key(self.budget.eval_rows[i]):raise ValueError('DEV source epoch')
        torch.set_rng_state(s['torch_rng'])
        if self.device=='cuda':torch.cuda.set_rng_state_all(s['cuda_rng'])
        random.setstate(s['python_rng']);n=s['numpy_rng'];np.random.set_state((n['name'],np.array(n['keys'],dtype=np.uint32),n['position'],n['gaussian'],n['cached']))
