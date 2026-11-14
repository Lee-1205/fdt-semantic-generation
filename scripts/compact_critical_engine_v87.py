"""V86 math unchanged; immutable bank references avoid checkpoint duplication."""
import copy
from pathlib import Path
from critical_bank_engine_v86 import CriticalBankEngine,SCHEMA as ENGINE_SCHEMA
from research import critical_bank_storage_v87 as storage
from run_semantic_binding_20261111_v61 import save_tensor
import text_generation_support_v1 as support

SCHEMA='fdt_fullFP32_critical_engine_immutable_bank_v87'
ROOT=Path(__file__).resolve().parents[1]

class CompactCriticalEngine(CriticalBankEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.bank_ref=None

    def commit_reference_files(self):
        if self.inflight is not None:raise RuntimeError('interrupted file publication; preserve state, no automatic replay')
        if not self.bank_ready or self.bank_ref is not None or self.attempts:raise ValueError('one immutable reference publication before head')
        self.inflight=dict(stage='reference_files_requested');features,targets=storage.packets(self.bank);folder=self.budget.output/'TRAIN_reference';folder.mkdir(exist_ok=True)
        f=folder/'bank.features.pt';t=folder/'bank.targets.pt';fs=save_tensor(f,features,self.budget);self.inflight['stage']='reference_features_saved'
        ts=save_tensor(t,targets,self.budget);reference=dict(features=str(f),features_SHA=fs,targets=str(t),targets_SHA=ts,bank_runtime_SHA=support.state_sha(self.bank.export()))
        rebuilt=storage.load(reference,ROOT,support.sha)
        if support.state_sha(rebuilt)!=reference['bank_runtime_SHA']:raise ValueError('bank exact reconstruction before phase transition')
        self.bank_ref=reference;self.inflight=None;return copy.deepcopy(reference)

    def step(self,force_overflow=False):
        if self.bank_ref is None:raise ValueError('reference files not durably verified')
        return super().step(force_overflow)

    def export(self):
        state=super().export();state['schema']=SCHEMA;state['bank_ref']=copy.deepcopy(self.bank_ref)
        if self.bank_ref is not None:state['bank_state']=None
        return state

    def restore(self,state):
        if state['schema']!=SCHEMA:raise ValueError('immutable-bank runtime schema')
        reference=state['bank_ref'];mapped=dict(state,schema=ENGINE_SCHEMA)
        if reference is not None:
            if state['bank_state'] is not None:raise ValueError('exactly one bank storage representation')
            mapped['bank_state']=storage.load(reference,ROOT,support.sha)
            if support.state_sha(mapped['bank_state'])!=reference['bank_runtime_SHA']:raise ValueError('immutable full bank digest')
        elif state['bank_state'] is None:raise ValueError('missing partial bank state')
        super().restore(mapped);self.bank_ref=copy.deepcopy(reference)
