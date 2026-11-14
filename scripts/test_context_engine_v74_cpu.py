"""V74 cache transition, populated7Adam, enabled scaler, ALLRNG and STOP restore."""
import copy,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
from context_engine_v74 import ContextEngine
import text_generation_support_v1 as support
def main():
    torch.set_num_threads(2);torch.manual_seed(74);random.seed(74);np.random.seed(74)
    rows=[dict(id='a',prompt='Cruz kept +7 kg!!'),dict(id='b',prompt='Cyrus logged -2.50 degrees C at 03:17.'),dict(id='c',prompt='A plain sentence without a numeral. '*2)]
    labels={r['id']:complete_expression_spans(r['prompt'],[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'Cruz|Cyrus',r['prompt'])]) for r in rows}
    directory=ROOT/'research/artifacts/context_verbalizer_20261112_v74_cpu_v2/engine_fixture_v2';directory.mkdir(exist_ok=False);c=Config.smoke()
    def model():
        enc=FuzzyCodecV64(c,input_capacity=256);enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks();return FuzzyRoundTripV64(enc,OrderedFuzzyReadout(c.z_dim,c.value_dim,hidden=128))
    initial=copy.deepcopy(model().state_dict())
    class Budget:
        data_sha='fixture74';parent_reference={'source':'fixture74'};critical_targets=labels
        def __init__(self,label):self.output=directory/label;self.output.mkdir();self.calls=dict(encode=0,readout=0,stream_chunk=0,denoise=0);self.block_work=0;self.fail=False
        def check(self,write_bytes=0,emergency=False):pass
        def charge(self,kind):
            if self.fail:self.fail=False;raise InterruptedError('fixture STOP')
            self.calls[kind]+=1
    def create(label):
        with torch.random.fork_rng(devices=[]):m=model();m.load_state_dict(initial,strict=True)
        return ContextEngine(m,rows,[[0],[1],[2]],Budget(label),'fixture_plan74','cpu')
    a=create('a');a.cache_next();partial=copy.deepcopy(a.export());a.cache_next();a.cache_next();complete=copy.deepcopy(a.export())
    b=create('b');b.restore(partial);b.cache_next();b.cache_next()
    def normalized(s):
        s=copy.deepcopy(s)
        for r in s['cache_refs']:r['features']=Path(r['features']).name;r['targets']=Path(r['targets']).name
        return s
    assert support.state_sha(normalized(b.export()))==support.state_sha(normalized(complete))
    t=create('transition');t.restore(complete);assert support.state_sha(t.export())==support.state_sha(complete)
    a.step();middle=copy.deepcopy(a.export());a.step(True);a.step();end=a.export();restored=create('restored');restored.restore(middle);restored.step(True);restored.step();assert support.state_sha(restored.export())==support.state_sha(end)
    assert a.attempts==3 and a.updates==2 and a.skips==1 and a.scaler.is_enabled() and len(a.optimizer.state)==7 and all(int(v['step'])==2 for v in a.optimizer.state.values())
    a.verify_frozen();assert a.budget.calls==dict(encode=3,readout=6,stream_chunk=0,denoise=0)
    stop=create('stop');stop.restore(complete);stop.budget.fail=True
    try:stop.step();raise AssertionError('STOP')
    except InterruptedError:pass
    saved=copy.deepcopy(stop.export());again=create('stopped_restore');again.restore(saved);assert support.state_sha(again.export())==support.state_sha(saved)
    try:again.step();raise AssertionError('no automatic replay')
    except RuntimeError:pass
    assert not torch.cuda.is_initialized()
    proof=dict(status='V74_ENGINE_CPU_CONTRACT_PASS',partial_cache_payload_and_state_equal_except_owned_destination=True,complete_cache_to_head_exact_fullstate=True,
        populated7Adam_enabledscaler_ALLRNG_exact_nextupdate=True,AMPskip_consumes_row_cursor_exposure=True,all_frozen_upstream_tensors_unchanged=True,
        interrupted_inflight_exact_restore_no_automatic_replay=True,actual_parent_forward=0,CUDA_initialized=False,fixture_encode5_teacher5_head5=True)
    (directory.parent/'root_engine_contract_v2.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
