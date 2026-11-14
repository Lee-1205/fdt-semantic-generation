"""Exact finite teacher/live recovery with populated23Adam and enabled scaler."""
import copy,json,random,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
from fuzzy_extraction_engine_v73 import ExtractionEngine
import text_generation_support_v1 as support
def main():
    torch.set_num_threads(2);torch.manual_seed(73);random.seed(73);np.random.seed(73)
    rows=[dict(id='a',prompt='Yara has 7 gold!!'),dict(id='b',prompt='Isolde left at 03:17 with -2.50 kg of flour.'),dict(id='c',prompt='A plain sentence without a numeral. '*2)]
    labels={r['id']:complete_expression_spans(r['prompt'],[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'Yara|Isolde',r['prompt'])]) for r in rows}
    directory=ROOT/'research/artifacts/fuzzy_extraction_20261112_v73_cpu_v2/engine_fixture';directory.mkdir(exist_ok=False);c=Config.smoke()
    def model():
        enc=FuzzyCodecV64(c,input_capacity=256);enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks();return FuzzyRoundTripV64(enc,OrderedFuzzyReadout(c.z_dim,c.value_dim,hidden=128))
    initial=copy.deepcopy(model().state_dict())
    class Budget:
        data_sha='fixture73';parent_reference={'source':'fixture73'};critical_targets=labels
        def __init__(self,label):self.output=directory/label;self.output.mkdir();self.calls=dict(encode=0,readout=0,stream_chunk=0,denoise=0);self.block_work=0;self.fail_next=False
        def check(self,write_bytes=0,emergency=False):pass
        def charge(self,kind):
            if self.fail_next:self.fail_next=False;raise InterruptedError('fixture interruption')
            self.calls[kind]+=1
    def create(label):
        with torch.random.fork_rng(devices=[]):m=model();m.load_state_dict(initial,strict=True)
        return ExtractionEngine(m,rows,[[0],[1],[2]],Budget(label),'fixture_plan73','cpu')
    a=create('a');a.teacher_next();partial=copy.deepcopy(a.export());a.teacher_next();a.teacher_next();complete=copy.deepcopy(a.export())
    b=create('b');b.restore(partial);b.teacher_next();b.teacher_next()
    def normalized(s):
        s=copy.deepcopy(s)
        for r in s['teacher_refs']:r['file']=Path(r['file']).name
        return s
    assert support.state_sha(normalized(b.export()))==support.state_sha(normalized(complete))
    t=create('transition');t.restore(complete);assert support.state_sha(t.export())==support.state_sha(complete)
    a.step();middle=copy.deepcopy(a.export());a.step(True);a.step();expected=a.export()
    restored=create('restored');restored.restore(middle);restored.step(True);restored.step();assert support.state_sha(restored.export())==support.state_sha(expected)
    assert a.attempts==a.cursor==3 and a.updates==2 and a.skips==1 and a.scaler.is_enabled() and len(a.optimizer.state)==23
    assert all(int(v['step'])==2 for v in a.optimizer.state.values()) and a.budget.calls==dict(encode=6,readout=6,stream_chunk=0,denoise=0)
    assert a.exposure['teacher_rows']==a.exposure['live_rows']==3 and a.exposure['live_UTF8_bytes']==sum(len(r['prompt'].encode()) for r in rows)
    a.verify_frozen();assert all(p.grad is None for p in a.model.readout.readout.parameters())
    stop=create('stop');stop.restore(complete);stop.budget.fail_next=True
    try:stop.step();raise AssertionError('fixture stop')
    except InterruptedError:pass
    stopped=copy.deepcopy(stop.export());assert stopped['inflight']['stage']=='source_started' and stopped['attempts']==1 and not stopped['history']
    again=create('stopped_restore');again.restore(stopped);assert support.state_sha(again.export())==support.state_sha(stopped)
    try:again.step();raise AssertionError('no automatic inflight replay')
    except RuntimeError:pass
    try:a.step();raise AssertionError('finite epoch')
    except StopIteration:pass
    assert not torch.cuda.is_initialized()
    proof=dict(status='V73_ACTUAL_ENGINE_CPU_CONTRACT_PASS',partial_teacher_payload_and_state_identical_except_separate_fixture_destination=True,teacher_complete_to_live_exact_fullstate=True,
        populated23Adam_enabledscaler_ALLRNG_exact_nextupdate=True,AMPskip_consumes_row_cursor_exposure=True,fixed_readout7_and_all_nonactive_unchanged=True,
        interrupted_inflight_state_exact_restore_no_automatic_replay=True,finite_epoch_rejects_extra_attempt=True,fixture_teacher_encode5_readout5=True,fixture_live_encode5_readout5=True,
        actual_parent_model_forward=0,CUDA_initialized=False)
    (directory.parent/'root_engine_contract.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
