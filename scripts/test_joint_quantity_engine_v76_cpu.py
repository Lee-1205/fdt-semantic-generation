"""New live-source phase recovery and parent-linked state, tiny CPU only."""
import copy,json,random,re,sys,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
from research import linked_runtime_v75 as linked
from joint_quantity_engine_v76 import RetentionEngine
import text_generation_support_v1 as support

def main():
    started=time.monotonic()
    if shutil.disk_usage(ROOT).free<100*1024**3+32*1024**2:
        raise OSError('tiny CPU fixture needs100GiB protected+32MiB reserve')
    torch.set_num_threads(2);torch.manual_seed(76);random.seed(76);np.random.seed(76)
    rows=[dict(id='a',prompt='Cruz stored +7 kg!!'),dict(id='b',prompt='Cyrus logged -2.50 degrees C at 03:17.'),dict(id='c',prompt='A plain sentence without a numeral. '*2)]
    labels={r['id']:complete_expression_spans(r['prompt'],[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'Cruz|Cyrus',r['prompt'])]) for r in rows}
    directory=ROOT/'research/artifacts/joint_quantity_20261113_v76_cpu_v1/engine_fixture_v2';directory.mkdir(exist_ok=False);config=Config.smoke()
    def model():
        e=FuzzyCodecV64(config,input_capacity=256);e.enable_position_adapter();e.enable_long_capacity();e.enable_local_banks();return FuzzyRoundTripV64(e,OrderedFuzzyReadout(config.z_dim,config.value_dim,hidden=128))
    initial=copy.deepcopy(model().state_dict());parent=directory/'parent.pt';torch.save(dict(model=initial),parent)
    reference=dict(path=str(parent.resolve()),SHA=linked.file_sha(parent),model_key='model',model_SHA=support.state_sha(initial))
    class Budget:
        data_sha='fixture76';parent_reference={'source':'fixture76'};critical_targets=labels
        def __init__(self,label):self.output=directory/label;self.output.mkdir();self.calls=dict(encode=0,readout=0,stream_chunk=0,denoise=0);self.block_work=0;self.fail=False
        def check(self,write_bytes=0,emergency=False):
            if time.monotonic()-started>60:raise TimeoutError('bounded tiny CPU contract60sec')
            if shutil.disk_usage(ROOT).free<100*1024**3+write_bytes:raise OSError('100GiB CPU write guard')
            owned=sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
            if owned+write_bytes>32*1024**2:raise OSError('32MiB fixture output cap including pending write')
        def charge(self,kind):
            if self.fail:self.fail=False;raise InterruptedError('fixture STOP')
            self.calls[kind]+=1
    def create(label):
        with torch.random.fork_rng(devices=[]):m=model();m.load_state_dict(initial,strict=True)
        return RetentionEngine(m,rows,[[0],[1],[2]],Budget(label),'fixture_plan76','cpu')
    a=create('a');a.cache_next();partial=copy.deepcopy(a.export());a.cache_next();a.cache_next();complete=copy.deepcopy(a.export())
    b=create('b');b.restore(partial);b.cache_next();b.cache_next()
    def normalized(s):
        s=copy.deepcopy(s)
        for r in s['cache_refs']:r['targets']=Path(r['targets']).name
        return s
    assert support.state_sha(normalized(b.export()))==support.state_sha(normalized(complete))
    transition=create('transition');transition.restore(complete);assert support.state_sha(transition.export())==support.state_sha(complete)
    assert not list(a.budget.output.rglob('*.features.pt'))
    a.step();middle=copy.deepcopy(a.export());env=linked.pack(middle,initial,reference,a.names,support.state_sha)
    a.budget.check(write_bytes=16*1024**2)
    meta=linked.save_verified(directory/'populated_runtime.pt',env,ROOT,support.state_sha)
    linked_state=linked.load_verified(meta['path'],meta['SHA'],ROOT,support.state_sha);assert support.state_sha(linked_state)==support.state_sha(middle)
    a.step(True);a.step();expected=a.export();restored=create('restored');restored.restore(linked_state);restored.step(True);restored.step()
    assert support.state_sha(restored.export())==support.state_sha(expected)
    assert a.attempts==a.cursor==3 and a.updates==2 and a.skips==1 and a.scaler.is_enabled() and len(a.optimizer.state)==7
    assert all(int(v['step'])==2 for v in a.optimizer.state.values()) and a.budget.calls==dict(encode=6,readout=6,stream_chunk=0,denoise=0)
    assert a.exposure['encoder_rows']==a.exposure['head_rows']==a.exposure['head_encoder_rows']==3
    assert a.exposure['head_encoder_UTF8_bytes']==a.exposure['head_UTF8_bytes']==sum(len(r['prompt'].encode()) for r in rows)
    a.verify_frozen()
    for stage in ('teacher','head'):
        stopped=create('stop_'+stage)
        if stage=='head':stopped.restore(complete)
        stopped.budget.fail=True
        try:
            stopped.cache_next() if stage=='teacher' else stopped.step();raise AssertionError('expected STOP')
        except InterruptedError:pass
        saved=copy.deepcopy(stopped.export());again=create('restored_stop_'+stage);again.restore(saved)
        assert support.state_sha(again.export())==support.state_sha(saved)
        try:
            again.cache_next() if stage=='teacher' else again.step();raise AssertionError('automatic replay forbidden')
        except RuntimeError:pass
    bad=copy.deepcopy(middle);bad['embedding_sha']='wrong'
    try:restored.restore(bad);raise AssertionError('source epoch')
    except ValueError:pass
    assert a.objective_report['wrong_critical_max_position_amplification']==8.
    assert a.objective_report['wrong_critical_denominator']=='max(W,C/8)'
    assert all(h['wrong_critical_position_amplification']<=8. for h in a.history)
    assert not torch.cuda.is_initialized()
    proof=dict(status='V76_CAPPED_LIVE_ENGINE_CPU_CONTRACT_PASS',partial_teacher_packet_state_equal_except_owned_path=True,complete_teacher_to_live_head_exact_fullstate=True,
        no_feature_cache_written=True,head_source_actually_reencoded_including_skipped_attempt=True,populated7Adam_enabledscaler_ALLRNG_linked_exact_nextupdate=True,
        AMPskip_consumes_row_order_and_exposure=True,all_upstream_tensors_frozen=True,teacher_and_head_inflight_STOP_exact_restore_no_replay=True,
        wrong_source_epoch_rejected=True,source_model_actual_forward=0,CUDA_initialized=False,one_complete_fixture_calls=dict(encode=6,readout=6),V76_capped_objective_wired=True,maximum_wrong_critical_position_amplification=8.,actual_fullsize_parent_used=False,actual_worker_ready=False,elapsed_seconds=time.monotonic()-started,fixture_output_bytes=sum(f.stat().st_size for f in directory.rglob('*') if f.is_file()),code_SHA={str(f):support.sha(f) for f in [ROOT/'scripts/joint_quantity_engine_v76.py',ROOT/'research/critical_retention_objective_v76.py',ROOT/'research/linked_runtime_v75.py',Path(__file__)]})
    (directory.parent/'root_engine_contract_v1.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof))
if __name__=='__main__':main()
