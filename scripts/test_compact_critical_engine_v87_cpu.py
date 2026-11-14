"""Bounded CPU integration: tiny encoder, actual512wide final Linear, no GPU."""
import copy,io as memoryio,json,random,re,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.fp32_ordered_readout_v86 import FP32OrderedFuzzyReadout
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
from research import linked_lastlinear_runtime_v79 as linked
from research import linked_runtime_v75 as source_storage
from research.actual_update_retention_v80 import TrainDecision
from compact_critical_engine_v87 import CompactCriticalEngine as CriticalBankEngine
import text_generation_support_v1 as support
import learning_io_v2 as io

def main():
    began=time.monotonic();torch.set_num_threads(2);torch.manual_seed(86);random.seed(86);np.random.seed(86)
    directory=ROOT/'research/artifacts/compact_critical_engine_20261113_v87_cpu_v1';directory.mkdir(exist_ok=False)
    rows=[dict(id='CPU_TRAIN:u',prompt='Cruz stored 27 pins.'),dict(id='CPU_TRAIN:s',prompt='Cyrus logged +2.50 kg at 03:17.'),dict(id='CPU_TRAIN:o',prompt='A plain sentence without a numeral.')]
    labels={r['id']:complete_expression_spans(r['prompt'],[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'Cruz|Cyrus',r['prompt'])]) for r in rows}
    config=Config.smoke();orders=[[0,1,2,0,1]]*3;profile=dict(weights=[1.]*257+[0.],source_SHA='CPU_synthetic_symbol_profile')
    def model():
        e=FuzzyCodecV64(config,input_capacity=256);e.enable_position_adapter();e.enable_long_capacity();e.enable_local_banks()
        return FuzzyRoundTripV64(e,FP32OrderedFuzzyReadout(config.z_dim,config.value_dim,hidden=512))
    initial=copy.deepcopy(model().state_dict());storage=copy.deepcopy(initial);initial['readout.readout.net.0.bias'].add_(.125)
    parent=ROOT/'research/artifacts/critical_bank_engine_20261113_v86_cpu_v1/CPU_parent.pt'
    assert support.state_sha(torch.load(parent,map_location='cpu',weights_only=True)['model'])==support.state_sha(storage)
    parentref=dict(path=str(parent.resolve()),SHA=support.sha(parent),model_key='model',model_SHA=support.state_sha(storage))
    class Budget:
        data_sha='CPU_v86';parent_reference={'source':'CPU_v86'};parent_SHA='f'*64;anchor_indices=[0,1,2];critical_targets=labels;frozen_symbol_profile=profile
        def __init__(self,label):
            self.output=directory/label;self.output.mkdir();self.calls=dict(encode=0,readout=0,reference_linear=0,denoise=0);self.block_work=0;self.fail_at=None
        def check(self,write_bytes=0,emergency=False):
            if time.monotonic()-began>90:raise TimeoutError('fixed CPU90sec contract')
            if shutil.disk_usage(ROOT).free<100*1024**3+write_bytes:raise OSError('100GiB plus pending write protected')
            if sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())+write_bytes>96*1024**2:raise OSError('fixed CPU96MiB output cap')
        def charge(self,kind):
            self.check()
            if sum(self.calls.values())>=128:raise RuntimeError('fixed CPU128call cap per fixture engine')
            if self.fail_at is not None and sum(self.calls.values())>=self.fail_at:self.fail_at=None;raise InterruptedError('CPU cooperative STOP')
            self.calls[kind]+=1
    def create(name):
        with torch.random.fork_rng(devices=[]):m=model();m.load_state_dict(initial,strict=True)
        return CriticalBankEngine(m,rows,orders,Budget(name),'CPU_plan_v86','cpu')
    def norm(s):
        s=copy.deepcopy(s)
        for ref in s['cache_refs']:ref['targets']=Path(ref['targets']).name
        if s['bank_ref'] is not None:
            for key in ('features','targets'):s['bank_ref'][key]=Path(s['bank_ref'][key]).name
        return s
    a=create('a');a.cache_next();partial=copy.deepcopy(a.export());a.cache_next();a.cache_next();a.initialize_reference();a.commit_reference_files();complete=copy.deepcopy(a.export())
    b=create('b');b.restore(partial);b.cache_next();b.cache_next();b.initialize_reference();b.commit_reference_files();assert support.state_sha(norm(b.export()))==support.state_sha(norm(complete))
    c=create('boundary');c.restore(complete);assert support.state_sha(c.export())==support.state_sha(complete)
    features,_=a.bank.components()
    with torch.no_grad():z=a.model.readout.readout.net[2](features['hidden']).reshape(-1,16,258)
    assert a.bank.decision(z)==a._device_reference.decision(z)
    a.step();middle=copy.deepcopy(a.export());assert len(a.optimizer.state)==2 and a.scaler.is_enabled() and a.updates==1
    try:source_storage.pack(middle,storage,parentref,a.names,support.state_sha);raise AssertionError('would lose frozen5prefix')
    except ValueError:pass
    envelope=linked.pack(middle,storage,parentref,support.state_sha);stream=memoryio.BytesIO();torch.save(envelope,stream);stream.seek(0);saved=linked.unpack(torch.load(stream,map_location='cpu',weights_only=True),ROOT,support.state_sha)
    io.atomic_json(directory/'CPU_inmemory_fullstate_proof.json',dict(envelope_bytes=len(stream.getvalue()),algorithm_SHA=support.state_sha(saved),source_parent_SHA=support.sha(parent),purpose='Exact safeload/reconstruction without duplicate synthetic weight files; original physical revision1 runtime preserved.'))
    # Meaningful RNG consumption outside deterministic model eval demonstrates
    # that serialized ALL RNG, rather than model determinism, controls recovery.
    r1=(torch.rand(3),random.random(),np.random.rand());a.step(True);a.step();expected=copy.deepcopy(a.export())
    restored=create('restored');restored.restore(saved);r2=(torch.rand(3),random.random(),np.random.rand());assert torch.equal(r1[0],r2[0]) and r1[1:]==r2[1:]
    restored.step(True);restored.step();assert support.state_sha(restored.export())==support.state_sha(expected)
    assert a.attempts==a.cursor==3 and a.skips==1 and a.updates==2 and a.rejections==0 and all(int(v['step'])==2 for v in a.optimizer.state.values())
    assert a.budget.calls['encode']==a.budget.calls['readout']==18 and a.exposure['head_rows']==a.exposure['head_encoder_rows']==15
    assert all(h['clip_and_Adam_calls']==1 and len(h['micros'])==5 and abs(h['loss']-sum(v['loss']*.2 for v in h['micros']))<1e-8 for h in a.history)
    a.verify_frozen()
    rejected=create('rejected');rejected.restore(middle);old=copy.deepcopy(rejected.export());actual=rejected.reference_decision
    def reject():
        v=actual();return TrainDecision(False,dict(v.measurements,CPU_forced_reject=1.))
    rejected.reference_decision=reject;rejected.step();after=copy.deepcopy(rejected.export())
    assert support.state_sha(old['model'])==support.state_sha(after['model']) and support.state_sha(old['optimizer'])==support.state_sha(after['optimizer'])
    assert after['attempts']==2 and after['rejections']==1 and after['scaler']['_growth_tracker']==old['scaler']['_growth_tracker']+1 and len(after['history'][-1]['update_receipt']['reference_checks'])==4
    reject_restore=create('reject_restore');reject_restore.restore(after);assert support.state_sha(reject_restore.export())==support.state_sha(after)
    stop_proofs={}
    for phase in ('teacher','teacher_packet_saved','head_one_micro','proposal'):
        e=create('STOP_'+phase)
        if phase.startswith('head') or phase=='proposal':e.restore(complete)
        if phase=='teacher_packet_saved':
            def stop_ingest(*args,**kwargs):raise InterruptedError('CPU STOP after actual teacher packet, before bank ingest')
            e.bank.ingest=stop_ingest
        else:e.budget.fail_at=sum(e.budget.calls.values())+(2 if phase=='head_one_micro' else 10 if phase=='proposal' else 0)
        try:e.cache_next() if phase.startswith('teacher') else e.step();raise AssertionError('expected STOP')
        except InterruptedError:pass
        snapshot=copy.deepcopy(e.export());again=create('restore_STOP_'+phase);again.restore(snapshot);assert support.state_sha(again.export())==support.state_sha(snapshot)
        try:again.cache_next() if phase.startswith('teacher') else again.step();raise AssertionError('automatic replay forbidden')
        except RuntimeError:pass
        if phase=='teacher_packet_saved':assert len(snapshot['cache_refs'])==1 and snapshot['cache_cursor']==snapshot['bank_state']['cursor']==0
        if phase=='head_one_micro':assert len(snapshot['inflight']['completed_micros'])==1 and snapshot['exposure']['head_rows']==5 and snapshot['exposure']['head_encoder_rows']==1
        if phase=='proposal':assert len(snapshot['inflight']['completed_micros'])==5 and not snapshot['optimizer']['state']
        stop_proofs[phase]=True
    for key in ('precision_policy','orders','symbol_weights','embedding_sha'):
        bad=copy.deepcopy(middle)
        if key=='orders':bad[key][0]=list(reversed(bad[key][0]))
        elif key=='symbol_weights':bad[key][0]+=1
        else:bad[key]='mismatch'
        try:restored.restore(bad);raise AssertionError('mismatch accepted')
        except ValueError:pass
    assert not torch.cuda.is_initialized();output=sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
    proof=dict(status='ROOT_V87_IMMUTABLE_BANK_CPU_ENGINE_PASS',tiny_language_encoder=True,actual_hidden512_final_projection_params=2117664,actual_full123M_model_calls=0,
        actual_GPU_calls=0,CUDA_initialized=False,real_CPU_Adam_enabled_scaler_used=True,immutable_bank_reference_fullstate_exact=True,separate_reference_feature_target_files=True,CPU_synthetic_updates_NOT_research_model_training=True,
        teacher_partial_boundary_and_reference_calibration_exact=True,seven_stored_deltas_two_optimizer_states=True,all_RNG_populated_Adam_and_enabled_scaler_next_group_exact=True,
        reject_restores_weights_Adam_but_consumes_scaler_cursor_exposure=True,AMPskip_consumes_group=True,CPU_reference_callback_equivalent=True,stops_exact_no_replay=stop_proofs,
        original_source_and_five_prefix_frozen=True,initial_parent_unchanged=True,fixture_seconds=time.monotonic()-began,fixture_output_bytes=output,
        actual_worker_preflight_or_GPU_ready=False,limitation='Tiny E/context and3CPU TRAIN sentences; actual learned full-size source restoration, fixed plan schema/caps/OSowner and preflight still required. Bank is separate feature/target immutable storage with exact digest reconstruction; actual learned full-source worker/preflight and whole output caps still required.')
    io.atomic_json(directory/'root_engine_cpu_review.json',proof);io.atomic_json(directory/'evidence_SHA.json',{str(f):support.sha(f) for f in (Path(__file__).resolve(),ROOT/'scripts/critical_bank_engine_v86.py',ROOT/'scripts/compact_critical_engine_v87.py',ROOT/'research/critical_bank_storage_v87.py',ROOT/'research/fp32_ordered_readout_v86.py',ROOT/'research/critical_reference_device_v86.py',parent,directory/'CPU_inmemory_fullstate_proof.json',directory/'root_engine_cpu_review.json')});print(json.dumps(proof))

if __name__=='__main__':main()
