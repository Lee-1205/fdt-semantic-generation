"""Grouped microbatch and exact interrupted-state contracts; tiny CPU only."""
import copy,json,random,re,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.frozen_verbalizer_objective_v72 import complete_expression_spans
from research.fuzzy_factual_objective_v70 import training_weights
from research import linked_lastlinear_runtime_v79 as linked
from research import linked_runtime_v75 as original_storage
from update_retention_engine_v80 import MarginEngine
import text_generation_support_v1 as support


def main():
    started=time.monotonic();torch.set_num_threads(2);torch.manual_seed(78);random.seed(78);np.random.seed(78)
    root=ROOT/'research/artifacts/update_retention_20261113_v80_cpu_v1';directory=root/'engine_fixture_v4';directory.mkdir(exist_ok=False)
    rows=[dict(id='u',prompt='Cruz stored 27 pins.'),dict(id='s',prompt='Cyrus logged +2.50 kg at 03:17.'),dict(id='o',prompt='A plain sentence without a numeral. '*2)]
    labels={r['id']:complete_expression_spans(r['prompt'],[dict(start=m.start(),end=m.end(),text=m.group(),kind='name') for m in re.finditer(r'Cruz|Cyrus',r['prompt'])]) for r in rows}
    orders=[[0,1,2,0,1]]*3;config=Config.smoke();_,source_report=training_weights(rows)
    source_report.update(class_weights=[1.,1.,1.,1.],normalization=1.);weights=[1.]*258;weights[257]=0.
    profile=dict(weights=weights,parent_objective_report=source_report,source_SHA='fixture_TRAIN_only')
    def model():
        enc=FuzzyCodecV64(config,input_capacity=256);enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks()
        return FuzzyRoundTripV64(enc,OrderedFuzzyReadout(config.z_dim,config.value_dim,hidden=128))
    initial=copy.deepcopy(model().state_dict());storage_initial=copy.deepcopy(initial);initial['readout.readout.net.0.bias'].add_(.125);parent=directory/'parent.pt';torch.save(dict(model=storage_initial),parent)
    reference=dict(path=str(parent.resolve()),SHA=linked.file_sha(parent),model_key='model',model_SHA=support.state_sha(storage_initial))
    class Budget:
        data_sha='fixture80';parent_reference={'source':'fixture80'};anchor_indices=[0,1,2];anchor_hidden_rows=sum((len(r['prompt'].encode())+15)//16+1 for r in rows);critical_targets=labels;frozen_symbol_profile=profile
        def __init__(self,label):
            self.output=directory/label;self.output.mkdir();self.calls=dict(encode=0,readout=0,reference_linear=0,stream_chunk=0,denoise=0);self.block_work=0;self.fail_at=None
        def check(self,write_bytes=0,emergency=False):
            if time.monotonic()-started>60:raise TimeoutError('fixed CPU60sec')
            if shutil.disk_usage(ROOT).free<100*1024**3+write_bytes:raise OSError('100GiB+pending write')
            if sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())+write_bytes>32*1024**2:raise OSError('fixed32MiB')
        def charge(self,kind):
            if self.fail_at is not None and sum(self.calls.values())>=self.fail_at:self.fail_at=None;raise InterruptedError('fixture cooperative STOP')
            self.calls[kind]+=1
    def create(label):
        with torch.random.fork_rng(devices=[]):m=model();m.load_state_dict(initial,strict=True)
        return MarginEngine(m,rows,orders,Budget(label),'fixture_plan79','cpu')
    def normalized(s):
        s=copy.deepcopy(s)
        for ref in s['cache_refs']:ref['targets']=Path(ref['targets']).name
        for ref in s['anchor_refs']:
            for key in ('features','targets'):ref[key]=Path(ref[key]).name
        if s['bank_ref'] is not None:
            for key in ('features','targets'):s['bank_ref'][key]=Path(s['bank_ref'][key]).name
        return s
    a=create('a');a.cache_next();partial=copy.deepcopy(a.export());a.cache_next();a.cache_next();a.initialize_reference_bank();complete=copy.deepcopy(a.export())
    b=create('b');b.restore(partial);b.cache_next();b.cache_next();b.initialize_reference_bank()
    assert support.state_sha(normalized(b.export()))==support.state_sha(normalized(complete))
    transition=create('transition');transition.restore(complete);assert support.state_sha(transition.export())==support.state_sha(complete)
    for i in range(3):
        packet=a.packet(i);assert packet['teacher_margin'].shape==packet['labels'].shape and packet['teacher_competitor'].dtype==torch.int16
    a.step();middle=copy.deepcopy(a.export());assert a.optimizer.param_groups[0]['lr']==1e-5
    
    try:original_storage.pack(middle,storage_initial,reference,a.names,support.state_sha);raise AssertionError('storing only2 would lose frozen V75 prefix')
    except ValueError:pass
    envelope=linked.pack(middle,storage_initial,reference,support.state_sha);assert len(envelope['active_names'])==7 and len(middle['optimizer_names'][0])==2;
    a.budget.check(write_bytes=16*1024**2)
    meta=linked.save_verified(directory/'populated_group_runtime.pt',envelope,ROOT,support.state_sha);stored=linked.load_verified(meta['path'],meta['SHA'],ROOT,support.state_sha)
    a.step(True);a.step();expected=a.export()
    restored=create('restored');restored.restore(stored);restored.step(True);restored.step()
    assert support.state_sha(restored.export())==support.state_sha(expected)
    assert a.attempts==a.cursor==3 and a.updates+a.rejections==2 and a.skips==1 and a.scaler.is_enabled() and len(a.optimizer.state)==2
    assert all(int(v['step'])==a.updates for v in a.optimizer.state.values()) and a.budget.calls['encode']==a.budget.calls['readout']==18
    assert a.exposure['teacher_rows']==3 and a.exposure['head_rows']==a.exposure['head_encoder_rows']==15
    assert all(h['clip_and_Adam_calls']==1 and len(h['micros'])==5 and abs(h['loss']-sum(m['loss']*.2 for m in h['micros']))<1e-8 for h in a.history)
    assert a.exposure['head_UTF8_bytes']==a.exposure['head_encoder_UTF8_bytes']==sum(len(rows[i]['prompt'].encode()) for group in orders for i in group)
    a.verify_frozen();assert list(directory.rglob('bank.features.pt'));a.finalize_reference_bank();assert a.reference_final_ref['initial_protected_flips']==0
    rejected=create('rejected');rejected.restore(middle);before_reject=copy.deepcopy(rejected.export());real_reference=rejected.reference_decision
    from research.actual_update_retention_v80 import TrainDecision
    def force_reject():
        decision=real_reference();return TrainDecision(False,dict(decision.measurements,CPU_fixture_forced_rejection=1.))
    rejected.reference_decision=force_reject;rejected.step();after_reject=copy.deepcopy(rejected.export())
    assert rejected.rejections==1 and rejected.updates==1 and rejected.attempts==rejected.cursor==2 and len(rejected.history[-1]['update_receipt']['reference_checks'])==4
    assert support.state_sha(before_reject['model'])==support.state_sha(after_reject['model']) and support.state_sha(before_reject['optimizer'])==support.state_sha(after_reject['optimizer'])
    assert after_reject['scaler']['_growth_tracker']==before_reject['scaler']['_growth_tracker']+1 and after_reject['exposure']['head_encoder_rows']==10
    reject_restore=create('reject_restore');reject_restore.restore(after_reject);assert support.state_sha(reject_restore.export())==support.state_sha(after_reject)
    for phase in ('teacher','head','proposal'):
        stopped=create('stop_'+phase)
        if phase!='teacher':stopped.restore(complete)
        stopped.budget.fail_at=sum(stopped.budget.calls.values())+(2 if phase=='head' else 10 if phase=='proposal' else 0)
        try:stopped.cache_next() if phase=='teacher' else stopped.step();raise AssertionError('expected STOP')
        except InterruptedError:pass
        snapshot=copy.deepcopy(stopped.export());again=create('restored_stop_'+phase);again.restore(snapshot)
        assert support.state_sha(again.export())==support.state_sha(snapshot)
        if phase=='head':assert len(snapshot['inflight']['completed_micros'])==1 and snapshot['exposure']['head_rows']==5 and snapshot['exposure']['head_encoder_rows']==1
        if phase=='proposal':assert len(snapshot['inflight']['completed_micros'])==5 and snapshot['exposure']['head_encoder_rows']==5 and not snapshot['optimizer']['state']
        try:again.cache_next() if phase=='teacher' else again.step();raise AssertionError('automatic replay forbidden')
        except RuntimeError:pass
    for key in ('embedding_sha','symbol_weights','orders'):
        bad=copy.deepcopy(middle)
        if key=='embedding_sha':bad[key]='wrong'
        elif key=='symbol_weights':bad[key][0]+=1
        else:bad[key][0]=list(reversed(bad[key][0]))
        try:restored.restore(bad);raise AssertionError('identity mismatch accepted')
        except ValueError:pass
    assert not torch.cuda.is_initialized()
    proof=dict(status='V80_FULL_ENGINE_REFERENCE_DELTA_CPU_PASS',teacher_partial_boundary_exact=True,
        loss_only_teacher_margin_and_competitor_separated=True,five_live_micros_same_parameters_one_unscale_clip_Adam=True,
        populated2Adam_enabledCPUscaler_ALLRNG_parentlinked_exact_next_group=True,AMPskip_consumes_whole_group15rows_three_attempts=True,
        head_STOP_after_one_micro_consumes5_requested1_actual_exact_restore_NOreplay=True,teacher_STOP_exact_restore_NOreplay=True,
        frozen_weight_source_order_epoch_mismatch_rejected=True,upstream_and_five_prefix_tensors_frozen_only2lastlinear=True,seven_storage_deltas_two_optimizer_states_exact=True,original_two_delta_writer_correctly_rejects_lost_frozen_prefix=True,
        fixture_calls=dict(encode=18,readout=18),initial_reference_feature_target_separation_and_batched_masks_verified=True,actual_Adam_accept_reject_AMPskip_ledger_exact=True,populated_Adam_all4reject_weight_Adam_exact_scaler_and_exposure_consumed_fullstate_restore=True,proposal_STOP_after_all5micros_exact_restore_NOreplay=True,final_reference_raw_argmax_saved=True,actual_fullsize_model_calls=0,actual_GPU_or_worker_ready=False,CUDA_initialized=False,
        seconds=time.monotonic()-started,output_bytes=sum(f.stat().st_size for f in directory.rglob('*') if f.is_file()),
        files_SHA={str(f):support.sha(f) for f in (ROOT/'research/update_retention_objective_v80.py',ROOT/'research/linked_lastlinear_runtime_v79.py',ROOT/'scripts/update_retention_engine_v80.py',Path(__file__))})
    assert not (root/'root_engine_CPU_contract_v80_v4.json').exists();(root/'root_engine_CPU_contract_v80_v4.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))


if __name__=='__main__':main()
