"""ROOT independent actual worker/source/CPU runtime/cost/owner validation."""
import gc,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'src')]
import torch
from run_feasible_repair_20261113_v91 import validate, SOURCE_SHA, ALGORITHM, ANCHOR_SHA, CAPS, COST, OUTPUT, START
import learning_io_v2 as io
import text_generation_support_v1 as support


def main():
    a=ROOT/'research/artifacts/feasible_repair_20261113_v91_cpu_v1'; path=a/'plan_unsealed.json'
    p,rows,orders=validate(path,Path(io.load_json(path)['output']))
    pre=io.load_json(a/'cpu_preflight_owned/final_status.json')
    assert not p['authorized'] and pre['status']=='CPU_PREFLIGHT_PASS' and pre['exit']['all_owned_processes_gone']
    assert pre['worker_exit_code']==0 and not pre['owner_imported_Torch']
    report=pre['report']
    assert report['actual_encoder_forwards']==report['actual_readout_forwards']==report['actual_reference_projections']==1
    assert not report['CUDA_initialized'] and report['optimizer_steps']==0 and report['actual_only2_gradient']
    assert report['original_floor_not_recalibrated'] and report['all_original_TRAIN_reference_pass']
    assert report['warm_teacher_identities_distinct'] and report['original_anchor_SHA']==ANCHOR_SHA
    storage=torch.load(p['storage_parent'],map_location='cpu',weights_only=True)
    source=torch.load(p['source'],map_location='cpu',weights_only=True)
    warm=dict(storage['model']); warm.update(source['active_weights'])
    whole_source=dict(source['runtime'],model=warm)
    assert support.state_sha(whole_source)==ALGORITHM and support.sha(p['source'])==SOURCE_SHA
    assert whole_source['updates']==16 and whole_source['attempts']==108
    states=[]
    for folder,verified in (('initial',False),('verified',True)):
        f=a/'cpu_preflight_owned/runtime'/folder/'runtime.pt'; meta=io.load_json(f.with_suffix('.json'))
        assert support.sha(f)==meta['SHA']
        envelope=torch.load(f,map_location='cpu',weights_only=True)
        model=dict(storage['model']); model.update(envelope['active_weights']); s=dict(envelope['runtime'],model=model)
        assert support.state_sha(s)==meta['reconstructed_fullstate_SHA']==envelope['reconstructed_fullstate_SHA']
        assert set(model)==set(warm) and all(torch.equal(v,warm[k]) for k,v in model.items())
        assert len(envelope['active_names'])==7 and len(s['optimizer_names'][0])==2 and not s['optimizer']['state']
        assert s['attempts']==s['updates']==s['skips']==s['rejections']==0 and s['inflight'] is None
        assert s['scaler']['scale']==1024 and s['scaler']['_growth_tracker']==0
        assert s['bank_ref']==whole_source['bank_ref'] and s['bank_state'] is None and s['bank_ready']
        assert s['warm_reference_verified']==verified and s['cache_refs']==whole_source['cache_refs']
        assert s['teacher_sha']==whole_source['teacher_sha'] and s['cache_cursor']==1024
        assert s['warm_readout_SHA']==support.state_sha({k.removeprefix('readout.readout.'):v for k,v in warm.items() if k.startswith('readout.readout.')})
        assert s['teacher_sha']!=s['warm_readout_SHA'] and s['representation']==io.load_json(p['anchor'])['representation']
        assert not any(s['exposure'].values()) and s['Exact_entries']==s['copy_bytes']==s['denoise_calls']==0
        assert s['calls']==(dict(encode=1,readout=1,reference_linear=1,denoise=0) if verified else dict(encode=0,readout=0,reference_linear=0,denoise=0))
        states.append(dict(phase=folder,all_285_tensor_values_equal_warm=True,bytes=f.stat().st_size,
            algorithm_SHA=support.state_sha(s),warm_reference_verified=verified,optimizer_updates0=True))
        del envelope,model,s; gc.collect()
    # Actual saved-baseline linkage is to V88 FINAL weights; no new baseline NN
    # is spent. Existing independently reviewed raw trace claims stay limited.
    result=io.load_json(p['source_result']); oldroot=io.load_json(ROOT/'runs/critical_repair_20261113_v88/root_result.json')
    assert result['model']['SHA']==SOURCE_SHA and oldroot['all_owned_gone_before_analysis']
    baseline=io.load_json(p['baseline_records']); assert len(baseline)==194
    for r in baseline: assert support.sha(r['trace'])==r['SHA']
    assert len(rows)==1024 and len(orders)==96 and sum(map(len,orders))==480
    assert COST==dict(head_encode=480,head_readout=480,reference_linear=194,final_cached_readout=194)
    assert sum(COST.values())==1348<1400 and p['reference_patch_work_cap']==8192*(2*96+2)
    assert sum(CAPS.values())==OUTPUT==144*1024**2 and p['no_accepted_window']==32
    old=(ROOT/'scripts/supervise_critical_repair_v88.py').read_text(encoding='utf-8')
    new=(ROOT/'scripts/supervise_feasible_repair_v91.py').read_text(encoding='utf-8')
    assert old.split("    state=dict(status='PREPARING'",1)[1]==new.split("    state=dict(status='PREPARING'",1)[1]
    assert "p['wall_seconds']!=600" in new and "p['max_calls']!=1400" in new and '108*1024**3+144*1024**2' in new
    compression=io.load_json(ROOT/'research/artifacts/closed_raw_compression_20261113_v2/root_result.json')
    assert compression['deletions']==0 and compression['physical_saved_bytes']==201805956
    assert all(r['logical_SHA_unchanged'] and r['CPU_safeload'] for r in compression['files'])
    free=shutil.disk_usage(ROOT).free; assert free>=START and not torch.cuda.is_initialized()
    reviewed=dict(status='ROOT_V91_ACTUAL_FEASIBLE_REPAIR_READY',pins=len(p['files']),actual_fullsize_CPU=report,
        CPU_fullstates=states,fresh2Adam_scope=2117664,all_warm_tensor_values_exact=True,
        original_teacher_bank_anchor_preserved=True,reused_teacher1024_not_new_NN_exposure=True,
        same194_saved_baseline_source_verified=True,new_baseline_model_calls=0,
        max_calls=1400,expected_max_calls=1348,cost=COST,owned_output_bytes_including_STOPtemporary=OUTPUT,
        component_caps=CAPS,Cfree=free,start_required=START,unchanged_reviewed_postPopen_cleanup=True,
        actual_new_GPU_calls=0,CUDA_initialized=False,
        limitations='Actual CPU original-bank parity and two gradient ownership tested. Synthetic engine transaction recovery is not actual CUDA restart. Same observed dev/long cached Commit panel; finite TRAIN guard only; no semantic completion, promotion or repaired92actualproposals claim.')
    io.atomic_json(a/'root_independent_validate.json',reviewed)
    print(json.dumps({k:v for k,v in reviewed.items() if k not in ('actual_fullsize_CPU','CPU_fullstates','component_caps')}))


if __name__=='__main__': main()
