"""ROOT independent source/fullstate/schema/resources proof; no model forward."""
import json,shutil,sys,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support
from research.lossless_raw_tensor_v80 import unpack
from saved_commit_reference_v76_v2 import resolve_saved_commit
from run_critical_repair_20261113_v88 import validate,CAPS,OUTPUT,START,PARENT_SHA,ALGORITHM

def main():
    a=ROOT/'research/artifacts/critical_repair_20261113_v88_cpu_v1';path=a/'plan_unsealed.json';p,rows,orders=validate(path,Path(io.load_json(path)['output']));pre=io.load_json(a/'cpu_preflight_owned/final_status.json')
    assert not p['authorized'] and pre['status']=='CPU_PREFLIGHT_PASS' and pre['exit']['all_owned_processes_gone'] and pre['worker_exit_code']==0 and not pre['owner_imported_Torch']
    assert pre['report']['actual_encoder_forwards']==pre['report']['actual_readout_forwards']==4 and pre['report']['optimizer_steps']==0 and not pre['report']['CUDA_initialized']
    storage=torch.load(p['storage_parent'],map_location='cpu',weights_only=True);sourceenv=torch.load(p['source'],map_location='cpu',weights_only=True);warm=dict(storage['model']);warm.update(sourceenv['active_weights']);source=dict(sourceenv['runtime'],model=warm)
    assert support.sha(p['source'])==PARENT_SHA and support.state_sha(source)==ALGORITHM and source['updates']==247 and len(source['optimizer']['state'])==2
    proof=[]
    for folder,cursor in [('initial',0),('teacher_complete',3)]:
        f=a/'cpu_preflight_owned/runtime'/folder/'runtime.pt';meta=io.load_json(f.with_suffix('.json'));assert support.sha(f)==meta['SHA']
        env=torch.load(f,map_location='cpu',weights_only=True);model=dict(storage['model']);model.update(env['active_weights']);s=dict(env['runtime'],model=model)
        assert support.state_sha(s)==env['reconstructed_fullstate_SHA']==meta['reconstructed_fullstate_SHA']
        assert set(s['model'])==set(warm) and all(torch.equal(v,warm[k]) for k,v in model.items())
        assert s['schema']=='fdt_fullFP32_critical_engine_immutable_bank_v87' and s['precision_policy']==p['precision_policy'] and len(env['active_names'])==7 and len(s['optimizer_names'][0])==2 and not s['optimizer']['state']
        assert s['scaler']['scale']==1024 and s['scaler']['_growth_tracker']==0 and s['cache_cursor']==len(s['cache_refs'])==s['bank_state']['cursor']==cursor
        assert s['attempts']==s['updates']==s['skips']==s['rejections']==0 and s['inflight'] is None and s['bank_ref'] is None and s['bank_ready'] is False
        assert all(part['hidden'].dtype==torch.float32 and part['hidden'].shape[-1]==512 for part in s['bank_state']['parts']) and s['bank_state']['config']['precision_policy']==p['precision_policy']
        assert s['embedding_sha']==source['embedding_sha'] and s['storage_delta_names']==sourceenv['active_names'] and s['Exact_entries']==s['copy_bytes']==s['denoise_calls']==0
        proof.append(dict(phase=folder,bytes=f.stat().st_size,all_model_tensors_equal_source=True,bank_cursor=cursor,algorithm_SHA=support.state_sha(s)))
        del env,model,s;gc.collect()
    items=io.load_json(p['eval_inputs']);assert len(items)==194 and len({x['row']['id'] for x in items})==194
    total=0
    for item in items:
        packet=unpack(torch.load(item['raw'],map_location='cpu',weights_only=True));tensor,_=resolve_saved_commit(packet,item['row'],source['embedding_sha'],p['files'],ROOT,512)
        assert packet['Exact_entries']==0 and all(0<=i<packet['targets'].shape[0] for i in item['sample_patches']);total+=packet['targets'].numel();del packet,tensor
    assert total==589936
    assert p['cost']==dict(teacher_encode=1024,teacher_readout=1024,head_encode_max=960,head_readout_max=960,reference_linear_max=770,baseline_cached_readout=130,final_cached_readout=194)
    assert sum(p['cost'].values())==5062<5100 and sum(CAPS.values())==192*1024**2==OUTPUT and max(CAPS[k] for k in ('model','emergency','temporary'))==32*1024**2
    assert p['no_accepted_window']==32 and p['attempts']==192 and p['teacher_epochs']==1 and len(orders)==192 and p['reference_patch_work_cap']==8192*770
    old=(ROOT/'scripts/supervise_update_retention_v80.py').read_text(encoding='utf-8');new=(ROOT/'scripts/supervise_critical_repair_v88.py').read_text(encoding='utf-8');assert old.split("    state=dict(status='PREPARING'",1)[1]==new.split("    state=dict(status='PREPARING'",1)[1]
    compression=io.load_json(ROOT/'research/artifacts/closed_raw_compression_20261113_v1/root_result.json');assert compression['deletions']==0 and compression['physical_saved_bytes']==100757570 and all(x['logical_SHA_unchanged'] for x in compression['files'])
    free=shutil.disk_usage(ROOT).free;assert free>=START and not torch.cuda.is_initialized()
    report=dict(status='ROOT_V88_ACTUAL_REPAIR_READY',pins=len(p['files']),actual_fullsize_CPU=pre['report'],source_model_SHA=support.state_sha(warm),CPU_fullstates=proof,
        max_calls=5100,expected_max=5062,fixed192attempts=True,teacher1024once=True,head960not_unique_epoch=True,source_head_unchanged_at_CPU_preflight=True,encoder_updates0=True,
        Cfree=free,start_required=START,owned192MiB_includes_STOPtemp=True,caps=CAPS,unchanged_reviewed_postPopen_cleanup=True,eval_same_observed_not_blind=True,
        long_same_cached_Commit_not_fresh_encoder=True,trace_NOT_full258_raw=True,wholelanguage_complete=False,actual_new_GPU_calls=0,CUDA_initialized=False,
        limitations='Changes FP32 arithmetic and weak-critical TRAIN guard together; actual accepts may be fewer than192. Critical reference protection finite TRAIN only. Stop after32noaccepted groups; no cap widening or automatic retry. SyntheticCPU engine recovery is not CUDA restart evidence.')
    io.atomic_json(a/'root_independent_validate.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ('actual_fullsize_CPU','CPU_fullstates','caps')}))

if __name__=='__main__':main()
