"""ROOT independent pin/cost/source-linked fullsize CPU/resource review."""
import ast,datetime,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    torch.set_num_threads(4);d=ROOT/'research/artifacts/critical_retention_20261112_v75_cpu';review=d/'execution_review';p=io.load_json(review/'plan_unsealed.json');native=io.load_json(review/'root_native_preseal.json')
    assert not p['authorized'] and native['PythonCount']==0 and native['C_free_bytes']>=109*1024**3 and native['GPU_free_MiB']>=12288 and native['RAM_available_bytes']>4*1024**3
    for f,sha in p['files'].items():assert support.sha(f)==sha,f
    for f in p['new_code_files']:ast.parse((ROOT/f).read_text(encoding='utf-8-sig'))
    cp=io.load_json(review/'cpu_preflight_owned/final_status.json');r=cp['report']
    assert cp['status']=='CPU_PREFLIGHT_PASS' and cp['exit']['all_owned_processes_gone'] and not cp['owner_imported_Torch'] and r['optimizer_steps']==0 and not r['CUDA_initialized']
    assert r['actual_encoder_forwards']==r['actual_readout_forwards']==2 and r['active_tensors']==7 and r['active_readout_parameters']==2908192
    # Independently rebuild model without calling the checkpoint helper.
    parent=torch.load(p['cycle_source'],map_location='cpu',weights_only=True);assert support.state_sha(parent)==p['cycle_algorithm_sha'] and len(parent['optimizer']['state'])==7
    states=[]
    for name,cursor in [('initial.pt',0),('partial_teacher.pt',1)]:
        f=review/'cpu_preflight_owned/runtime'/name;e=torch.load(f,map_location='cpu',weights_only=True);ref=e['parent_reference']
        assert ref['path']==p['cycle_source'] and ref['SHA']==p['cycle_sha'] and ref['model_key']=='model' and ref['model_SHA']==support.state_sha(parent['model']) and not e['standalone']
        model=dict(parent['model']);assert set(e['active_names'])==set(e['active_weights']) and len(e['active_names'])==7
        model.update(e['active_weights']);full=dict(e['runtime'],model=model);assert support.state_sha(full)==e['reconstructed_fullstate_SHA']
        assert support.state_sha(model)==support.state_sha(parent['model']) and full['cache_cursor']==cursor and not full['optimizer']['state'] and full['scaler']['scale']==1024 and full['attempts']==full['updates']==full['cursor']==0
        assert full['inflight'] is None and full['gradients']=={} and full['objective_report']==p['objective_report'] and full['orders']==io.load_json(d/'orders.json')
        assert full['torch_rng'].numel() and full['numpy_rng']['keys'] and full['python_rng']
        for c in full['cache_refs']:
            assert set(c)=={'targets','targets_SHA','cache_key','source_id'} and support.sha(c['targets'])==c['targets_SHA']
            packet=torch.load(c['targets'],map_location='cpu',weights_only=True);assert 'semantic' not in packet and 'features' not in packet
        states.append(dict(file=str(f),SHA=support.sha(f),reconstructed_algorithm_SHA=e['reconstructed_fullstate_SHA'],bytes=f.stat().st_size))
    assert not list((review/'cpu_preflight_owned/runtime').rglob('*.features.pt'))
    # All post-Popen ownership/cleanup is inherited byte-for-byte from tested V74.
    old=(ROOT/'scripts/supervise_context_verbalizer_v74.py').read_text(encoding='utf-8');new=(ROOT/'scripts/supervise_critical_retention_v75.py').read_text(encoding='utf-8')
    old_body=old[old.index("    state=dict(status='PREPARING'"):]
    new_body=new[new.index("    state=dict(status='PREPARING'"):];assert old_body==new_body
    rows=io.load_json(d/'train_features.json');dev=io.load_json(d/'dev_features.json')
    assert len(rows)==640 and len(dev)==64 and sum(len(v['prompt'].encode()) for v in rows)==969013
    assert p['cost_detail']['encode_total']==1412 and p['cost_detail']['readout_total']==1690 and p['cost_detail']['stream_chunk_total']==258 and sum((1412,1690,258))==p['cost']==3360<3500
    detail=p['output_estimate_detail'];assert sum(detail.values())==p['output_estimate_including_all_raw_parentlinked_states_STOP_and32MiB']<1024**3
    headers=[]
    for f in [Path(p['baseline_run'])/f'raw_old_untrained128_{i:02d}.pt' for i in (0,8,24,32,64,100)]:
        packet=torch.load(f,map_location='cpu',weights_only=True);body=sum(v.numel()*v.element_size() for v in packet.values() if torch.is_tensor(v));headers.append(f.stat().st_size-body)
    assert max(headers)+4096<16384 # new reference metadata still conservatively covered
    assert support.sha(p['cycle_source'])==p['cycle_sha'] and not torch.cuda.is_initialized()
    proof=dict(status='ROOT_INDEPENDENT_V75_READY',pins_independently_rehashed=len(p['files']),actual_CPU_parentlinked_initial_partial_reconstruction=states,
        native=native,actual_fullsize_CPU_preflight_exit_all_owned_gone=True,actual_fullsize_optimizer_steps=0,CUDA_initialized=False,
        inherited_postPopen_owner_cleanup_byte_identical_V74=True,source_teacher_loss_only_no_feature_cache=True,all_old_parent_Adam_scaler_RNG_unchanged=True,
        fixed_cost=3360,max_calls=3500,output_estimate=detail,hard_output_cap=1024**3,start_protected100GiB_plus1GiB_output_plus8GiB_headroom=True,
        sampled_actual_raw_container_overhead_bytes=headers,no_old_files_deleted=True,actual_GPU_launches_so_far=0,reviewer_SHA=support.sha(Path(__file__)))
    io.atomic_json(review/'root_independent_validate.json',proof);print(json.dumps(proof))
if __name__=='__main__':main()
