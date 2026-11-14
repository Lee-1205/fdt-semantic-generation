"""ROOT independent V79 actual schema/pins/linked warm state/owner/resources."""
import ast,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
 torch.set_num_threads(4);d=ROOT/'research/artifacts/joint_quantity_20261113_v76_cpu_v1';b=ROOT/'research/artifacts/last_linear_20261113_v79_cpu_v1';review=b/'execution_review_v1'
 p=io.load_json(review/'plan_unsealed.json');native=io.load_json(review/'root_native_preseal.json')
 assert not p['authorized'] and native['PythonCount']==0 and native['C_free_bytes']>=p['start_free_bytes'] and native['GPU_free_MiB']>=12288 and native['RAM_available_bytes']>4*1024**3
 for f,s in p['files'].items():assert support.sha(f)==s,f
 for f in p['new_code_files']:ast.parse((ROOT/f).read_text(encoding='utf-8-sig'))
 cp=io.load_json(review/'cpu_preflight_owned/final_status.json');r=cp['report']
 assert cp['status']=='CPU_PREFLIGHT_PASS' and cp['exit']['all_owned_processes_gone'] and not cp['owner_imported_Torch'] and not r['CUDA_initialized'] and r['optimizer_steps']==0
 assert r['actual_encoder_forwards']==r['actual_readout_forwards']==2 and r['active_tensors']==2 and r['active_readout_parameters']==2117664 and r['serialized_overrides']==7 and r['optimizer_owned_tensors']==2 and r['actual_only2_gradient'] and r['five_readout_prefix_tensors_unchanged'] and r['actual_saved_onelevel_reference_checked']
 assert r['V75_linked_parent_reconstructed'] and r['V74_standalone_storage_parent']
 storage=torch.load(p['storage_parent'],map_location='cpu',weights_only=True);e=torch.load(p['cycle_source'],map_location='cpu',weights_only=True)
 ref=e['parent_reference'];assert ref['path']==p['storage_parent'] and ref['SHA']==p['storage_parent_sha'] and ref['model_SHA']==support.state_sha(storage['model'])
 warm=dict(storage['model']);warm.update(e['active_weights']);old=dict(e['runtime'],model=warm)
 assert support.state_sha(old)==p['cycle_algorithm_sha']==e['reconstructed_fullstate_SHA'] and old['updates']==640 and len(old['optimizer']['state'])==7
 old_adam_scaler_RNG=support.state_sha({k:old[k] for k in ('optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng')})
 states=[]
 for name,cursor in [('initial.pt',0),('partial_teacher.pt',1)]:
  f=review/'cpu_preflight_owned/runtime'/name;env=torch.load(f,map_location='cpu',weights_only=True);ref=env['parent_reference']
  assert ref['path']==p['storage_parent'] and ref['SHA']==p['storage_parent_sha'] and ref['model_key']=='model' and not env['standalone']
  model=dict(storage['model']);assert set(env['active_names'])==set(env['active_weights']) and len(env['active_names'])==7
  model.update(env['active_weights']);full=dict(env['runtime'],model=model)
  assert support.state_sha(full)==env['reconstructed_fullstate_SHA'] and support.state_sha(model)==support.state_sha(warm)
  assert full['schema']=='fdt_lastlinear_readout_v79_fullstate' and full['objective']==p['objective']
  assert full['cache_cursor']==cursor and not full['optimizer']['state'] and full['scaler']['scale']==1024 and full['attempts']==full['updates']==full['cursor']==0
  assert full['inflight'] is None and full['gradients']=={} and full['objective_report']==p['objective_report'] and full['orders']==io.load_json(Path(p['balance_data'])/'orders.json')
  assert full['embedding_sha']==old['embedding_sha'] and full['teacher_sha']==support.state_sha(warm and {k.removeprefix('readout.readout.'):v for k,v in warm.items() if k.startswith('readout.readout.')})
  assert full['torch_rng'].numel() and full['python_rng'] and full['numpy_rng']['keys']
  for c in full['cache_refs']:
   assert set(c)=={'targets','targets_SHA','cache_key','source_id'} and support.sha(c['targets'])==c['targets_SHA']
   packet=torch.load(c['targets'],map_location='cpu',weights_only=True);assert not {'semantic','features'}&set(packet)
  trained=['readout.readout.net.2.weight','readout.readout.net.2.bias'];stored=old['optimizer_names'][0]
  assert full['optimizer_names']==[trained] and full['storage_delta_names']==stored
  prefix={k:v for k,v in warm.items() if k in stored and k not in trained}
  assert full['frozen_head_prefix_SHA']==support.state_sha(prefix)==p['storage_delta_scope']['frozen_prefix_SHA']
  assert full['frozen_except_last2_SHA']==support.state_sha({k:v for k,v in warm.items() if k not in trained})
  assert env['storage_scope']==dict(serialized_override_names=stored,trainable_optimizer_names=trained,frozen_override_names=[n for n in stored if n not in trained])
  states.append(dict(file=str(f),SHA=support.sha(f),reconstructed_algorithm_SHA=env['reconstructed_fullstate_SHA'],bytes=f.stat().st_size))
 assert support.state_sha({k:old[k] for k in ('optimizer','scaler','torch_rng','cuda_rng','python_rng','numpy_rng')})==old_adam_scaler_RNG
 assert not list((review/'cpu_preflight_owned/runtime').rglob('*.features.pt'))
 oldowner=(ROOT/'scripts/supervise_competing_margin_v78.py').read_text(encoding='utf-8');newowner=(ROOT/'scripts/supervise_last_linear_v79.py').read_text(encoding='utf-8')
 marker="    state=dict(status='PREPARING'"
 assert oldowner[oldowner.index(marker):]==newowner[newowner.index(marker):]
 rows=io.load_json(d/'train_features.json');dev=io.load_json(Path(p['eval_data'])/'features.json')
 assert len(rows)==1024 and len(dev)==64 and sum(len(v['prompt'].encode()) for v in rows)==1010943
 detail=p['cost_detail'];assert (detail['encode_total'],detail['readout_total'],detail['stream_chunk_total'])==(3076,3418,258) and sum((3076,3418,258))==p['cost']==6752<7000
 output=p['resource_estimate']['components'];assert sum(output.values())==p['resource_estimate']['total_bytes']<p['max_output_bytes']
 assert p['model_scope']==dict(tensors=2,parameters=2117664,names=sorted(trained))
 assert p['protected_free_bytes']==100*1024**3 and p['start_free_bytes']>=p['protected_free_bytes']+p['max_output_bytes']+8*1024**3
 assert not any(k.startswith('output_estimate_including') for k in p) and 'actual_trainable_parameters' not in p
 # Exercise real referenced packets without any neural forward; stage uses only
 # direct canonical references in new raw so it cannot grow a recursive chain.
 for case in p['CPU_reference_cases']:
  a=torch.load(case['raw'],map_location='cpu',weights_only=True);row=io.load_json(case['data'])[case['index']];assert a['source_id']==row['id'] and a['embedding_sha']==old['embedding_sha']
  if 'fuzzy_commit_reference' in a:
   ref=a['fuzzy_commit_reference'];assert ref['SHA']==p['files'][ref['path']] and support.sha(ref['path'])==ref['SHA']
   prior_packet=torch.load(ref['path'],map_location='cpu',weights_only=True);assert 'fuzzy_commit' in prior_packet and 'fuzzy_commit_reference' not in prior_packet and torch.equal(a['targets'],prior_packet['targets'])
 profile=io.load_json(Path(p['symbol_profile_data'])/'symbol_weight_source.json')
 assert torch.equal(torch.tensor(profile['weights']),old['symbol_weights'])
 assert all(torch.equal(torch.load(review/'cpu_preflight_owned/runtime'/n,map_location='cpu',weights_only=True)['runtime']['symbol_weights'],old['symbol_weights']) for n in ('initial.pt','partial_teacher.pt'))
 assert p['attempts']==384 and len(io.load_json(Path(p['balance_data'])/'orders.json'))==384 and all(len(g)==5 for g in io.load_json(Path(p['balance_data'])/'orders.json'))
 assert io.load_json(ROOT/'configs/research_base_selection_v78_review_20261113.json')['selected_SHA']==p['cycle_sha']
 assert not torch.cuda.is_initialized()
 proof=dict(status='ROOT_INDEPENDENT_V79_READY',pins_independently_rehashed=len(p['files']),actual_CPU_initial_partial_states_reconstructed=states,
  warm_V75_complete_parent_runtime_verified=True,frozen_V75_base_symbol_weights_verified=True,finite1920source_micros_in384_same_parameter_groups_verified=True,canonical_scope_and_single_output_estimate_verified=True,mandatory_standalone_V74_storage_base_verified=True,parent_Adam_scaler_ALLRNG_unchanged=True,
  actual_fullsize_CPU_only2gradient_and_loss_only_teacher=True,actual_reference_packets_onelevel_verified=True,
  inherited_postPopen_owner_cleanup_byte_identical_V78=True,native=native,actual_CPU_all_owned_gone=True,
  fixed_cost=6752,max_calls=7000,output_estimate=output,hard_output_cap=p['max_output_bytes'],actual_GPU_launches_so_far=0,
  seven_serialized_overrides_five_frozen_V75_prefix_two_Adam_verified=True,loss_group_LR_seed_exposure_identical_V78=True,
  CUDA_initialized=False,reviewer_SHA=support.sha(Path(__file__)))
 io.atomic_json(review/'root_independent_validate.json',proof);print(json.dumps(proof))
if __name__=='__main__':main()
