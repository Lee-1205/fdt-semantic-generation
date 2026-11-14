"""Actual V76 finite plan/schema/storage estimate. Seal only after ROOT validation."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_update_balance_20261113_v77 import validate,PARENT_SHA,ALGORITHM_SHA,STORAGE_PARENT_SHA,OUTPUT_CAP,START_FREE
from research.update_balance_objective_v77 import OBJECTIVE,COEFFICIENTS,MAX_POSITION_AMPLIFICATION,frozen_objective_report
from research.fuzzy_factual_objective_v70 import training_weights

def main(seal):
 d=ROOT/'research/artifacts/joint_quantity_20261113_v76_cpu_v1';b=ROOT/'research/artifacts/update_balance_20261113_v77_cpu_v1';ed=ROOT/'research/artifacts/joint_quantity_eval_20261113_v77_cpu_v1';review=b/'execution_review_v3';unsealed=review/'plan_unsealed.json'
 if seal:
  p=io.load_json(unsealed);out=Path(p['output']);validate(unsealed,out);assert not out.exists()
  cp=review/'cpu_preflight_owned/final_status.json';c=io.load_json(cp);r=c['report'];ind=review/'root_independent_validate.json'
  assert c['status']=='CPU_PREFLIGHT_PASS' and c['exit']['all_owned_processes_gone'] and not c['owner_imported_Torch']
  assert all(r[k] for k in ('fullsize_engine_restore','actual_linked_initial_and_partial_teacher_exact','actual_only7_gradient','live_source_reencoded','no_feature_cache','long_wrapper_all_tensors_exact','V75_linked_parent_reconstructed','V74_standalone_storage_parent','actual_saved_onelevel_reference_checked','all_parent_tensors_exact','fresh7Adam','encoder_binding_frozen','Exact_entries0'))
  assert not r['CUDA_initialized'] and r['optimizer_steps']==0 and r['objective_report']==p['objective_report']
  assert io.load_json(ind)['status']=='ROOT_INDEPENDENT_V77_READY'
  for f in (cp,ind):p['files'][str(f)]=support.sha(f)
  now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900);out.mkdir();path=out/'plan.json';io.atomic_json(path,p);validate(path,out)
  snap=out/'code_snapshot';snap.mkdir()
  for n in p['new_code_files']:(snap/Path(n).name).write_bytes((ROOT/n).read_bytes())
  print(json.dumps(dict(plan_SHA=support.sha(path),cost=p['cost'],max_calls=p['max_calls'],output_cap=OUTPUT_CAP,start_free_bytes=START_FREE)));return
 assert not unsealed.exists();base=ROOT/'runs/critical_retention_20261112_v75';old=io.load_json(ROOT/'runs/joint_quantity_20261113_v76/plan.json');p=dict(old);source=base/'model/runtime.pt';storage=Path(old['storage_parent'])
 names=['research/update_balance_objective_v77.py','scripts/prepare_update_balance_v77_cpu.py','scripts/review_update_balance_v77_cpu.py',
  'scripts/update_balance_engine_v77.py','scripts/test_update_balance_engine_v77_cpu.py','scripts/update_balance_assess_v77.py',
  'scripts/run_update_balance_20261113_v77.py','scripts/supervise_update_balance_v77.py','scripts/prepare_update_balance_plan_v77.py','scripts/review_update_balance_v77_preseal_cpu.py']

 rows=io.load_json(d/'train_features.json');dev=io.load_json(ed/'features.json');_,obj=training_weights(rows)
 obj=frozen_objective_report(obj,io.load_json(b/'symbol_weight_source.json'))
 p['files']={k:v for k,v in old['files'].items() if k.endswith('.py')}
 p.update(schema='fdt_update_balance_v77',authorized=False,root=str(ROOT),output=str(ROOT/'runs/update_balance_20261113_v77'),data=str(d),balance_data=str(b),eval_data=str(ed),
  cycle_source=str(source),cycle_sha=PARENT_SHA,cycle_algorithm_sha=ALGORITHM_SHA,storage_parent=str(storage),storage_parent_sha=STORAGE_PARENT_SHA,
  baseline_run=str(base),observed_V75_data=old['observed_V75_data'],start=None,cutoff=None,deadline=None,attempts=1920,epochs=1,batch=1,wall_seconds=900,max_calls=7000,cost=6752,
  cost_detail=dict(TRAIN1024_teacher_encode_readout=2048,TRAIN1920_live_encode_readout=3840,newDEV64_baseline_encode_readout=128,
   final_old340_cached_readout=340,final_newDEV64_encode_readout=128,fresh_long2_encode_readout=4,final_currenthead_cachedlong_comparison2=2,
   native_stream_parity4NN_plus2chunks=6,longchunks256=256,encode_total=3076,readout_total=3418,stream_chunk_total=258,block_work=2322,denoise=0),
  objective=OBJECTIVE,objective_report=obj,new_code_files=names,
  purpose='actual finite unsigned40/signed40/OWT20 update frequency and frozen V75 TRAIN base symbol weights, starting selected V75; repair observed sign interference without raw/exact bypass',
  optimizer='fresh final7AdamW LR1e-5 WD.01 clip1 BF16 scaler1024; upstream frozen; V75 old populatedAdam/scaler/RNG preserved',
  source_encoder_optimizer_updates=0,source_encoder_optimizer_policy='fully frozen V74/V73 source actually reencoded in BOTH finite teacher/head phases; no feature cache',
  exposure=dict(unique_TRAIN_rows=1024,authored896=True,correlated_quartets224=True,OWT_originalTRAIN128=True,finite_exposure_schedule=True,teacher_source_rows=1024,teacher_source_UTF8_bytes=1010943,head_source_rows=1920,head_source_UTF8_bytes=2720189,head_critical_bytes=86108,head_EOS_targets=1920,source_repeats=dict(signed=1,unsigned=6,OWT=3)),
  checkpoints=['initial/runtime.pt linked','teacher_complete/runtime.pt linked','model/runtime.pt linked'],maximum_additional_STOP_checkpoint=1,
  initial_restore_contract='warm NN from complete linked V75; all NEW runtime serialized against pinned STANDALONE V74 model-only base +active7; no linked-parent recursion/Exact text retrieval',
  baseline_contract='grammar-revised64 actual source before/after:32unchanged32edited/sameobservedfacts; observed340 fixed-upstream Commit-only; two262KiB fresh source plus native-stream4096 and currenthead saved-Commit parity',
  max_output_bytes=OUTPUT_CAP,start_free_bytes=START_FREE,stream_chunk_cap=258,long_block_work_cap=2322,
  cache_contract='loss-only teacher; head newTRAIN actual reencoded; observed retention one-level pinned direct neural Commit reference, no rawsource/GTprefix/GTlength decoder features',
  limitations='same1024uniqueTRAIN/1920headexposures(signed1,unsigned6,OWT3),frozenV75baseweights;same64evalfactswith32grammar-edits32unchanged/heldnamesobserved;notnewblindnaturalcorpus;old340/long2retention;noabstractlosslesscertificate/denoise0')
 p['CPU_reference_cases']=[dict(raw=str(base/'raw_OWT_00.pt'),data=p['retention_OWT'],index=0),dict(raw=str(base/'raw_new_English_00.pt'),data=str(Path(p['observed_V75_data'])/'dev_features.json'),index=0)]
 for case in p['CPU_reference_cases']:
  features=io.load_json(case['data'])
  if not isinstance(features,list) or not 0<=case['index']<len(features) or not {'id','prompt'}.issubset(features[case['index']]):
   raise ValueError('CPU compatibility case must resolve its actual baseline feature source')
 files=[ROOT/n for n in names]+[source,storage,base/'result.json',base/'root_result.json',base/'root_saved_review.json',base/'root_regression_severity_rollback_review.json',ROOT/'scripts/owned_cpu_preflight_v50.py']
 files.extend(d/n for n in ('train_features.json','train_targets.json','orders.json','manifest.json','builder_SHA.json','root_CPU_review.json','root_component_pins.json','root_engine_contract_v1.json','root_commit_reference_contract_v2.json','root_engine_reference_pins.json','root_legacy_commit_reference_contract.json'))
 files.extend(ed/n for n in ('features.json','targets.json','manifest.json','builder_SHA.json'))
 files.extend(b/n for n in ('orders.json','manifest.json','symbol_weight_source.json','builder_SHA.json','root_data_profile_CPU_review.json','root_engine_CPU_contract.json'))
 files.append(ROOT/'configs/research_base_selection_v76_review_20261113.json')
 for k in ('retention_OWT',):files.append(Path(p[k]))
 for key,ns in [('parent_data',('retained_English.json','dev_features.json','manifest.json')),('old_eval_data',('short_features.json','short_targets.json','long_features.json','long_targets.json','manifest.json')),('old_expression_data',('dev_features.json','dev_targets.json')),('observed_V74_data',('dev_features.json','dev_targets.json')),('observed_V75_data',('dev_features.json','dev_targets.json'))]:
  files.extend(Path(p[key])/n for n in ns)
 for label,count in [('OWT',16),('retained_English',36),('old_untrained128',128),('old_expression',32),('observed_V74_DEV',64),('new_English',64),('long',2)]:
  files.extend(base/f'raw_{label}_{i:02d}.pt' for i in range(count));files.append(base/(label+'_reconstructions.json'))
 # V75's indirect saved Commit packets point to direct V74 packets already
 # pinned in its plan. Carry those pins instead of opening arbitrary paths.
 files.extend(Path(f) for f in old['files'] if Path(f).name.startswith('raw_'))
 for f in files:p['files'][str(f)]=support.sha(f)
 assert support.sha(source)==PARENT_SHA and support.sha(storage)==STORAGE_PARENT_SHA
 for f,s in p['files'].items():
  assert support.sha(f)==s,f
  if f.endswith('.py'):ast.parse(Path(f).read_text(encoding='utf-8-sig'))
 longrows=io.load_json(Path(p['old_eval_data'])/'long_features.json')[:2]
 retained=io.load_json(p['retention_OWT'])+io.load_json(Path(p['parent_data'])/'retained_English.json')+io.load_json(Path(p['old_eval_data'])/'short_features.json')+io.load_json(Path(p['old_expression_data'])/'dev_features.json')+io.load_json(Path(p['observed_V74_data'])/'dev_features.json')+io.load_json(Path(p['observed_V75_data'])/'dev_features.json')
 assert len(retained)==340
 def rawbytes(row,commit):
  slots=(len(row['prompt'].encode())+15)//16
  return (slots+1)*16*(258*4+8)+((slots+2)*512*4 if commit else 0)+(32768 if commit else 16384)
 raw=sum(rawbytes(r,False) for r in retained)+sum(rawbytes(r,True) for r in dev+dev+longrows)+2*rawbytes(dict(prompt='x'*4096),True)
 teacher=sum(((len(r['prompt'].encode())+15)//16+1)*16*14+8192 for r in rows)
 linked_states=4*(2908192*4+8*1024**2)+2*2908192*8
 detail=dict(raw_and_parity=raw,loss_only_teacher=teacher,all_linked_states_including_STOP=linked_states,metadata_and_slack=32*1024**2)
 estimate=sum(detail.values());assert estimate<OUTPUT_CAP,(estimate,OUTPUT_CAP)
 p['output_estimate_including_all_raw_parentlinked_states_STOP_and32MiB']=estimate;p['output_estimate_detail']=detail
 p['disk_headroom_contract']='100GiB protected+1.25GiB hard owned including temp/current/STOP+8GiB headroom=109.25GiB start; no existing deletions'
 review.mkdir();io.atomic_json(unsealed,p);validate(unsealed,Path(p['output']))
 print(json.dumps(dict(actual_schema_PASS=True,pins=len(p['files']),cost=6752,cap=7000,estimated_output=estimate,detail=detail,hard_cap=OUTPUT_CAP,start_free_bytes=START_FREE)))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--seal',action='store_true');main(a.parse_args().seal)
