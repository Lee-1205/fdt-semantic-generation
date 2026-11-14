"""Fix actual V75 schema, costs/storage, independent CPU readiness, ONE clock."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_critical_retention_20261112_v75 import validate,PARENT_SHA,ALGORITHM_SHA,OUTPUT_CAP
from research.critical_retention_objective_v75 import OBJECTIVE,COEFFICIENTS
from research.fuzzy_factual_objective_v70 import training_weights

def main(seal):
    d=ROOT/'research/artifacts/critical_retention_20261112_v75_cpu';review=d/'execution_review';unsealed=review/'plan_unsealed.json'
    if seal:
        p=io.load_json(unsealed);output=Path(p['output']);validate(unsealed,output);assert not output.exists()
        cpu=review/'cpu_preflight_owned/final_status.json';proof=io.load_json(cpu);r=proof['report']
        assert proof['status']=='CPU_PREFLIGHT_PASS' and proof['exit']['all_owned_processes_gone'] and not proof['owner_imported_Torch']
        assert all(r[k] for k in ('fullsize_engine_restore','actual_linked_initial_and_partial_teacher_exact','actual_only7_gradient','live_source_reencoded','no_feature_cache','long_wrapper_all_tensors_exact','all_parent_tensors_exact','all_old_Adam_exact','fresh7Adam','encoder_binding_frozen','Exact_entries0'))
        assert not r['CUDA_initialized'] and r['optimizer_steps']==0 and r['active_tensors']==7 and r['active_readout_parameters']==2908192 and r['objective_report']==p['objective_report'] and r['engine_schema']=='fdt_live_critical_retention_v75_fullstate'
        independent=review/'root_independent_validate.json';assert io.load_json(independent)['status']=='ROOT_INDEPENDENT_V75_READY'
        for f in (cpu,independent):p['files'][str(f)]=support.sha(f)
        now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900);output.mkdir();path=output/'plan.json';io.atomic_json(path,p);validate(path,output)
        snap=output/'code_snapshot';snap.mkdir()
        for name in p['new_code_files']:
            f=ROOT/name;(snap/f.name).write_bytes(f.read_bytes())
        print(json.dumps(dict(plan_SHA=support.sha(path),cost=p['cost'],max_calls=3500,wall=900,reserve=120,output_cap=OUTPUT_CAP,start_free_bytes=p['start_free_bytes'])));return
    assert not unsealed.exists();base=ROOT/'runs/context_verbalizer_20261112_v74';old=io.load_json(base/'plan.json');p=dict(old);source=base/'model/checkpoints/state_0640.pt'
    names=['scripts/prepare_critical_retention_v75_cpu.py','scripts/review_critical_retention_v75_data_cpu.py','research/critical_retention_objective_v75.py','scripts/test_critical_retention_objective_v75_cpu.py',
        'research/linked_runtime_v75.py','scripts/test_linked_runtime_v75_cpu.py','scripts/critical_retention_engine_v75.py','scripts/test_critical_retention_engine_v75_cpu.py',
        'scripts/critical_retention_assess_v75.py','scripts/run_critical_retention_20261112_v75.py','scripts/supervise_critical_retention_v75.py','scripts/prepare_critical_retention_plan_v75.py']
    rows=io.load_json(d/'train_features.json');_,obj=training_weights(rows)
    for k in ('critical_CE_coefficient','critical_masks_TRAIN_only'):obj.pop(k,None)
    obj.update(objective=OBJECTIVE,coefficients=COEFFICIENTS,decoder_LR=1e-5,critical_groups_independently_normalized=True,teacher_cache='loss-only; no feature cache; live frozen source reencoded for head')
    p['files']={k:v for k,v in old['files'].items() if k.endswith('.py')}
    p.update(schema='fdt_critical_retention_v75',authorized=False,output=str(ROOT/'runs/critical_retention_20261112_v75'),data=str(d),cycle_source=str(source),cycle_sha=PARENT_SHA,cycle_algorithm_sha=ALGORITHM_SHA,
        baseline_run=str(base),observed_V74_data=old['data'],start=None,cutoff=None,deadline=None,attempts=640,epochs=1,batch=1,wall_seconds=900,max_calls=3500,cost=3360,
        cost_detail=dict(TRAIN640_teacher_encode_readout=1280,TRAIN640_live_encode_readout=1280,newDEV64_baseline_encode_readout=128,
            final_old276_cached_readout=276,final_newDEV64_encode_readout=128,fresh_long2_encode_readout=4,final_currenthead_cachedlong_comparison2=2,
            native_stream_parity4NN_plus2chunks=6,longchunks256=256,encode_total=1412,readout_total=1690,stream_chunk_total=258,block_work=2322,denoise=0),
        objective=OBJECTIVE,objective_report=obj,new_code_files=names,
        purpose='reduce observed correction-retention interference using mixed unsigned/signed/physical-stock contexts, separately normalized correct-critical preservation and wrong-critical correction; fixed actual fuzzy Commit boundary',
        optimizer='fresh final7AdamW LR1e-5 WD.01 clip1 BF16 scaler1024; upstream fully frozen; preserve all old V74 populated7Adam/scaler/RNG',
        source_encoder_optimizer_updates=0,source_encoder_optimizer_policy='fully frozen V74/V73 source; actual source reencoded in BOTH finite teacher/head phases, no feature cache',
        exposure=dict(TRAIN_rows=640,authored512=True,OWT_originalTRAIN128=True,onefiniteepoch=True,teacher_source_rows640=True,head_source_rows640=True,target_UTF8_bytes=969013,critical_bytes=29114,EOS_targets=640),
        checkpoints=['initial/runtime.pt linked','teacher_complete/runtime.pt linked','model/runtime.pt linked'],maximum_additional_STOP_checkpoint=1,
        initial_restore_contract='EVERY runtime is parent-required: all runtime +active7weights +pinned standalone V74 model-only parent, complete reconstructed recursive digest verified; NOT standalone/Exact text retrieval',
        baseline_contract='newDEV64 actual source baseline; observed V74 panels reused; final old276 Commit-only/new64actualsource/two262KiBactualsource; native-stream4096 parity and current-head old-Commit comparison without duplicated long raw',
        max_output_bytes=OUTPUT_CAP,start_free_bytes=109*1024**3,stream_chunk_cap=258,long_block_work_cap=2322,
        cache_contract='teacher targets/probabilities/flags only; no TRAIN feature cache; actual source→K_P/K_C→bound Commit each head attempt; no H/raw/gold prefix/GT length in decoder',
        limitations='new authored numerical contexts after observed V74 problems, correlated templates/pairs; same128oldOWTTRAIN; observed276short/2long are retention not blind benchmark; newDEV64 untrained names/surfaces; no abstract semantic/lossless certificate; denoise0')
    files=[ROOT/n for n in names]+[source,base/'result.json',base/'root_result.json',base/'root_saved_review.json',base/'root_retention_code_review.json',Path(p['retention_OWT']),ROOT/'scripts/owned_cpu_preflight_v50.py']
    files.extend(d/n for n in ('train_features.json','train_targets.json','dev_features.json','dev_targets.json','orders.json','manifest.json','root_data_review.json','root_engine_contract_v2.json','builder_snapshot_sha.json'))
    files.extend(Path(p['parent_data'])/n for n in ('retained_English.json','dev_features.json','manifest.json'))
    files.extend(Path(p['old_eval_data'])/n for n in ('short_features.json','short_targets.json','long_features.json','long_targets.json','manifest.json'))
    files.extend(Path(p['old_expression_data'])/n for n in ('dev_features.json','dev_targets.json'))
    files.extend(Path(p['observed_V74_data'])/n for n in ('dev_features.json','dev_targets.json'))
    for label,count in [('OWT',16),('retained_English',36),('old_untrained128',128),('old_expression',32),('new_English',64),('long',2)]:
        files.extend(base/f'raw_{label}_{i:02d}.pt' for i in range(count));files.append(base/(label+'_reconstructions.json'))
    for f in files:p['files'][str(f)]=support.sha(f)
    assert support.sha(source)==PARENT_SHA
    for file,sha in p['files'].items():
        assert support.sha(file)==sha,file
        if file.endswith('.py'):ast.parse(Path(file).read_text(encoding='utf-8-sig'))
    dev=io.load_json(d/'dev_features.json');longrows=io.load_json(Path(p['old_eval_data'])/'long_features.json')[:2]
    retained=io.load_json(p['retention_OWT'])+io.load_json(Path(p['parent_data'])/'retained_English.json')+io.load_json(Path(p['old_eval_data'])/'short_features.json')+io.load_json(Path(p['old_expression_data'])/'dev_features.json')+io.load_json(Path(p['observed_V74_data'])/'dev_features.json')
    def rawbytes(row,commit):
        slots=(len(row['prompt'].encode())+15)//16
        return (slots+1)*16*(258*4+8)+((slots+2)*512*4 if commit else 0)+(32768 if commit else 16384)
    raw=sum(rawbytes(r,False) for r in retained)+sum(rawbytes(r,True) for r in dev+dev+longrows)+2*rawbytes(dict(prompt='x'*4096),True)
    teacher=sum(((len(r['prompt'].encode())+15)//16+1)*16*14+8192 for r in rows)
    linked_states=4*(2908192*4+4*1024**2)+2*2908192*8
    estimate=raw+teacher+linked_states+32*1024**2
    assert estimate<OUTPUT_CAP,(estimate,OUTPUT_CAP)
    p['output_estimate_including_all_raw_parentlinked_states_STOP_and32MiB']=estimate
    p['output_estimate_detail']=dict(raw_and_parity=raw,loss_only_teacher=teacher,all_linked_states_including_STOP=linked_states,metadata_and_slack=32*1024**2)
    p['disk_headroom_contract']='100GiB protected+1GiB hard output INCLUDING temp/current/STOP+8GiB headroom=109GiB start; no existing file deletions'
    review.mkdir();io.atomic_json(unsealed,p);validate(unsealed,Path(p['output']))
    print(json.dumps(dict(actual_schema_PASS=True,pins=len(p['files']),cost=3360,cap=3500,estimated_output=estimate,detail=p['output_estimate_detail'],hard_cap=OUTPUT_CAP,start_free_bytes=p['start_free_bytes'],objective=obj)))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seal',action='store_true');main(a.parse_args().seal)
