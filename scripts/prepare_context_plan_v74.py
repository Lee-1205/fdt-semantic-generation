"""A new bounded readout/data hypothesis; prior terminal budgets remain closed."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_context_verbalizer_20261112_v74 import validate,PARENT_SHA,ALGORITHM_SHA,EXACT_POLICY,OUTPUT_CAP
from research.frozen_verbalizer_objective_v72 import OBJECTIVE
from research.fuzzy_factual_objective_v70 import training_weights
def main(seal):
    d=ROOT/'research/artifacts/context_verbalizer_20261112_v74_cpu_v2';review=d/'execution_review';unsealed=review/'plan_unsealed.json'
    if seal:
        p=json.loads(unsealed.read_text());output=Path(p['output']);validate(unsealed,output);assert not output.exists()
        proof=review/'cpu_preflight_owned/final_status.json';a=json.loads(proof.read_text());r=a['report']
        assert a['status']=='CPU_PREFLIGHT_PASS' and a['exit']['all_owned_processes_gone'] and not a['owner_imported_Torch']
        assert all(r[k] for k in ('fullsize_engine_restore','source_linked_initial_exact_restore','actual_cache_partial_restore','actual_only7_gradient','long_wrapper_all_tensors_exact','all_parent_tensors_exact','all_old_Adam_exact','fresh7Adam','encoder_binding_frozen','Exact_entries0'))
        assert not r['CUDA_initialized'] and r['optimizer_steps']==0 and r['active_readout_parameters']==2908192 and r['active_tensors']==7
        assert r['engine_schema']=='fdt_context_verbalizer_v74_fullstate' and r['objective_report']==p['objective_report']
        independent=review/'root_independent_validate.json';assert json.loads(independent.read_text(encoding='utf-8-sig'))['status']=='ROOT_INDEPENDENT_V74_READY'
        for file in (proof,independent):p['files'][str(file)]=support.sha(file)
        now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900)
        output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,indent=2)+'\n');validate(path,output);snap=output/'code_snapshot';snap.mkdir()
        for name in p['new_code_files']:
            file=ROOT/name;(snap/file.name).write_bytes(file.read_bytes())
        print(json.dumps(dict(plan_SHA=support.sha(path),cost=2592,max_calls=2700,wall=900,reserve=120,output_cap=OUTPUT_CAP)));return
    assert not unsealed.exists();oldrun=ROOT/'runs/fuzzy_extraction_20261112_v73';old=json.loads((oldrun/'plan.json').read_text());p=dict(old);parent=oldrun/'model/checkpoints/state_0960.pt'
    names=['scripts/prepare_context_verbalizer_v74_cpu.py','scripts/review_context_verbalizer_v74_data_cpu.py','scripts/context_engine_v74.py','scripts/test_context_engine_v74_cpu.py',
        'scripts/context_assess_v74.py','scripts/run_context_verbalizer_20261112_v74.py','scripts/supervise_context_verbalizer_v74.py','scripts/prepare_context_plan_v74.py']
    rows=json.loads((d/'train_features.json').read_text());_,obj=training_weights(rows);obj.update(objective=OBJECTIVE,anchor_coefficient=.05,decoder_LR=3e-5)
    p['files']={k:v for k,v in old['files'].items() if k.endswith('.py')}
    p.update(schema='fdt_context_verbalizer_v74',authorized=False,output=str(ROOT/'runs/context_verbalizer_20261112_v74'),data=str(d),cycle_source=str(parent),cycle_sha=PARENT_SHA,cycle_algorithm_sha=ALGORITHM_SHA,
        baseline_run=str(oldrun),old_eval_data=old['eval_data'],start=None,cutoff=None,deadline=None,attempts=640,epochs=1,batch=1,wall_seconds=900,max_calls=2700,cost=2592,
        cost_detail=dict(TRAIN_encode640_teacher640=1280,head_training640=640,newDEV64_baseline_encode_readout=128,final276short_cached_readout=276,oldlong2_cached_readout=2,
            freshlong2_encode_readout=4,freshlong_chunks256=256,execution_parity4NN_plus2chunks=6,encode_total=708,readout_total=1626,stream_chunks_total=258,block_work=2322,denoise=0),
        objective=OBJECTIVE,objective_report=obj,exact_policy=EXACT_POLICY,new_code_files=names,
        purpose='cross signed-unit predicate and y-internal/z-final name contexts to improve actual fixed fuzzy Commit reading; no held-name training leakage',
        optimizer='fresh final7AdamW LR3e-5 WD.01 clip1 BF16 enabledscaler1024; all upstream fully frozen; preserve V73 old23Adam/scaler/ALLRNG',
        source_encoder_optimizer_updates=0,source_encoder_optimizer_policy='all V73 upstream frozen; actual per-input K_P/K_C formed',
        exposure=dict(TRAIN_rows640=True,new_authored512=True,OWT_originalTRAIN128=True,onefiniteepoch=True,target_UTF8_bytes=1039965,critical_bytes=34194,EOS_targets=640),
        checkpoints=['source-linked initial_runtime.pt + preserved standalone V73 model','cache_complete/0 full','model/640 full'],maximum_additional_STOP_checkpoint=1,
        initial_restore_contract='initial fresh optimizer/scaler/ALLRNG/cursors/orders/weights signature stored small; model supplied ONLY by pinned immutable standalone V73 parent; full recursive digest verified',
        baseline_contract='newDEV64 actual raw-source baseline; existing V73 panels reused; final276short+2long cached readout, two largest long actual fresh source checks with exact cache parity',
        max_output_bytes=OUTPUT_CAP,start_free_bytes=223*1024**3//2,stream_chunk_cap=258,long_block_work_cap=2322,
        cache_contract='fixed upstream actual bound Commit keyed full nonhead digest/source id/version/SHA; targets and teacher separate; no H/raw/gold length/prefix in decoder',
        limitations='targeted authored cross-coverage after V73 results, correlated sign pairs; old panels already observed, newDEV64 untrained different wrappers/names; cached-boundary tests not new encoder training; two fresh long checks; no blind natural benchmark/abstract semantic/lossless certificate')
    files=[ROOT/n for n in names]+[parent,oldrun/'result.json',oldrun/'root_result.json',oldrun/'root_saved_review.json',oldrun/'root_context_coverage_cpu.json',Path(p['retention_OWT']),ROOT/'scripts/owned_cpu_preflight_v50.py']
    files.extend(d/n for n in ('train_features.json','train_targets.json','dev_features.json','dev_targets.json','orders.json','manifest.json','root_data_review.json','root_engine_contract_v2.json'))
    files.extend(Path(p['parent_data'])/n for n in ('retained_English.json','dev_features.json','manifest.json'))
    files.extend(Path(p['old_eval_data'])/n for n in ('short_features.json','short_targets.json','long_features.json','long_targets.json','manifest.json'))
    files.extend(Path(p['old_expression_data'])/n for n in ('dev_features.json','dev_targets.json'))
    for label,count in [('OWT',16),('retained_English',36),('new_English',128),('old_expression',32),('long',2)]:
        files.extend(oldrun/f'raw_{label}_{i:02d}.pt' for i in range(count));files.append(oldrun/(label+'_reconstructions.json'))
    for f in files:p['files'][str(f)]=support.sha(f)
    assert support.sha(parent)==PARENT_SHA
    for file,digest in p['files'].items():
        assert support.sha(file)==digest,file
        if file.endswith('.py'):ast.parse(Path(file).read_text(encoding='utf-8-sig'))
    def rawbytes(r):
        slots=(len(r['prompt'].encode())+15)//16;return (slots+1)*16*(258*4+8)+(slots+2)*512*4+32768
    dev=json.loads((d/'dev_features.json').read_text());long=json.loads((Path(p['old_eval_data'])/'long_features.json').read_text())[:2]
    retained=json.loads(Path(p['retention_OWT']).read_text())+json.loads((Path(p['parent_data'])/'retained_English.json').read_text())+json.loads((Path(p['old_eval_data'])/'short_features.json').read_text())+json.loads((Path(p['old_expression_data'])/'dev_features.json').read_text())
    raw=sum(rawbytes(r) for r in dev+retained+dev+long+long);model=123497080*4;adam=2908192*8
    cache=sum(((len(r['prompt'].encode())+15)//16+2)*512*4+((len(r['prompt'].encode())+15)//16+1)*16*14+32768 for r in rows+dev)
    estimate=raw+2*model+adam+(model+adam)+cache+3*4*1024**2+16*1024**2+64*1024**2
    assert estimate<OUTPUT_CAP
    p['output_estimate_including_additional_STOP_initial_runtime_and64MiB']=estimate
    p['disk_headroom_contract']='100GiB protected +3.5GiB hard owned output cap INCLUDES checkpoint temp/current/STOP +8GiB headroom =111.5GiB start; no hidden checkpoint copies'
    review.mkdir();unsealed.write_text(json.dumps(p,indent=2)+'\n');validate(unsealed,Path(p['output']))
    print(json.dumps(dict(actual_schema_PASS=True,pins=len(p['files']),cost=2592,cap=2700,estimated_output=estimate,hard_output_cap=OUTPUT_CAP,start_floor=p['start_free_bytes'],objective=obj)))
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--seal',action='store_true');main(a.parse_args().seal)
