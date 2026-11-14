"""Actual fixed-cost plan, then new sealed clock after independent review."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_frozen_verbalizer_20261112_v72 import validate,PARENT_SHA,ALGORITHM_SHA,EXACT_POLICY
from research.fuzzy_factual_objective_v70 import training_weights
from research.frozen_verbalizer_objective_v72 import OBJECTIVE

def main(seal):
    d=ROOT/'research/artifacts/frozen_verbalizer_20261112_v72_cpu';review=d/'execution_review';unsealed=review/'plan_unsealed.json'
    if seal:
        p=json.loads(unsealed.read_text());output=Path(p['output']);validate(unsealed,output);assert not output.exists()
        proof=review/'cpu_preflight_owned/final_status.json';a=json.loads(proof.read_text());r=a['report']
        assert a['status']=='CPU_PREFLIGHT_PASS' and a['exit']['all_owned_processes_gone'] and not a['owner_imported_Torch']
        assert all(r[k] for k in ('fullsize_engine_restore','actual_cache_partial_restore','actual_only7_gradient','all_parent_tensors_exact','all_old_Adam_exact','fresh7Adam','encoder_and_role_projection_frozen','Exact_entries0'))
        assert not r['CUDA_initialized'] and r['optimizer_steps']==0 and r['active_readout_parameters']==2908192 and r['active_tensor_count']==7
        assert r['engine_schema']=='fdt_frozen_verbalizer_v72_fullstate' and r['objective_report']==p['objective_report']
        independent=review/'root_independent_validate.json';assert json.loads(independent.read_text(encoding='utf-8-sig'))['status']=='ROOT_INDEPENDENT_V72_READY'
        for f in (proof,independent):p['files'][str(f)]=support.sha(f)
        now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900)
        output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,indent=2)+'\n');validate(path,output)
        snapshots=output/'code_snapshot';snapshots.mkdir()
        for name in p['new_code_files']:
            f=ROOT/name;(snapshots/f.name).write_bytes(f.read_bytes())
        print(json.dumps(dict(plan_sha=support.sha(path),cost=p['cost'],max_calls=p['max_calls'],attempts=640,wall=900,reserve=120)));return
    assert not unsealed.exists();oldrun=ROOT/'runs/factual_cycle_20261112_v70';old=json.loads((oldrun/'plan.json').read_text());p=dict(old)
    parent=oldrun/'model/checkpoints/state_0512.pt';parentdata=Path(old['data'])
    names=['research/frozen_verbalizer_objective_v72.py','scripts/frozen_verbalizer_engine_v72.py','scripts/frozen_verbalizer_assess_v72.py',
        'scripts/run_frozen_verbalizer_20261112_v72.py','scripts/supervise_frozen_verbalizer_v72.py','scripts/prepare_frozen_verbalizer_plan_v72.py',
        'scripts/test_frozen_engine_v72_cpu.py','scripts/test_frozen_verbalizer_v72_cpu.py','scripts/prepare_frozen_verbalizer_v72_cpu.py']
    rows=json.loads((d/'train_features.json').read_text());_,objective=training_weights(rows);objective.update(objective=OBJECTIVE,anchor_coefficient=.05,decoder_LR=3e-5)
    p['files']={k:v for k,v in old['files'].items() if k.endswith('.py')}
    p.update(schema='fdt_frozen_verbalizer_v72',authorized=False,output=str(ROOT/'runs/frozen_verbalizer_20261112_v72'),data=str(d),
        cycle_source=str(parent),cycle_sha=PARENT_SHA,cycle_algorithm_sha=ALGORITHM_SHA,baseline_run=str(oldrun),parent_data=str(parentdata),
        start=None,cutoff=None,deadline=None,attempts=640,epochs=1,wall_seconds=900,max_calls=2200,cost=2110,
        cost_detail=dict(TRAIN_encode640_teacher640=1280,newDEV32_fresh_initial_encode_readout=64,head_training640=640,
            cached_final_OWT16_English36_priorEnglish32_long2_newDEV32=118,fresh_full_path4_encode_readout=8,encode_total=676,readout_total=1434,denoise=0),
        checkpoints=['initial/0','cache_complete/0','model/1','model/320','model/640'],maximum_additional_STOP_checkpoint=1,new_code_files=names,
        objective_report=objective,objective=OBJECTIVE,exact_policy=EXACT_POLICY,source_encoder_optimizer_updates=0,
        purpose='fixed actual fuzzy extraction/binding; train only final seven readout tensors to reduce factual expression drift with TRAIN-only noncritical-correct probability anchor',
        optimizer='FRESH seven-tensor AdamW LR3e-5 WD.01 clip1 BF16 enabled GradScaler1024; V70 30Adam/scaler/ALLRNG preserved',
        exposure=dict(TRAIN_source_rows=640,initial_teacher_rows=640,head_row_selections=640,finite_epoch=1,target_UTF8_bytes=954100,critical_bytes=43992,EOS_targets=640),
        cache_contract='actual fixed-weight dynamic bound Commit; provenance/digest keys; TRAIN teacher probability/labels/masks separate from features; no source H/bytes in decoder; no encoder training claimed',
        baseline_contract='newDEV32 actual source path before training; old V70 86 cached Commit compare with unchanged upstream; final 4 fresh full-path checks; no fresh long encoder evaluation',
        max_output_bytes=4*1024**3,progress_stall_seconds=120,
        limitations='English observed/design-selected correlated panels and authored expression families; final86 old+32new cached-boundary readout evaluations, four fresh pipeline checks; fixed upstream cannot correct all upstream damage; not independent abstract meaning certificate or losslessness; Exact dormant; denoise0')
    files=[ROOT/n for n in names]+[parent,oldrun/'result.json',oldrun/'root_saved_review.json',oldrun/'root_result.json',
        Path(p['retention_OWT']),ROOT/'scripts/owned_cpu_preflight_v50.py']
    files.extend(d/n for n in ('train_features.json','train_targets.json','dev_features.json','dev_targets.json','orders.json','manifest.json','root_CPU_contract.json','root_data_review.json','root_engine_contract_v2.json'))
    files.extend(parentdata/n for n in ('retained_English.json','dev_features.json','manifest.json'))
    files.extend(Path(p['long_data'])/n for n in ('features.json','targets.json','manifest.json'))
    files.extend(oldrun/f'raw_final_{i:02d}.pt' for i in range(16));files.extend(oldrun/f'raw_retained_English_{i:02d}.pt' for i in range(36))
    files.extend(oldrun/f'raw_new_English_{i:02d}.pt' for i in range(32));files.extend(oldrun/f'raw_long_{i}.pt' for i in range(2))
    for f in files:p['files'][str(f)]=support.sha(f)
    assert support.sha(parent)==PARENT_SHA
    for name,digest in p['files'].items():
        assert support.sha(name)==digest,name
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    review.mkdir();unsealed.write_text(json.dumps(p,indent=2)+'\n');validate(unsealed,Path(p['output']))
    print(json.dumps(dict(actual_schema_PASS=True,cost=2110,max_calls=2200,pins=len(p['files']),objective=objective)))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
