"""New finite full-cycle factual fidelity training; old budgets remain closed."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_factual_cycle_20261112_v70 import validate,EXACT_POLICY

def main(seal):
    d=ROOT/'research/artifacts/factual_cycle_20261112_v70_cpu_v2';review=d/'execution_review';unsealed=review/'plan_unsealed.json'
    if seal:
        p=json.loads(unsealed.read_text());output=Path(p['output']);validate(unsealed,output);assert not output.exists()
        proof=review/'cpu_preflight_owned/final_status.json';a=json.loads(proof.read_text());r=a['report']
        assert a['status']=='CPU_PREFLIGHT_PASS' and a['exit']['all_owned_processes_gone'] and not a['owner_imported_Torch']
        assert r['fullsize_engine_restore'] and r['actual_warm_size_roundtrip_gradient'] and r['actual_new_objective_gradient'] and r['all_V67_tensors_exact']
        assert r['fresh_joint_optimizer'] and not r['parent_Adam_used_for_training'] and r['architecture_unchanged'] and r['long_wrapper_all_tensors_exact']
        assert r['engine_schema']=='fdt_live_fuzzy_roundtrip_v70_fullstate' and r['objective_report']==p['objective_report']
        assert r['Exact_entries0'] and not r['CUDA_initialized'] and r['optimizer_steps']==0 and r['actual_trainable_parameters']==17677856
        independent=review/'root_independent_validate.json';assert json.loads(independent.read_text(encoding='utf-8-sig'))['status']=='ROOT_INDEPENDENT_V70_READY'
        p['files'][str(proof)]=support.sha(proof);p['files'][str(independent)]=support.sha(independent)
        now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900)
        output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,indent=2)+'\n');validate(path,output)
        snapshots=output/'code_snapshot';snapshots.mkdir()
        for name in p['new_code_files']:
            f=ROOT/name;(snapshots/f.name).write_bytes(f.read_bytes())
        print(json.dumps(dict(plan_sha=support.sha(path),cost=1452,max_calls=1500,attempts512=True,wall900reserve120=True,objective=p['objective_report'])));return
    assert not unsealed.exists();oldrun=ROOT/'runs/hard_fuzzy_cycle_20261112_v67';old=json.loads((oldrun/'plan.json').read_text());p=dict(old)
    parent=oldrun/'model/checkpoints/state_1024.pt';proof=json.loads((d/'root_data_review.json').read_text())
    names=['research/fuzzy_factual_objective_v70.py','scripts/factual_cycle_engine_v70.py','scripts/run_factual_cycle_20261112_v70.py',
        'scripts/supervise_factual_cycle_v70.py','scripts/test_factual_cycle_v70_cpu.py','scripts/prepare_factual_cycle_v70_cpu.py',
        'scripts/review_factual_cycle_v70_data_cpu.py','scripts/prepare_factual_cycle_plan_v70.py','scripts/factual_cycle_long_eval_v70.py']
    p['files']={k:v for k,v in old['files'].items() if k.endswith('.py')}
    p.update(schema='fdt_factual_cycle_v70',authorized=False,output=str(ROOT/'runs/factual_cycle_20261112_v70'),data=str(d),
        cycle_source=str(parent),cycle_sha='b47c9bdf920605b153c1ebfaa31a28579eb35c1784de6037ba54f4674cdc18b9',
        cycle_algorithm_sha='d8b2ba555605913d38415bf7dc427b16f50245e34b542e5f91c831f059b6cd20',
        baseline_run=str(oldrun),baseline_long_run=str(ROOT/'runs/long_extrapolation_20261112_v69'),
        retention_OWT=str(ROOT/'research/artifacts/broad_long_codec_20261111_v53_cpu/dev.json'),long_data=str(ROOT/'research/artifacts/long_extrapolation_20261112_v69_cpu'),
        start=None,cutoff=None,deadline=None,attempts=512,epochs=1,wall_seconds=900,max_calls=1500,cost=1452,
        cost_detail=dict(train_encode_readout=1024,new_English_DEV=64,retained_OWT=32,retained_English=72,long_encode_readout=4,long_stream_chunks=256,final_long_block_work=2304,fresh_baseline_calls=0),
        checkpoints=[0,1,256,512],maximum_additional_STOP_checkpoint=1,new_code_files=names,objective_report=proof['objective_report'],objective=proof['objective_report']['objective'],
        purpose='reduce numeric/entity source-content loss in actual live fuzzy languagecycle; prospective independent TRAIN coverage and .25 critical name/number CE; retain old English and two observed 16x inputs',
        optimizer='fresh LR groups1e-6/3e-6/1e-4 AdamW WD.01 clip1 BF16 enabled scaler1024; preserve V67 populated30Adam',
        exposure=dict(TRAIN_rows=512,authored384=True,OWT128=True,row_selections=512,target_UTF8_bytes=940237,critical_bytes=38480,EOS_targets=512),
        baseline_contract='reuse V67 final OWT16/English36 and V69 largest2 saved outputs; no fresh baseline calls or Korean calls',
        stream_chunk_cap=256,long_block_work_cap=2304,max_output_bytes=4*1024**3,
        limitations='changed prospective coverage plus content loss; correlated authored facts/observed retention; no unique abstract-meaning or whole languagecycle completion claim; Exact dormant')
    files=[ROOT/n for n in names]+[parent,oldrun/'result.json',oldrun/'root_saved_review.json',Path(p['retention_OWT']),
        Path(p['baseline_long_run'])/'root_meaning_review_v2.json',Path(p['baseline_long_run'])/'result.json',
        ROOT/'research/streaming_fuzzy_codec_v69.py',ROOT/'research/fixed_utf8_readout_v69.py',ROOT/'scripts/owned_cpu_preflight_v50.py']
    files.extend(d/n for n in ('train_features.json','train_targets.json','dev_features.json','dev_targets.json','orders.json','retained_English.json','manifest.json','root_engine_contract_v2.json','root_data_review.json'))
    files.extend(Path(p['long_data'])/n for n in ('features.json','targets.json','manifest.json','root_CPU_contract.json'))
    for f in files:p['files'][str(f)]=support.sha(f)
    assert support.sha(parent)==p['cycle_sha']
    for name,digest in p['files'].items():
        assert support.sha(name)==digest,name
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    review.mkdir();unsealed.write_text(json.dumps(p,indent=2)+'\n');validate(unsealed,Path(p['output']))
    print(json.dumps(dict(actual_schema_PASS=True,cost=1452,max_calls=1500,pins=len(p['files']),objective=proof['objective_report'])))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
