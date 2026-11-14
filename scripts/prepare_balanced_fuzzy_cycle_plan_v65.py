"""Fixed V65 scope: TRAIN-derived rare-byte emphasis and wider live cycle."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_balanced_fuzzy_cycle_20261112_v65 import validate,EXACT_POLICY

def main(seal):
    d=ROOT/'research/artifacts/balanced_fuzzy_cycle_20261112_v65_cpu';review=d/'execution_review';unsealed=review/'plan_unsealed.json'
    if seal:
        p=json.loads(unsealed.read_text());output=Path(p['output']);validate(unsealed,output);assert not output.exists()
        proof=review/'cpu_preflight_owned/final_status.json';r=json.loads(proof.read_text());a=r['report']
        assert r['status']=='CPU_PREFLIGHT_PASS' and r['exit']['all_owned_processes_gone'] and not r['owner_imported_Torch']
        assert a['fullsize_engine_restore'] and a['actual_warm_size_roundtrip_gradient'] and a['all_V64_tensors_exact'] and a['architecture_unchanged']
        assert a['fresh_joint_optimizer'] and not a['parent_Adam_used_for_training'] and a['baseline_reused_not_fresh_assessment']
        assert a['engine_schema']=='fdt_live_fuzzy_roundtrip_v65_fullstate' and a['objective_report']==p['objective_report']
        assert a['Exact_entries0'] and not a['CUDA_initialized'] and a['optimizer_steps']==0 and a['actual_trainable_parameters']==17677856
        p['files'][str(proof)]=support.sha(proof);now=time.time();p.update(authorized=True,start=now,cutoff=now+1020,deadline=now+1200)
        output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,indent=2)+'\n');validate(path,output)
        snapshots=output/'code_snapshot';snapshots.mkdir()
        for name in p['new_code_files']:
            f=ROOT/name;(snapshots/f.name).write_bytes(f.read_bytes())
        summary=dict(plan_sha=support.sha(path),cost=1060,cap=1100,wall_seconds=1200,reserve_seconds=180,objective=p['objective_report'],Exact_policy=EXACT_POLICY)
        (output/'root_prelaunch_review.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary));return
    assert not unsealed.exists()
    oldrun=ROOT/'runs/fuzzy_roundtrip_20261112_v64';old=json.loads((oldrun/'plan.json').read_text());p=dict(old)
    p['files']={k:v for k,v in old['files'].items() if k.endswith('.py')};newcode=[
        'research/fuzzy_cycle_objective_v65.py','scripts/balanced_fuzzy_cycle_engine_v65.py','scripts/run_balanced_fuzzy_cycle_20261112_v65.py',
        'scripts/supervise_balanced_fuzzy_cycle_v65.py','scripts/test_balanced_fuzzy_cycle_v65_cpu.py','scripts/prepare_balanced_fuzzy_cycle_v65_cpu.py',
        'scripts/review_balanced_fuzzy_cycle_data_v65_cpu.py','scripts/prepare_balanced_fuzzy_cycle_plan_v65.py']
    cycle=oldrun/'model/checkpoints/state_0256.pt';data_review=json.loads((d/'root_data_review.json').read_text())
    p.update(schema='fdt_balanced_fuzzy_cycle_v65',authorized=False,output=str(ROOT/'runs/balanced_fuzzy_cycle_20261112_v65'),data=str(d),
        cycle_source=str(cycle),cycle_sha='231a6287f3fc94e0e5ce426beb716203c35cfc93f75699688f3e4b95fcbf8d9e',
        cycle_algorithm_sha='f97942875c73f3567ef210c6ab648199d0a15ab469193aa21aabf1fd2a1f3b96',baseline_run=str(oldrun),
        start=None,cutoff=None,deadline=None,attempts=512,epochs=1,max_calls=1100,cost=1060,
        cost_detail=dict(train_encode_readout=1024,final_encode_readout=32,zero_fuzzy_readout=4,fresh_baseline_calls=0),
        checkpoints=[0,1,256,512],new_code_files=newcode,objective_report=data_review['objective_report'],
        purpose='correct rare uppercase/number/symbol precision in actual live language-fuzzy-language cycle, with TRAIN-only bounded class balance and wider finite exposure',
        objective=data_review['objective_report']['objective'],
        optimizer='fresh same LR groups1e-6/3e-6/1e-4 AdamW WD.01 clip1 BF16 scaler1024; preserve closed V64 populated30Adam',
        exposure=dict(TRAIN_rows=512,unique_source_documents=505,row_selections=512,target_UTF8_bytes=3724282,EOS_targets=512),
        baseline_contract='reuse identical V64 final weights/architecture/DEV raw and metrics, not a fresh independent assessment',
        limitations='loss weighting and wider TRAIN exposure both change; observed adaptive DEV; same-input fuzzy codec is not abstract semantic uniqueness proof; Exact remains dormant')
    files=[ROOT/n for n in newcode]+[Path(p['source']),Path(p['head_source']),cycle,oldrun/'result.json',oldrun/'root_saved_review.json',oldrun/'final_reconstructions.json']
    files.extend(oldrun/f'raw_final_{i:02d}.pt' for i in range(16))
    files.extend(d/name for name in ('train.json','dev.json','orders.json','manifest.json','root_engine_contract.json','root_data_review.json'))
    for f in files:p['files'][str(f)]=support.sha(f)
    assert support.sha(cycle)==p['cycle_sha']
    for name,digest in p['files'].items():
        assert support.sha(name)==digest,name
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    review.mkdir();unsealed.write_text(json.dumps(p,indent=2)+'\n');validate(unsealed,Path(p['output']))
    print(json.dumps(dict(actual_schema_PASS=True,cost=1060,max_calls=1100,objective=p['objective_report'],pins=len(p['files']))))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
