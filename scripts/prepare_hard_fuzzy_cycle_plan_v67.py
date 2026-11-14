"""New finite whole-reconstruction objective; no closed-clock reuse."""
import argparse,ast,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_hard_fuzzy_cycle_20261112_v67 import validate,EXACT_POLICY
def main(seal):
    d=ROOT/'research/artifacts/hard_fuzzy_cycle_20261112_v67_cpu_v2';review=d/'execution_review';unsealed=review/'plan_unsealed.json'
    if seal:
        p=json.loads(unsealed.read_text());output=Path(p['output']);validate(unsealed,output);assert not output.exists()
        proof=review/'cpu_preflight_owned/final_status.json';a=json.loads(proof.read_text());r=a['report']
        assert a['status']=='CPU_PREFLIGHT_PASS' and a['exit']['all_owned_processes_gone'] and not a['owner_imported_Torch']
        assert r['fullsize_engine_restore'] and r['actual_warm_size_roundtrip_gradient'] and r['actual_new_objective_gradient'] and r['all_V65_tensors_exact']
        assert r['fresh_joint_optimizer'] and not r['parent_Adam_used_for_training'] and r['architecture_unchanged']
        assert r['engine_schema']=='fdt_live_fuzzy_roundtrip_v67_fullstate' and r['objective_report']==p['objective_report']
        assert r['Exact_entries0'] and not r['CUDA_initialized'] and r['optimizer_steps']==0 and r['actual_trainable_parameters']==17677856
        p['files'][str(proof)]=support.sha(proof);now=time.time();p.update(authorized=True,start=now,cutoff=now+1020,deadline=now+1200)
        output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,indent=2)+'\n');validate(path,output)
        snapshots=output/'code_snapshot';snapshots.mkdir()
        for name in p['new_code_files']:
            f=ROOT/name;(snapshots/f.name).write_bytes(f.read_bytes())
        info=dict(plan_sha=support.sha(path),cost=2156,max_calls=2200,attempts1024=True,wall1200reserve180=True,objective=p['objective_report'],all_V66_eval_only=True)
        (output/'root_prelaunch_review.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info));return
    assert not unsealed.exists();oldrun=ROOT/'runs/balanced_fuzzy_cycle_20261112_v65';old=json.loads((oldrun/'plan.json').read_text());p=dict(old)
    v66=ROOT/'runs/fresh_fuzzy_eval_20261112_v66';parent=oldrun/'model/checkpoints/state_0512.pt';proof=json.loads((d/'root_data_review.json').read_text())
    names=['research/fuzzy_cycle_objective_v67.py','scripts/hard_fuzzy_cycle_engine_v67.py','scripts/run_hard_fuzzy_cycle_20261112_v67.py',
        'scripts/supervise_hard_fuzzy_cycle_v67.py','scripts/test_hard_fuzzy_cycle_v67_cpu.py','scripts/prepare_hard_fuzzy_cycle_v67_cpu.py',
        'scripts/review_hard_fuzzy_cycle_data_v67_cpu.py','scripts/prepare_hard_fuzzy_cycle_plan_v67.py']
    p['files']={k:v for k,v in old['files'].items() if k.endswith('.py')}
    p.update(schema='fdt_hard_fuzzy_cycle_v67',authorized=False,output=str(ROOT/'runs/hard_fuzzy_cycle_20261112_v67'),data=str(d),
        cycle_source=str(parent),cycle_sha='3351bae2358ed660effbb67c642650b3401b9142f6b9b09e19d823310f9cc7a4',
        cycle_algorithm_sha='bdc7409c0da8fa43fe8c314cab742e9c50c41ef611706897231883c55485dceb',
        baseline_run=str(oldrun),baseline_English_run=str(v66),start=None,cutoff=None,deadline=None,attempts=1024,epochs=1,max_calls=2200,cost=2156,
        cost_detail=dict(train_encode_readout=2048,final_original_DEV=32,retained_untrained_English=72,zero_fuzzy_readout=4,fresh_baseline_calls=0),
        checkpoints=[0,1,512,1024],new_code_files=names,objective_report=proof['objective_report'],objective=proof['objective_report']['objective'],
        purpose='reduce rare incorrect bytes and neural last-slot false EOS hidden by mean CE; broaden finite TRAIN and all16 long EOS offsets',
        optimizer='fresh same LR groups1e-6/3e-6/1e-4 AdamW WD.01 clip1 BF16 scaler1024; preserve immutable V65 populated30Adam',
        exposure=dict(TRAIN_rows=1024,unique_source_documents=1011,row_selections=1024,target_UTF8_bytes=7978178,EOS_targets=1024),
        baseline_contract='reuse identical V65 final originalDEV plus V66 frozen English saved raw/metrics; no new baseline or Korean calls',
        limitations='hard-byte/tailEOS losses plus broader TRAIN/ragged tail allchange; observed retention36 notblind; no Korean improvement measurement; Exact dormant')
    files=[ROOT/n for n in names]+[Path(p['source']),Path(p['head_source']),parent,oldrun/'result.json',oldrun/'root_saved_review.json',v66/'root_saved_review.json',v66/'reconstructions.json']
    files.extend(oldrun/f'raw_final_{i:02d}.pt' for i in range(16))
    held=json.loads((ROOT/'research/artifacts/fresh_fuzzy_eval_20261112_v66_cpu/rows.json').read_text())
    files.extend(v66/f'raw_{i:02d}.pt' for i,r in enumerate(held) if r['language']=='English')
    files.extend(d/n for n in ('train.json','dev.json','orders.json','retained_English.json','manifest.json','root_engine_contract.json','root_data_review.json'))
    for f in files:p['files'][str(f)]=support.sha(f)
    assert support.sha(parent)==p['cycle_sha']
    for name,digest in p['files'].items():
        assert support.sha(name)==digest,name
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    review.mkdir();unsealed.write_text(json.dumps(p,indent=2)+'\n');validate(unsealed,Path(p['output']))
    print(json.dumps(dict(actual_schema_PASS=True,cost=2156,max_calls=2200,pins=len(p['files']),objective=proof['objective_report'])))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
