"""Freeze new held text before ONE bounded zero-optimizer assessment."""
import argparse,ast,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_fresh_fuzzy_eval_20261112_v66 import validate,PARENT_SHA
def main(seal):
    data=ROOT/'research/artifacts/fresh_fuzzy_eval_20261112_v66_cpu';review=data/'execution_review';path=review/'plan_unsealed.json'
    if seal:
        p=json.loads(path.read_text());output=Path(p['output']);validate(path,output);assert not output.exists()
        proof=review/'cpu_preflight_owned/final_status.json';state=json.loads(proof.read_text());r=state['report']
        assert state['status']=='CPU_PREFLIGHT_PASS' and state['exit']['all_owned_processes_gone'] and not state['owner_imported_Torch']
        assert r['fullsize_model_restore'] and r['all_parent_tensors_exact'] and r['all_old_Adam_exact'] and r['parent_scaler_ALLRNG_preserved']
        assert not r['CUDA_initialized'] and r['actual_parent_forward']==r['optimizer_steps']==r['trainable_parameters']==0
        assert r['runtime_schema']=='readonly_fuzzy_cycle_v66'
        p['files'][str(proof)]=support.sha(proof);now=time.time();p.update(authorized=True,start=now,cutoff=now+135,deadline=now+180)
        output.mkdir();sealed=output/'plan.json';sealed.write_text(json.dumps(p,indent=2)+'\n');validate(sealed,output)
        snapshots=output/'code_snapshot';snapshots.mkdir()
        for name in p['new_code_files']:
            f=ROOT/name;(snapshots/f.name).write_bytes(f.read_bytes())
        print(json.dumps(dict(plan_sha=support.sha(sealed),rows=52,cost104=True,cap112=True,optimizer0=True,wall180reserve45=True)));return
    assert not path.exists();oldrun=ROOT/'runs/balanced_fuzzy_cycle_20261112_v65';old=json.loads((oldrun/'plan.json').read_text());p=dict(old)
    rows=json.loads((data/'rows.json').read_text());manifest=json.loads((data/'manifest.json').read_text())
    assert len(rows)==52 and len({r['prompt'] for r in rows})==52 and sum(len(r['prompt'].encode()) for r in rows)==66306
    assert [len(r['prompt'].encode()) for r in rows]==manifest['actual_lengths']
    assert sum(r['language']=='Korean' for r in rows)==16 and sum(r['kind']=='multi_sentence' for r in rows)==12
    for name in manifest['checked_TRAIN_sources']:
        corpus=json.loads(Path(name).read_text(encoding='utf-8-sig'));assert not any(r['prompt'] in old['prompt'] for r in rows for old in corpus)
    proof=dict(status='ROOT_INDEPENDENT_FRESH_V66_DATA_PASS',authored_after_frozen_model=True,rows=52,training_exposure_rows=0,
        unique_literal_prompts52=True,checked_TRAIN_substring_overlap0=True,feature_keys='id,prompt; metadata language/kind/group score only',
        E2_registry_access=0,model_calls=0,limitations=manifest['limitations'])
    data_proof=data/'root_data_review.json';assert not data_proof.exists();data_proof.write_text(json.dumps(proof,indent=2)+'\n')
    names=['scripts/run_fresh_fuzzy_eval_20261112_v66.py','scripts/supervise_fresh_fuzzy_eval_v66.py','scripts/owned_cpu_preflight_v66.py',
        'scripts/prepare_fresh_fuzzy_eval_v66_cpu.py','scripts/prepare_fresh_fuzzy_eval_plan_v66.py']
    p['files']={k:v for k,v in old['files'].items() if k.endswith('.py')}
    parent=oldrun/'model/checkpoints/state_0512.pt';assert support.sha(parent)==PARENT_SHA
    p.update(schema='fdt_fresh_fuzzy_eval_v66',authorized=False,output=str(ROOT/'runs/fresh_fuzzy_eval_20261112_v66'),data=str(data),
        cycle_source=str(parent),cycle_sha=PARENT_SHA,new_code_files=names,attempts=0,epochs=0,wall_seconds=180,start=None,cutoff=None,deadline=None,
        max_calls=112,cost=104,cost_detail=dict(encode=52,readout=52,denoise=0),max_output_bytes=256*1024**2,
        purpose='user requested untrained sentences and explain byte accuracy versus whole reconstruction by length; no optimizer or corpus inclusion',
        exposure=dict(eval_rows=52,eval_target_UTF8_bytes=66306,training_rows=0),checkpoints=[],maximum_additional_STOP_checkpoint=0,
        baseline_contract='new held assessment on frozen V65 only; no repeated old DEV or automatic training',
        limitations=manifest['limitations'])
    for obsolete in ('objective_report','objective','optimizer','baseline_run','cycle_algorithm_sha'):p.pop(obsolete,None)
    files=[ROOT/n for n in names]+[parent,oldrun/'root_saved_review.json',data/'rows.json',data/'manifest.json',data_proof]
    for f in files:p['files'][str(f)]=support.sha(f)
    for name,digest in p['files'].items():
        assert support.sha(name)==digest,name
        if name.endswith('.py'):ast.parse(Path(name).read_text(encoding='utf-8-sig'))
    review.mkdir();path.write_text(json.dumps(p,indent=2)+'\n');validate(path,Path(p['output']))
    print(json.dumps(dict(actual_schema_PASS=True,data_review=proof,cost=104,max_calls=112,pins=len(p['files']))))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');main(parser.parse_args().seal)
