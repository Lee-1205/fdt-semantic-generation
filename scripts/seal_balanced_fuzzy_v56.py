"""Seal after actual fullsize CPU restore and independent owned-OS validation."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_balanced_fuzzy_20261111_v56 import validate

def main():
    artifact=ROOT/'research/artifacts/balanced_fuzzy_20261111_v56_cpu';unsealed=artifact/'plan_unsealed.json'
    p=json.loads(unsealed.read_text(encoding='utf-8-sig'));output=Path(p['output']);validate(unsealed,output)
    if output.exists():raise FileExistsError('old budget cannot reopen')
    proof=artifact/'cpu_preflight_owned/final_status.json';review=json.loads(proof.read_text(encoding='utf-8-sig'))
    if (review['status']!='CPU_PREFLIGHT_PASS' or not review['exit']['all_owned_processes_gone'] or review['owner_imported_Torch']
        or not review['report']['all8_V55_Adam_scaler_RNG_exact'] or not review['report']['fullsize_engine_restore']):raise ValueError('owned CPU proof')
    p['files'][str(proof)]=support.sha(proof);p['files'][str(Path(__file__).resolve())]=support.sha(__file__)
    now=time.time();p.update(authorized=True,start=now,cutoff=now+780,deadline=now+900)
    output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');validate(path,output)
    snapshots=output/'code_snapshot';snapshots.mkdir()
    for file in ('scripts/balanced_fuzzy_engine_v56.py','scripts/run_balanced_fuzzy_20261111_v56.py',
        'scripts/supervise_balanced_fuzzy_v56.py','research/codec_commit_stop_v56.py','scripts/seal_balanced_fuzzy_v56.py'):
        source=ROOT/file;(snapshots/source.name).write_bytes(source.read_bytes())
    proof=dict(plan_sha=support.sha(path),cost=p['cost'],active8Adam_preserved=True,source_frozen=True,new_budget=True)
    (output/'root_prelaunch_review.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
if __name__=='__main__':main()
