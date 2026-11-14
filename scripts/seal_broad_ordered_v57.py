"""Seal a new broad document epoch after actual owned fullsize CPU restore."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import text_generation_support_v1 as support
from run_broad_ordered_20261111_v57 import validate

def main():
    artifact=ROOT/'research/artifacts/broad_ordered_20261111_v57_cpu';unsealed=artifact/'plan_unsealed.json'
    p=json.loads(unsealed.read_text(encoding='utf-8-sig'));output=Path(p['output']);validate(unsealed,output)
    if output.exists():raise FileExistsError('old terminal budget cannot reopen')
    proof=artifact/'cpu_preflight_owned/final_status.json';review=json.loads(proof.read_text(encoding='utf-8-sig'))
    if (review['status']!='CPU_PREFLIGHT_PASS' or not review['exit']['all_owned_processes_gone'] or review['owner_imported_Torch']
        or not review['report']['all8_V56_Adam_scaler_RNG_exact'] or not review['report']['fullsize_engine_restore']):raise ValueError('actual ownership proof')
    p['files'][str(proof)]=support.sha(proof);p['files'][str(Path(__file__).resolve())]=support.sha(__file__)
    now=time.time();p.update(authorized=True,start=now,cutoff=now+3000,deadline=now+3300)
    output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');validate(path,output)
    snapshot=output/'code_snapshot';snapshot.mkdir()
    for name in ('broad_ordered_engine_v57.py','run_broad_ordered_20261111_v57.py','supervise_broad_ordered_v57.py','seal_broad_ordered_v57.py'):
        source=ROOT/'scripts'/name;(snapshot/name).write_bytes(source.read_bytes())
    proof=dict(plan_sha=support.sha(path),cost=16416,new_budget=True,source_frozen=True,active8Adam_preserved=True)
    (output/'root_prelaunch_review.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof))
if __name__=='__main__':main()
