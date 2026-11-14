"""Root seals a new fixed budget after actual worker CPU/OS ownership review."""
import json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from run_ordered_fuzzy_20261111_v55 import validate
import text_generation_support_v1 as support

def main():
    artifact=ROOT/'research/artifacts/ordered_fuzzy_20261111_v55_cpu'
    p=json.loads((artifact/'plan_unsealed.json').read_text());output=Path(p['output'])
    if output.exists():raise FileExistsError('old run must not be reopened')
    proof=artifact/'cpu_preflight_owned/final_status.json';review=json.loads(proof.read_text())
    if (review['status']!='CPU_PREFLIGHT_PASS' or not review['exit']['all_owned_processes_gone']
        or review['owner_imported_Torch'] or review['report']['encoder_trainable_parameters']!=0
        or review['report']['parent_Adam_used_for_training'] or not review['report']['fullsize_engine_restore']):raise ValueError('actual CPU ownership evidence')
    validate(artifact/'plan_unsealed.json',output)
    if sum(p['cost'][k] for k in ('encode','train_readout','normal_and_role_reversal_readout','bank_readout'))!=4208:raise ValueError('actual call count')
    p['files'][str(proof)]=support.sha(proof);p['files'][str(Path(__file__).resolve())]=support.sha(__file__)
    start=time.time();p.update(authorized=True,start=start,cutoff=start+780,deadline=start+900)
    output.mkdir();path=output/'plan.json';path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    validate(path,output)
    snapshot=output/'code_snapshot';snapshot.mkdir()
    for source in [ROOT/'research/ordered_fuzzy_readout_v55.py',ROOT/'scripts/ordered_fuzzy_engine_v55.py',
        ROOT/'scripts/run_ordered_fuzzy_20261111_v55.py',ROOT/'scripts/supervise_ordered_fuzzy_v55.py',Path(__file__)]:
        (snapshot/source.name).write_bytes(source.read_bytes())
    result=dict(plan_sha=support.sha(path),actual_CPU_fullsize_reference_restore=True,parent_frozen=True,
        cost=p['cost'],source_Adam_required_in_parent=True,new_readout_Adam_fresh=True,head_checkpoints_not_standalone=True)
    (output/'root_prelaunch_review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
if __name__=='__main__':main()
