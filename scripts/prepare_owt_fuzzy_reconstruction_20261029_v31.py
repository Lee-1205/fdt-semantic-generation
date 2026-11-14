"""New focused64 allocation from V30; preserve broad corpus, all old runs and budgets."""
import copy,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src'),str(R/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_owt_fuzzy_reconstruction_20261029_v31 import SCHEMA,LIMITS,COST,SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM,MANIFEST

def main():
    old=R/'runs/owt_fuzzy_reconstruction_20261029_v30';out=R/'runs/owt_fuzzy_reconstruction_20261029_v31'
    focused=R/'research/artifacts/owt_focused_reconstruction_20261029_v31_cpu'
    if out.exists():raise FileExistsError('new output only')
    if support.sha(focused/'manifest.json')!=MANIFEST:raise ValueError('immutable focused artifact')
    p=copy.deepcopy(io.load_json(old/'plan.json'));source=old/'model/checkpoints/state_8192.pt'
    if support.sha(source)!=SOURCE_SHA or support.sha(old/'plan.json')!=SOURCE_PLAN:raise ValueError('actual source')
    audit=io.load_json(old/'root_result.json');terminal=io.load_json(old/'supervisor/final_status.json')
    if audit['global_updates']!=100287 or terminal['status']!='COMPLETE' or not terminal['exit']['all_owned_processes_gone']:raise ValueError('source closed/reviewed')
    p.update(schema=SCHEMA,authorized=False,output=str(out),source=str(source),source_sha=SOURCE_SHA,source_plan_sha=SOURCE_PLAN,source_algorithm=SOURCE_ALGORITHM,
        source_global_attempts=100324,source_global_updates=100287,limits=LIMITS,cost_contract=COST,epochs=256,
        focused_artifact=str(focused),focused_manifest_sha=MANIFEST,allocation_start=None,deadline=None,cutoff=None,
        training_exposure={'unique_training_rows':64,'new_rows':16384,'target_UTF8_bytes':1101824,'finite_epochs':256},
        import_policy='All81 populated Adam/3groups/model/scaler/allRNG/frozen scope and complete ancestry exact from actual V30; no fresh moments/parameters/unfreezing.',
        reference_policy='NEW focused64 training subset baseline/final, no extra initial broad held baseline; held final only; previous broad source results retained.',
        interpretation='Finite concentrated fit of64existing training spans, includes memorization; same fuzzy input meaning to same input reconstruction; not blind LM or independent reasoning proof.')
    names=('scripts/build_owt_v31_adapters.py','scripts/owt_fuzzy_engine_v31.py','scripts/run_owt_fuzzy_reconstruction_20261029_v31.py',
        'scripts/supervise_owt_fuzzy_reconstruction_20261029_v31.py','scripts/prepare_owt_fuzzy_reconstruction_20261029_v31.py',
        'tests/test_owt_focused_reconstruction_v31.py','scripts/prepare_owt_focused_reconstruction_v31_cpu.py')
    files=[R/n for n in names]+[source,old/'plan.json',old/'root_result.json',old/'result.json',old/'supervisor/final_status.json']
    files+=[focused/n for n in ('manifest.json','features.json','targets.json','orders.json')]
    p['files'].update({str(f.resolve()):support.sha(f) for f in files})
    out.mkdir();io.atomic_json(out/'plan.json',p);snap=out/'code_snapshot';snap.mkdir()
    for filename in p['files']:
        f=Path(filename)
        if f.suffix=='.py' and f.parent.name in ('scripts','research','tests'):(snap/f.name).write_bytes(f.read_bytes())
    print({'draft_sha256':support.sha(out/'plan.json'),'attempts':2048,'rows':16384,'calls_max':12000})
if __name__=='__main__':main()
