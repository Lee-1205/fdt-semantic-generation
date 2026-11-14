"""New finite allocation from the actual V28 fullstate; training prefix p50 hold, no architecture/corpus change."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_owt_fuzzy_reconstruction_20261029_v29 import SCHEMA,SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM,LIMITS,COST
from owt_resource_budget_v29 import RESOURCE_POLICY
from research.owt_extraction_tail_v29 import OBJECTIVE,TRAINABILITY
from owt_fuzzy_engine_v29 import OWNERSHIP

def main():
    old=ROOT/'runs/owt_fuzzy_reconstruction_20261029_v28'
    baseline=old
    output=ROOT/'runs/owt_fuzzy_reconstruction_20261029_v29'
    if output.exists():raise FileExistsError('new output only')
    p=copy.deepcopy(io.load_json(old/'plan.json'))
    source=old/'model/checkpoints/state_8192.pt'
    if support.sha(source)!=SOURCE_SHA or support.sha(old/'plan.json')!=SOURCE_PLAN:raise ValueError('actual partial identity')
    audit=io.load_json(old/'root_result.json');r=io.load_json(baseline/'result.json')
    if audit['global_updates']!=88005 or audit['new_updates']!=8188 or r['latest']['sha256']!=SOURCE_SHA:raise ValueError('actual partial/reference')
    p.update(schema=SCHEMA,authorized=False,output=str(output),source=str(source),source_sha=SOURCE_SHA,
        source_plan_sha=SOURCE_PLAN,source_algorithm=SOURCE_ALGORITHM,source_global_attempts=88036,source_global_updates=88005,
        limits=LIMITS,cost_contract=COST,epochs=1,trainability=TRAINABILITY,ownership_policy=OWNERSHIP,baseline_run=str(baseline),allocation_start=None,deadline=None,cutoff=None,resource_policy=RESOURCE_POLICY,training_objective=OBJECTIVE,
        external_closed_assessment_calls=copy.deepcopy(p['external_closed_assessment_calls']),
        import_policy='Actual V28 fullstate model/allRNG/scaler and old63 Adam subcomponents exact; append18 existing tensors with empty fresh Adam group; source oldfrozen and phase frozen scopes separately retained.',
        reference_policy='Saved V28 assessment at88005updates; same held256 and fixedtrain16, no initial GPU baseline repeat.',
        interpretation='One finite epoch; thaw existing language blocks5-6 at3e-6, original fuzzy mechanism/decoder/inference/p50 unchanged. No output-meaning reasoning training; no old budget reuse.')
    p['training_exposure']={k:v//2 for k,v in p['training_exposure'].items()}
    names=('scripts/build_owt_v29_driver.py','scripts/owt_resource_budget_v29.py','research/owt_extraction_tail_v29.py','scripts/owt_fuzzy_engine_v29.py','scripts/run_owt_fuzzy_reconstruction_20261029_v29.py',
        'scripts/supervise_owt_fuzzy_reconstruction_20261029_v29.py','scripts/prepare_owt_fuzzy_reconstruction_20261029_v29.py','tests/test_owt_extraction_tail_runtime_v29.py')
    files=[ROOT/n for n in names]+[source,old/'plan.json',old/'root_result.json',old/'supervisor/final_status.json']
    files += [baseline/n for n in ('plan.json','result.json','root_result.json','final_metrics.json','final_logits.pt','final_generation.json','final_train_reconstruction.json')]
    p['files'].update({str(f.resolve()):support.sha(f) for f in files})
    output.mkdir();io.atomic_json(output/'plan.json',p);snap=output/'code_snapshot';snap.mkdir()
    for name in p['files']:
        f=Path(name)
        if f.suffix=='.py' and f.parent.name in ('scripts','research','tests'):(snap/f.name).write_bytes(f.read_bytes())
    print(json.dumps({'draft_sha256':support.sha(output/'plan.json'),'source':SOURCE_SHA,'max_calls':p['limits']['max_calls'],'resource_policy':RESOURCE_POLICY}))
if __name__=='__main__':main()
