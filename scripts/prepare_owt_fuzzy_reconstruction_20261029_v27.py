"""New finite allocation from the actual V26 fullstate; training prefix curriculum, no architecture/corpus change."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_owt_fuzzy_reconstruction_20261029_v27 import SCHEMA,SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM
from owt_resource_budget_v24 import RESOURCE_POLICY
from research.owt_prefix_curriculum_v27 import OBJECTIVE

def main():
    old=ROOT/'runs/owt_fuzzy_reconstruction_20261029_v26'
    baseline=old
    output=ROOT/'runs/owt_fuzzy_reconstruction_20261029_v27'
    if output.exists():raise FileExistsError('new output only')
    p=copy.deepcopy(io.load_json(old/'plan.json'))
    source=old/'model/checkpoints/state_8192.pt'
    if support.sha(source)!=SOURCE_SHA or support.sha(old/'plan.json')!=SOURCE_PLAN:raise ValueError('actual partial identity')
    audit=io.load_json(old/'root_result.json');r=io.load_json(baseline/'result.json')
    if audit['global_updates']!=71627 or audit['new_updates']!=8189 or r['latest']['sha256']!=SOURCE_SHA:raise ValueError('actual partial/reference')
    p.update(schema=SCHEMA,authorized=False,output=str(output),source=str(source),source_sha=SOURCE_SHA,
        source_plan_sha=SOURCE_PLAN,source_algorithm=SOURCE_ALGORITHM,source_global_attempts=71652,source_global_updates=71627,
        baseline_run=str(baseline),allocation_start=None,deadline=None,cutoff=None,resource_policy=RESOURCE_POLICY,training_objective=OBJECTIVE,
        external_closed_assessment_calls=copy.deepcopy(p['external_closed_assessment_calls']),
        import_policy='Actual V26 full8192/8189/cursor65536; all populated Adam/scaler/RNG/full ancestry preserved. New8192attempt finite allocation, no old752attempt-time-call reuse.',
        reference_policy='V26 saved assessment at71627updates, stored same256held and fixed16train. No initial GPU baseline repeat.',
        interpretation='Single training-only past-byte mask curriculum from.15at1to.50at4096thenhold8192. Same fuzzy model, clean0.5+corrupt0.5 weights, deterministic reduction, inference/corpus/resource guards. Compare saved V26, no initial GPU repeat. Different corruption losses are not directly comparable.')
    names=('scripts/owt_resource_budget_v24.py','research/owt_prefix_curriculum_v27.py','scripts/owt_fuzzy_engine_v27.py','scripts/run_owt_fuzzy_reconstruction_20261029_v27.py',
        'scripts/supervise_owt_fuzzy_reconstruction_20261029_v27.py','scripts/prepare_owt_fuzzy_reconstruction_20261029_v27.py','tests/test_owt_prefix_curriculum_v27.py')
    files=[ROOT/n for n in names]+[source,old/'plan.json',old/'root_result.json',old/'supervisor/final_status.json']
    files += [baseline/n for n in ('plan.json','result.json','root_result.json','final_metrics.json','final_logits.pt','final_generation.json','final_train_reconstruction.json')]
    p['files'].update({str(f.resolve()):support.sha(f) for f in files})
    output.mkdir();io.atomic_json(output/'plan.json',p);snap=output/'code_snapshot';snap.mkdir()
    for name in p['files']:
        f=Path(name)
        if f.suffix=='.py' and f.parent.name in ('scripts','research','tests'):(snap/f.name).write_bytes(f.read_bytes())
    print(json.dumps({'draft_sha256':support.sha(output/'plan.json'),'source':SOURCE_SHA,'max_calls':p['limits']['max_calls'],'resource_policy':RESOURCE_POLICY}))
if __name__=='__main__':main()
