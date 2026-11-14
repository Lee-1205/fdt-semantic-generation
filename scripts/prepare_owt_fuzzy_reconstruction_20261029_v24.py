"""New finite allocation from the actual V23 partial; no corpus or model change."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
import text_generation_support_v1 as support
from run_owt_fuzzy_reconstruction_20261029_v24 import SCHEMA,SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM
from owt_resource_budget_v24 import RESOURCE_POLICY

def main():
    old=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v23_final'
    baseline=ROOT/'runs/owt_partial_assessment_20261028_v23_readonly'
    output=ROOT/'runs/owt_fuzzy_reconstruction_20261029_v24'
    if output.exists():raise FileExistsError('new output only')
    p=copy.deepcopy(io.load_json(old/'plan.json'))
    source=old/'emergency/checkpoints/state_7440.pt'
    if support.sha(source)!=SOURCE_SHA or support.sha(old/'plan.json')!=SOURCE_PLAN:raise ValueError('actual partial identity')
    audit=io.load_json(old/'root_partial_state_review.json');r=io.load_json(baseline/'result.json')
    if audit['global_updates']!=47061 or audit['new_updates']!=7437 or r['source_sha256']!=SOURCE_SHA:raise ValueError('actual partial/reference')
    p.update(schema=SCHEMA,authorized=False,output=str(output),source=str(source),source_sha=SOURCE_SHA,
        source_plan_sha=SOURCE_PLAN,source_algorithm=SOURCE_ALGORITHM,source_global_attempts=47076,source_global_updates=47061,
        baseline_run=str(baseline),allocation_start=None,deadline=None,cutoff=None,resource_policy=RESOURCE_POLICY,
        external_closed_assessment_calls={'run':str(baseline),'plan_sha256':support.sha(baseline/'plan.json'),'calls':r['actual_calls'],'total':1936,'included_in_source_ledger':False},
        import_policy='Actual V23 partial7440/7437/cursor59520; all populated Adam/scaler/RNG/full ancestry preserved. New8192attempt finite allocation, no old752attempt-time-call reuse.',
        reference_policy='Changed partial assessment at47061updates, stored same256held and fixed16train. No initial GPU baseline repeat.',
        interpretation='Same corpus/model/fuzzy/AMP/loss/inference; only expensive filesystem scans bounded at1second and forced before large writes. Immediate STOP-clock-callcap and independent1second OS supervision retained. CPU timing is not a GPU gain claim.')
    names=('scripts/owt_resource_budget_v24.py','scripts/owt_fuzzy_engine_v24.py','scripts/run_owt_fuzzy_reconstruction_20261029_v24.py',
        'scripts/supervise_owt_fuzzy_reconstruction_20261029_v24.py','scripts/prepare_owt_fuzzy_reconstruction_20261029_v24.py','tests/test_owt_resource_continuation_v24.py')
    files=[ROOT/n for n in names]+[source,old/'plan.json',old/'root_partial_state_review.json',old/'supervisor/final_status.json',old/'incident.json']
    files += [baseline/n for n in ('plan.json','result.json','root_result.json','final_metrics.json','final_logits.pt','final_generation.json','final_train_reconstruction.json')]
    p['files'].update({str(f.resolve()):support.sha(f) for f in files})
    output.mkdir();io.atomic_json(output/'plan.json',p);snap=output/'code_snapshot';snap.mkdir()
    for name in p['files']:
        f=Path(name)
        if f.suffix=='.py' and f.parent.name in ('scripts','research','tests'):(snap/f.name).write_bytes(f.read_bytes())
    print(json.dumps({'draft_sha256':support.sha(output/'plan.json'),'source':SOURCE_SHA,'max_calls':p['limits']['max_calls'],'resource_policy':RESOURCE_POLICY}))
if __name__=='__main__':main()
