"""Prepare NEW two-epoch continuation after persistent AMP guard correction."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from run_owt_fuzzy_reconstruction_20261028_v17 import SCHEMA,LIMITS,SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM,COST
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    a=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v15';parent=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v13'
    baseline=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v15';source_run=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v16';source=source_run/'emergency/checkpoints/state_6867.pt'
    output=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v17'
    if output.exists():raise FileExistsError('new output only')
    names=('research/owt_fuzzy_reconstruction_v12.py','scripts/owt_fuzzy_engine_v12.py','scripts/owt_fuzzy_engine_v13.py',
        'scripts/owt_fuzzy_engine_v14.py','scripts/owt_fuzzy_engine_v15.py','scripts/owt_fuzzy_engine_v17_base.py','scripts/owt_fuzzy_engine_v17.py','scripts/run_owt_fuzzy_reconstruction_20261028_v12.py',
        'scripts/run_owt_fuzzy_reconstruction_20261028_v17.py','scripts/supervise_owt_fuzzy_reconstruction_20261028_v17.py',
        'scripts/prepare_owt_fuzzy_reconstruction_20261028_v17.py','tests/test_owt_fuzzy_continuation_v17.py',
        'research/text_diffusion_v1.py','research/text_context_decoder_v2.py','scripts/text_generation_support_v1.py',
        'scripts/text_language_runtime_support_v6.py','scripts/learning_io_v2.py','scripts/supervise_querycount_partial9_20261027_v3.py')
    files=[ROOT/n for n in names]+list(sorted((ROOT/'src/fdt_local').glob('*.py')))
    files+=list(sorted(p for p in a.rglob('*') if p.is_file()))+list(sorted(p for p in parent.rglob('*') if p.is_file()))
    files+=[source,source_run/'plan.json',source_run/'incident.json',source_run/'root_stop_review.json',baseline/'result.json']+[baseline/('final_'+n) for n in ('metrics.json','logits.pt','generation.json')]
    if support.sha(source)!=SOURCE_SHA or support.sha(source_run/'plan.json')!=SOURCE_PLAN:raise ValueError('source identity')
    previous=io.load_json(source_run/'root_stop_review.json')
    if (previous['global_attempts'],previous['global_updates'])!=(19155,19152):raise ValueError('source cumulative counters')
    m=io.load_json(a/'manifest.json')
    plan={'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),'artifact':str(a),'parent_artifact':str(parent),
        'source':str(source),'source_sha':SOURCE_SHA,'source_plan_sha':SOURCE_PLAN,'source_algorithm':SOURCE_ALGORITHM,
        'source_global_attempts':19155,'source_global_updates':19152,'baseline_run':str(baseline),'limits':LIMITS,
        'files':{str(p.resolve()):support.sha(p) for p in files},'epochs':2,'cost_contract':COST,'generation_indices':[4*i for i in range(16)],
        'reasoning_diffusion_training':False,'E2_final_accessed':False,'allocation_start':None,'deadline':None,'cutoff':None,
        'training_exposure':{'new_rows':16384,'target_UTF8_bytes':2*m['sentence_UTF8_bytes']['train'],
            'old_document_rows':8192,'new_document_rows':8192,'old_target_UTF8_bytes':2*m['old_train_sentence_UTF8_bytes'],
            'new_target_UTF8_bytes':2*m['new_train_sentence_UTF8_bytes']},
        'fullstate_policy':['import0','mid1024','final2048','emergency_if_needed'],
        'latest_human_instruction':'Continue original fuzzy-anchor sentence -> extracted meaning -> reconstruction on OpenWebText, reasoning separate later. ROOT only.',
        'import_policy':'Actual V16 emergency schema; exact all63Adam/scaler/RNG/model and recursive ancestry retained; new2048attempt finite clock/order, no reset of source allocation.',
        'reference_policy':'V15 last actual same256held quality reference; comparison includes6864V16source updates and newV17 updates; no extra initial GPU forward.',
        'amp_stop_policy':{'consecutive':3,'window':16,'in_window':3,'total':8},'automatic_restart':False,'interpretation':'Same2048documents and original fuzzy-AE; two NEW finite epochs after fixing cumulative AMP guard, source FAILED record and all useful updates preserved; old1325unattempted updates and unusedcalls/time not reused.'}
    output.mkdir();io.atomic_json(output/'plan.json',plan);snap=output/'code_snapshot';snap.mkdir()
    for p in files:
        if p.suffix=='.py' and p.parent.name in ('scripts','research','tests'):(snap/p.name).write_bytes(p.read_bytes())
    print(json.dumps({'draft_plan_sha256':support.sha(output/'plan.json'),'new_training_exposure':plan['training_exposure'],'max_calls':COST['total_max']}))
if __name__=='__main__':main()
