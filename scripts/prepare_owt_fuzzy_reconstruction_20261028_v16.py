"""Prepare eight new finite epochs on the unchanged broadened OWT pool."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from run_owt_fuzzy_reconstruction_20261028_v16 import SCHEMA,LIMITS,SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM,COST
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    a=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v15';parent=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v13'
    baseline=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v15';source=baseline/'model/checkpoints/state_4096.pt'
    output=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v16'
    if output.exists():raise FileExistsError('new output only')
    names=('research/owt_fuzzy_reconstruction_v12.py','scripts/owt_fuzzy_engine_v12.py','scripts/owt_fuzzy_engine_v13.py',
        'scripts/owt_fuzzy_engine_v14.py','scripts/owt_fuzzy_engine_v15.py','scripts/owt_fuzzy_engine_v16.py','scripts/run_owt_fuzzy_reconstruction_20261028_v12.py',
        'scripts/run_owt_fuzzy_reconstruction_20261028_v16.py','scripts/supervise_owt_fuzzy_reconstruction_20261028_v16.py',
        'scripts/prepare_owt_fuzzy_reconstruction_20261028_v16.py','tests/test_owt_fuzzy_continuation_v16.py',
        'research/text_diffusion_v1.py','research/text_context_decoder_v2.py','scripts/text_generation_support_v1.py',
        'scripts/text_language_runtime_support_v6.py','scripts/learning_io_v2.py','scripts/supervise_querycount_partial9_20261027_v3.py')
    files=[ROOT/n for n in names]+list(sorted((ROOT/'src/fdt_local').glob('*.py')))
    files+=list(sorted(p for p in a.rglob('*') if p.is_file()))+list(sorted(p for p in parent.rglob('*') if p.is_file()))
    files+=[source,baseline/'plan.json',baseline/'result.json']+[baseline/('final_'+n) for n in ('metrics.json','logits.pt','generation.json')]
    if support.sha(source)!=SOURCE_SHA or support.sha(baseline/'plan.json')!=SOURCE_PLAN:raise ValueError('source identity')
    previous=io.load_json(baseline/'result.json')
    if (previous['global_attempts'],previous['global_updates'])!=(12288,12288):raise ValueError('source cumulative counters')
    m=io.load_json(a/'manifest.json')
    plan={'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),'artifact':str(a),'parent_artifact':str(parent),
        'source':str(source),'source_sha':SOURCE_SHA,'source_plan_sha':SOURCE_PLAN,'source_algorithm':SOURCE_ALGORITHM,
        'source_global_attempts':12288,'source_global_updates':12288,'baseline_run':str(baseline),'limits':LIMITS,
        'files':{str(p.resolve()):support.sha(p) for p in files},'epochs':8,'cost_contract':COST,'generation_indices':[4*i for i in range(16)],
        'reasoning_diffusion_training':False,'E2_final_accessed':False,'allocation_start':None,'deadline':None,'cutoff':None,
        'training_exposure':{'new_rows':65536,'target_UTF8_bytes':8*m['sentence_UTF8_bytes']['train'],
            'old_document_rows':32768,'new_document_rows':32768,'old_target_UTF8_bytes':8*m['old_train_sentence_UTF8_bytes'],
            'new_target_UTF8_bytes':8*m['new_train_sentence_UTF8_bytes']},
        'fullstate_policy':['import0','mid4096','final8192','emergency_if_needed'],
        'latest_human_instruction':'Continue original fuzzy-anchor sentence -> extracted meaning -> reconstruction on OpenWebText, reasoning separate later. ROOT only.',
        'import_policy':'Actual V15 schema; exact all63Adam/scaler/RNG/model and recursive ancestry retained; new8192attempt finite clock/order, no reset of source allocation.',
        'reference_policy':'V15 final stored same256held/samearchitecture/samecontrols reference; no additional baseline GPU forward.',
        'automatic_restart':False,'interpretation':'Same2048training documents and original fuzzy-AE; eight new finite epochs increase learning dose without corpus/architecture/penalty/reasoning changes.'}
    output.mkdir();io.atomic_json(output/'plan.json',plan);snap=output/'code_snapshot';snap.mkdir()
    for p in files:
        if p.suffix=='.py' and p.parent.name in ('scripts','research','tests'):(snap/p.name).write_bytes(p.read_bytes())
    print(json.dumps({'draft_plan_sha256':support.sha(output/'plan.json'),'new_training_exposure':plan['training_exposure'],'max_calls':COST['total_max']}))
if __name__=='__main__':main()
