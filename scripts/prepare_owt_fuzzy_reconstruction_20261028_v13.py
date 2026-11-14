"""Draft the new fixed expanded-corpus continuation; all older allocations closed."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from run_owt_fuzzy_reconstruction_20261028_v13 import SCHEMA,LIMITS,SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM,COST
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    artifact=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v13';parent=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v12'
    baseline=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v12';source=baseline/'model/checkpoints/state_2048.pt'
    output=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v13'
    if output.exists():raise FileExistsError('new output only')
    names=('research/owt_fuzzy_reconstruction_v12.py','scripts/owt_fuzzy_engine_v12.py','scripts/owt_fuzzy_engine_v13.py',
        'scripts/run_owt_fuzzy_reconstruction_20261028_v12.py','scripts/run_owt_fuzzy_reconstruction_20261028_v13.py',
        'scripts/supervise_owt_fuzzy_reconstruction_20261028_v13.py','scripts/prepare_owt_expansion_v13.py',
        'scripts/prepare_owt_fuzzy_reconstruction_20261028_v13.py','tests/test_owt_fuzzy_continuation_v13.py',
        'research/text_diffusion_v1.py','research/text_context_decoder_v2.py','scripts/text_generation_support_v1.py',
        'scripts/text_language_runtime_support_v6.py','scripts/learning_io_v2.py','scripts/supervise_querycount_partial9_20261027_v3.py')
    files=[ROOT/n for n in names]+list(sorted((ROOT/'src/fdt_local').glob('*.py')))
    files+=list(sorted(p for p in artifact.rglob('*') if p.is_file()))+list(sorted(p for p in parent.rglob('*') if p.is_file()))
    files+=[source,baseline/'plan.json']+[baseline/('final_'+n) for n in ('metrics.json','logits.pt','generation.json')]
    if support.sha(source)!=SOURCE_SHA or support.sha(baseline/'plan.json')!=SOURCE_PLAN:raise ValueError('source changed')
    m=io.load_json(artifact/'manifest.json')
    plan={'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),'artifact':str(artifact),'parent_artifact':str(parent),
        'source':str(source),'source_sha':SOURCE_SHA,'source_plan_sha':SOURCE_PLAN,'source_algorithm':SOURCE_ALGORITHM,'baseline_run':str(baseline),
        'limits':LIMITS,'files':{str(p.resolve()):support.sha(p) for p in files},'epochs':4,'cost_contract':COST,'generation_indices':[4*i for i in range(16)],
        'reasoning_diffusion_training':False,'E2_final_accessed':False,'allocation_start':None,'deadline':None,'cutoff':None,
        'training_exposure':{'new_rows':16384,'target_UTF8_bytes':4*m['sentence_UTF8_bytes']['train'],
            'old_document_rows':8192,'new_document_rows':8192,'old_target_UTF8_bytes':4*m['old_train_sentence_UTF8_bytes'],'new_target_UTF8_bytes':4*m['new_train_sentence_UTF8_bytes']},
        'latest_human_instruction':'Continue original fuzzy anchors on OpenWebText sentence -> extracted meaning -> reconstruction first; reasoning/diffusion separate later. ROOT only.',
        'import_policy':'Exact V12 model/63Adam/scaler/allRNG import; new2048attempt allocation/four finite epochs211-214, no source clock/calls reuse.',
        'reference_policy':'Source V12 final held metrics/logits/generation reused; architecture and64helddocuments/256sentences/controlconditions unchanged, no extra GPU baseline.',
        'automatic_restart':False,'interpretation':'Expand training diversity512->1024documents with fixed observed held; same fuzzy-AE, no reset/penalty/reasoning change.'}
    output.mkdir();io.atomic_json(output/'plan.json',plan);snap=output/'code_snapshot';snap.mkdir()
    for p in files:
        if p.suffix=='.py' and p.parent.name in ('scripts','research','tests'):(snap/p.name).write_bytes(p.read_bytes())
    print(json.dumps({'draft_plan_sha256':support.sha(output/'plan.json'),'new_target_UTF8_exposure':plan['training_exposure'],'maximum_calls':COST['total_max']}))
if __name__=='__main__':main()
