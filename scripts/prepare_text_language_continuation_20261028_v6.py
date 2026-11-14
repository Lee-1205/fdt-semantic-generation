"""New bounded explicit human-continued allocation, not reopening old terminal."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
from text_generation_support_v1 import sha
from run_text_language_continuation_20261028_v6 import SCHEMA,LIMITS,SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM,MANIFEST_SHA,generation_indices


def make_plan(output):
    source_run=ROOT/'runs/text_language_continuation_20261028_v5_deterministic_v2'
    source=source_run/'emergency/checkpoints/state_0149.pt'
    artifact=ROOT/'research/artifacts/text_story_continuation_20261028_v5_cpu'
    files=[ROOT/p for p in ('research/text_language_continuation_v6.py','research/text_diffusion_v1.py',
        'research/text_context_decoder_v2.py','research/text_corpus_contract_v5.py',
        'research/text_language_continuation_v5_deterministic_v2.py','scripts/text_language_engine_v6.py',
        'scripts/text_language_runtime_support_v6.py','scripts/text_language_engine_v5.py',
        'scripts/run_text_language_continuation_20261028_deterministic_v2.py',
        'scripts/run_text_language_continuation_20261028_v6.py','scripts/supervise_text_language_continuation_20261028_v6.py',
        'scripts/prepare_text_language_continuation_20261028_v6.py','scripts/text_generation_support_v1.py',
        'scripts/learning_io_v2.py','scripts/supervise_querycount_partial9_20261027_v3.py',
        'tests/test_text_language_runtime_v6.py')]
    files+=list(sorted((ROOT/'src/fdt_local').glob('*.py')))+list(sorted(artifact.glob('*.json')))
    files+=[source,source_run/'plan.json']+[source_run/n for n in ('baseline_NLL.json','baseline_NLL_logits.pt','baseline_generation.json')]
    return {'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),
        'source':str(source),'source_sha':SOURCE_SHA,'source_plan_sha':SOURCE_PLAN,'source_algorithm':SOURCE_ALGORITHM,
        'source_attempts':149,'global_attempt_cap':409,'epochs':1,'artifact':str(artifact),'manifest_sha':MANIFEST_SHA,
        'baseline_run':str(source_run),'generation_indices':generation_indices(io.load_json(artifact/'targets.json')['dev']),
        'files':{str(p):sha(p) for p in files},'limits':LIMITS,'cost_contract':{'training':9880,'NLL':1998,'generation':1100,'total_max':12978},
        'allocation_start':None,'deadline':None,'cutoff':None,'E2_final_accessed':False,
        'continuation_contract':'Preserve model/populatedAdam/scaler/allRNG/history/cursor1192; first new global150; no retry of failed149; new independent600sec/14000call cap.',
        'prior_gradient_observation':'Old148 updates reached language/decoder and some VP parameters; AMP no-grad cast cache left70 VP matrix parameters without Adam states. Preserve this history; fix cache before next update.',
        'changes':['vanished atomic temp accounting race only','no-grad rollout autocast cache disabled','require every trainable parameter gradient'],
        'automatic_restart':False,'latest_human_instruction':'그리고 학습계속해'}


def create():
    output=ROOT/'runs/text_language_continuation_20261028_v6'
    if output.exists():raise FileExistsError('never overwrite new allocation')
    plan=make_plan(output)
    if sha(plan['source'])!=SOURCE_SHA:raise ValueError('source SHA')
    output.mkdir();io.atomic_json(output/'plan.json',plan);snapshot=output/'code_snapshot';snapshot.mkdir()
    for path in plan['files']:
        p=Path(path)
        if p.suffix=='.py' and p.parent.name in ('scripts','research','tests'):(snapshot/p.name).write_bytes(p.read_bytes())
    print(json.dumps({'output':str(output),'draft_plan_sha256':sha(output/'plan.json'),'authorized':False}))
if __name__=='__main__':create()
