"""Prospective two-epoch allocation after the user's explicit continue-learning instruction."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
from text_generation_support_v1 import sha
from run_text_language_continuation_20261028_v7 import SCHEMA,LIMITS,SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM,MANIFEST_SHA,generation_indices

CURRICULUM={'epochs':2,'seeds':[62,63],'training_eos_CE_weight':4.0,'base_language_LR':5e-6,
    'base_other_LR':1.5e-5,'scheduler':'cosine0.1floor_new818attempts','quality_metrics_stop_training':False,
    'safety_stops_retained':True,'new_row_exposures':6542,'new_target_bytes':769350,'new_EOF':1024}

def make_plan(output):
    parent=ROOT/'runs/text_language_continuation_20261028_v6'
    artifact=ROOT/'research/artifacts/text_story_continuation_20261028_v5_cpu'
    source=parent/'model/checkpoints/state_0409.pt'
    names=('research/text_language_continuation_v7.py','research/text_language_continuation_v6.py',
        'research/text_language_continuation_v5_deterministic_v2.py','research/text_diffusion_v1.py',
        'research/text_context_decoder_v2.py','research/text_corpus_contract_v5.py',
        'scripts/text_language_engine_v7.py','scripts/text_language_engine_v6.py','scripts/text_language_engine_v5.py',
        'scripts/text_language_runtime_support_v6.py','scripts/run_text_language_continuation_20261028_deterministic_v2.py',
        'scripts/run_text_language_continuation_20261028_v7.py','scripts/supervise_text_language_continuation_20261028_v7.py',
        'scripts/prepare_text_language_continuation_20261028_v7.py','scripts/text_generation_support_v1.py',
        'scripts/learning_io_v2.py','scripts/supervise_querycount_partial9_20261027_v3.py','tests/test_text_language_runtime_v7.py')
    files=[ROOT/n for n in names]+list(sorted((ROOT/'src/fdt_local').glob('*.py')))+list(sorted(artifact.glob('*.json')))
    files+=[source,parent/'plan.json']+[parent/n for n in ('final_NLL.json','final_NLL_logits.pt','final_generation.json')]
    return {'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),
        'source':str(source),'source_sha':SOURCE_SHA,'source_plan_sha':SOURCE_PLAN,'source_algorithm':SOURCE_ALGORITHM,
        'source_attempts':409,'source_updates':408,'global_attempt_cap':1227,'epochs':2,'artifact':str(artifact),
        'manifest_sha':MANIFEST_SHA,'baseline_run':str(parent),'curriculum':CURRICULUM,
        'generation_indices':generation_indices(io.load_json(artifact/'targets.json')['dev']),
        'files':{str(p):sha(p) for p in files},'limits':LIMITS,
        'cost_contract':{'training':31084,'NLL':1998,'generation':1100,'total_max':34182},
        'allocation_start':None,'deadline':None,'cutoff':None,'E2_final_accessed':False,
        'automatic_restart':False,'latest_human_instruction':'절대 학습량이 아직 모자라니 검증에 너무 목매달지 말고 문제 해결되면 계속 학습해.',
        'interpretation':'Early small-corpus learning; relation success retained. Quality metrics are milestones, not evidence of whole-model failure or automatic training stops. New two-epoch allocation, no old terminal/call/time reuse.',
        'optimizer_policy':'Retain all parent populatedAdam/scaler/RNG/model; explicitly new lower-base LR schedule/cursors, no fresh moments.',
        'objective_change':'Train-only CE EOS weight4, unweighted symbol normalization; VMSE/clean unchanged. Relative EOF CE mass0.5296%, inference and evaluation unchanged.'}

def create():
    output=ROOT/'runs/text_language_continuation_20261028_v7'
    if output.exists():raise FileExistsError('new output only')
    p=make_plan(output)
    if sha(p['source'])!=SOURCE_SHA:raise ValueError('source SHA')
    output.mkdir();io.atomic_json(output/'plan.json',p);snapshot=output/'code_snapshot';snapshot.mkdir()
    for path in p['files']:
        q=Path(path)
        if q.suffix=='.py' and q.parent.name in ('scripts','research','tests'):(snapshot/q.name).write_bytes(q.read_bytes())
    print(json.dumps({'output':str(output),'draft_plan_sha256':sha(output/'plan.json'),'authorized':False}))
if __name__=='__main__':create()
