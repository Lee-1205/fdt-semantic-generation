"""First causal interface repair arm, beta0; future repetition arms are separate."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
from text_generation_support_v1 import sha
from research.text_decoder_addressing_v9 import ADDRESS_POLICY
from run_text_decoder_addressing_20261028_v9 import SCHEMA,LIMITS,SOURCE_SHA,SOURCE_PLAN,SOURCE_ALGORITHM,MANIFEST_SHA,generation_indices

CURRICULUM={'epochs':4,'seeds':[81,82,83,84],'reconstruction_attempts':1636,'joint_attempts':0,
    'training_eos_CE_weight':4.0,'span_UL_beta':0.0,'base_language_LR':1e-5,'base_other_LR':3e-5,
    'decoder_rawcontext_dropout':0.5,'codec_noise_sigma_max':0.15,'memory_position_init':'retainV8',
    'quality_metrics_stop_training':False,'new_row_exposures':13084,'new_target_bytes':1538700}
COST={'reconstruction':6544,'NLL':1998,'generation':1100,'held_reconstruction':24,'oracle_causal_generation_max':516,'total_max':10182}

def make_plan(output):
    parent=ROOT/'runs/text_decoder_alignment_20261028_v8';source=parent/'model/checkpoints/state_1636.pt'
    artifact=ROOT/'research/artifacts/text_story_continuation_20261028_v5_cpu'
    negative=ROOT/'research/artifacts/text_decoder_alignment_training_20261028_v8'
    names=('research/text_decoder_addressing_v9.py','scripts/text_decoder_addressing_engine_v9.py','research/text_language_continuation_v7.py','research/text_language_continuation_v5_deterministic_v2.py',
        'research/text_decoder_alignment_v8.py','research/text_decoder_alignment_training_v8.py','research/text_diffusion_v1.py',
        'research/text_context_decoder_v2.py','research/text_corpus_contract_v5.py','scripts/text_language_engine_v7.py',
        'scripts/text_language_engine_v6.py','scripts/text_language_engine_v5.py','scripts/text_decoder_alignment_engine_v8.py',
        'scripts/text_decoder_alignment_step_v8.py','scripts/text_decoder_alignment_data_v8.py','scripts/text_decoder_alignment_probes_v8.py',
        'scripts/text_language_runtime_support_v6.py','scripts/run_text_language_continuation_20261028_deterministic_v2.py',
        'scripts/run_text_decoder_addressing_20261028_v9.py','scripts/supervise_text_decoder_addressing_20261028_v9.py',
        'scripts/prepare_text_decoder_addressing_20261028_v9.py','scripts/text_generation_support_v1.py',
        'scripts/learning_io_v2.py','scripts/supervise_querycount_partial9_20261027_v3.py','tests/test_decoder_addressing_runtime_v9.py')
    files=[ROOT/n for n in names]+list(sorted((ROOT/'src/fdt_local').glob('*.py')))+list(sorted(artifact.glob('*.json')))+list(sorted(negative.glob('*.json')))
    files+=[source,parent/'plan.json']+[parent/n for n in ('final_NLL.json','final_NLL_logits.pt','final_generation.json')]
    return {'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),'source':str(source),
        'source_sha':SOURCE_SHA,'source_plan_sha':SOURCE_PLAN,'source_algorithm':SOURCE_ALGORITHM,'source_attempts':1636,
        'global_attempt_cap':4499,'epochs':4,'artifact':str(artifact),'manifest_sha':MANIFEST_SHA,
        'negative_artifact':str(negative),'negative_manifest_sha':sha(negative/'manifest.json'),
        'baseline_run':str(parent),'reference_is_parent_architecture':True,'curriculum':CURRICULUM,'address_policy':ADDRESS_POLICY,
        'generation_indices':generation_indices(io.load_json(artifact/'targets.json')['dev']),
        'files':{str(p):sha(p) for p in files},'limits':LIMITS,'cost_contract':COST,
        'allocation_start':None,'deadline':None,'cutoff':None,'E2_final_accessed':False,'automatic_restart':False,
        'latest_human_instruction':'Latest HUMAN 해봐 authorizes continue explicit latent addressing and faithful reconstruction; beta0 isolates penalty.',
        'optimizer_policy':'All V8weights/145populatedAdam states/scaler/allRNG retained exactly, no new parameter or position reset. EntireVP frozen; decoder-only learning.',
        'interpretation':'V8 continuation with explicit same-index soft cross-attention addressing; retain all learned positions/weights/Adam/RNG, no new parameters. Beta0 and VP frozen, four finite exposure epochs. Initial/final clean/noisy/oracle probes measure interface fidelity only; final dev generation uses predicted VP. No generic semantic completion claim.',
        'reference_policy':'Parent V8reference is NOT exact new initial baseline because addressing changes the function. Initial/final narrow interface/oracle diagnostics are measured, no broad repeated GPU baseline.',
        'long_term_contract':'Develop explicit internally settled semantic state before language, with permitted exact memory only. Current byte-codec repair is an interface step, not that whole architecture achieved.'}

def create():
    output=ROOT/'runs/text_decoder_addressing_20261028_v9'
    if output.exists():raise FileExistsError('new output only')
    p=make_plan(output)
    if sha(p['source'])!=SOURCE_SHA:raise ValueError('source SHA')
    output.mkdir();io.atomic_json(output/'plan.json',p);snapshot=output/'code_snapshot';snapshot.mkdir()
    for path in p['files']:
        q=Path(path)
        if q.suffix=='.py' and q.parent.name in ('scripts','research','tests'):(snapshot/q.name).write_bytes(q.read_bytes())
    print(json.dumps({'output':str(output),'draft_plan_sha256':sha(output/'plan.json'),'authorized':False}))
if __name__=='__main__':create()
