"""New unauthorized draft and immutable code snapshot; no clock seal/launch."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import learning_io_v2 as io
from text_generation_support_v1 import sha
from run_text_language_continuation_20261028 import SCHEMA,LIMITS,PARENT_SHA,MANIFEST_SHA,cost_contract,generation_indices
from text_language_engine_v5 import finite_order


def make_plan(output):
    artifact=ROOT/'research/artifacts/text_story_continuation_20261028_v5_cpu'
    parent=ROOT/'runs/text_span_alignment_20261028_v4/realizer/checkpoints/state_1024.pt'
    paths=[ROOT/p for p in ('research/text_language_continuation_v5.py','research/text_corpus_contract_v5.py',
        'research/text_context_decoder_v2.py','research/text_diffusion_v1.py','scripts/text_language_engine_v5.py',
        'scripts/run_text_language_continuation_20261028.py','scripts/supervise_text_language_continuation_20261028.py',
        'scripts/prepare_text_language_continuation_20261028.py','scripts/text_generation_support_v1.py',
        'scripts/learning_io_v2.py','scripts/supervise_querycount_partial9_20261027_v3.py',
        'tests/test_text_language_runtime_v5.py')]
    paths+=list(sorted((ROOT/'src/fdt_local').glob('*.py')))+list(sorted(artifact.glob('*.json')))+[parent]
    targets=io.load_json(artifact/'targets.json')
    return {'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),
        'artifact':str(artifact),'manifest_sha':MANIFEST_SHA,'parent':str(parent),'parent_sha':PARENT_SHA,
        'files':{str(p):sha(p) for p in paths},'limits':LIMITS,'epochs':1,'order':finite_order(3271),
        'generation_indices':generation_indices(targets['dev']),'cost_contract':cost_contract(),
        'allocation_start':None,'deadline':None,'cutoff':None,'E2_final_accessed':False,
        'checkpoints':[0,1,128,256,384,409],'row_exposure_cap':3271,'target_byte_exposure_cap':384675,
        'objective':'VPvMSE+.25cleanMSE+.5byteCE+ARCE; live language tail7/8; AR free proposal no gold',
        'stops':['nonfinite','frozen/gradient ownership','3AMP skips','loss>1000/20x','fixed time/calls/oneepoch','disk/output','OSprocess'],
        'purpose':'One new finite natural-document byte-continuation trial, not resume or generalLM completion'}


def create():
    output=ROOT/'runs/text_language_continuation_20261028_v5'
    if output.exists():raise FileExistsError('never overwrite execution draft')
    plan=make_plan(output)
    if sha(plan['parent'])!=PARENT_SHA or sha(Path(plan['artifact'])/'manifest.json')!=MANIFEST_SHA:raise ValueError('source identity')
    output.mkdir();io.atomic_json(output/'plan.json',plan);snapshot=output/'code_snapshot';snapshot.mkdir()
    for name in plan['files']:
        path=Path(name)
        if path.suffix=='.py' and path.parent.name in ('scripts','research','tests'):(snapshot/path.name).write_bytes(path.read_bytes())
    print(json.dumps({'output':str(output),'draft_plan_sha256':sha(output/'plan.json'),'authorized':False}))
if __name__=='__main__':create()
