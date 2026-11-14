"""Unsealed NEW V2 contract; old pilot remains closed and unchanged."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from text_generation_support_v1 import sha
from run_text_context_decoder_20261028 import LIMITS,SCHEMA
import learning_io_v2 as io

def create():
    output=ROOT/'runs/text_context_decoder_20261028_v2';output.mkdir(exist_ok=False)
    data=ROOT/'research/artifacts/text_generation_20261028_v1'
    parent=ROOT/'runs/text_generation_20261028_v1/checkpoints/state_0256.pt'
    parent_sha='0e3947eea37d5141ef299044c7c24369aeb79a18048f283b526cb29c82ce2163'
    paths=[ROOT/p for p in ('research/text_diffusion_v1.py','research/text_context_decoder_v2.py','research/text_generation_data_v1.py',
        'scripts/run_text_context_decoder_20261028.py','scripts/supervise_text_context_decoder_20261028.py',
        'scripts/text_generation_support_v1.py','scripts/supervise_querycount_partial9_20261027_v3.py',
        'scripts/learning_io_v2.py','tests/test_text_context_decoder_v2.py')]
    paths+=list(sorted((ROOT/'src/fdt_local').glob('*.py')))+list(sorted(data.glob('*.json')))+[parent]
    plan={'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),'data':str(data),
        'parent':str(parent),'parent_sha':parent_sha,'limits':LIMITS,'allocation_start':None,'deadline':None,'cutoff':None,
        'e2_final_accessed':False,'files':{str(p):sha(p) for p in paths},
        'purpose':'Joint-byte contextual decoding of frozen VP proposal; test EOS/UTF8/exact and zero-latent control',
        'new_train_pool':1024,'prospective_finite_epochs':4,'max_train_exposures':4096,'optimizer':'fresh AdamW, decoder only',
        'cache_gold_allowed':False,'decoder_teacher_forcing':'shifted previous bytes, causal mask only',
        'targets':{'well_formed_each_dev':.8,'exact_each_dev':.25},
        'stops':['numerical','frozen generator/ownership','3AMPskips','disk/output','fixed time/call/exposure','owned process'],
        'interpretation':'Hybrid causal realizer; raw prompt also accessible. VP contribution requires measured zero-latent comparison. Not a general corpus/final benchmark.'}
    if plan['files'][str(parent)]!=parent_sha:raise ValueError('parent identity')
    io.atomic_json(output/'plan.json',plan)
    snapshots=output/'code_snapshot';snapshots.mkdir()
    for p in paths:
        if p.suffix=='.py' and p.parent.name in ('research','scripts','tests'):(snapshots/p.name).write_bytes(p.read_bytes())
    print(json.dumps({'output':str(output),'draft_plan_sha':sha(output/'plan.json')}))
if __name__=='__main__':create()
