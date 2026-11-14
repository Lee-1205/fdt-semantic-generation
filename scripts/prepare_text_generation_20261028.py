"""Create once; sealing is a separate root action after actual CPU review."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from text_generation_support_v1 import sha,SCHEMA,SOURCE_SHA
import learning_io_v2 as io

def create():
    output=ROOT/'runs/text_generation_20261028_v1';output.mkdir(exist_ok=False)
    data=ROOT/'research/artifacts/text_generation_20261028_v1'
    source=ROOT/'runs/querycount_alignment_20261027_v1/model/payload_000010.pt'
    dependencies=[ROOT/p for p in ('research/text_diffusion_v1.py','research/text_generation_data_v1.py',
        'scripts/run_text_generation_20261028.py','scripts/supervise_text_generation_20261028.py',
        'scripts/text_generation_support_v1.py','scripts/supervise_querycount_partial9_20261027_v3.py',
        'scripts/learning_io_v2.py','tests/test_text_diffusion_v1.py')]
    dependencies+=list(sorted((ROOT/'src/fdt_local').glob('*.py')))
    dependencies+=list(sorted(data.glob('*.json')))+[source]
    plan={'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),'data':str(data),
          'source':str(source),'source_sha':SOURCE_SHA,
          'optimizer_policy':'fresh AdamW on new text branch; frozen copied language',
          'limits':{'attempts':256,'batch':8,'wall':900,'reserve':90,'max_calls':1300,
                    'max_output_bytes':12*1024**3,'start_free_bytes':36*1024**3,'min_gpu_free_mib':12288},
          'allocation_start':None,'deadline':None,'cutoff':None,'e2_final_accessed':False,
          'files':{str(p):sha(p) for p in dependencies},
          'purpose':'First raw-prompt -> Gaussian VP latent -> UTF8 bytes/EOS generation; separate from typed relation model',
          'stops':['nonfinite','frozen language drift','unexpected gradient ownership','3AMP skips','loss>100',
                   'fixed time/calls/exposure','disk/output','owned process supervision'],
          'evaluation':'16 train probes +32 held entity-pair/language groups +32 new wording development; same seed17 before/after and shuffled prompt control',
          'prohibited':['old allocation reuse','automatic restart','E2 final','runtime templates/retrieval/gold/entity roles','agents']}
    if plan['files'][str(source)]!=SOURCE_SHA:raise ValueError('source SHA')
    io.atomic_json(output/'plan.json',plan)
    snapshots=output/'code_snapshot';snapshots.mkdir()
    for p in dependencies:
        if p.suffix=='.py' and p.parent.name in ('scripts','research','tests'):
            (snapshots/p.name).write_bytes(p.read_bytes())
    print(json.dumps({'output':str(output),'plan_sha':sha(output/'plan.json'),'authorized':False}))
if __name__=='__main__':create()
