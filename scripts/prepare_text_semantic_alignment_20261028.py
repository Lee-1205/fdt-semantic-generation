"""NEW CPU contract after verified V2 grammar-but-not-binding result."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
import learning_io_v2 as io
from text_generation_support_v1 import sha
from run_text_semantic_alignment_20261028 import LIMITS,SCHEMA
from research.text_semantic_targets_v3 import build_metadata

def create():
    out=ROOT/'runs/text_semantic_alignment_20261028_v3';out.mkdir(exist_ok=False)
    data=ROOT/'research/artifacts/text_generation_20261028_v1'
    parent=ROOT/'runs/text_context_decoder_20261028_v2/checkpoints/state_0512.pt'
    ph='b843d90b71f69419402fe3a6064522224c85058ab046b6f6fd1536348ca7d063'
    cache=ROOT/'runs/text_context_decoder_20261028_v2/frozen_context_cache.pt'
    ch='c0fe127015ddd17319e46d352be526264ffec0693990679585916051bca3fce9'
    if sha(parent)!=ph or sha(cache)!=ch:raise ValueError('V2parent/cache hash')
    p=torch.load(parent,map_location='cpu',weights_only=True)
    rows={k:io.load_json(data/(k+'.json')) for k in ('train','train_probe','dev_pairs','dev_wording')}
    artifact=ROOT/'research/artifacts/text_semantic_targets_20261028_v3';artifact.mkdir(exist_ok=False)
    metadata=artifact/'targets.json';io.atomic_json(metadata,build_metadata(rows,p['selected_train_indices']));del p
    paths=[ROOT/f for f in ('research/text_diffusion_v1.py','research/text_context_decoder_v2.py','research/text_semantic_reader_v3.py',
        'research/text_semantic_targets_v3.py','research/text_generation_data_v1.py','scripts/run_text_semantic_alignment_20261028.py',
        'scripts/supervise_text_semantic_alignment_20261028.py','scripts/text_generation_support_v1.py',
        'scripts/supervise_querycount_partial9_20261027_v3.py','scripts/learning_io_v2.py','tests/test_text_semantic_reader_v3.py')]
    paths+=list(sorted((ROOT/'src/fdt_local').glob('*.py')))+list(sorted(data.glob('*.json')))+[parent,cache,metadata]
    plan={'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(out),'data':str(data),'parent':str(parent),'parent_sha':ph,
        'cache':str(cache),'cache_sha':ch,'metadata':str(metadata),'limits':LIMITS,'allocation_start':None,'deadline':None,'cutoff':None,
        'e2_final_accessed':False,'files':{str(f):sha(f) for f in paths},
        'purpose':'Learn raw prompt query subject/object/relation/language then condition learned byte generation; no supplied roles at inference',
        'optimizer':'fresh AdamW on new reader + warm decoder, frozen VP; separate stage',
        'auxiliary_targets':'training loss and score only; prohibited in model context/generate',
        'finite_pool':1024,'epochs':4,'max_row_exposures':4096,'new_VP_model_calls':0,'max_reader_decoder_calls':2500,
        'development_targets':{'dev_exact':.75,'positive_exact':.60,'well_formed':.90},
        'scope':'A-H bounded bilingual synthetic comparison; not broad NER or general language-model completion'}
    io.atomic_json(out/'plan.json',plan);sn=out/'code_snapshot';sn.mkdir()
    for f in paths:
        if f.suffix=='.py' and f.parent.name in ('research','scripts','tests'):(sn/f.name).write_bytes(f.read_bytes())
    print(json.dumps({'draft_plan_sha':sha(out/'plan.json'),'output':str(out)}))
if __name__=='__main__':create()
