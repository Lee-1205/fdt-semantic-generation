"""New fixed AE allocation draft, after corpus/architecture contracts are concrete."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from run_owt_fuzzy_reconstruction_20261028_v12 import SCHEMA,LIMITS,SOURCES,COST
import learning_io_v2 as io
import text_generation_support_v1 as support

def main():
    artifact=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v12';output=ROOT/'runs/owt_fuzzy_reconstruction_20261028_v12'
    if output.exists():raise FileExistsError('new output only')
    names=('research/owt_fuzzy_reconstruction_v12.py','scripts/owt_fuzzy_engine_v12.py',
        'scripts/run_owt_fuzzy_reconstruction_20261028_v12.py','scripts/supervise_owt_fuzzy_reconstruction_20261028_v12.py',
        'scripts/prepare_owt_sentences_v12.py','scripts/prepare_owt_fuzzy_reconstruction_20261028_v12.py','tests/test_owt_fuzzy_reconstruction_v12.py',
        'research/text_diffusion_v1.py','research/text_context_decoder_v2.py','scripts/text_generation_support_v1.py',
        'scripts/text_language_runtime_support_v6.py','scripts/learning_io_v2.py','scripts/supervise_querycount_partial9_20261027_v3.py')
    files=[ROOT/n for n in names]+list(sorted((ROOT/'src/fdt_local').glob('*.py')))+list(sorted(p for p in artifact.rglob('*') if p.is_file()))
    for relative,h in SOURCES.values():
        p=ROOT/relative
        if support.sha(p)!=h:raise ValueError('parent fullstate changed')
        files.append(p)
    plan={'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),'artifact':str(artifact),'limits':LIMITS,
        'files':{str(p.resolve()):support.sha(p) for p in files},'epochs':8,'cost_contract':COST,'generation_indices':[4*i for i in range(16)],
        'reasoning_diffusion_training':False,'E2_final_accessed':False,'allocation_start':None,'deadline':None,'cutoff':None,
        'latest_human_instruction':'Use OpenWebText. Keep original fuzzy anchors; sentence -> fuzzy meaning extraction -> reconstruction from extracted meaning cycle first. Reasoning/output-meaning diffusion trained separately later. ROOT only.',
        'optimizer_policy':'Warmstart typed core and V9 language/34decoder tensors; fresh composed AE AdamW/scaler/allRNG. Source fullstates and all source AdamRNG preserved.',
        'representation':'Original Router topk sigmoid nonexclusive mu and probabilistic-sum union, anchor prior/dynamic value/condition; two extracted fuzzy views to decoder only. No raw byte/context bypass.',
        'interpretation':'Reconstruction plus swapped/zero meaning controls tests information preservation/content usage. Does not alone identify a unique universal semantic space, prove reasoning or general LM completion.',
        'automatic_restart':False}
    output.mkdir();io.atomic_json(output/'plan.json',plan);snapshot=output/'code_snapshot';snapshot.mkdir()
    for path in files:
        if path.suffix=='.py' and path.parent.name in ('scripts','research','tests'):(snapshot/path.name).write_bytes(path.read_bytes())
    print(json.dumps({'draft_plan_sha256':support.sha(output/'plan.json'),'maximum_calls':COST['total_max'],'attempts':2048,'epochs':8,'launches':0}))
if __name__=='__main__':main()
