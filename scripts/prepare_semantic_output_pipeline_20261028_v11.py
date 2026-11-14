"""Create NEW immutable targets/orders/draft; never reuses any old clock."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from research.semantic_output_data_v11 import build
from research.text_span_data_v4 import finite_pair_orders
from run_semantic_output_pipeline_20261028_v11 import SCHEMA,LIMITS,SOURCES,SEEDS
import text_generation_support_v1 as support
import learning_io_v2 as io

COST={'input_training':2048,'meaning_training':3072,'frozen_prediction_cache':5040,'language_training':1024,
    'initial_meaning':432,'final_generation_max':960,'oracle_diagnostic_max':80,'total_max':12656}

def prepare():
    base=ROOT/'research/artifacts/text_surface_alignment_20261028_v5_cpu'
    artifact=ROOT/'research/artifacts/semantic_output_20261028_v11_cpu'
    output=ROOT/'runs/semantic_output_pipeline_20261028_v11'
    if artifact.exists() or output.exists():raise FileExistsError('new output only')
    manifest=io.load_json(base/'manifest.json')
    for name,h in manifest['files'].items():
        if support.sha(base/name)!=h:raise ValueError('old source artifact changed')
    f=io.load_json(base/'features.json');y=build(f,io.load_json(base/'targets.json'))
    pairs=io.load_json(base/'pairs.json')['train'];orders={p:finite_pair_orders(pairs,2048,4,s) for p,s in SEEDS.items()}
    assert all(len(v)==8192 for v in orders.values())
    # finite_pair_orders is a flat sequence; actual driver consumes 8-row batches.
    orders={p:[order[i:i+8] for i in range(0,len(order),8)] for p,order in orders.items()}
    artifact.mkdir();io.atomic_json(artifact/'features.json',f);io.atomic_json(artifact/'targets.json',y);io.atomic_json(artifact/'orders.json',orders)
    io.atomic_json(artifact/'manifest.json',{'schema':'input_meaning_output_meaning_targets_v11','source_manifest_sha256':support.sha(base/'manifest.json'),
        'feature_file_identical_to_surface_v5':support.sha(base/'features.json')==support.sha(artifact/'features.json'),
        'raw_feature_rows_unchanged':f==io.load_json(base/'features.json'),'new_raw_rows':0,'input_output_targets_separate':True,
        'sizes':{k:len(v) for k,v in f.items()},'train_unique_facts':64,'finite_epochs_each_phase':4,'row_exposure_each_phase':8192,
        'language_target_UTF8_bytes':4*sum(len(r['answer'].encode('utf-8')) for r in y['train']),
        'E2_final_accessed':False,'source_holdout_imported_to_train':False,
        'files':{n:support.sha(artifact/n) for n in ('features.json','targets.json','orders.json')}})
    files=[ROOT/'research/semantic_output_pipeline_v11.py',ROOT/'research/semantic_output_data_v11.py',
        ROOT/'scripts/semantic_output_engine_v11.py',ROOT/'scripts/run_semantic_output_pipeline_20261028_v11.py',
        ROOT/'scripts/supervise_semantic_output_pipeline_20261028_v11.py',Path(__file__),ROOT/'tests/test_semantic_output_v11.py',
        ROOT/'research/text_diffusion_v1.py',ROOT/'research/text_context_decoder_v2.py',ROOT/'research/text_span_data_v4.py',
        ROOT/'scripts/text_generation_support_v1.py',ROOT/'scripts/text_language_runtime_support_v6.py',ROOT/'scripts/learning_io_v2.py',
        ROOT/'scripts/supervise_querycount_partial9_20261027_v3.py']
    files+=list(sorted((ROOT/'src/fdt_local').glob('*.py')))+list(sorted(artifact.glob('*.json')))+list(sorted(base.glob('*.json')))
    for relative,h in SOURCES.values():
        q=ROOT/relative
        if support.sha(q)!=h:raise ValueError('source fullstate changed')
        files.append(q)
    p={'schema':SCHEMA,'authorized':False,'root':str(ROOT),'output':str(output),'artifact':str(artifact),'limits':LIMITS,
        'files':{str(path.resolve()):support.sha(path) for path in files},'phase_limits':{p:1024 for p in SEEDS},'epochs_each_phase':4,
        'output_meaning_target':'query role16 + answer relation4 + constant1','cost_contract':COST,
        'allocation_start':None,'deadline':None,'cutoff':None,'E2_final_accessed':False,'evaluation_teacher_allowed':False,
        'latest_human_instruction':'Continue improve/train/validate: language -> INPUT meaning extraction -> OUTPUT-specific meaning denoising -> committed-state language realization. ROOT only.',
        'training_policy':'Fresh composed-model phase optimizers/scaler; warmstart only declared modules, all source fullstates/AdamRNG preserved. Input then output-meaning then language; predicted input always; teacher output conditions training-only and predicted-matched transition.',
        'interpretation':'Narrow A-H one-fact comparison/negation composition;64training facts with correlated paraphrase/reversal views and already-observed controls. No raw prompt/answer byte latent to language realizer. No broad semantic/LM completion.'}
    output.mkdir();io.atomic_json(output/'plan.json',p);snap=output/'code_snapshot';snap.mkdir()
    for path in files:
        if path.suffix=='.py' and path.parent.name in ('scripts','research','tests'):(snap/path.name).write_bytes(path.read_bytes())
    print(json.dumps({'output':str(output),'draft_plan_sha256':support.sha(output/'plan.json'),'new_raw_rows':0,'maximum_calls':COST['total_max']}))
if __name__=='__main__':prepare()
