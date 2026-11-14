"""New broad-data contract; preserves V33 and never launches or opens its clock."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import run_three_memory_codec_20261029_v34 as worker
import text_generation_support_v1 as support

def main():
    artifact=ROOT/'research/artifacts/three_memory_codec_20261029_v34_worker_cpu';artifact.mkdir(exist_ok=False)
    old=ROOT/'runs/three_memory_codec_20261029_v33';p=json.loads((old/'plan.json').read_text(encoding='utf-8'))
    report=json.loads((old/'root_result.json').read_text(encoding='utf-8'))
    if report['status']!='V33_COMPLETE_ROOT_SAVED_ONLY_REVIEWED' or not report['all_owned_and_proxy_gone_before_analysis']:raise ValueError('parent review/exit')
    output=ROOT/'runs/three_memory_codec_20261029_v34'
    if output.exists():raise FileExistsError('fresh output only')
    files=dict(p['files'])
    for name in ('scripts/run_three_memory_codec_20261029_v34.py','scripts/supervise_three_memory_codec_20261029_v34.py',
        'scripts/three_memory_codec_engine_v34.py','scripts/prepare_three_memory_worker_v34_cpu.py',
        'tests/test_three_memory_worker_v34.py'):
        files[str((ROOT/name).resolve())]=support.sha(ROOT/name)
    for path in (old/'model/checkpoints/state_2048.pt',old/'plan.json',old/'root_result.json'):
        files[str(path)]=support.sha(path)
    p.update(schema=worker.SCHEMA,authorized=False,output=str(output),source=str(old/'model/checkpoints/state_2048.pt'),
        source_sha=worker.SOURCE_SHA,source_plan_sha=worker.SOURCE_PLAN,source_algorithm=worker.SOURCE_ALGORITHM,
        source_plan_path=str(old/'plan.json'),source_global_attempts=104420,source_global_updates=104383,
        limits=worker.LIMITS,cost_contract=worker.COST,continuation_policy=worker.CONTINUATION,epochs=1,order_seed_range=[34001,34001],
        initialization_seed=None,training_exposure={'unique_training_docs':8192,'unique_training_rows':32768,'selected_rows':32768,
            'target_UTF8_bytes':2926478,'finite_epochs':1},allocation_start=None,deadline=None,cutoff=None,
        purpose='Broaden newly connected three-memory language codec from64 repeated TRAIN spans to one existing32768-span TRAIN epoch. SAME input meaning reconstruction, no denoising/copy bypass/held-to-train.',
        source_baseline_reference={'run':str(old),'root_result_sha256':support.sha(old/'root_result.json'),
            'same_function':True,'repeat_initial_GPU_baseline':False},files=files)
    path=artifact/'plan_unsealed.json'
    with path.open('x',encoding='utf-8') as f:json.dump(p,f,ensure_ascii=False,indent=2)
    worker.validate(path,output,worker.__file__)
    print(json.dumps({'status':'V34_NEW_ACTUAL_SCHEMA_CPU_PROSPECTIVE','plan':str(path),'cost_max':worker.COST['total_max'],'model_calls':0}))
if __name__=='__main__':main()
