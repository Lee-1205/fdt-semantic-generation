"""One NEW measured 1200sec epoch, replacing the unlaunched split proposal."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import run_three_memory_codec_20261029_v38 as worker
import text_generation_support_v1 as support

def main():
    artifact=ROOT/'research/artifacts/three_memory_codec_20261029_v38_worker_cpu';artifact.mkdir(exist_ok=False)
    old=ROOT/'runs/three_memory_codec_20261029_v37';p=json.loads((old/'plan.json').read_text(encoding='utf-8'))
    review=json.loads((old/'root_result.json').read_text(encoding='utf-8'))
    if review['status']!='V37_COMPLETE_THIRD_BROAD_EPOCH_ROOT_REVIEWED' or not review['all_owned_proxy_gone_before_analysis']:raise ValueError('completed source review')
    output=ROOT/'runs/three_memory_codec_20261029_v38'
    if output.exists():raise FileExistsError('fresh output only')
    files=dict(p['files'])
    for name in ('scripts/run_three_memory_codec_20261029_v38.py','scripts/supervise_three_memory_codec_20261029_v38.py',
        'scripts/three_memory_codec_engine_v38.py','scripts/prepare_three_memory_worker_v38_cpu.py','tests/test_three_memory_worker_v38.py'):
        files[str((ROOT/name).resolve())]=support.sha(ROOT/name)
    source=old/'model/checkpoints/state_4096.pt'
    for path in (source,old/'plan.json',old/'root_result.json'):files[str(path)]=support.sha(path)
    p.update(schema=worker.SCHEMA,authorized=False,output=str(output),source=str(source),source_sha=worker.SOURCE_SHA,
        source_plan_sha=worker.SOURCE_PLAN,source_algorithm=worker.SOURCE_ALGORITHM,source_plan_path=str(old/'plan.json'),
        source_global_attempts=116708,source_global_updates=116665,limits=worker.LIMITS,cost_contract=worker.COST,
        continuation_policy=worker.CONTINUATION,time_cost_contract=worker.TIME_COST,epochs=1,learning_epoch_number=4,order_seed_range=[37001,37001],
        training_exposure={'unique_training_docs':8192,'unique_training_rows':32768,'selected_rows':32768,'target_UTF8_bytes':2926478,'finite_epochs':1},
        allocation_start=None,deadline=None,cutoff=None,
        source_baseline_reference={'run':str(old),'root_result_sha256':support.sha(old/'root_result.json'),'same_function':True,'repeat_initial_GPU_baseline':False},
        purpose='Fourth finite broad SAME-input three-memory language codec epoch; one NEW measured1200sec allocation BEFORE any model launch. ALL91Adam/scaler/RNG preserved, no denoising/whole-copy/held training. All older allocations remain closed; this is a separate measured1200sec purpose.',files=files)
    path=artifact/'plan_unsealed.json'
    with path.open('x',encoding='utf-8') as f:json.dump(p,f,ensure_ascii=False,indent=2)
    worker.validate(path,output,worker.__file__)
    print(json.dumps({'status':'V38_NEW_FOURTH_EPOCH_PROSPECTIVE','cost_max':15060,'time_budget':worker.TIME_COST,'GPU_calls':0}))
if __name__=='__main__':main()
