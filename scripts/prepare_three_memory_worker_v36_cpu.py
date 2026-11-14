"""One NEW measured 1200sec epoch, replacing the unlaunched split proposal."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import run_three_memory_codec_20261029_v36 as worker
import text_generation_support_v1 as support

def main():
    artifact=ROOT/'research/artifacts/three_memory_codec_20261029_v36_worker_cpu';artifact.mkdir(exist_ok=False)
    old=ROOT/'runs/three_memory_codec_20261029_v35';p=json.loads((old/'plan.json').read_text(encoding='utf-8'))
    review=json.loads((old/'root_result.json').read_text(encoding='utf-8'))
    if review['status']!='V35_COMPLETE_ONE_FINITE_BROAD_EPOCH_ROOT_REVIEWED' or not review['all_owned_proxy_gone_before_analysis']:raise ValueError('completed source review')
    output=ROOT/'runs/three_memory_codec_20261029_v36'
    if output.exists():raise FileExistsError('fresh output only')
    files=dict(p['files'])
    for name in ('scripts/run_three_memory_codec_20261029_v36.py','scripts/supervise_three_memory_codec_20261029_v36.py',
        'scripts/three_memory_codec_engine_v36.py','scripts/prepare_three_memory_worker_v36_cpu.py','tests/test_three_memory_worker_v36.py'):
        files[str((ROOT/name).resolve())]=support.sha(ROOT/name)
    source=old/'model/checkpoints/state_0689.pt'
    for path in (source,old/'plan.json',old/'root_result.json'):files[str(path)]=support.sha(path)
    p.update(schema=worker.SCHEMA,authorized=False,output=str(output),source=str(source),source_sha=worker.SOURCE_SHA,
        source_plan_sha=worker.SOURCE_PLAN,source_algorithm=worker.SOURCE_ALGORITHM,source_plan_path=str(old/'plan.json'),
        source_global_attempts=108516,source_global_updates=108478,limits=worker.LIMITS,cost_contract=worker.COST,
        continuation_policy=worker.CONTINUATION,time_cost_contract=worker.TIME_COST,epochs=1,learning_epoch_number=2,order_seed_range=[35001,35001],
        training_exposure={'unique_training_docs':8192,'unique_training_rows':32768,'selected_rows':32768,'target_UTF8_bytes':2926478,'finite_epochs':1},
        allocation_start=None,deadline=None,cutoff=None,
        source_baseline_reference={'run':str(old),'root_result_sha256':support.sha(old/'root_result.json'),'same_function':True,'repeat_initial_GPU_baseline':False},
        purpose='Second finite broad SAME-input three-memory language codec epoch; one NEW measured1200sec allocation BEFORE any model launch. ALL91Adam/scaler/RNG preserved, no denoising/whole-copy/held training. Replaces only unlaunched two-part prospective proposal; all old clocks remain closed.',files=files)
    path=artifact/'plan_unsealed.json'
    with path.open('x',encoding='utf-8') as f:json.dump(p,f,ensure_ascii=False,indent=2)
    worker.validate(path,output,worker.__file__)
    print(json.dumps({'status':'V36_NEW_SECOND_EPOCH_PROSPECTIVE','cost_max':15060,'time_budget':worker.TIME_COST,'GPU_calls':0}))
if __name__=='__main__':main()
