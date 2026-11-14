"""Explicit new remainder purpose; old terminal/status/clock remain closed."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import run_three_memory_codec_20261029_v35 as worker
import text_generation_support_v1 as support

def main():
    artifact=ROOT/'research/artifacts/three_memory_codec_20261029_v35_worker_cpu';artifact.mkdir(exist_ok=False)
    old=ROOT/'runs/three_memory_codec_20261029_v34';p=json.loads((old/'plan.json').read_text(encoding='utf-8'))
    review=json.loads((old/'root_result.json').read_text(encoding='utf-8'))
    if review['status']!='V34_BOUNDED_PARTIAL_EPOCH_ROOT_REVIEWED' or not review['all_owned_proxy_gone_before_analysis']:raise ValueError('source partial independent review')
    output=ROOT/'runs/three_memory_codec_20261029_v35'
    if output.exists():raise FileExistsError('new output')
    files=dict(p['files'])
    for name in ('scripts/run_three_memory_codec_20261029_v35.py','scripts/supervise_three_memory_codec_20261029_v35.py',
        'scripts/three_memory_codec_engine_v35.py','scripts/prepare_three_memory_worker_v35_cpu.py','tests/test_three_memory_worker_v35.py'):
        files[str((ROOT/name).resolve())]=support.sha(ROOT/name)
    source=old/'emergency/checkpoints/state_3407.pt'
    for path in (source,old/'plan.json',old/'root_result.json'):files[str(path)]=support.sha(path)
    p.update(schema=worker.SCHEMA,authorized=False,output=str(output),source=str(source),source_sha=worker.SOURCE_SHA,
        source_plan_sha=worker.SOURCE_PLAN,source_algorithm=worker.SOURCE_ALGORITHM,source_plan_path=str(old/'plan.json'),
        source_global_attempts=107827,source_global_updates=107789,limits=worker.LIMITS,cost_contract=worker.COST,
        continuation_policy=worker.CONTINUATION,training_exposure={'epoch_total_training_docs':8192,'epoch_total_training_rows':32768,
            'consumed_before_rows':27256,'unique_training_rows':5512,'selected_rows':5512,'target_UTF8_bytes':492660,'origin_finite_epochs':1},
        allocation_start=None,deadline=None,cutoff=None,
        purpose='New bounded completion of unconsumed original finite epoch suffix5512rows/689batches, retaining original order/cosine position and ALL91Adam/scaler/RNG; then same-input reconstruction assessment. No old clock/call reuse, no repeated consumed rows.',files=files)
    path=artifact/'plan_unsealed.json'
    with path.open('x',encoding='utf-8') as f:json.dump(p,f,ensure_ascii=False,indent=2)
    worker.validate(path,output,worker.__file__)
    print(json.dumps({'status':'V35_NEW_REMAINDER_SCHEMA_CPU_PROSPECTIVE','cost_max':4839,'old_allocation_reopened':False,'GPU_calls':0}))
if __name__=='__main__':main()
