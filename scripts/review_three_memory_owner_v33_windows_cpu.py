"""Real Windows owner/proxy/native-child/exit check; CPU fixture, model calls0."""
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import supervise_three_memory_codec_20261029_v33 as owner


def main():
    artifact = ROOT/'research/artifacts/three_memory_codec_20261029_v33_worker_cpu'
    revision = sys.argv[1] if len(sys.argv) == 2 else 'r2'
    if revision not in ('r2', 'r3'): raise ValueError('fixed CPU review revision')
    record = artifact/('real_windows_owner_review.json' if revision == 'r2' else 'real_windows_owner_review_r3.json')
    if record.exists(): raise FileExistsError(record)
    assert 'torch' not in sys.modules
    output = artifact/('real_owner_CPU_fixture' if revision == 'r2' else 'real_owner_CPU_fixture_r3'); output.mkdir(exist_ok=False)
    fixture = artifact/('run_three_memory_owner_CPU_fixture.py' if revision == 'r2' else 'run_three_memory_owner_CPU_fixture_r3.py')
    fixture.write_text('''import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');p.add_argument('--preflight',action='store_true');a=p.parse_args()
if a.preflight:
 print('CPU_PREFLIGHT_PASS');raise SystemExit(0)
o=Path(a.output)
(o/'progress.json').write_text(json.dumps({'phase':'CPU_FIXTURE_ONLY','attempts':0,'allocation_calls':{'encode':0,'denoise':0,'decoder':0},'native_pid':os.getpid(),'parent_pid':os.getppid()}),encoding='utf-8')
time.sleep(2)
(o/'result.json').write_text(json.dumps({'status':'THREE_MEMORY_CODEC_V33_COMPLETE','CPU_fixture_only':True,'model_calls':0}),encoding='utf-8')
''', encoding='utf-8')
    plan = json.loads((artifact/('plan_unsealed_'+revision+'.json')).read_text(encoding='utf-8'))
    plan.update(output=str(output), authorized=True, allocation_start=time.time(), CPU_fixture_only=True)
    plan.update(deadline=plan['allocation_start']+900, cutoff=plan['allocation_start']+780)
    plan['files'][str(fixture)] = owner.native.sha256(fixture)
    path = output/'plan.json'; path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    original_resources = owner.resources
    # The CPU-only fixture is not a model launch and may run below the model's
    # declared model start floor. The actual model owner function stays unchanged.
    owner.resources = lambda p: {'CPU_fixture_only': True, 'C_free_bytes': shutil.disk_usage(ROOT).free, 'GPU_access': False}
    try: code = owner.supervise(path, output, fixture)
    finally: owner.resources = original_resources
    final = json.loads((output/'supervisor/final_status.json').read_text(encoding='utf-8'))
    assert code == 0 and final['status'] == 'COMPLETE' and final['launches'] == 1
    assert final['exit']['all_owned_processes_gone'] and final['exit_code'] == 0
    assert final['actual'] and final['actual']['identity_verified']
    assert 'torch' not in sys.modules
    result = {'status': 'REAL_WINDOWS_CPU_OWNER_PROXY_NATIVE_CHILD_NATURAL_EXIT_PASS',
        'CPU_fixture_only': True, 'actual_model_worker_used': False, 'neural_model_calls': 0,
        'CUDA_initialized': False, 'owner_imported_Torch': False,
        'owner_sha256': owner.native.sha256(Path(owner.__file__)), 'plan_sha256': owner.native.sha256(path),
        'actual_native_identity': final['actual'], 'exit': final['exit'], 'exit_code': final['exit_code'],
        'GPU_start_floor_bypassed_for_actual_model': False,
        'actual_model_CPU_preflight': 'Separate prospective plan owner --validate; see actual_model_worker_preflight record',
        'actual_GPU_training': False}
    with record.open('x', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({'status': result['status'], 'native': result['actual_native_identity'], 'all_owned_gone': True}))


if __name__ == '__main__': main()
