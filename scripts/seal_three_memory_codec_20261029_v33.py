"""Seal ONE fresh fixed allocation after actual CPU/OS checks; never launches it."""
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
    if owner.native.sha256(artifact/'manifest.json') != 'f0d2cf6b5f4e1038212509eb7e1efc12be4ea93f31d1d2eae3260d7eaad42178':
        raise ValueError('CPU worker evidence pin')
    prospective = artifact/'plan_unsealed_r3.json'; p = owner.validate(prospective, ROOT/'runs/three_memory_codec_20261029_v33')
    output = Path(p['output'])
    if output.exists() or p['authorized'] or any(p[k] is not None for k in ('allocation_start', 'deadline', 'cutoff')):
        raise FileExistsError('fresh allocation only')
    real = json.loads((artifact/'real_windows_owner_review_r3.json').read_text(encoding='utf-8'))
    preflight = json.loads((artifact/'actual_model_worker_preflight_r3.json').read_text(encoding='utf-8'))
    if not real['exit']['all_owned_processes_gone'] or real['owner_sha256'] != owner.native.sha256(Path(owner.__file__)) or preflight['status'] != 'INDEPENDENT_VALIDATE_PASS':
        raise ValueError('actual preflight/native owner evidence')
    worker = ROOT/'scripts/run_three_memory_codec_20261029_v33.py'
    if owner.duplicates(worker) or owner.native.process_inventory(ROOT): raise RuntimeError('native duplicate model')
    resources = owner.resources(p)
    output.mkdir(exist_ok=False); snapshot = output/'code_snapshot'; snapshot.mkdir()
    for source in p['files']:
        source = Path(source)
        if source.suffix == '.py':
            target = snapshot/source.relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    p['files'][str(Path(__file__).resolve())] = owner.native.sha256(Path(__file__))
    shutil.copyfile(Path(__file__), snapshot/Path(__file__).name)
    ready = {'status': 'CPU_OS_RESOURCE_REVIEW_COMPLETE_BEFORE_ONE_NEW_SEAL', 'resources': resources,
        'worker_CPU_manifest_sha256': owner.native.sha256(artifact/'manifest.json'),
        'actual_model_preflight': preflight, 'actual_Windows_CPU_fixture': str(artifact/'real_windows_owner_review_r3.json'),
        'storage_policy_amended_before_any_model_Popen': p['start_storage_contract'],
        'GPU_execution_evidence': False, 'historical_budget_reused': False}
    (output/'readiness.json').write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding='utf-8')
    start = time.time(); p.update(authorized=True, allocation_start=start, deadline=start+900, cutoff=start+780)
    path = output/'plan.json'
    with path.open('x', encoding='utf-8') as f: json.dump(p, f, ensure_ascii=False, indent=2)
    owner.validate(path, output)
    print(json.dumps({'status': 'ONE_FRESH_PLAN_SEALED_NOT_LAUNCHED', 'output': str(output),
        'plan_sha256': owner.native.sha256(path), 'allocation_start': start, 'cutoff': p['cutoff'], 'deadline': p['deadline'],
        'resources': resources, 'start_floor_GiB': 35, 'historical_closed_budgets_changed': False}))


if __name__ == '__main__': main()
