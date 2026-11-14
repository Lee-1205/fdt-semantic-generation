"""Freeze the tested worker/OS evidence. No GPU seal/clock/Popen."""
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import supervise_three_memory_codec_20261029_v33 as owner


def main():
    artifact = ROOT/'research/artifacts/three_memory_codec_20261029_v33_worker_cpu'
    manifest = artifact/'manifest.json'
    if manifest.exists(): raise FileExistsError(manifest)
    p = json.loads((artifact/'plan_unsealed_r3.json').read_text(encoding='utf-8'))
    owner.validate(artifact/'plan_unsealed_r3.json', Path(p['output']))
    assert p['authorized'] is False and p['deadline'] is None and not Path(p['output']).exists()
    for filename, count in (('worker_cpu_r2.xml', 7), ('resource_owner_delta_r3.xml', 5)):
        suites = ET.parse(artifact/filename).getroot().findall('testsuite')
        assert sum(int(s.attrib['tests']) for s in suites) == count
        assert sum(int(s.attrib['errors'])+int(s.attrib['failures']) for s in suites) == 0
    real = json.loads((artifact/'real_windows_owner_review_r3.json').read_text(encoding='utf-8'))
    assert real['exit']['all_owned_processes_gone'] and real['actual_native_identity']['identity_verified']
    assert real['owner_sha256'] == owner.native.sha256(Path(owner.__file__))
    preflight = json.loads((artifact/'actual_model_worker_preflight_r3.json').read_text(encoding='utf-8'))
    assert preflight['status'] == 'INDEPENDENT_VALIDATE_PASS' and preflight['exit_code'] == 0
    snapshot = artifact/'code_snapshot'; snapshot.mkdir(exist_ok=False)
    for name in ('run_three_memory_codec_20261029_v33.py', 'supervise_three_memory_codec_20261029_v33.py',
                 'prepare_three_memory_worker_v33_cpu.py', 'review_three_memory_owner_v33_windows_cpu.py',
                 'finalize_three_memory_worker_v33_cpu.py'):
        shutil.copyfile(ROOT/'scripts'/name, snapshot/name)
    shutil.copyfile(ROOT/'tests/test_three_memory_worker_v33.py', snapshot/'test_three_memory_worker_v33.py')
    report = ROOT.parent/'outputs/FDT_THREE_MEMORY_WORKER_V33_READY_20261029_KR.md'
    shutil.copyfile(report, artifact/report.name)
    summary = {'status': 'ACTUAL_WORKER_AND_INDEPENDENT_OS_OWNER_CPU_READY_NOT_GPU_EXECUTED',
        'new_focused_distinct_CPU_tests': 7, 'resource_delta_retests': 5,
        'actual_model_CPU_preflight_independent_validate': True, 'real_Windows_CPU_fixture_all_owned_gone': True,
        'actual_GPU_launches': 0, 'model_forward_calls': 0, 'start_storage_contract': p['start_storage_contract'],
        'prospective_plan_SHA': owner.native.sha256(artifact/'plan_unsealed_r3.json'),
        'explicit_before_any_GPU_amendment_from_prospective_C40_to_calculated_C35': True,
        'historical_closed_limits_changed': False, 'source_originals_deleted_or_moved': False,
        'failure_r1_external_ledger_complete_provenance_fixed': True}
    (artifact/'root_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    data = {'kind': 'CPU_WORKER_EVIDENCE_NOT_GPU_SEAL', 'summary': summary,
        'files': {str(f.relative_to(artifact)).replace('\\', '/'): {'bytes': f.stat().st_size, 'sha256': owner.native.sha256(f)}
            for f in sorted(artifact.rglob('*')) if f.is_file()}}
    data['self_excluding_payload_sha256'] = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    with manifest.open('x', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
    print(json.dumps({'status': summary['status'], 'manifest_FILE_sha256': owner.native.sha256(manifest),
        'payload_sha256': data['self_excluding_payload_sha256'], 'files': len(data['files'])}))


if __name__ == '__main__': main()
