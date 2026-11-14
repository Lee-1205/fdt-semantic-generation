"""Snapshot CPU code/evidence only; never authorizes or launches GPU work."""
import json
import hashlib
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import text_generation_support_v1 as support


def main():
    artifact = ROOT/'research/artifacts/three_memory_codec_20261029_v33_cpu'
    manifest = artifact/'manifest.json'
    if manifest.exists(): raise FileExistsError(manifest)
    source_review = json.loads((artifact/'root_source_review.json').read_text(encoding='utf-8'))
    driver_review = json.loads((artifact/'root_driver_source_review_r2.json').read_text(encoding='utf-8'))
    assert source_review['candidate_sha256'] == driver_review['candidate_sha256'] == support.sha(ROOT/'research/three_memory_codec_v33.py')
    assert driver_review['engine_sha256'] == support.sha(ROOT/'scripts/three_memory_codec_engine_v33.py')
    tests = []
    for filename, count in (('focused_cpu_r2.xml', 7), ('driver_cpu_r3.xml', 2)):
        suites = ET.parse(artifact/filename).getroot().findall('testsuite')
        assert sum(int(s.attrib['tests']) for s in suites) == count
        assert sum(int(s.attrib['failures'])+int(s.attrib['errors']) for s in suites) == 0
        tests.append({'record': filename, 'passed': count, 'seconds': sum(float(s.attrib['time']) for s in suites)})
    original_core = support.core_sha(ROOT)
    assert original_core == 'eb68ced289218c5a9ab1fc0481f52106da9f8c5892cb67022e70759d24a4f352'
    model_file_sha = support.sha(ROOT/'src/fdt_local/model_masked.py')
    source_plan = json.loads((ROOT/'runs/owt_fuzzy_reconstruction_20261029_v31/plan.json').read_text(encoding='utf-8'))
    assert model_file_sha == source_plan['files'][str(ROOT/'src/fdt_local/model_masked.py')]
    native = json.loads((artifact/'root_terminal_native_and_space.json').read_text(encoding='utf-8'))
    assert native['native'] == [] and native['GPU_new_launches'] == 0
    files = ['research/three_memory_codec_v33.py', 'tests/test_three_memory_codec_v33.py',
        'scripts/three_memory_codec_engine_v33.py', 'tests/test_three_memory_codec_engine_v33.py',
        'scripts/review_three_memory_codec_v33_cpu.py', 'scripts/review_three_memory_driver_source_v33_cpu.py',
        'scripts/finalize_three_memory_codec_v33_cpu.py']
    snapshot = artifact/'code_snapshot'; snapshot.mkdir(exist_ok=False)
    for name in files: shutil.copyfile(ROOT/name, snapshot/Path(name).name)
    report = ROOT.parent/'outputs/FDT_THREE_MEMORY_CODEC_V33_CPU_READY_20261029_KR.md'
    shutil.copyfile(report, artifact/report.name)
    summary = {'status': 'NEURAL_THREE_MEMORY_CODEC_AND_DRIVER_CPU_READY_NOT_GPU_TRAINED',
        'phase': 'language_meaning_language_codec_only', 'denoise_calls': 0, 'copy_bytes': 0,
        'focused_distinct_CPU_tests': 9, 'test_records': tests,
        'actual_source_forward': 0, 'actual_candidate_forward': 0, 'new_GPU_launches': 0,
        'parameter_counts': {k: source_review[k] for k in ('total_parameters', 'trainable_parameters', 'added_parameters', 'trainable_tensor_count')},
        'old_model_tensors_exact': 264, 'old_populated_Adam_tensors_exact': 81,
        'new_Adam_tensors_initially_populated': 0,
        'source_sha256': driver_review['source_sha256'],
        'source_algorithm_before_after': driver_review['source_algorithm_before_after'],
        'original_core_sha256': original_core,
        'original_model_masked_FILE_sha256': model_file_sha,
        'source_actual_calls_preserved': driver_review['source_actual_calls_retained'],
        'new_allocation_calls': driver_review['allocation_calls'],
        'full_vNext_certified_persistent_writer': False, 'full_layer_prefix_memory_integration': False,
        'learned_literal_copy_selector_or_intent': False, 'automatic_whole_source_copy': False,
        'GPU_performance_or_general_meaning_proof': False,
        'next': 'new worker/plan-schema/actual CPU preflight/independent OS owner and separate disk-headroom purpose before bounded codec GPU',
        'start_floor_currently_met': native['C_free_bytes'] >= native['start_floor_bytes']}
    with (artifact/'root_summary.json').open('x', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    references = ['src/fdt_local/model_masked.py', 'research/three_memory_contract_v32.py',
        'research/owt_extraction_tail_v29.py', 'research/owt_prefix_hold_v28.py',
        'research/owt_prefix_reconstruction_v19.py', 'research/owt_fuzzy_reconstruction_v12.py',
        'research/text_context_decoder_v2.py', 'research/text_diffusion_v1.py',
        'scripts/owt_fuzzy_engine_v17_base.py', 'scripts/owt_resource_budget_v29.py',
        'scripts/text_generation_support_v1.py']
    data = {'kind': 'CPU_CODE_AND_EVIDENCE_SNAPSHOT_NOT_EXECUTION_PLAN',
        'summary': summary,
        'references': {name: support.sha(ROOT/name) for name in references},
        'files': {p.relative_to(artifact).as_posix(): {'bytes': p.stat().st_size, 'sha256': support.sha(p)}
                  for p in sorted(artifact.rglob('*')) if p.is_file()}}
    data['self_excluding_payload_sha256'] = hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest()
    with manifest.open('x', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
    for name, record in data['files'].items():
        p = artifact/name
        assert p.stat().st_size == record['bytes'] and support.sha(p) == record['sha256']
    print(json.dumps({'status': summary['status'], 'manifest_FILE_sha256': support.sha(manifest),
        'payload_sha256': data['self_excluding_payload_sha256'], 'files': len(data['files']),
        'tests': summary['focused_distinct_CPU_tests'], 'start_floor_met': summary['start_floor_currently_met']}))


if __name__ == '__main__': main()
