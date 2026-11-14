"""NEW11th/12th epoch source-bound driver, gzip initial + owned CPU preflight.

Never edit earlier sealed code, clocks, checkpoints, or the frozen gzip helper.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPLACE = {
    'three_memory_codec_20261030_v41': 'three_memory_codec_20261030_v42',
    'three_memory_codec_20261029_v40': 'three_memory_codec_20261030_v41',
    'build_v41_delta_20261030': 'build_v42_delta_20261030',
    'V40_COMPLETE_SEVENTH_EIGHTH_BROAD_EPOCHS_ROOT_REVIEWED': 'V41_COMPLETE_NINTH_TENTH_BROAD_EPOCHS_ROOT_REVIEWED',
    'codec_initial_reference_v39': 'codec_initial_reference_v42_gzip',
    'codec_single_snapshot_v40': 'codec_single_snapshot_v42',
    'v41': 'v42', 'V41': 'V42', 'v40': 'v41', 'V40': 'V41',
    '40001': '41001', '40002': '41002', '39001': '40001',
    '137188': '145380', '137138': '145326',
    'fbaf29eca2c5c9e1df12e3ea71702aeee40c0bda5c7f9ec73cba339acd4cd677': 'f242fc902bd65f331cde625378bf832cbb54c5e26ef4842cf6352ec64fb48887',
    '00aca0d669b2228c49fd075ca45a9425ed655d7ce40857bc0a171c1905635a70': '7205572abeff36b18d202d652749ae0ef636e476f877593e3372265b119584d8',
    '3262ad6fd5e6f9f38552c4ee1207735e76f42a9f61cc8c005d01179b65aa4199': 'b74eb3f6f7137c5967fe1d6a53d54e49ea3f97ca9fd10c28568cdc26ef6d201e',
    '(8192,8189,65536)': '(8192,8188,65536)',
    'torch.manual_seed(841)': 'torch.manual_seed(842)',
    'learning_epoch_number=9': 'learning_epoch_number=11',
}
PREFLIGHT_POLICY = {'worker_seconds_max': 90, 'termination_verification_reserve_seconds': 30,
    'total_seconds_max': 120, 'independent_OS_hard_timeout': True,
    'all_postPopen_exceptions_verified_owned_cleanup': True,
    'unconfirmed_exit_blocks_model_launch': True, 'one_preflight_per_output': True}
INITIAL_POLICY = {'format': 'fdt_initial_parent_reference_v42_gzip_v1', 'initial_only': True,
    'parent_required': True, 'shared_fields': ['model','optimizer','optimizer_parameter_names','scaler',
        'frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng'],
    'shared_state_must_be_identical': True, 'updated_checkpoints': 'self_contained', 'quantization': False,
    'payload_reference_schema': 'fdt_initial_parent_reference_v39', 'compression': 'gzip',
    'maximum_compressed_bytes': 32*1024**2, 'maximum_serialized_bytes': 64*1024**2, 'gzip_level': 6,
    'concatenated_or_trailing_data': 'reject', 'bounded_decompression_and_CRC_required': True}


def main():
    pattern = re.compile('|'.join(re.escape(k) for k in sorted(REPLACE, key=len, reverse=True)))
    sources = ['scripts/three_memory_codec_engine_v41.py', 'scripts/run_three_memory_codec_20261030_v41.py',
        'scripts/supervise_three_memory_codec_20261030_v41.py', 'scripts/prepare_three_memory_worker_v41_cpu.py',
        'scripts/review_three_memory_owner_v41_windows_cpu.py', 'scripts/seal_three_memory_codec_20261030_v41.py',
        'tests/test_three_memory_worker_v41.py']
    for name in sources:
        dest = pattern.sub(lambda m: REPLACE[m.group()], name)
        text = pattern.sub(lambda m: REPLACE[m.group()], (ROOT/name).read_text(encoding='utf-8-sig'))
        text = text.replace('codec_single_snapshot_v42', 'codec_single_snapshot_v42')
        text = text.replace("'unchanged_parent_reference_v39'", "'lossless_gzip_parent_reference_v42'")
        if '/run_three_memory_codec_' in name or '/supervise_three_memory_codec_' in name:
            text = text.replace("'startup_seconds':90", "'startup_seconds':120").replace("'startup_seconds': 90", "'startup_seconds': 120")
            text = text.replace("'calculated_total_seconds':2029", "'calculated_total_seconds':2059")
            text = text.replace('1639+90+60+60+180!=2029', '1639+120+60+60+180!=2059')
            text = text.replace("ROOT = Path(__file__).resolve().parents[1]", "ROOT = Path(__file__).resolve().parents[1]\nCPU_PREFLIGHT_POLICY = "+repr(PREFLIGHT_POLICY))
            if '/run_' in name:  # compact worker ROOT assignment has no spaces
                text = text.replace("ROOT=Path(__file__).resolve().parents[1]", "ROOT=Path(__file__).resolve().parents[1]\nCPU_PREFLIGHT_POLICY="+repr(PREFLIGHT_POLICY))
                text = text.replace('import argparse,ctypes,sys,time,traceback', 'import argparse,ctypes,json,sys,time,traceback')
                text = text.replace("('single_snapshot_contract',snapshot_storage.POLICY)]", "('single_snapshot_contract',snapshot_storage.POLICY),('cpu_preflight_policy',CPU_PREFLIGHT_POLICY)]")
                old = "restored,size=checkpoint_storage.safe_memory_roundtrip(engine.export(),state,p['source'],SOURCE_SHA,SOURCE_ALGORITHM)\n        if support.state_sha(restored)!=support.state_sha(engine.export()):raise ValueError('actual initial reference fullstate roundtrip')\n        report.update(actual_fullsize_initial_reference_bytes=size,actual_initial_reference_CPU_roundtrip=True)"
                new = "restored,sizes=checkpoint_storage.safe_memory_roundtrip(engine.export(),state,p['source'],SOURCE_SHA,SOURCE_ALGORITHM)\n        if restored['attempts']!=0 or restored['plan_sha256']!=plan_sha:raise ValueError('initial reference allocation identity')\n        report.update(actual_fullsize_initial_reference_bytes=sizes['compressed_bytes'],actual_fullsize_initial_reference_serialized_bytes=sizes['serialized_bytes'],actual_initial_reference_CPU_roundtrip=True,initial_reference_checkpoint_format=checkpoint_storage.SCHEMA)"
                assert old in text; text = text.replace(old, new)
                text = text.replace("load_import(p,f,y,orders,RecoveryBudget(p),support.sha(a.plan),'cpu')", "engine,report=load_import(p,f,y,orders,RecoveryBudget(p),support.sha(a.plan),'cpu')")
                text = text.replace("print('CPU_PREFLIGHT_PASS');return 0", "print('CPU_PREFLIGHT_REPORT '+json.dumps(report));print('CPU_PREFLIGHT_PASS');return 0")
            else:
                begin = text.index('INITIAL_CHECKPOINT_CONTRACT='); end = text.index('\nSTORAGE_BOUND=', begin)
                text = text[:begin]+'INITIAL_CHECKPOINT_CONTRACT='+repr(INITIAL_POLICY)+text[end:]
                begin = text.index('def preflight('); end = text.index('\n\ndef duplicates', begin)
                text = text[:begin]+"def preflight(plan_path, output, worker, seconds):\n    from owned_cpu_preflight_v42 import preflight as checked\n    return checked(plan_path,output,worker,seconds)\n"+text[end:]
                text = text.replace("plan.get('time_cost_contract')!=TIME_COST", "plan.get('cpu_preflight_policy')!=CPU_PREFLIGHT_POLICY or plan.get('time_cost_contract')!=TIME_COST")
                text = text.replace("min(90, plan['cutoff']-time.time())", "min(120, plan['cutoff']-time.time())")
                text = text.replace('preflight(args.plan, args.output, args.worker, 90)', 'preflight(args.plan, args.output, args.worker, 120)')
        if 'prepare_three_memory_worker' in name:
            text = text.replace("'scripts/codec_single_snapshot_v42.py')", "'scripts/codec_single_snapshot_v42.py','scripts/owned_cpu_preflight_v42.py','tests/test_owned_cpu_preflight_v42.py')")
            text = text.replace('single_snapshot_contract=worker.snapshot_storage.POLICY,', 'single_snapshot_contract=worker.snapshot_storage.POLICY,cpu_preflight_policy=worker.CPU_PREFLIGHT_POLICY,')
            text = text.replace("source=old/'model/checkpoints/state_8192.pt'", "storage_manifest=ROOT/'research/artifacts/initial_reference_gzip_20261030_v42_cpu_ready/manifest.json'\n    files[str(storage_manifest)]=support.sha(storage_manifest)\n    source=old/'model/checkpoints/state_8192.pt'")
        if 'test_three_memory_worker' in name:
            text = text.replace("'initial_checkpoint_contract'", "'initial_checkpoint_contract'")
            text = text.replace('1639+90+60+60+180==2029', '1639+120+60+60+180==2059')
            text = text.replace('reference.make_reference(', 'reference.base.make_reference(').replace('reference.materialize(', 'reference.base.materialize(')
        if 'seal_three_memory' in name:
            text = text.replace("suite=ET.parse(a/'cpu_delta.xml').getroot().find('testsuite')", "suite=ET.parse(a/'cpu_delta.xml').getroot().find('testsuite')\n    pretests=ET.parse(a/'preflight_cpu.xml').getroot().find('testsuite')\n    if int(pretests.attrib['tests'])!=6 or int(pretests.attrib['failures']) or int(pretests.attrib['errors']):raise ValueError('owned CPU preflight contracts')")
            text = text.replace("'CPU_tests':12", "'CPU_tests':18")
        with (ROOT/dest).open('x', encoding='utf-8', newline='\n') as file:
            file.write(text)
    print('V42_NEW_GZIP_SOURCE_BOUND_DRIVER_FILES_ONLY_NO_GPU')


if __name__ == '__main__':
    main()
