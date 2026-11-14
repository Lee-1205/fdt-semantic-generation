"""Actual V31 fullstate -> new codec driver CPU migration; no model forward."""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from research.three_memory_codec_v33 import ThreeMemoryCodec
from three_memory_codec_engine_v33 import CodecEngine, old_optimizer_view
from owt_resource_budget_v29 import RecoveryBudget, RESOURCE_POLICY
import text_generation_support_v1 as support


def main():
    started = time.monotonic(); torch.set_num_threads(2)
    artifact = ROOT/'research/artifacts/three_memory_codec_20261029_v33_cpu'
    record = sys.argv[1] if len(sys.argv) == 2 else 'root_driver_source_review.json'
    if record not in ('root_driver_source_review.json', 'root_driver_source_review_r2.json'):
        raise ValueError('fixed CPU review record name')
    output = artifact/record
    if output.exists(): raise FileExistsError(output)
    source = ROOT/'runs/owt_fuzzy_reconstruction_20261029_v31/model/checkpoints/state_2048.pt'
    source_sha = '1d02d0d52c44683a582e70273e3e951e84384ecb406700470af620e21e8c252f'
    assert support.sha(source) == source_sha
    state = torch.load(source, map_location='cpu', weights_only=True)
    before = support.state_sha(state)
    assert before == '66316e12eeeb9ae3d6c3515f5983deca0363f9d54f2166c1bc6210a0c604bf19'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(833); model = ThreeMemoryCodec(Config(**state['core']))
    policy = {'output': str(artifact), 'cutoff': time.time()+180, 'resource_policy': RESOURCE_POLICY,
        'external_closed_assessment_calls': state['external_closed_assessment_calls'],
        'limits': {'max_calls': 3, 'max_output_bytes': 1024**3}}
    # A declaration for constructor validation ONLY, not a dataset or trial.
    features = [{'id': 'migration-fixture-only', 'prompt': 'CPU migration only'}]
    targets = [{'id': 'migration-fixture-only', 'target': 'CPU migration only'}]
    engine = CodecEngine(model, features, targets, [[0]], RecoveryBudget(policy), 'CPU_SOURCE_REVIEW_NO_GPU_PLAN', 'cpu',
        source_parameter_groups=state['optimizer_parameter_names'], data_revision='CPU_DECLARATION_NOT_CORPUS')
    migration = engine.import_parent(state, state['plan_sha256'], source_sha)
    exported = engine.export()
    assert all(torch.equal(exported['model'][n], v) for n, v in state['model'].items())
    assert support.state_sha(old_optimizer_view(exported['optimizer'])) == support.state_sha(state['optimizer'])
    assert len(exported['optimizer']['state']) == 81 and list(map(len, engine.names)) == [19, 44, 18, 10]
    for key in ('scaler', 'frozen_sha', 'torch_rng', 'cuda_rng', 'python_rng', 'numpy_rng'):
        assert support.state_sha(exported[key]) == support.state_sha(state[key])
    assert exported['attempts'] == exported['updates'] == exported['cursor'] == 0
    assert exported['allocation_calls'] == {'encode': 0, 'denoise': 0, 'decoder': 0}
    assert exported['actual_calls'] == state['actual_calls']
    assert support.state_sha(state) == before and support.sha(source) == source_sha
    engine.verify_frozen(); assert not torch.cuda.is_initialized()
    seconds = time.monotonic()-started; assert seconds < 180
    result = {'status': 'ACTUAL_SOURCE_CPU_DRIVER_MIGRATION_PASS_NO_FORWARD_NO_GPU',
        'source_sha256': source_sha, 'source_algorithm_before_after': before,
        'all_old_model_tensors_exact': len(state['model']), 'old_populated_Adam_tensors_exact': 81,
        'new_named_tensors': engine.names[3], 'new_Adam_states_at_import': 0,
        'optimizer_group_counts': list(map(len, engine.names)),
        'source_scaler_frozen_all_stored_RNG_exact': True, 'migration': migration,
        'allocation_calls': exported['allocation_calls'], 'source_actual_calls_retained': exported['actual_calls'],
        'candidate_algorithm_after_import': support.state_sha(exported),
        'actual_model_forward': 0, 'actual_optimizer_steps': 0, 'CUDA_initialized': False,
        'actual_GPU_fullrestore_replay': False, 'actual_new_fullstate_disk_serialization': False,
        'tiny_driver_fullstate_serialization_exact_next_including_AMPskip_last7': 'driver_cpu_r3.xml',
        'worker_plan_OSpreflight_seal_launch': 'NOT_IMPLEMENTED',
        'candidate_sha256': support.sha(ROOT/'research/three_memory_codec_v33.py'),
        'engine_sha256': support.sha(ROOT/'scripts/three_memory_codec_engine_v33.py'), 'seconds': seconds}
    with output.open('x', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ('status', 'all_old_model_tensors_exact', 'old_populated_Adam_tensors_exact', 'new_Adam_states_at_import', 'seconds')}))


if __name__ == '__main__': main()
