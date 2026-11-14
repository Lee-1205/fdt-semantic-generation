"""ROOT source inventory/warmstart check, no actual parent/candidate forward."""
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from research.three_memory_codec_v33 import ThreeMemoryCodec
from research.three_memory_contract_v32 import assert_codec_training_only
import text_generation_support_v1 as support


def main():
    start = time.monotonic(); torch.set_num_threads(2)
    artifact = ROOT/'research/artifacts/three_memory_codec_20261029_v33_cpu'
    output = artifact/'root_source_review.json'
    if output.exists(): raise FileExistsError(output)
    source = ROOT/'runs/owt_fuzzy_reconstruction_20261029_v31/model/checkpoints/state_2048.pt'
    expected_sha = '1d02d0d52c44683a582e70273e3e951e84384ecb406700470af620e21e8c252f'
    assert support.sha(source) == expected_sha
    state = torch.load(source, map_location='cpu', weights_only=True)
    assert support.state_sha(state) == '66316e12eeeb9ae3d6c3515f5983deca0363f9d54f2166c1bc6210a0c604bf19'
    before = support.state_sha(state)
    rng_before = support.state_sha({'torch': torch.get_rng_state(), 'python': random.getstate(),
        'numpy': tuple(str(x) for x in np.random.get_state())})
    # Scoped construction initializes ONLY this prospective branch; no global
    # training RNG is silently advanced and no parent optimizer is restored.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(833)
        model = ThreeMemoryCodec(Config())
        added = model.import_legacy_weights(state['model'])
    rng_after = support.state_sha({'torch': torch.get_rng_state(), 'python': random.getstate(),
        'numpy': tuple(str(x) for x in np.random.get_state())})
    assert rng_before == rng_after
    assert all(torch.equal(model.state_dict()[n], v) for n, v in state['model'].items())
    ownership = state['optimizer_parameter_names']
    old_names = [n for group in ownership for n in group]
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    new_names = [n for n in trainable if n not in set(old_names)]
    assert len(old_names) == len(set(old_names)) == 81
    assert set(trainable) == set(old_names) | set(new_names)
    assert set(added) == set(new_names) and all(n.startswith('memory_codec.') for n in new_names)
    assert len(state['optimizer']['state']) == 81
    assert_codec_training_only(model.named_parameters(), 0)
    assert support.state_sha(state) == before and support.sha(source) == expected_sha
    assert not torch.cuda.is_initialized()
    elapsed = time.monotonic()-start
    assert elapsed < 180
    result = {
        'status': 'CPU_CANDIDATE_SOURCE_STRICT_COPY_PASS_NOT_GPU_EXECUTION',
        'source': str(source), 'source_sha256': expected_sha, 'source_algorithm_before_after': before,
        'source_model_tensors_exact': len(state['model']), 'old_populated_Adam_tensors_preserved_in_source': len(old_names),
        'old_Adam_import_into_new_driver': 'NOT_IMPLEMENTED_YET',
        'old_optimizer_group_counts': [len(x) for x in ownership],
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in trainable.values()),
        'added_parameters': sum(trainable[n].numel() for n in new_names),
        'trainable_tensor_count': len(trainable),
        'new_named_tensors': {n: {'shape': list(trainable[n].shape), 'numel': trainable[n].numel()} for n in new_names},
        'caller_Python_numpy_Torch_rng_unchanged': rng_before == rng_after,
        'source_actual_model_forward': 0, 'candidate_actual_model_forward': 0,
        'tiny_contract_tests_not_actual_model_quality': True,
        'CUDA_initialized': torch.cuda.is_initialized(), 'seconds': elapsed,
        'candidate_sha256': support.sha(ROOT/'research/three_memory_codec_v33.py'),
        'test_sha256': support.sha(ROOT/'tests/test_three_memory_codec_v33.py'),
        'limitations': ['prior-only persistent memory; certified Writer absent',
            'input-complete token contextual snapshot, not full per-layer prefix read/write',
            'Exact learned index connected but new heads UNTRAINED; literal selector/copy intent absent',
            'Commit-only realizer and zero copy bytes; no automatic whole-source copying',
            'no denoising, new worker/plan/seal/GPU or performance improvement',
            'full new optimizer/scaler/RNG/cursor/worker restore not implemented'],
    }
    with output.open('x', encoding='utf-8') as f: json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ('status', 'total_parameters', 'trainable_parameters', 'added_parameters', 'trainable_tensor_count', 'seconds')}))


if __name__ == '__main__': main()
