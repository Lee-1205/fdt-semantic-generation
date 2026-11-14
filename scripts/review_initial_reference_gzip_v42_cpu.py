"""ONE bounded saved-state storage probe, not a V42 worker/model trial.

Real parity uses the existing V41 INITIAL state and its preserved V40 parent.
An explicitly synthetic +4MiB metadata field exercises the old-cap boundary;
it is not a measurement of a future V42 engine export or training data.
"""
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'scripts')]
import codec_initial_reference_v39 as old
import codec_initial_reference_v42_gzip as new
import text_generation_support_v1 as support

ARTIFACT = ROOT/'research/artifacts/initial_reference_gzip_20261030_v42_cpu'
LIMITS = {'wall_seconds': 120, 'max_output_bytes': 64*1024**2,
          'model_calls': 0, 'actual_fullsize_model_construction': 0,
          'synthetic_extra_metadata_bytes': 4*1024**2}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8') as file:
        json.dump(value, file, ensure_ascii=False, indent=2)


def main():
    torch.set_num_threads(2)
    run = ROOT/'runs/three_memory_codec_20261030_v41'
    plan = read(run/'plan.json')
    source = Path(plan['source'])  # V40 parent of the actual V41 initial reference.
    source_sha, source_algorithm = plan['source_sha'], plan['source_algorithm']
    initial_path = run/'import/checkpoints/state_0000_reference.pt'
    marker = read(initial_path.with_suffix('.json'))
    if (ARTIFACT/'execution_started.json').exists():
        raise FileExistsError('CPU storage allocation already consumed')
    if read(run/'supervisor/final_status.json')['status'] != 'COMPLETE':
        raise ValueError('closed source only')
    source_plan = {'schema': 'CPU_GZIP_STORAGE_PARITY_ONLY_V42', 'limits': LIMITS,
        'actual_state': 'existing V41 initial, parent V40; NOT future V42 engine import',
        'source': str(source), 'source_sha256': source_sha,
        'initial_reference': str(initial_path), 'initial_reference_sha256': marker['sha256'],
        'helper_sha256': support.sha(Path(new.__file__)),
        'old_helper_sha256': support.sha(Path(old.__file__)),
        'script_sha256': support.sha(Path(__file__)), 'policy': new.POLICY,
        'all_prior_files_preserved': True, 'GPU_training': False}
    write(ARTIFACT/'plan.json', source_plan)
    started = time.monotonic()
    write(ARTIFACT/'execution_started.json', {'started_unix': time.time(),
        'plan_sha256': support.sha(ARTIFACT/'plan.json'), 'fixed_wall_seconds': 120})

    def check():
        if time.monotonic()-started > LIMITS['wall_seconds']:
            raise TimeoutError('fixed CPU storage deadline')
        total = sum(p.stat().st_size for p in ARTIFACT.rglob('*') if p.is_file())
        if total > LIMITS['max_output_bytes']:
            raise OSError('fixed CPU storage output cap')

    try:
        assert support.sha(source) == source_sha and support.sha(initial_path) == marker['sha256']
        parent = torch.load(source, weights_only=True, map_location='cpu')
        reference = torch.load(initial_path, weights_only=True, map_location='cpu')
        initial = old.materialize(reference, parent, source, source_sha, source_algorithm)
        assert support.state_sha(initial) == marker['algorithm_sha256']
        assert len(initial['optimizer']['state']) == 91 and initial['attempts'] == initial['cursor'] == 0
        check()
        rng = torch.get_rng_state().clone()
        restored, sizes = new.safe_memory_roundtrip(initial, parent, source, source_sha, source_algorithm)
        assert support.state_sha(restored) == marker['algorithm_sha256']
        assert torch.equal(torch.get_rng_state(), rng)
        check()
        # This extra field is explicitly synthetic; do not call it real next-run metadata.
        enlarged = dict(initial, CPU_STORAGE_ONLY_SYNTHETIC_EXTRA=b'x'*LIMITS['synthetic_extra_metadata_bytes'])
        enlarged_roundtrip, large_sizes = new.safe_memory_roundtrip(
            enlarged, parent, source, source_sha, source_algorithm)
        assert large_sizes['serialized_bytes'] > 32*1024**2
        assert large_sizes['compressed_bytes'] <= 32*1024**2
        assert support.state_sha(enlarged_roundtrip) == support.state_sha(enlarged)
        check()
        saved = new.save_initial_reference(ARTIFACT/'real_v41_initial_parity', initial,
                                          source, source_sha, source_algorithm)
        loaded = new.load_initial_reference(ARTIFACT/'real_v41_initial_parity/checkpoints'/saved['file'],
                                            source, source_sha, source_algorithm)
        assert support.state_sha(loaded) == marker['algorithm_sha256']
        assert torch.equal(torch.get_rng_state(), rng) and not torch.cuda.is_initialized()
        assert support.sha(source) == source_sha and support.sha(initial_path) == marker['sha256']
        assert support.core_sha(ROOT) == support.CORE_SHA
        check()
        result = {'status': 'CPU_GZIP_REFERENCE_STORAGE_PARITY_COMPLETE_NOT_V42_WORKER',
            'seconds': time.monotonic()-started, 'actual_existing_V41_initial_state_digest': marker['algorithm_sha256'],
            'actual_existing_V41_initial_gzip_sizes': sizes,
            'actual_existing_V41_initial_disk_roundtrip': saved,
            'synthetic_extra_metadata_bytes': LIMITS['synthetic_extra_metadata_bytes'],
            'synthetic_beyond_old_cap_gzip_sizes': large_sizes,
            'synthetic_case_is_NOT_future_V42_export_measurement': True,
            'fullsize_shared_model91Adam_scaler_ALLstoredRNG_exact': True,
            'global_Torch_RNG_unchanged': True, 'all_prior_logical_files_unchanged': True,
            'actual_fullsize_model_construction': 0, 'model_forward_calls': 0, 'GPU_calls': 0,
            'CUDA_initialized': False, 'actual_V42_fullsize_source_import_performed': False,
            'V42_worker_plan_owner_seal_launch_created': False,
            'next_required': 'Actual V41-final -> V42 engine import/export, enabled-scaler full-driver recovery, '
                             'actual worker preflight + independent owner validate with fresh caps before ONE new training.'}
        write(ARTIFACT/'root_cpu_parity.json', result)
        print(json.dumps({k: result[k] for k in ('status', 'seconds',
            'actual_existing_V41_initial_gzip_sizes', 'synthetic_beyond_old_cap_gzip_sizes',
            'actual_fullsize_model_construction', 'model_forward_calls', 'GPU_calls')}))
    except BaseException as exc:
        write(ARTIFACT/'incident.json', {'error': repr(exc), 'seconds': time.monotonic()-started,
            'model_calls': 0, 'all_candidates_preserved': True, 'old_budget_reused': False})
        raise


if __name__ == '__main__':
    main()
