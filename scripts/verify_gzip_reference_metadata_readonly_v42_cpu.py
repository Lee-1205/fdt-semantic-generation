"""New90sec owned CPU READ-ONLY proof; no repeated model/fullstate hashing.

Compare every old/new reference payload field, binding the preserved parent by
file SHA. Do not instantiate/load parent model tensors, rewrite candidates,
recompress, repeat synthetic cases, or reuse the earlier failed120sec clock.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import supervise_querycount_partial9_20261027_v3 as native
import learning_io_v2 as io

ARTIFACT = ROOT/'research/artifacts/initial_reference_gzip_20261030_v42_readonly_cpu'


def worker(plan_path, output):
    plan = io.load_json(plan_path)
    if native.sha256(Path(__file__)) != plan['script_sha256']:
        raise ValueError('pinned readonly worker')
    io.atomic_json(output/'progress.json', {'native_pid': os.getpid(), 'parent_pid': os.getppid(),
                                           'phase': 'REFERENCE_PAYLOAD_COMPARE_CPU_ONLY'})
    import torch
    import codec_initial_reference_v42_gzip as storage
    assert native.sha256(Path(storage.__file__)) == plan['helper_sha256']
    torch.set_num_threads(2)
    old_path, new_path = Path(plan['old_reference']), Path(plan['gzip_reference'])
    assert native.sha256(old_path) == plan['old_reference_sha256']
    assert native.sha256(new_path) == plan['gzip_reference_sha256']
    rng = torch.get_rng_state().clone()
    old = torch.load(old_path, weights_only=True, map_location='cpu')
    new, serialized_bytes = storage._unpack(storage._read_blob(new_path))
    counters = {'nodes_compared': 0, 'metadata_tensors_compared': 0}

    def same(left, right):
        counters['nodes_compared'] += 1
        if type(left) is not type(right):
            return False
        if isinstance(left, torch.Tensor):
            counters['metadata_tensors_compared'] += 1
            return left.dtype == right.dtype and torch.equal(left, right)
        if isinstance(left, dict):
            return left.keys() == right.keys() and all(same(v, right[k]) for k, v in left.items())
        if isinstance(left, (tuple, list)):
            return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
        return left == right

    assert same(old, new)
    assert new['source_sha256'] == plan['parent_sha256']
    assert new['reconstructed_algorithm_sha256'] == plan['fullstate_algorithm_sha256']
    marker = io.load_json(new_path.with_suffix('.json'))
    assert serialized_bytes == marker['serialized_bytes']
    assert marker['bytes'] == new_path.stat().st_size
    assert native.sha256(Path(plan['parent'])) == plan['parent_sha256']
    assert native.sha256(old_path) == plan['old_reference_sha256']
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    io.atomic_json(output/'result.json', {'status': 'REFERENCE_METADATA_READONLY_VERIFIED',
        **counters, 'old_reference_bytes': old_path.stat().st_size,
        'new_serialized_bytes': serialized_bytes, 'new_compressed_bytes': new_path.stat().st_size,
        'every_reference_payload_field_identical': True, 'parent_file_SHA_verified': True,
        'reconstructed_fullstate_algorithm_sha256': plan['fullstate_algorithm_sha256'],
        'parent_model_tensors_loaded': False, 'model_construction_forward': 0,
        'global_Torch_RNG_unchanged': True, 'CUDA_initialized': False, 'GPU_calls': 0,
        'candidate_or_prior_files_changed': False, 'actual_V42_worker_import_tested': False})
    return 0


def owner():
    # This OS owner intentionally imports no Torch and uses literal CPU-only caps.
    ARTIFACT.mkdir(exist_ok=False)
    previous = ROOT/'research/artifacts/initial_reference_gzip_20261030_v42_cpu'
    old_run = ROOT/'runs/three_memory_codec_20261030_v41'
    old_path = old_run/'import/checkpoints/state_0000_reference.pt'
    new_path = previous/'real_v41_initial_parity/checkpoints/state_0000_reference_gzip.pt'
    marker = io.load_json(new_path.with_suffix('.json'))
    start = time.time()
    plan = {'schema': 'CPU_READONLY_REFERENCE_IDENTITY_NEW_PURPOSE', 'wall_seconds': 90,
        'worker_cutoff': start+75, 'deadline': start+90, 'max_output_bytes': 1024**2,
        'script_sha256': native.sha256(Path(__file__)),
        'helper_sha256': native.sha256(ROOT/'scripts/codec_initial_reference_v42_gzip.py'),
        'old_reference': str(old_path), 'old_reference_sha256': native.sha256(old_path),
        'gzip_reference': str(new_path), 'gzip_reference_sha256': marker['sha256'],
        'parent': marker['source_path'], 'parent_sha256': marker['source_sha256'],
        'fullstate_algorithm_sha256': marker['algorithm_sha256'],
        'old_failed_allocation_reopened': False, 'no_new_compression_or_checkpoint_write': True}
    plan_path = ARTIFACT/'plan.json'; io.atomic_json(plan_path, plan)
    child = actual = None; reason = None
    state = {'status': 'PREPARING_CPU_ONLY', 'owner_pid': os.getpid(), 'started_unix': start,
             'plan_sha256': native.sha256(plan_path), 'launches': 0}
    stdout = stderr = None
    try:
        if native.sha256(Path(plan['parent'])) != plan['parent_sha256']:
            raise ValueError('preserved parent file SHA')
        stdout = (ARTIFACT/'cpu_worker.stdout.log').open('xb')
        stderr = (ARTIFACT/'cpu_worker.stderr.log').open('xb')
        command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__).resolve()),
                   '--worker', '--plan', str(plan_path), '--output', str(ARTIFACT)]
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout,
                                 stderr=stderr, creationflags=native._no_window(), shell=False)
        state.update(status='RUNNING_CPU_ONLY', launches=1, worker_proxy_pid=child.pid, command=command)
        io.atomic_json(ARTIFACT/'state.json', state)
        while child.poll() is None:
            if time.time() >= plan['worker_cutoff'] or (ARTIFACT/'STOP').exists():
                reason = 'fixed CPU cutoff or cooperative STOP'; break
            progress = io.load_json(ARTIFACT/'progress.json') or {}
            pid = progress.get('native_pid')
            if pid and actual is None and int(pid) != child.pid and child.poll() is None:
                actual = native.process_identity(int(pid), child.pid, Path(__file__), plan_path, ARTIFACT)
                state['actual'] = actual
            if sum(p.stat().st_size for p in ARTIFACT.rglob('*') if p.is_file()) > plan['max_output_bytes']:
                reason = 'CPU output cap'; break
            time.sleep(.25)
    except BaseException as exc:
        reason = repr(exc); state['traceback'] = traceback.format_exc()
    finally:
        if child is not None:
            state['exit'] = native.verify_exit(child, Path(__file__), plan_path, ARTIFACT, ROOT,
                                               actual, terminate=child.poll() is None)
            result = io.load_json(ARTIFACT/'result.json') or {}
            if not state['exit']['all_owned_processes_gone']:
                state['status'] = 'UNCONFIRMED_EXIT'
            elif reason:
                state['status'] = 'STOPPED'
            elif child.returncode == 0 and result.get('status') == 'REFERENCE_METADATA_READONLY_VERIFIED':
                state['status'] = 'COMPLETE'
            else:
                state['status'] = 'FAILED'
            state['worker_exit_code'] = child.poll()
        else:
            state['status'] = 'FAILED_BEFORE_POPEN'
        if stdout: stdout.close()
        if stderr: stderr.close()
        state.update(reason=reason, seconds=time.time()-start, model_GPU_calls=0,
                     old_failed_allocation_reopened=False, owner_imported_Torch='torch' in sys.modules)
        io.atomic_json(ARTIFACT/'final_status.json', state)
    print(json.dumps(state))
    return 0 if state['status'] == 'COMPLETE' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    raise SystemExit(worker(args.plan, args.output) if args.worker else owner())
