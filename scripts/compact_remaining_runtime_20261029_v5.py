"""NEW bounded lossless remaining-runtime storage purpose; no model or GPU work.

Exact workspace Torch DLLs only, excluding the three already compressed in v2.
No checkpoints, source/core/archives/corpus/user files, deletion, move, download
or serialization change. Every logical length/SHA must remain identical.
"""
import ctypes
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from compact_owned_language_checkpoints_20261029_v4 import sha, allocated

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/artifacts/owt_runtime_lossless_space_20261029_v5'
MAX_FILES = 14; MAX_LOGICAL = 3*1024**3; WALL = 180; PER_FILE = 30
TARGET_FREE = int(40.5*1024**3)
NAMES = ('torch_cpu.dll', 'cusparse64_12.dll', 'cudnn_adv64_9.dll', 'cufft64_11.dll',
    'cusolver64_11.dll', 'cusolverMg64_11.dll', 'cudnn_ops64_9.dll', 'cublas64_12.dll',
    'nvrtc64_120_0.alt.dll', 'nvrtc64_120_0.dll', 'nvJitLink_120_0.dll', 'curand64_10.dll',
    'cudnn_heuristic64_9.dll', 'nvperf_host.dll')


def save(name, obj):
    target = OUT/name; temp = target.with_suffix(target.suffix+'.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8'); os.replace(temp, target)


def main():
    if os.name != 'nt' or OUT.exists(): raise RuntimeError('new Windows operation only')
    library = (ROOT/'.venv/Lib/site-packages/torch/lib').resolve(strict=True)
    paths = []
    for name in NAMES:
        p = library/name
        if not p.is_file() or p.is_symlink() or p.resolve(strict=True).parent != library: raise ValueError('exact installed runtime file')
        if allocated(p) < p.stat().st_size*.99: continue
        paths.append(p)
    if len(paths) > MAX_FILES or sum(p.stat().st_size for p in paths) > MAX_LOGICAL: raise ValueError('fixed scope cap')
    OUT.mkdir(); began = time.monotonic(); before = shutil.disk_usage(ROOT).free
    save('plan.json', {'kind': 'NEW_REMAINING_RUNTIME_LOSSLESS_SPACE_NOT_MODEL_ALLOCATION',
        'paths': [str(p) for p in paths], 'max_files': MAX_FILES, 'max_logical_bytes': MAX_LOGICAL,
        'wall_seconds': WALL, 'per_file_seconds': PER_FILE, 'target_C_free_bytes': TARGET_FREE,
        'method': 'compact.exe /c /exe:lzx /q EXACT_FILE',
        'official_reference': 'https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact',
        'already_compressed_excluded': ['torch_cuda.dll', 'cublasLt64_12.dll', 'cudnn_engines_precompiled64_9.dll'],
        'script_sha256': sha(Path(__file__)), 'C_free_before': before,
        'model_calls': 0, 'GPU_calls': 0, 'file_deletions': 0, 'file_moves': 0})
    rows = []; status = 'BOUNDED_COMPLETE'
    for path in paths:
        if shutil.disk_usage(ROOT).free >= TARGET_FREE: status = 'SPACE_TARGET_REACHED'; break
        if time.monotonic()-began+PER_FILE+10 >= WALL: status = 'FIXED_TIME_CAP'; break
        row = {'file': str(path), 'logical_before': path.stat().st_size, 'sha_before': sha(path), 'allocated_before': allocated(path)}
        rows.append(row); save('progress.json', {'phase': 'prepared', 'records': rows})
        child = None; error = None; started = time.monotonic()
        try:
            command = ['compact.exe', '/c', '/exe:lzx', '/q', str(path)]
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                shell=False, creationflags=subprocess.CREATE_NO_WINDOW)
            row.update(owned_pid=child.pid, command=command); save('progress.json', {'phase': 'owned_compact', 'records': rows})
            stdout, stderr = child.communicate(timeout=PER_FILE)
            row.update(exit_code=child.returncode, stdout_hex=stdout.hex(), stderr_hex=stderr.hex())
            if child.returncode: raise RuntimeError('compact nonzero exit')
        except BaseException as exc: error = repr(exc)
        finally:
            if child is not None:
                if child.poll() is None: child.kill()
                child.wait(timeout=10); row['owned_exit_confirmed'] = True
        row.update(logical_after=path.stat().st_size, sha_after=sha(path), allocated_after=allocated(path),
            error=error, seconds=time.monotonic()-started)
        row['logical_identical'] = row['logical_before'] == row['logical_after'] and row['sha_before'] == row['sha_after']
        save('progress.json', {'phase': 'verified', 'records': rows})
        print(json.dumps({'file': path.name, 'logical_identical': row['logical_identical'],
            'saved_allocated_bytes': row['allocated_before']-row['allocated_after'], 'C_free_bytes': shutil.disk_usage(ROOT).free}), flush=True)
        if error or not row['logical_identical']: status = 'STOP_ERROR_OR_MISMATCH'; break
    result = {'status': status, 'records': rows, 'files_processed': len(rows), 'seconds': time.monotonic()-began,
        'allocated_bytes_saved': sum(r['allocated_before']-r['allocated_after'] for r in rows),
        'all_logical_SHA_length_identical': all(r['logical_identical'] for r in rows),
        'all_owned_compact_gone': all(r.get('owned_exit_confirmed') for r in rows),
        'C_free_before': before, 'C_free_after': shutil.disk_usage(ROOT).free, 'target_C_free_bytes': TARGET_FREE,
        'model_calls': 0, 'GPU_calls': 0, 'file_deletions': 0, 'file_moves': 0}
    save('result.json', result); print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
    return 0 if result['all_logical_SHA_length_identical'] and result['all_owned_compact_gone'] and not status.startswith('STOP') else 2


if __name__ == '__main__': raise SystemExit(main())
