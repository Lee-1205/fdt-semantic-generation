"""ONE bounded NTFS allocation for closed typed intermediate checkpoints.

Preserve every logical byte and marker. Exclude each run's terminal checkpoint,
all QueryCount partial9 states, the current V39 source, and all user/source data.
This new purpose does not reopen any previous compression or model allocation.
"""
import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import compact_closed_logits_20261029_v8 as helpers
import supervise_three_memory_codec_20261029_v40 as owner

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/artifacts/typed_ancestor_lossless_space_20261030_v9'
SELECTION = {
    'query_mix_diffusion_20261027_v1': (0, 65, 130, 195, 257, 258, 259, 260, 261),
    'role8_diffusion_20261027_v1': (0, 126, 127, 128, 129),
}
LIMITS = {'max_files': 14, 'max_logical_bytes': 8*1024**3,
          'wall_seconds': 300, 'per_file_seconds': 45,
          'target_C_free_bytes': int(30.25*1024**3),
          'protected_C_free_bytes': 20*1024**3}


def save(name, value):
    path = OUT/name
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def checked(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError('resolved absolute checkpoint only')
    rel = path.relative_to(ROOT/'runs')
    if len(rel.parts) != 3 or rel.parts[0] not in SELECTION or rel.parts[1] != 'model':
        raise ValueError('outside exact closed model folders')
    allowed = {f'payload_{g:06d}.pt' for g in SELECTION[rel.parts[0]]}
    if path.name not in allowed:
        raise ValueError('outside enumerated intermediate files')
    terminal = helpers.read(path.parent.parent/'final_status.json')
    if terminal['status'] != 'COMPLETE' or not terminal['temporary_sleep_prevention_released']:
        raise ValueError('terminal complete evidence required')
    marker = path.with_name(path.name.replace('payload_', 'complete_')).with_suffix('.json')
    if marker.resolve() == Path(terminal['last_marker']).resolve():
        raise ValueError('terminal checkpoint protected')
    meta = helpers.read(marker)
    if meta['file'] != path.name or meta['bytes'] != path.stat().st_size:
        raise ValueError('original complete marker identity')
    return path, meta


def ensure_idle():
    if owner.duplicates(None) or owner.native.process_inventory(ROOT):
        raise RuntimeError('model worker active; no storage work while training')


def prepare():
    if os.name != 'nt' or OUT.exists():
        raise FileExistsError('new Windows storage allocation only')
    ensure_idle()
    rows = []
    for rn, generations in SELECTION.items():
        for generation in generations:
            path, marker = checked(ROOT/'runs'/rn/'model'/f'payload_{generation:06d}.pt')
            if helpers.allocated(path) < marker['bytes']:
                continue
            rows.append({'path': str(path), 'logical_bytes': marker['bytes'],
                         'original_sha256': marker['sha256']})
    if len(rows) > LIMITS['max_files'] or sum(r['logical_bytes'] for r in rows) > LIMITS['max_logical_bytes']:
        raise ValueError('fixed scope cap')
    OUT.mkdir(exist_ok=False)
    plan = {'schema': 'typed_intermediate_NTFS_lossless_v9', 'limits': LIMITS,
            'paths': rows, 'script_sha256': helpers.sha(Path(__file__)),
            'helper_sha256': helpers.sha(Path(helpers.__file__)),
            'method': 'compact.exe /c /exe:lzx /q EXACT_FILE',
            'official_reference': 'https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact',
            'planned_logical_bytes': sum(r['logical_bytes'] for r in rows),
            'C_free_before': shutil.disk_usage(ROOT).free,
            'terminal_payloads_protected': True, 'QueryCount_partial9_untouched': True,
            'current_V39_source_untouched': True, 'file_deletions': 0,
            'file_moves': 0, 'model_calls': 0, 'old_allocation_reopened': False}
    save('plan.json', plan)
    print(json.dumps({k: v for k, v in plan.items() if k != 'paths'}), flush=True)


def execute():
    plan = helpers.read(OUT/'plan.json')
    if (plan['limits'] != LIMITS or plan['script_sha256'] != helpers.sha(Path(__file__))
            or plan['helper_sha256'] != helpers.sha(Path(helpers.__file__))):
        raise ValueError('fixed code and caps')
    if (OUT/'execution_started.json').exists():
        raise FileExistsError('closed allocation never replayed')
    ensure_idle()
    for row in plan['paths']:
        checked(row['path'])
    started = time.monotonic()
    before = shutil.disk_usage(ROOT).free
    save('execution_started.json', {'unix': time.time(), 'wall_seconds': LIMITS['wall_seconds'],
                                    'plan_sha256': helpers.sha(OUT/'plan.json')})
    records = []
    status = 'BOUNDED_COMPLETE'
    for row in plan['paths']:
        if (OUT/'STOP').exists():
            status = 'COOPERATIVE_STOP'; break
        free = shutil.disk_usage(ROOT).free
        if free >= LIMITS['target_C_free_bytes']:
            status = 'SPACE_TARGET_REACHED'; break
        if free < LIMITS['protected_C_free_bytes']:
            status = 'STOP_PROTECTED_DISK'; break
        if time.monotonic()-started+60 >= LIMITS['wall_seconds']:
            status = 'FIXED_TIME_CAP'; break
        ensure_idle()
        path, marker = checked(row['path'])
        rec = {'path': str(path), 'sha256_before': helpers.sha(path),
               'logical_bytes_before': path.stat().st_size,
               'allocated_before': helpers.allocated(path)}
        if rec['sha256_before'] != row['original_sha256'] or rec['logical_bytes_before'] != row['logical_bytes']:
            raise ValueError('original logical bytes differ before compression')
        records.append(rec)
        save('progress.json', {'status': 'PREPARED', 'records': records})
        child = None
        error = None
        began = time.monotonic()
        try:
            command = ['compact.exe', '/c', '/exe:lzx', '/q', str(path)]
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, shell=False,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            rec.update(owned_pid=child.pid, command=command)
            save('progress.json', {'status': 'OWNED_COMPACT', 'records': records})
            stdout, stderr = child.communicate(timeout=LIMITS['per_file_seconds'])
            rec.update(exit_code=child.returncode, stdout_hex=stdout.hex(), stderr_hex=stderr.hex())
            if child.returncode:
                raise RuntimeError('compact nonzero exit')
        except BaseException as exc:
            error = repr(exc)
        finally:
            if child is not None:
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=10)
                rec['owned_compact_gone'] = True
        rec.update(sha256_after=helpers.sha(path), logical_bytes_after=path.stat().st_size,
                   allocated_after=helpers.allocated(path), error=error,
                   seconds=time.monotonic()-began, C_free_after=shutil.disk_usage(ROOT).free)
        rec['logical_identical'] = (rec['sha256_after'] == rec['sha256_before']
                                   and rec['logical_bytes_after'] == rec['logical_bytes_before'])
        save('progress.json', {'status': 'VERIFIED', 'records': records})
        print(json.dumps({'path': str(path.relative_to(ROOT)), 'identical': rec['logical_identical'],
                          'allocated_saved': rec['allocated_before']-rec['allocated_after'],
                          'C_free': rec['C_free_after']}), flush=True)
        if error or not rec['logical_identical']:
            status = 'STOP_ERROR_OR_CONTENT_MISMATCH'; break
    free = shutil.disk_usage(ROOT).free
    if free >= LIMITS['target_C_free_bytes'] and not status.startswith('STOP'):
        status = 'SPACE_TARGET_REACHED'
    result = {'status': status, 'files_processed': len(records), 'records': records,
              'seconds': time.monotonic()-started,
              'all_logical_SHA_length_identical': all(r['logical_identical'] for r in records),
              'all_owned_compact_gone': all(r.get('owned_compact_gone', False) for r in records),
              'allocated_saved': sum(r['allocated_before']-r['allocated_after'] for r in records),
              'C_free_before': before, 'C_free_after': free,
              'file_deletions': 0, 'file_moves': 0, 'tensor_reserialization': 0,
              'model_calls': 0, 'old_budget_reused': False,
              'terminal_typed_sources_and_partial9_and_V39_untouched': True}
    save('result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)
    return 0 if not status.startswith('STOP') and result['all_logical_SHA_length_identical'] and result['all_owned_compact_gone'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        raise SystemExit(execute())
    prepare()
