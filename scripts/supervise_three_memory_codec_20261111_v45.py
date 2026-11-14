"""V45 OS-only owner with the user-requested C: free-space reserve."""
from __future__ import annotations
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from filelock import FileLock

import learning_io_v2 as io
import supervise_querycount_partial9_20261027_v3 as native
from codec_resume_v45_contract import (
    ROOT, LIMITS, STATUS, STOP_FREE, validate as validate_pins,
)


def validate(plan_path, output):
    plan = validate_pins(plan_path, output, __file__)
    if output.exists():
        allowed = {'plan.json', 'readiness.json', 'code_snapshot', 'supervisor.stdout.log',
                   'supervisor.stderr.log', 'supervisor'}
        if {path.name for path in output.iterdir()} - allowed:
            raise FileExistsError('V45 execution output already exists')
    return plan


def owned_bytes(output):
    size = 0
    for path in Path(output).rglob('*'):
        try:
            if path.is_file(): size += path.stat().st_size
        except FileNotFoundError:
            if not path.name.endswith('.tmp'): raise
    return size


def preflight(plan_path, output, worker, seconds):
    from owned_cpu_preflight_v42 import preflight as checked
    return checked(plan_path,output,worker,seconds)


def duplicates(worker):
    query = "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    result = native.run_native(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', query], check=True,
        capture_output=True, text=True, timeout=8, creationflags=native._no_window())
    rows = json.loads(result.stdout) if result.stdout.strip() else []
    if isinstance(rows, dict): rows = [rows]
    root = str(ROOT).lower().replace('\\', '/')+'/'
    return [row for row in rows if (lambda script: script.startswith(root) and Path(script).name.startswith('run_'))(
        native.invoked_script(row.get('CommandLine') or '') or '')]


def resources(plan):
    free = shutil.disk_usage(ROOT).free
    if free < plan['limits']['start_free_bytes']: raise OSError('C: free space below the 110GiB launch reserve')
    result = native.run_native(['nvidia-smi', '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=10, creationflags=native._no_window())
    gpu = int(result.stdout.strip())
    if gpu < 12288: raise OSError('GPU0 12GiB free floor')
    return {'C_free_bytes': free, 'GPU0_free_MiB': gpu}


def supervise(plan_path, output, worker):
    plan = validate(plan_path, output)
    if not plan['authorized'] or plan['deadline'] is None: raise PermissionError('new plan unsealed')
    if native.sha256(worker) != plan['files'].get(str(worker.resolve())): raise ValueError('worker identity')
    if time.time() >= plan['cutoff']: raise TimeoutError('expired before Popen')
    directory = output/'supervisor'; directory.mkdir(exist_ok=True)
    state = {'status': 'PREPARING', 'pid': os.getpid(), 'parent_pid': os.getppid(),
        'plan_sha256': native.sha256(plan_path), 'deadline': plan['deadline'], 'cutoff': plan['cutoff'],
        'launches': 0, 'worker_pid': None, 'actual': None}
    child = actual = stdout = stderr = None; reason = None
    def persist():
        state['observed_at'] = time.time(); io.atomic_json(directory/'state.json', state)
    persist()
    lock_name = 'fdt_local_supervisor_'+native.hashlib.sha256(str(ROOT).lower().encode()).hexdigest()[:20]+'.lock'
    with FileLock(str(Path(native.tempfile.gettempdir())/lock_name), timeout=0):
        try:
            if duplicates(worker) or native.process_inventory(ROOT): raise RuntimeError('native duplicate model worker')
            preflight(plan_path, output, worker, min(120, plan['cutoff']-time.time()))
            state['resources'] = resources(plan)
            if time.time() >= plan['cutoff']: raise TimeoutError('expired during startup')
            stdout = (output/'worker.stdout.log').open('xb'); stderr = (output/'worker.stderr.log').open('xb')
            command = [sys.executable, '-B', '-X', 'utf8', str(worker), '--plan', str(plan_path), '--output', str(output)]
            child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                creationflags=native._no_window(), shell=False)
            state.update(status='RUNNING', launches=1, worker_pid=child.pid, command=command); persist()
            last_change = time.time(); signature = None
            while child.poll() is None:
                progress = io.load_json(output/'progress.json') or {}
                now_signature = (progress.get('phase'), progress.get('attempts'), tuple(sorted(progress.get('allocation_calls', {}).items())))
                if now_signature != signature: signature = now_signature; last_change = time.time()
                pid = progress.get('native_pid')
                if pid and actual is None and child.poll() is None and int(pid) != child.pid:
                    actual = native.process_identity(int(pid), child.pid, worker, plan_path, output); state['actual'] = actual
                state['progress'] = progress; persist()
                if time.time() >= plan['cutoff']: reason = 'fixed worker cutoff'; break
                if time.time()-last_change > 180: reason = '180sec progress stall'; break
                if shutil.disk_usage(ROOT).free < STOP_FREE: reason = '100GiB protected disk reserve'; break
                if owned_bytes(output) > plan['limits']['max_output_bytes']: reason = 'output cap'; break
                if (output/'STOP').exists(): reason = 'cooperative STOP requested'; break
                time.sleep(1)
        except BaseException as exc:
            reason = repr(exc); state['exception_traceback'] = traceback.format_exc()
        finally:
            if child is not None:
                # Publication failure must never skip verified-owned cleanup.
                try:
                    if reason:
                        state['stop_reason'] = reason
                        try: io.atomic_text(output/'STOP', reason+'\n'); persist()
                        except BaseException as exc: state['STOP_publish_error'] = repr(exc)
                        try: child.wait(timeout=max(0, min(20, plan['deadline']-time.time())))
                        except subprocess.TimeoutExpired: pass
                        except BaseException as exc: state['grace_wait_error'] = repr(exc)
                finally:
                    try:
                        if reason:
                            cleanup = native.cleanup_owned_after_exception(child, output, worker, plan_path, output, ROOT, actual, reason)
                            state['cleanup'] = cleanup; state['exit'] = cleanup['exit_verification']
                        else:
                            state['exit'] = native.verify_exit(child, worker, plan_path, output, ROOT, actual, terminate=False)
                            if not state['exit'].get('all_owned_processes_gone'):
                                cleanup = native.cleanup_owned_after_exception(child, output, worker, plan_path, output, ROOT, actual, 'unconfirmed natural worker exit')
                                state['cleanup'] = cleanup; state['exit'] = cleanup['exit_verification']
                    except BaseException as exc:
                        reason = reason or 'exit verification exception: '+repr(exc)
                        try:
                            cleanup = native.cleanup_owned_after_exception(child, output, worker, plan_path, output, ROOT, actual, reason)
                            state['cleanup'] = cleanup; state['exit'] = cleanup['exit_verification']
                        except BaseException as cleanup_exc:
                            state['exit'] = {'all_owned_processes_gone': False, 'error': repr(cleanup_exc)}
                state['exit_code'] = child.poll(); result = io.load_json(output/'result.json')
                if not state['exit'].get('all_owned_processes_gone'): state['status'] = 'UNCONFIRMED_EXIT'
                elif reason: state['status'] = 'STOPPED'
                elif child.returncode == 0 and result and result.get('status') == STATUS: state['status'] = 'COMPLETE'
                else: state['status'] = 'FAILED'
            else: state.update(status='FAILED_BEFORE_POPEN', stop_reason=reason)
            if stdout: stdout.close()
            if stderr: stderr.close()
            persist(); io.atomic_json(directory/'final_status.json', state)
    return 0 if state['status'] == 'COMPLETE' else 2


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--validate', action='store_true'); args = parser.parse_args()
    if args.validate:
        validate(args.plan, args.output); preflight(args.plan, args.output, args.worker, 120)
        print('INDEPENDENT_VALIDATE_PASS'); return 0
    return supervise(args.plan.resolve(), args.output.resolve(), args.worker.resolve())


if __name__ == '__main__': raise SystemExit(main())
