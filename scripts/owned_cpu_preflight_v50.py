"""Torch-free hard OS ownership for ONE actual worker CPU preflight.

CPU worker<=90sec plus30sec termination/verification reserve. No next model
launch unless every Popen-owned native/proxy has been verified gone. Errors in
status/STOP publication cannot bypass verified-owned cleanup.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import learning_io_v2 as io
import supervise_querycount_partial9_20261027_v3 as native

POLICY = {'worker_seconds_max': 90, 'termination_verification_reserve_seconds': 30,
          'total_seconds_max': 120, 'independent_OS_hard_timeout': True,
          'all_postPopen_exceptions_verified_owned_cleanup': True,
          'unconfirmed_exit_blocks_model_launch': True, 'one_preflight_per_output': True}
REPORT_PREFIX = 'CPU_PREFLIGHT_REPORT '


def preflight(plan_path, output, worker, seconds):
    plan_path, output, worker = Path(plan_path).resolve(), Path(output).resolve(), Path(worker).resolve()
    plan = io.load_json(plan_path)
    if plan['files'].get(str(worker)) != native.sha256(worker):
        raise ValueError('actual preflight worker pin')
    total = min(float(seconds), 120.)
    if total <= 30:
        raise TimeoutError('preflight cleanup reserve unavailable')
    directory = output/'supervisor/cpu_preflight_owned' if plan['authorized'] else plan_path.parent/'cpu_preflight_owned'
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); worker_end = started+min(90., total-30.)
    state = {'status': 'PREPARING_CPU_PREFLIGHT', 'owner_pid': os.getpid(), 'launches': 0,
             'worker_seconds_cap': min(90., total-30.), 'total_seconds_cap': total}
    child = actual = stdout = stderr = None
    reason = None; parsed_report = None

    def persist():
        io.atomic_json(directory/'state.json', state)

    try:
        persist()
        stdout = (directory/'stdout.log').open('xb')
        stderr = (directory/'stderr.log').open('xb')
        command = [sys.executable, '-B', '-X', 'utf8', str(worker), '--plan', str(plan_path),
                   '--output', str(output), '--preflight']
        child = subprocess.Popen(command, cwd=Path(plan['root']), stdin=subprocess.DEVNULL,
            stdout=stdout, stderr=stderr, shell=False, creationflags=native._no_window())
        state.update(status='RUNNING_CPU_PREFLIGHT', launches=1, proxy_pid=child.pid, command=command)
        persist()
        while child.poll() is None:
            if time.monotonic() >= worker_end or (directory/'STOP').exists():
                raise TimeoutError('fixed owned CPU preflight deadline/STOP')
            if actual is None:
                rows = native.owned_descendants(child.pid)
                for row in rows:
                    text = (row.get('CommandLine') or '').lower().replace('\\', '/')
                    if (row.get('ProcessId') != child.pid and row.get('ParentProcessId') == child.pid
                            and row.get('CreationDate') and row.get('ExecutablePath')
                            and all(str(p).lower().replace('\\', '/') in text for p in (worker, plan_path, output))):
                        # Verify the complete observed CIM snapshot itself. Do not
                        # query a naturally exiting PID again merely to label it.
                        actual = dict(row, identity_verified=True); state['actual'] = actual; persist()
                        break
            time.sleep(.25)
        child.wait(timeout=1)
        stdout.flush(); stderr.flush()
        text = (directory/'stdout.log').read_text(encoding='utf-8', errors='replace').strip()
        lines = text.splitlines()
        if child.returncode or not lines or lines[-1] != 'CPU_PREFLIGHT_PASS':
            raise RuntimeError('actual CPU preflight failed: '+text[-300:])
        reports = [line[len(REPORT_PREFIX):] for line in lines if line.startswith(REPORT_PREFIX)]
        if not plan.get('CPU_fixture_only', False):
            if len(reports) != 1:
                raise ValueError('actual fullsize preflight report required')
            parsed_report = json.loads(reports[0])
            if not parsed_report.get('fullsize_engine_restore') or parsed_report['CUDA_initialized'] or not parsed_report['all_parent_tensors_exact'] or not parsed_report['all_old_Adam_exact']:
                raise ValueError('actual fullsize long codec restore contract')
    except BaseException as exc:
        reason = repr(exc)
    finally:
        if child is not None:
            try:
                if reason:
                    cleanup = native.cleanup_owned_after_exception(child, directory/'control', worker,
                        plan_path, output, Path(plan['root']), actual, reason)
                    state['cleanup'] = cleanup; state['exit'] = cleanup['exit_verification']
                else:
                    state['exit'] = native.verify_exit(child, worker, plan_path, output,
                                                       Path(plan['root']), actual, terminate=False)
                    if not state['exit']['all_owned_processes_gone']:
                        cleanup = native.cleanup_owned_after_exception(child, directory/'control', worker,
                            plan_path, output, Path(plan['root']), actual, 'unconfirmed natural CPU exit')
                        state['cleanup'] = cleanup; state['exit'] = cleanup['exit_verification']
            except BaseException as exc:
                reason = reason or 'CPU exit verification exception: '+repr(exc)
                try:
                    cleanup = native.cleanup_owned_after_exception(child, directory/'control', worker,
                        plan_path, output, Path(plan['root']), actual, reason)
                    state['cleanup'] = cleanup; state['exit'] = cleanup['exit_verification']
                except BaseException as cleanup_exc:
                    state['exit'] = {'all_owned_processes_gone': False, 'error': repr(cleanup_exc)}
            state['worker_exit_code'] = child.poll()
            if not state['exit'].get('all_owned_processes_gone'):
                state['status'] = 'UNCONFIRMED_EXIT'
            elif reason:
                state['status'] = 'STOPPED'
            else:
                state['status'] = 'CPU_PREFLIGHT_PASS'
        else:
            state['status'] = 'FAILED_BEFORE_POPEN'
        if stdout: stdout.close()
        if stderr: stderr.close()
        state.update(reason=reason, seconds=time.monotonic()-started, report=parsed_report,
                     owner_imported_Torch='torch' in sys.modules)
        io.atomic_json(directory/'final_status.json', state)
    if state['status'] != 'CPU_PREFLIGHT_PASS':
        raise RuntimeError('owned CPU preflight '+state['status']+': '+str(reason))
    return state
