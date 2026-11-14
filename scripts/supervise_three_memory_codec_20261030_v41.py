"""Independent OS-only single-launch owner; no Torch/neural imports.

Every post-Popen exception reaches owned-tree exit verification, even when STOP
or state publication itself fails. Uncertain identity/remaining child =>
UNCONFIRMED_EXIT. Reviewed V3 native Windows settling/identity helpers reused.
"""
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

ROOT = Path(__file__).resolve().parents[1]
INITIAL_CHECKPOINT_CONTRACT={'format':'fdt_initial_parent_reference_v39','initial_only':True,'parent_required':True,
    'shared_fields':['model','optimizer','optimizer_parameter_names','scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng'],
    'shared_state_must_be_identical':True,'updated_checkpoints':'self_contained','quantization':False}
STORAGE_BOUND={'maximum_full_files':1,'maximum_full_file_bytes':800*1024**2,'maximum_initial_reference_bytes':32*1024**2,
    'assessment_bytes':128*1024**2,'other_bytes':32*1024**2,'total_bound_bytes':992*1024**2,'output_cap_bytes':1024**3}
SINGLE_SNAPSHOT_CONTRACT={'intermediate_full_checkpoint':False,'maximum_updated_full_files':1,
    'initial_checkpoint':'unchanged_parent_reference_v39',
    'after_final_exception':'reuse_only_if_entire_saved_state_matches_except_worker_stage',
    'incomplete_full_write':'preserve_candidates_report_unconfirmed_no_second_write',
    'maximum_uncommitted_attempts':8192,'parent_and_initial_reference_preserved':True}
TIME_COST={'measured_V34_updates_per_second':5.46,'planning_updates_per_second':5.0,
    'training_seconds_ceiling':1639,'startup_seconds':90,'checkpoint_seconds':60,'final_assessment_seconds':60,
    'shutdown_reserve_seconds':180,'calculated_total_seconds':2029,'fixed_new_allocation_seconds':2100,
    'historical_deadlines_extended':False,'previous_unlaunched_split_proposal_replaced':False}
LIMITS = {'attempts': 8192, 'batch': 8, 'wall': 2100, 'reserve': 180, 'max_calls': 30000,
    'max_output_bytes': 1024**3, 'start_free_bytes': 30*1024**3, 'min_gpu_free_mib': 12288}
START_STORAGE = {'protected_free_bytes': 20*1024**3, 'max_owned_output_bytes': 1024**3,
    'temporary_checkpoint_bytes': 1024**3, 'additional_headroom_bytes': 8*1024**3,
    'minimum_start_bytes': 30*1024**3, 'previous_unlaunched_prospective_start_bytes': 40*1024**3,
    'historical_closed_run_floors_changed': False}
CONTRACT = {'phase': 'language_meaning_language_codec_only', 'context': 'input_complete_per_token_long_recent_FP64',
    'persistent': 'learned_prior_only_no_certified_writer', 'exact': 'fresh_input_UTF8_codepoint_index_top16_plus_NULL',
    'realizer': 'differentiable_Commit_only', 'copy_bytes': 0, 'denoise_calls': 0,
    'clean_CE_weight': .5, 'corrupted_prefix_CE_weight': .5, 'past_byte_mask_probability': .5,
    'loss_reduction': 'elementwise_none_then_sum'}
OWNERSHIP = {'old_populated_groups': 3, 'new_groups': 1, 'new_heads_base_lr': 1e-4,
    'new_heads_weight_decay': .01, 'old_Adam_moments_preserved': True,
    'warmstart_changes_extraction_function': True, 'strict_old_learning_trajectory_resume': False}
RESOURCE_POLICY = {'filesystem_interval_seconds': 1.0, 'force_on_start': True, 'invalidate_on_restore': True,
    'force_before_large_write': True, 'worker_disk_reserve_bytes': 22*1024**3,
    'checkpoint_write_reserve_bytes': 1024**3, 'STOP_clock_callcap_immediate': True,
    'emergency_resource_only_check': True, 'independent_owner_poll_seconds': 1.0}


def owned_bytes(output):
    size = 0
    for path in Path(output).rglob('*'):
        try:
            if path.is_file(): size += path.stat().st_size
        except FileNotFoundError:
            if not path.name.endswith('.tmp'): raise
    return size


def validate(plan_path, output):
    plan = io.load_json(plan_path)
    if plan.get('schema') != 'fdt_three_memory_language_codec_v41' or type(plan.get('authorized')) is not bool:
        raise ValueError('actual codec schema/authorization')
    if Path(plan['output']).resolve() != output.resolve() or Path(plan['root']).resolve() != ROOT:
        raise ValueError('root/output identity')
    if plan.get('limits') != LIMITS or plan.get('codec_contract') != CONTRACT or plan.get('ownership_policy') != OWNERSHIP or plan.get('resource_policy') != RESOURCE_POLICY:
        raise ValueError('actual independent caps/codec/ownership/resource contract')
    if plan.get('time_cost_contract')!=TIME_COST or 1639+90+60+60+180!=2029 or LIMITS['wall']!=2100:
        raise ValueError('new measured time-cost contract; historical clocks immutable')
    if plan.get('start_storage_contract') != START_STORAGE or LIMITS['start_free_bytes'] != 20*1024**3+1024**3+1024**3+8*1024**3:
        raise ValueError('NEW prelaunch storage reserve arithmetic')
    if plan.get('initial_checkpoint_contract')!=INITIAL_CHECKPOINT_CONTRACT or plan.get('storage_bound')!=STORAGE_BOUND or plan.get('single_snapshot_contract')!=SINGLE_SNAPSHOT_CONTRACT:
        raise ValueError('initial reference and smaller owned output contract')
    if (800+32+128+32)*1024**2!=STORAGE_BOUND['total_bound_bytes'] or not STORAGE_BOUND['total_bound_bytes']<LIMITS['max_output_bytes']:
        raise ValueError('three fullstates plus reference/assessments/logs storage bound')
    if plan.get('epochs') != 2 or plan.get('order_seed_range') != [40001, 40002] or plan.get('cost_contract') != {
        'training': 24576, 'focused_final_teacher': 32, 'focused_final_BOS_max': 2080,
        'held_final_teacher': 128, 'held_final_BOS_max': 520, 'held16_three_memory_ablations': 12, 'total_max': 27348}:
        raise ValueError('finite exposure/cost')
    if plan.get('continuation_policy') != {'neural_function': 'unchanged_V33_three_memory_codec', 'all_populated_Adam_states': 91, 'new_parameters': 0, 'old_allocation_reopened': False, 'purpose': 'two_broad_finite_TRAIN_epochs', 'learning_rate_schedule': 'same_group_base_LRs_new_finite_cosine_no_moment_or_step_reset', 'source_epoch_completed': True, 'new_finite_epochs': 2}:
        raise ValueError('same91 continuation policy')
    if plan.get('source_sha') != 'fbaf29eca2c5c9e1df12e3ea71702aeee40c0bda5c7f9ec73cba339acd4cd677':
        raise ValueError('actual V40 source')
    if any(plan.get(k) is not False for k in ('reasoning_diffusion_training', 'E2_final_accessed', 'automatic_source_copy')):
        raise ValueError('codec-only/no whole-copy/holdout')
    if plan['deadline'] is not None and (plan['deadline']-plan['allocation_start'] != 2100 or plan['cutoff'] != plan['deadline']-180):
        raise ValueError('fresh fixed clock')
    for name, digest in plan['files'].items():
        if native.sha256(Path(name)) != digest: raise ValueError('pin changed '+name)
    if str(Path(__file__).resolve()) not in plan['files']: raise ValueError('owner unpinned')
    if output.exists():
        allow = {'plan.json', 'readiness.json', 'code_snapshot', 'supervisor.stdout.log', 'supervisor.stderr.log', 'supervisor'}
        if {p.name for p in output.iterdir()}-allow: raise FileExistsError('execution output already exists')
    return plan


def preflight(plan_path, output, worker, seconds):
    plan = io.load_json(plan_path)
    if str(worker.resolve()) not in plan['files']: raise ValueError('actual worker not pinned')
    result = native.run_native([sys.executable, '-B', '-X', 'utf8', str(worker), '--plan', str(plan_path),
        '--output', str(output), '--preflight'], check=False, capture_output=True, text=True,
        timeout=max(1, seconds), creationflags=native._no_window())
    if result.returncode or result.stdout.strip() != 'CPU_PREFLIGHT_PASS':
        raise RuntimeError('actual worker CPU preflight '+result.stdout[-300:]+result.stderr[-1200:])


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
    if free < plan['limits']['start_free_bytes']: raise OSError('new C30GiB=20protected+1output+1temporary+8headroom start floor')
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
            preflight(plan_path, output, worker, min(90, plan['cutoff']-time.time()))
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
                if shutil.disk_usage(ROOT).free < 22*1024**3: reason = 'disk reserve'; break
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
                    try: state['exit'] = native.verify_exit(child, worker, plan_path, output, ROOT, actual, terminate=child.poll() is None)
                    except BaseException as exc: state['exit'] = {'all_owned_processes_gone': False, 'error': repr(exc)}
                state['exit_code'] = child.poll(); result = io.load_json(output/'result.json')
                if not state['exit'].get('all_owned_processes_gone'): state['status'] = 'UNCONFIRMED_EXIT'
                elif reason: state['status'] = 'STOPPED'
                elif child.returncode == 0 and result and result.get('status') == 'THREE_MEMORY_CODEC_V41_COMPLETE': state['status'] = 'COMPLETE'
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
        validate(args.plan, args.output); preflight(args.plan, args.output, args.worker, 90)
        print('INDEPENDENT_VALIDATE_PASS'); return 0
    return supervise(args.plan.resolve(), args.output.resolve(), args.worker.resolve())


if __name__ == '__main__': raise SystemExit(main())
