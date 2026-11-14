"""Torch-free, fixed-clock Windows owner for one feasible actual-Adam original-anchor V91 trial."""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from filelock import FileLock
import learning_io_v2 as io
from owned_output_size_v52 import owned_output_bytes
import supervise_querycount_partial9_20261027_v3 as native

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--validate',action='store_true')
    args=parser.parse_args()
    plan_path=args.plan.resolve()
    p=io.load_json(plan_path)
    output=Path(p['output'])
    worker=ROOT/'scripts/run_latent_identity_v110_cpu.py'
    if (p['schema']!='fdt_fullrank_LATENT_ONLY_identity_thought_v110' or p['wall_seconds']!=600 or p['max_calls']!=13200 or p['max_attempts']!=67 or p['train_rows']!=536 or p['max_output_bytes']!=640*1024**2):
        raise ValueError('fixed NEW selected134 identity-only thought scope')
    for name,digest in p['files'].items():
        if native.sha256(Path(name))!=digest:raise ValueError('input pin')
    if args.validate:
        from owned_cpu_preflight_v50 import preflight
        report=preflight(plan_path,output,worker,120)
        print(json.dumps(report))
        return 0
    if not p['authorized'] or p['deadline']-p['start']!=600 or p['cutoff']!=p['deadline']-120:
        raise ValueError('new bounded clock required')
    if (output/'owner_started.json').exists() or time.time()>=p['cutoff']:
        raise ValueError('one unexpired profile')
    for name,digest in p['files'].items():
        if native.sha256(Path(name))!=digest:
            raise ValueError('profile pin changed')
    if shutil.disk_usage(ROOT).free<108*1024**3+1024**2 or shutil.disk_usage(output).free<8*1024**3+640*1024**2:
        raise OSError('C100+8GiB+1MiB metadata; D8GiB+640MiB whole trial')
    inventory=native.run_native(['powershell.exe','-NoProfile','-NonInteractive','-Command',
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
        check=True,capture_output=True,text=True,timeout=8,creationflags=native._no_window())
    rows=json.loads(inventory.stdout) if inventory.stdout.strip() else []
    if isinstance(rows,dict):
        rows=[rows]
    for row in rows:
        invoked=native.invoked_script(row.get('CommandLine') or '') or ''
        if str(ROOT).lower().replace('\\','/') in invoked and Path(invoked).name.startswith(('run_','profile_long_')):
            raise RuntimeError('another model/profile worker exists')
    if p.get('device')!='cpu':raise ValueError('CPU-only profile; GPU execution forbidden')
    state=dict(status='PREPARING',owner_pid=os.getpid(),parent_pid=os.getppid(),launches=0)
    io.atomic_json(output/'owner_started.json',state)
    child=actual=None
    reason=None
    out=err=None
    lock=Path(native.tempfile.gettempdir())/('fdt_local_supervisor_'+native.hashlib.sha256(str(ROOT).lower().encode()).hexdigest()[:20]+'.lock')
    with FileLock(str(lock),timeout=0):
        try:
            out=(output/'worker.stdout.log').open('xb')
            err=(output/'worker.stderr.log').open('xb')
            child=subprocess.Popen([sys.executable,'-B','-X','utf8',str(worker),'--plan',str(plan_path),
                                    '--output',str(output)],cwd=ROOT,stdin=subprocess.DEVNULL,
                                    stdout=out,stderr=err,creationflags=native._no_window())
            launched_at=time.time()
            state.update(status='RUNNING',launches=1,worker_proxy_pid=child.pid)
            while child.poll() is None:
                progress=io.load_json(output/'progress.json') or {}
                state['progress']=progress
                state['observed_at']=time.time()
                io.atomic_json(output/'state.json',state)
                if actual is None:
                    candidates=native.owned_descendants(child.pid)
                    for row in candidates:
                        command=(row.get('CommandLine') or '').lower().replace('\\','/')
                        if (row.get('ParentProcessId')==child.pid and row.get('CreationDate')
                                and row.get('ExecutablePath') and str(worker).lower().replace('\\','/') in command):
                            actual=dict(row,identity_verified=True)
                            state['actual']=actual
                            break
                if time.time()>=p['cutoff']:
                    reason='fixed training cutoff'
                    break
                if shutil.disk_usage(ROOT).free<104*1024**3:
                    reason='100GiB protected floor'
                    break
                if shutil.disk_usage(output).free<1024**3:
                    reason='D1GiB output reserve'
                    break
                if owned_output_bytes(output)>p['max_output_bytes']:
                    reason='training output cap'
                    break
                if (output/'STOP').exists():
                    reason='cooperative STOP'
                    break
                if time.time()-progress.get('time',launched_at)>120:
                    reason='fixed 120sec progress publication stall'
                    break
                time.sleep(.5)
        except BaseException as error:
            reason=repr(error)
        finally:
            if child is not None:
                if reason:
                    try: io.atomic_text(output/'STOP',reason)
                    except BaseException: pass
                try:
                    if reason:
                        exit_info=native.cleanup_owned_after_exception(child,output,worker,plan_path,output,ROOT,actual,reason)['exit_verification']
                    else:
                        exit_info=native.verify_exit(child,worker,plan_path,output,ROOT,actual,terminate=False)
                        if not exit_info.get('all_owned_processes_gone'):
                            exit_info=native.cleanup_owned_after_exception(child,output,worker,plan_path,output,ROOT,actual,'unconfirmed natural exit')['exit_verification']
                except BaseException as error:
                    try:
                        exit_info=native.cleanup_owned_after_exception(child,output,worker,plan_path,output,ROOT,actual,str(error))['exit_verification']
                    except BaseException as final_error:
                        exit_info=dict(all_owned_processes_gone=False,error=repr(final_error))
                state.update(exit=exit_info,exit_code=child.poll(),reason=reason)
                state['status']=('UNCONFIRMED_EXIT' if not exit_info.get('all_owned_processes_gone')
                                 else 'STOPPED' if reason else 'COMPLETE' if child.returncode==0 and (output/'result.json').exists() else 'FAILED')
            else:
                state.update(status='FAILED_BEFORE_POPEN',reason=reason)
            if out:out.close()
            if err:err.close()
            io.atomic_json(output/'final_status.json',state)
    return 0 if state['status']=='COMPLETE' else 2


if __name__=='__main__':
    raise SystemExit(main())
