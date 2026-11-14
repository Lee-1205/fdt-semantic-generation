"""One root-owned independent OS supervisor; reviewed native exit handling reused."""
from __future__ import annotations
import argparse,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from filelock import FileLock
import learning_io_v2 as io
from text_language_runtime_support_v6 import owned_bytes
import supervise_querycount_partial9_20261027_v3 as native

ROOT=Path(__file__).resolve().parents[1]

def validate(plan_path,output):
    plan=io.load_json(plan_path)
    if plan.get('schema')!='fdt_decoder_boundary_readonly_v1' or type(plan.get('authorized')) is not bool: raise ValueError('schema/authorization')
    if Path(plan['output']).resolve()!=output.resolve() or Path(plan['root']).resolve()!=ROOT: raise ValueError('root/output')
    if plan['limits']!={'attempts':0,'batch':4,'micro':4,'wall':120,'reserve':30,'max_calls':160,
        'max_output_bytes':1024**3,'start_free_bytes':40*1024**3,'min_gpu_free_mib':12288}: raise ValueError('caps')
    if plan.get('epochs')!=0 or plan.get('E2_final_accessed') is not False or plan.get('cost_contract',{}).get('total_max')!=140:raise ValueError('epoch/holdout/cost')
    if plan.get('deadline') is not None and (plan['deadline']-plan['allocation_start']!=120 or plan['cutoff']!=plan['deadline']-30): raise ValueError('fixed clock')
    for path,value in plan['files'].items():
        if native.sha256(Path(path))!=value: raise ValueError('pinned file changed '+path)
    if str(Path(__file__).resolve()) not in plan['files']: raise ValueError('owner not pinned')
    if output.exists():
        allow={'plan.json','readiness.json','code_snapshot','supervisor.stdout.log','supervisor.stderr.log','supervisor'}
        if {p.name for p in output.iterdir()}-allow: raise FileExistsError('execution output already exists')
    return plan

def preflight(plan_path,output,worker,seconds):
    declaration=io.load_json(plan_path)
    if str(worker.resolve()) not in declaration['files']: raise ValueError('actual invoked worker not pinned')
    proc=native.run_native([sys.executable,'-B','-X','utf8',str(worker),'--plan',str(plan_path),'--output',str(output),'--preflight'],
        check=False,capture_output=True,text=True,timeout=max(1,seconds),creationflags=native._no_window())
    if proc.returncode or proc.stdout.strip()!='CPU_PREFLIGHT_PASS': raise RuntimeError('worker CPU preflight '+proc.stdout[-300:]+proc.stderr[-1200:])

def duplicates(worker):
    # First invoked .py only; supervisor --worker is not a duplicate worker.
    query="Get-CimInstance Win32_Process -Filter \"Name='python.exe' OR Name='pythonw.exe'\" | Select-Object ProcessId,ParentProcessId,CreationDate,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    p=native.run_native(['powershell.exe','-NoProfile','-NonInteractive','-Command',query],check=True,capture_output=True,text=True,timeout=8,creationflags=native._no_window())
    rows=json.loads(p.stdout) if p.stdout.strip() else []
    if isinstance(rows,dict): rows=[rows]
    invoked={str(worker).lower().replace('\\','/')}
    invoked.update(str(ROOT/'scripts'/name).lower().replace('\\','/') for name in
        ('run_text_generation_20261028.py','run_text_context_decoder_20261028.py','run_text_semantic_alignment_20261028.py','run_text_span_alignment_20261028.py'))
    return [r for r in rows if native.invoked_script(r.get('CommandLine') or '') in invoked]

def resources(plan):
    free=shutil.disk_usage(ROOT).free
    if free<plan['limits']['start_free_bytes']: raise OSError('C20GiB + temporary/headroom floor')
    result=native.run_native(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=10,creationflags=native._no_window())
    gpu=max(int(s) for s in result.stdout.splitlines() if s.strip())
    if gpu<12288: raise OSError('GPU12GiB free floor')
    return {'C_free_bytes':free,'GPU_free_MiB':gpu}

def supervise(plan_path,output,worker):
    plan=validate(plan_path,output)
    if not plan['authorized'] or plan.get('deadline') is None: raise PermissionError('unsealed')
    if native.sha256(worker)!=plan['files'].get(str(worker.resolve())):raise ValueError('invoked worker mismatch')
    if time.time()>=plan['cutoff']: raise TimeoutError('allocation expired before Popen')
    owner=output/'supervisor';owner.mkdir(exist_ok=True)
    state={'status':'PREPARING','pid':os.getpid(),'parent_pid':os.getppid(),'plan_sha256':native.sha256(plan_path),
           'deadline':plan['deadline'],'cutoff':plan['cutoff'],'launches':0,'worker_pid':None,'actual':None}
    child=None;actual=None;reason=None;stdout=stderr=None;worker_state={}
    def persist(): state['observed_at']=time.time();io.atomic_json(owner/'state.json',state)
    persist()
    lock_name='fdt_local_supervisor_'+native.hashlib.sha256(str(ROOT).lower().encode()).hexdigest()[:20]+'.lock'
    with FileLock(str(Path(native.tempfile.gettempdir())/lock_name),timeout=0):
        try:
            if duplicates(worker) or native.process_inventory(ROOT): raise RuntimeError('native duplicate model worker')
            preflight(plan_path,output,worker,min(45,plan['cutoff']-time.time()))
            state['resources']=resources(plan)
            if time.time()>=plan['cutoff']: raise TimeoutError('expired during startup/preflight')
            stdout=(output/'worker.stdout.log').open('xb');stderr=(output/'worker.stderr.log').open('xb')
            command=[sys.executable,'-B','-X','utf8',str(worker),'--plan',str(plan_path),'--output',str(output)]
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=stdout,stderr=stderr,
                creationflags=native._no_window(),shell=False)
            state.update(status='RUNNING',launches=1,worker_pid=child.pid,command=command);persist()
            last_change=time.time();last_signature=None
            while child.poll() is None:
                worker_state=io.load_json(output/'progress.json') or {}
                signature=(worker_state.get('phase'),worker_state.get('attempts'),tuple(sorted(worker_state.get('actual_calls',{}).items())))
                if signature!=last_signature: last_signature=signature;last_change=time.time()
                pid=worker_state.get('native_pid')
                if pid and actual is None and child.poll() is None:
                    if int(pid)!=child.pid:
                        actual=native.process_identity(int(pid),child.pid,worker,plan_path,output)
                        state['actual']=actual
                state['progress']=worker_state;persist()
                if time.time()>=plan['cutoff']: reason='fixed worker cutoff';break
                if time.time()-last_change>180: reason='180sec progress stall';break
                if shutil.disk_usage(ROOT).free<22*1024**3: reason='disk reserve';break
                if owned_bytes(output)>plan['limits']['max_output_bytes']: reason='output cap';break
                if (output/'STOP').exists(): reason='cooperative STOP requested';break
                time.sleep(1)
        except BaseException as exc:
            reason=repr(exc);state['exception_traceback']=traceback.format_exc()
        finally:
            if child is not None:
                if reason:
                    io.atomic_text(output/'STOP',reason+'\n');state['stop_reason']=reason;persist()
                    wait=max(0,min(20,plan['deadline']-time.time()))
                    try: child.wait(timeout=wait)
                    except subprocess.TimeoutExpired: pass
                # ANY post-Popen exception uses owned descendant/identity cleanup.
                try:
                    exit_info=native.verify_exit(child,worker,plan_path,output,ROOT,actual,terminate=child.poll() is None)
                except BaseException as exc: exit_info={'all_owned_processes_gone':False,'error':repr(exc)}
                state['exit']=exit_info;state['exit_code']=child.poll()
                result=io.load_json(output/'result.json')
                if not exit_info.get('all_owned_processes_gone'): state['status']='UNCONFIRMED_EXIT'
                elif reason: state['status']='STOPPED'
                elif child.returncode==0 and result and result.get('status')=='DECODER_BOUNDARY_READONLY_COMPLETE': state['status']='COMPLETE'
                else: state['status']='FAILED'
            else: state['status']='FAILED_BEFORE_POPEN';state['stop_reason']=reason
            if stdout: stdout.close()
            if stderr: stderr.close()
            persist();io.atomic_json(owner/'final_status.json',state)
    return 0 if state['status']=='COMPLETE' else 2

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',type=Path,required=True);p.add_argument('--validate',action='store_true');a=p.parse_args()
    if a.validate:
        validate(a.plan,a.output);preflight(a.plan,a.output,a.worker,60);print('INDEPENDENT_VALIDATE_PASS');return 0
    return supervise(a.plan.resolve(),a.output.resolve(),a.worker.resolve())
if __name__=='__main__': raise SystemExit(main())
