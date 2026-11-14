"""One fixed saved-only ROOT review; two dependent jobs, zero model forwards."""
import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import learning_io_v2 as io
import supervise_querycount_partial9_20261027_v3 as native

def main():
    run=ROOT/'runs/three_memory_codec_20261030_v42'
    terminal=io.load_json(run/'code_snapshot/root_terminal_native_before_analysis.json')
    assert terminal['native']==[] and terminal['all_owned_proxy_gone_before_analysis']
    directory=run/'root_saved_review_cpu';directory.mkdir(exist_ok=False)
    jobs=[ROOT/'scripts'/name for name in ('review_v42_three_memory_readouts_cpu.py','review_three_memory_codec_v42_result_cpu.py')]
    started=time.monotonic();start_unix=time.time();cutoff=started+270
    plan={'purpose':'saved-only 2-checkpoint ALL91 state/logit/ALL80 ROOT reading review',
          'root':str(ROOT),'run':str(run),'model_calls':0,'GPU_calls':0,'seconds':300,'reserve':30,
          'start':start_unix,'worker_cutoff':start_unix+270,'deadline':start_unix+300,
          'workers':{str(p):native.sha256(p) for p in jobs},'automatic_retry':False}
    path=directory/'plan.json';io.atomic_json(path,plan)
    stages=[];status='COMPLETE'
    for worker in jobs:
        child=actual=out=err=None;reason=None;stage={'worker':str(worker)}
        try:
            if time.monotonic()>=cutoff:raise TimeoutError('fixed saved-review cutoff')
            if native.sha256(worker)!=plan['workers'][str(worker)]:raise ValueError('review code changed')
            out=(directory/(worker.stem+'.stdout.log')).open('xb')
            err=(directory/(worker.stem+'.stderr.log')).open('xb')
            command=[sys.executable,'-B','-X','utf8',str(worker),'--plan',str(path),'--output',str(directory)]
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,
                                   shell=False,creationflags=native._no_window())
            stage.update(proxy_pid=child.pid,command=command)
            while child.poll() is None:
                if time.monotonic()>=cutoff or (directory/'STOP').exists():raise TimeoutError('saved-review fixed CPU cutoff/STOP')
                if actual is None:
                    for row in native.owned_descendants(child.pid):
                        cmd=(row.get('CommandLine') or '').lower().replace('\\','/')
                        if (row.get('ParentProcessId')==child.pid and row.get('CreationDate') and row.get('ExecutablePath')
                            and all(str(p).lower().replace('\\','/') in cmd for p in (worker,path,directory))):
                            actual=dict(row,identity_verified=True);stage['actual']=actual;break
                time.sleep(.25)
            child.wait(timeout=1)
            if child.returncode:raise RuntimeError('saved-only CPU review returned '+str(child.returncode))
        except BaseException as exc:reason=repr(exc)
        finally:
            if child is not None:
                try:
                    if reason:
                        stage['cleanup']=native.cleanup_owned_after_exception(child,directory,worker,path,directory,ROOT,actual,reason)
                        stage['exit']=stage['cleanup']['exit_verification']
                    else:
                        stage['exit']=native.verify_exit(child,worker,path,directory,ROOT,actual,terminate=False)
                        if not stage['exit']['all_owned_processes_gone']:raise RuntimeError('unconfirmed natural review exit')
                except BaseException as exc:
                    reason=reason or repr(exc)
                    stage['cleanup']=native.cleanup_owned_after_exception(child,directory,worker,path,directory,ROOT,actual,reason)
                    stage['exit']=stage['cleanup']['exit_verification']
                stage['exit_code']=child.poll()
            if out:out.close()
            if err:err.close()
            stage['reason']=reason;stages.append(stage)
            if not stage.get('exit',{}).get('all_owned_processes_gone'):status='UNCONFIRMED_EXIT'
            elif reason:status='STOPPED'
            io.atomic_json(directory/'state.json',{'status':status,'stages':stages,'seconds':time.monotonic()-started})
        if status!='COMPLETE':break
    result={'status':status,'stages':stages,'seconds':time.monotonic()-started,
            'model_forward_calls':0,'GPU_calls':0,'owner_imported_Torch':'torch' in sys.modules}
    io.atomic_json(directory/'final_status.json',result)
    print(json.dumps(result))
    return 0 if status=='COMPLETE' else 2
if __name__=='__main__':raise SystemExit(main())
