from pathlib import Path
import json,sys,time,subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import supervise_querycount_alignment_20261027_recovery_v2 as sup

class Child:
    pid=77123
    def __init__(self, exited=False): self.code=0 if exited else None
    def poll(self): return self.code
    def wait(self,timeout=None):
        if self.code is None: raise subprocess.TimeoutExpired('fake',timeout)
        return self.code

def _setup(tmp_path,monkeypatch,child,exception=False):
    output=tmp_path/'out'; run_dir=output/'model'; output.mkdir()
    plan={'authorized':True,'wall_seconds':900,'absolute_deadline_unix':time.time()+300,'run_dir':str(run_dir),'output_dir':str(output),
          'worker_sha256':'x','supervisor_sha256':'x','io_sha256':'x','runtime_source_sha256':'x','config_sha256':'x',
          'reserve_seconds':120,'max_attempts':119,'attempts_at_start':9,'successful_at_start':9,'target_successful_updates':128,
          'stall_seconds':600,'save_stale_seconds':3000,'min_free_bytes':1}
    pp=tmp_path/'plan.json'; pp.write_text(json.dumps(plan),encoding='utf-8')
    worker=tmp_path/'worker.py'; worker.write_text('pass',encoding='utf-8')
    monkeypatch.setattr(sup,'validate_plan',lambda *a:(plan,run_dir,output))
    monkeypatch.setattr(sup,'validate_worker_preflight',lambda *a:{'status':'CPU_PREFLIGHT_PASS'})
    monkeypatch.setattr(sup,'load_json',lambda p: ({'status':'COMPLETE'} if str(p).endswith('final_status.json') else ({'worker_pid':888} if str(p).endswith('worker_progress.json') else None)))
    monkeypatch.setattr(sup,'latest_durable',lambda *a: (_ for _ in ()).throw(RuntimeError('injected monitor failure')) if exception else (None,None))
    monkeypatch.setattr(sup,'evaluate_stop',lambda *a:None)
    monkeypatch.setattr(sup,'process_identity',lambda *a:None)
    monkeypatch.setattr(sup,'owned_descendants',lambda pid:[])
    def kill(ch,**kw): ch.code=1; return 1,True
    monkeypatch.setattr(sup,'terminate_owned_tree',kill)
    monkeypatch.setattr(sup,'POLL_SECONDS',0)
    return pp,worker,output

def test_full_supervise_path_natural_exit_after_popen(tmp_path,monkeypatch):
    child=Child(exited=True); pp,worker,out=_setup(tmp_path,monkeypatch,child)
    rc=sup.supervise(pp,worker,out,popen=lambda *a,**k:child,resource_check=lambda *a:{'ok':True},process_check=lambda *a:[])
    assert rc==0 and child.poll()==0
    state=json.loads((out/'supervisor/final.json').read_text(encoding='utf-8'))
    assert state['status']=='COMPLETE' and state['worker_proxy_alive'] is False

def test_full_supervise_path_exception_cleans_owned_child(tmp_path,monkeypatch):
    child=Child(); pp,worker,out=_setup(tmp_path,monkeypatch,child,exception=True)
    with pytest.raises(RuntimeError,match='injected monitor failure'):
        sup.supervise(pp,worker,out,popen=lambda *a,**k:child,resource_check=lambda *a:{'ok':True},process_check=lambda *a:[])
    assert child.poll()==1
    state=json.loads((out/'supervisor/final.json').read_text(encoding='utf-8'))
    assert state['status']=='OWNER_EXCEPTION_CHILD_EXITED' and state['cleanup_verified'] is True




def test_exception_cleanup_does_not_claim_detached_descendant_gone(tmp_path,monkeypatch):
    child=Child(); pp,worker,out=_setup(tmp_path,monkeypatch,child,exception=True)
    calls={'n':0}
    def inventory(pid):
        calls['n']+=1
        if calls['n']==1: return [{'ProcessId':child.pid}]
        return [{'ProcessId':991,'ParentProcessId':child.pid,'CommandLine':'python detached.py'}]
    monkeypatch.setattr(sup,'owned_descendants',inventory)
    with pytest.raises(RuntimeError,match='injected monitor failure'):
        sup.supervise(pp,worker,out,popen=lambda *a,**k:child,resource_check=lambda *a:{'ok':True},process_check=lambda *a:[])
    state=json.loads((out/'supervisor/final.json').read_text(encoding='utf-8'))
    assert child.poll()==1
    assert state['status']=='UNCONFIRMED_EXIT' and state['cleanup_verified'] is False
    assert state['owned_descendants_after_cleanup'][0]['ProcessId']==991


def test_exception_after_popen_without_progress_still_checks_owned_tree(tmp_path,monkeypatch):
    child=Child(); pp,worker,out=_setup(tmp_path,monkeypatch,child,exception=True)
    monkeypatch.setattr(sup,'load_json',lambda p: ({'status':'COMPLETE'} if str(p).endswith('final_status.json') else None))
    snapshots=iter([[],[]]); monkeypatch.setattr(sup,'owned_descendants',lambda pid:next(snapshots))
    with pytest.raises(RuntimeError,match='injected monitor failure'):
        sup.supervise(pp,worker,out,popen=lambda *a,**k:child,resource_check=lambda *a:{'ok':True},process_check=lambda *a:[])
    state=json.loads((out/'supervisor/final.json').read_text(encoding='utf-8'))
    assert child.poll()==1 and state['cleanup_verified'] is True
    assert state['owned_descendants_before_cleanup']==[] and state['owned_descendants_after_cleanup']==[]
