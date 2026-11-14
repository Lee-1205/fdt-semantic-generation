import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import supervise_querycount_partial9_20261027_v3 as owner

REAL_PLAN=ROOT/'runs/querycount_partial9_readonly_20261027_v1/plan.json'
WORKER=ROOT/'scripts/evaluate_querycount_partial9_20261027.py'
OWNER=ROOT/'scripts/supervise_querycount_partial9_20261027_v3.py'

def _test_plan(tmp_path, *, authorized=True, deadline=None):
    plan=copy.deepcopy(json.loads(REAL_PLAN.read_text(encoding='utf-8')))
    output=tmp_path/'out'; output.mkdir()
    plan['output_dir']=str(output.resolve()); plan['control_dir']=str((output/'control').resolve())
    plan['authorized']=authorized
    plan['immutable_owner_deadline_unix']=float(deadline if deadline is not None else time.time()+90)
    plan['worker_cutoff_unix']=plan['immutable_owner_deadline_unix']-owner.SHUTDOWN_RESERVE_SECONDS
    plan['dependencies']['supervisor']=str(OWNER.resolve())
    plan['dependency_sha256']['supervisor']=owner.sha256(OWNER.resolve())
    plan['supervisor_sha256']=owner.sha256(OWNER.resolve())
    path=output/'plan.json'; path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    return path,WORKER.resolve(),output,plan

def test_full_supervise_real_plan_schema_builds_state_without_legacy_key(tmp_path,monkeypatch):
    path,worker,output,plan=_test_plan(tmp_path)
    assert 'parent_payload_sha256' not in plan
    class Lock:
        def __init__(self,*a,**k): pass
        def acquire(self): pass
        def release(self): pass
    monkeypatch.setattr(owner,'FileLock',Lock)
    monkeypatch.setattr(owner,'worker_preflight',lambda *a:{'status':'CPU_PREFLIGHT_PASS'})
    monkeypatch.setattr(owner,'inspect_resources',lambda *a:{'disk_free_bytes':10**12,'gpu_free_mib':15000})
    monkeypatch.setattr(owner,'process_inventory',lambda *a:[])
    monkeypatch.setattr(owner,'owned_descendants',lambda *a:[])
    class Child:
        pid=12345
        def poll(self): return 0
    def popen(*a,**k):
        (output/'diagnostic_result.json').write_text('{}',encoding='utf-8')
        return Child()
    monkeypatch.setattr(owner.subprocess,'Popen',popen)
    assert owner.supervise(path,worker,output)==0
    state=json.loads((output/'supervisor/state.json').read_text(encoding='utf-8'))
    final=json.loads((output/'supervisor/final.json').read_text(encoding='utf-8'))
    assert state['checkpoint_payload_sha256']==plan['checkpoint_payload_sha256']
    assert 'parent_payload_sha256' not in state
    assert final['status']=='COMPLETE' and final['exit_verification']['all_owned_processes_gone'] is True
    assert state['deadline_unix']==plan['immutable_owner_deadline_unix']

def test_expired_original_allocation_rejects_before_preflight_or_popen(tmp_path,monkeypatch):
    expired=1793096705.987614
    path,worker,output,_=_test_plan(tmp_path,authorized=False,deadline=expired)
    called={'preflight':False,'popen':False}
    monkeypatch.setattr(owner,'worker_preflight',lambda *a:called.update(preflight=True))
    monkeypatch.setattr(owner.subprocess,'Popen',lambda *a,**k:called.update(popen=True))
    with pytest.raises(TimeoutError,match='original one-shot allocation deadline has expired'):
        owner.supervise(path,worker,output)
    assert called=={'preflight':False,'popen':False}



def test_partial_cim_exit_row_is_unverified_until_absence(tmp_path,monkeypatch):
    row={"ProcessId":26016,"ParentProcessId":20960,"CreationDate":"same-creation","CommandLine":None,"ExecutablePath":None}
    responses=iter([json.dumps(row),""])
    def fake_run(*a,**kw):
        assert kw["encoding"]=="utf-8" and kw["errors"]=="replace"
        return type("Result",(),{"stdout":next(responses),"returncode":0})()
    monkeypatch.setattr(owner.subprocess,"run",fake_run)
    first=owner.process_identity(26016,20960,WORKER,tmp_path/"plan.json",tmp_path)
    assert first["identity_status"]=="PARTIAL_NATIVE_ROW" and first["identity_verified"] is False
    assert owner.process_identity(26016,20960,WORKER,tmp_path/"plan.json",tmp_path) is None


def test_natural_exit_ignores_stale_progress_identity(tmp_path,monkeypatch):
    path,worker,output,plan=_test_plan(tmp_path)
    class Lock:
        def __init__(self,*a,**k): pass
        def acquire(self): pass
        def release(self): pass
    monkeypatch.setattr(owner,"FileLock",Lock)
    monkeypatch.setattr(owner,"worker_preflight",lambda *a:{"status":"CPU_PREFLIGHT_PASS"})
    monkeypatch.setattr(owner,"inspect_resources",lambda *a:{"disk_free_bytes":10**12,"gpu_free_mib":15000})
    monkeypatch.setattr(owner,"process_inventory",lambda *a:[])
    monkeypatch.setattr(owner,"owned_descendants",lambda *a:[])
    original_load=owner.load_json
    monkeypatch.setattr(owner,"load_json",lambda p: {"worker_pid":26016,"stage":"terminal"} if Path(p).name=="worker_progress.json" else original_load(p))
    monkeypatch.setattr(owner,"process_identity",lambda *a:pytest.fail("must not inspect stale progress PID after proxy exit"))
    class Child:
        pid=20960
        def poll(self): return 0
    def popen(*a,**k):
        (output/"diagnostic_result.json").write_text("{}",encoding="utf-8")
        return Child()
    monkeypatch.setattr(owner.subprocess,"Popen",popen)
    assert owner.supervise(path,worker,output)==0
    assert json.loads((output/"supervisor/final.json").read_text())["status"]=="COMPLETE"


def test_verify_exit_settles_partial_row_and_preserves_uncertainty(tmp_path,monkeypatch):
    class Child:
        pid=20960
        def poll(self): return 0
    partial={"ProcessId":26016,"ParentProcessId":20960,"identity_verified":False,"identity_status":"PARTIAL_NATIVE_ROW"}
    answers=iter([partial,None])
    monkeypatch.setattr(owner,"process_identity",lambda *a:next(answers))
    monkeypatch.setattr(owner,"owned_descendants",lambda *a:[])
    settled=owner.verify_exit(Child(),WORKER,tmp_path/"p.json",tmp_path,ROOT,{"ProcessId":26016})
    assert settled["all_owned_processes_gone"] is True
    monkeypatch.setattr(owner,"process_identity",lambda *a:partial)
    monkeypatch.setattr(owner,"owned_descendants",lambda *a:[partial])
    unresolved=owner.verify_exit(Child(),WORKER,tmp_path/"p.json",tmp_path,ROOT,{"ProcessId":26016})
    assert unresolved["all_owned_processes_gone"] is False
    assert unresolved["actual_process_after_exit"]["identity_verified"] is False


def test_localized_native_diagnostic_does_not_break_utf8_reader():
    payload="이미 종료된 프로세스".encode("cp949")
    result=owner.run_native([sys.executable,"-B","-c",f"import sys; sys.stdout.buffer.write({payload!r})"],capture_output=True,text=True,check=True,creationflags=owner._no_window())
    assert isinstance(result.stdout,str) and result.returncode==0


def test_powershell_json_utf8_is_preserved():
    result=owner.run_native(["powershell.exe","-NoProfile","-NonInteractive","-Command","'한글 JSON' | ConvertTo-Json -Compress"],capture_output=True,text=True,check=True,creationflags=owner._no_window())
    assert json.loads(result.stdout)=="한글 JSON"


def test_real_native_full_owner_path_with_cpu_only_child(tmp_path):
    # Actual Windows Popen proxy/child and CIM exit verification, no mocks, model0.
    path,_,output,plan=_test_plan(tmp_path,deadline=time.time()+60)
    stub=tmp_path/"cpu_only_owner_child.py"
    stub.write_text("""import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');p.add_argument('--preflight',action='store_true');a=p.parse_args()
if a.preflight:
 print('CPU_PREFLIGHT_PASS');raise SystemExit(0)
out=Path(a.output)
(out/'worker_progress.json').write_text(json.dumps({'worker_pid':os.getpid(),'stage':'terminal','actual_model_calls':0}),encoding='utf-8')
(out/'diagnostic_result.json').write_text(json.dumps({'cpu_only':True,'actual_model_calls':0}),encoding='utf-8')
time.sleep(.15)
""",encoding="utf-8")
    plan["worker_sha256"]=owner.sha256(stub)
    plan["dependencies"]["worker"]=str(stub.resolve())
    plan["dependency_sha256"]["worker"]=owner.sha256(stub)
    path.write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding="utf-8")
    assert owner.supervise(path,stub,output)==0
    final=json.loads((output/"supervisor/final.json").read_text(encoding="utf-8"))
    assert final["status"]=="COMPLETE" and final["worker_exit_code"]==0
    assert final["exit_verification"]["all_owned_processes_gone"] is True
    assert json.loads((output/"diagnostic_result.json").read_text())["actual_model_calls"]==0
