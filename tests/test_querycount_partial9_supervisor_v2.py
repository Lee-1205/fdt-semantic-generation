import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import supervise_querycount_partial9_20261027_v2 as owner

REAL_PLAN=ROOT/'runs/querycount_partial9_readonly_20261027_v1/plan.json'
WORKER=ROOT/'scripts/evaluate_querycount_partial9_20261027.py'
OWNER=ROOT/'scripts/supervise_querycount_partial9_20261027_v2.py'

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

