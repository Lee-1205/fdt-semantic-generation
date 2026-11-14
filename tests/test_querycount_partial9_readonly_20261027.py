from pathlib import Path
import json,sys,subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
import supervise_querycount_partial9_20261027 as owner
import evaluate_querycount_partial9_20261027 as worker

def test_worker_panel_contract_alias_and_call_budget_cpu():
    plan=worker.io.load_json(ROOT/'runs/querycount_partial9_readonly_20261027_v1/plan.json')
    rows=worker.load_panel_rows(plan); baseline=worker.io.load_json(plan['baseline_summary_path'])
    counts=worker.validate_panels(rows,baseline,plan)
    assert counts['physical_panels']==13 and counts['physical_rows']==560 and counts['batches']==140
    assert counts['query_score_positions']==5392 and counts['expected_actual_encode_denoise_calls']==700
    assert 'querycount_dev_q16' not in rows
    budget=worker.ActualBudget(10**20,720,ROOT/'does_not_exist_stop')
    for _ in range(140): budget.charge('encode')
    for _ in range(560): budget.charge('denoise')
    for _ in range(280): budget.charge('readout')
    assert (budget.encode,budget.denoise,budget.readout)==(140,560,280)

def _owner_fixture(tmp_path,monkeypatch,mode):
    output=tmp_path/'out'; output.mkdir(); (output/'plan.json').write_text('{}',encoding='utf-8')
    worker_path=tmp_path/'worker.py'; worker_path.write_text('pass',encoding='utf-8'); plan_path=output/'plan.json'
    plan={'output_dir':str(output),'authorized':True,'worker_sha256':'x','supervisor_sha256':'x','parent_payload_sha256':worker.SOURCE_SHA,
          'min_free_bytes':1,'min_gpu_free_mib':12288,'max_actual_encode_denoise_calls':720,'max_output_bytes':64*1024**2}
    class FakeLock:
        def __init__(self,*a,**k): pass
        def acquire(self): pass
        def release(self): pass
    monkeypatch.setattr(owner,'FileLock',FakeLock)
    monkeypatch.setattr(owner,'STOP_GRACE_SECONDS',0)
    monkeypatch.setattr(owner,'validate_plan',lambda *a:(plan,ROOT))
    monkeypatch.setattr(owner,'worker_preflight',lambda *a:{'status':'CPU_PREFLIGHT_PASS'})
    monkeypatch.setattr(owner,'inspect_resources',lambda *a:{'disk_free_bytes':10**12,'gpu_free_mib':15000})
    inv={'n':0}
    orphan={'alive':True}
    def inventory(*a):
        inv['n']+=1
        if mode=='detached' and inv['n']>1:
            return [{'ProcessId':9002,'ParentProcessId':9001,'CommandLine':'python unrelated_child.py --detached'}]
        return []
    monkeypatch.setattr(owner,'process_inventory',inventory)
    def descendants(proxy_pid):
        proxy={'ProcessId':proxy_pid,'ParentProcessId':100,'CreationDate':'proxy-created','CommandLine':'python owner.py'}
        if mode=='orphan_one' and orphan['alive']:
            return [{'ProcessId':9002,'ParentProcessId':proxy_pid,'CreationDate':'orphan-created','CommandLine':'python unrelated_child.py'}]
        if mode=='orphan_one': return []
        if mode=='detached':
            child_row={'ProcessId':9002,'ParentProcessId':proxy_pid,'CreationDate':'child-created','CommandLine':'python unrelated_child.py --detached'}
            return [proxy,child_row]
        return [] if child.code is not None else [proxy]
    monkeypatch.setattr(owner,'owned_descendants',descendants)
    if mode in ('exception','orphan_one'):
        monkeypatch.setattr(owner,'load_json',lambda p: (_ for _ in ()).throw(RuntimeError('injected after Popen')) if str(p).endswith('worker_progress.json') else json.loads(Path(p).read_text(encoding='utf-8')))
    if mode=='detached':
        monkeypatch.setattr(owner,'load_json',lambda p: {'anomaly':'trigger detached-child verification'} if str(p).endswith('worker_progress.json') else json.loads(Path(p).read_text(encoding='utf-8')))
    class Child:
        pid=9001
        def __init__(self):
            self.code=0 if mode=='natural' else None
        def poll(self): return self.code
        def wait(self,timeout=None):
            if self.code is None: raise subprocess.TimeoutExpired('fake',timeout)
            return self.code
    child=Child()
    def fake_popen(args,**kwargs):
        if mode=='natural': (output/'diagnostic_result.json').write_text('{}',encoding='utf-8')
        return child
    monkeypatch.setattr(owner.subprocess,'Popen',fake_popen)
    def fake_run(args,**kwargs):
        if args[0]=='taskkill.exe':
            if str(args[2])=='9002': orphan['alive']=False
            else: child.code=1
        return type('P',(),{'returncode':0,'stdout':'','stderr':''})()
    monkeypatch.setattr(owner.subprocess,'run',fake_run)
    return plan_path,worker_path,output,child

def test_owner_full_path_natural_proxy_and_actual_exit(tmp_path,monkeypatch):
    pp,w,out,child=_owner_fixture(tmp_path,monkeypatch,'natural')
    rc=owner.supervise(pp,w,out)
    final=json.loads((out/'supervisor/final.json').read_text(encoding='utf-8'))
    assert rc==0 and child.poll()==0 and final['status']=='COMPLETE'
    assert final['exit_verification']['all_owned_processes_gone'] is True

def test_owner_exception_without_progress_stops_and_verifies_proxy(tmp_path,monkeypatch):
    pp,w,out,child=_owner_fixture(tmp_path,monkeypatch,'exception')
    rc=owner.supervise(pp,w,out)
    final=json.loads((out/'supervisor/final.json').read_text(encoding='utf-8'))
    assert rc==1 and child.poll()==1 and final['status']=='STOPPED'
    assert final['exit_verification']['all_owned_processes_gone'] is True

def test_owner_detached_child_survival_is_unconfirmed(tmp_path,monkeypatch):
    pp,w,out,child=_owner_fixture(tmp_path,monkeypatch,'detached')
    rc=owner.supervise(pp,w,out)
    final=json.loads((out/'supervisor/final.json').read_text(encoding='utf-8'))
    assert rc==1 and final['status']=='UNCONFIRMED_EXIT'
    assert final['exit_verification']['all_owned_processes_gone'] is False

def test_owner_exception_kills_single_orphan_after_proxy_exits(tmp_path,monkeypatch):
    pp,w,out,child=_owner_fixture(tmp_path,monkeypatch,'orphan_one')
    rc=owner.supervise(pp,w,out)
    final=json.loads((out/'supervisor/final.json').read_text(encoding='utf-8'))
    assert rc==1 and child.poll()==1 and final['status']=='STOPPED'
    assert final['exit_verification']['all_owned_processes_gone'] is True
    assert final['exit_verification']['owned_descendants_after_exit']==[]





def test_score_panel_uses_real_integer_metric_inputs_and_counts_calls_cpu(tmp_path):
    import torch, json, time, math
    from fdt_local.model_masked import FDT100
    plan=worker.io.load_json(ROOT/'runs/querycount_partial9_readonly_20261027_v1/plan.json')
    rows=worker.load_panel_rows(plan)['querycount_dev_q04'][:4]
    cfg=worker.Config()
    class Tiny:
        def __init__(self): self.rollout_noise=None; self.times=[]
        def encode(self,features):
            b=features['query_mask'].shape[0]
            return {'init':torch.zeros((b,16,cfg.z_dim)), 'ground':torch.zeros((b,12,4)), 'query_mask':features['query_mask']}
        def readout(self,z):
            out=torch.zeros((z.shape[0],z.shape[1],4)); out[...,0]=1; return out
        def denoise(self,z,t,encoded,ablate=False): return torch.zeros_like(z)
        def rollout(self,encoded,noise):
            self.rollout_noise=noise.detach().clone()
            self.times=[]
            original=self.denoise
            def track(z,t,enc,ablate=False):
                self.times.append(float(t[0].item()))
                return original(z,t,enc,ablate)
            self.denoise=track
            try: return FDT100.rollout(self,encoded,noise)
            finally: self.denoise=original
    budget=worker.ActualBudget(time.time()+20,720,tmp_path/'NO_STOP')
    tiny=Tiny()
    result=worker.score_panel(tiny,rows,cfg,budget,'tiny',device='cpu')
    assert result['ddim_4']['queries']==16 and len(result['ddim_4']['confusion'])==4
    assert (budget.encode,budget.denoise,budget.readout)==(1,4,2)
    expected=torch.randn((4,16,cfg.z_dim),generator=torch.Generator(device='cpu').manual_seed(worker.SEED))
    assert torch.equal(tiny.rollout_noise,expected)
    assert tiny.times==[1.,.75,.5,.25]

def test_config_declaration_binds_fixed_protocol_and_checkpoint_config():
    plan=worker.io.load_json(ROOT/'runs/querycount_partial9_readonly_20261027_v1/plan.json')
    bound=worker.validate_declaration_config(plan)
    assert bound['declaration_seed']==17 and bound['declaration_query_counts']==[1,4,8,16]
    assert bound['declaration_batch_size']==4 and bound['declaration_rollout_steps']==4
    policy=worker.io.load_json(plan['dependencies']['config'])
    policy['seed']=18
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'bad.json'; p.write_text(json.dumps(policy),encoding='utf-8')
        plan2=dict(plan); plan2['dependencies']=dict(plan['dependencies']); plan2['dependencies']['config']=str(p)
        plan2['dependency_sha256']=dict(plan['dependency_sha256'])
        plan2['dependency_sha256']['config']=worker.sha256(p)
        with pytest.raises(ValueError,match='policy config mismatch'):
            worker.validate_declaration_config(plan2)


