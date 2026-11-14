import copy,json,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from owt_resource_budget_v24 import RecoveryBudget,RESOURCE_POLICY
import owt_resource_budget_v24 as guard
from test_owt_reconstruction_continuation_v23 import fixture as make_source
from test_decoder_alignment_runtime_v8 import Child
from research.owt_prefix_reconstruction_v19 import PrefixRobustFuzzyAE
from owt_fuzzy_engine_v24 import ContinuationEngine,COMPONENTS,cumulative_count
import text_generation_support_v1 as support
import run_owt_fuzzy_reconstruction_20261029_v24 as worker
import supervise_owt_fuzzy_reconstruction_20261029_v24 as owner
torch.set_num_threads(2)

def plan(path):return {'output':str(path),'cutoff':time.time()+3600,'resource_policy':RESOURCE_POLICY,'limits':{'max_calls':100000,'max_output_bytes':2*1024**3}}

def test_immediate_stop_clock_callcap_and_one_second_resource_limit(tmp_path):
    p=plan(tmp_path);b=RecoveryBudget(p)
    with patch.object(guard.time,'monotonic',return_value=10.) as clock,patch.object(guard.shutil,'disk_usage',return_value=SimpleNamespace(free=30*1024**3)) as disk,patch.object(guard,'owned_bytes',return_value=0) as scan:
        b.check();b.charge('encode');assert scan.call_count==disk.call_count==1
        clock.return_value=10.999;b.check();assert scan.call_count==1
        clock.return_value=11.;disk.return_value=SimpleNamespace(free=21*1024**3)
        with pytest.raises(OSError):b.check()
        disk.return_value=SimpleNamespace(free=30*1024**3);scan.return_value=p['limits']['max_output_bytes']
        with pytest.raises(OSError):b.check()
        scan.return_value=0;b.check();clock.return_value=11.01
        (tmp_path/'STOP').write_text('test')
        with pytest.raises(InterruptedError):b.charge('decoder')
        # Resource-only forced path must allow a cooperative emergency save.
        b.check_resources(force=True,upcoming_bytes=1024)
        (tmp_path/'STOP').unlink();p['cutoff']=0
        with pytest.raises(TimeoutError):b.check()
        p['cutoff']=time.time()+3600;b.allocation_calls['encode']=100000
        with pytest.raises(RuntimeError):b.check()

def test_forced_large_write_checks_and_restore_invalidates_cache(tmp_path):
    b=RecoveryBudget(plan(tmp_path))
    with patch.object(guard.time,'monotonic',return_value=10.),patch.object(guard.shutil,'disk_usage',return_value=SimpleNamespace(free=30*1024**3)),patch.object(guard,'owned_bytes',return_value=0) as scan:
        b.check();b.check();assert scan.call_count==1
        b.check(force=True,upcoming_bytes=768*1024**2);assert scan.call_count==2
        b.invalidate_resources();b.check();assert scan.call_count==3
        with pytest.raises(OSError):b.check_resources(force=True,upcoming_bytes=2*1024**3)

def fixture(path):
    source,_,_=make_source();source.step();source.step();s=copy.deepcopy(source.export())
    b=RecoveryBudget(plan(path));m=PrefixRobustFuzzyAE(source.model.c)
    e=ContinuationEngine(m,source.features,source.targets,worker.finite_orders(23,[321,322]),b,'CPU24','cpu')
    digest=e.import_parent(s,'CPU23','CPU_SOURCE23');return e,s,digest

def test_actual_source_all_components_mask_scaler_rng_exact_next_and_last7(tmp_path):
    e,s,digest=fixture(tmp_path);assert s['schema']=='fdt_owt_fuzzy_AE_v23_fullstate'
    assert digest==support.state_sha({k:s[k] for k in COMPONENTS})==support.state_sha({k:e.export()[k] for k in COMPONENTS})
    before=support.state_sha(s);e.step();assert before==support.state_sha(s)
    hook=e.model.core.anchors.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:assert e.step()['AMP_skip']
    finally:hook.remove()
    saved=support.save_full(tmp_path,copy.deepcopy(e.export()));snap=torch.load(tmp_path/'checkpoints'/saved['file'],weights_only=True)
    expected=e.step();final=copy.deepcopy(e.export());other,_,_=fixture(tmp_path);other.budget.last_resource_check=time.monotonic();other.restore(snap)
    assert other.budget.last_resource_check is None
    assert other.step()==expected and support.state_sha(other.export())==support.state_sha(final)
    while other.attempts<len(other.orders):other.step()
    assert other.cursor==46 and other.attempts==6 and other.updates==5
    assert other.budget.allocation_calls=={'encode':6,'denoise':0,'decoder':12}
    assert cumulative_count(other.export(),'updates')==cumulative_count(s,'updates')+5
    other.verify_frozen()

def test_unchanged_data_new_finite_orders_cost():
    p=json.loads((ROOT/'runs/owt_fuzzy_reconstruction_20261029_v24/plan.json').read_text())
    _,f,_,orders=worker.validate(ROOT/'runs/owt_fuzzy_reconstruction_20261029_v24/plan.json',Path(p['output']),worker.__file__)
    assert len(orders)==8192 and p['external_closed_assessment_calls']['total']==1936
    assert worker.COST['total_max']==26776 and 2*sum(len(r['prompt'].encode()) for r in f['train'])==5852956
    for epoch in range(2):assert sorted(i for b in orders[epoch*4096:(epoch+1)*4096] for i in b)==list(range(32768))

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_schema_postPopen_cleanup(tmp_path,crash,confirmed):
    p=json.loads((ROOT/'runs/owt_fuzzy_reconstruction_20261029_v24/plan.json').read_text())
    p.update(authorized=True,output=str(tmp_path),allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p));owner.validate(path,tmp_path)
    worker.validate(path,tmp_path,worker.__file__);patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'OWT_FUZZY_RECONSTRUCTION_V24_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            code=owner.supervise(path,tmp_path,Path(worker.__file__))
        end=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2) and end['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()
