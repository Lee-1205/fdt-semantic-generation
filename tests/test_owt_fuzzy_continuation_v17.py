import copy,json,sys,time
from pathlib import Path
from unittest.mock import patch
import torch,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_owt_fuzzy_continuation_v16 import fixture as make_source,Budget
from test_decoder_alignment_runtime_v8 import Child
from research.owt_fuzzy_reconstruction_v12 import FuzzySentenceAE
from owt_fuzzy_engine_v17 import ContinuationEngine,COMPONENTS,cumulative_count
import text_generation_support_v1 as support
import run_owt_fuzzy_reconstruction_20261028_v17 as worker
import supervise_owt_fuzzy_reconstruction_20261028_v17 as owner
from owt_fuzzy_engine_v17_base import amp_stop_reason
torch.set_num_threads(2)

def fixture():
    source,_,_=make_source();source.step();source.step();state=copy.deepcopy(source.export())
    m=FuzzySentenceAE(source.model.c)
    e=ContinuationEngine(m,source.features,source.targets,worker.finite_orders(23,[221,222]),Budget(),'CPU17','cpu')
    digest=e.import_parent(state,'CPU16','CPU_SOURCE16');return e,state,digest

def test_V16_actual_schema_exact_components_and_recursive_counters_no_source_mutation():
    e,s,digest=fixture();out=e.export()
    assert s['schema']=='fdt_owt_fuzzy_AE_v16_fullstate' and out['schema']=='fdt_owt_fuzzy_AE_v17_fullstate'
    assert digest==support.state_sha({k:s[k] for k in COMPONENTS})==support.state_sha({k:out[k] for k in COMPONENTS})
    assert cumulative_count(s,'updates')==10 and e.ancestry['source_global_updates']==10 and cumulative_count(out,'updates')==10
    assert not e.history and e.attempts==e.updates==e.cursor==0 and sum(e.budget.allocation_calls.values())==0
    assert e.ancestry['ancestry']==s['ancestry'];before=support.state_sha(s);e.step();assert support.state_sha(s)==before

def test_new_fullstate_exact_next_last7_allRNG_and_budget_ancestry(tmp_path):
    e,s,_=fixture();e.step();e.step();saved=support.save_full(tmp_path,copy.deepcopy(e.export()))
    snapshot=torch.load(tmp_path/'checkpoints'/saved['file'],map_location='cpu',weights_only=True)
    expected=e.step();end=copy.deepcopy(e.export());n,_,_=fixture();n.restore(snapshot)
    assert n.step()==expected and support.state_sha(n.export())==support.state_sha(end)
    while n.attempts<len(n.orders):n.step()
    assert n.cursor==46 and n.updates==6 and cumulative_count(n.export(),'updates')==16
    assert sum(n.budget.allocation_calls.values())==12 and n.budget.calls['encode']==s['actual_calls']['encode']+6
    n.verify_frozen()

def test_expanded_corpus_finite_learning_dose_no_extra_epoch():
    orders=worker.finite_orders(8192,range(251,253));assert len(orders)==2048
    for epoch in range(2):assert sorted(i for batch in orders[epoch*1024:(epoch+1)*1024] for i in batch)==list(range(8192))
    assert worker.LIMITS['attempts']==2048 and worker.COST['total_max']==5776

def test_isolated_real_V16_backoffs_do_not_mean_persistent_overflow():
    history=[{'AMP_skip':False} for _ in range(6867)]
    for i in (2613,4719,6867):history[i-1]['AMP_skip']=True
    assert amp_stop_reason(history,[],3,True,False) is None
    assert amp_stop_reason([{'AMP_skip':True}]*3,[],3,True,False)=='3 consecutive AMP skips'
    flags=[{'AMP_skip':i in (0,7,15)} for i in range(16)]
    assert amp_stop_reason(flags,[],3,True,False)=='3 AMP skips within16attempts'

def test_guard_crosses_source_boundary_and_retains_other_numeric_limits():
    assert amp_stop_reason([{'AMP_skip':True}]*2,[{'AMP_skip':True}],2,True,False)=='3 consecutive AMP skips'
    assert amp_stop_reason([{'AMP_skip':True}],[],8,True,False)=='8 AMP skips in new finite allocation'
    assert amp_stop_reason([{'AMP_skip':False}],[],0,False,False)=='successful update has nonfinite gradient'

def test_enabled_CPU_scaler_skip_preserves_weights_moments_and_exact_next_restore():
    e,_,_=fixture();before_model=support.state_sha(e.model.state_dict());before_moments=support.state_sha(e.optimizer.state_dict()['state'])
    initial_scale=e.scaler.get_scale();handle=e.model.core.anchors.register_hook(lambda grad:torch.full_like(grad,float('inf')))
    try:record=e.step()
    finally:handle.remove()
    assert record['AMP_skip'] and e.updates==0 and e.attempts==1 and e.cursor==8 and e.skips==1
    assert e.scaler.get_scale()==initial_scale/2 and before_model==support.state_sha(e.model.state_dict())
    assert before_moments==support.state_sha(e.optimizer.state_dict()['state'])
    snapshot=copy.deepcopy(e.export());expected=e.step();end=copy.deepcopy(e.export());other,_,_=fixture();other.restore(snapshot)
    assert other.step()==expected and support.state_sha(other.export())==support.state_sha(end)

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_V17_schema_owner_cleanup_after_any_Popen(tmp_path,crash,confirmed):
    p=json.loads((ROOT/'runs/owt_fuzzy_reconstruction_20261028_v17/plan.json').read_text())
    p.update(authorized=True,output=str(tmp_path),allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8')
    owner.validate(path,tmp_path);worker.validate(path,tmp_path,ROOT/'scripts/run_owt_fuzzy_reconstruction_20261028_v17.py')
    patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'OWT_FUZZY_RECONSTRUCTION_V17_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            code=owner.supervise(path,tmp_path,ROOT/'scripts/run_owt_fuzzy_reconstruction_20261028_v17.py')
        d=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2) and d['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for p in patches:p.stop()
