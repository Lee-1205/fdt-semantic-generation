import copy,json,sys,time
from pathlib import Path
from unittest.mock import patch
import torch,pytest
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src'),str(R/'scripts'),str(R/'tests')]
from test_owt_extraction_tail_runtime_v29 import fixture as previous,policy
from test_decoder_alignment_runtime_v8 import Child
from research.owt_extraction_tail_v29 import ExtractionTailFuzzyAE
from owt_fuzzy_engine_v30 import ContinuationEngine,COMPONENTS,cumulative_count
from owt_resource_budget_v29 import RecoveryBudget,RESOURCE_POLICY
import text_generation_support_v1 as support
import run_owt_fuzzy_reconstruction_20261029_v30 as worker
import supervise_owt_fuzzy_reconstruction_20261029_v30 as owner
torch.set_num_threads(2)

def fixture(path):
    old,_,_=previous(path);old.step();old.step();s=copy.deepcopy(old.export())
    e=ContinuationEngine(ExtractionTailFuzzyAE(old.model.c),old.features,old.targets,worker.finite_orders(23,[381,382]),RecoveryBudget(policy(path,RESOURCE_POLICY)),'CPU30','cpu',source_parameter_groups=s['optimizer_parameter_names'])
    report=e.import_parent(s,'CPU29','CPU_SOURCE29');return e,s,report

def test_all81_three_populated_groups_exact_and_enabled_skip_full_restore_last7(tmp_path):
    e,s,report=fixture(tmp_path);assert len(e.optimizer.state)==81 and report['all_source_components_exact']
    assert support.state_sha({k:s[k] for k in COMPONENTS})==support.state_sha({k:e.export()[k] for k in COMPONENTS})
    original=support.state_sha(s);e.step();before=support.state_sha(e.model.state_dict())
    h=e.model.core.anchors.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:assert e.step()['AMP_skip']
    finally:h.remove()
    assert support.state_sha(e.model.state_dict())==before and e.scaler.is_enabled() and support.state_sha(s)==original
    saved=support.save_full(tmp_path,copy.deepcopy(e.export()));snap=torch.load(tmp_path/'checkpoints'/saved['file'],weights_only=True)
    assert len(e.next_indices())==7
    expected=e.step();final=copy.deepcopy(e.export());other,_,_=fixture(tmp_path);other.restore(snap)
    assert other.budget.last_resource_check is None and other.step()==expected and support.state_sha(other.export())==support.state_sha(final)
    while other.attempts<len(other.orders):other.step()
    assert other.cursor==46 and other.updates==5 and other.attempts==6 and other.skips==1
    assert cumulative_count(other.export(),'updates')==cumulative_count(s,'updates')+5
    assert other.budget.allocation_calls=={'encode':6,'denoise':0,'decoder':12}
    for gi,group in enumerate(other.optimizer.param_groups):
        source_steps={int(s['optimizer']['state'][i]['step']) for i in s['optimizer']['param_groups'][gi]['params']}
        assert {int(other.optimizer.state[p]['step']) for p in group['params']}=={x+5 for x in source_steps}
    other.verify_frozen();assert other.ancestry['ancestry']==s['ancestry'] and other.migration==s['migration']

def test_actual_plan_unchanged_data_two_finite_epochs():
    path=R/'runs/owt_fuzzy_reconstruction_20261029_v30/plan.json';p=json.loads(path.read_text())
    _,f,_,orders=worker.validate(path,Path(p['output']),worker.__file__)
    assert len(orders)==8192 and p['training_exposure']['target_UTF8_bytes']==5852956 and worker.COST['total_max']==26776
    for i in (0,4096):assert sorted(j for b in orders[i:i+4096] for j in b)==list(range(32768))
    assert p['source_global_updates']==92099 and p['limits']['start_free_bytes']==40*1024**3

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_schema_postPopen_cleanup(tmp_path,crash,confirmed):
    p=json.loads((R/'runs/owt_fuzzy_reconstruction_20261029_v30/plan.json').read_text())
    p.update(authorized=True,output=str(tmp_path),allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p));owner.validate(path,tmp_path);worker.validate(path,tmp_path,worker.__file__);patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            q=patch.object(owner.io,'atomic_json',injected);q.start();patches.append(q)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'OWT_FUZZY_RECONSTRUCTION_V30_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:code=owner.supervise(path,tmp_path,Path(worker.__file__))
        end=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2) and end['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for q in patches:q.stop()
