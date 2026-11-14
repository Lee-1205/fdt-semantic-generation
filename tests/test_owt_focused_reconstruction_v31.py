import copy,json,sys,time
from pathlib import Path
from unittest.mock import patch
import torch,pytest
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src'),str(R/'scripts'),str(R/'tests')]
from test_owt_extraction_tail_continuation_v30 import fixture as previous
from test_owt_extraction_tail_runtime_v29 import policy
from test_decoder_alignment_runtime_v8 import Child
from research.owt_extraction_tail_v29 import ExtractionTailFuzzyAE
from owt_fuzzy_engine_v31 import ContinuationEngine,COMPONENTS,cumulative_count
from owt_resource_budget_v29 import RecoveryBudget,RESOURCE_POLICY
import text_generation_support_v1 as support
import run_owt_fuzzy_reconstruction_20261029_v31 as worker
import supervise_owt_fuzzy_reconstruction_20261029_v31 as owner
torch.set_num_threads(2)

def fixture(path):
    old,_,_=previous(path);old.step();old.step();s=copy.deepcopy(old.export())
    e=ContinuationEngine(ExtractionTailFuzzyAE(old.model.c),old.features,old.targets,worker.finite_orders(23,[391,392]),RecoveryBudget(policy(path,RESOURCE_POLICY)),'CPU31','cpu',source_parameter_groups=s['optimizer_parameter_names'])
    review=e.import_parent(s,'CPU30','CPU_SOURCE30');return e,s,review

def test_all81_components_enabled_scaler_skip_stage_and_exact_next_last7(tmp_path):
    e,s,review=fixture(tmp_path);assert review['all_source_components_exact'] and len(e.optimizer.state)==81
    assert support.state_sha({k:s[k] for k in COMPONENTS})==support.state_sha({k:e.export()[k] for k in COMPONENTS})
    digest=support.state_sha(s);e.worker_stage='training';e.evaluations={'focused_initial_metrics':'CPU_reference'};e.step()
    before=support.state_sha(e.model.state_dict());h=e.model.core.anchors.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:assert e.step()['AMP_skip']
    finally:h.remove()
    assert e.scaler.is_enabled() and support.state_sha(e.model.state_dict())==before and support.state_sha(s)==digest
    saved=support.save_full(tmp_path,copy.deepcopy(e.export()));snap=torch.load(tmp_path/'checkpoints'/saved['file'],weights_only=True)
    assert len(e.next_indices())==7
    expected=e.step();final=copy.deepcopy(e.export());n,_,_=fixture(tmp_path);n.restore(snap)
    assert n.worker_stage=='training' and n.evaluations==snap['evaluations'] and n.budget.last_resource_check is None
    assert n.step()==expected and support.state_sha(n.export())==support.state_sha(final)
    while n.attempts<len(n.orders):n.step()
    assert n.cursor==46 and n.updates==5 and n.skips==1 and n.budget.allocation_calls=={'encode':6,'denoise':0,'decoder':12}
    assert cumulative_count(n.export(),'updates')==cumulative_count(s,'updates')+5;n.verify_frozen()

def test_actual_subset_selection_and_finite_exposures():
    path=R/'runs/owt_fuzzy_reconstruction_20261029_v31/plan.json';p=json.loads(path.read_text())
    _,f,y,orders=worker.validate(path,Path(p['output']),worker.__file__)
    assert len(f['train'])==len(y['train'])==64 and len(f['dev'])==256 and len(orders)==2048
    assert sum(len(row['prompt'].encode()) for row in f['train'])==4304
    for start in range(0,2048,8):assert sorted(i for b in orders[start:start+8] for i in b)==list(range(64))
    assert worker.COST['total_max']==11016 and p['limits']['max_calls']==12000
    assert p['training_exposure']['target_UTF8_bytes']==1101824 and p['limits']['start_free_bytes']==40*1024**3

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_schema_postPopen_cleanup(tmp_path,crash,confirmed):
    p=json.loads((R/'runs/owt_fuzzy_reconstruction_20261029_v31/plan.json').read_text())
    p.update(authorized=True,output=str(tmp_path),allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p));owner.validate(path,tmp_path);worker.validate(path,tmp_path,worker.__file__);patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            q=patch.object(owner.io,'atomic_json',injected);q.start();patches.append(q)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'OWT_FUZZY_RECONSTRUCTION_V31_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:code=owner.supervise(path,tmp_path,Path(worker.__file__))
        end=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2) and end['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for q in patches:q.stop()
