import copy,json,sys,time
from pathlib import Path
from unittest.mock import patch
import torch,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_owt_fuzzy_reconstruction_v12 import fixture as source_fixture
from test_decoder_alignment_runtime_v8 import Child
from research.owt_fuzzy_reconstruction_v12 import FuzzySentenceAE
from owt_fuzzy_engine_v13 import ContinuationEngine,COMPONENTS
import text_generation_support_v1 as support
import run_owt_fuzzy_reconstruction_20261028_v13 as worker
import supervise_owt_fuzzy_reconstruction_20261028_v13 as owner
torch.set_num_threads(2)

class Budget:
    def __init__(self):self.calls={'encode':0,'denoise':0,'decoder':0};self.allocation_calls=dict(self.calls)
    def check(self):pass
    def charge(self,k):self.calls[k]+=1;self.allocation_calls[k]+=1

def fixture():
    parent=source_fixture();parent.step();parent.step();state=copy.deepcopy(parent.export())
    f=parent.features+parent.features[:8];y=parent.targets+parent.targets[:8]
    model=FuzzySentenceAE(parent.model.c)
    e=ContinuationEngine(model,f,y,worker.finite_orders(23,[211,212]),Budget(),'CPU13','cpu')
    digest=e.import_parent(state,'CPU12','CPU_SOURCE')
    return e,state,digest

def test_exact_model_allAdam_scaler_allRNG_import_and_new_allocation_only():
    e,state,digest=fixture();out=e.export()
    assert digest==support.state_sha({k:state[k] for k in COMPONENTS})==support.state_sha({k:out[k] for k in COMPONENTS})
    assert (e.attempts,e.updates,e.cursor)==(0,0,0) and not e.history
    assert e.budget.calls==state['actual_calls'] and sum(e.budget.allocation_calls.values())==0
    assert e.ancestry['history']==state['history'] and not torch.cuda.is_initialized()
    original=support.state_sha(state);e.step();assert support.state_sha(state)==original

def test_new_fullstate_exact_next_update_last7_and_cumulative_vs_allocation_calls(tmp_path):
    e,state,_=fixture();e.step();e.step();saved=support.save_full(tmp_path,copy.deepcopy(e.export()))
    snapshot=torch.load(tmp_path/'checkpoints'/saved['file'],map_location='cpu',weights_only=True)
    expected=e.step();end=copy.deepcopy(e.export());n,_,_=fixture();n.restore(snapshot)
    assert n.step()==expected and support.state_sha(n.export())==support.state_sha(end)
    while n.attempts<len(n.orders):n.step()
    assert n.cursor==46 and n.updates==6 and sum(n.budget.allocation_calls.values())==12
    assert n.budget.calls['encode']==state['actual_calls']['encode']+6 and n.budget.calls['denoise']==0
    n.verify_frozen()

def test_old_held_rows_and_old_train_prefix_unchanged_expanded_finite_orders():
    a=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v13';b=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v12'
    for name,key in [('features.json','dev'),('targets.json','dev'),('documents.json','dev')]:
        assert json.loads((a/name).read_text())[key]==json.loads((b/name).read_text())[key]
    f=json.loads((a/'features.json').read_text());old=json.loads((b/'features.json').read_text())
    assert f['train'][:2048]==old['train'] and len(f['train'])==4096
    orders=worker.finite_orders(4096,range(211,215));assert len(orders)==2048
    for epoch in range(4):assert sorted(i for batch in orders[epoch*512:(epoch+1)*512] for i in batch)==list(range(4096))

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_new_schema_owner_postPopen_cleanup(tmp_path,crash,confirmed):
    p=json.loads((ROOT/'runs/owt_fuzzy_reconstruction_20261028_v13/plan.json').read_text())
    p.update(authorized=True,output=str(tmp_path),allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8')
    owner.validate(path,tmp_path);worker.validate(path,tmp_path,ROOT/'scripts/run_owt_fuzzy_reconstruction_20261028_v13.py')
    patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'OWT_FUZZY_RECONSTRUCTION_V13_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            code=owner.supervise(path,tmp_path,ROOT/'scripts/run_owt_fuzzy_reconstruction_20261028_v13.py')
        d=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2) and d['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for p in patches:p.stop()
