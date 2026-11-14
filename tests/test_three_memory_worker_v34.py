import copy,json,sys,time
from pathlib import Path
from unittest.mock import patch
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_three_memory_codec_engine_v33 import fixture,policy
from research.three_memory_codec_v33 import ThreeMemoryCodec
from three_memory_codec_engine_v34 import BroadCodecEngine,CONTINUATION
from owt_resource_budget_v29 import RecoveryBudget
import run_three_memory_codec_20261029_v34 as worker
import supervise_three_memory_codec_20261029_v34 as owner
import text_generation_support_v1 as support
PLAN=ROOT/'research/artifacts/three_memory_codec_20261029_v34_worker_cpu/plan_unsealed.json'

def continued(path):
    parent,_,_=fixture(path);parent.step();state=copy.deepcopy(parent.export())
    with torch.random.fork_rng(devices=[]):model=ThreeMemoryCodec(parent.model.c)
    engine=BroadCodecEngine(model,parent.features,parent.targets,parent.orders,RecoveryBudget(policy(path)),
        'CPU_V34_FIXTURE','cpu',source_parameter_groups=parent.names[:3],data_revision=parent.data_revision)
    engine.import_codec_parent(state,parent.plan_sha,'CPU_V33_FIXTURE_SOURCE')
    return engine,state

def test_complete91_import_and_enabled_scaler_skip_exact_next_last7_restore(tmp_path):
    engine,parent=continued(tmp_path);initial=copy.deepcopy(engine.export())
    parts=('model','optimizer','optimizer_parameter_names','scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')
    assert support.state_sha({k:parent[k] for k in parts})==support.state_sha({k:initial[k] for k in parts})
    assert len(engine.optimizer.state)==91 and initial['allocation_calls']=={'encode':0,'denoise':0,'decoder':0}
    engine.worker_stage='broad_codec_training';engine.step();before=support.state_sha(engine.model.state_dict())
    hook=engine.model.core.anchors.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:assert engine.step()['AMP_skip']
    finally:hook.remove()
    assert support.state_sha(engine.model.state_dict())==before and engine.scaler.is_enabled()
    record=support.save_full(tmp_path,copy.deepcopy(engine.export()))
    state=torch.load(tmp_path/'checkpoints'/record['file'],weights_only=True)
    assert len(engine.next_indices())==7
    expected=engine.step();after=copy.deepcopy(engine.export())
    restored,_=continued(tmp_path);restored.restore(state)
    assert restored.step()==expected and support.state_sha(restored.export())==support.state_sha(after)
    assert restored.cursor==23 and restored.updates==2 and restored.skips==1
    assert restored.export()['continuation']['policy']==CONTINUATION
    with pytest.raises(StopIteration):restored.step()
    assert not torch.cuda.is_initialized()

def test_actual_schema_one_broad_epoch_no_dev_training_or_initial_baseline_repeat():
    p=json.loads(PLAN.read_text(encoding='utf-8'));_,f,y,orders=worker.validate(PLAN,Path(p['output']),worker.__file__)
    owner.validate(PLAN,Path(p['output']))
    assert len(orders)==4096 and sorted(i for b in orders for i in b)==list(range(32768))
    assert len(f['focused'])==64 and len(f['dev'])==256 and len(f['train'])==32768
    assert {x['id'] for x in f['train']}.isdisjoint({x['id'] for x in f['dev']})
    assert sum(v for k,v in worker.COST.items() if k!='total_max')==15060
    assert p['source_baseline_reference']['repeat_initial_GPU_baseline'] is False
    assert worker.LIMITS==owner.LIMITS and worker.CONTRACT==owner.CONTRACT and worker.START_STORAGE==owner.START_STORAGE
    assert not p['authorized'] and p['deadline'] is None and not Path(p['output']).exists()

class Child:
    pid=999984;returncode=0
    def poll(self):return self.returncode
    def wait(self,timeout=None):return self.returncode

@pytest.mark.parametrize('crash,confirmed,fail_stop',[(False,True,False),(True,True,False),(False,False,False),(True,True,True)])
def test_new_owner_all_postPopen_exit_paths(tmp_path,crash,confirmed,fail_stop):
    p=json.loads(PLAN.read_text(encoding='utf-8'));p.update(authorized=True,output=str(tmp_path),allocation_start=time.time())
    p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+780)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8');owner.validate(path,tmp_path)
    original_json,original_text=owner.io.atomic_json,owner.io.atomic_text
    def js(dest,value):
        if crash and Path(dest).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('CPU_POST_POPEN')
        return original_json(dest,value)
    def txt(dest,value):
        if fail_stop and Path(dest).name=='STOP':raise OSError('CPU_STOP_PUBLICATION')
        return original_text(dest,value)
    def launch(*a,**k):
        (tmp_path/'result.json').write_text(json.dumps({'status':'THREE_MEMORY_CODEC_V34_COMPLETE'}),encoding='utf-8');return Child()
    with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),\
        patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),\
        patch.object(owner.io,'atomic_json',side_effect=js),patch.object(owner.io,'atomic_text',side_effect=txt),\
        patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
        code=owner.supervise(path,tmp_path,Path(worker.__file__))
    end=json.loads((tmp_path/'supervisor/final_status.json').read_text(encoding='utf-8'))
    assert cleanup.call_count==1 and end['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    assert code==(0 if end['status']=='COMPLETE' else 2)
    if fail_stop:assert 'CPU_STOP_PUBLICATION' in end['STOP_publish_error']
