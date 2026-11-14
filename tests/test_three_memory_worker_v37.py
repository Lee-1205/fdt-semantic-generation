import copy,json,sys,time
from pathlib import Path
from unittest.mock import patch
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_three_memory_worker_v36 import new_epoch as second_epoch
from test_three_memory_worker_v35 import components
from test_three_memory_codec_engine_v33 import policy
from research.three_memory_codec_v33 import ThreeMemoryCodec
from three_memory_codec_engine_v37 import ThirdEpochCodecEngine,CONTINUATION
from owt_resource_budget_v29 import RecoveryBudget
import run_three_memory_codec_20261029_v37 as worker
import supervise_three_memory_codec_20261029_v37 as owner
import text_generation_support_v1 as support
PLAN=ROOT/'research/artifacts/three_memory_codec_20261029_v37_worker_cpu/plan_unsealed.json'

def new_epoch(path):
    parent,_=second_epoch(path)
    while parent.attempts<len(parent.orders):parent.step()
    parent.worker_stage='final_commit';source=copy.deepcopy(parent.export())
    orders=[list(range(8)),list(range(8,16)),list(range(16,23))]
    with torch.random.fork_rng(devices=[]):model=ThreeMemoryCodec(parent.model.c)
    e=ThirdEpochCodecEngine(model,parent.features,parent.targets,orders,RecoveryBudget(policy(path)),
        'CPU_THIRD_EPOCH','cpu',source_parameter_groups=parent.names[:3],data_revision=parent.data_revision)
    e.import_completed_v36(source,parent.plan_sha,'CPU_COMPLETED_V36_SOURCE')
    return e,source

def test_ALL91_exact_import_new_finite_cosine_without_Adam_step_reset(tmp_path):
    e,s=new_epoch(tmp_path);initial=copy.deepcopy(e.export())
    assert components(s)==components(initial) and len(e.optimizer.state)==91
    assert initial['allocation_calls']=={'encode':0,'denoise':0,'decoder':0} and not initial['evaluations']
    assert initial['ancestry']['source_schema']==s['schema'] and initial['ancestry']['continuation']==s['continuation']
    before={i:int(v['step']) for i,v in s['optimizer']['state'].items()}
    e.step();after=e.export()
    assert all(int(v['step'])==before[i]+1 for i,v in after['optimizer']['state'].items())
    assert all(g['lr']==g['base_lr'] for g in e.optimizer.param_groups)
    assert e.continuation['policy']==CONTINUATION and not torch.cuda.is_initialized()

def test_second_epoch_enabled_scaler_skip_last7_full_restore_exact_next(tmp_path):
    e,_=new_epoch(tmp_path);e.worker_stage='second_broad_codec_training';e.step()
    before=support.state_sha(e.model.state_dict());hook=e.model.core.anchors.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:assert e.step()['AMP_skip']
    finally:hook.remove()
    assert e.scaler.is_enabled() and support.state_sha(e.model.state_dict())==before
    marker=support.save_full(tmp_path,copy.deepcopy(e.export()));state=torch.load(tmp_path/'checkpoints'/marker['file'],weights_only=True)
    expected=e.step();final=copy.deepcopy(e.export());restored,_=new_epoch(tmp_path);restored.restore(state)
    assert restored.model.hold_attempt==2 and len(restored.next_indices())==7
    assert restored.step()==expected and support.state_sha(restored.export())==support.state_sha(final)
    assert restored.attempts==3 and restored.updates==2 and restored.skips==1 and restored.cursor==23
    with pytest.raises(StopIteration):restored.step()

def test_actual_new_second_epoch_plan_independent_cost_and_source_scope():
    p=json.loads(PLAN.read_text(encoding='utf-8'));_,f,y,orders=worker.validate(PLAN,Path(p['output']),worker.__file__)
    owner.validate(PLAN,Path(p['output']))
    assert len(orders)==4096 and sorted(i for b in orders for i in b)==list(range(32768))
    assert orders!=worker.finite_orders(32768,[35001]) and p['source_baseline_reference']['repeat_initial_GPU_baseline'] is False
    assert {x['id'] for x in f['train']}.isdisjoint({x['id'] for x in f['dev']})
    assert worker.TIME_COST==owner.TIME_COST and worker.LIMITS==owner.LIMITS
    assert 820+60+60+60+120==1120<1200 and sum(v for k,v in worker.COST.items() if k!='total_max')==15060
    assert not p['authorized'] and p['deadline'] is None and not Path(p['output']).exists()

class Child:
    pid=999986;returncode=0
    def poll(self):return self.returncode
    def wait(self,timeout=None):return self.returncode

@pytest.mark.parametrize('crash,confirmed,fail_stop',[(False,True,False),(True,True,False),(False,False,False),(True,True,True)])
def test_current_owner_all_postPopen_exit_paths(tmp_path,crash,confirmed,fail_stop):
    p=json.loads(PLAN.read_text(encoding='utf-8'));p.update(authorized=True,output=str(tmp_path),allocation_start=time.time())
    p.update(deadline=p['allocation_start']+1200,cutoff=p['allocation_start']+1080)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8');owner.validate(path,tmp_path)
    original_json,original_text=owner.io.atomic_json,owner.io.atomic_text
    def js(dest,value):
        if crash and Path(dest).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('CPU_POST_POPEN')
        return original_json(dest,value)
    def txt(dest,value):
        if fail_stop and Path(dest).name=='STOP':raise OSError('CPU_STOP_PUBLICATION')
        return original_text(dest,value)
    def launch(*a,**k):
        (tmp_path/'result.json').write_text(json.dumps({'status':'THREE_MEMORY_CODEC_V37_COMPLETE'}),encoding='utf-8');return Child()
    with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),\
        patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),\
        patch.object(owner.io,'atomic_json',side_effect=js),patch.object(owner.io,'atomic_text',side_effect=txt),\
        patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
        code=owner.supervise(path,tmp_path,Path(worker.__file__))
    end=json.loads((tmp_path/'supervisor/final_status.json').read_text(encoding='utf-8'))
    assert cleanup.call_count==1 and end['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    assert code==(0 if end['status']=='COMPLETE' else 2)
    if fail_stop:assert 'CPU_STOP_PUBLICATION' in end['STOP_publish_error']
