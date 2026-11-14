import copy,json,math,sys,time
from pathlib import Path
from unittest.mock import patch
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_three_memory_worker_v34 import continued
from test_three_memory_codec_engine_v33 import policy
from research.three_memory_codec_v33 import ThreeMemoryCodec
from three_memory_codec_engine_v35 import RemainderCodecEngine,CONTINUATION
from owt_resource_budget_v29 import RecoveryBudget
import run_three_memory_codec_20261029_v35 as worker
import supervise_three_memory_codec_20261029_v35 as owner
import text_generation_support_v1 as support
PLAN=ROOT/'research/artifacts/three_memory_codec_20261029_v35_worker_cpu/plan_unsealed.json'

def make_remainder(path):
    parent,_=continued(path);parent.step();snapshot=copy.deepcopy(parent.export())
    with torch.random.fork_rng(devices=[]):model=ThreeMemoryCodec(parent.model.c)
    engine=RemainderCodecEngine(model,parent.features,parent.targets,parent.orders[parent.attempts:],RecoveryBudget(policy(path)),
        'CPU_V35_FIXTURE','cpu',source_parameter_groups=parent.names[:3],data_revision=parent.data_revision)
    engine.import_v34_parent(snapshot,parent.plan_sha,'CPU_PARTIAL_V34_SOURCE')
    return engine,parent,snapshot

def components(s):
    keys=('model','optimizer','optimizer_parameter_names','scaler','frozen_sha','torch_rng','cuda_rng','python_rng','numpy_rng')
    return support.state_sha({k:s[k] for k in keys})

def test_exact91_import_and_original_cosine_next_update_matches_uninterrupted_epoch(tmp_path):
    engine,parent,source=make_remainder(tmp_path)
    initial=copy.deepcopy(engine.export());assert components(initial)==components(source)
    assert len(engine.optimizer.state)==91 and len(engine.orders)==2 and initial['allocation_calls']=={'encode':0,'denoise':0,'decoder':0}
    expected=parent.step();expected_state=copy.deepcopy(parent.export())
    # Re-import resets every stored RNG to the exact saved pre-update state.
    engine.import_v34_parent(source,parent.plan_sha,'CPU_PARTIAL_V34_SOURCE')
    rec=engine.step();after=engine.export()
    assert rec['rows']==expected['rows'] and rec['loss']==expected['loss'] and components(after)==components(expected_state)
    assert rec['epoch_attempt']==2 and rec['schedule_factor']==.1+.9*(1+math.cos(math.pi/3))/2
    assert all(g['lr']==g['base_lr']*rec['schedule_factor'] for g in engine.optimizer.param_groups)
    assert not torch.cuda.is_initialized()

def test_enabled_scaler_skip_suffix_cursor_last7_full_restore_exact_next(tmp_path):
    engine,_,_=make_remainder(tmp_path);engine.worker_stage='finite_epoch_remainder_training';before=support.state_sha(engine.model.state_dict())
    hook=engine.model.core.anchors.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:assert engine.step()['AMP_skip']
    finally:hook.remove()
    assert support.state_sha(engine.model.state_dict())==before and engine.scaler.is_enabled()
    record=support.save_full(tmp_path,copy.deepcopy(engine.export()))
    snap=torch.load(tmp_path/'checkpoints'/record['file'],weights_only=True)
    expected=engine.step();after=copy.deepcopy(engine.export())
    restored,_,_=make_remainder(tmp_path);restored.restore(snap)
    assert restored.model.hold_attempt==2 and len(restored.next_indices())==7
    assert restored.step()==expected and support.state_sha(restored.export())==support.state_sha(after)
    assert restored.cursor==15 and restored.attempts==2 and restored.updates==1 and restored.skips==1
    assert restored.epoch_schedule['offset_batches']+restored.attempts==3
    with pytest.raises(StopIteration):restored.step()

def test_actual_new_plan_only_unconsumed5512_rows_never_replays_or_changes_epoch_schedule():
    p=json.loads(PLAN.read_text(encoding='utf-8'));_,f,y,orders=worker.validate(PLAN,Path(p['output']),worker.__file__)
    owner.validate(PLAN,Path(p['output']))
    full=worker.finite_orders(32768,[34001]);assert orders==full[3407:] and len(orders)==689
    seen={i for b in full[:3407] for i in b};left={i for b in orders for i in b}
    assert len(left)==5512 and left.isdisjoint(seen) and len(left|seen)==32768
    assert sum(len(f['train'][i]['prompt'].encode()) for i in left)==492660
    assert sum(v for k,v in worker.COST.items() if k!='total_max')==4839
    assert worker.LIMITS==owner.LIMITS and p['limits']['wall']==480 and p['limits']['reserve']==90
    assert p['continuation_policy']==CONTINUATION and not p['authorized'] and p['deadline'] is None

class Child:
    pid=999985;returncode=0
    def poll(self):return self.returncode
    def wait(self,timeout=None):return self.returncode

@pytest.mark.parametrize('crash,confirmed,fail_stop',[(False,True,False),(True,True,False),(False,False,False),(True,True,True)])
def test_new_owner_postPopen_exit_paths(tmp_path,crash,confirmed,fail_stop):
    p=json.loads(PLAN.read_text(encoding='utf-8'));p.update(authorized=True,output=str(tmp_path),allocation_start=time.time())
    p.update(deadline=p['allocation_start']+480,cutoff=p['allocation_start']+390)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8');owner.validate(path,tmp_path)
    original_json,original_text=owner.io.atomic_json,owner.io.atomic_text
    def js(dest,value):
        if crash and Path(dest).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('CPU_POST_POPEN')
        return original_json(dest,value)
    def txt(dest,value):
        if fail_stop and Path(dest).name=='STOP':raise OSError('CPU_STOP_PUBLICATION')
        return original_text(dest,value)
    def launch(*a,**k):
        (tmp_path/'result.json').write_text(json.dumps({'status':'THREE_MEMORY_CODEC_V35_COMPLETE'}),encoding='utf-8');return Child()
    with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),\
        patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),\
        patch.object(owner.io,'atomic_json',side_effect=js),patch.object(owner.io,'atomic_text',side_effect=txt),\
        patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
        code=owner.supervise(path,tmp_path,Path(worker.__file__))
    end=json.loads((tmp_path/'supervisor/final_status.json').read_text(encoding='utf-8'))
    assert cleanup.call_count==1 and end['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    assert code==(0 if end['status']=='COMPLETE' else 2)
    if fail_stop:assert 'CPU_STOP_PUBLICATION' in end['STOP_publish_error']
