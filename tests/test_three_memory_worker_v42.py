import copy,json,sys,time
from pathlib import Path
from unittest.mock import patch
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_three_memory_codec_engine_v33 import fixture
from test_three_memory_worker_v35 import components
from test_three_memory_codec_engine_v33 import policy
from research.three_memory_codec_v33 import ThreeMemoryCodec
from three_memory_codec_engine_v42 import TwoEpochCodecEngine,CONTINUATION
from three_memory_codec_engine_v41 import CONTINUATION as SOURCE_POLICY
from owt_resource_budget_v29 import RecoveryBudget
import run_three_memory_codec_20261030_v42 as worker
import supervise_three_memory_codec_20261030_v42 as owner
import text_generation_support_v1 as support
import codec_initial_reference_v42_gzip as reference
import codec_single_snapshot_v42 as snapshot
PLAN=ROOT/'research/artifacts/three_memory_codec_20261030_v42_worker_cpu/plan_unsealed.json'

def new_epoch(path):
    # Canonical tiny populated91 state, not a replay of every historical trial.
    # Only the completed-source metadata is synthetic. The real full-size V41
    # state is independently imported by the actual worker CPU preflight.
    parent,_,_=fixture(path)
    while parent.attempts<len(parent.orders):parent.step()
    parent.worker_stage='final_commit';source=copy.deepcopy(parent.export())
    source.update(schema='fdt_three_memory_language_codec_v41_fullstate',
        continuation={'policy':copy.deepcopy(SOURCE_POLICY),'CPU_synthetic_history_only':True})
    orders=[list(range(8)),list(range(8,16)),list(range(16,23))]*2
    with torch.random.fork_rng(devices=[]):model=ThreeMemoryCodec(parent.model.c)
    e=TwoEpochCodecEngine(model,parent.features,parent.targets,orders,RecoveryBudget(policy(path)),
        'CPU_TWO_EPOCH_EPOCH','cpu',source_parameter_groups=parent.names[:3],data_revision=parent.data_revision)
    e.import_completed_v41(source,parent.plan_sha,'CPU_COMPLETED_V41_SOURCE')
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

def test_two_epoch_epoch_enabled_scaler_skip_last7_full_restore_exact_next(tmp_path):
    e,_=new_epoch(tmp_path);e.worker_stage='two_epoch_broad_codec_training';e.step()
    before=support.state_sha(e.model.state_dict());hook=e.model.core.anchors.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:assert e.step()['AMP_skip']
    finally:hook.remove()
    assert e.scaler.is_enabled() and support.state_sha(e.model.state_dict())==before
    for _ in range(3):e.step()
    marker=support.save_full(tmp_path,copy.deepcopy(e.export()));state=torch.load(tmp_path/'checkpoints'/marker['file'],weights_only=True)
    expected=e.step();final=copy.deepcopy(e.export());restored,_=new_epoch(tmp_path);restored.restore(state)
    assert restored.model.hold_attempt==5 and len(restored.next_indices())==7
    assert restored.step()==expected and support.state_sha(restored.export())==support.state_sha(final)
    assert restored.attempts==6 and restored.updates==5 and restored.skips==1 and restored.cursor==46
    with pytest.raises(StopIteration):restored.step()

def test_actual_new_two_epoch_epoch_plan_independent_cost_and_source_scope():
    p=json.loads(PLAN.read_text(encoding='utf-8'));_,f,y,orders=worker.validate(PLAN,Path(p['output']),worker.__file__)
    owner.validate(PLAN,Path(p['output']))
    assert len(orders)==8192 and sorted(i for b in orders for i in b)==sorted(list(range(32768))*2)
    assert orders!=worker.finite_orders(32768,[40001]) and p['source_baseline_reference']['repeat_initial_GPU_baseline'] is False
    assert {x['id'] for x in f['train']}.isdisjoint({x['id'] for x in f['dev']})
    assert worker.TIME_COST==owner.TIME_COST and worker.LIMITS==owner.LIMITS
    assert worker.START_STORAGE==owner.START_STORAGE and reference.POLICY==owner.INITIAL_CHECKPOINT_CONTRACT
    assert snapshot.STORAGE_BOUND==owner.STORAGE_BOUND and 992*1024**2<1024**3
    assert snapshot.POLICY==owner.SINGLE_SNAPSHOT_CONTRACT
    assert 1639+120+60+60+180==2059<2100 and sum(v for k,v in worker.COST.items() if k!='total_max')==27348
    assert not p['authorized'] and p['deadline'] is None and not Path(p['output']).exists()

def test_initial_reference_roundtrip_and_exact_next_update(tmp_path):
    e,parent=new_epoch(tmp_path);initial=copy.deepcopy(e.export())
    source=tmp_path/'immutable_parent.pt';torch.save(parent,source)
    sha,algorithm=support.sha(source),support.state_sha(parent)
    dest=tmp_path/'reference';dest.mkdir()
    mark=reference.save_initial_reference(dest,initial,source,sha,algorithm)
    loaded=reference.load_initial_reference(dest/'checkpoints'/mark['file'],source,sha,algorithm)
    assert support.state_sha(loaded)==support.state_sha(initial) and not mark['standalone']
    expected=e.step();final=copy.deepcopy(e.export());restored,_=new_epoch(tmp_path);restored.restore(loaded)
    assert restored.step()==expected and support.state_sha(restored.export())==support.state_sha(final)
    assert support.sha(source)==sha and not torch.cuda.is_initialized()

def test_reference_rejects_wrong_parent_and_any_updated_state(tmp_path):
    e,parent=new_epoch(tmp_path);initial=copy.deepcopy(e.export());source=tmp_path/'parent.pt'
    algorithm=support.state_sha(parent);ref=reference.base.make_reference(initial,parent,source,'CPU_SHA',algorithm)
    with pytest.raises(ValueError):reference.base.materialize(ref,parent,source,'WRONG_SHA',algorithm)
    changed=copy.deepcopy(parent);next(iter(changed['model'].values())).view(-1)[0]+=1
    with pytest.raises(ValueError):reference.base.materialize(ref,changed,source,'CPU_SHA',algorithm)
    e.step()
    with pytest.raises(ValueError):reference.base.make_reference(e.export(),parent,source,'CPU_SHA',algorithm)

def test_single_full_slot_and_post_final_status_error_reuse(tmp_path):
    payload={'attempts':1,'updates':1,'cursor':8,'worker_stage':'final_commit',
             'model':{'x':torch.tensor([1.])},'optimizer':{},'RNG':torch.tensor([4])}
    slot=snapshot.SingleSnapshot();meta=slot.save(tmp_path,payload)
    changed_stage=dict(payload,worker_stage='complete')
    assert slot.verify_current(changed_stage)['second_full_write'] is False
    with pytest.raises(RuntimeError):slot.save(tmp_path,payload)
    assert len(list((tmp_path/'checkpoints').glob('*.pt')))==1
    assert support.sha(slot.path)==meta['sha256']

def test_existing_fullstate_rejects_changed_model_or_rng(tmp_path):
    payload={'attempts':1,'updates':1,'cursor':8,'worker_stage':'final_commit',
             'model':{'x':torch.tensor([1.])},'RNG':torch.tensor([4])}
    slot=snapshot.SingleSnapshot();slot.save(tmp_path,payload)
    changed=copy.deepcopy(payload);changed['RNG'][0]+=1
    with pytest.raises(ValueError):slot.verify_current(changed)
    changed=copy.deepcopy(payload);changed['model']['x'][0]+=1
    with pytest.raises(ValueError):slot.verify_current(changed)

def test_failed_full_write_preserved_without_second_attempt(tmp_path):
    slot=snapshot.SingleSnapshot();payload={'worker_stage':'final_commit'}
    def fail(*_):
        (tmp_path/'preserved_incomplete.tmp').write_bytes(b'CPU_PARTIAL_WRITE')
        raise OSError('CPU_WRITE_FAILURE')
    with patch.object(snapshot.support,'save_full',side_effect=fail) as writer:
        with pytest.raises(OSError):slot.save(tmp_path,payload)
        with pytest.raises(RuntimeError):slot.save(tmp_path,payload)
        with pytest.raises(RuntimeError):slot.verify_current(payload)
    assert writer.call_count==1 and (tmp_path/'preserved_incomplete.tmp').read_bytes()==b'CPU_PARTIAL_WRITE'

class Child:
    pid=999986;returncode=0
    def poll(self):return self.returncode
    def wait(self,timeout=None):return self.returncode

@pytest.mark.parametrize('crash,confirmed,fail_stop',[(False,True,False),(True,True,False),(False,False,False),(True,True,True)])
def test_current_owner_all_postPopen_exit_paths(tmp_path,crash,confirmed,fail_stop):
    p=json.loads(PLAN.read_text(encoding='utf-8'));p.update(authorized=True,output=str(tmp_path),allocation_start=time.time())
    p.update(deadline=p['allocation_start']+2100,cutoff=p['allocation_start']+1920)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8');owner.validate(path,tmp_path)
    original_json,original_text=owner.io.atomic_json,owner.io.atomic_text
    def js(dest,value):
        if crash and Path(dest).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('CPU_POST_POPEN')
        return original_json(dest,value)
    def txt(dest,value):
        if fail_stop and Path(dest).name=='STOP':raise OSError('CPU_STOP_PUBLICATION')
        return original_text(dest,value)
    def launch(*a,**k):
        (tmp_path/'result.json').write_text(json.dumps({'status':'THREE_MEMORY_CODEC_V42_COMPLETE'}),encoding='utf-8');return Child()
    with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),\
        patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),\
        patch.object(owner.io,'atomic_json',side_effect=js),patch.object(owner.io,'atomic_text',side_effect=txt),\
        patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
        code=owner.supervise(path,tmp_path,Path(worker.__file__))
    end=json.loads((tmp_path/'supervisor/final_status.json').read_text(encoding='utf-8'))
    assert cleanup.call_count==(2 if not confirmed and not crash else 1) and end['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    assert code==(0 if end['status']=='COMPLETE' else 2)
    if fail_stop:assert 'CPU_STOP_PUBLICATION' in end['STOP_publish_error']
