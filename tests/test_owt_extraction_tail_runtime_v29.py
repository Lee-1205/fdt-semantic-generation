import copy,json,random,sys,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from fdt_local.model_masked import Config
from research.owt_prefix_hold_v28 import PrefixHoldFuzzyAE
from research.owt_extraction_tail_v29 import ExtractionTailFuzzyAE,TRAINABILITY
from owt_fuzzy_engine_v28 import ContinuationEngine as OldEngine
from owt_fuzzy_engine_v29 import ContinuationEngine,old_optimizer_view,cumulative_count
from owt_resource_budget_v24 import RecoveryBudget as OldBudget,RESOURCE_POLICY as OLD_POLICY
from owt_resource_budget_v29 import RecoveryBudget,RESOURCE_POLICY
import owt_resource_budget_v29 as guard
import text_generation_support_v1 as support
from test_decoder_alignment_runtime_v8 import Child
import run_owt_fuzzy_reconstruction_20261029_v29 as worker
import supervise_owt_fuzzy_reconstruction_20261029_v29 as owner
torch.set_num_threads(2)

def policy(path,resource):
    return {'output':str(path),'cutoff':time.time()+3600,'resource_policy':resource,
        'external_closed_assessment_calls':{'total':1936,'included_in_source_ledger':False},
        'limits':{'max_calls':100000,'max_output_bytes':6*1024**3}}

def fixture(path):
    torch.manual_seed(371);random.seed(371);np.random.seed(371)
    c=Config(lang_dim=24,lang_heads=3,lang_ff=32,lang_layers=5,z_dim=32,core_heads=4,core_ff=40,diffusion_layers=1,anchors=16,key_dim=16,value_dim=16,top_k=4)
    a=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v23_final'
    f=json.loads((a/'features.json').read_text(encoding='utf-8'))['train'][:23]
    y=json.loads((a/'targets.json').read_text(encoding='utf-8'))['train'][:23]
    old=OldEngine(PrefixHoldFuzzyAE(c),f,y,worker.finite_orders(23,[361]),OldBudget(policy(path,OLD_POLICY)),'CPU28','cpu')
    old.step();old.step();s=copy.deepcopy(old.export())
    e=ContinuationEngine(ExtractionTailFuzzyAE(c),f,y,worker.finite_orders(23,[371]),RecoveryBudget(policy(path,RESOURCE_POLICY)),'CPU29','cpu',source_parameter_groups=s['optimizer_parameter_names'])
    migration=e.import_parent(s,'CPU28','CPU_SOURCE28');return e,s,migration

def test_old_populated_Adam_exact_append18_empty_source_RNG_frozen_scopes(tmp_path):
    e,s,m=fixture(tmp_path);out=e.export()
    assert len(e.names[2])==18 and len(e.optimizer.state)==63 and m['new_state_count_at_import']==0
    assert support.state_sha(out['model'])==support.state_sha(s['model'])
    assert support.state_sha(old_optimizer_view(out['optimizer'],s['optimizer']))==support.state_sha(s['optimizer'])
    assert m['source_frozen_sha']==s['frozen_sha'] and m['phase_frozen_sha']!=s['frozen_sha']
    assert m['new_parameter_names']==e.names[2] and out['trainability']==TRAINABILITY
    for k in ('scaler','torch_rng','cuda_rng','python_rng','numpy_rng'):assert support.state_sha(out[k])==support.state_sha(s[k])
    assert e.ancestry['ancestry']==s['ancestry'] and e.ancestry['external_closed_assessment_calls']==s['external_closed_assessment_calls']
    assert e.optimizer.param_groups[2]['base_lr']==3e-6
    before=support.state_sha(s);seen=set();handles=[]
    for n,p in e.model.named_parameters():
        if p.requires_grad:handles.append(p.register_hook(lambda g,name=n:seen.add(name)))
    try:e.step()
    finally:
        for h in handles:h.remove()
    assert seen==set(n for group in e.names for n in group)
    assert support.state_sha(s)==before and len(e.optimizer.state)==81
    for group in e.optimizer.param_groups[:2]:assert {int(e.optimizer.state[p]['step']) for p in group['params']}=={3}
    assert {int(e.optimizer.state[p]['step']) for p in e.optimizer.param_groups[2]['params']}=={1}
    e.verify_frozen()

def test_enabled_scaler_AMPskip_appendgroup_fullstate_exact_next_last7(tmp_path):
    e,s,_=fixture(tmp_path);e.step();before=support.state_sha(e.model.state_dict())
    hook=dict(e.model.named_parameters())[e.names[2][0]].register_hook(lambda g:torch.full_like(g,float('inf')))
    try:assert e.step()['AMP_skip']
    finally:hook.remove()
    assert support.state_sha(e.model.state_dict())==before and e.scaler.is_enabled()
    saved=support.save_full(tmp_path,copy.deepcopy(e.export()))
    snap=torch.load(tmp_path/'checkpoints'/saved['file'],weights_only=True)
    assert len(e.next_indices())==7 and snap['scaler']['scale']==512
    expected=e.step();end=copy.deepcopy(e.export());n,_,_=fixture(tmp_path);n.restore(snap)
    assert n.budget.last_resource_check is None and n.model.hold_attempt==2
    assert n.step()==expected and support.state_sha(n.export())==support.state_sha(end)
    assert n.attempts==3 and n.updates==2 and n.cursor==23 and n.skips==1
    assert n.budget.allocation_calls=={'encode':3,'denoise':0,'decoder':6}
    for group in n.optimizer.param_groups[:2]:assert {int(n.optimizer.state[p]['step']) for p in group['params']}=={4}
    assert {int(n.optimizer.state[p]['step']) for p in n.optimizer.param_groups[2]['params']}=={2}
    assert cumulative_count(n.export(),'updates')==4
    n.verify_frozen()
    with pytest.raises(StopIteration):n.step()

def test_new_write_reserve_and_same_corpus_finite_epoch_actual_plan():
    path=ROOT/'runs/owt_fuzzy_reconstruction_20261029_v29/plan.json'
    p=json.loads(path.read_text());_,f,_,orders=worker.validate(path,Path(p['output']),worker.__file__)
    assert RESOURCE_POLICY['checkpoint_write_reserve_bytes']==1024**3 and RESOURCE_POLICY['worker_disk_reserve_bytes']==22*1024**3
    assert len(orders)==4096 and sorted(i for b in orders for i in b)==list(range(32768))
    assert sum(len(r['prompt'].encode()) for r in f['train'])==2926478 and worker.COST['total_max']==14488
    assert p['source_global_updates']==88005 and p['limits']['max_calls']==15000

def test_projected_write_and_immediate_controls(tmp_path):
    p=policy(tmp_path,RESOURCE_POLICY);b=RecoveryBudget(p)
    with patch.object(guard.shutil,'disk_usage',return_value=SimpleNamespace(free=22*1024**3+1024**3-1)),patch.object(guard,'owned_bytes',return_value=0):
        with pytest.raises(OSError):b.check_resources(force=True,upcoming_bytes=1024**3)
    (tmp_path/'STOP').write_text('CPU')
    with pytest.raises(InterruptedError):b.charge('encode')

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_schema_postPopen_cleanup(tmp_path,crash,confirmed):
    p=json.loads((ROOT/'runs/owt_fuzzy_reconstruction_20261029_v29/plan.json').read_text())
    p.update(authorized=True,output=str(tmp_path),allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p));owner.validate(path,tmp_path);worker.validate(path,tmp_path,worker.__file__);patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'OWT_FUZZY_RECONSTRUCTION_V29_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:code=owner.supervise(path,tmp_path,Path(worker.__file__))
        end=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2) and end['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()
