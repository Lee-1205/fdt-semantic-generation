import copy,json,sys,time
from pathlib import Path
from unittest.mock import patch
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_owt_prefix_curriculum_v27 import fixture as make_source
from test_owt_resource_continuation_v24 import plan
from test_decoder_alignment_runtime_v8 import Child
from owt_resource_budget_v24 import RecoveryBudget,RESOURCE_POLICY
from research.owt_prefix_curriculum_v27 import PrefixCurriculumFuzzyAE as OldModel
from research.owt_prefix_hold_v28 import PrefixHoldFuzzyAE,mask_probability,OBJECTIVE,SOURCE_OBJECTIVE
from research.text_diffusion_v1 import PAD
from research.text_context_decoder_v2 import BOS
from owt_fuzzy_engine_v28 import ContinuationEngine,COMPONENTS,cumulative_count
import text_generation_support_v1 as support
import run_owt_fuzzy_reconstruction_20261029_v28 as worker
import supervise_owt_fuzzy_reconstruction_20261029_v28 as owner
torch.set_num_threads(2)

def fixture(path):
    source,_,_=make_source(path);source.step();source.step();s=copy.deepcopy(source.export())
    p=plan(path);p['external_closed_assessment_calls']=copy.deepcopy(s['external_closed_assessment_calls'])
    b=RecoveryBudget(p);m=PrefixHoldFuzzyAE(source.model.c)
    e=ContinuationEngine(m,source.features,source.targets,worker.finite_orders(23,[361,362]),b,'CPU28','cpu')
    digest=e.import_parent(s,'CPU27','CPU_SOURCE27');return e,s,digest

def test_actual_V27_import_enabled_scaler_AMPskip_complete_exact_next_last7(tmp_path):
    e,s,digest=fixture(tmp_path);assert s['schema']=='fdt_owt_fuzzy_AE_v27_fullstate'
    assert digest==support.state_sha({k:s[k] for k in COMPONENTS})==support.state_sha({k:e.export()[k] for k in COMPONENTS})
    before=support.state_sha(s);e.step();assert before==support.state_sha(s)
    hook=e.model.core.anchors.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:assert e.step()['AMP_skip']
    finally:hook.remove()
    saved=support.save_full(tmp_path,copy.deepcopy(e.export()));snap=torch.load(tmp_path/'checkpoints'/saved['file'],weights_only=True)
    assert snap['external_closed_assessment_calls']==e.budget.plan['external_closed_assessment_calls']
    expected=e.step();final=copy.deepcopy(e.export());other,_,_=fixture(tmp_path);other.restore(snap)
    assert other.budget.last_resource_check is None and other.model.hold_attempt==snap['attempts']
    assert snap['training_objective']==OBJECTIVE and s['training_objective']==SOURCE_OBJECTIVE
    assert snap['constant_mask_probability']==mask_probability(snap['attempts'])
    assert other.step()==expected and support.state_sha(other.export())==support.state_sha(final)
    while other.attempts<len(other.orders):other.step()
    assert other.cursor==46 and other.attempts==6 and other.updates==5
    assert other.budget.allocation_calls=={'encode':6,'denoise':0,'decoder':12}
    assert all(h['mask_probability']==mask_probability(h['attempt']) and h['hold_attempt']==h['attempt'] for h in other.history)
    assert cumulative_count(other.export(),'updates')==cumulative_count(s,'updates')+5
    assert other.ancestry['ancestry']==s['ancestry'] and other.ancestry['external_closed_assessment_calls']==s['external_closed_assessment_calls'];other.verify_frozen()

def test_same_corpus_new_finite_order_caps_and_external_ledger():
    path=ROOT/'runs/owt_fuzzy_reconstruction_20261029_v28/plan.json'
    p=json.loads(path.read_text());_,f,_,orders=worker.validate(path,Path(p['output']),worker.__file__)
    assert len(orders)==8192 and p['external_closed_assessment_calls']['total']==1936
    assert worker.COST['total_max']==26776 and 2*sum(len(r['prompt'].encode()) for r in f['train'])==5852956
    for epoch in range(2):assert sorted(i for b in orders[epoch*4096:(epoch+1)*4096] for i in b)==list(range(32768))

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_schema_postPopen_cleanup(tmp_path,crash,confirmed):
    p=json.loads((ROOT/'runs/owt_fuzzy_reconstruction_20261029_v28/plan.json').read_text())
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
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'OWT_FUZZY_RECONSTRUCTION_V28_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            code=owner.supervise(path,tmp_path,Path(worker.__file__))
        end=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2) and end['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()


def test_constant_p50_starts_without_reramp_matches_old_endpoint_and_rng(tmp_path):
    assert mask_probability(1)==mask_probability(4096)==mask_probability(8192)==.50
    for invalid in (0,8193,True,1.5):
        with pytest.raises(ValueError):mask_probability(invalid)
    e,source,_=fixture(tmp_path);model=e.model;old=OldModel(model.c);old.load_state_dict(source['model'],strict=True)
    assert set(model.state_dict())==set(old.state_dict())
    target=torch.full((2,129),PAD,dtype=torch.long);target[:,:60]=torch.arange(60);target[:,60]=256
    rng=torch.get_rng_state();model.train();old.train();model.set_training_attempt(1);old.set_training_attempt(4096)
    held=model.corrupt_teacher_prefix(target);after=torch.get_rng_state();torch.set_rng_state(rng);prior_end=old.corrupt_teacher_prefix(target)
    assert all(torch.equal(a,b) for a,b in zip(held,prior_end)) and torch.equal(after,torch.get_rng_state())
    torch.set_rng_state(rng);model.set_training_attempt(8192);last=model.corrupt_teacher_prefix(target)
    assert all(torch.equal(a,b) for a,b in zip(held,last)) and torch.equal(after,torch.get_rng_state())
    clean=torch.cat((torch.full_like(target[:,:1],BOS),target[:,:-1]),1)
    assert not held[1][(clean==256)|(clean==PAD)|(clean==BOS)].any()
    torch.set_rng_state(rng);draw=torch.rand(target.shape);assert torch.equal(after,torch.get_rng_state())
    assert torch.equal(held[1],(draw<.5)&held[2])
    model.eval();old.eval();meaning={'views':torch.randn(2,2,model.c.z_dim)}
    with torch.no_grad():assert torch.equal(model.teacher_logits(meaning,target),old.teacher_logits(meaning,target))
    with pytest.raises(RuntimeError):model.corrupt_teacher_prefix(target)


def test_future_bytes_lengths_do_not_change_past_masks_or_causal_logits(tmp_path):
    e,_,_=fixture(tmp_path);model=e.model;model.train();model.set_training_attempt(4095)
    a=torch.full((2,129),PAD,dtype=torch.long);a[:,:40]=torch.arange(40);a[:,40]=256
    b=a.clone();b[:,40:90]=123;b[:,90]=256
    rng=torch.get_rng_state();prefix_a,mask_a,_=model.corrupt_teacher_prefix(a);after=torch.get_rng_state()
    torch.set_rng_state(rng);prefix_b,mask_b,_=model.corrupt_teacher_prefix(b)
    assert torch.equal(after,torch.get_rng_state()) and torch.equal(prefix_a[:,:41],prefix_b[:,:41]) and torch.equal(mask_a[:,:41],mask_b[:,:41])
    model.eval();meaning={'views':torch.randn(2,2,model.c.z_dim)}
    with torch.no_grad():
        x=model.decoder(prefix_a,model.context(meaning));y=model.decoder(prefix_b,model.context(meaning))
        assert torch.equal(x[:,:41],y[:,:41])
