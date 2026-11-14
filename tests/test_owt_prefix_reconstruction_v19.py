import copy,json,sys,time
from pathlib import Path
from unittest.mock import patch
import torch,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_owt_prefix_reconstruction_v18 import fixture as make_source,Budget
from test_decoder_alignment_runtime_v8 import Child
from research.owt_fuzzy_reconstruction_v12 import FuzzySentenceAE
from research.owt_prefix_reconstruction_v19 import PrefixRobustFuzzyAE,OBJECTIVE
from research.text_diffusion_v1 import PAD
from research.text_context_decoder_v2 import BOS
from owt_fuzzy_engine_v19 import ContinuationEngine,COMPONENTS,cumulative_count
import text_generation_support_v1 as support
import run_owt_fuzzy_reconstruction_20261028_v19 as worker
import supervise_owt_fuzzy_reconstruction_20261028_v19 as owner
torch.set_num_threads(2)

def test_elementwise_CE_preserves_reduced_math_gradients_and_strict_determinism():
    old_setting=torch.are_deterministic_algorithms_enabled();torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(719);x=torch.randn(2,129,258,requires_grad=True);labels=torch.randint(0,257,(2,129));labels[:,90:]=-100
        count=(labels!=-100).sum()
        original=torch.nn.functional.cross_entropy(x.transpose(1,2),labels,ignore_index=-100,reduction='sum')/count
        corrected=torch.nn.functional.cross_entropy(x.transpose(1,2),labels,ignore_index=-100,reduction='none').sum()/count
        a=torch.autograd.grad(original,x,retain_graph=True)[0];b=torch.autograd.grad(corrected,x)[0]
        assert torch.allclose(original,corrected,atol=1e-6,rtol=1e-6) and torch.allclose(a,b,atol=1e-7,rtol=1e-6)
        e,_,_=fixture();e.step();assert torch.are_deterministic_algorithms_enabled()
    finally:torch.use_deterministic_algorithms(old_setting)

def fixture():
    source,_,_=make_source();source.step();source.step();state=copy.deepcopy(source.export())
    model=PrefixRobustFuzzyAE(source.model.c)
    engine=ContinuationEngine(model,source.features,source.targets,worker.finite_orders(23,[261,262]),Budget(),'CPU19','cpu')
    digest=engine.import_parent(state,'CPU18','CPU_SOURCE18');return engine,state,digest

def test_exact_source_import_same_inference_no_new_parameters_and_gradient_ownership():
    e,s,digest=fixture();out=e.export()
    assert s['schema']=='fdt_owt_fuzzy_AE_v18_fullstate' and out['schema']=='fdt_owt_fuzzy_AE_v19_fullstate'
    assert digest==support.state_sha({k:s[k] for k in COMPONENTS})==support.state_sha({k:out[k] for k in COMPONENTS})
    assert cumulative_count(s,'updates')==14 and cumulative_count(out,'updates')==14 and sum(e.budget.allocation_calls.values())==0
    base=FuzzySentenceAE(e.model.c);base.load_state_dict(s['model'],strict=True)
    assert list(base.state_dict())==list(e.model.state_dict())
    assert sum(p.numel() for p in base.parameters())==sum(p.numel() for p in e.model.parameters())
    # The existing encoder PAD is256; the separate decoder PAD is257.
    base.eval();e.model.eval();tokens=torch.tensor([[65,32,66,256]],dtype=torch.long);target=torch.full((1,129),PAD,dtype=torch.long);target[0,:4]=torch.tensor([65,32,66,256])
    with torch.no_grad():
        old=base.extract(tokens);new=e.model.extract(tokens)
        assert all(torch.equal(old[k],new[k]) for k in old)
        assert torch.equal(base.teacher_logits(old,target),e.model.teacher_logits(new,target))
    before=support.state_sha(s);record=e.step();assert support.state_sha(s)==before
    assert abs(record['loss']-.5*(record['clean_CE']+record['corrupted_prefix_CE']))<1e-5
    assert e.budget.allocation_calls=={'encode':1,'denoise':0,'decoder':2}
    assert 0<record['masked_prefix_bytes']<record['eligible_prefix_bytes']
    assert e.exposure['eligible_prefix_bytes']==e.exposure['target_UTF8_bytes']
    assert e.exposure['decoder_supervised_symbol_occurrences']==2*(e.exposure['target_UTF8_bytes']+e.exposure['processed_rows'])
    e.verify_frozen()

def test_masked_rng_fullstate_exact_next_last7_and_budget(tmp_path):
    e,_,_=fixture();e.step();e.step();saved=support.save_full(tmp_path,copy.deepcopy(e.export()))
    snapshot=torch.load(tmp_path/'checkpoints'/saved['file'],map_location='cpu',weights_only=True)
    expected=e.step();end=copy.deepcopy(e.export());other,_,_=fixture();other.restore(snapshot)
    assert other.step()==expected and support.state_sha(other.export())==support.state_sha(end)
    while other.attempts<len(other.orders):other.step()
    assert other.cursor==46 and other.updates==6 and cumulative_count(other.export(),'updates')==20
    assert other.budget.allocation_calls=={'encode':6,'denoise':0,'decoder':12}
    assert other.export()['training_objective']==OBJECTIVE

def test_enabled_scaler_skip_preserves_weights_moments_and_mask_rng_exact_restore():
    e,_,_=fixture();model_before=support.state_sha(e.model.state_dict());moment_before=support.state_sha(e.optimizer.state_dict()['state']);scale=e.scaler.get_scale()
    hook=e.model.core.anchors.register_hook(lambda grad:torch.full_like(grad,float('inf')))
    try:rec=e.step()
    finally:hook.remove()
    assert rec['AMP_skip'] and e.attempts==1 and e.updates==0 and e.cursor==8 and e.scaler.get_scale()==scale/2
    assert model_before==support.state_sha(e.model.state_dict()) and moment_before==support.state_sha(e.optimizer.state_dict()['state'])
    assert e.exposure['masked_prefix_bytes']==rec['masked_prefix_bytes']
    snapshot=copy.deepcopy(e.export());expected=e.step();end=copy.deepcopy(e.export());other,_,_=fixture();other.restore(snapshot)
    assert other.step()==expected and support.state_sha(other.export())==support.state_sha(end)

def test_corruption_uses_only_past_bytes_fixed_draws_and_causal_logits():
    e,_,_=fixture();model=e.model;model.train();target=torch.arange(129,dtype=torch.long)[None].repeat(2,1)%256
    target[:,64]=256;target[:,65:]=PAD
    rng=torch.get_rng_state();prefix,selected,eligible=model.corrupt_teacher_prefix(target)
    assert (prefix[:,0]==BOS).all() and not selected[:,0].any()
    clean=torch.cat((torch.full_like(target[:,:1],BOS),target[:,:-1]),1)
    assert not selected[(clean==256)|(clean==PAD)].any() and torch.equal(prefix[~selected],clean[~selected])
    assert (prefix[selected]==PAD).all()
    torch.set_rng_state(rng);_draw=torch.rand(target.shape);after=torch.get_rng_state();assert torch.equal(selected,(_draw<.15)&eligible)
    changed=target.clone();changed[:,80:]=123;torch.set_rng_state(rng);second,_,_=model.corrupt_teacher_prefix(changed)
    assert torch.equal(torch.get_rng_state(),after) and torch.equal(prefix[:,:81],second[:,:81])
    meaning={'views':torch.randn(2,2,model.c.z_dim)}
    with torch.no_grad():
        first_logits=model.decoder(prefix,model.context(meaning));other_logits=model.decoder(second,model.context(meaning))
        assert torch.equal(first_logits[:,:81],other_logits[:,:81])
    model.eval()
    with pytest.raises(RuntimeError):model.corrupt_teacher_prefix(target)

def test_actual_finite_orders_and_three_call_cost():
    orders=worker.finite_orders(8192,range(271,273));assert len(orders)==2048
    for epoch in range(2):assert sorted(i for b in orders[1024*epoch:1024*(epoch+1)] for i in b)==list(range(8192))
    assert worker.COST['training']==3*2048 and worker.COST['total_max']==7824 and worker.LIMITS['max_calls']==8500

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_V19_plan_postPopen_owner_cleanup(tmp_path,crash,confirmed):
    p=json.loads((ROOT/'runs/owt_fuzzy_reconstruction_20261028_v19/plan.json').read_text())
    p.update(authorized=True,output=str(tmp_path),allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8');owner.validate(path,tmp_path)
    worker.validate(path,tmp_path,ROOT/'scripts/run_owt_fuzzy_reconstruction_20261028_v19.py');patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'OWT_FUZZY_RECONSTRUCTION_V19_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            code=owner.supervise(path,tmp_path,ROOT/'scripts/run_owt_fuzzy_reconstruction_20261028_v19.py')
        result=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2) and result['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()
