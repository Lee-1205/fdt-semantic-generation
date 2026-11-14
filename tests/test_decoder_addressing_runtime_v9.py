import copy,json,sys,time
from pathlib import Path
from unittest.mock import patch
import torch,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from test_decoder_alignment_runtime_v8 import fixture as make_v8,Budget,Child
from research.text_decoder_addressing_v9 import AddressedLanguage,AddressedContextDecoder,ADDRESS_POLICY
from scripts.text_decoder_addressing_engine_v9 import AddressEngine
import text_generation_support_v1 as support
import run_text_decoder_addressing_20261028_v9 as worker
import prepare_text_decoder_addressing_20261028_v9 as prepare
import supervise_text_decoder_addressing_20261028_v9 as owner
torch.set_num_threads(2)

def fixture():
    parent,_,_=make_v8();parent.step();parent.step();state=copy.deepcopy(parent.export())
    m=AddressedLanguage(parent.model.core_config,parent.model.policy)
    e=AddressEngine(m,parent.features,parent.targets,parent.negatives,Budget(),'V9CPU','cpu')
    e.import_parent(state,'V8CPU');return e,parent,state

def test_visible_memory_count_gives_half_prior_to_own_slot_without_length_or_gold():
    e,_,_=fixture();m=e.model.decoder;valid=torch.ones(2,16,dtype=torch.bool);valid[1,:3]=False
    bias=m.address_bias(valid,12);prob=bias.softmax(-1);slots=4+torch.arange(12)
    assert torch.allclose(prob[:,:,torch.arange(12),slots],torch.full((2,1,12),.5),atol=1e-6)
    assert prob[1,:,:,:3].sum()==0 and ADDRESS_POLICY['uses_gold_length_or_positions'] is False
    with pytest.raises(ValueError):m.address_bias(valid,13)

def test_order_sensitive_context_and_causal_prefix():
    e,_,_=fixture();m=e.model.decoder.eval();prompt=torch.randn(2,4,264);mask=torch.ones(2,4,dtype=torch.bool)
    z=torch.randn(2,12,264);prefix=torch.ones(2,12,dtype=torch.long);later=prefix.clone();later[:,6:]=7
    with torch.no_grad():
        a=m(prefix,m.context(prompt,mask,z));b=m(prefix,m.context(prompt,mask,z.flip(1)));c=m(later,m.context(prompt,mask,z))
    assert (a-b).abs().max()>.01 and torch.equal(a[:,:6],c[:,:6])

def test_all_V8_tensors_moments_scaler_RNG_imported_no_new_parameters_or_position_reset():
    e,p,state=fixture();assert support.state_sha(e.model.state_dict())==support.state_sha(p.model.state_dict())
    assert support.state_sha(e.optimizer.state_dict())==support.state_sha(p.optimizer.state_dict())
    assert e.scaler.state_dict()==p.scaler.state_dict() and e.ancestry['old_component_sha256']
    assert torch.count_nonzero(e.model.decoder.memory_positions.weight)>0
    assert e.parameter_names==p.parameter_names

def test_fullstate_exact_next_and_four_last7_epoch_batches(tmp_path):
    e,_,_=fixture();vp=support.state_sha(e.model.vp.state_dict());e.step();e.step();assert e.cursor==15
    saved=support.save_full(tmp_path,copy.deepcopy(e.export()));state=torch.load(tmp_path/'checkpoints'/saved['file'],map_location='cpu',weights_only=True)
    expected=e.step();end=copy.deepcopy(e.export());n,_,_=fixture();n.restore(state)
    assert n.step()==expected and support.state_sha(n.export())==support.state_sha(end)
    while n.attempts<n.limit:n.step()
    assert n.cursor==60 and n.updates==8 and sum(len(b)==7 for b in n.batches)==4
    assert support.state_sha(n.model.vp.state_dict())==vp and all(r['span_UL']==0 for r in n.history)

def test_real_enabled_scaler_skip_consumes_without_replay_or_frozen_mutation():
    e,_,_=fixture();before=support.state_sha(e.model.state_dict());scale=e.scaler.get_scale()
    hook=e.model.decoder.memory_positions.weight.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:e.step()
    finally:hook.remove()
    assert e.cursor==8 and e.updates==0 and e.skips==1 and e.scaler.get_scale()==scale/2
    assert before==support.state_sha(e.model.state_dict());n,_,_=fixture();n.restore(copy.deepcopy(e.export()));assert len(n.next_indices())==7

def test_float_attention_bias_under_CPU_autocast_preserves_all_required_gradients():
    e,_,_=fixture()
    with torch.autocast('cpu',dtype=torch.bfloat16):record=e.step()
    assert record['outcome']=='UPDATED' and e.updates==1
    assert all(e.optimizer.state[p] for n,p in e.model.named_parameters() if p.requires_grad)

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_addressed_schema_postPopen_cleanup(tmp_path,crash,confirmed):
    p=prepare.make_plan(tmp_path);p.update(authorized=True,allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8')
    owner.validate(path,tmp_path);worker.validate(path,tmp_path,ROOT/'scripts/run_text_decoder_addressing_20261028_v9.py')
    patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'TEXT_DECODER_ADDRESSING_V9_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            code=owner.supervise(path,tmp_path,ROOT/'scripts/run_text_decoder_addressing_20261028_v9.py')
        d=json.loads((tmp_path/'supervisor/final_status.json').read_text())
        assert cleanup.call_count==1 and code==(0 if confirmed and not crash else 2)
        assert d['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()
