import copy,json,random,sys,time
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from research.text_language_continuation_v6 import LanguageContinuation,ContinuationConfig
from text_language_engine_v6 import RecoveryEngine as LanguageEngine,finite_order,row_noise
import run_text_language_continuation_20261028_v6 as worker
import supervise_text_language_continuation_20261028_v6 as owner
import prepare_text_language_continuation_20261028_v6 as prepare
import text_generation_support_v1 as support
torch.set_num_threads(2)


class FakeBudget:
    def __init__(self):self.calls=Counter(encode=0,denoise=0,decoder=0);self.stopped=False
    def check(self):
        if self.stopped:raise InterruptedError('CPU STOP fixture')
    def charge(self,kind):self.check();self.calls[kind]+=1


def fixture(count=15):
    torch.manual_seed(91);random.seed(91);np.random.seed(91)
    c=Config(name='CPU-runtime-fixture',lang_dim=32,lang_heads=4,lang_ff=64,lang_layers=3,
        z_dim=264,core_heads=4,core_ff=64,diffusion_layers=1)
    model=LanguageContinuation(c,ContinuationConfig(32,12,2,1));features=[];targets=[]
    for i in range(count):
        text=' wet.' if i%2 else ' sunshine.'
        features.append({'id':str(i),'prompt':'Rain fell on a little roof.'})
        targets.append({'target':text,'target_ids':list(text.encode())+[256],
            'source_eof':True,'document_id':str(i)})
    return LanguageEngine(model,features,targets,finite_order(count),FakeBudget(),'CPU-PLAN','cpu')


def test_full_driver_model_Adam_enabled_scaler_all_rng_exact_next_and_last_seven(tmp_path):
    engine=fixture();engine.step();state=copy.deepcopy(engine.export())
    saved=support.save_full(tmp_path,state);loaded=torch.load(tmp_path/'checkpoints'/saved['file'],map_location='cpu',weights_only=True)
    assert loaded['optimizer']['state'] and loaded['scaler']['scale']==1024
    expected=engine.step();end=copy.deepcopy(engine.export())
    new=fixture();new.restore(loaded);assert len(new.next_indices())==7
    actual=new.step();assert actual==expected and support.state_sha(new.export())==support.state_sha(end)
    assert new.cursor==15 and new.exposure['selected_rows']==15 and new.updates==2
    with pytest.raises(StopIteration):new.next_indices()
    new.verify_frozen()


def test_actual_enabled_AMPskip_consumes_no_replay_and_three_stops():
    engine=fixture(24);before=support.state_sha(engine.model.state_dict())
    hook=engine.model.decoder.head.weight.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:
        engine.step();engine.step()
        with pytest.raises(FloatingPointError,match='AMP'):engine.step()
    finally:hook.remove()
    assert engine.attempts==3 and engine.updates==0 and engine.cursor==24 and engine.skips==3
    assert engine.scaler.state_dict()['scale']==128 and not engine.optimizer.state
    assert engine.exposure['processed_rows']==24 and before==support.state_sha(engine.model.state_dict())
    restored=fixture(24);restored.restore(copy.deepcopy(engine.export()));assert restored.cursor==24


def test_failed_attempt_selected_vs_processed_and_STOP_before_consume():
    engine=fixture()
    with patch.object(engine.model,'training_loss',side_effect=FloatingPointError('CPU injected failure')):
        with pytest.raises(FloatingPointError):engine.step()
    assert engine.cursor==8 and engine.updates==0 and engine.exposure['processed_rows']==0
    assert engine.history[-1]['outcome']=='FAILED_ATTEMPT'
    restored=fixture();restored.restore(copy.deepcopy(engine.export()));assert len(restored.next_indices())==7
    restored.budget.stopped=True
    with pytest.raises(InterruptedError):restored.step()
    assert restored.attempts==1


def test_restore_rejects_changed_order_missing_scaler_or_corrupt_history():
    e=fixture();e.step();p=copy.deepcopy(e.export())
    for change in ('order','scaler','history'):
        bad=copy.deepcopy(p)
        if change=='order':bad['order'][:2]=bad['order'][:2][::-1]
        elif change=='scaler':bad['scaler']={}
        else:bad['history'][0]['rows']=[]
        with pytest.raises(ValueError):fixture().restore(bad)


def test_microbatch_objective_weight_uses_actual_active_symbol_mass():
    e=fixture(8)
    def analytic(tokens,ids,t,noise,proposal,charge):
        charge('encode');charge('denoise');charge('decoder')
        active=ids!=257;value=ids.float().masked_fill(~active,0).sum()/active.sum()
        loss=sum(p.sum()*0 for p in e.model.parameters() if p.requires_grad)+value
        return loss,{k:value for k in ('velocity_mse','clean_mse','VP_byte_ce','AR_byte_ce')}
    expected=sum(sum(t['target_ids']) for t in e.targets)/sum(len(t['target_ids']) for t in e.targets)
    with patch.object(e.model,'training_loss',side_effect=analytic):record=e.step()
    assert abs(record['loss']-expected)<1e-4


def test_goldfree_generation_reuses_proposal_zero_control_and_NLL_is_separate():
    e=fixture(4);normal=worker.free_generate(e.model,e.features,e.budget,'cpu')
    assert len(normal['normal'])==len(normal['zero'])==4
    assert e.budget.calls['encode']==1 and e.budget.calls['denoise']==2
    with patch.object(e.model,'training_loss',side_effect=AssertionError('training loss during eval')):
        nll,tensors=worker.evaluate_nll(e.model,e.features,e.targets,e.budget,'cpu')
    assert nll['symbols']==sum(len(t['target_ids']) for t in e.targets) and tensors['logits'].shape==(4,12,258)
    assert nll['true_EOF']==4
    a=row_noise(e.features,12,264,17,'cpu');b=torch.cat([row_noise(e.features[:2],12,264,17,'cpu'),row_noise(e.features[2:],12,264,17,'cpu')])
    assert torch.equal(a,b)


def execution_plan(tmp_path):
    p=prepare.make_plan(tmp_path);p['authorized']=True;p['allocation_start']=time.time()
    p['deadline']=p['allocation_start']+600;p['cutoff']=p['deadline']-90
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8');return p,path


class Child:
    pid=567;returncode=None
    def poll(self):self.returncode=0;return 0
    def wait(self,timeout=None):self.returncode=0;return 0


@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_full_postPopen_actual_schema_natural_exception_and_unconfirmed(tmp_path,crash,confirmed):
    plan,path=execution_plan(tmp_path);owner.validate(path,tmp_path)
    worker.validate(path,tmp_path,ROOT/'scripts/run_text_language_continuation_20261028_v6.py')
    patches=[]
    def popen(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def bad(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            p=patch.object(owner.io,'atomic_json',bad);p.start();patches.append(p)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'TEXT_LANGUAGE_V6_BOUNDED_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=popen),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            rc=owner.supervise(path,tmp_path,ROOT/'scripts/run_text_language_continuation_20261028_v6.py')
        result=json.loads((tmp_path/'supervisor/final_status.json').read_text(encoding='utf-8'))
        assert cleanup.call_count==1 and rc==(0 if confirmed and not crash else 2)
        assert result['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for p in patches:p.stop()


def test_expired_schema_rejects_before_preflight_or_Popen(tmp_path):
    plan,path=execution_plan(tmp_path);plan.update(allocation_start=time.time()-1000)
    plan.update(deadline=plan['allocation_start']+600,cutoff=plan['allocation_start']+510)
    path.write_text(json.dumps(plan),encoding='utf-8')
    with patch.object(owner.subprocess,'Popen') as launch:
        with pytest.raises(TimeoutError):owner.supervise(path,tmp_path,ROOT/'scripts/run_text_language_continuation_20261028_v6.py')
        launch.assert_not_called()


@pytest.mark.skipif(sys.platform!='win32',reason='real Windows only')
def test_REAL_Windows_full_owner_Popen_proxy_child_CIM_exit_CPU(tmp_path):
    output=tmp_path/'native';output.mkdir();stub=tmp_path/'stub.py'
    stub.write_text("import argparse,os,json,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');p.add_argument('--preflight',action='store_true');a=p.parse_args()\nif a.preflight:print('CPU_PREFLIGHT_PASS');raise SystemExit(0)\no=Path(a.output);(o/'progress.json').write_text(json.dumps({'phase':'cpu','attempts':0,'native_pid':os.getpid()}));time.sleep(2);(o/'result.json').write_text(json.dumps({'status':'TEXT_LANGUAGE_V6_BOUNDED_COMPLETE','model_calls':0}))\n",encoding='utf-8')
    plan,path=execution_plan(output)
    plan['files']={str(Path(owner.__file__).resolve()):support.sha(owner.__file__),str(stub.resolve()):support.sha(stub)}
    path.write_text(json.dumps(plan),encoding='utf-8')
    with patch.object(owner,'resources',return_value={'CPU_fixture_only':True}):rc=owner.supervise(path,output,stub)
    final=json.loads((output/'supervisor/final_status.json').read_text(encoding='utf-8'))
    assert rc==0 and final['exit']['all_owned_processes_gone']


def test_masked_nonreduced_CE_equals_original_CPU_value_and_gradient():
    from torch.nn import functional as F
    torch.manual_seed(72)
    a=torch.randn(3,11,258,requires_grad=True);b=a.detach().clone().requires_grad_()
    gold=torch.randint(0,257,(3,11));gold[0,5:]=-100;gold[1,8:]=-100
    old=F.cross_entropy(a.transpose(1,2),gold,ignore_index=-100)
    new=F.cross_entropy(b.transpose(1,2),gold,ignore_index=-100,reduction='none').sum()/(gold!=-100).sum()
    old.backward();new.backward()
    assert torch.allclose(old,new,atol=1e-6) and torch.allclose(a.grad,b.grad,atol=1e-7)


def test_CPU_AMP_all_required_gradients_and_forward_value_preserved():
    from research.text_language_continuation_v5_deterministic_v2 import LanguageContinuation as Old
    from research.text_diffusion_v1 import pack_prompts
    e=fixture(4);old=Old(e.model.core_config,e.model.policy);old.load_state_dict(e.model.state_dict())
    tokens=pack_prompts([r['prompt'] for r in e.features],32)
    ids=torch.full((4,12),257,dtype=torch.long)
    for i,t in enumerate(e.targets):ids[i,:len(t['target_ids'])]=torch.tensor(t['target_ids'])
    args=(tokens,ids,torch.full((4,),.5),torch.randn(4,12,264),torch.randn(4,12,264))
    with torch.autocast('cpu',dtype=torch.bfloat16):a,_=old.training_loss(*args)
    a.backward();assert old.vp.velocity.weight.grad is None
    with torch.autocast('cpu',dtype=torch.bfloat16):b,_=e.model.training_loss(*args)
    b.backward();assert torch.equal(a.detach(),b.detach())
    assert all(p.grad is not None for p in e.model.parameters() if p.requires_grad)


def test_missing_trainable_gradient_rejected_before_optimizer_update():
    e=fixture(8)
    original=e.model.training_loss
    def detached(*args,**kw):
        loss,out=original(*args,**kw)
        return loss.detach()+e.model.decoder.head.weight.sum()*0,out
    with patch.object(e.model,'training_loss',side_effect=detached):
        with pytest.raises(RuntimeError,match='missing required'):e.step()
    assert e.updates==0 and not e.optimizer.state


def test_vanished_atomic_temp_accounting_only_ignores_temporary_path(tmp_path):
    from text_language_runtime_support_v6 import owned_bytes
    temp=tmp_path/'state.json.atomic.tmp';temp.write_text('abcdef')
    durable=tmp_path/'important.pt';durable.write_text('12345')
    original=Path.stat
    def raced(path,*a,**kw):
        if path==temp:raise FileNotFoundError('atomic rename')
        return original(path,*a,**kw)
    with patch.object(Path,'stat',raced):assert owned_bytes(tmp_path)==5
    # A durable path disappearing between is_file and stat is not silenced.
    with patch.object(Path,'is_file',return_value=True),patch.object(Path,'stat',side_effect=FileNotFoundError('missing durable')):
        with pytest.raises(FileNotFoundError):owned_bytes(tmp_path)


def test_import_full_algorithm_preserves_model_populated_Adam_scaler_rng_cursor(tmp_path):
    e=fixture();e.step();source=copy.deepcopy(e.export());before=support.state_sha(source)
    new=fixture();new.plan_sha='NEW-CPU-PLAN'
    fingerprint=new.import_source(source,'CPU-PLAN')
    expected=support.state_sha({k:v for k,v in source.items() if k!='plan_sha256'})
    assert fingerprint==expected and new.next_indices()==e.next_indices()
    assert support.state_sha(source)==before
    assert new.optimizer.state and new.scaler.state_dict()==source['scaler']
    state=new.export();support.save_full(tmp_path,state)
