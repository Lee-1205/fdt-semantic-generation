import copy,json,sys,time
from pathlib import Path
from unittest.mock import patch
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from research.text_context_decoder_v2 import ContextDecoder,BOS
from research.text_diffusion_v1 import TextDiffusion,TextConfig,pack_prompts,pack_answers
import run_text_context_decoder_20261028 as worker
import supervise_text_context_decoder_20261028 as owner
import text_generation_support_v1 as support
torch.set_num_threads(2)

def cfg():return Config(lang_dim=32,lang_heads=4,lang_ff=64,lang_layers=1,z_dim=272,core_heads=8,core_ff=64,diffusion_layers=2)

def test_causal_decoder_cannot_see_future_gold_and_has_prompt_gradient():
    d=ContextDecoder(cfg(),length=6,layers=1)
    p=torch.randn(2,5,272);m=torch.ones(2,5,dtype=torch.bool);z=torch.randn(2,6,272)
    context=d.context(p,m,z);a=torch.tensor([[BOS,65,66,67,68,256]]*2);b=a.clone();b[:,3:]=70
    la=d(a,context);lb=d(b,context)
    assert torch.allclose(la[:,:3],lb[:,:3],atol=1e-6) and not torch.allclose(la[:,3:],lb[:,3:])
    loss=d.loss(pack_answers(['AB','CD'],6),context);loss.backward()
    assert torch.isfinite(loss) and d.context_types.grad.abs().sum()>0
    assert d.head.weight.grad.abs().sum()>0

def test_cached_proposal_is_passive_same_as_original_sampler():
    g=TextDiffusion(cfg(),TextConfig(prompt_bytes=8,answer_bytes=6,steps=16)).eval()
    tokens=pack_prompts(['AB'],8);noise=torch.randn(1,6,272);calls=[]
    before=support.state_sha(g.state_dict());rng=torch.get_rng_state().clone()
    enc,z=worker.proposal(g,tokens,noise,calls.append);_,reference=g.generate(tokens,noise)
    assert torch.equal(z,reference) and calls==['encode']+['denoise']*16
    assert before==support.state_sha(g.state_dict()) and torch.equal(rng,torch.get_rng_state())

def test_zero_latent_keeps_prompt_and_raw_logits_finite():
    d=ContextDecoder(cfg(),length=6,layers=1)
    p=torch.randn(2,5,272);mask=torch.ones(2,5,dtype=torch.bool);z=torch.randn(2,6,272)
    a=d.context(p,mask,z);b=d.context(p,mask,z,True)
    assert torch.equal(a[0][:,:5],b[0][:,:5]) and not torch.equal(a[0][:,5:],b[0][:,5:])
    calls=[];ids=d.generate(a,calls.append)
    assert ids.shape==(2,6) and 1<=len(calls)<=6 and (ids<258).all()

class Child:
    pid=567;returncode=None
    def poll(self):self.returncode=0;return 0
    def wait(self,timeout=None):self.returncode=0;return 0

@pytest.mark.parametrize('crash',[False,True])
def test_actual_v2_plan_schema_full_post_popen_cleanup(tmp_path,crash):
    pp=ROOT/'runs/text_context_decoder_20261028_v2/plan.json';plan=json.loads(pp.read_text(encoding='utf-8'))
    plan.update(output=str(tmp_path),authorized=True,allocation_start=time.time());plan.update(deadline=plan['allocation_start']+900,cutoff=plan['allocation_start']+810)
    testplan=tmp_path/'plan.json';testplan.write_text(json.dumps(plan));owner.validate(testplan,tmp_path)
    worker.validate(testplan,tmp_path,ROOT/'scripts/run_text_context_decoder_20261028.py')
    patches=[]
    def popen(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def bad(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('V2postPopen')
                return original(path,value)
            monkey=patch.object(owner.io,'atomic_json',bad);monkey.start();patches.append(monkey)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'CONTEXT_DECODER_V2_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=popen),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':True}) as cleanup:
            rc=owner.supervise(testplan,tmp_path,ROOT/'scripts/run_text_context_decoder_20261028.py')
        state=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert rc==(2 if crash else 0) and state['status']==('STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()

def test_expired_v2_rejects_before_popen(tmp_path):
    source=ROOT/'runs/text_context_decoder_20261028_v2/plan.json';plan=json.loads(source.read_text())
    plan.update(output=str(tmp_path),authorized=True,allocation_start=time.time()-1000)
    plan.update(deadline=plan['allocation_start']+900,cutoff=plan['allocation_start']+810)
    pp=tmp_path/'plan.json';pp.write_text(json.dumps(plan))
    with patch.object(owner.subprocess,'Popen') as p:
        with pytest.raises(TimeoutError):owner.supervise(pp,tmp_path,ROOT/'scripts/run_text_context_decoder_20261028.py')
        p.assert_not_called()

@pytest.mark.skipif(sys.platform!='win32',reason='real Windows')
def test_v2_real_windows_complete_owner_cpu_only(tmp_path):
    out=tmp_path/'native';out.mkdir();stub=tmp_path/'stub.py'
    stub.write_text("import argparse,os,json,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');p.add_argument('--preflight',action='store_true');a=p.parse_args()\nif a.preflight:print('CPU_PREFLIGHT_PASS');raise SystemExit(0)\no=Path(a.output);(o/'progress.json').write_text(json.dumps({'phase':'cpu','attempts':0,'native_pid':os.getpid()}));time.sleep(2);(o/'result.json').write_text(json.dumps({'status':'CONTEXT_DECODER_V2_COMPLETE','model_calls':0}))\n")
    source=ROOT/'runs/text_context_decoder_20261028_v2/plan.json';plan=json.loads(source.read_text());plan.update(output=str(out),authorized=True,allocation_start=time.time());plan.update(deadline=plan['allocation_start']+900,cutoff=plan['allocation_start']+810)
    plan['files']={str(Path(owner.__file__).resolve()):support.sha(owner.__file__),str(stub.resolve()):support.sha(stub)}
    pp=out/'plan.json';pp.write_text(json.dumps(plan))
    with patch.object(owner,'resources',return_value={'cpu_fixture':True}):rc=owner.supervise(pp,out,stub)
    result=json.loads((out/'supervisor/final_status.json').read_text());assert rc==0 and result['exit']['all_owned_processes_gone']
