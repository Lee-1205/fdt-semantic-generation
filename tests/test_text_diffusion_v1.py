import copy,importlib.util,json,sys,time
from pathlib import Path
from unittest.mock import patch
import pytest
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from research.text_diffusion_v1 import *
from research.text_generation_data_v1 import build,check,write_new
import text_generation_support_v1 as support
import supervise_text_generation_20261028 as owner
torch.set_num_threads(2)

def tiny():
    return TextDiffusion(Config(lang_dim=32,lang_heads=4,lang_ff=64,lang_layers=1,z_dim=272,
        core_heads=8,core_ff=64,diffusion_layers=1),TextConfig(prompt_bytes=8,answer_bytes=4,steps=2))

def test_codec_all_bytes_eos_pad_roundtrip():
    model=tiny();ids=torch.arange(258).view(1,-1)
    assert torch.equal(model.readout(model.target(ids)).argmax(-1),ids)
    assert decode_bytes([65,256,257])['text']=='A'
    assert not decode_bytes([255,256])['utf8_valid']
    assert not decode_bytes([65,257,256])['well_formed']
    with pytest.raises(ValueError):pack_answers(['가'*20],40)

def test_generation_only_prompt_noise_and_exact_method_counts():
    model=tiny().eval();tokens=pack_prompts(['AB','BA'],8);noise=torch.randn(2,4,272)
    seen=[];before=support.state_sha(model.state_dict());rng=torch.get_rng_state().clone()
    a,z=model.generate(tokens,noise,seen.append);b,_=model.generate(tokens,noise)
    assert torch.equal(a,b) and torch.equal(rng,torch.get_rng_state())
    assert seen==['encode','denoise','denoise']
    assert torch.isfinite(z).all() and before==support.state_sha(model.state_dict())

def test_vp_objective_gradient_ownership_and_prompt_dependence():
    model=tiny();tokens=pack_prompts(['AB','BA'],8);ids=pack_answers(['x','y'],4)
    noise=torch.randn(2,4,272);loss,parts=model.objective(model.encode(tokens),ids,torch.tensor([.3,.8]),noise)
    loss.backward();assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.language.parameters())
    assert model.prompt_project.weight.grad.abs().sum()>0
    model.eval();a,_=model.generate(tokens,noise);_,za=model.generate(tokens,noise);_,zb=model.generate(tokens.flip(0),noise)
    assert not torch.equal(za,zb)

def test_fixed_data_partition_labels_and_no_overwrite(tmp_path):
    a=build();b=build();assert a==b;review=check(a);assert review['rows']['train']>=2048
    assert set(r['group'] for r in a['train']).isdisjoint(r['group'] for r in a['dev_pairs'])
    assert all(r['template']==4 for r in a['dev_wording'])
    write_new(tmp_path/'data')
    with pytest.raises(FileExistsError):write_new(tmp_path/'data')

def test_checkpoint_full_optimizer_rng_and_cursor_restore(tmp_path):
    model=tiny();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    ids=pack_answers(['x'],4);loss,_=model.objective(model.encode(pack_prompts(['AB'],8)),ids,torch.tensor([.5]),torch.randn(1,4,272));loss.backward();optimizer.step()
    payload={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':torch.amp.GradScaler('cpu',init_scale=1024.).state_dict(),
             'torch_rng':torch.get_rng_state(),'attempts':1,'updates':1,'cursor':8,'order':list(range(16))}
    meta=support.save_full(tmp_path,payload);restored=torch.load(tmp_path/'checkpoints'/meta['file'],weights_only=True)
    assert support.state_sha(restored)==support.state_sha(payload)
    assert restored['order'][restored['cursor']:][:8]==list(range(8,16))
    assert restored['optimizer']['state'] and restored['scaler']['scale']==1024

def test_budget_attempt_calls_and_expiry(tmp_path):
    plan={'output':str(tmp_path),'cutoff':time.time()+30,'limits':{'max_calls':2,'max_output_bytes':10000}}
    budget=support.Budget(plan);budget.charge('encode');budget.charge('denoise')
    with pytest.raises(RuntimeError):budget.charge('encode')
    plan['cutoff']=time.time()-1
    with pytest.raises(TimeoutError):budget.check()

class Child:
    pid=123;returncode=None
    def poll(self):self.returncode=0;return 0
    def wait(self,timeout=None):self.returncode=0;return 0

@pytest.mark.parametrize('crash',[False,True])
def test_entire_owner_post_popen_natural_and_exception_cleanup(tmp_path,crash):
    plan={'authorized':True,'deadline':time.time()+50,'cutoff':time.time()+30,
          'limits':{'max_output_bytes':100000}}
    worker=tmp_path/'worker.py';worker.write_text('# CPU test stub');plan['files']={str(worker.resolve()):owner.native.sha256(worker)}
    pp=tmp_path/'plan.json';pp.write_text('{}')
    def popen(*args,**kwargs):
        if crash:
            # Persist fails immediately AFTER a real Popen return in the loop.
            original=owner.io.atomic_json
            def bad(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('post-Popen injected')
                return original(path,value)
            monkey=patch.object(owner.io,'atomic_json',bad);monkey.start();patches.append(monkey)
        else: (tmp_path/'result.json').write_text(json.dumps({'status':'BOUNDED_TEXT_GENERATION_PILOT_COMPLETE'}))
        return Child()
    patches=[]
    try:
        with patch.object(owner,'validate',return_value=plan),patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=popen),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':True}) as cleanup:
            rc=owner.supervise(pp,tmp_path,worker)
        assert cleanup.call_count==1
        state=json.loads((tmp_path/'supervisor/final_status.json').read_text())
        assert state['launches']==1 and state['status']==('STOPPED' if crash else 'COMPLETE')
        assert rc==(2 if crash else 0)
    finally:
        for p in patches:p.stop()

def test_real_plan_schema_and_expired_no_launch():
    pp=ROOT/'runs/text_generation_20261028_v1/plan.json'
    plan=owner.io.load_json(pp);out=Path(plan['output'])
    # Uses the real schema/files, without adding obsolete migration fields.
    owner.validate(pp,out)
    loaded,rows,review=support.validate(pp,out,ROOT/'scripts/run_text_generation_20261028.py')
    assert len(rows['train'])==3008 and not loaded['authorized']
    expired=copy.deepcopy(plan);expired.update(authorized=True,deadline=time.time()-1,cutoff=time.time()-91)
    with patch.object(owner,'validate',return_value=expired),patch.object(owner.subprocess,'Popen') as p:
        with pytest.raises(TimeoutError):owner.supervise(pp,out,ROOT/'scripts/run_text_generation_20261028.py')
        p.assert_not_called()

@pytest.mark.skipif(sys.platform!='win32',reason='real Windows CIM integration')
def test_real_windows_entire_owner_popen_proxy_exit_cpu_only(tmp_path):
    output=tmp_path/'native';output.mkdir()
    worker=tmp_path/'native_cpu_stub.py'
    worker.write_text("import argparse,json,os,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');p.add_argument('--preflight',action='store_true');a=p.parse_args()\nif a.preflight:print('CPU_PREFLIGHT_PASS');raise SystemExit(0)\no=Path(a.output);(o/'progress.json').write_text(json.dumps({'phase':'cpu_only','native_pid':os.getpid(),'attempts':0}))\ntime.sleep(2)\n(o/'result.json').write_text(json.dumps({'status':'BOUNDED_TEXT_GENERATION_PILOT_COMPLETE','model_calls':0}))\n",encoding='utf-8')
    source=ROOT/'runs/text_generation_20261028_v1/plan.json';plan=json.loads(source.read_text(encoding='utf-8'))
    plan.update(output=str(output),authorized=True,allocation_start=time.time())
    plan.update(deadline=plan['allocation_start']+900,cutoff=plan['allocation_start']+810)
    plan['files']={str(Path(owner.__file__).resolve()):owner.native.sha256(Path(owner.__file__)),str(worker.resolve()):owner.native.sha256(worker)}
    pp=output/'plan.json';pp.write_text(json.dumps(plan),encoding='utf-8')
    # No CUDA/model forward. Native subprocess/Popen/CIM/exit path is real.
    with patch.object(owner,'resources',return_value={'cpu_only_fixture':True}):
        rc=owner.supervise(pp,output,worker)
    state=json.loads((output/'supervisor/final_status.json').read_text(encoding='utf-8'))
    assert rc==0 and state['status']=='COMPLETE' and state['launches']==1
    assert state['exit']['all_owned_processes_gone']
