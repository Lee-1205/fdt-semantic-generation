import copy,json,sys,time,inspect
from pathlib import Path
from unittest.mock import patch
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from research.text_semantic_reader_v3 import SemanticGenerator
from research.text_semantic_targets_v3 import targets,build_metadata
from research.text_generation_data_v1 import build
from research.text_diffusion_v1 import pack_answers
import run_text_semantic_alignment_20261028 as worker
import supervise_text_semantic_alignment_20261028 as owner
import text_generation_support_v1 as support
torch.set_num_threads(2)

def cfg():return Config(lang_dim=32,lang_heads=4,lang_ff=64,lang_layers=1,z_dim=272,core_heads=8,core_ff=64,diffusion_layers=2)

def test_no_gold_in_inference_and_joint_reader_decoder_gradients():
    model=SemanticGenerator(cfg());prompt=torch.randn(2,7,272);mask=torch.ones(2,7,dtype=torch.bool);latent=torch.randn(2,40,272)
    assert list(inspect.signature(model.context).parameters)==['prompt','mask','latent','zero_latent']
    context,logits=model.context(prompt,mask,latent)
    labels=torch.tensor([[0,1,0,0],[1,2,2,1]])
    loss,ar,aux=model.loss(pack_answers(['A','B']),context,logits,labels);loss.backward()
    assert torch.isfinite(loss) and model.reader.heads[0].weight.grad.abs().sum()>0
    assert model.reader.values[0].weight.grad.abs().sum()>0 and model.decoder.head.weight.grad.abs().sum()>0

def test_all_training_and_eval_query_targets_separate_from_inputs():
    data=build();selected=list(range(1024));meta=build_metadata(data,selected)
    assert len(meta['train'])==1024
    for rows in data.values():
        for row in rows:
            t=targets(row);assert 0<=t[0]<8 and 0<=t[1]<8 and t[0]!=t[1]
            assert t[2]==('greater','less','unknown').index(row['label'])
    assert set(data['train'][0])=={'id','group','prompt','answer','language','label','operator','reverse','template'}

def test_zero_latent_does_not_change_reader_predictions():
    m=SemanticGenerator(cfg()).eval();p=torch.randn(2,7,272);mask=torch.ones(2,7,dtype=torch.bool);z=torch.randn(2,40,272)
    a,la=m.context(p,mask,z);b,lb=m.context(p,mask,z,True)
    assert all(torch.equal(x,y) for x,y in zip(la,lb)) and torch.equal(a[0][:,:4],b[0][:,:4])
    assert not torch.equal(a[0][:,4:],b[0][:,4:])

class Child:
    pid=567;returncode=None
    def poll(self):self.returncode=0;return 0
    def wait(self,timeout=None):self.returncode=0;return 0

@pytest.mark.parametrize('crash',[False,True])
def test_actual_v2_plan_schema_full_post_popen_cleanup(tmp_path,crash):
    pp=ROOT/'runs/text_semantic_alignment_20261028_v3/plan.json';plan=json.loads(pp.read_text(encoding='utf-8'))
    plan.update(output=str(tmp_path),authorized=True,allocation_start=time.time());plan.update(deadline=plan['allocation_start']+900,cutoff=plan['allocation_start']+810)
    testplan=tmp_path/'plan.json';testplan.write_text(json.dumps(plan));owner.validate(testplan,tmp_path)
    worker.validate(testplan,tmp_path,ROOT/'scripts/run_text_semantic_alignment_20261028.py')
    patches=[]
    def popen(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def bad(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('V2postPopen')
                return original(path,value)
            monkey=patch.object(owner.io,'atomic_json',bad);monkey.start();patches.append(monkey)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'TEXT_SEMANTIC_ALIGNMENT_V3_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=popen),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':True}) as cleanup:
            rc=owner.supervise(testplan,tmp_path,ROOT/'scripts/run_text_semantic_alignment_20261028.py')
        state=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert rc==(2 if crash else 0) and state['status']==('STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()

def test_expired_v2_rejects_before_popen(tmp_path):
    source=ROOT/'runs/text_semantic_alignment_20261028_v3/plan.json';plan=json.loads(source.read_text())
    plan.update(output=str(tmp_path),authorized=True,allocation_start=time.time()-1000)
    plan.update(deadline=plan['allocation_start']+900,cutoff=plan['allocation_start']+810)
    pp=tmp_path/'plan.json';pp.write_text(json.dumps(plan))
    with patch.object(owner.subprocess,'Popen') as p:
        with pytest.raises(TimeoutError):owner.supervise(pp,tmp_path,ROOT/'scripts/run_text_semantic_alignment_20261028.py')
        p.assert_not_called()

@pytest.mark.skipif(sys.platform!='win32',reason='real Windows')
def test_v2_real_windows_complete_owner_cpu_only(tmp_path):
    out=tmp_path/'native';out.mkdir();stub=tmp_path/'stub.py'
    stub.write_text("import argparse,os,json,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--plan');p.add_argument('--output');p.add_argument('--preflight',action='store_true');a=p.parse_args()\nif a.preflight:print('CPU_PREFLIGHT_PASS');raise SystemExit(0)\no=Path(a.output);(o/'progress.json').write_text(json.dumps({'phase':'cpu','attempts':0,'native_pid':os.getpid()}));time.sleep(2);(o/'result.json').write_text(json.dumps({'status':'TEXT_SEMANTIC_ALIGNMENT_V3_COMPLETE','model_calls':0}))\n")
    source=ROOT/'runs/text_semantic_alignment_20261028_v3/plan.json';plan=json.loads(source.read_text());plan.update(output=str(out),authorized=True,allocation_start=time.time());plan.update(deadline=plan['allocation_start']+900,cutoff=plan['allocation_start']+810)
    plan['files']={str(Path(owner.__file__).resolve()):support.sha(owner.__file__),str(stub.resolve()):support.sha(stub)}
    pp=out/'plan.json';pp.write_text(json.dumps(plan))
    with patch.object(owner,'resources',return_value={'cpu_fixture':True}):rc=owner.supervise(pp,out,stub)
    result=json.loads((out/'supervisor/final_status.json').read_text());assert rc==0 and result['exit']['all_owned_processes_gone']
