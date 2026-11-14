import json,sys,time
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import torch,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import probe_decoder_boundary_20261028 as worker
import supervise_decoder_boundary_20261028 as owner
from fdt_local.model_masked import Config
from research.text_language_continuation_v7 import LanguageContinuation,ContinuationConfig
torch.set_num_threads(2)

def test_components_swap_mask_and_readonly_trace_cost():
    c=Config(name='probeCPU',lang_dim=32,lang_heads=4,lang_ff=64,lang_layers=3,z_dim=264,core_heads=4,core_ff=64,diffusion_layers=1)
    m=LanguageContinuation(c,ContinuationConfig(32,12,2,1)).eval().requires_grad_(False)
    enc=(torch.randn(4,8,264),torch.ones(4,8,dtype=torch.bool));enc[1][0,3:]=False;z=torch.randn(4,12,264)
    ctx=worker.context_variant(m,enc,z,'swapPrompt');assert torch.equal(ctx[1][:,:8],enc[1].roll(1,0))
    ctx=worker.context_variant(m,enc,z,'zeroVP');assert torch.equal(ctx[0][:,8:],m.decoder.context_types[1].expand(4,12,264))
    class Budget:
        def __init__(self):self.calls=Counter()
        def charge(self,k):self.calls[k]+=1
    b=Budget();rows=[{'id':str(i),'prompt':'Rain fell.'} for i in range(4)]
    ids=worker.packed_targets([[97,98,256]]*4,12,'cpu');own=worker.packed_targets([[97]*12]*4,12,'cpu')
    before={k:v.clone() for k,v in m.state_dict().items()};tr=worker.trace_batch(m,rows,ids,own,b,'cpu')
    assert sum(b.calls.values())==16 and b.calls['decoder']==12 and tr['normal_teacher'].shape==(4,12,258)
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items())
    assert not any(p.grad is not None for p in m.parameters())

class Child:
    pid=567;returncode=None
    def poll(self):self.returncode=0;return 0
    def wait(self,timeout=None):self.returncode=0;return 0

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_readonly_schema_postPopen_cleanup(tmp_path,crash,confirmed):
    p=worker.make_plan(tmp_path);p.update(authorized=True,allocation_start=time.time())
    p.update(deadline=p['allocation_start']+120,cutoff=p['allocation_start']+90)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8')
    owner.validate(path,tmp_path);worker.validate(path,tmp_path,ROOT/'scripts/probe_decoder_boundary_20261028.py')
    patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'DECODER_BOUNDARY_READONLY_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            rc=owner.supervise(path,tmp_path,ROOT/'scripts/probe_decoder_boundary_20261028.py')
        d=json.loads((tmp_path/'supervisor/final_status.json').read_text())
        assert cleanup.call_count==1 and rc==(0 if confirmed and not crash else 2)
        assert d['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()
