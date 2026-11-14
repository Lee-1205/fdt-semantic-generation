import copy,json,random,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch,pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from fdt_local.model_masked import Config,Router
from research.owt_fuzzy_reconstruction_v12 import FuzzySentenceAE
from research.text_diffusion_v1 import pack_prompts
from owt_fuzzy_engine_v12 import AEEngine
import run_owt_fuzzy_reconstruction_20261028_v12 as worker
import supervise_owt_fuzzy_reconstruction_20261028_v12 as owner
import text_generation_support_v1 as support
from test_decoder_alignment_runtime_v8 import Child
torch.set_num_threads(2)

class Budget:
    def __init__(self):self.calls={'encode':0,'denoise':0,'decoder':0}
    def check(self):pass
    def charge(self,k):self.calls[k]+=1

def fixture():
    torch.manual_seed(201);random.seed(201);np.random.seed(201)
    c=Config(lang_dim=24,lang_heads=3,lang_ff=32,lang_layers=3,z_dim=32,core_heads=4,core_ff=40,diffusion_layers=1,anchors=16,key_dim=16,value_dim=16,top_k=4)
    model=FuzzySentenceAE(c)
    a=ROOT/'research/artifacts/owt_fuzzy_reconstruction_20261028_v12'
    f=json.loads((a/'features.json').read_text(encoding='utf-8'))['train'][:15]
    y=json.loads((a/'targets.json').read_text(encoding='utf-8'))['train'][:15]
    return AEEngine(model,f,y,worker.finite_orders(15,[201,202]),Budget(),'CPU12','cpu')

def test_original_fuzzy_mu_is_nonexclusive_and_decoder_sees_extracted_state_only():
    e=fixture();m=e.model;tokens=pack_prompts([r['prompt'] for r in e.features[:2]],128,'cpu')
    s=m.extract(tokens);assert isinstance(m.core.phrase_router,Router)
    assert set(s)=={'views','memberships','union_memberships','retrieval_weights','confidence'}
    mu=s['memberships'];assert (mu>=0).all() and (mu<=1).all() and ((mu>0).sum(-1)<=4).all()
    assert torch.allclose(s['union_memberships'],1-(1-mu).prod(1))
    assert torch.allclose(s['retrieval_weights'],mu/mu.sum(-1,keepdim=True).clamp_min(1e-8))
    a=m.context(s)[0].detach().clone()
    with torch.no_grad():
        for p in m.core.parameters():p.add_(1)
    assert torch.equal(a,m.context(s)[0])
    import inspect
    assert list(inspect.signature(m.reconstruct).parameters)==['meaning','charge','mode']

def test_actual_fullstate_restores_exact_next_update_allRNG_and_last7(tmp_path):
    e=fixture();e.step();e.step();assert e.cursor==15
    snapshot=copy.deepcopy(e.export());saved=support.save_full(tmp_path,snapshot)
    s=torch.load(tmp_path/'checkpoints'/saved['file'],map_location='cpu',weights_only=True)
    rec=e.step();end=copy.deepcopy(e.export());n=fixture();n.restore(s)
    assert n.step()==rec and support.state_sha(n.export())==support.state_sha(end)
    n.step();assert n.cursor==30 and n.updates==4 and n.budget.calls['denoise']==0
    n.verify_frozen();assert len(n.optimizer.state)==sum(len(x) for x in n.names)

def test_enabled_scaler_skip_keeps_model_and_consumes_attempt_cursor():
    e=fixture();old=support.state_sha(e.model.state_dict());scale=e.scaler.get_scale()
    h=e.model.core.anchors.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:e.step()
    finally:h.remove()
    assert e.attempts==1 and e.updates==0 and e.cursor==8 and e.skips==1 and e.scaler.get_scale()==scale/2
    assert support.state_sha(e.model.state_dict())==old
    n=fixture();n.restore(copy.deepcopy(e.export()));assert len(n.next_indices())==7

def test_CPU_autocast_fuzzy_routes_FP32_and_all_required_gradients():
    e=fixture()
    with torch.autocast('cpu',dtype=torch.bfloat16):rec=e.step()
    assert rec['outcome']=='UPDATED' and len(e.optimizer.state)==sum(len(x) for x in e.names)
    assert e.model.core.phrase_router.sigma in e.optimizer.state

def test_saved_controls_charge_one_extraction_and_three_decoders_each_batch():
    e=fixture();budget=Budget();f=e.features[:8];y=e.targets[:8]
    metrics,tensors=worker.evaluate(e.model,f,y,budget,'cpu')
    assert budget.calls=={'encode':1,'denoise':0,'decoder':3}
    assert metrics['normal']['symbols']==sum(len(x['target_ids']) for x in y)
    assert torch.equal(tensors['target_ids'][:,:1],torch.tensor([[x['target_ids'][0]] for x in y]))
    assert tensors['memberships'].shape==(8,2,16)

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_AE_schema_postPopen_cleanup(tmp_path,crash,confirmed):
    p=json.loads((ROOT/'runs/owt_fuzzy_reconstruction_20261028_v12/plan.json').read_text())
    p.update(authorized=True,output=str(tmp_path),allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8')
    owner.validate(path,tmp_path);worker.validate(path,tmp_path,ROOT/'scripts/run_owt_fuzzy_reconstruction_20261028_v12.py')
    patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'OWT_FUZZY_RECONSTRUCTION_V12_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            code=owner.supervise(path,tmp_path,ROOT/'scripts/run_owt_fuzzy_reconstruction_20261028_v12.py')
        d=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2) and d['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for p in patches:p.stop()
