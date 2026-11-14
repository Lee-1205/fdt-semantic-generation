import copy,json,random,sys,time
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from research.text_language_continuation_v7 import LanguageContinuation,ContinuationConfig
from research.text_diffusion_v1 import pack_prompts,PAD
from text_language_engine_v7 import CurriculumEngine
import text_generation_support_v1 as support
import run_text_language_continuation_20261028_v7 as worker
import supervise_text_language_continuation_20261028_v7 as owner
import prepare_text_language_continuation_20261028_v7 as prepare
torch.set_num_threads(2)

class Budget:
    def __init__(self):self.calls=Counter(encode=0,denoise=0,decoder=0)
    def check(self):pass
    def charge(self,k):self.calls[k]+=1

def fixture(count=15,seeds=(62,63)):
    torch.manual_seed(91);random.seed(91);np.random.seed(91)
    c=Config(name='CPU-v7',lang_dim=32,lang_heads=4,lang_ff=64,lang_layers=3,z_dim=264,core_heads=4,core_ff=64,diffusion_layers=1)
    m=LanguageContinuation(c,ContinuationConfig(32,12,2,1));f=[];t=[]
    for i in range(count):
        text=' wet.' if i%2 else ' sunshine.'
        f.append({'id':str(i),'prompt':'Rain fell on a little roof.'})
        t.append({'target':text,'target_ids':list(text.encode())+([256] if i%2 else []),'source_eof':bool(i%2),'document_id':str(i)})
    return CurriculumEngine(m,f,t,Budget(),'CPU-V7','cpu',seeds)

def test_two_finite_epochs_boundary_last_seven_and_exact_next_restore(tmp_path):
    e=fixture();e.step();e.step();assert e.cursor==15 and len(e.history[-1]['rows'])==7
    state=copy.deepcopy(e.export());saved=support.save_full(tmp_path,state)
    state=torch.load(tmp_path/'checkpoints'/saved['file'],map_location='cpu',weights_only=True)
    expected=e.step();end=copy.deepcopy(e.export());n=fixture();n.restore(state)
    assert n.step()==expected and support.state_sha(n.export())==support.state_sha(end)
    n.step();assert n.cursor==30 and len(n.history[-1]['rows'])==7
    assert all(sorted([i for b in n.batches[j:j+2] for i in b])==list(range(15)) for j in (0,2))
    with pytest.raises(StopIteration):n.step()

def test_parent_adam_scaler_rng_components_exact_before_new_curriculum():
    e=fixture(seeds=(61,));e.step();e.step();p=copy.deepcopy(e.export());p['plan_sha256']='PARENT'
    n=fixture();digest=n.import_parent(p,'PARENT');assert digest==n.ancestry['imported_component_sha256']
    assert n.attempts==n.updates==n.cursor==0 and n.optimizer.state and n.scaler.state_dict()['scale']==1024
    assert n.ancestry['updates']==2 and n.ancestry['cursor']==15
    n.step();assert n.updates==1 and all(g['base_lr'] in (5e-6,1.5e-5) for g in n.optimizer.param_groups)

def test_weight4_exact_CE_delta_noisy_gold_not_in_free_proposal():
    e=fixture(4);m=e.model;m.train();tokens=pack_prompts([r['prompt'] for r in e.features],32,'cpu')
    ids=torch.full((4,12),PAD,dtype=torch.long)
    for j,t in enumerate(e.targets):ids[j,:len(t['target_ids'])]=torch.tensor(t['target_ids'])
    t=torch.full((4,),.4);noise=torch.randn(4,12,264);proposal=torch.randn_like(noise)
    loss,a=m.training_loss(tokens,ids,t,noise,proposal)
    labels=ids.masked_fill(ids==PAD,-100);ce=torch.nn.functional.cross_entropy(a['AR_logits'].transpose(1,2),labels,ignore_index=-100,reduction='none')
    expected=(ce*(ids!=PAD)*torch.where(ids==256,4.,1.)).sum()/(ids!=PAD).sum()
    assert torch.allclose(expected,a['AR_byte_ce']);loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    m.eval()
    with pytest.raises(RuntimeError):m.training_loss(tokens,ids,t,noise,proposal)

def test_skip_consumes_selected_batch_and_restore_rejects_curriculum_change():
    e=fixture();before=support.state_sha(e.model.state_dict())
    hook=e.model.decoder.head.weight.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:e.step()
    finally:hook.remove()
    assert e.cursor==8 and e.updates==0 and e.skips==1 and before==support.state_sha(e.model.state_dict())
    p=copy.deepcopy(e.export());n=fixture();n.restore(p);assert len(n.next_indices())==7
    p['curriculum']['seeds']=[62,64]
    with pytest.raises(ValueError):fixture().restore(p)

class Child:
    pid=567;returncode=None
    def poll(self):self.returncode=0;return 0
    def wait(self,timeout=None):self.returncode=0;return 0

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_new_schema_postPopen_cleanup(tmp_path,crash,confirmed):
    p=prepare.make_plan(tmp_path);p.update(authorized=True,allocation_start=time.time())
    p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8')
    owner.validate(path,tmp_path);worker.validate(path,tmp_path,ROOT/'scripts/run_text_language_continuation_20261028_v7.py')
    patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'TEXT_LANGUAGE_V7_BOUNDED_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            rc=owner.supervise(path,tmp_path,ROOT/'scripts/run_text_language_continuation_20261028_v7.py')
        final=json.loads((tmp_path/'supervisor/final_status.json').read_text())
        assert cleanup.call_count==1 and rc==(0 if confirmed and not crash else 2)
        assert final['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()
