import copy,json,random,sys,time
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from research.text_language_continuation_v7 import LanguageContinuation,ContinuationConfig
from research.text_decoder_alignment_training_v8 import AlignedLanguage,vector_span_loss
from research.text_decoder_alignment_v8 import span_unlikelihood
from text_language_engine_v7 import CurriculumEngine
from text_decoder_alignment_engine_v8 import AlignmentEngine
import text_generation_support_v1 as support
import prepare_text_decoder_alignment_20261028_v8 as prepare
import run_text_decoder_alignment_20261028_v8 as worker
import supervise_text_decoder_alignment_20261028_v8 as owner
torch.set_num_threads(2)

class Budget:
    def __init__(self):self.calls=Counter(encode=0,denoise=0,decoder=0)
    def check(self):pass
    def charge(self,k):self.calls[k]+=1

def fixture():
    torch.manual_seed(51);random.seed(51);np.random.seed(51)
    c=Config(name='v8CPU',lang_dim=32,lang_heads=4,lang_ff=64,lang_layers=3,z_dim=264,core_heads=4,core_ff=64,diffusion_layers=1)
    policy=ContinuationConfig(32,12,2,1);features=[];targets=[];negatives=[]
    for i in range(15):
        text=' wet.' if i%2 else ' sunshine.'
        features.append({'id':str(i),'prompt':'Rain fell on a little roof.'})
        targets.append({'target':text,'target_ids':list(text.encode())+[256],'source_eof':True,'document_id':str(i)})
        negatives.append({'id':str(i),'negative_ids':[32,97]*6,'negative_spans':[[8,10],[10,12]]})
    parent=CurriculumEngine(LanguageContinuation(c,policy),features,targets,Budget(),'PARENT','cpu',seeds=(61,))
    parent.step();parent.step();state=copy.deepcopy(parent.export())
    engine=AlignmentEngine(AlignedLanguage(c,policy),features,targets,negatives,Budget(),'V8CPU','cpu')
    engine.import_parent(state,'PARENT')
    return engine,parent,state

def test_zero_position_initialization_preserves_parent_function_and_all_old_components():
    e,p,state=fixture();assert e.ancestry['old_component_sha256']
    assert len(e.optimizer.state)==len(p.optimizer.state) and not torch.count_nonzero(e.model.decoder.memory_positions.weight)
    prompt=torch.randn(2,4,264);mask=torch.ones(2,4,dtype=torch.bool);z=torch.randn(2,12,264);ids=torch.ones(2,12,dtype=torch.long)
    with torch.no_grad():
        a=p.model.decoder(ids,p.model.decoder.context(prompt,mask,z));b=e.model.decoder(ids,e.model.decoder.context(prompt,mask,z))
    assert torch.equal(a,b) and e.scaler.state_dict()==p.scaler.state_dict()

def test_actual_driver_fullrestore_last7_epoch_boundary_and_frozen_VP(tmp_path):
    e,_,_=fixture();vp=support.state_sha(e.model.vp.state_dict());calls=e.budget.calls.copy()
    e.step();e.step();assert e.cursor==15 and len(e.history[-1]['rows'])==7 and e.history[-1]['span_UL']==0
    assert support.state_sha(e.model.vp.state_dict())==vp and e.budget.calls['denoise']==calls['denoise']
    saved=support.save_full(tmp_path,copy.deepcopy(e.export()));state=torch.load(tmp_path/'checkpoints'/saved['file'],map_location='cpu',weights_only=True)
    expected=e.step();end=copy.deepcopy(e.export());n,_,_=fixture();n.restore(state)
    assert n.step()==expected and support.state_sha(n.export())==support.state_sha(end)
    while n.attempts<n.limit:n.step()
    assert n.cursor==60 and n.updates==8 and all(r['phase']=='reconstruction' for r in n.history)
    assert support.state_sha(n.model.vp.state_dict())==vp and any(v['step']>2 for v in n.optimizer.state.values())
    n.verify_frozen()

def test_AMP_skip_consumes_rows_without_mutating_parent_or_new_parameter():
    e,_,_=fixture();before=support.state_sha(e.model.state_dict())
    hook=e.model.decoder.memory_positions.weight.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:e.step()
    finally:hook.remove()
    assert e.attempts==1 and e.cursor==8 and e.updates==0 and e.skips==1 and e.scaler.get_scale()==512
    assert before==support.state_sha(e.model.state_dict());n,_,_=fixture();n.restore(copy.deepcopy(e.export()));assert len(n.next_indices())==7

def test_vectorized_strong_span_loss_matches_joint_probability_reference_value_and_gradient():
    torch.manual_seed(5);x=torch.randn(2,12,258,requires_grad=True);y=x.detach().clone().requires_grad_(True)
    ids=torch.tensor([[32,97]*6]*2);spans=[(0,6,8),(0,8,10),(1,8,12)]
    a=vector_span_loss(x,ids,spans,10);b=span_unlikelihood(y,ids,spans,10)
    a.backward();b.backward();assert torch.allclose(a,b,atol=1e-7) and torch.allclose(x.grad,y.grad,atol=1e-7)

class Child:
    pid=567;returncode=None
    def poll(self):self.returncode=0;return 0
    def wait(self,timeout=None):self.returncode=0;return 0

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_new_schema_owned_postPopen_cleanup(tmp_path,crash,confirmed):
    p=prepare.make_plan(tmp_path);p.update(authorized=True,allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8')
    owner.validate(path,tmp_path);worker.validate(path,tmp_path,ROOT/'scripts/run_text_decoder_alignment_20261028_v8.py')
    patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'TEXT_DECODER_ALIGNMENT_V8_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            code=owner.supervise(path,tmp_path,ROOT/'scripts/run_text_decoder_alignment_20261028_v8.py')
        d=json.loads((tmp_path/'supervisor/final_status.json').read_text())
        assert cleanup.call_count==1 and code==(0 if confirmed and not crash else 2)
        assert d['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()
