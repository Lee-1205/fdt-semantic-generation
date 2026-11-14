import copy,json,random,sys,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts'),str(ROOT/'tests')]
from fdt_local.model_masked import Config
from research.semantic_output_pipeline_v11 import SemanticPipeline,PHASES
from research.semantic_output_data_v11 import load
from research.text_diffusion_v1 import pack_prompts
from semantic_output_engine_v11 import SemanticEngine
import text_generation_support_v1 as support
import run_semantic_output_pipeline_20261028_v11 as worker
import supervise_semantic_output_pipeline_20261028_v11 as owner
from test_decoder_alignment_runtime_v8 import Child
torch.set_num_threads(2)

class Budget:
    def __init__(self,path):self.output=path;self.calls={k:0 for k in ('encode','reader','denoise','decoder')}
    def check(self):pass
    def charge(self,kind):self.calls[kind]+=1

def fixture(path):
    torch.manual_seed(151);random.seed(151);np.random.seed(151)
    c=Config(lang_dim=24,lang_heads=3,lang_ff=32,lang_layers=3,z_dim=32,core_heads=4,core_ff=40,diffusion_layers=1)
    m=SemanticPipeline(c)
    f,y=load(ROOT/'research/artifacts/text_surface_alignment_20261028_v5_cpu')
    f={'train':f['train'][:15]};y={'train':y['train'][:15]}
    orders={p:[list(range(8)),list(range(8,15))]*2 for p in PHASES}
    return SemanticEngine(m,f,y,orders,Budget(path),'CPU11','cpu')

def attach_cache(e):
    labels=torch.tensor([y['output_labels'] for y in e.targets['train']]);labels[0,2]=(labels[0,2]+1)%4
    cache={'train':{'labels':labels}};path=e.budget.output/'meaning_cache.pt';torch.save(cache,path)
    e.cache=cache;e.cache_reference={'path':path.name,'sha256':support.sha(path)}

def test_input_and_output_targets_distinct_and_no_gold_in_neural_features():
    f,y=load(ROOT/'research/artifacts/text_surface_alignment_20261028_v5_cpu')
    assert all(set(r)=={'id','prompt'} for rows in f.values() for r in rows)
    rows=y['train'];normal=next(r for r in rows if r['source_fact']['operator']=='greater' and not r['reverse'])
    reverse=next(r for r in rows if r['source_fact']==normal['source_fact'] and r['reverse'])
    assert normal['input_labels'][2:5]==reverse['input_labels'][2:5]
    assert normal['output_labels'][2]==1 and reverse['output_labels'][2]==2
    assert normal['output_labels'][:2]==list(reversed(reverse['output_labels'][:2]))
    assert all(r['output_labels'][2]==0 for r in rows if r['source_fact']['operator'].startswith('not '))

def test_actual_semantic_target_is_21basis_not_byte_codec(tmp_path):
    e=fixture(tmp_path);m=e.model.meaning;labels=torch.tensor([[0,1,1,0],[1,0,2,0]])
    z=m.target(labels);heads=m.logits(z)
    assert z.shape==(2,1,32) and m.basis.shape==(21,32)
    assert torch.equal(torch.stack([x.argmax(-1) for x in heads],1),labels[:,:3])
    tokens=pack_prompts([r['prompt'] for r in e.features['train'][:2]],160,'cpu')
    read=e.model.extract(tokens);assert set(read)=={'entities','operator','language','pointer_logits','operator_logits','language_logits','candidate_mask'}

def test_all_three_phase_fullstate_exact_next_step_and_last7_rows(tmp_path):
    e=fixture(tmp_path)
    for phase in PHASES:
        if phase!='input':e.set_phase(phase)
        if phase=='language':attach_cache(e)
        e.step();e.step();assert e.cursors[phase]==15
        snapshot=copy.deepcopy(e.export());saved=support.save_full(tmp_path/phase,snapshot) if (tmp_path/phase).mkdir() is None else None
        restored=torch.load(tmp_path/phase/'checkpoints'/saved['file'],map_location='cpu',weights_only=True)
        expected=e.step();final=copy.deepcopy(e.export())
        n=fixture(tmp_path);n.restore(restored);assert n.step()==expected and support.state_sha(n.export())==support.state_sha(final)
        e.step();e.verify_frozen()
    assert e.attempts==12 and e.cursor==90 and set(e.optimizers)==set(PHASES)
    assert e.exposure['language_target_bytes']>0 and e.exposure['predicted_rows']>0 and e.exposure['fallback_rows']>0

def test_real_enabled_scaler_skip_consumes_rows_without_replay(tmp_path):
    e=fixture(tmp_path);before=support.state_sha(e.model.state_dict());scale=e.scaler.get_scale()
    h=e.model.input.queries.register_hook(lambda g:torch.full_like(g,float('inf')))
    try:e.step()
    finally:h.remove()
    assert e.attempts==1 and e.updates==0 and e.skips==1 and e.cursor==8 and e.scaler.get_scale()==scale/2
    assert before==support.state_sha(e.model.state_dict())
    n=fixture(tmp_path);n.restore(copy.deepcopy(e.export()));assert len(n.next_indices())==7

def test_committed_output_is_only_realizer_input_no_raw_prompt_or_input_fact_access(tmp_path):
    e=fixture(tmp_path);m=e.model;labels=torch.tensor([[0,1,1,0],[1,0,2,0]])
    with torch.no_grad():
        a=m.realizer.context(labels)
        for p in m.encoder.parameters():p.add_(123)
        for p in m.input.parameters():p.add_(123)
        for p in m.meaning.parameters():p.add_(123)
        b=m.realizer.context(labels)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
    assert not torch.equal(a[0][0],a[0][1])
    import inspect
    assert list(inspect.signature(m.realizer.generate).parameters)==['committed','charge']

@pytest.mark.parametrize('crash,confirmed',[(False,True),(True,True),(False,False)])
def test_actual_new_schema_owner_cleanup_after_Popen(tmp_path,crash,confirmed):
    p=json.loads((ROOT/'runs/semantic_output_pipeline_20261028_v11/plan.json').read_text())
    p.update(authorized=True,output=str(tmp_path),allocation_start=time.time());p.update(deadline=p['allocation_start']+900,cutoff=p['allocation_start']+810)
    path=tmp_path/'plan.json';path.write_text(json.dumps(p),encoding='utf-8')
    owner.validate(path,tmp_path);worker.validate(path,tmp_path,ROOT/'scripts/run_semantic_output_pipeline_20261028_v11.py')
    patches=[]
    def launch(*args,**kwargs):
        if crash:
            original=owner.io.atomic_json
            def injected(path,value):
                if Path(path).name=='state.json' and value.get('launches')==1 and not value.get('stop_reason'):raise RuntimeError('postPopenCPU')
                return original(path,value)
            patcher=patch.object(owner.io,'atomic_json',injected);patcher.start();patches.append(patcher)
        else:(tmp_path/'result.json').write_text(json.dumps({'status':'SEMANTIC_OUTPUT_PIPELINE_V11_COMPLETE'}))
        return Child()
    try:
        with patch.object(owner,'preflight'),patch.object(owner,'resources',return_value={}),patch.object(owner,'duplicates',return_value=[]),patch.object(owner.native,'process_inventory',return_value=[]),patch.object(owner.subprocess,'Popen',side_effect=launch),patch.object(owner.native,'verify_exit',return_value={'all_owned_processes_gone':confirmed}) as cleanup:
            code=owner.supervise(path,tmp_path,ROOT/'scripts/run_semantic_output_pipeline_20261028_v11.py')
        d=json.loads((tmp_path/'supervisor/final_status.json').read_text());assert cleanup.call_count==1
        assert code==(0 if confirmed and not crash else 2) and d['status']==('UNCONFIRMED_EXIT' if not confirmed else 'STOPPED' if crash else 'COMPLETE')
    finally:
        for patcher in patches:patcher.stop()
