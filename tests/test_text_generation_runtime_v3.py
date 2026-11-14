import inspect,sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
from fdt_local.model_masked import Config
from research.text_diffusion_v1 import TextDiffusion,TextConfig
from research.text_semantic_reader_v3 import SemanticGenerator
from research.text_generation_runtime_v3 import generate_text,load_models
from text_generation_support_v1 import state_sha
torch.set_num_threads(2)

def cfg():return Config(lang_dim=32,lang_heads=4,lang_ff=64,lang_layers=1,z_dim=272,core_heads=8,core_ff=64,diffusion_layers=2)

def test_raw_prompt_generation_has_no_gold_and_is_repeatable():
    vp=TextDiffusion(cfg()).eval();text=SemanticGenerator(cfg()).eval();calls=[]
    before=(state_sha(vp.state_dict()),state_sha(text.state_dict()));rng=torch.get_rng_state().clone()
    a=generate_text(vp,text,'A is greater than B.',17,calls.append);b=generate_text(vp,text,'A is greater than B.',17)
    assert a==b and torch.equal(torch.get_rng_state(),rng)
    assert calls[:18]==['encode']+['denoise']*16+['reader'] and 19<=len(calls)<=58
    assert before==(state_sha(vp.state_dict()),state_sha(text.state_dict()))
    assert list(inspect.signature(generate_text).parameters)==['vp','text','prompt','seed','charge']

def test_capacity_failure_before_forward():
    vp=TextDiffusion(cfg());text=SemanticGenerator(cfg());calls=[]
    with pytest.raises(ValueError):generate_text(vp,text,'가'*100,charge=calls.append)
    assert not calls

def test_full_research_runtime_checkpoint_roundtrip(tmp_path):
    from dataclasses import asdict
    vp=TextDiffusion(cfg());text=SemanticGenerator(cfg());path=tmp_path/'state.pt'
    torch.save({'schema':'fdt_text_semantic_alignment_v3','core':asdict(cfg()),
                'model':{'generator':vp.state_dict(),'semantic_generator':text.state_dict()}},path)
    a,b=load_models(path,'cpu')
    assert state_sha(a.state_dict())==state_sha(vp.state_dict()) and state_sha(b.state_dict())==state_sha(text.state_dict())
    assert all(not p.requires_grad for p in list(a.parameters())+list(b.parameters()))
