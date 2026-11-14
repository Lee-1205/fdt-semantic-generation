import sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from fdt_local.model_masked import Config
from research.text_context_decoder_v2 import ContextDecoder
from research.text_decoder_alignment_v8 import AlignedContextDecoder,span_unlikelihood
torch.set_num_threads(2)

def config():return Config(lang_dim=32,lang_heads=4,lang_ff=64,lang_layers=3,z_dim=264,core_heads=4,core_ff=64,diffusion_layers=1)

def test_exact_parent_import_and_position_information_reaches_crossattention():
    torch.manual_seed(5);parent=ContextDecoder(config(),length=8).eval();new=AlignedContextDecoder(config(),length=8).eval()
    result=new.import_parent(parent.state_dict());assert result['new_parameters']==8*264
    for k,v in parent.state_dict().items():assert torch.equal(v,new.state_dict()[k])
    assert torch.equal(new.memory_positions.weight,new.positions.weight)
    prompt=torch.randn(2,4,264);mask=torch.ones(2,4,dtype=torch.bool);z=torch.randn(2,8,264);prefix=torch.ones(2,8,dtype=torch.long)
    with torch.no_grad():
        a=parent(prefix,parent.context(prompt,mask,z));b=parent(prefix,parent.context(prompt,mask,z.flip(1)))
        c=new(prefix,new.context(prompt,mask,z));d=new(prefix,new.context(prompt,mask,z.flip(1)))
    assert torch.allclose(a,b,atol=2e-6) and (c-d).abs().max()>1e-4

def test_joint_span_probability_and_strong_coefficient_gradient():
    x=torch.zeros(1,4,258,requires_grad=True);ids=torch.tensor([[32,116,104,101]])
    p=(1/258)**4;loss=span_unlikelihood(x,ids,[(0,0,4)],beta=10.)
    assert abs(float(loss)-10*(-torch.log1p(torch.tensor(-p))).item())<1e-12
    loss.backward();assert (x.grad.gather(-1,ids[...,None])>0).all()
    with pytest.raises(ValueError):span_unlikelihood(x,torch.tensor([[256,1,2,3]]),[(0,0,4)])

def test_clean_reconstruction_training_only_and_noise_gradient_path():
    m=AlignedContextDecoder(config(),length=8);prompt=torch.randn(2,4,264);mask=torch.ones(2,4,dtype=torch.bool)
    ids=torch.tensor([[65,66,256,257,257,257,257,257]]*2);codec=torch.randn(258,264);noise=torch.randn(2,8,264)
    loss=m.training_reconstruction(prompt,mask,ids,codec,noise);loss.backward()
    assert m.memory_positions.weight.grad is not None and m.head.weight.grad is not None
    m.eval()
    with pytest.raises(RuntimeError):m.training_reconstruction(prompt,mask,ids,codec,noise)
