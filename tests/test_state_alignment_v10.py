import copy,sys
from pathlib import Path
import pytest,torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from fdt_local.model_masked import Config
from research.text_language_continuation_v7 import ContinuationConfig
from research.text_state_alignment_v10 import StateAlignedLanguage,STATE_ALIGNMENT_POLICY
from research.text_diffusion_v1 import PAD
torch.set_num_threads(2)

def fixture():
    torch.manual_seed(103)
    core=Config(lang_dim=24,lang_heads=3,lang_ff=32,lang_layers=3,z_dim=264,core_heads=4,core_ff=40,diffusion_layers=1)
    m=StateAlignedLanguage(core,ContinuationConfig(prompt_bytes=8,target_bytes_with_eos=12,diffusion_steps=2,train_language_tail_blocks=1))
    tokens=torch.tensor([[65,66,67,256],[67,65,256,256]])
    ids=torch.full((2,12),PAD,dtype=torch.long);ids[0,:4]=torch.tensor([65,32,66,256]);ids[1,:3]=torch.tensor([66,65,256])
    return m,tokens,ids,torch.tensor([.3,1.]),torch.randn(2,12,264)

def test_frozen_decoder_transmits_CE_gradient_to_VP_and_language_tail_without_prompt_bypass():
    m,x,y,t,n=fixture();calls=[];seen=[]
    hook=m.decoder.blocks[0].cross.register_forward_pre_hook(lambda module,args:seen.append(args[1]))
    try:loss,aux=m.training_loss(x,y,t,n,charge=calls.append)
    finally:hook.remove()
    aux['predicted_clean'].retain_grad();aux['AR_byte_ce'].backward()
    assert aux['predicted_clean'].grad.abs().sum()>0
    assert m.vp.velocity.weight.grad.abs().sum()>0 and m.vp.prompt_project.weight.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.vp.language.blocks[-1].parameters())
    assert all(p.grad is None and not p.requires_grad for p in m.decoder.parameters())
    prompt=seen[0][:,:x.shape[1]+1]
    assert torch.allclose(prompt,m.decoder.context_types[0].expand_as(prompt))
    assert calls==['encode','denoise','decoder'] and torch.isfinite(loss)

def test_time_one_denoiser_input_is_target_independent_and_inactive_slots_have_no_PAD_code():
    m,x,y,_,n=fixture();t=torch.ones(2);seen=[]
    def collect(module,args):seen.append(args[0].detach().clone())
    hook=m.vp.refiner[0].register_forward_pre_hook(collect)
    try:
        m.training_loss(x,y,t,n)
        other=y.clone();other[other<256]=90;m.training_loss(x,other,t,n)
    finally:hook.remove()
    assert torch.allclose(seen[0],seen[1],atol=2e-6,rtol=0)
    # At t=1 all input z is noise; small cos(pi/2) FP residual is tolerated.
    assert STATE_ALIGNMENT_POLICY['runtime_gold_features'] is False

def test_actual_trainable_ownership_decoder_unchanged_and_vMSE_cleanMSE_identity():
    m,x,y,t,n=fixture();before=copy.deepcopy(m.decoder.state_dict())
    optim=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5)
    loss,aux=m.training_loss(x,y,t,n);loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    assert all(not p.requires_grad for p in m.vp.language.blocks[0].parameters())
    optim.step()
    assert all(torch.equal(v,m.decoder.state_dict()[k]) for k,v in before.items())
    # With a common t, clean-error MSE equals sin(t*pi/2)^2 * velocity-error MSE.
    _,aux=m.training_loss(x,y,torch.full((2,),.7),n)
    assert torch.allclose(aux['clean_mse'],aux['velocity_mse']*torch.sin(torch.tensor(.7*torch.pi/2)).square(),atol=1e-5,rtol=1e-5)

def test_training_targets_rejected_in_eval_and_inference_signature_unchanged():
    m,x,y,t,n=fixture();m.eval()
    with pytest.raises(RuntimeError):m.training_loss(x,y,t,n)
    import inspect
    assert list(inspect.signature(m.generate).parameters)==['tokens','proposal_noise','charge','zero_latent']
