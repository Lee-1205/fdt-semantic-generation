"""Offline span coverage, anchor gradient and frozen upstream tiny contract."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.text_diffusion_v1 import pack_prompts
from research.fuzzy_factual_objective_v70 import critical_mask
from research.frozen_verbalizer_objective_v72 import complete_expression_spans,anchor_loss,decoder_losses,freeze_extraction_and_binding
import text_generation_support_v1 as support

def main():
    torch.set_num_threads(2)
    text='Calder logged -$12.50 at 03:17 on 2024-11-07; +2.75 kg, 18%, 8.1K and 3-7 kg.'
    pieces=[s['text'] for s in complete_expression_spans(text)]
    assert pieces==['-$12.50','03:17','2024-11-07','+2.75 kg','18%','8.1K','3-7 kg'],pieces
    z=torch.zeros(3,16,258,requires_grad=True);y=torch.full((3,16),257,dtype=torch.long);y.flatten()[:35]=65;y.flatten()[35]=256
    with torch.no_grad():z[0,0,65]=8
    risk=torch.zeros_like(y,dtype=torch.bool);risk[0,1:4]=True;correct=torch.ones_like(risk);correct[0,4]=False
    p0=z.detach().softmax(-1).gather(-1,y[...,None]).squeeze(-1)
    kl,eligible=anchor_loss(z,y,risk,p0,correct);grad=torch.autograd.grad(kl,z)[0];assert grad.abs().max()<1e-7 and grad[~eligible].count_nonzero()==0
    shifted=z.detach().clone();shifted[0,0,65]-=2;shifted.requires_grad_(True);kl,_=anchor_loss(shifted,y,risk,p0,correct);grad=torch.autograd.grad(kl,shifted)[0]
    assert grad[0,0,65]<0 and grad[risk].count_nonzero()==0 and grad[y==257].count_nonzero()==0
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(72);c=Config.smoke();enc=FuzzyCodecV64(c,input_capacity=256);enc.enable_position_adapter();enc.enable_long_capacity();enc.enable_local_banks()
        model=FuzzyRoundTripV64(enc,OrderedFuzzyReadout(c.z_dim,c.value_dim,hidden=64));names=freeze_extraction_and_binding(model)
        frozen=support.state_sha({n:v for n,v in model.state_dict().items() if n not in names});optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-5)
        model.eval();source='Vera has 8 gold.';assert len(source.encode())==16
        with torch.no_grad():
            commit,episode=model.extract(pack_prompts([source],16,'cpu'),source_ids=('fixture72',),source_versions=('v1',),model_epoch='fixture72')
            initial=model.verbalize(commit)
        target=torch.full(initial.shape[:2],257,dtype=torch.long);target.flatten()[:16]=torch.tensor(list(source.encode()));target.flatten()[16]=256
        mask=critical_mask(target,[dict(start=9,end=10)],16);prob=initial.softmax(-1).gather(-1,target[...,None]).squeeze(-1);was_correct=initial.argmax(-1)==target
        prediction=model.verbalize(commit);weights=torch.ones(258);weights[257]=0
        loss,_=decoder_losses(prediction,target,weights,mask,prob,was_correct);loss.backward()
        assert all(p.grad is not None for n,p in model.named_parameters() if n in names) and all(p.grad is None for n,p in model.named_parameters() if n not in names)
        optimizer.step();assert support.state_sha({n:v for n,v in model.state_dict().items() if n not in names})==frozen
        assert not episode.memories.exact.valid.any()
    assert not torch.cuda.is_initialized()
    report=dict(status='FROZEN_VERBALIZER_V72_CPU_CANDIDATE_PASS',complete_numeric_expression_fixture=True,anchor_stationary_and_recovery_gradient=True,
        critical_PAD_incorrect_teacher_anchor_excluded=True,upstream_encoder_and_role_projection_frozen=True,only7_final_readout_tensors_trainable=True,
        tiny_fixture_model_calls=3,actual_parent_model_calls=0,actual_V72_optimizer_trial=False,CUDA_initialized=False,
        limitation='tiny one-step contract, not full runner/cursor/cache/scaler/ALLRNG restoration or GPU capability')
    file=ROOT/'research/artifacts/frozen_verbalizer_20261112_v72_cpu/root_CPU_contract.json';assert not file.exists();file.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
