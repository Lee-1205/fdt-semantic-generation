"""Actual fuzzy-only roundtrip boundary and live encoder/decoder gradients."""
import copy,json,sys
from dataclasses import replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from fdt_local.model_masked import Config
from research.fuzzy_roundtrip_v64 import FuzzyCodecV64,FuzzyRoundTripV64,enable_cycle_training,EXACT_POLICY
from research.ordered_fuzzy_readout_v55 import OrderedFuzzyReadout
from research.text_diffusion_v1 import pack_prompts


def main():
    torch.set_num_threads(2);torch.manual_seed(64)
    c=replace(Config.smoke(),gradient_checkpointing=True)
    encoder=FuzzyCodecV64(c,input_capacity=256);encoder.enable_position_adapter();encoder.enable_long_capacity();encoder.enable_local_banks()
    with torch.no_grad():encoder.core.prior.normal_(0,.02);encoder.memory_codec.local_bank_gains.fill_(.002)
    model=FuzzyRoundTripV64(encoder,OrderedFuzzyReadout(c.z_dim,c.value_dim,hidden=128))
    names=enable_cycle_training(model)
    text='The blue box holds 314159. 한글 문장.';x=pack_prompts([text],len(text.encode()),'cpu')
    kwargs=dict(source_ids=('cpu:one',),source_versions=('v1',),model_epoch='cpu:64')
    logits,commit,episode=model(x,**kwargs)
    exact=episode.memories.exact
    assert not exact.valid.any() and not exact.keys.count_nonzero() and not exact.values.count_nonzero() and not any(exact.spans)
    assert all(set(vars(s))=={'source_id','version','sha256'} and not hasattr(s,'raw') for s in exact.sources)
    assert torch.equal(episode.diagnostics['null_weight'],torch.ones_like(episode.diagnostics['null_weight']))
    sealed=commit.seal();reference=model.verbalize(sealed).detach().clone()
    x.fill_(0)
    assert torch.equal(reference,model.verbalize(sealed))
    try:model.verbalize(episode)
    except TypeError:pass
    else:raise AssertionError('episode/raw boundary accepted by language reader')
    try:model(x,**kwargs,copy_request=True)
    except TypeError:pass
    else:raise AssertionError('current cycle accepts copy request')
    y=torch.full(logits.shape[:2],257,dtype=torch.long);flat=y.flatten();b=text.encode();flat[:len(b)]=torch.tensor(list(b));flat[len(b)]=256
    loss=torch.nn.functional.cross_entropy(logits.transpose(1,2),y,ignore_index=257);loss.backward()
    named=dict(model.named_parameters())
    assert all(named[n].grad is not None for n in names)
    assert any(p.grad.abs().sum()>0 for n,p in named.items() if n.startswith('encoder.core.language.blocks.') and p.requires_grad)
    assert named['encoder.core.value.weight'].grad.abs().sum()>0
    assert any(p.grad.abs().sum()>0 for n,p in named.items() if n.startswith('readout.'))
    assert all(p.grad is None for n,p in named.items() if not p.requires_grad)
    assert not any(p.requires_grad for n,p in named.items() if any(z in n for z in ('core.refiner.','core.velocity.','core.time.','core.rel_message.')))
    result=dict(status='FUZZY_ROUNDTRIP_V64_CPU_PASS',exact_policy=EXACT_POLICY,all_natural_and_numeric_input_exact_entries0=True,
        Exact_has_no_raw_source_or_codepoint_index=True,Exact_NULL_only=True,Commit_only_reader=True,sealed_roundtrip_input_mutation_invariant=True,
        reconstruction_loss_reaches_language_encoder_and_value_and_decoder=True,no_reasoning_or_denoising_gradients=True,
        copy_request_api_rejected_in_current_stage=True,tiny_fixture_not_model_ability=True,CUDA_initialized=torch.cuda.is_initialized())
    out=ROOT/'research/artifacts/fuzzy_roundtrip_20261112_v64_cpu';out.mkdir(exist_ok=True)
    (out/'root_boundary_gradient_contract.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))

if __name__=='__main__':main()
