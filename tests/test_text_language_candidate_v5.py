import copy
import inspect
from collections import Counter
from pathlib import Path
import sys
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from fdt_local.model_masked import Config
from research.text_diffusion_v1 import TextDiffusion, TextConfig, pack_prompts, PAD
from research.text_span_reader_v4 import SpanSemanticGenerator
from research.text_language_continuation_v5 import LanguageContinuation, ContinuationConfig

torch.set_num_threads(2)


def model():
    core = Config(name='tiny-contract', lang_dim=32, lang_heads=4, lang_ff=64, lang_layers=3,
                  z_dim=264, core_heads=4, core_ff=64, diffusion_layers=1)
    policy = ContinuationConfig(prompt_bytes=32, target_bytes_with_eos=12,
                                diffusion_steps=2, train_language_tail_blocks=1)
    return LanguageContinuation(core, policy)


def inputs():
    tokens = pack_prompts(['A child took a walk.', 'Rain fell on the roof.'], 32)
    ids = torch.full((2, 12), PAD, dtype=torch.long)
    ids[0, :6] = torch.tensor(list(b' Fine.') ); ids[0, 6] = 256
    ids[1, :5] = torch.tensor(list(b' Wet.') ); ids[1, 5] = 256
    return tokens, ids, torch.tensor([.25, .75]), torch.randn(2, 12, 264), torch.randn(2, 12, 264)


def test_live_language_tail_receives_both_losses_frozen_prefix_never_gradients():
    m = model(); args = inputs()
    loss, out = m.training_loss(*args)
    loss.backward()
    assert torch.isfinite(loss)
    assert m.vp.language.blocks[-1].attn.q.weight.grad.abs().sum() > 0
    assert m.vp.velocity.weight.grad.abs().sum() > 0 and m.decoder.head.weight.grad.abs().sum() > 0
    assert m.vp.language.embedding.weight.grad is None
    assert all(p.grad is None for b in m.vp.language.blocks[:-1] for p in b.parameters())
    assert not out['free_proposal'].requires_grad
    m.zero_grad(set_to_none=True)
    _, out = m.training_loss(*args); out['AR_byte_ce'].backward()
    assert m.vp.language.blocks[-1].attn.q.weight.grad.abs().sum() > 0
    assert m.vp.velocity.weight.grad is None  # AR cannot backpropagate through a detached rollout.


def test_free_proposal_no_gold_target_lookahead_and_real_call_count():
    m = model(); args = inputs(); counts = Counter()
    _, first = m.training_loss(*args, charge=lambda k: counts.update([k]))
    changed = list(args); changed[1] = args[1].clone(); changed[1][:, 1] = ord('z')
    _, second = m.training_loss(*changed)
    assert torch.equal(first['free_proposal'], second['free_proposal'])
    assert counts == {'encode': 1, 'denoise': 3, 'decoder': 1}
    assert sum(counts.values()) == m.policy.diffusion_steps+3
    assert list(inspect.signature(m.free_proposal).parameters) == ['encoded', 'noise', 'charge']
    assert list(inspect.signature(m.generate).parameters) == ['tokens', 'proposal_noise', 'charge', 'zero_latent']


def test_decoder_causality_and_ignored_padding_labels():
    m = model(); args = inputs()
    _, a = m.training_loss(*args)
    other = list(args); other[1] = args[1].clone(); other[1][:, 4] = ord('x')
    _, b = m.training_loss(*other)
    assert torch.equal(a['AR_logits'][:, :5], b['AR_logits'][:, :5])
    assert not torch.equal(a['AR_logits'][:, 5], b['AR_logits'][:, 5])
    m.eval()
    with pytest.raises(RuntimeError, match='training'):
        m.training_loss(*args)


def test_strict_warmstart_copies_all_except_declared_positions_without_source_mutation():
    m = model(); oldvp = TextDiffusion(m.core_config, TextConfig(32, 8, 2))
    oldspan = SpanSemanticGenerator(m.core_config)
    vp = copy.deepcopy(oldvp.state_dict()); span = copy.deepcopy(oldspan.state_dict())
    before = vp['positions.weight'].clone()
    info = m.warmstart(vp, span)
    assert info['resized_only'] == ['vp.positions.weight', 'decoder.positions.weight']
    assert torch.equal(vp['positions.weight'], before)
    assert torch.equal(m.vp.velocity.weight, oldvp.velocity.weight)
    assert torch.equal(m.decoder.head.weight, oldspan.decoder.head.weight)
    bad = dict(vp); bad.pop('velocity.weight')
    with pytest.raises(ValueError, match='missing VP'):
        m.warmstart(bad, span)


def test_exact_candidate_model_optimizer_rng_next_update_cpu_restore():
    m = model(); args = inputs(); opt = torch.optim.AdamW((p for p in m.parameters() if p.requires_grad), lr=1e-4)
    loss, _ = m.training_loss(*args); loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    state, adam, rng = copy.deepcopy(m.state_dict()), copy.deepcopy(opt.state_dict()), torch.get_rng_state().clone()
    # Fixed input/noise is part of this tiny contract, not a complete runner restore.
    loss, _ = m.training_loss(*args); loss.backward(); opt.step()
    restored = model(); restored.load_state_dict(state, strict=True)
    opt2 = torch.optim.AdamW((p for p in restored.parameters() if p.requires_grad), lr=1e-4)
    opt2.load_state_dict(adam); torch.set_rng_state(rng)
    loss2, _ = restored.training_loss(*args); loss2.backward(); opt2.step()
    assert torch.equal(loss.detach(), loss2.detach())
    assert all(torch.equal(a, restored.state_dict()[k]) for k, a in m.state_dict().items())
