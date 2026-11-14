import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from fdt_local.model_masked import Config
from research.three_memory_codec_v33 import (
    ThreeMemoryCodec, contextual_snapshot, DifferentiableCodecCommit,
)
from research.three_memory_contract_v32 import ContextualFuzzyBank, assert_codec_training_only
from research.owt_extraction_tail_v29 import ExtractionTailFuzzyAE

torch.set_num_threads(2)


def config():
    return Config(name='v33-tiny-contract', lang_dim=24, lang_heads=3, lang_ff=48,
        lang_layers=5, z_dim=32, core_heads=4, core_ff=64, init_layers=1,
        diffusion_layers=1, anchors=16, key_dim=16, value_dim=16, top_k=4, rank=4)


def candidate():
    torch.manual_seed(83)
    return ThreeMemoryCodec(config())


def data(texts=('not 25', '25 now'), padded=20):
    tokens = torch.full((len(texts), padded), 256, dtype=torch.long)
    targets = torch.full((len(texts), 129), 257, dtype=torch.long)
    for row, text in enumerate(texts):
        raw = list(text.encode('utf-8'))
        tokens[row, :len(raw)] = torch.tensor(raw)
        targets[row, :len(raw)+1] = torch.tensor(raw+[256])
    metadata = dict(source_ids=tuple('doc'+str(i) for i in range(len(texts))),
                    source_versions=('v1',)*len(texts), model_epoch='fixture-update0')
    return tokens, targets, metadata


def test_original_language_pool_exact_and_source_local_padding_invariance():
    m = candidate().eval(); tokens, _, meta = data()
    with torch.no_grad():
        _, pooled, _ = m.token_features(tokens)
        assert torch.equal(pooled, m.core.language(tokens[:, None])[:, 0])
        a = m.encode_input(tokens, **meta)
        b = m.encode_input(tokens[:, :6], **meta)
        solo_meta = dict(source_ids=('doc0',), source_versions=('v1',), model_epoch=meta['model_epoch'])
        solo = m.encode_input(tokens[:1], **solo_meta)
    for name in ('long_N', 'long_D', 'recent_N', 'recent_D'):
        assert torch.allclose(getattr(a.memories.contextual, name), getattr(b.memories.contextual, name), atol=1e-7, rtol=1e-6)
    assert torch.allclose(a.commit.reader_tensor(), b.commit.reader_tensor(), atol=1e-6)
    assert torch.allclose(a.commit.reader_tensor()[:1], solo.commit.reader_tensor(), atol=1e-6)
    assert not torch.equal(a.diagnostics['token_values'][0, 0], a.diagnostics['token_values'][0, 1])
    assert a.memories.persistent.certified_D.count_nonzero() == 0


def test_vectorized_fp64_context_matches_sequential_reference_including_invalid_steps():
    torch.manual_seed(7)
    mu = torch.rand(2, 5, 3, requires_grad=True)
    values = torch.randn(2, 5, 4, requires_grad=True)
    valid = torch.tensor([[True, False, True, True, False], [False, True, True, False, True]])
    decay = torch.tensor([.6, .8, .9], requires_grad=True)
    out = contextual_snapshot(mu, values, valid, decay, (('a', 'v1', 'sha'), ('b', 'v1', 'sha')), 'e1')
    ref = ContextualFuzzyBank.empty(2, 3, 4, decay, 'batch-fixture', 'v1')
    for i in range(5): ref = ref.write_after_read(mu[:, i], values[:, i], valid[:, i])
    for name in ('long_N', 'long_D', 'recent_N', 'recent_D'):
        assert getattr(out, name).dtype == torch.float64
        assert torch.allclose(getattr(out, name), getattr(ref, name), atol=1e-14, rtol=1e-14)
    out.recent_N.sum().backward()
    assert values.grad is not None and mu.grad is not None and decay.grad.abs().sum() > 0
    empty = contextual_snapshot(mu.detach(), values.detach(), torch.zeros_like(valid), decay.detach(), out.source_revisions, 'e1')
    assert all(x.count_nonzero() == 0 for x in empty.means())


def test_each_memory_reaches_neural_semantic_commit_without_raw_context_or_denoising():
    m = candidate().eval(); tokens, _, meta = data()
    with torch.no_grad():
        normal = m.encode_input(tokens, **meta)
        z = normal.commit.reader_tensor()
        for memory in ('persistent', 'contextual', 'exact'):
            other = m.encode_input(tokens, **meta, disabled_memories=(memory,)).commit.reader_tensor()
            assert (z-other).abs().max() > 1e-6
        ctx, mask = m.context(normal.commit)
    assert ctx.shape[1] == m.c.meaning_views and mask.all()
    assert normal.diagnostics['denoise_calls'] == normal.diagnostics['copy_bytes'] == 0
    assert set(vars(normal.commit)) == {'_semantic', 'memory_revisions', 'model_epoch', 'phase'}
    for forbidden in (normal, normal.memories, {'views': z}, tokens):
        with pytest.raises(TypeError): m.context(forbidden)


def test_exact_utf8_identity_epoch_null_mass_and_conflicting_source_revisions():
    m = candidate().eval(); tokens, _, meta = data(('Cafe\u0301', 'Café'))
    with torch.no_grad():
        episode = m.encode_input(tokens, **meta)
        exact = episode.memories.exact
        assert exact.sources[0].sha256 != exact.sources[1].sha256
        for source, spans in zip(exact.sources, exact.spans):
            assert b''.join(source.span(a, b) for a, b in spans) == source.raw
            assert all(len(source.span(a, b).decode('utf-8')) == 1 for a, b in spans)
        d = episode.diagnostics
        assert torch.allclose(d['candidate_weights'].sum(-1)+d['null_weight'], torch.ones(2, 2))
        assert (d['null_weight'] > 0).all()
        # Candidate tables are padded; invalid entries retain zero probability.
        for row in range(2):
            chosen_valid = exact.valid[row][d['candidate_ids'][row]]
            assert d['candidate_weights'][row][~chosen_valid].count_nonzero() == 0
        with pytest.raises(ValueError): m.read_exact(exact, episode.commit.reader_tensor(), 'after-update1')
        m.memory_codec.exact_null.fill_(30.)
        read, diagnostics = m.read_exact(exact, episode.commit.reader_tensor(), meta['model_epoch'])
        assert (diagnostics['null_weight'] > .99999).all() and read.abs().max() < 1e-9
    with pytest.raises(ValueError):
        m.encode_input(tokens, **{**meta, 'source_ids': ('same', 'same')})


def test_differentiable_commit_and_byte_seal_have_independent_readers():
    z = torch.randn(2, 2, 32, requires_grad=True)
    before = z.detach().clone()
    c = DifferentiableCodecCommit.from_semantic(z, (('a', 'v1', 'sha'), ('b', 'v1', 'sha')), 'e1')
    reader = c.reader_tensor(); reader.zero_()
    assert torch.equal(c.reader_tensor(), before) and torch.equal(z.detach(), before)
    c.reader_tensor().square().sum().backward()
    assert torch.equal(z.grad, 2*before)
    sealed = c.seal(); output = sealed.reader_tensor(); output.add_(100)
    assert torch.equal(sealed.reader_tensor(), before) and not sealed.reader_tensor().requires_grad
    m = candidate().eval()
    assert torch.equal(m.context(c)[0], m.context(sealed)[0])


def test_actual_codec_loss_keeps_encoder_gradients_and_frozen_reasoning_untouched():
    m = candidate().train(); tokens, targets, meta = data()
    frozen = {n: p.detach().clone() for n, p in m.named_parameters() if not p.requires_grad}
    calls = []
    # The core's semantic refiner is deliberately unusable in this stage.
    for block in m.core.refiner:
        block.forward = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError('denoising called'))
    loss, episode = m.reconstruction_loss(tokens, targets, **meta, charge=calls.append)
    assert calls == ['encode', 'decoder', 'decoder'] and torch.isfinite(loss)
    loss.backward()
    for name, parameter in m.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else: assert parameter.grad is None, name
    for name in ('core.language.blocks.4.attn.q.weight', 'memory_codec.exact_key.weight',
                 'memory_codec.exact_query.weight', 'memory_codec.recent_decay_logits', 'memory_codec.long_gate.0.weight'):
        assert dict(m.named_parameters())[name].grad.abs().sum() > 0, name
    assert_codec_training_only(m.named_parameters(), episode.diagnostics['denoise_calls'])
    optimizer = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    optimizer.step()
    assert all(torch.equal(dict(m.named_parameters())[n], v) for n, v in frozen.items())
    assert not torch.cuda.is_initialized()


def test_strict_old_tensor_warmstart_new_keys_only_and_same_input_targets():
    legacy = ExtractionTailFuzzyAE(config()); m = candidate()
    source = {k: v.clone() for k, v in legacy.state_dict().items()}
    added = m.import_legacy_weights(source)
    assert added and all(n.startswith('memory_codec.') for n in added)
    assert all(torch.equal(m.state_dict()[n], v) for n, v in source.items())
    assert all(torch.equal(legacy.state_dict()[n], v) for n, v in source.items())
    with pytest.raises(ValueError): m.import_legacy_weights({k: v for k, v in source.items() if k != 'core.prior'})
    tokens, targets, meta = data(); targets[0, 0] = 65
    with pytest.raises(ValueError, match='SAME'): m.reconstruction_loss(tokens, targets, **meta)
    with pytest.raises(TypeError): m.extract(tokens)
    with pytest.raises(ValueError): m.encode_input(tokens.masked_fill(tokens == 256, 257), **meta)
    bad = tokens.clone(); bad[0, 1] = 256
    with pytest.raises(ValueError): m.encode_input(bad, **meta)
    with pytest.raises(ValueError): assert_codec_training_only(m.named_parameters(), 1)
