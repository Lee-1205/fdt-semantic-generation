from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))
from fdt_local.model_masked import Config, FDT100
from query_bound_refiner_v1 import QueryBoundRefiner, bound_core_forward, build_query_binding_mask


def features(batch=2, facts=4, queries=16):
    gen = torch.Generator(device="cpu").manual_seed(1617)
    tokens = torch.full((batch, facts, 8), 256, dtype=torch.long)
    tokens[:, :, :5] = torch.randint(0, 256, (batch, facts, 5), generator=gen)
    roles = torch.zeros(batch, facts, 16)
    qroles = torch.zeros(batch, queries, 16)
    for b in range(batch):
        for f in range(facts):
            roles[b, f, (f + b) % 8] = 1
            roles[b, f, 8 + (f + b + 2) % 8] = 1
        for q in range(queries):
            qroles[b, q, q % 8] = 1
            qroles[b, q, 8 + (q * 5 + 1) % 8] = 1
    qmask = torch.ones(batch, queries)
    qmask[0, 13:] = 0
    fmask = torch.ones(batch, facts)
    fmask[1, -1] = 0
    return {"tokens": tokens, "roles": roles, "fact_mask": fmask,
            "queries": qroles, "query_mask": qmask}


def setup():
    torch.manual_seed(181)
    cfg = Config.smoke()
    base = FDT100(cfg).eval()
    candidate = QueryBoundRefiner(cfg).eval()
    candidate.load_state_dict(base.state_dict(), strict=True)
    feat = features()
    enc = candidate.encode(feat)
    z = torch.randn(2, 16, cfg.z_dim, generator=torch.Generator().manual_seed(199))
    times = torch.tensor([0.25, 0.75])
    return base, candidate, feat, enc, z, times


def test_strict_state_keys_counts_and_no_gold_or_extra_parameters():
    base, candidate, feat, enc, _, _ = setup()
    assert list(candidate.state_dict()) == list(base.state_dict())
    assert candidate.counts()["parameters"] == base.counts()["parameters"]
    params = list(candidate.parameters())
    assert len(params) == len({id(p) for p in params})
    assert not ({"labels", "gold", "target"} & set(feat))
    assert not ({"labels", "gold", "target"} & set(enc))
    candidate.load_state_dict(base.state_dict(), strict=True)


def test_binding_mask_assigns_only_own_recalled_and_init_all_valid_facts_and_null():
    qmask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    fmask = torch.tensor([[1, 0]], dtype=torch.bool)
    mask = build_query_binding_mask(qmask, fmask)
    assert mask.shape == (1, 3, 2 * 3 + 2 + 1)
    # recalled range [0:Q], init range [Q:2Q]
    assert torch.equal(mask[0, 0], torch.tensor([1,0,0, 1,0,0, 1,0, 1], dtype=torch.bool))
    assert torch.equal(mask[0, 1], torch.tensor([0,1,0, 0,1,0, 1,0, 1], dtype=torch.bool))
    assert torch.equal(mask[0, 2], torch.tensor([0,0,0, 0,0,0, 1,0, 1], dtype=torch.bool))


def test_original_encoder_and_query_relational_context_paths_are_unchanged():
    base, candidate, feat, enc, z, _ = setup()
    base_enc = base.encode(feat)
    for key in base_enc:
        assert torch.equal(base_enc[key], enc[key]), key
    # Candidate inherits the original condition_at implementation unchanged.
    base_context = base.condition_at(z, base_enc)
    candidate_context = candidate.condition_at(z, enc)
    assert torch.equal(base_context, candidate_context)
    # Slot binding changes only cross-attention access; shape remains original context layout.
    assert base_context.shape[1] == 2 * z.shape[1] + enc["facts"].shape[1] + 1
    binding = build_query_binding_mask(enc["query_mask"], enc["fact_mask"])
    q = 4
    queries = z.shape[1]
    assert binding[1, q, q]  # this slot's recalled vector
    assert binding[1, q, queries + q]  # this slot's initializer vector
    assert not binding[1, q, q - 1]
    assert not binding[1, q, queries + q - 1]
    assert binding[1, q, 2 * queries:2 * queries + enc["facts"].shape[1] - 1].all()
    assert binding[1, q, -1]  # null context


def test_full_query_self_attention_path_matches_original_core_when_cross_mask_is_open():
    base, _, _, _, _, _ = setup()
    block = base.refiner[0]
    qmask = torch.ones(2, 5, dtype=torch.bool)
    x = torch.randn(2, 5, base.c.z_dim, generator=torch.Generator().manual_seed(555))
    context = torch.randn(2, 7, base.c.z_dim, generator=torch.Generator().manual_seed(556))
    full_context_mask = torch.ones(2, 7, dtype=torch.bool)
    open_per_query_mask = full_context_mask[:, None, :].expand(2, 5, 7)
    expected = block(x, context, qmask, full_context_mask)
    actual = bound_core_forward(block, x, context, qmask, open_per_query_mask)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)

    # Cross context is fixed and open, so another query can affect slot 0 through
    # the unchanged full-query self-attention path.
    changed = x.clone()
    changed[:, 1] += 2.5
    altered = bound_core_forward(block, changed, context, qmask, open_per_query_mask)
    assert not torch.allclose(altered[:, 0], actual[:, 0], atol=1e-7, rtol=1e-7)


def test_denoise_is_finite_masked_and_ablate_path_is_supported():
    _, candidate, _, enc, z, times = setup()
    out = candidate.denoise(z, times, enc)
    out_ablate = candidate.denoise(z, times, enc, ablate=True)
    assert out.shape == z.shape and out_ablate.shape == z.shape
    assert torch.isfinite(out).all() and torch.isfinite(out_ablate).all()
    assert torch.equal(out[0, 13:], torch.zeros_like(out[0, 13:]))
    assert torch.equal(out_ablate[0, 13:], torch.zeros_like(out_ablate[0, 13:]))
    all_masked = dict(enc)
    all_masked["query_mask"] = torch.zeros_like(enc["query_mask"])
    result = candidate.denoise(z, times, all_masked)
    assert torch.isfinite(result).all()
    assert torch.equal(result, torch.zeros_like(result))


def test_aligned_query_permutation_preserves_outputs():
    _, candidate, _, enc, z, times = setup()
    expected = candidate.denoise(z, times, enc)
    perm = torch.tensor([8, 2, 15, 0, 12, 4, 9, 1, 14, 6, 3, 11, 5, 7, 13, 10])
    perm_enc = dict(enc)
    for key in ("init", "query_roles", "query_mask"):
        perm_enc[key] = enc[key][:, perm]
    actual = candidate.denoise(z[:, perm], times, perm_enc)
    assert torch.allclose(actual, expected[:, perm], atol=1e-6, rtol=1e-6)


def test_backward_gradients_are_finite_and_parameter_ownership_is_shared():
    _, candidate, _, enc, z, times = setup()
    z = z.requires_grad_(True)
    output = candidate.denoise(z, times, enc)
    output.square().mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    grads = [p.grad for p in candidate.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    params = list(candidate.parameters())
    assert len(params) == len({id(p) for p in params})
    assert not hasattr(candidate, "optimizer")
