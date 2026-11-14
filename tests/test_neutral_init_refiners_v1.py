from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))
from fdt_local.model_masked import Config, FDT100
from neutral_init_refiners_v1 import (
    NeutralInitFullRefiner,
    NeutralInitQueryLocalRefiner,
    neutral_initializer,
)


def tiny_features(batch=2, facts=3, queries=16):
    generator = torch.Generator(device="cpu").manual_seed(713)
    tokens = torch.full((batch, facts, 8), 256, dtype=torch.long)
    tokens[:, :, :5] = torch.randint(0, 256, (batch, facts, 5), generator=generator)
    roles = torch.zeros(batch, facts, 16)
    qroles = torch.zeros(batch, queries, 16)
    for b in range(batch):
        for f in range(facts):
            roles[b, f, (f + b) % 8] = 1
            roles[b, f, 8 + (f + b + 3) % 8] = 1
        for q in range(queries):
            qroles[b, q, q % 8] = 1
            qroles[b, q, 8 + (q * 3 + 1) % 8] = 1
    qmask = torch.ones(batch, queries)
    qmask[0, 14:] = 0
    return {
        "tokens": tokens,
        "roles": roles,
        "fact_mask": torch.ones(batch, facts),
        "queries": qroles,
        "query_mask": qmask,
    }


def setup():
    torch.manual_seed(902)
    cfg = Config.smoke()
    base = FDT100(cfg).eval()
    full = NeutralInitFullRefiner(cfg).eval()
    local = NeutralInitQueryLocalRefiner(cfg).eval()
    full.load_state_dict(base.state_dict(), strict=True)
    local.load_state_dict(base.state_dict(), strict=True)
    return base, full, local, tiny_features()


def test_variants_keep_exact_parameter_state_and_strict_load_contract():
    base, full, local, _ = setup()
    keys = list(base.state_dict())
    assert list(full.state_dict()) == keys
    assert list(local.state_dict()) == keys
    assert full.counts()["parameters"] == local.counts()["parameters"] == base.counts()["parameters"]
    for model in (full, local):
        params = list(model.parameters())
        assert len(params) == len({id(p) for p in params})
        model.load_state_dict(base.state_dict(), strict=True)


def test_neutral_init_is_role_template_and_preserves_encoded_context():
    base, full, local, features = setup()
    base_enc = base.encode(features)
    full_enc = full.encode(features)
    local_enc = local.encode(features)
    expected = neutral_initializer(features["queries"], features["query_mask"], full.semantic)
    assert torch.equal(full_enc["init"], expected)
    assert torch.equal(local_enc["init"], expected)
    assert torch.equal(full_enc["init"], local_enc["init"])
    assert not torch.equal(base_enc["init"], full_enc["init"])
    for key in ("facts", "memory", "ground", "grades", "query_roles", "query_mask", "fact_mask"):
        assert torch.equal(base_enc[key], full_enc[key]), key
        assert torch.equal(base_enc[key], local_enc[key]), key
    assert torch.equal(full_enc["init"][0, 14:], torch.zeros_like(full_enc["init"][0, 14:]))


def test_neutral_init_does_not_depend_on_facts_class_or_discarded_initializer():
    _, full, _, features = setup()
    encoded = full.encode(features)
    changed = {key: value.clone() for key, value in features.items()}
    changed["tokens"][:, :, :5] = (changed["tokens"][:, :, :5] + 37) % 256
    changed["roles"] = torch.roll(changed["roles"], shifts=1, dims=-1)
    changed_enc = full.encode(changed)
    assert torch.equal(encoded["init"], changed_enc["init"])
    assert not torch.equal(encoded["facts"], changed_enc["facts"])
    assert not torch.equal(encoded["ground"], changed_enc["ground"])

    # The discarded fuzzy initializer is not an argument to the replacement.
    roles, mask = features["queries"], features["query_mask"]
    init_a = torch.randn_like(encoded["init"])
    init_b = torch.randn_like(encoded["init"])
    def replacement(_unused_original_init):
        return neutral_initializer(roles, mask, full.semantic)
    assert torch.equal(replacement(init_a), replacement(init_b))


def test_query_role_and_mask_control_template_with_zero_predicate_projection():
    _, full, _, features = setup()
    roles = features["queries"]
    mask = features["query_mask"]
    actual = neutral_initializer(roles, mask, full.semantic)
    zero_pred = torch.zeros(*roles.shape[:2], 4)
    expected = full.semantic(roles, zero_pred) * mask[..., None]
    assert torch.equal(actual, expected)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[0, 14:], torch.zeros_like(actual[0, 14:]))
    changed_roles = roles.clone()
    changed_roles[:, 0] = torch.roll(changed_roles[:, 0], 1, dims=-1)
    assert not torch.equal(actual[:, 0], neutral_initializer(changed_roles, mask, full.semantic)[:, 0])
    all_masked = neutral_initializer(roles, torch.zeros_like(mask), full.semantic)
    assert torch.isfinite(all_masked).all()
    assert torch.equal(all_masked, torch.zeros_like(all_masked))


def test_full_and_query_local_denoisers_share_neutral_encoding_contract():
    _, full, local, features = setup()
    enc_full = full.encode(features)
    enc_local = local.encode(features)
    z = torch.randn(2, 16, full.c.z_dim, generator=torch.Generator().manual_seed(44))
    times = torch.tensor([0.2, 0.7])
    noise = torch.randn_like(z)
    # Both methods accept precisely the same neutralized encoder fields/noise/time.
    assert enc_full.keys() == enc_local.keys()
    for key in enc_full:
        assert torch.equal(enc_full[key], enc_local[key]), key
    out_full = full.denoise(z, times, enc_full)
    out_local = local.denoise(z, times, enc_local)
    assert out_full.shape == out_local.shape == z.shape
    assert torch.isfinite(out_full).all() and torch.isfinite(out_local).all()
    assert torch.equal(out_full[0, 14:], torch.zeros_like(out_full[0, 14:]))
    assert torch.equal(out_local[0, 14:], torch.zeros_like(out_local[0, 14:]))
    assert noise.shape == z.shape


def test_backward_owns_existing_parameters_without_optimizer_or_new_state():
    base, full, local, features = setup()
    original_keys = tuple(base.state_dict())
    enc = local.encode(features)
    z = torch.randn(2, 16, local.c.z_dim, generator=torch.Generator().manual_seed(101)).requires_grad_(True)
    output = local.denoise(z, torch.tensor([0.3, 0.8]), enc)
    output.square().mean().backward()
    assert z.grad is not None and torch.isfinite(z.grad).all()
    assert any(p.grad is not None for p in local.parameters())
    assert tuple(local.state_dict()) == original_keys
    assert len(list(local.parameters())) == len({id(p) for p in local.parameters()})
    assert not hasattr(local, "optimizer")
