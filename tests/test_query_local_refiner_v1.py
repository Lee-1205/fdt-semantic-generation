from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from query_local_refiner_v1 import QueryLocalRefiner, flatten_single_query_views
from fdt_local.model_masked import Config, FDT100


def tiny_features(batch=2, facts=3, queries=16):
    generator = torch.Generator(device="cpu").manual_seed(407)
    tokens = torch.full((batch, facts, 8), 256, dtype=torch.long)
    tokens[:, :, :5] = torch.randint(0, 256, (batch, facts, 5), generator=generator)
    roles = torch.zeros(batch, facts, 16)
    qroles = torch.zeros(batch, queries, 16)
    for b in range(batch):
        for f in range(facts):
            s, o = (f + b) % 8, (f + b + 3) % 8
            roles[b, f, s] = 1
            roles[b, f, 8 + o] = 1
        for q in range(queries):
            s, o = q % 8, (q * 3 + 1) % 8
            qroles[b, q, s] = 1
            qroles[b, q, 8 + o] = 1
    mask = torch.ones(batch, queries)
    mask[0, 14:] = 0
    return {"tokens": tokens, "roles": roles, "fact_mask": torch.ones(batch, facts),
            "queries": qroles, "query_mask": mask}


def explicit_singleton_loop(model, z, times, enc, ablate=False):
    batch, queries, _ = z.shape
    outputs = []
    for b in range(batch):
        row = []
        for q in range(queries):
            single = {}
            for key, value in enc.items():
                single[key] = value[b:b+1, q:q+1] if key in ("init", "query_roles", "query_mask") else value[b:b+1]
            row.append(model.denoise(z[b:b+1, q:q+1], times[b:b+1], single, ablate=ablate))
        outputs.append(torch.cat(row, dim=1))
    return torch.cat(outputs, dim=0)


def setup_models():
    torch.manual_seed(121)
    cfg = Config.smoke()
    original = FDT100(cfg).eval()
    candidate = QueryLocalRefiner(cfg).eval()
    candidate.load_state_dict(original.state_dict(), strict=True)
    features = tiny_features()
    encoded = original.encode(features)
    z = torch.randn(2, 16, cfg.z_dim, generator=torch.Generator().manual_seed(901))
    times = torch.tensor([0.125, 0.875])
    return original, candidate, features, encoded, z, times


def test_candidate_preserves_parameter_names_count_and_strict_load():
    original, candidate, features, _, _, _ = setup_models()
    assert list(candidate.state_dict()) == list(original.state_dict())
    assert candidate.counts()["parameters"] == original.counts()["parameters"]
    assert len(list(candidate.parameters())) == len({id(p) for p in candidate.parameters()})
    assert candidate.encode.__func__ is FDT100.encode
    candidate.load_state_dict(original.state_dict(), strict=True)
    with torch.no_grad():
        base_encoding=original.encode(features)
        candidate_encoding=candidate.encode(features)
    assert base_encoding.keys() == candidate_encoding.keys()
    for key in base_encoding:
        assert torch.equal(base_encoding[key], candidate_encoding[key]), key


def test_batched_query_local_denoise_matches_explicit_original_singleton_loop():
    original, candidate, _, encoded, z, times = setup_models()
    expected = explicit_singleton_loop(original, z, times, encoded)
    actual = candidate.denoise(z, times, encoded)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    expected_ablate = explicit_singleton_loop(original, z, times, encoded, ablate=True)
    actual_ablate = candidate.denoise(z, times, encoded, ablate=True)
    assert torch.allclose(actual_ablate, expected_ablate, atol=1e-6, rtol=1e-6)


def test_flattening_keeps_per_graph_time_and_repeats_only_static_context():
    _, _, _, encoded, z, times = setup_models()
    single, flat_z, flat_t = flatten_single_query_views(encoded, z, times)
    assert flat_z.shape == (32, 1, z.shape[-1])
    assert flat_t.tolist() == [0.125] * 16 + [0.875] * 16
    assert single["query_mask"].shape == (32, 1)
    for key in ("facts", "memory", "ground", "grades", "fact_mask"):
        for b in range(2):
            assert torch.equal(single[key][b*16:(b+1)*16], encoded[key][b:b+1].expand(16, *encoded[key].shape[1:]))


def test_partial_and_all_masked_queries_remain_finite_and_zeroed():
    _, candidate, _, encoded, z, times = setup_models()
    partial = candidate.denoise(z, times, encoded)
    assert torch.isfinite(partial).all()
    assert torch.equal(partial[0, 14:], torch.zeros_like(partial[0, 14:]))
    all_masked = dict(encoded)
    all_masked["query_mask"] = torch.zeros_like(encoded["query_mask"])
    output = candidate.denoise(z, times, all_masked)
    assert torch.isfinite(output).all()
    assert torch.equal(output, torch.zeros_like(output))


def test_query_local_denoiser_is_permutation_equivariant_and_slot_isolated():
    _, candidate, _, encoded, z, times = setup_models()
    reference = candidate.denoise(z, times, encoded)
    perm = torch.tensor([7, 2, 15, 0, 11, 4, 9, 1, 14, 6, 3, 12, 5, 8, 13, 10])
    permuted_enc = dict(encoded)
    for key in ("init", "query_roles", "query_mask"):
        permuted_enc[key] = encoded[key][:, perm]
    permuted = candidate.denoise(z[:, perm], times, permuted_enc)
    assert torch.allclose(permuted, reference[:, perm], atol=1e-6, rtol=1e-6)

    changed_z = z.clone(); changed_z[:, 1] += 4
    changed_enc = dict(encoded)
    changed_enc["init"] = encoded["init"].clone(); changed_enc["init"][:, 1] -= 3
    changed_enc["query_roles"] = encoded["query_roles"].clone(); changed_enc["query_roles"][:, 1] = torch.flip(encoded["query_roles"][:, 1], dims=(-1,))
    changed = candidate.denoise(changed_z, times, changed_enc)
    assert torch.allclose(changed[:, 0], reference[:, 0], atol=1e-6, rtol=1e-6)


def test_backward_gradients_match_reference_loop_without_duplicate_parameter_ownership():
    original, candidate, _, encoded, z, times = setup_models()
    encoded_actual = {key: value.detach() for key, value in encoded.items()}
    encoded_expected = {key: value.detach() for key, value in encoded.items()}
    z1=z.clone().requires_grad_(True)
    actual=candidate.denoise(z1,times,encoded_actual)
    actual.square().sum().backward()
    actual_grads={name: p.grad.detach().clone() for name,p in candidate.named_parameters() if p.grad is not None}
    actual_z_grad=z1.grad.detach().clone()

    z2=z.clone().requires_grad_(True)
    expected=explicit_singleton_loop(original,z2,times,encoded_expected)
    expected.square().sum().backward()
    expected_grads={name:p.grad.detach().clone() for name,p in original.named_parameters() if p.grad is not None}
    assert actual_grads.keys() == expected_grads.keys()
    for name in actual_grads:
        assert torch.allclose(actual_grads[name], expected_grads[name], atol=2e-6, rtol=2e-5), name
    assert torch.allclose(actual_z_grad,z2.grad,atol=2e-6,rtol=2e-5)
    named=list(candidate.named_parameters())
    assert len(named)==len({id(p) for _,p in named})
    assert len(actual_grads)==len(expected_grads)
