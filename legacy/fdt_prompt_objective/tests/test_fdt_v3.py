import torch

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model
from fdt_rlm.models.causal_lm import dense_causal_prefix_aggregate
from fdt_rlm.models.fdt_v3 import (
    sparse_chunked_prefix_summaries,
    sparse_segmented_prefix_summaries,
)


def v3_config():
    return ModelConfig(
        vocab_size=79,
        pad_token_id=0,
        eos_token_id=1,
        model_type="fdt_v3",
        dim=32,
        n_layers=2,
        n_heads=4,
        mlp_ratio=2,
        max_seq_len=16,
        dropout=0.0,
        use_rope=False,
        num_anchors=11,
        top_k=3,
        local_attention_window=4,
        anchor_recency_bias=3.0,
        aggregation_impl="sparse_chunked_scan",
        anchor_scan_chunk_size=4,
        use_cached_normalized_anchors=True,
    )


def test_sparse_segmented_scan_matches_dense_prefix_mean():
    torch.manual_seed(31)
    bsz, seq_len, top_k, anchors, dim = 2, 7, 3, 9, 5
    logits = torch.randn(bsz, seq_len, anchors)
    top_logits, indices = torch.topk(logits, top_k, dim=-1)
    membership = top_logits.softmax(-1)
    values = torch.randn(bsz, seq_len, dim)
    mask = torch.ones(bsz, seq_len)
    long_summary, recent_summary = sparse_segmented_prefix_summaries(
        membership,
        indices,
        values,
        mask,
        anchors,
        seq_len,
        0.0,
    )
    sparse_context = torch.einsum("bnk,bnkd->bnd", membership, long_summary)
    dense_context = dense_causal_prefix_aggregate(membership, indices, values, mask, anchors)
    assert torch.allclose(sparse_context, dense_context, atol=1e-5, rtol=1e-5)
    assert torch.allclose(long_summary, recent_summary, atol=1e-6, rtol=1e-6)


def test_sparse_chunked_scan_matches_dense_prefix_mean():
    torch.manual_seed(32)
    bsz, seq_len, top_k, anchors, dim = 2, 11, 3, 9, 5
    logits = torch.randn(bsz, seq_len, anchors)
    top_logits, indices = torch.topk(logits, top_k, dim=-1)
    membership = top_logits.softmax(-1)
    values = torch.randn(bsz, seq_len, dim)
    mask = torch.ones(bsz, seq_len)
    long_summary, recent_summary = sparse_chunked_prefix_summaries(
        membership,
        indices,
        values,
        mask,
        anchors,
        seq_len,
        0.0,
        4,
    )
    sparse_context = torch.einsum("bnk,bnkd->bnd", membership, long_summary)
    dense_context = dense_causal_prefix_aggregate(membership, indices, values, mask, anchors)
    assert torch.allclose(sparse_context, dense_context, atol=1e-5, rtol=1e-5)
    assert torch.allclose(long_summary, recent_summary, atol=1e-6, rtol=1e-6)


def test_sparse_chunked_scan_matches_dense_across_multiple_slabs():
    torch.manual_seed(320)
    bsz, seq_len, top_k, anchors, dim = 1, 137, 2, 7, 3
    logits = torch.randn(bsz, seq_len, anchors)
    top_logits, indices = torch.topk(logits, top_k, dim=-1)
    membership = top_logits.softmax(-1)
    values = torch.randn(bsz, seq_len, dim)
    mask = torch.ones(bsz, seq_len)
    mask[:, 51] = 0

    long_summary, recent_summary = sparse_chunked_prefix_summaries(
        membership,
        indices,
        values,
        mask,
        anchors,
        16_384,
        2.5,
        4,
    )
    sparse_long = torch.einsum("bnk,bnkd->bnd", membership, long_summary)
    sparse_recent = torch.einsum("bnk,bnkd->bnd", membership, recent_summary)
    reference_long, reference_recent = sparse_segmented_prefix_summaries(
        membership, indices, values, mask, anchors, 16_384, 2.5
    )
    dense_long = torch.einsum("bnk,bnkd->bnd", membership, reference_long)
    dense_recent = torch.einsum("bnk,bnkd->bnd", membership, reference_recent)

    assert torch.allclose(sparse_long, dense_long, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sparse_recent, dense_recent, atol=1e-5, rtol=1e-5)


def test_sparse_chunked_scan_gradients_match_dense_prefix_mean():
    torch.manual_seed(33)
    bsz, seq_len, top_k, anchors, dim = 2, 7, 2, 6, 4
    routing_logits = torch.randn(bsz, seq_len, top_k, requires_grad=True)
    membership = routing_logits.softmax(-1)
    indices = torch.stack(
        (
            torch.arange(seq_len).view(1, -1).expand(bsz, -1) % anchors,
            (torch.arange(seq_len).view(1, -1).expand(bsz, -1) + 2) % anchors,
        ),
        dim=-1,
    )
    values = torch.randn(bsz, seq_len, dim, requires_grad=True)
    mask = torch.ones(bsz, seq_len)
    long_summary, _ = sparse_chunked_prefix_summaries(
        membership,
        indices,
        values,
        mask,
        anchors,
        seq_len,
        0.0,
        4,
    )
    sparse_context = torch.einsum("bnk,bnkd->bnd", membership, long_summary)
    sparse_grads = torch.autograd.grad(
        sparse_context.square().mean(),
        (routing_logits, values),
        retain_graph=True,
    )

    dense_context = dense_causal_prefix_aggregate(membership, indices, values, mask, anchors)
    dense_grads = torch.autograd.grad(dense_context.square().mean(), (routing_logits, values))
    assert torch.allclose(sparse_context, dense_context, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sparse_grads[0], dense_grads[0], atol=2e-5, rtol=2e-5)
    assert torch.allclose(sparse_grads[1], dense_grads[1], atol=2e-5, rtol=2e-5)


def test_sparse_chunked_scan_gradients_match_dense_across_slabs():
    torch.manual_seed(330)
    bsz, seq_len, top_k, anchors, dim = 1, 73, 2, 5, 2
    routing_logits = torch.randn(bsz, seq_len, top_k, requires_grad=True)
    membership = routing_logits.softmax(-1)
    positions = torch.arange(seq_len).view(1, -1)
    indices = torch.stack(
        (positions.remainder(anchors), (positions + 2).remainder(anchors)), dim=-1
    )
    values = torch.randn(bsz, seq_len, dim, requires_grad=True)
    mask = torch.ones(bsz, seq_len)
    long_summary, recent_summary = sparse_chunked_prefix_summaries(
        membership, indices, values, mask, anchors, 16_384, 1.5, 4
    )
    sparse = (
        torch.einsum("bnk,bnkd->bnd", membership, long_summary)
        + torch.einsum("bnk,bnkd->bnd", membership, recent_summary)
    )
    sparse_grads = torch.autograd.grad(
        sparse.square().mean(), (routing_logits, values), retain_graph=True
    )

    reference_long, reference_recent = sparse_segmented_prefix_summaries(
        membership, indices, values, mask, anchors, 16_384, 1.5
    )
    dense = (
        torch.einsum("bnk,bnkd->bnd", membership, reference_long)
        + torch.einsum("bnk,bnkd->bnd", membership, reference_recent)
    )
    dense_grads = torch.autograd.grad(
        dense.square().mean(), (routing_logits, values)
    )

    assert torch.allclose(sparse, dense, atol=1e-5, rtol=1e-5)
    assert torch.allclose(sparse_grads[0], dense_grads[0], atol=3e-5, rtol=3e-5)
    assert torch.allclose(sparse_grads[1], dense_grads[1], atol=3e-5, rtol=3e-5)


@torch.no_grad()
def test_v3_incremental_cache_matches_recompute():
    torch.manual_seed(37)
    model = build_model(v3_config()).eval()
    prefix = torch.tensor([[2, 3, 4, 5, 6]])
    mask = torch.ones_like(prefix)
    expected_prefix = model(prefix, attention_mask=mask)["logits"][:, -1:]
    cached_prefix, cache = model.prefill(prefix, mask)
    assert cache["backend"] == "fdt_v3_incremental"
    assert torch.allclose(expected_prefix, cached_prefix["logits"], atol=3e-5, rtol=3e-5)

    for token in (7, 8, 9, 10, 11):
        next_id = torch.tensor([[token]])
        decoded, cache = model.decode_step(next_id, cache)
        prefix = torch.cat((prefix, next_id), dim=1)
        expected = model(prefix, attention_mask=torch.ones_like(prefix))["logits"][:, -1:]
        assert torch.allclose(expected, decoded["logits"], atol=3e-5, rtol=3e-5)


@torch.no_grad()
def test_recency_reference_preserves_prefix_when_context_is_extended():
    torch.manual_seed(39)
    short_config = v3_config()
    short_model = build_model(short_config).eval()
    long_config = v3_config()
    long_config.max_seq_len = 32
    long_config.anchor_recency_reference_len = short_config.max_seq_len
    long_model = build_model(long_config).eval()

    state = dict(short_model.state_dict())
    positions = long_model.position_embedding.weight.detach().clone()
    positions[: short_config.max_seq_len].copy_(state["position_embedding.weight"])
    state["position_embedding.weight"] = positions
    long_model.load_state_dict(state, strict=True)

    ids = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]])
    mask = torch.ones_like(ids)
    short_logits = short_model(ids, attention_mask=mask)["logits"]
    long_logits = long_model(ids, attention_mask=mask)["logits"]
    assert torch.allclose(short_logits, long_logits, atol=3e-5, rtol=3e-5)


def test_v3_backward_is_finite():
    torch.manual_seed(41)
    model = build_model(v3_config())
    ids = torch.randint(2, 79, (2, 10))
    loss = model(ids, attention_mask=torch.ones_like(ids))["logits"].float().square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_dual_memory_loads_a_v3_parent_and_trains_copy_path():
    torch.manual_seed(43)
    parent = build_model(v3_config())
    config = v3_config()
    config.model_type = "fdt_v3_dual_memory"
    child = build_model(config)
    missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(name.startswith("exact_pointer.") for name in missing)

    ids = torch.tensor([[2, 3, 4, 5, 2, 3, 4, 6]])
    output = child(ids, attention_mask=torch.ones_like(ids))
    pointer = child.exact_memory_loss(output["hidden"], ids, ids, torch.ones_like(ids))
    total = output["logits"].float().square().mean() + 0.05 * pointer.loss
    total.backward()
    assert torch.isfinite(total)
    assert any(parameter.grad is not None for parameter in child.exact_pointer.parameters())


def test_anchor_indexed_dual_memory_preserves_parent_and_trains_exact_path():
    torch.manual_seed(47)
    parent = build_model(v3_config())
    config = v3_config()
    config.model_type = "fdt_v3_anchor_indexed_dual_memory"
    child = build_model(config)
    missing, unexpected = child.load_state_dict(parent.state_dict(), strict=False)
    assert not unexpected
    assert missing and all(name.startswith("exact_pointer.") for name in missing)

    ids = torch.tensor([[2, 3, 4, 5, 2, 3, 4, 6]])
    mask = torch.ones_like(ids)
    output = child(ids, attention_mask=mask)
    routes = child.exact_route_indices(output["hidden"])
    memory = child.build_exact_memory(output["hidden"], ids, mask)
    pointer = child.exact_memory_loss(output["hidden"], ids, ids, mask)

    assert routes.shape == (1, ids.size(1), config.top_k)
    assert memory.token_ids.tolist() == ids.int().tolist()
    assert torch.isfinite(pointer.loss)
    pointer.loss.backward()
    assert child.exact_pointer.anchor_match_bias.grad is not None
