import pytest
import torch
import torch.nn.functional as F

from fdt_rlm.lexical_pointer import (
    AnchorIndexedExactMemory,
    ExactTokenMemory,
    LexicalPointerDecodeState,
    SparseLexicalPointer,
    causal_pointer_windows,
    sequential_prompt_copy_mask,
)


def test_exact_token_memory_is_lossless_compact_and_immutable_from_source():
    ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    memory = ExactTokenMemory.from_prompt(ids, source_length=3)
    ids[0, 1] = 999

    assert memory.token_ids.dtype == torch.int32
    assert memory.token_ids.tolist() == [[10, 20, 30]]
    assert memory.token_at(torch.tensor([[2, 0]])).tolist() == [[30, 10]]
    assert memory.storage_bytes() == 15


def test_exact_token_memory_rejects_invalid_positions():
    memory = ExactTokenMemory.from_prompt(torch.tensor([[1, 2, 3]]))
    try:
        memory.token_at(torch.tensor([[3]]))
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-range exact-memory access must fail")


def test_anchor_indexed_memory_selects_matching_chunks_and_keeps_raw_tokens():
    ids = torch.tensor([[10, 11, 12, 13, 20, 21, 22, 23]])
    routes = torch.tensor([[[1, 2], [1, 2], [1, 3], [1, 3], [7, 8], [7, 8], [7, 9], [7, 9]]])
    memory = AnchorIndexedExactMemory.from_prompt(
        ids,
        routes,
        chunk_size=4,
        commit_scores=torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]]),
    )
    positions, valid = memory.candidate_key_positions(torch.tensor([[7, 9]]), max_chunks=1)

    assert memory.token_ids.tolist() == ids.int().tolist()
    assert positions[0, 0].item() == 4
    assert valid[0, :3].all()
    assert memory.storage_bytes() < 256


def test_pointer_windows_are_strictly_causal_and_return_next_token_ids():
    hidden = torch.arange(5, dtype=torch.float32).view(1, 5, 1)
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    mask = torch.ones_like(ids)
    keys, candidates, valid = causal_pointer_windows(hidden, ids, mask, window=3)

    assert not valid[0, 0].any()
    assert keys[0, 3, 0, 0].item() == 2
    assert candidates[0, 3, 0].item() == 13
    assert keys[0, 3, 2, 0].item() == 0
    assert candidates[0, 3, 2].item() == 11


def test_pointer_loss_is_finite_with_copyable_and_novel_targets():
    pointer = SparseLexicalPointer(hidden_dim=8, pointer_dim=4, window=4)
    hidden = torch.randn(2, 6, 8)
    ids = torch.tensor([[1, 2, 3, 2, 3, 4], [1, 5, 6, 7, 8, 9]])
    labels = ids.clone()
    labels[:, :3] = -100
    mask = torch.ones_like(ids)

    result = pointer.training_loss(hidden, ids, labels, mask)

    assert torch.isfinite(result.loss)
    result.loss.backward()


def test_pointer_can_learn_causal_copy_spans_from_plain_lm_labels():
    pointer = SparseLexicalPointer(hidden_dim=8, pointer_dim=4, window=6)
    hidden = torch.randn(1, 8, 8)
    ids = torch.tensor([[10, 20, 30, 40, 10, 20, 30, 50]])
    result = pointer.training_loss(
        hidden,
        ids,
        ids,
        torch.ones_like(ids),
        min_copy_span=3,
        prompt_sources_only=False,
    )

    assert torch.isfinite(result.loss)
    assert result.copyable_rate > 0


def test_sequential_copy_mask_requires_prompt_span():
    ids = torch.tensor([[10, 20, 30, 40, 20, 30, 99]])
    labels = torch.tensor([[-100, -100, -100, -100, 20, 30, 99]])
    mask = torch.ones_like(ids)
    hidden = torch.randn(1, ids.size(1), 8)
    _, _, valid = causal_pointer_windows(hidden, ids, mask, window=6)
    prediction_length = ids.size(1) - 1
    positions = (
        torch.arange(prediction_length).view(1, prediction_length, 1)
        - torch.arange(6).view(1, 1, 6)
    )

    copied = sequential_prompt_copy_mask(
        ids, labels, mask, positions, valid, min_span=2
    )

    assert copied[0, 3].item()
    assert copied[0, 4].item()
    assert not copied[0, 5].item()


def test_fixed_prompt_source_excludes_generated_tokens_and_threshold_can_disable():
    pointer = SparseLexicalPointer(hidden_dim=8, pointer_dim=4, window=8)
    torch.nn.init.zeros_(pointer.gate_proj.weight)
    torch.nn.init.constant_(pointer.gate_proj.bias, -10.0)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    hidden = torch.randn(1, 5, 8)
    mask = torch.ones_like(ids)
    base = torch.randn(1, 16)

    mixed, diagnostics = pointer.mix_next_logits(
        base, hidden, ids, mask, source_length=3, min_gate=0.8
    )

    candidates = {value for row in diagnostics["candidate_ids"] for value in row}
    assert candidates <= {2, 3}
    assert diagnostics["mix_gate"] == 0.0
    assert torch.equal(mixed, base)


def test_decode_state_searches_the_full_fixed_source_without_expanding_training_window():
    pointer = SparseLexicalPointer(hidden_dim=2, pointer_dim=2, window=2)
    with torch.no_grad():
        pointer.q_proj.weight.copy_(torch.eye(2))
        pointer.k_proj.weight.copy_(torch.eye(2))
        pointer.gate_proj.weight.zero_()
        pointer.gate_proj.bias.fill_(10.0)
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    hidden = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0], [1.0, 0.0]]]
    )
    mask = torch.ones_like(ids)
    base = torch.zeros(1, 32)

    _, local = pointer.mix_next_logits(base, hidden, ids, mask, source_length=5)
    state = LexicalPointerDecodeState(source_length=5)
    _, full = state.prepare_logits(pointer, base, hidden, ids, mask, min_gate=0.0)

    assert local["search_span"] == 2
    assert local["candidate_ids"][0][0] != 11
    assert full["search_span"] == 4
    assert full["candidate_ids"][0][0] == 11
    assert full["source_positions"][0][0] == 0


def test_decode_cursor_without_span_metadata_ignores_heuristic_boundary():
    state = LexicalPointerDecodeState(source_length=5, max_activation_steps=2)
    source = torch.tensor([[10, 20, 30, 40, 50]])
    diagnostics = {
        "mode": "hard_copy",
        "hard_copy_eligible": True,
        "cursor_commit_eligible": True,
        "mix_gate": 0.95,
        "source_positions": [[1]],
        "candidate_ids": [[30]],
    }
    state.commit(30, diagnostics)

    base = torch.randn(1, 64)
    logits, cursor_diagnostics = state.prepare_logits(
        None, base, torch.empty(1, 1, 1), source, torch.ones_like(source)
    )
    assert logits.argmax(-1).item() == 40
    assert cursor_diagnostics["mode"] == "cursor"

    state.commit(40, cursor_diagnostics, boundary=True)
    assert not state.completed

    logits, cursor_diagnostics = state.prepare_logits(
        None, base, torch.empty(1, 1, 1), source, torch.ones_like(source)
    )
    assert logits.argmax(-1).item() == 50
    state.commit(50, cursor_diagnostics)
    assert state.completed
    assert state.source_token_position is None


def test_decode_cursor_reads_the_frozen_exact_memory():
    state = LexicalPointerDecodeState(source_length=4, max_activation_steps=2)
    source = torch.tensor([[10, 20, 30, 40]])
    base = torch.randn(1, 64)

    state.exact_memory = ExactTokenMemory.from_prompt(source)
    state.source_token_position = 1
    source[0, 2] = 55
    logits, diagnostics = state.prepare_logits(
        None, base, torch.empty(1, 1, 1), source, torch.ones_like(source)
    )

    assert diagnostics["mode"] == "cursor"
    assert logits.argmax(-1).item() == 30


def test_decode_cursor_losslessly_emits_oracle_selected_spans_at_audit_lengths():
    for span_length in (4, 8, 16, 32, 64):
        source = torch.arange(span_length + 3, dtype=torch.long).view(1, -1) + 17
        expected = source[0, 1 : span_length + 1].tolist()
        state = LexicalPointerDecodeState(
            source_length=source.size(1),
            max_activation_steps=span_length,
            max_copy_tokens=span_length,
        )
        first = {
            "mode": "hard_copy",
            "hard_copy_eligible": True,
            "cursor_commit_eligible": True,
            "mix_gate": 1.0,
            "source_positions": [[0]],
            "candidate_ids": [[expected[0]]],
        }
        state.commit(expected[0], first)
        emitted = [expected[0]]
        base = torch.zeros(1, int(source.max()) + 2)
        for expected_id in expected[1:]:
            logits, diagnostics = state.prepare_logits(
                None,
                base,
                torch.empty(1, 1, 1),
                source,
                torch.ones_like(source),
            )
            assert diagnostics["mode"] == "cursor"
            selected = int(logits.argmax(dim=-1))
            assert selected == expected_id
            emitted.append(selected)
            state.commit(selected, diagnostics)
        assert emitted == expected
        assert state.completed


def test_cursor_uses_explicit_span_end_and_returns_to_generation():
    source = torch.tensor([[10, 20, 30, 40, 50]])
    state = LexicalPointerDecodeState(source_length=5, max_copy_tokens=16)
    first = {
        "mode": "hard_copy",
        "hard_copy_eligible": True,
        "cursor_commit_eligible": True,
        "mix_gate": 1.0,
        "source_positions": [[0]],
        "candidate_ids": [[20]],
        "span_end_positions": [[2]],
    }
    state.commit(20, first)
    logits, diagnostics = state.prepare_logits(
        None,
        torch.zeros(1, 64),
        torch.empty(1, 1, 1),
        source,
        torch.ones_like(source),
    )
    assert int(logits.argmax(dim=-1)) == 30
    state.commit(30, diagnostics)
    assert state.completed
    assert state.source_token_position is None
    assert state.source_span_end_position is None


def test_anchor_memory_preserves_explicit_span_metadata():
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    routes = torch.tensor([[[0], [0], [1], [1], [2]]])
    span_ends = torch.tensor([[2, 2, 2, 4, 4]])
    memory = AnchorIndexedExactMemory.from_prompt(
        ids,
        routes,
        chunk_size=2,
        chunk_anchor_count=1,
        span_end_positions=span_ends,
    )
    assert torch.equal(memory.span_end_positions.long(), span_ends)


def test_anchor_memory_infers_span_end_from_trained_commit_scores():
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    routes = torch.tensor([[[0], [0], [1], [1], [2]]])
    commits = torch.tensor([[0.9, 0.8, 0.1, 0.9, 0.0]])
    memory = AnchorIndexedExactMemory.from_prompt(
        ids,
        routes,
        chunk_size=2,
        chunk_anchor_count=1,
        commit_scores=commits,
    )

    ends = memory.inferred_span_end_positions(torch.tensor([[0, 2, 3]]))
    assert ends.tolist() == [[3, 3, 4]]


def test_pointer_projections_follow_parameter_dtype_and_return_fp32_scores():
    pointer = SparseLexicalPointer(hidden_dim=8, pointer_dim=4, window=6).to(
        dtype=torch.bfloat16
    )
    hidden = torch.randn(1, 5, 8, dtype=torch.bfloat16)

    assert pointer.project_queries(hidden).dtype == torch.float32
    assert pointer.project_keys(hidden).dtype == torch.float32
    assert pointer.project_gate(hidden).dtype == torch.float32
    assert pointer.commit_scores(hidden).dtype == torch.float32


def test_global_prompt_exact_loss_gives_all_selection_heads_finite_nonzero_gradients():
    torch.manual_seed(41)
    pointer = SparseLexicalPointer(hidden_dim=12, pointer_dim=6, window=8)
    prompt = torch.tensor([41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89])
    copied = prompt[3:10]
    ids = torch.cat((prompt, copied)).view(1, -1)
    labels = torch.full_like(ids, -100)
    labels[:, prompt.numel() :] = ids[:, prompt.numel() :]
    hidden = torch.randn(1, ids.size(1), 12)
    mask = torch.ones_like(ids)

    result = pointer.training_loss(
        hidden,
        ids,
        labels,
        mask,
        mode="global_prompt",
        source_chunk_size=4,
        query_chunk_size=3,
        max_global_queries=32,
        proposal_chunk_size=4,
        max_candidate_chunks=2,
    )

    assert result.contract_valid.item() == 1.0
    assert torch.isfinite(result.loss)
    result.loss.backward()
    for parameter in (
        pointer.q_proj.weight,
        pointer.k_proj.weight,
        pointer.gate_proj.weight,
        pointer.commit_proj.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0.0


def test_explicit_copy_contract_rejects_invalid_source_mapping():
    pointer = SparseLexicalPointer(hidden_dim=8, pointer_dim=4, window=6)
    ids = torch.tensor([[1, 10, 20, 10]])
    labels = torch.tensor([[-100, -100, -100, 10]])
    hidden = torch.randn(1, 4, 8)
    source_positions = torch.tensor([[-1, -1, -1, 2]])
    target_mask = torch.tensor([[0, 0, 0, 1]])

    with pytest.raises(ValueError, match="does not equal"):
        pointer.training_loss(
            hidden,
            ids,
            labels,
            torch.ones_like(ids),
            copy_source_positions=source_positions,
            copy_target_mask=target_mask,
            source_boundary=torch.tensor([3]),
        )


def test_explicit_copy_contract_rejects_target_inside_prompt_boundary():
    pointer = SparseLexicalPointer(hidden_dim=8, pointer_dim=4, window=6)
    ids = torch.tensor([[1, 10, 10, 20]])
    labels = torch.tensor([[-100, -100, 10, -100]])
    hidden = torch.randn(1, 4, 8)
    source_positions = torch.tensor([[-1, -1, 1, -1]])
    target_mask = torch.tensor([[0, 0, 1, 0]])

    with pytest.raises(ValueError, match="earlier prompt"):
        pointer.training_loss(
            hidden,
            ids,
            labels,
            torch.ones_like(ids),
            copy_source_positions=source_positions,
            copy_target_mask=target_mask,
            source_boundary=torch.tensor([3]),
        )


def test_explicit_copy_contract_rejects_inactive_target():
    pointer = SparseLexicalPointer(hidden_dim=8, pointer_dim=4, window=6)
    ids = torch.tensor([[1, 10, 10]])
    labels = torch.tensor([[-100, -100, 10]])
    source_positions = torch.tensor([[-1, -1, 1]])
    target_mask = torch.tensor([[0, 0, 1]])
    attention = torch.tensor([[1, 1, 0]])

    with pytest.raises(ValueError, match="active tokens"):
        pointer.training_loss(
            torch.randn(1, 3, 8),
            ids,
            labels,
            attention,
            copy_source_positions=source_positions,
            copy_target_mask=target_mask,
            source_boundary=torch.tensor([2]),
        )


def test_explicit_copy_commit_targets_reduce_duplicate_source_keys():
    pointer = SparseLexicalPointer(hidden_dim=8, pointer_dim=4, window=8)
    ids = torch.tensor([[1, 10, 20, 10, 10]])
    labels = torch.tensor([[-100, -100, -100, 10, 10]])
    hidden = torch.randn(1, 5, 8)
    source_positions = torch.tensor([[-1, -1, -1, 1, 1]])
    target_mask = torch.tensor([[0, 0, 0, 1, 1]])

    result = pointer.training_loss(
        hidden,
        ids,
        labels,
        torch.ones_like(ids),
        copy_source_positions=source_positions,
        copy_target_mask=target_mask,
        source_boundary=torch.tensor([3]),
    )

    assert torch.isfinite(result.loss)
    assert torch.isfinite(result.commit_loss)


def test_explicit_copy_proposal_recall_measures_routed_source_chunk() -> None:
    pointer = SparseLexicalPointer(hidden_dim=8, pointer_dim=4, window=8)
    ids = torch.tensor([[1, 10, 20, 30, 40, 50, 60, 70, 10, 20]])
    labels = torch.tensor([[-100, -100, -100, -100, -100, -100, -100, -100, 10, 20]])
    source_positions = torch.tensor([[-1, -1, -1, -1, -1, -1, -1, -1, 1, 2]])
    target_mask = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1]])
    routes = torch.zeros((1, ids.size(1), 2), dtype=torch.long)
    routes[:, 0:2] = torch.tensor([3, 13])
    routes[:, 2:4] = torch.tensor([7, 17])
    routes[:, 4:6] = torch.tensor([9, 19])
    routes[:, 6:8] = torch.tensor([11, 21])
    routes[:, 7:9] = torch.tensor([3, 13])

    hidden = torch.randn(1, ids.size(1), 8)

    def measured_and_runtime_recall(current_routes: torch.Tensor) -> tuple[float, float]:
        result = pointer.training_loss(
            hidden,
            ids,
            labels,
            torch.ones_like(ids),
            route_indices=current_routes,
            proposal_chunk_size=2,
            proposal_chunk_anchors=2,
            max_candidate_chunks=1,
            copy_source_positions=source_positions,
            copy_target_mask=target_mask,
            source_boundary=torch.tensor([8]),
        )
        memory = AnchorIndexedExactMemory.from_prompt(
            ids[:, :8],
            current_routes[:, :8],
            chunk_size=2,
            chunk_anchor_count=2,
            commit_scores=pointer.commit_scores(hidden[:, :8]).detach(),
        )
        hits = []
        for target_position, mapped_token in ((8, 1), (9, 2)):
            candidates, valid = memory.candidate_key_positions(
                current_routes[:, target_position - 1], max_chunks=1
            )
            mapped_key = mapped_token - 1
            hits.append(bool(((candidates == mapped_key) & valid).any()))
        return float(result.proposal_recall), sum(hits) / len(hits)

    hit, hit_runtime = measured_and_runtime_recall(routes)
    routes[:, 7:9] = torch.tensor([7, 17])
    miss, miss_runtime = measured_and_runtime_recall(routes)

    assert hit == hit_runtime
    assert miss == miss_runtime


def test_explicit_copy_proposal_recall_can_be_skipped_off_log_steps() -> None:
    pointer = SparseLexicalPointer(hidden_dim=8, pointer_dim=4, window=8)
    ids = torch.tensor([[1, 10, 20, 30, 10]])
    labels = torch.tensor([[-100, -100, -100, -100, 10]])
    source_positions = torch.tensor([[-1, -1, -1, -1, 1]])
    target_mask = torch.tensor([[0, 0, 0, 0, 1]])
    routes = torch.arange(ids.size(1)).view(1, -1, 1)
    result = pointer.training_loss(
        torch.randn(1, ids.size(1), 8),
        ids,
        labels,
        torch.ones_like(ids),
        route_indices=routes,
        proposal_chunk_size=2,
        proposal_chunk_anchors=2,
        max_candidate_chunks=1,
        measure_proposal_recall=False,
        copy_source_positions=source_positions,
        copy_target_mask=target_mask,
        source_boundary=torch.tensor([4]),
    )
    assert result.proposal_recall.item() == 0.0


def test_16k_exact_store_preserves_tokens_and_returns_bounded_chunk_candidates():
    length = 16_384
    ids = ((torch.arange(length, dtype=torch.long) * 37) % 1009 + 3).view(1, -1)
    routes = torch.arange(length, dtype=torch.long).view(1, length, 1).remainder(256)
    routes = torch.cat(tuple((routes + offset).remainder(256) for offset in range(8)), dim=-1)
    memory = AnchorIndexedExactMemory.from_prompt(
        ids,
        routes,
        chunk_size=32,
        chunk_anchor_count=4,
        commit_scores=torch.zeros_like(ids, dtype=torch.float32),
    )
    positions, valid = memory.candidate_key_positions(
        torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]]),
        max_chunks=4,
    )

    assert memory.source_length == length
    assert torch.equal(memory.token_ids.long(), ids)
    assert memory.key_vectors is None
    assert positions.shape == (1, 4 * 32)
    assert positions.numel() < length - 1
    assert bool((positions >= 0).all())
    assert bool((positions < length - 1).all())
    assert bool(valid.any())


def test_decode_state_respects_disabled_full_scan_fallback():
    pointer = SparseLexicalPointer(hidden_dim=4, pointer_dim=2, window=4)
    with torch.no_grad():
        pointer.gate_proj.weight.zero_()
        pointer.gate_proj.bias.fill_(10.0)
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    routes = torch.tensor([[[0, 1], [0, 1], [2, 3], [2, 3], [4, 5]]])
    hidden = torch.randn(1, 5, 4)
    keys = F.normalize(pointer.k_proj(hidden.float()), dim=-1).half()
    memory = AnchorIndexedExactMemory.from_prompt(
        ids,
        routes,
        chunk_size=2,
        chunk_anchor_count=2,
        key_vectors=keys,
    )
    state = LexicalPointerDecodeState(source_length=5)
    _, diagnostics = state.prepare_logits(
        pointer,
        torch.zeros(1, 32),
        hidden,
        ids,
        torch.ones_like(ids),
        min_gate=0.0,
        anchor_memory=memory,
        query_anchor_ids=torch.tensor([[0, 1]]),
        max_candidate_chunks=1,
        full_scan_fallback=False,
    )

    assert diagnostics["full_scan_attempted"] is False
    assert diagnostics["full_scan_count"] == 0


def test_closed_need_gate_does_not_consume_the_one_full_scan_budget():
    pointer = SparseLexicalPointer(hidden_dim=4, pointer_dim=2, window=4)
    with torch.no_grad():
        pointer.gate_proj.weight.zero_()
        pointer.gate_proj.bias.fill_(-10.0)
    ids = torch.tensor([[10, 11, 12, 13, 14]])
    routes = torch.tensor([[[0, 1], [0, 1], [2, 3], [2, 3], [4, 5]]])
    hidden = torch.randn(1, 5, 4)
    keys = F.normalize(pointer.k_proj(hidden.float()), dim=-1).half()
    memory = AnchorIndexedExactMemory.from_prompt(
        ids,
        routes,
        chunk_size=2,
        chunk_anchor_count=2,
        key_vectors=keys,
    )
    state = LexicalPointerDecodeState(source_length=5)
    base = torch.zeros(1, 32)
    _, diagnostics = state.prepare_logits(
        pointer,
        base,
        hidden,
        ids,
        torch.ones_like(ids),
        min_gate=0.8,
        anchor_memory=memory,
        query_anchor_ids=torch.tensor([[0, 1]]),
        max_candidate_chunks=1,
        full_scan_fallback=True,
    )

    assert diagnostics["skipped_by_exact_need_gate"] is True
    assert diagnostics["full_scan_attempted"] is False
    assert diagnostics["full_scan_count"] == 0


def test_decode_cursor_rejects_a_repeated_activation_token():
    state = LexicalPointerDecodeState(source_length=6, max_activation_steps=1)
    diagnostics = {
        "mode": "hard_copy",
        "hard_copy_eligible": True,
        "cursor_commit_eligible": True,
        "mix_gate": 0.95,
        "source_positions": [[1]],
        "candidate_ids": [[30]],
        "candidate_occurrences": [[3]],
    }

    state.commit(30, diagnostics)

    assert state.source_token_position is None
    assert state.generated_tokens == 1


def test_decode_state_does_not_activate_after_budget():
    state = LexicalPointerDecodeState(source_length=3, max_activation_steps=1)
    state.generated_tokens = 1
    base = torch.randn(1, 16)
    logits, diagnostics = state.prepare_logits(
        None,
        base,
        torch.empty(1, 1, 1),
        torch.tensor([[1, 2, 3]]),
        torch.ones(1, 3, dtype=torch.long),
    )
    assert diagnostics["mode"] == "base"
    assert torch.equal(logits, base)
