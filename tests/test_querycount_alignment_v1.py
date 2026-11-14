import json

from research import querycount_alignment_v1 as module


def stored():
    output = module.ROOT / "research/artifacts/querycount_alignment_20261027_v1"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    train = {q: json.loads((output / f"train_q{q:02d}.json").read_text(encoding="utf-8")) for q in module.QUERY_COUNTS}
    dev = {q: json.loads((output / f"dev_q{q:02d}.json").read_text(encoding="utf-8")) for q in module.QUERY_COUNTS}
    return output, manifest, train, dev


def test_nested_views_have_expected_source_and_view_counts():
    output, manifest, train, dev = stored()
    assert manifest["summary"]["train_source_graphs"] == 1024
    assert manifest["summary"]["train_views"] == 4096
    assert manifest["summary"]["dev_source_graphs"] == 64
    assert manifest["summary"]["dev_views"] == 256
    assert all(len(train[q]) == 1024 for q in module.QUERY_COUNTS)
    assert all(len(dev[q]) == 64 for q in module.QUERY_COUNTS)
    assert [len(train[q][0]["queries"]) for q in module.QUERY_COUNTS] == [1, 4, 8, 16]
    assert [len(dev[q][0]["queries"]) for q in module.QUERY_COUNTS] == [1, 4, 8, 16]


def test_full_query_view_preserves_original_source_row_exactly_except_view_metadata():
    _, _, train, dev = stored()
    source_train, source_dev, _, _ = module.read_sources()
    metadata = {"source_graph_index", "source_split", "query_count", "query_view_of_original_16"}
    for source_rows, q16, split in ((source_train, train[16], "train"), (source_dev, dev[16], "dev")):
        for i, row in enumerate(q16):
            projected = {k: v for k, v in row.items() if k not in metadata}
            assert projected == source_rows[i]
            assert row["source_graph_index"] == i and row["source_split"] == split


def test_all_view_rows_revalidate_oracle_bfs_facts_prefixes_masks_and_transfer_exclusion():
    _, manifest, train, dev = stored()
    source_train, source_dev, excluded, _ = module.read_sources()
    summary = module.validate_views(source_train, source_dev, train, dev, excluded)
    assert summary["transfer_graphs_excluded_from_train"] == 16
    assert summary["train_distinct_base_queries"] == 16384
    assert summary["dev_distinct_base_queries"] == 1024
    assert summary["train_active_query_positions_across_prefix_views"] == 29696
    assert summary["dev_active_query_positions_across_prefix_views"] == 1856
    assert summary["train_view_repeated_premise_phrase_utf8_bytes"] == 4 * summary["train_physical_premise_phrase_utf8_bytes_once"]
    assert summary["dev_view_repeated_premise_phrase_utf8_bytes"] == 4 * summary["dev_physical_premise_phrase_utf8_bytes_once"]
    assert summary == manifest["summary"]
    assert manifest["construction"]["independence_claimed"] is False


def _batch_projection(batch):
    return [(row["source_graph_index"], row["query_count"], row["queries"]) for row in batch]


def test_sampler_is_deterministic_and_has_two_contexts_per_query_count():
    _, _, train, _ = stored()
    first = module.BalancedQueryCountBatchSampler(train, seed=25)
    second = module.BalancedQueryCountBatchSampler(train, seed=25)
    seen = []
    for _ in range(128):
        batch_a = first.next_batch()
        batch_b = second.next_batch()
        assert _batch_projection(batch_a) == _batch_projection(batch_b)
        assert len(batch_a) == 8
        assert len({row["source_graph_index"] for row in batch_a}) == 8
        assert {q: sum(row["query_count"] == q for row in batch_a) for q in module.QUERY_COUNTS} == {1: 2, 4: 2, 8: 2, 16: 2}
        seen.extend(row["source_graph_index"] for row in batch_a)
    assert len(seen) == 1024 and len(set(seen)) == 1024
    assert first.next_batch() is None


def test_sampler_serializable_state_restores_exact_next_batch():
    _, _, train, _ = stored()
    sampler = module.BalancedQueryCountBatchSampler(train, seed=25)
    sampler.next_batch()
    state = sampler.state_dict()
    serializable = json.loads(json.dumps(state))
    expected = sampler.next_batch()
    restored = module.BalancedQueryCountBatchSampler.from_state(train, serializable)
    actual = restored.next_batch()
    assert _batch_projection(actual) == _batch_projection(expected)


def test_sampler_state_rejects_identity_or_cursor_corruption():
    _, _, train, _ = stored()
    sampler = module.BalancedQueryCountBatchSampler(train, seed=25)
    state = sampler.state_dict()
    state["cursor"] = 1
    try:
        module.BalancedQueryCountBatchSampler.from_state(train, state)
    except ValueError:
        pass
    else:
        raise AssertionError("non-batch-aligned cursor must fail")


def test_builder_refuses_to_overwrite_existing_artifact():
    existing = module.ROOT / "research/artifacts/querycount_alignment_20261027_v1"
    try:
        module.write_candidate(existing)
    except FileExistsError:
        pass
    else:
        raise AssertionError("existing data artifact must be preserved")
