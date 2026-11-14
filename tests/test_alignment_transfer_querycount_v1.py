import json
from pathlib import Path

from research import alignment_transfer_querycount_v1 as candidate


def built():
    output = candidate.ROOT / "research/artifacts/alignment_transfer_querycount_20261027_v1"
    assert output.is_dir()
    report = candidate.validate_candidate(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    selection = manifest["selection"]
    return output, {"selection": selection, "validation": report,
        "manifest_payload_sha256_without_self": manifest["manifest_payload_sha256_without_self"],
        "manifest_file_sha256": candidate.sha256_file(output / "manifest.json")}


def test_fixed_bounded_selection_and_named_novelty():
    output, result = built()
    summary = result["selection"]
    assert summary["candidate_scan_count"] <= 50_000
    assert summary["accepted_per_original_target_stratum"] == [4, 4, 4, 4]
    assert summary["accepted_graphs"] == 16
    registry = json.loads((output / "selection_registry.json").read_text(encoding="utf-8"))
    keys = [row["canonical_graph_key"] for row in registry["graphs"]]
    assert len(set(keys)) == 16
    excluded, evidence = candidate._load_querymix_exclusions()
    assert not (set(keys) & excluded)
    assert evidence["query_mix"]["distinct_abstract_graphs"] == 1088
    assert evidence["original_v2_observed_exclusions"]["distinct_abstract_graphs"] == 112
    assert "not an all-history guarantee" in evidence["claim_scope"]


def test_all_query_count_views_are_nested_prefixes_over_same_facts():
    output, _ = built()
    views = {q: json.loads((output / f"q{q:02d}.json").read_text(encoding="utf-8"))
             for q in candidate.QUERY_COUNTS}
    assert all(len(rows) == 16 for rows in views.values())
    by_q = {q: {row["canonical_graph_key"]: row for row in rows} for q, rows in views.items()}
    assert all(set(by_q[q]) == set(by_q[16]) for q in candidate.QUERY_COUNTS)
    for key, full in by_q[16].items():
        assert len(full["queries"]) == 16 == len(set(map(tuple, full["queries"])))
        for q in candidate.QUERY_COUNTS:
            row = by_q[q][key]
            assert row["facts"] == full["facts"]
            assert row["queries"] == full["queries"][:q]
            assert row["labels"] == full["labels"][:q]
            assert row["hops"] == full["hops"][:q]


def test_every_stored_row_matches_original_oracle_and_independent_bfs():
    output, _ = built()
    for q in candidate.QUERY_COUNTS:
        for row in json.loads((output / f"q{q:02d}.json").read_text(encoding="utf-8")):
            original = candidate.oracle(8, row["facts"], row["queries"])
            independent = candidate.v2.independent_oracle(8, row["facts"], row["queries"])
            assert original == independent == (row["labels"], row["hops"])
            assert candidate._active_ids(row["facts"]) == set(range(8))
            assert len(row["facts"]) <= candidate.Config().facts
            assert max(len(f[-1].encode("utf-8")) for f in row["facts"]) <= 48


def test_five_feature_input_contract_and_metadata_is_outside_inputs():
    output, _ = built()
    rows = json.loads((output / "q16.json").read_text(encoding="utf-8"))
    for row in rows:
        features, targets, _ = candidate.collate([row], candidate.Config(), "cpu")
        assert set(features) == candidate.FEATURE_KEYS
        assert int(features["query_mask"].sum()) == 16
        assert targets["labels"].shape == (1, 16)
        assert "original_sampling_target_class_stratum" in row
        assert "original_sampling_target_class_stratum" not in features
        assert "labels" not in features and "hops" not in features


def test_manifest_payload_file_and_data_hashes_are_distinct_and_valid():
    output, result = built()
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared = manifest["manifest_payload_sha256_without_self"]
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256_without_self")
    assert candidate.sha256_bytes(candidate.canonical_json_bytes(payload)) == declared
    assert result["manifest_payload_sha256_without_self"] == declared
    assert result["manifest_file_sha256"] == candidate.sha256_file(manifest_path)
    for name, info in manifest["files"].items():
        assert candidate.sha256_file(output / name) == info["sha256"]
    verified = candidate.validate_candidate(output)
    assert verified["all_64_rows_original_and_bfs_verified"]
    assert verified["all_views_prefix_and_same_facts_verified"]
    assert verified["all_features_exactly_five_and_gold_free"]
    assert verified["novelty_scope_limited_to_named_sets"]
