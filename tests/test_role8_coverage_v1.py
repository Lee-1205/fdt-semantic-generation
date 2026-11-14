import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))

from fdt_local.data import collate, make_split
from fdt_local.data import oracle
from fdt_local.model_masked import Config
import role8_coverage_v1 as role8


@pytest.fixture(scope="module")
def source_row():
    return make_split("train", 1, seed=301)[0]


def test_canonical_key_is_invariant_to_entity_permutation(source_row):
    ids = list(reversed(range(source_row["n"])))
    permutation = {old: ids[old] for old in range(source_row["n"])}
    facts = [(permutation[s], permutation[o], kind, phrase)
             for s, o, kind, phrase in source_row["facts"]]
    assert role8.canonical_graph_key(source_row["n"], facts) == role8.canonical_graph_key(
        source_row["n"], source_row["facts"])
    with pytest.raises(ValueError, match="n in 4..6"):
        role8.canonical_graph_key(8, facts)


def test_eight_cyclic_rows_roundtrip_and_preserve_oracle(source_row):
    rendered = role8.eight_rotations(source_row)
    assert len(rendered) == 8
    assert len({row["graph_key"] for row in rendered}) == 8
    for shift, row in enumerate(rendered):
        assert row["n"] == 8 and row["source_n"] == source_row["n"]
        assert row["rotation"] == shift
        assert row["labels"] == source_row["labels"]
        assert row["hops"] == source_row["hops"]
        for old, new in zip(source_row["facts"], row["facts"]):
            s, o, kind, phrase = new
            assert ((s - shift) % 8, (o - shift) % 8, kind, phrase) == old
            assert 0 <= s < 8 and 0 <= o < 8
        for old, new in zip(source_row["queries"], row["queries"]):
            s, o = new
            assert ((s - shift) % 8, (o - shift) % 8) == old


def test_eight_rotations_balance_fact_and_query_entity_role_ids(source_row):
    summary = role8.summarize_rows(role8.eight_rotations(source_row))
    for key in ("query_source_entity_id_counts", "query_object_entity_id_counts",
                "fact_source_entity_id_counts", "fact_object_entity_id_counts"):
        values = summary[key]
        assert len(values) == 8 and len(set(values)) == 1 and values[0] > 0
    assert summary["rendered_rows"] == summary["physical_graph_keys"] == 8


def test_role_rows_keep_supervision_outside_the_five_neural_features(source_row):
    row = role8.cyclic_renaming(source_row, 3)
    features, targets, _ = collate([row], Config(), "cpu")
    assert set(features) == {"tokens", "roles", "fact_mask", "queries", "query_mask"}
    assert set(targets) == {"labels", "ground", "hops"}
    assert targets["labels"][0, :len(row["labels"])].tolist() == row["labels"]
    assert targets["hops"][0, :len(row["hops"])].tolist() == row["hops"]


def test_selection_rejects_canonical_overlap_and_keeps_rotations_together(source_row):
    other = make_split("train", 2, seed=9001)[1]
    forbidden = {role8.canonical_graph_key(source_row["n"], source_row["facts"])}
    groups, stats = role8.choose_train_bases([source_row, copy.deepcopy(source_row), other],
        forbidden, count=1, max_candidates=3)
    assert len(groups) == 1 and stats["accepted_bases"] == 1
    assert stats["rejected"]["overlaps_olddev_or_test_control"] == 2
    source, iso, rows = groups[0]
    assert len(rows) == 8 and len({row["canonical_source_key"] for row in rows}) == 1
    assert iso not in forbidden
    assert {row["source_graph_key"] for row in rows} == {source["graph_key"]}


def test_generated_candidate_manifest_and_all_rows_match_contract():
    artifact = ROOT / "research" / "artifacts" / "role8_coverage_20261026_v1"
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    train = json.loads((artifact / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((artifact / "dev.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "CPU_ROLE8_COVERAGE_CANDIDATE_NOT_TRAINED"
    assert manifest["lineage_and_limits"]["e2_final_data_accessed"] is False
    assert manifest["lineage_and_limits"]["historical_isomorphic_leakage_to_e1_model"].startswith("unknown")
    assert all(manifest["canonical_partition_checks"][key] for key in (
        "train_source_vs_olddev96", "train_source_vs_newtest16", "train_source_vs_held_control16",
        "train_source_vs_dev_source", "dev_source_is_subset_of_olddev96",
        "dev_source_vs_newtest16", "dev_source_vs_held_control16"))
    assert len(train) == 512 and len(dev) == 128
    assert len({row["graph_key"] for row in train + dev}) == 640
    for rows, expected_bases in ((train, 64), (dev, 16)):
        assert {row["n"] for row in rows} == {8}
        assert {row["source_index"] for row in rows} == set(range(expected_bases))
        for index in range(expected_bases):
            group = [row for row in rows if row["source_index"] == index]
            assert sorted(row["rotation"] for row in group) == list(range(8))
            assert len({row["canonical_source_key"] for row in group}) == 1
        assert all(oracle(row["n"], row["facts"], row["queries"]) ==
                   (row["labels"], row["hops"]) for row in rows)
    for split, rows in (("train", train), ("dev", dev)):
        summary = manifest[split]
        assert summary["rendered_rows"] == len(rows)
        for key in ("query_source_entity_id_counts", "query_object_entity_id_counts",
                    "fact_source_entity_id_counts", "fact_object_entity_id_counts"):
            assert len(set(summary[key])) == 1
