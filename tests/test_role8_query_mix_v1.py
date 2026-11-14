from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import role8_query_mix_v1 as builder

OUT = ROOT / "research/artifacts/role8_query_mix_20261027_v1"
V2 = ROOT / "research/artifacts/role8_active_balanced_20261027_v2"


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def test_manifest_payload_and_actual_manifest_file_hashes_are_distinct():
    path = OUT / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    payload_hash = manifest.pop("manifest_payload_sha256_without_self")
    assert hashlib.sha256(builder.canonical_json_bytes(manifest)).hexdigest() == payload_hash
    actual_file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(actual_file_hash) == 64 and actual_file_hash != payload_hash
    assert manifest["schema"] == builder.SCHEMA
    assert manifest["status"] == builder.STATUS
    assert manifest["construction"]["new_abstract_graphs_added"] == 0
    assert manifest["inputs"]["baseline_stored_hops"] is False
    assert manifest["inputs"]["baseline_hops_reconstructed_by_original_and_bfs"] is True


def test_rows_preserve_every_v2_fact_graph_family_and_split_identity():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for split, count in (("train", 1024), ("dev", 64)):
        before = json.loads((V2 / f"{split}.json").read_text(encoding="utf-8"))
        after = load(f"{split}.json")
        assert len(after) == count
        assert [[r["graph_key"], r["canonical_graph_key"], r["facts"]] for r in before] == \
               [[r["graph_key"], r["canonical_graph_key"], r["facts"]] for r in after]
        assert all("target_class" not in row for row in after)
        assert all(0 <= row["original_sampling_target_class"] <= 3 for row in after)
        assert all(row["source_family_id"] == row["canonical_graph_key"] for row in after)
        assert len({row["canonical_graph_key"] for row in after}) == count
        assert manifest["summary"][split]["facts_projection_sha256"] == \
               manifest["checks"][f"{split}_facts_projection_sha256_v2_equivalent"]
    assert not ({r["canonical_graph_key"] for r in load("train.json")} &
                {r["canonical_graph_key"] for r in load("dev.json")})


def test_fixed_uniform_queries_match_v3_seed17_baseline_raw_and_both_oracles():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    train = json.loads((V2 / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((V2 / "dev.json").read_text(encoding="utf-8"))
    baseline = json.loads(builder.V3_BASELINE.read_text(encoding="utf-8"))
    actual_train = load("train.json")
    actual_dev = load("dev.json")
    expected_train = builder.sample_split(train, seed=builder.TRAIN_SEED, split="train")
    expected_dev = builder.sample_split(dev, seed=builder.DEV_SEED, split="dev", baseline_raw=baseline["raw"])
    assert [r["queries"] for r in actual_train] == [r["queries"] for r in expected_train]
    assert [r["queries"] for r in actual_dev] == [r["queries"] for r in expected_dev]
    assert len(baseline["raw"]) == 64 and baseline["noise_seeds"] == [17]
    by_key = {r["graph_key"]: r for r in baseline["raw"]}
    for row in actual_dev:
        raw = by_key[row["graph_key"]]
        assert row["facts"] == raw["facts"]
        assert row["queries"] == raw["queries"]
        assert row["labels"] == raw["gold"]
        assert (row["labels"], row["hops"]) == builder.oracle(8, row["facts"], row["queries"])
        assert (row["labels"], row["hops"]) == builder.v2.independent_oracle(8, row["facts"], row["queries"])
    assert manifest["inputs"]["v3_baseline_facts_queries_gold_exact"] is True


def test_all_1088_rows_meet_query_fact_phrase_mask_and_oracle_contracts():
    for split, rows in (("train", load("train.json")), ("dev", load("dev.json"))):
        summary = builder.validate_rows(rows, split=split)
        assert summary["graphs"] == len(rows)
        assert all(summary["fact_entity_exposure"])
        assert all(summary["query_entity_exposure"])
        assert summary["max_phrase_bytes"] <= 48
        assert len(summary["query_label_counts"]) == 4
        assert all(len(row["facts"]) <= 12 and len(row["queries"]) == 16 for row in rows)
        assert all(len(set(map(tuple, row["queries"]))) == 16 for row in rows)
        assert all("target_class" not in row for row in rows)


def test_v2_sources_are_pinned_and_unchanged_after_build():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for name in ("train.json", "dev.json", "manifest.json"):
        path = V2 / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == builder.V2_PINNED[name]
        assert manifest["inputs"]["v2_files"][name] == builder.V2_PINNED[name]
    assert hashlib.sha256((ROOT / "research/role8_active_balanced_v2.py").read_bytes()).hexdigest() == \
           builder.V2_PINNED["builder.py"]


def test_unbalanced_sample_is_not_label_reassigned_or_claimed_independent():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    rows = load("train.json") + load("dev.json")
    assert any(len(set(row["labels"])) > 1 for row in rows)
    assert all("original_sampling_target_class" in row for row in rows)
    assert manifest["construction"]["random_query_label_assignment"] is False
    assert manifest["construction"]["independence_claimed"] is False
    assert manifest["e2_final_files_accessed"] is False
    assert manifest["model_or_checkpoint_loaded"] is False
    assert manifest["optimizer_updates"] == 0
