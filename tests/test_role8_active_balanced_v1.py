from __future__ import annotations

import itertools
import hashlib
import json
import hashlib
import json
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "src"))
import role8_active_balanced_v1 as builder
from fdt_local.data import make_split, oracle


def relabel(facts, mapping):
    return [(mapping[s], mapping[o], kind, phrase) for s, o, kind, phrase in facts]


def test_canonical_key_is_invariant_under_random_eight_id_renamings():
    facts = [(0, 1, 0, "a"), (1, 2, 0, "a"), (2, 3, 1, "b"),
             (3, 4, 2, "c"), (4, 5, 0, "a"), (5, 6, 1, "b"), (6, 7, 3, "d")]
    key = builder.canonical_typed_graph(8, facts)
    rng = random.Random(814)
    for _ in range(12):
        perm = list(range(8)); rng.shuffle(perm)
        assert builder.canonical_typed_graph(8, relabel(facts, perm)) == key


def test_refinement_canonicalizer_matches_bruteforce_isomorphism_on_small_fixtures():
    fixtures = [
        [(0, 1, 0, "x"), (1, 2, 1, "y"), (2, 3, 2, "z")],
        [(0, 2, 0, "x"), (2, 1, 1, "y"), (1, 3, 2, "z")],  # relabelled first fixture
        [(0, 1, 0, "x"), (1, 2, 0, "x"), (2, 3, 0, "x")],
        [(0, 1, 0, "x"), (1, 2, 0, "x"), (2, 0, 0, "x")],
    ]
    for left, right in itertools.combinations(fixtures, 2):
        brute = builder.brute_force_isomorphic(4, left, right)
        canon_equal = builder.canonical_typed_graph(4, left) == builder.canonical_typed_graph(4, right)
        assert canon_equal == brute
    eight_cycle = [(i, (i + 1) % 8, i % 4, "x") for i in range(8)]
    perm = [3, 6, 1, 7, 0, 5, 2, 4]
    relabelled_cycle = relabel(eight_cycle, perm)
    eight_path = [(i, i + 1, i % 4, "x") for i in range(7)]
    for left, right in ((eight_cycle, relabelled_cycle),
                        (eight_cycle, eight_path),
                        (relabelled_cycle, eight_path)):
        assert (builder.canonical_typed_graph(8, left) == builder.canonical_typed_graph(8, right)) == \
               builder.brute_force_isomorphic(8, left, right)
    eight_cycle = [(i, (i + 1) % 8, i % 4, "x") for i in range(8)]
    perm = [3, 6, 1, 7, 0, 5, 2, 4]
    relabelled_cycle = relabel(eight_cycle, perm)
    eight_path = [(i, i + 1, i % 4, "x") for i in range(7)]
    for left, right in ((eight_cycle, relabelled_cycle),
                        (eight_cycle, eight_path),
                        (relabelled_cycle, eight_path)):
        assert (builder.canonical_typed_graph(8, left) == builder.canonical_typed_graph(8, right)) == \
               builder.brute_force_isomorphic(8, left, right)
    eight_cycle = [(i, (i + 1) % 8, i % 4, "x") for i in range(8)]
    perm = [3, 6, 1, 7, 0, 5, 2, 4]
    relabelled_cycle = relabel(eight_cycle, perm)
    eight_path = [(i, i + 1, i % 4, "x") for i in range(7)]
    for left, right in ((eight_cycle, relabelled_cycle),
                        (eight_cycle, eight_path),
                        (relabelled_cycle, eight_path)):
        assert (builder.canonical_typed_graph(8, left) == builder.canonical_typed_graph(8, right)) == \
               builder.brute_force_isomorphic(8, left, right)


def test_canonical_permutation_count_has_explicit_8_factorial_cap():
    complete_one_type = [(s, o, 0, "p") for s in range(8) for o in range(8) if s != o]
    key, stats = builder.canonical_typed_graph(8, complete_one_type, return_stats=True)
    assert len(key) == 64
    assert stats["permutations_enumerated"] == 40320
    assert stats["permutations_enumerated"] <= stats["permutation_cap"]


def test_independent_path_oracle_matches_frozen_global_conflict_and_hops():
    pairs = builder.all_pairs(8)
    examples = [
        [(0, 1, 0, "a"), (1, 2, 0, "a"), (2, 3, 0, "a")],
        [(0, 1, 0, "a"), (1, 2, 0, "a"), (2, 0, 0, "a")],
        [(0, 2, 0, "a"), (0, 2, 2, "c")],
        [(0, 1, 1, "b"), (2, 1, 0, "a")],
    ]
    rng = random.Random(20261027)
    for seed in range(80):
        examples.append(builder.candidate(seed + 9_000_000, n_range=(8, 8))["facts"])
    for facts in examples:
        assert builder.independent_oracle(8, facts, pairs) == oracle(8, facts, pairs)


def test_bounded_old_panel_reconstruction_matches_frozen_make_split():
    bounded, info = builder.bounded_frozen_split("test", 16, seed=60001, n_range=(4, 6))
    reference = make_split("test", 16, seed=60001, n_range=(4, 6))
    assert [row["graph_key"] for row in bounded] == [row["graph_key"] for row in reference]
    assert info["examined"] <= builder.MAX_CANDIDATES_PER_SPLIT


def test_small_candidate_build_balances_real_oracle_queries_and_active_ids():
    train, dev, selection = builder.build_candidate(
        train_per_class=2, dev_per_class=1, max_candidates=2000)
    train2, dev2, selection2 = builder.build_candidate(
        train_per_class=2, dev_per_class=1, max_candidates=2000)
    assert [r["canonical_graph_key"] for r in train] == [r["canonical_graph_key"] for r in train2]
    assert [r["canonical_graph_key"] for r in dev] == [r["canonical_graph_key"] for r in dev2]
    assert selection == selection2
    train2, dev2, selection2 = builder.build_candidate(
        train_per_class=2, dev_per_class=1, max_candidates=2000)
    assert [r["canonical_graph_key"] for r in train] == [r["canonical_graph_key"] for r in train2]
    assert [r["canonical_graph_key"] for r in dev] == [r["canonical_graph_key"] for r in dev2]
    assert selection == selection2
    train2, dev2, selection2 = builder.build_candidate(
        train_per_class=2, dev_per_class=1, max_candidates=2000)
    assert [r["canonical_graph_key"] for r in train] == [r["canonical_graph_key"] for r in train2]
    assert [r["canonical_graph_key"] for r in dev] == [r["canonical_graph_key"] for r in dev2]
    assert selection == selection2
    assert len(train) == 8 and len(dev) == 4
    assert builder.summarize(train)["query_labels"] == [16] * 4
    assert builder.summarize(dev)["query_labels"] == [8] * 4
    assert not ({r["canonical_graph_key"] for r in train} &
                {r["canonical_graph_key"] for r in dev})
    for rows in (train, dev):
        assert all(builder._active_nodes(row["facts"]) == set(range(8)) for row in rows)
        assert all(len(row["queries"]) == 8 and len(set(map(tuple, row["queries"]))) == 8
                   for row in rows)
        assert all(set(builder.independent_oracle(8, row["facts"], row["queries"])[0]) ==
                   {row["target_class"]} for row in rows)
    assert selection["split_disjoint"]
    assert selection["train"]["candidates_examined"] <= 2000
    assert selection["dev"]["candidates_examined"] <= 2000


def test_persisted_full_artifact_matches_manifest_and_full_balanced_quotas():
    artifact = ROOT / "research" / "artifacts" / "role8_active_balanced_20261027_v1"
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest); recorded = unsigned.pop("manifest_sha256")
    recomputed = hashlib.sha256(builder.canonical_json_bytes(unsigned)).hexdigest()
    assert manifest["status"] == builder.STATUS and recorded == recomputed
    train = json.loads((artifact / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((artifact / "dev.json").read_text(encoding="utf-8"))
    assert len(train) == 1024 and len(dev) == 64
    for name, rows, per_class in (("train.json", train, 256), ("dev.json", dev, 16)):
        file_sha = hashlib.sha256((artifact / name).read_bytes()).hexdigest()
        assert manifest["files"][name]["sha256"] == file_sha
        summary = builder.summarize(rows)
        assert summary["query_labels"] == [per_class * 8] * 4
        assert summary["abstract_graphs"] == len(rows)
        assert all(summary["fact_entity_exposure"]) and all(summary["query_entity_exposure"])
        assert all(builder._active_nodes(r["facts"]) == set(range(8)) for r in rows)
        assert all(len(r["facts"]) <= builder.Config().facts and len(r["queries"]) == 8 for r in rows)
    assert not ({r["canonical_graph_key"] for r in train} &
                {r["canonical_graph_key"] for r in dev})


def test_persisted_full_artifact_matches_manifest_and_full_balanced_quotas():
    artifact = ROOT / "research" / "artifacts" / "role8_active_balanced_20261027_v1"
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest); recorded = unsigned.pop("manifest_sha256")
    recomputed = hashlib.sha256(builder.canonical_json_bytes(unsigned)).hexdigest()
    assert manifest["status"] == builder.STATUS and recorded == recomputed
    train = json.loads((artifact / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((artifact / "dev.json").read_text(encoding="utf-8"))
    assert len(train) == 1024 and len(dev) == 64
    for name, rows, per_class in (("train.json", train, 256), ("dev.json", dev, 16)):
        file_sha = hashlib.sha256((artifact / name).read_bytes()).hexdigest()
        assert manifest["files"][name]["sha256"] == file_sha
        summary = builder.summarize(rows)
        assert summary["query_labels"] == [per_class * 8] * 4
        assert summary["abstract_graphs"] == len(rows)
        assert all(summary["fact_entity_exposure"]) and all(summary["query_entity_exposure"])
        assert all(builder._active_nodes(r["facts"]) == set(range(8)) for r in rows)
        assert all(len(r["facts"]) <= builder.Config().facts and len(r["queries"]) == 8 for r in rows)
    assert not ({r["canonical_graph_key"] for r in train} &
                {r["canonical_graph_key"] for r in dev})


def test_persisted_full_artifact_matches_manifest_and_full_balanced_quotas():
    artifact = ROOT / "research" / "artifacts" / "role8_active_balanced_20261027_v1"
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unsigned = dict(manifest); recorded = unsigned.pop("manifest_sha256")
    recomputed = hashlib.sha256(builder.canonical_json_bytes(unsigned)).hexdigest()
    assert manifest["status"] == builder.STATUS and recorded == recomputed
    train = json.loads((artifact / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((artifact / "dev.json").read_text(encoding="utf-8"))
    assert len(train) == 1024 and len(dev) == 64
    for name, rows, per_class in (("train.json", train, 256), ("dev.json", dev, 16)):
        file_sha = hashlib.sha256((artifact / name).read_bytes()).hexdigest()
        assert manifest["files"][name]["sha256"] == file_sha
        summary = builder.summarize(rows)
        assert summary["query_labels"] == [per_class * 8] * 4
        assert summary["abstract_graphs"] == len(rows)
        assert all(summary["fact_entity_exposure"]) and all(summary["query_entity_exposure"])
        assert all(builder._active_nodes(r["facts"]) == set(range(8)) for r in rows)
        assert all(len(r["facts"]) <= builder.Config().facts and len(r["queries"]) == 8 for r in rows)
    assert not ({r["canonical_graph_key"] for r in train} &
                {r["canonical_graph_key"] for r in dev})
