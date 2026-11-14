"""Independent CPU contracts for E2 data; no model training."""
import copy
import importlib.util
from pathlib import Path

import pytest

from fdt_local.data import collate, oracle
from fdt_local.model import Config

spec = importlib.util.spec_from_file_location("e2", Path(__file__).with_name("e2_data_v1.py"))
e2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2)


def independent_labels(n, facts, queries):
    # Enumerate all simple directed paths, not the production Floyd closure.
    adjacency = {i: set() for i in range(n)}
    for s, o, kind, _ in facts:
        if kind == 0:
            adjacency[s].add(o)
        elif kind == 1:
            adjacency[o].add(s)
    def reaches(a, b):
        pending, seen = list(adjacency[a]), set()
        while pending:
            value = pending.pop()
            if value == b:
                return True
            if value not in seen:
                seen.add(value)
                pending.extend(adjacency[value])
        return False
    conflict = any(reaches(i, i) for i in range(n))
    conflict |= any((kind == 2 and reaches(s, o)) or (kind == 3 and reaches(o, s))
                    for s, o, kind, _ in facts)
    return [3 if conflict else 1 if reaches(s, o) else 2 if reaches(o, s) else 0 for s, o in queries]


@pytest.fixture(scope="module")
def candidate():
    return e2.build({"train": 32, "dev": 16, "final": 16})[0]


def test_exposure_families_and_no_oracle_input(candidate):
    for split, rows in candidate.items():
        summary = e2.summarize(rows)
        assert summary["query_counts"] == [len(rows) // 4] * 4
        assert summary["grounding_counts"] == [len(rows) * 2] * 4
        for family in e2.FAMILIES[split]:
            subset = [row for row in rows if row["family_id"] == family]
            assert e2.summarize(subset)["query_counts"] == [len(subset) // 4] * 4
        features, targets, size = collate(rows[:2], Config())
        assert set(features) == {"tokens", "roles", "fact_mask", "queries", "query_mask"}
        assert set(targets) == {"labels", "ground", "hops"}
        assert size == sum(len(p.encode("utf-8")) for row in rows[:2] for *_, p in row["facts"])
    e2.validate(candidate)


def test_independent_path_oracle_and_negative_counterexamples(candidate):
    for rows in candidate.values():
        for row in rows:
            assert independent_labels(row["n"], row["facts"], row["queries"]) == row["labels"]
    negative_only = [(0, 1, 2, "not greater")]
    assert oracle(3, negative_only, [(0, 1), (1, 0)])[0] == [0, 0]
    transitive_conflict = [(0, 1, 0, "greater"), (1, 2, 0, "greater"), (0, 2, 2, "not greater")]
    assert independent_labels(3, transitive_conflict, [(2, 0), (0, 1)]) == [3, 3]


def test_isomorphic_identity_and_cross_split_leakage(candidate):
    row = candidate["train"][0]
    p = list(reversed(range(row["n"])))
    renamed = [(p[s], p[o], k, phrase) for s, o, k, phrase in reversed(row["facts"])]
    assert e2.isomorphic_key(row["n"], renamed) == row["isomorphic_key"]
    damaged = copy.deepcopy(candidate)
    # A valid same-family row moved into dev must be rejected, not relabeled.
    damaged["dev"][0] = copy.deepcopy(row)
    with pytest.raises(ValueError):
        e2.validate(damaged)


def test_invalid_labels_phrases_and_supply_fail_closed(candidate):
    bad = copy.deepcopy(candidate)
    bad["train"][0]["labels"] = [(bad["train"][0]["labels"][0] + 1) % 4]
    with pytest.raises(ValueError, match="oracle"):
        e2.validate(bad)
    bad = copy.deepcopy(candidate)
    bad["train"][0]["facts"][0] = (*bad["train"][0]["facts"][0][:3], "wrong semantic phrase")
    with pytest.raises(ValueError, match="phrase"):
        e2.validate(bad)
    with pytest.raises(RuntimeError, match="bounded_supply_shortfall"):
        e2.build_split("train", 32, 1, set(), max_candidates=1)


def test_deterministic_rebuild(candidate):
    rebuilt, _ = e2.build({"train": 32, "dev": 16, "final": 16})
    assert e2.digest(rebuilt) == e2.digest(candidate)
