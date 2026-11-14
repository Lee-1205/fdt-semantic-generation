"""CPU-only candidate for cyclic entity-ID coverage on the frozen graph task.

Every rendered row is derived from a graph in the original E1 training pool.
The eight rotations are correlated views of one abstract graph, not independent
examples. This builder does not access model checkpoints or E2 final data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Iterable

from fdt_local.data import graph_key, make_split, oracle

VERSION = "role8-cyclic-coverage-candidate-v1"
TRAIN_BASES = 64
DEV_BASES = 16
ROTATIONS = 8
MAX_SOURCE_CANDIDATES = 4096
TRAIN_POOL_SEED = 301
OLD_DEV_SEED = 301
NEW_TEST_SEED = 60001
HELD_CONTROL_SEED = 70001


def canonical_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_graph_key(n: int, facts: Iterable[tuple]) -> str:
    """Exact typed-graph isomorphism key for the bounded original n<=6 graphs."""
    if not isinstance(n, int) or not 4 <= n <= 6:
        raise ValueError("source canonicalization is defined only for n in 4..6")
    triples = {(int(s), int(o), int(kind)) for s, o, kind, _ in facts}
    best = min(tuple(sorted((permutation[s], permutation[o], kind)
                            for s, o, kind in triples))
               for permutation in itertools.permutations(range(n)))
    return sha256_bytes(canonical_bytes([n, best]))


def cyclic_renaming(row: dict, shift: int) -> dict:
    if not 0 <= shift < ROTATIONS:
        raise ValueError("shift must be in 0..7")
    old_n = row["n"]
    if not 4 <= old_n <= 6:
        raise ValueError("source graph entity count must be 4..6")
    rename = lambda entity: (int(entity) + shift) % ROTATIONS
    facts = [(rename(s), rename(o), int(kind), phrase)
             for s, o, kind, phrase in row["facts"]]
    queries = [(rename(s), rename(o)) for s, o in row["queries"]]
    labels, hops = oracle(ROTATIONS, facts, queries)
    if labels != row["labels"] or hops != row["hops"]:
        raise ValueError("adding unused entities or cyclic renaming changed oracle targets")
    result = {
        "n": ROTATIONS,
        "source_n": old_n,
        "facts": facts,
        "queries": queries,
        "labels": labels,
        "hops": hops,
        "graph_key": graph_key(ROTATIONS, facts),
        "source_graph_key": row["graph_key"],
        "canonical_source_key": canonical_graph_key(old_n, row["facts"]),
        "rotation": shift,
    }
    return result


def eight_rotations(row: dict) -> list[dict]:
    rendered = [cyclic_renaming(row, shift) for shift in range(ROTATIONS)]
    keys = [item["graph_key"] for item in rendered]
    if len(set(keys)) != ROTATIONS:
        raise ValueError("source rejected: cyclic rotations have duplicate physical graph keys")
    return rendered


def choose_train_bases(source_pool: list[dict], forbidden_iso: set[str],
                       count: int = TRAIN_BASES, max_candidates: int = MAX_SOURCE_CANDIDATES):
    accepted, accepted_iso, physical_keys = [], set(), set()
    rejected = Counter()
    examined = 0
    for row in source_pool[:max_candidates]:
        examined += 1
        iso = canonical_graph_key(row["n"], row["facts"])
        if iso in forbidden_iso:
            rejected["overlaps_olddev_or_test_control"] += 1
            continue
        if iso in accepted_iso:
            rejected["duplicate_source_isomorphism"] += 1
            continue
        try:
            rendered = eight_rotations(row)
        except ValueError:
            rejected["duplicate_rotation_physical_key"] += 1
            continue
        candidate_physical = {item["graph_key"] for item in rendered}
        if candidate_physical & physical_keys:
            rejected["duplicate_accepted_physical_key"] += 1
            continue
        accepted.append((row, iso, rendered))
        accepted_iso.add(iso)
        physical_keys.update(candidate_physical)
        if len(accepted) >= count:
            break
    return accepted, {"candidates_examined": examined,
                      "candidate_limit": min(max_candidates, len(source_pool)),
                      "rejected": dict(rejected), "accepted_bases": len(accepted),
                      "shortfall": max(0, count - len(accepted))}


def choose_dev_bases(old_dev: list[dict], forbidden_iso: set[str],
                     forbidden_physical: set[str], count: int = DEV_BASES):
    accepted, accepted_iso, physical_keys = [], set(), set()
    rejected = Counter()
    for row in old_dev:
        iso = canonical_graph_key(row["n"], row["facts"])
        if iso in forbidden_iso:
            rejected["overlaps_train_or_test_control"] += 1
            continue
        if iso in accepted_iso:
            rejected["duplicate_dev_isomorphism"] += 1
            continue
        try:
            rendered = eight_rotations(row)
        except ValueError:
            rejected["duplicate_rotation_physical_key"] += 1
            continue
        candidate_physical = {item["graph_key"] for item in rendered}
        if candidate_physical & (forbidden_physical | physical_keys):
            rejected["duplicate_physical_key"] += 1
            continue
        accepted.append((row, iso, rendered))
        accepted_iso.add(iso)
        physical_keys.update(candidate_physical)
        if len(accepted) >= count:
            break
    return accepted, {"source_pool_rows": len(old_dev), "accepted_bases": len(accepted),
                      "rejected": dict(rejected), "shortfall": max(0, count - len(accepted))}


def flatten_groups(groups: list[tuple[dict, str, list[dict]]], split: str) -> list[dict]:
    rows = []
    for index, (source, iso, rendered) in enumerate(groups):
        for item in rendered:
            rows.append({**item, "split": split, "source_index": index,
                         "source_mode": source.get("mode")})
    return rows


def summarize_rows(rows: list[dict]) -> dict:
    query_class, ground_class = Counter(), Counter()
    query_source, query_object = Counter(), Counter()
    fact_source, fact_object = Counter(), Counter()
    input_bytes = 0
    for row in rows:
        query_class.update(row["labels"])
        for source, target in row["queries"]:
            query_source[source] += 1
            query_object[target] += 1
        for source, target, kind, phrase in row["facts"]:
            fact_source[source] += 1
            fact_object[target] += 1
            ground_class[kind] += 1
            input_bytes += len(phrase.encode("utf-8"))
    expand = lambda counter: [counter[index] for index in range(ROTATIONS)]
    return {
        "rendered_rows": len(rows),
        "abstract_source_graphs": len({row["source_graph_key"] for row in rows}),
        "physical_graph_keys": len({row["graph_key"] for row in rows}),
        "canonical_source_graphs": len({row["canonical_source_key"] for row in rows}),
        "query_class_counts": [query_class[index] for index in range(4)],
        "ground_class_counts": [ground_class[index] for index in range(4)],
        "query_source_entity_id_counts": expand(query_source),
        "query_object_entity_id_counts": expand(query_object),
        "fact_source_entity_id_counts": expand(fact_source),
        "fact_object_entity_id_counts": expand(fact_object),
        "input_utf8_bytes": input_bytes,
        "content_sha256": sha256_bytes(canonical_bytes(rows)),
    }


def _partition_base_sets(train_groups, dev_groups, old_dev, new_test, held_control):
    keys = lambda rows: {canonical_graph_key(row["n"], row["facts"]) for row in rows}
    train_keys = {iso for _, iso, _ in train_groups}
    dev_keys = {iso for _, iso, _ in dev_groups}
    old_dev_keys = keys(old_dev)
    test_keys = keys(new_test)
    held_keys = keys(held_control)
    return {"train_source_vs_olddev96": not (train_keys & old_dev_keys),
            "train_source_vs_newtest16": not (train_keys & test_keys),
            "train_source_vs_held_control16": not (train_keys & held_keys),
            "train_source_vs_dev_source": not (train_keys & dev_keys),
            "dev_source_is_subset_of_olddev96": dev_keys <= old_dev_keys,
            "dev_source_vs_newtest16": not (dev_keys & test_keys),
            "dev_source_vs_held_control16": not (dev_keys & held_keys),
            "train_source_count": len(train_keys), "dev_source_count": len(dev_keys),
            "olddev96_unique_canonical_count": len(old_dev_keys),
            "newtest16_unique_canonical_count": len(test_keys),
            "held_control16_unique_canonical_count": len(held_keys)}


def build_candidate(*, train_bases: int = TRAIN_BASES, dev_bases: int = DEV_BASES,
                    max_candidates: int = MAX_SOURCE_CANDIDATES) -> tuple[dict, dict, dict]:
    source_pool = make_split("train", MAX_SOURCE_CANDIDATES, seed=TRAIN_POOL_SEED)
    old_dev = make_split("dev", 96, seed=OLD_DEV_SEED)
    new_test = make_split("test", 16, seed=NEW_TEST_SEED)
    held_control = make_split("test", 16, seed=HELD_CONTROL_SEED)

    old_dev_iso = {canonical_graph_key(row["n"], row["facts"]) for row in old_dev}
    test_iso = {canonical_graph_key(row["n"], row["facts"]) for row in new_test}
    held_iso = {canonical_graph_key(row["n"], row["facts"]) for row in held_control}
    forbidden = old_dev_iso | test_iso | held_iso
    train_groups, train_stats = choose_train_bases(source_pool, forbidden,
        count=train_bases, max_candidates=max_candidates)

    train_iso = {iso for _, iso, _ in train_groups}
    train_physical = {item["graph_key"] for _, _, rows in train_groups for item in rows}
    dev_forbidden = train_iso | test_iso | held_iso
    dev_groups, dev_stats = choose_dev_bases(old_dev, dev_forbidden, train_physical, count=dev_bases)
    train_rows = flatten_groups(train_groups, "train")
    dev_rows = flatten_groups(dev_groups, "dev_seen_original_development")

    # Independently assert physical identity, oracle preservation, and exact ID balance.
    all_physical = [row["graph_key"] for row in train_rows + dev_rows]
    if len(all_physical) != len(set(all_physical)):
        raise ValueError("rendered train/dev physical graph keys are not unique")
    expected_train_rows = len(train_groups) * ROTATIONS
    expected_dev_rows = len(dev_groups) * ROTATIONS
    train_summary, dev_summary = summarize_rows(train_rows), summarize_rows(dev_rows)
    for summary, expected, split in ((train_summary, expected_train_rows, "train"),
                                     (dev_summary, expected_dev_rows, "dev")):
        if summary["rendered_rows"] != expected or summary["physical_graph_keys"] != expected:
            raise ValueError(f"{split} split shortfall or physical key collision")
        for key in ("query_source_entity_id_counts", "query_object_entity_id_counts",
                    "fact_source_entity_id_counts", "fact_object_entity_id_counts"):
            if len(set(summary[key])) != 1:
                raise ValueError(f"{split} {key} is not balanced over all eight IDs: {summary[key]}")
    partition = _partition_base_sets(train_groups, dev_groups, old_dev, new_test, held_control)
    failed = [name for name, value in partition.items() if isinstance(value, bool) and not value]
    if failed:
        raise ValueError(f"canonical partition overlap/selection failure: {failed}")

    root = Path(__file__).resolve().parents[1]
    data_source = root / "src" / "fdt_local" / "data.py"
    source_pool_bytes = canonical_bytes(source_pool)
    manifest = {
        "schema": "fdt_role8_coverage_v1",
        "version": VERSION,
        "status": "CPU_ROLE8_COVERAGE_CANDIDATE_NOT_TRAINED",
        "seed": {"train_source_pool": TRAIN_POOL_SEED, "old_dev96": OLD_DEV_SEED,
                 "new_test16_control": NEW_TEST_SEED, "held_control16": HELD_CONTROL_SEED},
        "construction": {"source_pool": "make_split('train', 4096) from frozen original source",
            "selected_source_bases": train_bases, "rendered_rows_per_source": ROTATIONS,
            "training_rendered_rows": len(train_rows), "dev_source_bases": dev_bases,
            "dev_rendered_rows": len(dev_rows), "cyclic_map": "id -> (id + shift) mod 8 for shift 0..7",
            "all_rendered_rows_mark_n_8": True, "original_source_n_range": [4, 6],
            "unused_entity_slots_are_isolated": True,
            "rendered_rows_are_independent_examples": False,
            "same_base_graph_rotations_stay_in_one_partition": True,
            "labels_and_hops_recomputed_with_original_oracle": True,
            "oracle_targets_identical_to_source": True,
            "phrase_policy_changed": False, "grounding_policy_changed": False,
            "feature_whitelist": ["tokens", "roles", "fact_mask", "queries", "query_mask"]},
        "lineage_and_limits": {"source_graphs_are_from_original_e1_training_pool": True,
            "new_graph_generalization_claim": False,
            "historical_isomorphic_leakage_to_e1_model": "unknown; E1 train source pool is known seen data",
            "dev_is_subset_of_original_observed_dev96": True,
            "dev_is_final_claim_material": False,
            "new_test16_used_only_as_canonical_exclusion_control": True,
            "held_control16_used_only_as_canonical_exclusion_control": True,
            "e2_final_data_accessed": False,
            "e2_final_data_evaluated": False,
            "optimizer_updates": 0,
            "model_evaluation": False},
        "source_sha256": {"frozen_data_py": sha256_bytes(data_source.read_bytes()),
            "source_pool_rows_canonical": sha256_bytes(source_pool_bytes),
            "selected_source_rows_canonical": sha256_bytes(canonical_bytes([row for row, _, _ in train_groups])),
            "old_dev96_canonical": sha256_bytes(canonical_bytes(old_dev)),
            "new_test16_canonical": sha256_bytes(canonical_bytes(new_test)),
            "held_control16_canonical": sha256_bytes(canonical_bytes(held_control))},
        "source_pool": {"requested": MAX_SOURCE_CANDIDATES, "generated": len(source_pool),
            "train_partition_rows": len(source_pool), "selection": train_stats},
        "dev_selection": dev_stats,
        "canonical_partition_checks": partition,
        "train": train_summary,
        "dev": dev_summary,
        "final_held_unused": True,
        "shortfall": {"train_bases": train_stats["shortfall"], "dev_bases": dev_stats["shortfall"]},
    }
    payloads = {"train": train_rows, "dev": dev_rows}
    return payloads, manifest, {"train_groups": train_groups, "dev_groups": dev_groups,
        "source_pool": source_pool, "old_dev96": old_dev, "new_test16": new_test,
        "held_control16": held_control}


def write_candidate(output: Path, *, train_bases: int = TRAIN_BASES,
                    dev_bases: int = DEV_BASES, max_candidates: int = MAX_SOURCE_CANDIDATES) -> dict:
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    payloads, manifest, _ = build_candidate(train_bases=train_bases,
                                             dev_bases=dev_bases,
                                             max_candidates=max_candidates)
    output.mkdir(parents=True)
    for split, rows in payloads.items():
        data = canonical_bytes(rows)
        (output / f"{split}.json").write_bytes(data)
    data_source = Path(__file__).resolve().parents[1] / "src" / "fdt_local" / "data.py"
    manifest["builder_sha256"] = sha256_bytes(Path(__file__).read_bytes())
    manifest["builder_path"] = str(Path(__file__).resolve())
    manifest["runtime_source_sha256"] = sha256_bytes(b"".join(
        path.name.encode() + path.read_bytes()
        for path in sorted(data_source.parent.glob("*.py"))))
    manifest["files"] = {path.name: {"sha256": sha256_bytes(path.read_bytes()),
        "bytes": path.stat().st_size} for path in sorted(output.glob("*.json"))}
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8")
    (output / "manifest.json").write_bytes(manifest_bytes)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-bases", type=int, default=TRAIN_BASES)
    parser.add_argument("--dev-bases", type=int, default=DEV_BASES)
    parser.add_argument("--max-candidates", type=int, default=MAX_SOURCE_CANDIDATES)
    args = parser.parse_args()
    manifest = write_candidate(args.output, train_bases=args.train_bases,
        dev_bases=args.dev_bases, max_candidates=args.max_candidates)
    print(json.dumps({"status": manifest["status"], "train": manifest["train"],
        "dev": manifest["dev"], "shortfall": manifest["shortfall"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
