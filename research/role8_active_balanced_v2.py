"""CPU-only active-eight, class-balanced graph dataset candidate.

Rows use distinct abstract eight-node typed graphs, rather than rotations of a
small set. This module reads the frozen generator and prior non-final role8
artifacts only; it never loads a model or accesses the E2 final holdout.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fdt_local.data import candidate, collate, graph_key, graph_partition, oracle
from fdt_local.model_masked import Config

SCHEMA = "fdt_role8_active_balanced_v2"
STATUS = "CPU_ROLE8_ACTIVE_BALANCED_V2_CANDIDATE_NOT_TRAINED"
TRAIN_PER_CLASS = 256
DEV_PER_CLASS = 16
QUERIES_PER_GRAPH = 8
MAX_CANDIDATES_PER_SPLIT = 50_000
TRAIN_SEED_BASE = 3_010_000
DEV_SEED_BASE = 4_010_000
MAX_CANONICAL_PERMUTATIONS = 40_320
EXCLUDED_OLD_SEEDS = {"old_normal16": 60_001, "old_large16": 80_001}
RENAMING_MAP = [6, 7, 2, 3, 4, 5, 0, 1]


def canonical_json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _color_ids(signatures) -> tuple[int, ...]:
    ordered = {value: i for i, value in enumerate(sorted(set(signatures)))}
    return tuple(ordered[value] for value in signatures)


def _refined_cells(n: int, triples: set[tuple[int, int, int]]) -> list[tuple[int, ...]]:
    outgoing = [[] for _ in range(n)]
    incoming = [[] for _ in range(n)]
    for source, target, kind in triples:
        outgoing[source].append((kind, target))
        incoming[target].append((kind, source))
    initial = []
    for node in range(n):
        out_counts = tuple(sum(k == kind for k, _ in outgoing[node]) for kind in range(4))
        in_counts = tuple(sum(k == kind for k, _ in incoming[node]) for kind in range(4))
        initial.append((out_counts, in_counts))
    colors = _color_ids(initial)
    while True:
        signatures = []
        for node in range(n):
            out_sig = tuple(sorted((kind, colors[other]) for kind, other in outgoing[node]))
            in_sig = tuple(sorted((kind, colors[other]) for kind, other in incoming[node]))
            signatures.append((colors[node], out_sig, in_sig))
        new_colors = _color_ids(signatures)
        if new_colors == colors:
            break
        colors = new_colors
    groups = defaultdict(list)
    for node, color in enumerate(colors):
        groups[color].append(node)
    return [tuple(groups[color]) for color in sorted(groups)]


def canonical_typed_graph(n: int, facts: Iterable[tuple], *, return_stats=False):
    """Exact directed typed-graph key after 1-WL refinement.

    Only permutations within stable color cells are enumerated; the product of
    cell factorials is bounded by 8! (40,320) for n=8. Edge phrases are
    intentionally excluded from abstract graph identity.
    """
    if not isinstance(n, int) or not 1 <= n <= 8:
        raise ValueError("canonicalization supports 1..8 vertices")
    triples = {(int(s), int(o), int(kind)) for s, o, kind, _ in facts}
    if any(not (0 <= s < n and 0 <= o < n and s != o and 0 <= k < 4)
           for s, o, k in triples):
        raise ValueError("invalid typed directed edge")
    cells = _refined_cells(n, triples)
    count = 1
    for cell in cells:
        count *= math_factorial(len(cell))
    if count > MAX_CANONICAL_PERMUTATIONS:
        raise AssertionError("canonical permutation bound exceeded")
    best = None
    for orders in itertools.product(*(itertools.permutations(cell) for cell in cells)):
        order = tuple(node for cell_order in orders for node in cell_order)
        mapping = [0] * n
        for new, old in enumerate(order):
            mapping[old] = new
        encoded = tuple(sorted((mapping[s], mapping[o], kind) for s, o, kind in triples))
        if best is None or encoded < best:
            best = encoded
    key = sha256_bytes(canonical_json_bytes([n, best]))
    if return_stats:
        return key, {"refinement_cells": [len(cell) for cell in cells],
                     "permutations_enumerated": count,
                     "permutation_cap": MAX_CANONICAL_PERMUTATIONS}
    return key


def brute_force_isomorphic(n: int, left: Iterable[tuple], right: Iterable[tuple]) -> bool:
    """Slow exact reference used only by CPU tests and a few bounded checks."""
    a = {(int(s), int(o), int(k)) for s, o, k, *_ in left}
    b = {(int(s), int(o), int(k)) for s, o, k, *_ in right}
    if len(a) != len(b):
        return False
    for permutation in itertools.permutations(range(n)):
        if {(permutation[s], permutation[o], k) for s, o, k in a} == b:
            return True
    return False


def math_factorial(value: int) -> int:
    result = 1
    for number in range(2, value + 1):
        result *= number
    return result


def independent_oracle(n: int, facts: Iterable[tuple], queries: Iterable[tuple]):
    """Independent adjacency-list BFS oracle for labels and shortest hops."""
    adjacency = [[] for _ in range(n)]
    negatives = []
    for s, o, kind, _ in facts:
        if kind == 0:
            adjacency[s].append(o)
        elif kind == 1:
            adjacency[o].append(s)
        else:
            negatives.append((s, o, kind))
    distances = []
    for source in range(n):
        distance = [-1] * n
        distance[source] = 0
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for target in adjacency[node]:
                if distance[target] < 0:
                    distance[target] = distance[node] + 1
                    queue.append(target)
        distances.append(distance)
    positive_cycle = any(distances[target][source] >= 0
                         for source, targets in enumerate(adjacency) for target in targets)
    inconsistent = positive_cycle or any(
        (distances[s][o] >= 0 if kind == 2 else distances[o][s] >= 0)
        for s, o, kind in negatives)
    labels, hops = [], []
    for s, o in queries:
        if inconsistent:
            labels.append(3); hops.append(0)
        elif distances[s][o] >= 0:
            labels.append(1); hops.append(distances[s][o])
        elif distances[o][s] >= 0:
            labels.append(2); hops.append(distances[o][s])
        else:
            labels.append(0); hops.append(0)
    return labels, hops


def all_pairs(n: int = 8) -> list[tuple[int, int]]:
    return [(s, o) for s in range(n) for o in range(n) if s != o]


def _active_nodes(facts) -> set[int]:
    return {node for s, o, *_ in facts for node in (s, o)}


def bounded_frozen_split(split: str, count: int, *, seed: int, n_range,
                         max_candidates: int = MAX_CANDIDATES_PER_SPLIT):
    """Bounded reproduction of data.make_split's seeded partition selection."""
    if split not in ("train", "dev", "test") or count < 0:
        raise ValueError("invalid frozen split request")
    rows, seen, rejected = [], set(), Counter()
    examined = 0
    for offset in range(max_candidates):
        examined += 1
        row = candidate(seed + offset, n_range=n_range, heldout=False)
        if graph_partition(row["n"], row["facts"]) != split:
            rejected["other_partition"] += 1; continue
        if row["graph_key"] in seen:
            rejected["duplicate_physical_graph"] += 1; continue
        seen.add(row["graph_key"]); rows.append(row)
        if len(rows) >= count:
            break
    if len(rows) != count:
        raise RuntimeError(f"bounded frozen split shortfall {split}: {len(rows)}/{count} "
                           f"after {max_candidates} candidates; rejected={dict(rejected)}")
    return rows, {"requested": count, "examined": examined,
                  "rejected": dict(rejected), "seed": seed, "n_range": list(n_range)}


def _select_queries(seed: int, facts, target_class: int) -> list[tuple[int, int]] | None:
    pairs = all_pairs(8)
    labels, _ = independent_oracle(8, facts, pairs)
    eligible = [pair for pair, label in zip(pairs, labels) if label == target_class]
    if len(eligible) < QUERIES_PER_GRAPH:
        return None
    rng = random.Random(seed ^ (0x9E3779B9 * (target_class + 1)))
    return sorted(rng.sample(eligible, QUERIES_PER_GRAPH))


def _make_row(source, queries, target_class, split, canonical_key, seed):
    labels, hops = independent_oracle(8, source["facts"], queries)
    if labels != [target_class] * QUERIES_PER_GRAPH:
        raise ValueError("target class query selection changed under independent oracle")
    reference = oracle(8, source["facts"], queries)
    if reference != (labels, hops):
        raise ValueError("independent oracle disagrees with frozen oracle")
    return {"n": 8, "facts": [list(f) for f in source["facts"]],
        "queries": [list(q) for q in queries], "labels": labels, "hops": hops,
        "graph_key": graph_key(8, source["facts"]), "canonical_graph_key": canonical_key,
        "source_family_id": canonical_key, "split": split, "target_class": target_class,
        "source_seed": seed, "source_mode": source.get("mode")}


def build_exclusions(root: Path = ROOT) -> tuple[set[str], dict]:
    excluded = set()
    evidence = {}
    for name, seed in EXCLUDED_OLD_SEEDS.items():
        n_range = (7, 8) if name == "old_large16" else (4, 6)
        rows, stats = bounded_frozen_split("test", 16, seed=seed, n_range=n_range)
        keys = {canonical_typed_graph(8, row["facts"]) for row in rows}
        excluded.update(keys)
        evidence[name] = {**stats, "rows": len(rows), "abstract_graphs": len(keys),
                          "canonical_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(keys)))}
        if name == "old_normal16":
            renamed_keys = set()
            for row in rows:
                mapped = [(RENAMING_MAP[s], RENAMING_MAP[o], kind, phrase)
                          for s, o, kind, phrase in row["facts"]]
                if independent_oracle(8, mapped,
                        [(RENAMING_MAP[s], RENAMING_MAP[o]) for s, o in row["queries"]]) != (row["labels"], row["hops"]):
                    raise ValueError("known oldnormal paired rename changed oracle output")
                renamed_keys.add(canonical_typed_graph(8, mapped))
            if renamed_keys != keys:
                raise ValueError("oldnormal paired-ID rename is not canonical-invariant")
            excluded.update(renamed_keys)
            evidence["paired_new_ids16"] = {"graphs": len(rows), "mapping": RENAMING_MAP,
                "canonical_graphs": len(renamed_keys),
                "canonical_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(renamed_keys)))}

    role_dir = root / "research" / "artifacts" / "role8_coverage_20261026_v1"
    role_keys = set()
    for filename in ("train.json", "dev.json"):
        rows = json.loads((role_dir / filename).read_text(encoding="utf-8"))
        seen_source = set()
        for row in rows:
            source_key = row.get("source_graph_key")
            if source_key in seen_source:
                continue
            seen_source.add(source_key)
            shift = int(row.get("rotation", 0))
            facts = [((int(s) - shift) % 8, (int(o) - shift) % 8, int(k), phrase)
                     for s, o, k, phrase in row["facts"]]
            role_keys.add(canonical_typed_graph(8, facts))
        evidence[f"old_role8_{filename}"] = {"rendered_rows": len(rows),
            "source_families": len(seen_source), "canonical_graphs": len(seen_source)}
    excluded.update(role_keys)
    evidence["old_role8_source_groups"] = {"abstract_graphs": len(role_keys),
        "canonical_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(role_keys)))}
    evidence["exclusion_union"] = {"abstract_graphs": len(excluded),
        "canonical_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(excluded)))}
    return excluded, evidence


def summarize(rows: list[dict]) -> dict:
    labels = Counter(y for row in rows for y in row["labels"])
    fact_ids, query_ids = Counter(), Counter()
    fact_count, max_phrase = 0, 0
    for row in rows:
        for s, o, _, phrase in row["facts"]:
            fact_ids[s] += 1; fact_ids[o] += 1; fact_count += 1
            max_phrase = max(max_phrase, len(phrase.encode("utf-8")))
        for s, o in row["queries"]:
            query_ids[s] += 1; query_ids[o] += 1
    return {"graphs": len(rows), "query_labels": [labels[i] for i in range(4)],
        "abstract_graphs": len({r["canonical_graph_key"] for r in rows}),
        "physical_graphs": len({r["graph_key"] for r in rows}),
        "fact_entity_exposure": [fact_ids[i] for i in range(8)],
        "query_entity_exposure": [query_ids[i] for i in range(8)],
        "fact_count": fact_count, "max_phrase_bytes": max_phrase,
        "graph_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(r["graph_key"] for r in rows))),
        "canonical_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(r["canonical_graph_key"] for r in rows)))}


def build_candidate(*, train_per_class=TRAIN_PER_CLASS, dev_per_class=DEV_PER_CLASS,
                    max_candidates=MAX_CANDIDATES_PER_SPLIT, root: Path = ROOT):
    if max_candidates <= 0 or max_candidates > MAX_CANDIDATES_PER_SPLIT:
        raise ValueError("candidate budget must be in 1..50000")
    if train_per_class < 1 or dev_per_class < 1:
        raise ValueError("per-class graph quotas must be positive")
    excluded, exclusion_evidence = build_exclusions(root)
    config = Config()
    targets = {"train": train_per_class, "dev": dev_per_class}
    pools = {split: {label: [] for label in range(4)} for split in targets}
    unique_abstract = {split: set() for split in targets}
    physical_keys = {split: set() for split in targets}
    rejected = {split: Counter() for split in targets}
    examined = {split: 0 for split in targets}
    canonical_stats = Counter()
    seed_bases = {"train": TRAIN_SEED_BASE, "dev": DEV_SEED_BASE}
    for split, target in targets.items():
        seed_base = seed_bases[split]
        for offset in range(max_candidates):
            if all(len(pools[split][label]) >= target for label in range(4)):
                break
            seed = seed_base + offset
            examined[split] += 1
            row = candidate(seed, n_range=(8, 8), heldout=False)
            if row["n"] != 8:
                rejected[split]["wrong_entity_count"] += 1; continue
            if len(row["facts"]) > config.facts or not row["facts"]:
                rejected[split]["fact_capacity"] += 1; continue
            if _active_nodes(row["facts"]) != set(range(8)):
                rejected[split]["not_all_entities_active"] += 1; continue
            full_pairs = all_pairs(8)
            all_labels, all_hops = independent_oracle(8, row["facts"], full_pairs)
            if oracle(8, row["facts"], full_pairs) != (all_labels, all_hops):
                raise ValueError("source candidate oracle cross-check failed")
            feasible = [label for label in range(4)
                        if sum(y == label for y in all_labels) >= QUERIES_PER_GRAPH
                        and len(pools[split][label]) < target]
            if not feasible:
                rejected[split]["no_available_query_class_quota"] += 1; continue
            try:
                canonical_key, stat = canonical_typed_graph(8, row["facts"], return_stats=True)
            except (TypeError, ValueError) as exc:
                rejected[split]["invalid_candidate"] += 1; continue
            canonical_stats["canonical_graphs"] += 1
            canonical_stats["permutations_enumerated"] += stat["permutations_enumerated"]
            canonical_stats["max_permutations_enumerated"] = max(
                canonical_stats["max_permutations_enumerated"], stat["permutations_enumerated"])
            bucket = int(canonical_key[:16], 16) % 10
            assigned = "train" if bucket < 8 else "dev" if bucket == 8 else "test"
            if assigned != split:
                rejected[split]["stable_hash_other_split"] += 1; continue
            if canonical_key in excluded:
                rejected[split]["overlaps_observed_graphs"] += 1; continue
            if canonical_key in unique_abstract[split]:
                rejected[split]["duplicate_abstract_graph"] += 1; continue
            physical = graph_key(8, row["facts"])
            if physical in physical_keys[split]:
                rejected[split]["duplicate_physical_graph"] += 1; continue
            # A graph belongs to one class quota; deterministic round-robin among
            # currently-short classes avoids overfilling common target classes.
            label = min(feasible, key=lambda c: (len(pools[split][c]) / target, c))
            queries = _select_queries(seed, row["facts"], label)
            output = _make_row(row, queries, label, split, canonical_key, seed)
            features, supervision, _ = collate([output], config, "cpu")
            if set(features) != {"tokens", "roles", "fact_mask", "queries", "query_mask"}:
                raise ValueError("supervision leaked into feature contract")
            if max(len(str(f[3]).encode("utf-8")) for f in output["facts"]) > config.phrase_bytes:
                rejected[split]["phrase_bytes"] += 1; continue
            pools[split][label].append(output)
            unique_abstract[split].add(canonical_key)
            physical_keys[split].add(physical)
        short = {label: target - len(pools[split][label]) for label in range(4)
                 if len(pools[split][label]) < target}
        if short:
            raise RuntimeError(f"{split} candidate quota shortfall within {max_candidates}: {short}")

    train_rows = [row for label in range(4) for row in pools["train"][label]]
    dev_rows = [row for label in range(4) for row in pools["dev"][label]]
    train_iso = {row["canonical_graph_key"] for row in train_rows}
    dev_iso = {row["canonical_graph_key"] for row in dev_rows}
    if train_iso & dev_iso:
        raise ValueError("train/dev abstract graph overlap")
    for split, rows in (("train", train_rows), ("dev", dev_rows)):
        summary = summarize(rows)
        expected = targets[split]
        if len(rows) != expected * 4 or summary["query_labels"] != [expected * 8] * 4:
            raise ValueError(f"{split} graph/query class balance mismatch")
        if not all(summary["fact_entity_exposure"]) or not all(summary["query_entity_exposure"]):
            raise ValueError(f"{split} entity exposure has an unobserved ID")
        for row in rows:
            if _active_nodes(row["facts"]) != set(range(8)):
                raise ValueError(f"{split} graph lost an active entity")
            if len(row["queries"]) != QUERIES_PER_GRAPH or len(set(map(tuple, row["queries"]))) != QUERIES_PER_GRAPH:
                raise ValueError(f"{split} query selection is not eight distinct pairs")

    return train_rows, dev_rows, {
        "train": {"candidates_examined": examined["train"], "quotas": targets["train"],
            "accepted_by_query_class": [len(pools["train"][k]) for k in range(4)],
            "rejections": dict(rejected["train"]), "summary": summarize(train_rows)},
        "dev": {"candidates_examined": examined["dev"], "quotas": targets["dev"],
            "accepted_by_query_class": [len(pools["dev"][k]) for k in range(4)],
            "rejections": dict(rejected["dev"]), "summary": summarize(dev_rows)},
        "canonicalization": {"algorithm": "directed-typed color refinement + exact within-cell permutations",
            "max_permutations_per_graph": MAX_CANONICAL_PERMUTATIONS,
            "canonical_graphs_evaluated": canonical_stats["canonical_graphs"],
            "total_permutations_enumerated": canonical_stats["permutations_enumerated"],
            "worst_permutations_enumerated": canonical_stats["max_permutations_enumerated"]},
        "exclusions": exclusion_evidence,
        "split_rule": "canonical SHA256 first 64 bits modulo 10: 0..7 train, 8 dev, 9 unused test",
        "split_disjoint": True,
    }


def write_candidate(output: Path, *, train_per_class=TRAIN_PER_CLASS,
                    dev_per_class=DEV_PER_CLASS, max_candidates=MAX_CANDIDATES_PER_SPLIT):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"candidate output already exists: {output}")
    train, dev, selection = build_candidate(train_per_class=train_per_class,
        dev_per_class=dev_per_class, max_candidates=max_candidates)
    output.mkdir(parents=True)
    paths = {"train.json": output / "train.json", "dev.json": output / "dev.json"}
    for name, rows in (("train.json", train), ("dev.json", dev)):
        paths[name].write_bytes(canonical_json_bytes(rows) + b"\n")
    manifest = {"schema": SCHEMA, "status": STATUS,
        "construction": {"n": 8, "all_entities_active_in_premises": True,
            "training_graphs": len(train), "development_graphs": len(dev),
            "graphs_per_query_class": {"train": train_per_class, "dev": dev_per_class},
            "distinct_queries_per_graph": QUERIES_PER_GRAPH, "all_56_ordered_nonself_pairs_considered": True,
            "oracle_global_conflict_rule_changed": False,
            "distinct_abstract_graphs": True, "independence_claimed": False,
            "random_query_label_assignment": False,
            "feature_whitelist": ["tokens", "roles", "fact_mask", "queries", "query_mask"]},
        "limits": {"candidate_cap_per_split": max_candidates,
            "max_exact_permutations_per_graph": MAX_CANONICAL_PERMUTATIONS,
            "facts_max": Config().facts, "queries_per_graph": QUERIES_PER_GRAPH,
            "phrase_bytes_max": Config().phrase_bytes},
        "source_sha256": {"frozen_data_py": sha256_file(ROOT / "src" / "fdt_local" / "data.py"),
            "model_config_py": sha256_file(ROOT / "src" / "fdt_local" / "model_masked.py"),
            "builder_py": sha256_file(Path(__file__)),
            "old_role8_manifest": sha256_file(ROOT / "research" / "artifacts" / "role8_coverage_20261026_v1" / "manifest.json")},
        "selection": selection,
        "files": {name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                  for name, path in paths.items()},
        "nonclaims": ["No training or model evaluation was run.",
            "No guarantee is made about all historical graph overlap beyond the enumerated exclusions.",
            "The deterministic hash test bucket is held out and not included in train or dev."],
        "e2_final_files_accessed": False, "model_or_checkpoint_loaded": False,
        "optimizer_updates": 0}
    manifest["manifest_payload_sha256_without_self"] = sha256_bytes(canonical_json_bytes(manifest))
    (output / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-class", type=int, default=TRAIN_PER_CLASS)
    parser.add_argument("--dev-per-class", type=int, default=DEV_PER_CLASS)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES_PER_SPLIT)
    args = parser.parse_args()
    manifest = write_candidate(args.output.resolve(), train_per_class=args.train_per_class,
        dev_per_class=args.dev_per_class, max_candidates=args.max_candidates)
    manifest_file = args.output.resolve() / "manifest.json"
    print(json.dumps({"status": manifest["status"],
        "manifest_payload_sha256_without_self": manifest["manifest_payload_sha256_without_self"],
        "manifest_file_sha256": sha256_file(manifest_file),
        "train": manifest["selection"]["train"], "dev": manifest["selection"]["dev"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
