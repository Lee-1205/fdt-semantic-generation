"""CPU-only E2 candidate construction. Never changes the frozen local package.

One query per row makes query exposure explicit; every row has two facts of
each grounding kind. This is a proposed curriculum, not the baseline sampler.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random

from fdt_local.data import graph_key, oracle

VERSION = "e2-balanced-family-candidate-v1"
FAMILIES = {
    "train": {
        "en_plain": ("is greater than", "is less than", "is not greater than", "is not less than"),
        "ko_plain": ("더 크다", "더 작다", "크지 않다", "작지 않다"),
    },
    "dev": {
        "en_exceed": ("value exceeds", "value is below", "value does not exceed", "value is not below"),
        "ko_numeric": ("수치가 더 높다", "수치가 더 낮다", "수치가 높지는 않다", "수치가 낮지는 않다"),
    },
    "final": {
        "en_have_value": ("has a higher value than", "has a lower value than", "does not have a higher value than", "does not have a lower value than"),
        "ko_relative": ("값이 상대보다 높다", "값이 상대보다 낮다", "값이 상대보다 높지 않다", "값이 상대보다 낮지 않다"),
    },
}
N_RANGE = (4, 5)


def canonical_bytes(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def isomorphic_key(n, facts):
    """Exact entity-renaming invariant for this bounded n<=5 candidate.

    Kind IDs remain distinct, even when a reversed LESS fact has the same
    positive closure as GREATER. Duplicates and fact order do not affect keys.
    """
    if not 1 <= n <= 5:
        raise ValueError("exact canonicalization is bounded to n<=5")
    triples = {(int(s), int(o), int(k)) for s, o, k, _ in facts}
    best = min(tuple(sorted((p[s], p[o], k) for s, o, k in triples))
               for p in itertools.permutations(range(n)))
    return digest([n, best])


def typed_candidate(rng):
    n = rng.randint(*N_RANGE)
    order = list(range(n))
    rng.shuffle(order)
    possible = [(order[i], order[j]) for i in range(n) for j in range(i + 1, n)]
    # Positive DAG; negative contradictions are accepted only as CONFLICT.
    positives = rng.sample(possible, 4)
    facts = []
    for k in (0, 1):
        for a, b in positives[k * 2:k * 2 + 2]:
            s, o = (a, b) if k == 0 else (b, a)
            facts.append((s, o, k, ""))
    pairs = list(itertools.permutations(range(n), 2))
    for k in (2, 3):
        facts.extend((s, o, k, "") for s, o in rng.sample(pairs, 2))
    rng.shuffle(facts)
    labels, hops = oracle(n, facts, pairs)
    return n, facts, pairs, labels, hops


def build_split(split, count, seed, occupied, max_candidates=10000):
    if split not in FAMILIES or count <= 0 or count % 4:
        raise ValueError("known split and positive count divisible by four required")
    if max_candidates <= 0:
        raise ValueError("positive candidate limit required")
    rng = random.Random(seed)
    families = list(FAMILIES[split])
    rows, attempts = [], 0
    supplied = Counter()
    local_keys = set()
    while len(rows) < count and attempts < max_candidates:
        attempts += 1
        n, facts, pairs, labels, hops = typed_candidate(rng)
        supplied.update(set(labels))
        wanted = len(rows) % 4
        choices = [i for i, label in enumerate(labels) if label == wanted]
        if not choices:
            continue
        iso = isomorphic_key(n, facts)
        if iso in occupied or iso in local_keys:
            continue
        chosen = rng.choice(choices)
        # Four consecutive rows cover all query kinds. Alternate families per
        # block, rather than binding one family to one query label.
        family = families[(len(rows) // 4) % len(families)]
        phrases = FAMILIES[split][family]
        rendered = [(s, o, k, phrases[k]) for s, o, k, _ in facts]
        row = {"n": n, "facts": rendered, "queries": [pairs[chosen]],
               "labels": [labels[chosen]], "hops": [hops[chosen]],
               "graph_key": graph_key(n, facts), "isomorphic_key": iso,
               "family_id": family, "mode": "balanced_dag_with_negatives"}
        rows.append(row)
        local_keys.add(iso)
    if len(rows) != count:
        raise RuntimeError(json.dumps({"reason": "bounded_supply_shortfall", "split": split,
            "requested_rows": count, "built_rows": len(rows), "candidates": attempts,
            "remaining_query_quota": [count // 4 - sum(r["labels"] == [k] for r in rows) for k in range(4)]}))
    occupied.update(local_keys)
    return rows, {"candidates": attempts, "candidate_class_availability": dict(supplied),
                  "rows_accepted": count, "max_candidates": max_candidates}


def summarize(rows):
    query, ground, family = Counter(), Counter(), Counter()
    raw_bytes = 0
    for row in rows:
        query.update(row["labels"])
        family.update([row["family_id"]])
        for _, _, kind, phrase in row["facts"]:
            ground.update([kind])
            raw_bytes += len(phrase.encode("utf-8"))
    return {"rows": len(rows), "valid_queries": sum(query.values()),
            "valid_facts": sum(ground.values()), "query_counts": [query[k] for k in range(4)],
            "grounding_counts": [ground[k] for k in range(4)], "family_counts": dict(family),
            "input_utf8_bytes": raw_bytes, "isomorphic_graphs": len({r["isomorphic_key"] for r in rows}),
            "content_sha256": digest(rows)}


def validate(splits):
    """Strict structural/semantic-oracle contract, not learned performance."""
    all_graphs, all_phrases, all_families = set(), set(), set()
    for split, rows in splits.items():
        phrases = {p for family in FAMILIES[split].values() for p in family}
        families = set(FAMILIES[split])
        if phrases & all_phrases or families & all_families:
            raise ValueError("expression family or literal leakage")
        all_phrases.update(phrases)
        all_families.update(families)
        for row in rows:
            if row["family_id"] not in families or not N_RANGE[0] <= row["n"] <= N_RANGE[1]:
                raise ValueError("family or entity capacity mismatch")
            if len(row["facts"]) != 8 or len(row["queries"]) != 1:
                raise ValueError("v1 row exposure contract mismatch")
            kinds = Counter(k for _, _, k, _ in row["facts"])
            if [kinds[k] for k in range(4)] != [2] * 4:
                raise ValueError("grounding quota mismatch")
            for s, o, k, phrase in row["facts"]:
                if not (0 <= s < row["n"] and 0 <= o < row["n"] and s != o):
                    raise ValueError("invalid source/object")
                if phrase != FAMILIES[split][row["family_id"]][k] or len(phrase.encode("utf-8")) > 48:
                    raise ValueError("phrase semantics/UTF-8 capacity mismatch")
            if oracle(row["n"], row["facts"], row["queries"]) != (row["labels"], row["hops"]):
                raise ValueError("oracle changed or stale label")
            iso = isomorphic_key(row["n"], row["facts"])
            if iso != row["isomorphic_key"] or iso in all_graphs:
                raise ValueError("entity-renamed graph leakage or stale identity")
            if graph_key(row["n"], row["facts"]) != row["graph_key"]:
                raise ValueError("typed graph identity mismatch")
            all_graphs.add(iso)
        if summarize(rows)["query_counts"] != [len(rows) // 4] * 4:
            raise ValueError("query quota mismatch")


def build(counts=None, seed=20261026, max_candidates=10000):
    counts = counts or {"train": 128, "dev": 32, "final": 32}
    occupied, splits, stats = set(), {}, {}
    for i, (split, count) in enumerate(counts.items()):
        splits[split], stats[split] = build_split(split, count, seed + i * 100000, occupied, max_candidates)
    validate(splits)
    return splits, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("candidate output must be new; preserve prior artifacts")
    splits, stats = build()
    base = Path(__file__).resolve().parents[1]
    source = base / "src" / "fdt_local" / "data.py"
    record = {"version": VERSION, "status": "CPU_DATA_CANDIDATE_NOT_TRAINED",
              "base_data_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "seed": 20261026, "families": FAMILIES, "build_stats": stats,
              "splits": {k: summarize(v) for k, v in splits.items()},
              "neural_input_keys": ["tokens", "roles", "fact_mask", "queries", "query_mask"],
              "final_selection_allowed": False, "optimizer_updates": 0,
              "limitations": ["synthetic bounded typed graphs; not natural-language corpus",
                  "one query per row; differs from baseline's up-to-16-slot exposure",
                  "same family within row; no time/scope/identity extraction task",
                  "typed isomorphism only; semantically equivalent LESS/GREATER encodings may remain",
                  "not integrated into the frozen baseline training runner"]}
    args.output.mkdir(parents=True)
    for split, rows in splits.items():
        (args.output / f"{split}.json").write_bytes(canonical_bytes(rows))
    record["files"] = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                       for p in sorted(args.output.glob("*.json"))}
    (args.output / "manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": record["status"], "splits": record["splits"], "build_stats": stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
