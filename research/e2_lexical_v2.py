"""New lexical grounding curriculum; old development exposure is declared."""
from collections import Counter
import json
from pathlib import Path
import random

from e2_data_v1 import canonical_bytes, digest, isomorphic_key, summarize, typed_candidate
from prepare_e2_review import path_labels
from fdt_local.data import collate, graph_key, oracle
from fdt_local.model import Config

TRAIN = {
    "en_plain": ("is greater than", "is less than", "is not greater than", "is not less than"),
    "en_exceed": ("value exceeds", "value is below", "value does not exceed", "value is not below"),
    "en_size": ("is larger than", "is smaller than", "is no larger than", "is no smaller than"),
    "en_height": ("is higher than", "is lower than", "is not higher than", "is not lower than"),
    "en_compare": ("compares greater than", "compares less than", "does not compare greater than", "does not compare less than"),
    "en_numeric": ("has the greater numeric value", "has the smaller numeric value", "does not have the greater numeric value", "does not have the smaller numeric value"),
    "ko_plain": ("더 크다", "더 작다", "크지 않다", "작지 않다"),
    "ko_numeric": ("수치가 더 높다", "수치가 더 낮다", "수치가 높지는 않다", "수치가 낮지는 않다"),
}
DEV = {
    "en_value_order": ("is greater in value than", "is less in value than", "is not greater in value than", "is not less in value than"),
    "ko_numeric_order": ("수치는 더 크다", "수치는 더 작다", "수치는 더 크지 않다", "수치는 더 작지 않다"),
}


def split(families, count, seed, occupied):
    if count % (4 * len(families)):
        raise ValueError("full family/class blocks required")
    rng, rows, attempts = random.Random(seed), [], 0
    names = list(families)
    while len(rows) < count and attempts < 100000:
        attempts += 1
        n, facts, pairs, labels, hops = typed_candidate(rng)
        wanted = len(rows) % 4
        choices = [i for i, label in enumerate(labels) if label == wanted]
        if not choices:
            continue
        identity = isomorphic_key(n, facts)
        if identity in occupied:
            continue
        chosen = rng.choice(choices)
        family = names[(len(rows) // 4) % len(names)]
        rendered = [(s, o, kind, families[family][kind]) for s, o, kind, _ in facts]
        rows.append({"n": n, "facts": rendered, "queries": [pairs[chosen]], "labels": [labels[chosen]],
                     "hops": [hops[chosen]], "graph_key": graph_key(n, facts), "isomorphic_key": identity,
                     "family_id": family, "mode": "lexical_expansion_v2"})
        occupied.add(identity)
    if len(rows) != count:
        raise RuntimeError(f"bounded supply shortfall: {len(rows)}/{count}")
    return rows, attempts


def main():
    import argparse
    import hashlib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("new versioned data output required")
    root = Path(__file__).resolve().parents[1]
    old = root / "research/artifacts/e2_candidate_20261026_v1"
    # Identity-only exclusion is CPU split hygiene, not final model evaluation.
    occupied = {row["isomorphic_key"] for fold in ("train", "dev", "final")
                for row in json.loads((old / f"{fold}.json").read_text(encoding="utf-8"))}
    prior_keys = occupied.copy()
    train, train_attempts = split(TRAIN, 512, 20261027, occupied)
    dev, dev_attempts = split(DEV, 64, 20361027, occupied)
    if {p for family in TRAIN.values() for p in family} & {p for family in DEV.values() for p in family}:
        raise ValueError("literal train/dev leakage")
    for families, rows in ((TRAIN, train), (DEV, dev)):
        for row in rows:
            if row["isomorphic_key"] in prior_keys or row["labels"] != path_labels(row["n"], row["facts"], row["queries"]):
                raise ValueError("graph exclusion or independent label failed")
            if oracle(row["n"], row["facts"], row["queries"]) != (row["labels"], row["hops"]):
                raise ValueError("production oracle disagreement")
            if Counter(k for _, _, k, _ in row["facts"]) != Counter({k: 2 for k in range(4)}):
                raise ValueError("grounding exposure imbalance")
            if any(len(p.encode("utf-8")) > 48 or p != families[row["family_id"]][k]
                   for _, _, k, p in row["facts"]):
                raise ValueError("typed phrase semantics or UTF-8 capacity mismatch")
        for start in range(0, len(rows), 8):
            features, targets, _ = collate(rows[start:start + 8], Config())
            if set(features) != {"tokens", "roles", "fact_mask", "queries", "query_mask"}:
                raise ValueError("supervision leakage")
            if targets["labels"][:, 0].tolist() != [row["labels"][0] for row in rows[start:start + 8]]:
                raise ValueError("target contract mismatch")
    args.output.mkdir(parents=True)
    for fold, rows in (("train", train), ("dev", dev)):
        (args.output / f"{fold}.json").write_bytes(canonical_bytes(rows))
    manifest = {"status": "CPU_LEXICAL_CANDIDATE_NOT_TRAINED", "version": "lexical-v2",
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "families": {"train": TRAIN, "dev": DEV}, "train": summarize(train), "dev": summarize(dev),
        "candidate_attempts": {"train": train_attempts, "dev": dev_attempts},
        "old_development_families_explicitly_promoted_to_training": ["en_exceed", "ko_numeric"],
        "old_development_scores_become_seen_after_training": True,
        "final_model_evaluation": False, "final_rows_for_training": False,
        "identity_only_final_exclusion": True, "independent_path_label_checks": 576,
        "train_language_row_fraction": {"english": .75, "korean": .25},
        "not_token_or_gradient_fraction": True, "grounding_each_row_fraction": [.25] * 4,
        "stage_grounding_only_query_targets_not_supervised": True,
        "limits": ["synthetic fixed8facts/onequery/n4-5", "family vocabulary overlaps intentionally; exact templates disjoint",
                   "entity roles are provided, not learned NER", "no broad natural-language/general LM claim"]}
    manifest["files"] = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                         for p in args.output.glob("*.json")}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "train": manifest["train"], "dev": manifest["dev"],
                      "checks": 576, "promoted_prior_dev_declared": True}))


if __name__ == "__main__":
    main()
