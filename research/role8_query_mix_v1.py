"""CPU-only query resampling over the frozen Role8 active-balanced graphs.

This creates no new abstract graphs: it holds all V2 facts and split identities
fixed, and samples 16 distinct ordered nonself queries uniformly from all 56
available pairs per row. It never loads a model or reads an E2 final holdout.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))
from fdt_local.data import collate, oracle
from fdt_local.model_masked import Config
import role8_active_balanced_v2 as v2

SCHEMA = "fdt_role8_query_mix_v1"
STATUS = "CPU_ROLE8_QUERY_MIX_V1_CANDIDATE_NOT_TRAINED"
QUERY_COUNT = 16
ALL_PAIRS = [(s, o) for s in range(8) for o in range(8) if s != o]
TRAIN_SEED = 33_017
DEV_SEED = 22_017
V2_DIR = ROOT / "research" / "artifacts" / "role8_active_balanced_20261027_v2"
V3_BASELINE = ROOT / "runs" / "active_balanced_diffusion_20261027_runtime_v3" / "baseline" / "new_dev_mixed64.json"
V2_PINNED = {
    "train.json": "62b6326b251e0aeeca8a6d7eb19c513edd24f4beb3ef8a15c2276d9e97bd4628",
    "dev.json": "fdb790b99b7138dc5fe15cf019b9985c98acc1cd16a3ba7176a5ed9c9d57a657",
    "manifest.json": "2e67a3661f71545900934f0cd3717af58046848da3572f5ec884ca1230b0967e",
    "builder.py": "5433a8d9887c2fc3630bc2b8b8bb48852c2d7cc106701841776b45c71537f0a2",
}
FEATURE_KEYS = {"tokens", "roles", "fact_mask", "queries", "query_mask"}


def canonical_json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _facts_projection(rows):
    return [[row["graph_key"], row["canonical_graph_key"], row["facts"]] for row in rows]


def sample_split(rows: list[dict], *, seed: int, split: str,
                 baseline_raw: list[dict] | None = None) -> list[dict]:
    """Uniform per-row sample without replacement; preserve row order and all facts."""
    rng = random.Random(seed)
    out = []
    raw_by_graph = None
    if baseline_raw is not None:
        raw_by_graph = {str(row["graph_key"]): row for row in baseline_raw}
        if len(raw_by_graph) != len(baseline_raw):
            raise ValueError("baseline panel contains duplicate graph keys")
    for index, original in enumerate(rows):
        if original.get("n") != 8 or "target_class" not in original:
            raise ValueError("expected frozen V2 active-eight source rows")
        row = copy.deepcopy(original)
        row["original_sampling_target_class"] = int(row.pop("target_class"))
        queries = rng.sample(ALL_PAIRS, QUERY_COUNT)
        labels_floyd, hops_floyd = oracle(8, row["facts"], queries)
        labels_bfs, hops_bfs = v2.independent_oracle(8, row["facts"], queries)
        if (labels_floyd, hops_floyd) != (labels_bfs, hops_bfs):
            raise ValueError(f"original/BFS oracle mismatch in {split} row {index}")
        row["queries"] = [list(pair) for pair in queries]
        row["labels"] = labels_floyd
        row["hops"] = hops_floyd
        row["query_sample_seed"] = seed
        row["query_sample_index"] = index
        row["query_sampling"] = "uniform_without_replacement_from_all_56_ordered_nonself_pairs"
        if raw_by_graph is not None:
            baseline = raw_by_graph.get(str(row["graph_key"]))
            if baseline is None:
                raise ValueError("baseline graph keys do not match frozen V2 dev rows")
            if (baseline.get("facts") != row["facts"]
                    or baseline.get("queries") != row["queries"]
                    or baseline.get("gold") != labels_floyd
                    or int(baseline.get("seed", -1)) != 17):
                raise ValueError(f"new dev row differs from runtime v3 baseline raw row {index}")
            # The panel's raw payload omits hops; reconstruct them through both
            # independent implementations on those exact baseline facts/queries.
            baseline_hops = oracle(8, baseline["facts"], baseline["queries"])[1]
            if baseline_hops != hops_floyd or v2.independent_oracle(
                    8, baseline["facts"], baseline["queries"])[1] != baseline_hops:
                raise ValueError("baseline exact-pair hop reconstruction mismatch")
        out.append(row)
    if raw_by_graph is not None and set(raw_by_graph) != {str(row["graph_key"]) for row in rows}:
        raise ValueError("baseline panel includes graphs outside frozen dev split")
    return out


def validate_rows(rows: list[dict], *, split: str) -> dict:
    cfg = Config()
    query_class = Counter()
    facts_by_id = Counter()
    queries_by_id = Counter()
    facts_bytes = queries_bytes = 0
    fact_projection = []
    seen_graphs, seen_abstract = set(), set()
    for index, row in enumerate(rows):
        if row.get("n") != 8 or not 1 <= len(row["facts"]) <= cfg.facts:
            raise ValueError(f"{split}[{index}] entity/fact capacity mismatch")
        if len(row["queries"]) != QUERY_COUNT or len(set(map(tuple, row["queries"]))) != QUERY_COUNT:
            raise ValueError(f"{split}[{index}] requires 16 distinct query pairs")
        if row.get("target_class") is not None or not isinstance(row.get("original_sampling_target_class"), int):
            raise ValueError("source target class must be renamed metadata, not a current target")
        if row["graph_key"] in seen_graphs or row["canonical_graph_key"] in seen_abstract:
            raise ValueError(f"{split} contains duplicate physical/abstract graphs")
        seen_graphs.add(row["graph_key"])
        seen_abstract.add(row["canonical_graph_key"])
        if row.get("source_family_id") != row["canonical_graph_key"]:
            raise ValueError("canonical family identity changed")
        if int(row["original_sampling_target_class"]) not in range(4):
            raise ValueError("invalid original V2 sampling-target metadata")
        active = set()
        for s, o, kind, phrase in row["facts"]:
            if not (0 <= s < 8 and 0 <= o < 8 and s != o and kind in range(4)
                    and isinstance(phrase, str) and len(phrase.encode("utf-8")) <= cfg.phrase_bytes):
                raise ValueError("invalid fact/phrase capacity")
            active.update((s, o))
            facts_by_id[s] += 1; facts_by_id[o] += 1
            facts_bytes += len(phrase.encode("utf-8"))
        if active != set(range(8)):
            raise ValueError("all eight entity IDs must remain active in facts")
        for s, o in row["queries"]:
            if not (0 <= s < 8 and 0 <= o < 8 and s != o):
                raise ValueError("invalid query pair")
            queries_by_id[s] += 1; queries_by_id[o] += 1
            queries_bytes += len(str(s).encode("utf-8")) + len(str(o).encode("utf-8"))
        labels_floyd, hops_floyd = oracle(8, row["facts"], row["queries"])
        labels_bfs, hops_bfs = v2.independent_oracle(8, row["facts"], row["queries"])
        if (labels_floyd, hops_floyd) != (row["labels"], row["hops"]):
            raise ValueError("stored row truth/hops disagree with original Floyd oracle")
        if (labels_bfs, hops_bfs) != (row["labels"], row["hops"]):
            raise ValueError("stored row truth/hops disagree with independent BFS")
        query_class.update(labels_floyd)
        fact_projection.append([row["graph_key"], row["canonical_graph_key"], row["facts"]])
    features, targets, _ = collate(rows, cfg, "cpu")
    if set(features) != FEATURE_KEYS or features["query_mask"].sum(-1).tolist() != [QUERY_COUNT] * len(rows):
        raise ValueError("neural input is not exactly the five supported features or query capacity is exceeded")
    if targets["labels"].shape[1] != cfg.slots:
        raise ValueError("supervision slot capacity mismatch")
    if any(facts_by_id[i] == 0 or queries_by_id[i] == 0 for i in range(8)):
        raise ValueError("entity ID exposure is incomplete")
    return {"graphs": len(rows), "distinct_abstract_graphs": len(seen_abstract),
        "query_label_counts": [query_class[i] for i in range(4)],
        "fact_entity_exposure": [facts_by_id[i] for i in range(8)],
        "query_entity_exposure": [queries_by_id[i] for i in range(8)],
        "facts_utf8_bytes": facts_bytes, "query_id_utf8_bytes": queries_bytes,
        "max_phrase_bytes": max(len(phrase.encode("utf-8")) for row in rows for *_, phrase in row["facts"]),
        "graph_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(seen_graphs))),
        "canonical_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(seen_abstract))),
        "facts_projection_sha256": sha256_bytes(canonical_json_bytes(fact_projection))}


def read_v2_inputs():
    manifest_path = V2_DIR / "manifest.json"
    if not V3_BASELINE.exists():
        raise FileNotFoundError(f"required prior v3 baseline panel missing: {V3_BASELINE}")
    paths = {name: V2_DIR / name for name in ("train.json", "dev.json", "manifest.json")}
    actual_hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in V2_PINNED.items():
        if name in actual_hashes and actual_hashes[name] != expected:
            raise ValueError(f"pinned V2 {name} SHA mismatch")
    builder_path = ROOT / "research" / "role8_active_balanced_v2.py"
    if sha256_file(builder_path) != V2_PINNED["builder.py"]:
        raise ValueError("pinned V2 builder SHA mismatch")
    v2_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    v2_manifest_payload = dict(v2_manifest)
    declared = v2_manifest_payload.pop("manifest_payload_sha256_without_self", None)
    if sha256_bytes(canonical_json_bytes(v2_manifest_payload)) != declared:
        raise ValueError("V2 manifest self-excluding payload digest mismatch")
    if v2_manifest["files"]["train.json"]["sha256"] != actual_hashes["train.json"]:
        raise ValueError("V2 manifest train binding mismatch")
    if v2_manifest["files"]["dev.json"]["sha256"] != actual_hashes["dev.json"]:
        raise ValueError("V2 manifest dev binding mismatch")
    train = json.loads(paths["train.json"].read_text(encoding="utf-8"))
    dev = json.loads(paths["dev.json"].read_text(encoding="utf-8"))
    baseline = json.loads(V3_BASELINE.read_text(encoding="utf-8"))
    if baseline.get("graphs") != 64 or baseline.get("noise_seeds") != [17] or len(baseline.get("raw", [])) != 64:
        raise ValueError("runtime v3 baseline is not the seed17 raw 64-row panel")
    return train, dev, v2_manifest, actual_hashes, baseline


def write_candidate(output: Path) -> dict:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"new query-mix output must not exist: {output}")
    train, dev, v2_manifest, v2_hashes, baseline = read_v2_inputs()
    if len(train) != 1024 or len(dev) != 64:
        raise ValueError("V2 split counts mismatch")
    train_out = sample_split(train, seed=TRAIN_SEED, split="train")
    dev_out = sample_split(dev, seed=DEV_SEED, split="dev", baseline_raw=baseline["raw"])
    train_summary = validate_rows(train_out, split="train")
    dev_summary = validate_rows(dev_out, split="dev")
    original_train_facts = sha256_bytes(canonical_json_bytes(_facts_projection(train)))
    original_dev_facts = sha256_bytes(canonical_json_bytes(_facts_projection(dev)))
    if (train_summary["facts_projection_sha256"] != original_train_facts
            or dev_summary["facts_projection_sha256"] != original_dev_facts):
        raise ValueError("fact/canonical family/split identity changed from V2")
    train_keys = {row["canonical_graph_key"] for row in train_out}
    dev_keys = {row["canonical_graph_key"] for row in dev_out}
    if train_keys != {row["canonical_graph_key"] for row in train} or dev_keys != {row["canonical_graph_key"] for row in dev}:
        raise ValueError("query-mix data must preserve V2 canonical graph identities exactly")
    if train_keys & dev_keys:
        raise ValueError("V2 train/dev canonical split leak")
    output.mkdir(parents=True)
    files = {}
    for name, rows in (("train.json", train_out), ("dev.json", dev_out)):
        path = output / name
        path.write_bytes(canonical_json_bytes(rows) + b"\n")
        files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    source_paths = {
        "builder_py": Path(__file__),
        "v2_builder_py": ROOT / "research" / "role8_active_balanced_v2.py",
        "frozen_data_py": ROOT / "src" / "fdt_local" / "data.py",
        "model_config_py": ROOT / "src" / "fdt_local" / "model_masked.py",
        "v2_manifest_json": V2_DIR / "manifest.json",
        "v3_baseline_raw_json": V3_BASELINE,
    }
    baseline_projection = [[r["graph_key"], r["facts"], r["queries"], r["gold"]] for r in baseline["raw"]]
    manifest = {
        "schema": SCHEMA,
        "status": STATUS,
        "construction": {
            "facts_and_abstract_graphs_unchanged_from_v2": True,
            "new_abstract_graphs_added": 0,
            "train_rows": 1024,
            "dev_rows": 64,
            "query_count_per_row": QUERY_COUNT,
            "query_candidate_pool": "all 56 ordered nonself pairs",
            "uniform_without_replacement": True,
            "random_query_label_assignment": False,
            "source_target_class_renamed_to_metadata": "original_sampling_target_class",
            "feature_whitelist": ["tokens", "roles", "fact_mask", "queries", "query_mask"],
            "oracle_global_conflict_rule_changed": False,
            "independence_claimed": False,
        },
        "sampling": {"train_seed": TRAIN_SEED, "dev_seed": DEV_SEED,
            "train_rng": "one deterministic Python Random stream over source row order",
            "dev_rng": "one deterministic Python Random stream over source row order",
            "selection": "random.sample(all_56_pairs, 16) for each row"},
        "inputs": {"v2_files": v2_hashes,
            "v2_builder_sha256": V2_PINNED["builder.py"],
            "v2_manifest_payload_sha256_without_self": v2_manifest["manifest_payload_sha256_without_self"],
            "v3_baseline_raw_sha256": sha256_file(V3_BASELINE),
            "v3_baseline_seed": 17,
            "v3_baseline_rows": len(baseline["raw"]),
            "v3_baseline_facts_queries_gold_exact": True,
            "baseline_stored_hops": False,
            "baseline_hops_reconstructed_by_original_and_bfs": True,
            "baseline_projection_sha256": sha256_bytes(canonical_json_bytes(baseline_projection)),
            "source_sha256": {key: sha256_file(path) for key, path in source_paths.items()}},
        "checks": {"train_facts_projection_sha256_v2_equivalent": original_train_facts,
            "dev_facts_projection_sha256_v2_equivalent": original_dev_facts,
            "train_dev_canonical_disjoint": True,
            "all_sampled_labels_hops_match_floyd_and_bfs": True,
            "all_ids_active_in_facts_and_queries": True,
            "features_exactly_whitelisted": True,
            "max_facts": Config().facts, "max_phrase_bytes": Config().phrase_bytes,
            "query_slots": Config().slots},
        "summary": {"train": train_summary, "dev": dev_summary},
        "files": files,
        "nonclaims": ["No additional abstract graphs were generated.",
            "Query rows are not statistically independent from premise graphs or other queries.",
            "No training, optimizer, model, or final holdout was accessed."],
        "e2_final_files_accessed": False,
        "model_or_checkpoint_loaded": False,
        "optimizer_updates": 0,
    }
    manifest["manifest_payload_sha256_without_self"] = sha256_bytes(canonical_json_bytes(manifest))
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return {"status": STATUS,
        "manifest_payload_sha256_without_self": manifest["manifest_payload_sha256_without_self"],
        "manifest_file_sha256": sha256_file(manifest_path),
        "train_sha256": files["train.json"]["sha256"],
        "dev_sha256": files["dev.json"]["sha256"],
        "train": train_summary, "dev": dev_summary}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_candidate(args.output.resolve()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
