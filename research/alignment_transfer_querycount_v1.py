"""CPU-only transfer query-count views over 16 new abstract typed graphs."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))
from fdt_local.data import candidate, collate, graph_key, oracle
from fdt_local.model_masked import Config
import role8_active_balanced_v2 as v2

SCHEMA = "fdt_alignment_transfer_querycount_v1"
STATUS = "CPU_ALIGNMENT_TRANSFER_QUERYCOUNT_V1_NOT_TRAINED"
QUERY_COUNTS = (1, 4, 8, 16)
GRAPHS_PER_STRATUM = 4
MAX_CANDIDATES = 50_000
SELECTION_SEED_BASE = 5_271_000
QUERY_SAMPLE_SEED = 20_261_027
QUERY_MIX_DIR = ROOT / "research" / "artifacts" / "role8_query_mix_20261027_v1"
V2_DIR = ROOT / "research" / "artifacts" / "role8_active_balanced_20261027_v2"
V2_EXCLUSION_COUNT = 112
V2_EXCLUSION_KEYS_SHA256 = "9cd3025c8777a508efec5b5014d41f6a8a1dcab65b8eab6535f91566de9f0792"
PINNED_FILES = {
    "query_mix_train.json": "2a0c4e3f91cc9c41b80a009433b3de41b8f401e9c439b84cb0836727f7cac6c0",
    "query_mix_dev.json": "71a6eb832119a8dc7e6532bfdd10daca328ba8779f46df08f099483ffa1f0522",
    "query_mix_manifest.json": "430dcbb4afd52d00e436817279326b559161be294ecfef352de2f2036d02dff5",
    "v2_train.json": "62b6326b251e0aeeca8a6d7eb19c513edd24f4beb3ef8a15c2276d9e97bd4628",
    "v2_dev.json": "fdb790b99b7138dc5fe15cf019b9985c98acc1cd16a3ba7176a5ed9c9d57a657",
    "v2_manifest.json": "2e67a3661f71545900934f0cd3717af58046848da3572f5ec884ca1230b0967e",
    "v2_builder.py": "5433a8d9887c2fc3630bc2b8b8bb48852c2d7cc106701841776b45c71537f0a2",
    "query_mix_builder.py": "92012dd0ae0aeb5a4ac7943cb19e1d1b28bf0b8a79e63c54c125a8afaa0ef75b",
    "frozen_data.py": "7055bb63764c3461fd7b4462873642c3369151f77750816d7a4bd27d2cb3b121",
    "model_config.py": "ef0d9e646a1fe75ae3fd36b5136ada6ee4c9f3d389b04012c615180f6cf9d4d7",
}
ALL_PAIRS = v2.all_pairs(8)
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


def _load_querymix_exclusions(root: Path = ROOT):
    files = {
        "query_mix_train.json": root / "research/artifacts/role8_query_mix_20261027_v1/train.json",
        "query_mix_dev.json": root / "research/artifacts/role8_query_mix_20261027_v1/dev.json",
        "query_mix_manifest.json": root / "research/artifacts/role8_query_mix_20261027_v1/manifest.json",
        "v2_train.json": root / "research/artifacts/role8_active_balanced_20261027_v2/train.json",
        "v2_dev.json": root / "research/artifacts/role8_active_balanced_20261027_v2/dev.json",
        "v2_manifest.json": root / "research/artifacts/role8_active_balanced_20261027_v2/manifest.json",
        "v2_builder.py": root / "research/role8_active_balanced_v2.py",
        "query_mix_builder.py": root / "research/role8_query_mix_v1.py",
        "frozen_data.py": root / "src/fdt_local/data.py",
        "model_config.py": root / "src/fdt_local/model_masked.py",
    }
    actual = {name: sha256_file(path) for name, path in files.items()}
    if actual != PINNED_FILES:
        mismatches = {key: (PINNED_FILES.get(key), value) for key, value in actual.items()
                      if PINNED_FILES.get(key) != value}
        raise ValueError(f"named historical source digest mismatch: {mismatches}")
    qmix_manifest = json.loads(files["query_mix_manifest.json"].read_text(encoding="utf-8"))
    payload = dict(qmix_manifest)
    declared = payload.pop("manifest_payload_sha256_without_self", None)
    if sha256_bytes(canonical_json_bytes(payload)) != declared:
        raise ValueError("QueryMix manifest payload hash mismatch")
    qmix_keys = set()
    qmix_rows = []
    for name in ("query_mix_train.json", "query_mix_dev.json"):
        rows = json.loads(files[name].read_text(encoding="utf-8"))
        qmix_rows.extend(rows)
    for row in qmix_rows:
        key = v2.canonical_typed_graph(8, row["facts"])
        if key != row["canonical_graph_key"]:
            raise ValueError("stored QueryMix canonical identity mismatch")
        qmix_keys.add(key)
    if len(qmix_rows) != 1088 or len(qmix_keys) != 1088:
        raise ValueError("expected 1,088 distinct stored QueryMix abstract graphs")

    v2_keys, v2_evidence = v2.build_exclusions(root)
    v2_keys_digest = sha256_bytes(canonical_json_bytes(sorted(v2_keys)))
    if len(v2_keys) != V2_EXCLUSION_COUNT:
        raise ValueError("original V2 observed exclusion set must remain exactly 112 graphs")
    # The manifest records this exclusion union's canonical-key fingerprint.
    v2_manifest = json.loads(files["v2_manifest.json"].read_text(encoding="utf-8"))
    declared_v2 = v2_manifest["selection"]["exclusions"]["exclusion_union"]["canonical_keys_sha256"]
    if declared_v2 != v2_keys_digest or declared_v2 != V2_EXCLUSION_KEYS_SHA256:
        raise ValueError("reconstructed V2 observed exclusion-set hash mismatch")
    union = qmix_keys | v2_keys
    evidence = {
        "query_mix": {"rows": len(qmix_rows), "distinct_abstract_graphs": len(qmix_keys),
            "canonical_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(qmix_keys)))},
        "original_v2_observed_exclusions": {"distinct_abstract_graphs": len(v2_keys),
            "canonical_keys_sha256": v2_keys_digest, "selection_evidence": v2_evidence},
        "named_set_union": {"distinct_abstract_graphs": len(union),
            "canonical_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(union)))},
        "claim_scope": "Novelty is only checked against these 1,088 stored QueryMix graphs and the original V2 112-graph observed exclusion union; this is not an all-history guarantee.",
        "source_file_sha256": actual,
    }
    return union, evidence


def _active_ids(facts) -> set[int]:
    return {i for s, o, *_ in facts for i in (s, o)}


def _validate_fact_schema(facts, config):
    if not 1 <= len(facts) <= config.facts:
        raise ValueError("premise fact count outside model capacity")
    if _active_ids(facts) != set(range(8)):
        raise ValueError("all eight entity IDs must participate in premise facts")
    for s, o, kind, phrase in facts:
        if not (0 <= s < 8 and 0 <= o < 8 and s != o and kind in range(4)):
            raise ValueError("invalid typed premise edge")
        if not isinstance(phrase, str) or not phrase or len(phrase.encode("utf-8")) > config.phrase_bytes:
            raise ValueError("invalid phrase or phrase byte capacity")


def select_graphs(*, max_candidates=MAX_CANDIDATES, root: Path = ROOT):
    if not 1 <= max_candidates <= MAX_CANDIDATES:
        raise ValueError("candidate scan must be bounded to 1..50,000")
    excluded, exclusion_evidence = _load_querymix_exclusions(root)
    config = Config()
    selected = {label: [] for label in range(4)}
    seen_canonical, seen_physical = set(), set()
    rejected = Counter()
    examined = 0
    selected_by_seed = {}
    for offset in range(max_candidates):
        if all(len(selected[label]) == GRAPHS_PER_STRATUM for label in range(4)):
            break
        seed = SELECTION_SEED_BASE + offset
        examined += 1
        source = candidate(seed, n_range=(8, 8), heldout=False)
        if source["n"] != 8:
            rejected["entity_count"] += 1; continue
        try:
            _validate_fact_schema(source["facts"], config)
        except ValueError:
            rejected["fact_schema_or_active_ids"] += 1; continue
        pairs = ALL_PAIRS
        floyd = oracle(8, source["facts"], pairs)
        bfs = v2.independent_oracle(8, source["facts"], pairs)
        if floyd != bfs:
            raise ValueError("frozen oracle and independent BFS disagree for candidate")
        supports = [sum(label == c for label in bfs[0]) for c in range(4)]
        feasible = [c for c in range(4) if supports[c] >= 8 and len(selected[c]) < GRAPHS_PER_STRATUM]
        if not feasible:
            rejected["no_original_sampling_stratum_with_eight_query_support"] += 1; continue
        key, stats = v2.canonical_typed_graph(8, source["facts"], return_stats=True)
        if key in excluded:
            rejected["named_historical_graph_overlap"] += 1; continue
        if key in seen_canonical:
            rejected["duplicate_new_abstract_graph"] += 1; continue
        physical_key = graph_key(8, source["facts"])
        if physical_key in seen_physical:
            rejected["duplicate_new_physical_graph"] += 1; continue
        stratum = min(feasible, key=lambda c: (len(selected[c]), c))
        record = {"seed": seed, "stratum_class": stratum,
            "all56_query_label_support": supports, "canonical_graph_key": key,
            "physical_graph_key": physical_key, "mode": source.get("mode"),
            "facts": [list(f) for f in source["facts"]],
            "canonicalization": stats}
        selected[stratum].append(record)
        seen_canonical.add(key); seen_physical.add(physical_key)
        selected_by_seed[seed] = source
    short = {label: GRAPHS_PER_STRATUM - len(selected[label]) for label in range(4)
             if len(selected[label]) != GRAPHS_PER_STRATUM}
    if short:
        raise RuntimeError(f"bounded candidate shortfall within {max_candidates}: {short}")
    records = [record for label in range(4) for record in selected[label]]
    if len({r["canonical_graph_key"] for r in records}) != 16:
        raise ValueError("selected abstract graph identities are not unique")
    return records, {"candidate_scan_seed_base": SELECTION_SEED_BASE,
        "candidate_scan_count": examined, "candidate_cap": max_candidates,
        "accepted_graphs": len(records), "accepted_per_original_target_stratum":
            [len(selected[i]) for i in range(4)], "rejections": dict(rejected),
        "stratum_assignment_rule": "round-robin among classes with at least eight of the 56 oracle pairs, preserving four graph slots per original target class",
        "exclusions": exclusion_evidence}


def _register_selected_graphs(output: Path, graph_records, selection_summary):
    registry = {"schema": "fdt_alignment_transfer_selection_registry_v1",
        "status": "REGISTERED_BEFORE_QUERY_SAMPLING",
        "selection_seed_base": SELECTION_SEED_BASE,
        "query_sampling_seed": QUERY_SAMPLE_SEED,
        "graph_count": len(graph_records), "graphs": graph_records,
        "selection_summary": selection_summary,
        "query_sampling_not_yet_applied": True}
    registry_path = output / "selection_registry.json"
    registry_path.write_bytes(canonical_json_bytes(registry) + b"\n")
    return registry, sha256_file(registry_path)


def _make_views(graph_records, registry_sha256):
    rng = random.Random(QUERY_SAMPLE_SEED)
    rows_by_q = {q: [] for q in QUERY_COUNTS}
    sampled_order = {}
    for graph_index, record in enumerate(graph_records):
        queries16 = rng.sample(ALL_PAIRS, 16)
        if len(queries16) != 16 or len(set(queries16)) != 16:
            raise ValueError("query sampler must return 16 unique ordered pairs")
        sampled_order[record["canonical_graph_key"]] = [list(q) for q in queries16]
        for q_count in QUERY_COUNTS:
            queries = queries16[:q_count]
            floyd = oracle(8, record["facts"], queries)
            bfs = v2.independent_oracle(8, record["facts"], queries)
            if floyd != bfs:
                raise ValueError("Floyd and BFS labels/hops differ on a prefix view")
            row = {"n": 8, "facts": record["facts"],
                "queries": [list(q) for q in queries], "labels": floyd[0], "hops": floyd[1],
                "graph_key": record["physical_graph_key"],
                "canonical_graph_key": record["canonical_graph_key"],
                "transfer_graph_index": graph_index, "query_count": q_count,
                "original_sampling_target_class_stratum": record["stratum_class"],
                "selection_seed": record["seed"], "query_sample_seed": QUERY_SAMPLE_SEED,
                "query_sampling": "uniform_without_replacement_from_all_56_ordered_nonself_pairs_then_prefix",
                "selection_registry_sha256": registry_sha256,
                "mode_metadata": record["mode"]}
            features, targets, _ = collate([row], Config(), "cpu")
            if set(features) != FEATURE_KEYS:
                raise ValueError("model feature whitelist changed or gold/metadata leaked")
            if int(features["query_mask"].sum()) != q_count or targets["labels"].shape != (1, 16):
                raise ValueError("collate query mask/slot contract mismatch")
            rows_by_q[q_count].append(row)
    return rows_by_q, sampled_order


def _view_summary(rows):
    label_counts = Counter()
    class_by_position = [Counter() for _ in range(16)]
    unique_graphs = set()
    fact_bytes_by_graph = {}
    phrase_max = 0
    facts_total = 0
    query_bytes = 0
    for row in rows:
        unique_graphs.add(row["canonical_graph_key"])
        fb = sum(len(phrase.encode("utf-8")) for *_, phrase in row["facts"])
        fact_bytes_by_graph[row["canonical_graph_key"]] = fb
        facts_total += len(row["facts"])
        phrase_max = max(phrase_max, max(len(phrase.encode("utf-8")) for *_, phrase in row["facts"]))
        for position, (label, (s, o)) in enumerate(zip(row["labels"], row["queries"])):
            label_counts[int(label)] += 1
            class_by_position[position][int(label)] += 1
            query_bytes += len(str(s).encode("utf-8")) + len(str(o).encode("utf-8"))
    return {"rows": len(rows), "distinct_abstract_graphs": len(unique_graphs),
        "query_count": rows[0]["query_count"] if rows else None,
        "query_label_counts": [label_counts[i] for i in range(4)],
        "query_label_counts_by_slot_position": [[counts[i] for i in range(4)] for counts in class_by_position],
        "facts_utf8_bytes_once_per_graph": sum(fact_bytes_by_graph.values()),
        "query_id_utf8_bytes": query_bytes, "premise_fact_instances": facts_total,
        "max_phrase_bytes": phrase_max,
        "fact_bytes_identical_within_graph": True}


def validate_candidate(output: Path, *, max_candidates=MAX_CANDIDATES):
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest_obj = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload = dict(manifest_obj)
    declared_manifest_payload = manifest_payload.pop("manifest_payload_sha256_without_self", None)
    if sha256_bytes(canonical_json_bytes(manifest_payload)) != declared_manifest_payload:
        raise ValueError("candidate manifest payload hash mismatch")
    if manifest_obj["source_sha256"].get("builder_py") != sha256_file(Path(__file__)):
        raise ValueError("candidate builder SHA in manifest is stale")
    files = {f"q{q:02d}.json": json.loads((output / f"q{q:02d}.json").read_text(encoding="utf-8"))
             for q in QUERY_COUNTS}
    registry_raw = (output / "selection_registry.json").read_bytes()
    if sha256_file(output / "selection_registry.json") != manifest_obj["sampling"]["selection_registry_sha256"]:
        raise ValueError("selection registry hash mismatch")
    registry = json.loads(registry_raw)
    graph_keys = {graph["canonical_graph_key"] for graph in registry["graphs"]}
    if len(graph_keys) != 16 or registry["graph_count"] != 16:
        raise ValueError("selection registry must pin 16 distinct abstract graphs")
    excluded, evidence = _load_querymix_exclusions(ROOT)
    if graph_keys & excluded:
        raise ValueError("selected graph overlaps one of the named exclusion sets")
    if manifest_obj["exclusions"] != evidence:
        raise ValueError("manifest novelty evidence differs from pinned data")
    cfg = Config()
    fact_fingerprint = {}
    for q_count in QUERY_COUNTS:
        rows = files[f"q{q_count:02d}.json"]
        if len(rows) != 16:
            raise ValueError("each query-count view must have 16 rows")
        seen_views = set()
        for row in rows:
            if row["query_count"] != q_count or row["n"] != 8:
                raise ValueError("view row has wrong graph/query count")
            if len(row["queries"]) != q_count or len(set(map(tuple, row["queries"]))) != q_count:
                raise ValueError("query view must have unique query prefix of requested length")
            _validate_fact_schema(row["facts"], cfg)
            key = row["canonical_graph_key"]
            if key not in graph_keys:
                raise ValueError("view references graph absent from registered selection")
            if key in seen_views:
                raise ValueError("duplicate graph in one query-count view")
            seen_views.add(key)
            floyd = oracle(8, row["facts"], row["queries"])
            bfs = v2.independent_oracle(8, row["facts"], row["queries"])
            if floyd != bfs or floyd != (row["labels"], row["hops"]):
                raise ValueError("stored label/hops fail original oracle and independent BFS")
            canonical = v2.canonical_typed_graph(8, row["facts"])
            if canonical != key:
                raise ValueError("canonical family id inconsistent with fact graph")
            projected = canonical_json_bytes(row["facts"])
            prior = fact_fingerprint.setdefault(key, projected)
            if prior != projected:
                raise ValueError("facts differ across query-count views for same graph")
            features, targets, _ = collate([row], cfg, "cpu")
            if set(features) != FEATURE_KEYS or any(k in features for k in ("labels", "gold", "hops", "target_class")):
                raise ValueError("gold/stratum metadata leaked into neural input")
            if int(features["query_mask"].sum()) != q_count or targets["labels"].shape != (1, 16):
                raise ValueError("collated query slot mask/capacity mismatch")
        bykey = {row["canonical_graph_key"]: row for row in rows}
        full = files["q16.json"]
        for row in full:
            prefix = bykey[row["canonical_graph_key"]]
            if prefix["queries"] != row["queries"][:q_count]:
                raise ValueError("query-count views are not identical ordered prefixes")
    if len(fact_fingerprint) != 16:
        raise ValueError("expected exactly 16 distinct abstract graphs across views")
    actual_summaries = {f"q{q:02d}": _view_summary(files[f"q{q:02d}.json"]) for q in QUERY_COUNTS}
    return {"graphs": 16, "correlated_views": 64, "view_summaries": actual_summaries,
        "exclusion_union_graphs": len(excluded), "stored_querymix_graphs": evidence["query_mix"]["distinct_abstract_graphs"],
        "all_64_rows_original_and_bfs_verified": True,
        "all_views_prefix_and_same_facts_verified": True,
        "all_features_exactly_five_and_gold_free": True,
        "novelty_scope_limited_to_named_sets": True}


def build_candidate(*, max_candidates=MAX_CANDIDATES, root: Path = ROOT):
    selected, summary = select_graphs(max_candidates=max_candidates, root=root)
    return selected, summary


def write_candidate(output: Path, *, max_candidates=MAX_CANDIDATES):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"new candidate output must not exist: {output}")
    graph_records, selection = select_graphs(max_candidates=max_candidates)
    output.mkdir(parents=True)
    registry, registry_sha = _register_selected_graphs(output, graph_records, selection)
    # Sampling begins only after the selected graph registry has been durably written.
    rows_by_q, sampled_order = _make_views(graph_records, registry_sha)
    file_summaries = {}
    for q in QUERY_COUNTS:
        name = f"q{q:02d}.json"
        path = output / name
        path.write_bytes(canonical_json_bytes(rows_by_q[q]) + b"\n")
        file_summaries[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    view_summaries = {f"q{q:02d}": _view_summary(rows_by_q[q]) for q in QUERY_COUNTS}
    source_files = {
        "builder_py": Path(__file__),
        "v2_builder_py": ROOT / "research/role8_active_balanced_v2.py",
        "query_mix_builder_py": ROOT / "research/role8_query_mix_v1.py",
        "frozen_data_py": ROOT / "src/fdt_local/data.py",
        "model_config_py": ROOT / "src/fdt_local/model_masked.py",
    }
    manifest_obj = {"schema": SCHEMA, "status": STATUS,
        "construction": {"new_abstract_graphs": 16, "strata": [0, 1, 2, 3],
            "graphs_per_stratum": [4, 4, 4, 4], "query_count_views": list(QUERY_COUNTS),
            "uniform_sample_size_per_graph": 16,
            "sample_pool": "all 56 ordered nonself pairs", "sampling_without_replacement": True,
            "views_are_ordered_prefixes_of_same_sample": True,
            "correlated_views": 64, "distinct_abstract_graphs": 16,
            "independence_claimed": False, "original_oracle_rule_changed": False,
            "model_feature_whitelist": ["tokens", "roles", "fact_mask", "queries", "query_mask"]},
        "sampling": {"selection_seed_base": SELECTION_SEED_BASE,
            "selection_candidates_examined": selection["candidate_scan_count"],
            "candidate_cap": max_candidates, "query_sample_seed": QUERY_SAMPLE_SEED,
            "selection_registry_sha256": registry_sha,
            "query_sample_order_registered_after_graph_selection": True,
            "queries_by_graph_sha256": sha256_bytes(canonical_json_bytes(sampled_order))},
        "selection": selection,
        "exclusions": selection["exclusions"],
        "views": view_summaries,
        "source_sha256": {key: sha256_file(path) for key, path in source_files.items()},
        "files": file_summaries,
        "nonclaims": ["The 64 views are correlated query-count views over 16 graphs, not 64 independent graphs.",
            "Novelty is guaranteed only against the named stored QueryMix1088 and original V2 observed exclusion union112, not all historical data.",
            "No model inference, training, optimizer, checkpoint, or E2 final holdout was accessed."],
        "model_or_checkpoint_loaded": False, "optimizer_updates": 0, "e2_final_files_accessed": False}
    manifest_obj["manifest_payload_sha256_without_self"] = sha256_bytes(canonical_json_bytes(manifest_obj))
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest_obj) + b"\n")
    report = validate_candidate(output, max_candidates=max_candidates)
    return {"status": STATUS, "manifest_payload_sha256_without_self": manifest_obj["manifest_payload_sha256_without_self"],
        "manifest_file_sha256": sha256_file(manifest_path), "data_file_sha256": file_summaries,
        "selection": selection, "validation": report}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES)
    args = parser.parse_args()
    print(json.dumps(write_candidate(args.output.resolve(), max_candidates=args.max_candidates), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
