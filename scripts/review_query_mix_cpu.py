"""Independent stored-view checks; CPU only, no new model evaluation."""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.util
import json
from fdt_local.data import collate, oracle
from fdt_local.model_masked import Config
from fdt_local.runner import source_digest
from learning_io_v2 import atomic_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research/artifacts/role8_query_mix_20261027_v1"
SOURCE = ROOT / "research/artifacts/role8_active_balanced_20261027_v2"
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
candidate = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
if digest(SOURCE / "manifest.json") != "2e67a3661f71545900934f0cd3717af58046848da3572f5ec884ca1230b0967e":
    raise ValueError("original V2 candidate changed")
if digest(ROOT / "research/role8_query_mix_v1.py") != candidate["inputs"]["source_sha256"]["builder_py"]:
    raise ValueError("query-mix builder identity mismatch")
spec = importlib.util.spec_from_file_location("independent_paths", ROOT / "research/role8_active_balanced_v2.py")
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)
base_raw = json.loads((ROOT / "runs/active_balanced_diffusion_20261027_runtime_v3/baseline/new_dev_mixed64.json").read_text(encoding="utf-8"))["raw"]
summaries, abstract = {}, {}
for split, expected in (("train", 1024), ("dev", 64)):
    path = OUT / f"{split}.json"
    if digest(path) != candidate["files"][path.name]["sha256"]:
        raise ValueError("data digest mismatch")
    old = json.loads((SOURCE / path.name).read_text(encoding="utf-8"))
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != len(old) or len(rows) != expected:
        raise ValueError("view row count mismatch")
    classes = Counter()
    for index, (a, b) in enumerate(zip(old, rows)):
        for key in ("n", "facts", "graph_key", "canonical_graph_key", "source_family_id", "split"):
            if a[key] != b[key]:
                raise ValueError(f"premise/family identity changed: {split}/{index}/{key}")
        if "target_class" in b or b["original_sampling_target_class"] != a["target_class"]:
            raise ValueError("uniform-class metadata not renamed correctly")
        if len(b["queries"]) != 16 or len(set(map(tuple, b["queries"]))) != 16:
            raise ValueError("query duplication/capacity mismatch")
        reference = oracle(8, b["facts"], b["queries"])
        if reference != independent.independent_oracle(8, b["facts"], b["queries"]) or reference != (b["labels"], b["hops"]):
            raise ValueError("label/shortest-hop disagreement")
        if split == "dev" and any(b[key] != base_raw[index][raw_key] for key, raw_key in
                                  (("facts", "facts"), ("queries", "queries"), ("labels", "gold"))):
            raise ValueError("development view no longer matches observed baseline")
        classes.update(b["labels"])
    byte_count, query_positions = 0, 0
    for start in range(0, len(rows), 16):
        features, _, count = collate(rows[start:start+16], Config(), "cpu")
        if set(features) != {"tokens", "roles", "fact_mask", "queries", "query_mask"}:
            raise ValueError("supervision/metadata leaked into neural inputs")
        if not bool((features["query_mask"].sum(1) == 16).all()):
            raise ValueError("active query mask mismatch")
        byte_count += count; query_positions += int(features["query_mask"].sum())
    abstract[split] = {r["canonical_graph_key"] for r in rows}
    summaries[split] = {"graphs": len(rows), "distinct_abstract_graphs": len(abstract[split]),
        "active_query_positions": query_positions, "actual_premise_utf8_bytes": byte_count,
        "query_class_counts": [classes[i] for i in range(4)], "new_graphs_added": 0}
if abstract["train"] & abstract["dev"]:
    raise ValueError("abstract-family split leak")
if source_digest() != "eb68ced289218c5a9ab1fc0481f52106da9f8c5892cb67022e70759d24a4f352":
    raise ValueError("frozen core source changed")
record = {"status": "ROOT_QUERY_MIX_CPU_REVIEW_PASS", "manifest_file_sha256": digest(OUT / "manifest.json"),
    "summaries": summaries, "all_facts_and_graph_identities_unchanged": True,
    "dev_facts_queries_gold_match_observed_baseline": True,
    "hops_verified_by_original_and_independent_BFS_not_stored_in_raw": True,
    "five_neural_inputs_only": True, "model_forwards": 0, "optimizer_updates": 0,
    "E2_final_holdout_accessed": False, "new_graphs_added": 0}
atomic_json(OUT / "root_cpu_review.json", record)
print(json.dumps(record, indent=2))
