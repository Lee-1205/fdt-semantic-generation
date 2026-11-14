"""Verify the stored active-8 candidate independently of its generation loop."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
from fdt_local.data import collate, oracle
from fdt_local.model_masked import Config
from fdt_local.runner import source_digest
from learning_io_v2 import atomic_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--builder", type=Path, required=True)
args = parser.parse_args()
directory, builder_path = args.candidate.resolve(), args.builder.resolve()
spec = importlib.util.spec_from_file_location("candidate_reviewed", builder_path)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
candidate = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
if candidate["source_sha256"]["builder_py"] != digest(builder_path):
    raise ValueError("candidate is not bound to current builder")
if source_digest() != "eb68ced289218c5a9ab1fc0481f52106da9f8c5892cb67022e70759d24a4f352":
    raise ValueError("core source changed")
excluded, _ = builder.build_exclusions()
keys, summaries = {}, {}
for split, count in (("train", 1024), ("dev", 64)):
    path = directory / f"{split}.json"
    if digest(path) != candidate["files"][path.name]["sha256"]:
        raise ValueError("stored candidate data digest mismatch")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if len(rows) != count:
        raise ValueError("candidate quota mismatch")
    classes, canonical, physical, byte_count = Counter(), set(), set(), 0
    for row in rows:
        if row["n"] != 8 or {v for f in row["facts"] for v in f[:2]} != set(range(8)):
            raise ValueError("not all eight IDs active")
        if len(row["queries"]) != 8 or len(set(map(tuple, row["queries"]))) != 8:
            raise ValueError("query capacity/duplication error")
        reference = oracle(8, row["facts"], row["queries"])
        independently = builder.independent_oracle(8, row["facts"], row["queries"])
        if reference != independently or reference != (row["labels"], row["hops"]):
            raise ValueError("stored labels/hops disagree with frozen and independent oracle")
        key = builder.canonical_typed_graph(8, row["facts"])
        if key != row["canonical_graph_key"] or key in canonical or key in excluded:
            raise ValueError("graph family overlap or wrong canonical identity")
        bucket = int(key[:16], 16) % 10
        if (split == "train" and bucket >= 8) or (split == "dev" and bucket != 8):
            raise ValueError("abstract graph assigned to wrong split")
        canonical.add(key); physical.add(row["graph_key"]); classes.update(row["labels"])
    for start in range(0, len(rows), 16):
        features, targets, actual_bytes = collate(rows[start:start+16], Config(), "cpu")
        if set(features) != {"tokens", "roles", "fact_mask", "queries", "query_mask"}:
            raise ValueError("metadata or supervision leaked into input")
        byte_count += actual_bytes
    if [classes[i] for i in range(4)] != [count * 2] * 4 or len(physical) != count:
        raise ValueError("class quotas or distinct physical graph count wrong")
    keys[split] = canonical
    summaries[split] = {"rows": count, "distinct_abstract_graphs": len(canonical),
        "query_classes": [classes[i] for i in range(4)], "actual_input_utf8_bytes": byte_count}
if keys["train"] & keys["dev"]:
    raise ValueError("train/dev abstract graph overlap")
record = {"status": "ROOT_STORED_CPU_REVIEW_PASS", "manifest_file_sha256": digest(directory / "manifest.json"),
    "builder_sha256": digest(builder_path), "summaries": summaries, "excluded_observed_abstracts": len(excluded),
    "labels_and_shortest_hops_match": True, "all_stored_canonical_keys_reconstructed": True,
    "five_neural_inputs_only": True, "optimizer_updates": 0, "model_forwards": 0,
    "e2_final_files_accessed": False, "cpu_preparation_is_not_training": True}
atomic_json(directory / "root_cpu_review.json", record)
print(json.dumps(record, indent=2))
