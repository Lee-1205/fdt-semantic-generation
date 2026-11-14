"""Nested query-count views and a deterministic balanced-context CPU sampler."""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research")]
from fdt_local.data import collate, oracle
from fdt_local.model_masked import Config
import role8_active_balanced_v2 as v2

SCHEMA = "fdt_querycount_alignment_v1"
STATUS = "CPU_QUERYCOUNT_ALIGNMENT_V1_CANDIDATE_NOT_TRAINED"
QUERY_COUNTS = (1, 4, 8, 16)
SAMPLER_SEED = 25
BATCH_SIZE = 8
VIEWS_PER_QUERY_COUNT_PER_BATCH = 2
SOURCE_DIR = ROOT / "research/artifacts/role8_query_mix_20261027_v1"
TRANSFER_DIR = ROOT / "research/artifacts/alignment_transfer_querycount_20261027_v1"
PINNED = {
    "train.json": "2a0c4e3f91cc9c41b80a009433b3de41b8f401e9c439b84cb0836727f7cac6c0",
    "dev.json": "71a6eb832119a8dc7e6532bfdd10daca328ba8779f46df08f099483ffa1f0522",
    "manifest.json": "430dcbb4afd52d00e436817279326b559161be294ecfef352de2f2036d02dff5",
    "query_mix_builder.py": "92012dd0ae0aeb5a4ac7943cb19e1d1b28bf0b8a79e63c54c125a8afaa0ef75b",
    "v2_builder.py": "5433a8d9887c2fc3630bc2b8b8bb48852c2d7cc106701841776b45c71537f0a2",
    "data_source.py": "7055bb63764c3461fd7b4462873642c3369151f77750816d7a4bd27d2cb3b121",
    "model_config.py": "ef0d9e646a1fe75ae3fd36b5136ada6ee4c9f3d389b04012c615180f6cf9d4d7",
    "transfer_registry.json": "9802a0b00b2e912b659513163311b82656fcd3da98094677ee56dee1350b49f4",
    "transfer_manifest.json": "9f428e54ae184f5ab6f66021ba6dfa23e6fd6678ea84d8e20f46a9bf9932ae67",
    "transfer_builder.py": "5364fcdf6ade2308a60e549ec2b4fcb8bb696126aa18b122f4683c18f7bd47fa",
}
FEATURE_KEYS = {"tokens", "roles", "fact_mask", "queries", "query_mask"}


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


def read_sources():
    source_paths = {"train.json": SOURCE_DIR / "train.json", "dev.json": SOURCE_DIR / "dev.json",
        "manifest.json": SOURCE_DIR / "manifest.json",
        "query_mix_builder.py": ROOT / "research/role8_query_mix_v1.py",
        "v2_builder.py": ROOT / "research/role8_active_balanced_v2.py",
        "data_source.py": ROOT / "src/fdt_local/data.py",
        "model_config.py": ROOT / "src/fdt_local/model_masked.py",
        "transfer_registry.json": TRANSFER_DIR / "selection_registry.json",
        "transfer_manifest.json": TRANSFER_DIR / "manifest.json",
        "transfer_builder.py": ROOT / "research/alignment_transfer_querycount_v1.py"}
    actual = {name: sha256_file(path) for name, path in source_paths.items()}
    mismatch = {name: (PINNED.get(name), value) for name, value in actual.items() if PINNED.get(name) != value}
    if mismatch:
        raise ValueError(f"pinned source/transfer evidence SHA mismatch: {mismatch}")
    source_manifest = json.loads(source_paths["manifest.json"].read_text(encoding="utf-8"))
    payload = dict(source_manifest)
    declared = payload.pop("manifest_payload_sha256_without_self", None)
    if sha256_bytes(canonical_json_bytes(payload)) != declared:
        raise ValueError("source QueryMix manifest payload hash mismatch")
    for name in ("train.json", "dev.json"):
        if source_manifest["files"][name]["sha256"] != actual[name]:
            raise ValueError(f"source QueryMix manifest binding mismatch: {name}")
    transfer_registry = json.loads(source_paths["transfer_registry.json"].read_text(encoding="utf-8"))
    if len(transfer_registry.get("graphs", [])) != 16:
        raise ValueError("transfer registry must pin exactly 16 observed abstract graphs")
    excluded = {item["canonical_graph_key"] for item in transfer_registry["graphs"]}
    if len(excluded) != 16:
        raise ValueError("transfer registry canonical identities must be unique")
    train = json.loads(source_paths["train.json"].read_text(encoding="utf-8"))
    dev = json.loads(source_paths["dev.json"].read_text(encoding="utf-8"))
    if len(train) != 1024 or len(dev) != 64:
        raise ValueError("QueryMix source split sizes changed")
    return train, dev, excluded, {"source_file_sha256": actual,
        "source_manifest_payload_sha256_without_self": declared,
        "transfer_registry_graphs": len(excluded),
        "transfer_registry_canonical_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(excluded)))}


def _make_views(rows: list[dict], split: str, excluded: set[str]):
    output = {q: [] for q in QUERY_COUNTS}
    seen = set()
    for source_index, original in enumerate(rows):
        if original.get("split") != split or original.get("n") != 8:
            raise ValueError(f"source row {source_index} is not a QueryMix {split}/n8 row")
        key = original["canonical_graph_key"]
        if key in seen:
            raise ValueError(f"duplicate source graph in {split}: {key}")
        seen.add(key)
        if key != original.get("source_family_id"):
            raise ValueError("source family/canonical graph key mismatch")
        if key in excluded and split == "train":
            raise ValueError("transfer diagnostic graph leaked into training source")
        if len(original["queries"]) != 16 or len(set(map(tuple, original["queries"]))) != 16:
            raise ValueError("source QueryMix query order must contain 16 distinct pairs")
        for q in QUERY_COUNTS:
            row = copy.deepcopy(original)
            row.pop("query_count", None)
            row["queries"] = copy.deepcopy(original["queries"][:q])
            row["labels"] = copy.deepcopy(original["labels"][:q])
            row["hops"] = copy.deepcopy(original["hops"][:q])
            row["source_graph_index"] = source_index
            row["source_split"] = split
            row["query_count"] = q
            row["query_view_of_original_16"] = True
            output[q].append(row)
    return output


def _validate_view(rows: list[dict], split: str, q: int, source_rows: list[dict]):
    cfg = Config()
    counts = Counter()
    graph_keys = set()
    canonical_keys = set()
    source_facts = {i: source_rows[i]["facts"] for i in range(len(source_rows))}
    phrase_bytes_by_graph = {}
    active_positions = 0
    for offset in range(0, len(rows), 64):
        chunk = rows[offset:offset + 64]
        for row in chunk:
            index = row["source_graph_index"]
            original = source_rows[index]
            if row["query_count"] != q or row["source_split"] != split:
                raise ValueError("view row source/query-count metadata mismatch")
            if row.get("target_class") is not None:
                raise ValueError("metadata leaked into target_class")
            if row["facts"] != source_facts[index] or row["queries"] != original["queries"][:q]:
                raise ValueError("view facts/query order differ from unchanged source prefix")
            if row["labels"] != original["labels"][:q] or row["hops"] != original["hops"][:q]:
                raise ValueError("view gold/hops differ from unchanged source prefix")
            if row["canonical_graph_key"] != original["canonical_graph_key"] or row["graph_key"] != original["graph_key"]:
                raise ValueError("source graph identity changed")
            if oracle(8, row["facts"], row["queries"]) != (row["labels"], row["hops"]):
                raise ValueError("original oracle mismatch")
            if v2.independent_oracle(8, row["facts"], row["queries"]) != (row["labels"], row["hops"]):
                raise ValueError("independent BFS labels/hops mismatch")
            active = {int(entity) for s, o, *_ in row["facts"] for entity in (s, o)}
            if active != set(range(8)) or len(row["facts"]) > cfg.facts:
                raise ValueError("active-ID or fact-capacity contract mismatch")
            if any(len(str(f[3]).encode("utf-8")) > cfg.phrase_bytes for f in row["facts"]):
                raise ValueError("premise phrase exceeds byte capacity")
            graph_keys.add(row["graph_key"])
            canonical_keys.add(row["canonical_graph_key"])
            counts.update(int(label) for label in row["labels"])
            phrase_bytes_by_graph[index] = sum(len(str(f[3]).encode("utf-8")) for f in row["facts"])
            active_positions += q
        features, targets, _ = collate(chunk, cfg, "cpu")
        if set(features) != FEATURE_KEYS or features["query_mask"].sum(-1).tolist() != [q] * len(chunk):
            raise ValueError("feature whitelist or query padding mask mismatch")
        if targets["labels"].shape != (len(chunk), cfg.slots) or targets["hops"].shape != (len(chunk), cfg.slots):
            raise ValueError("target slot capacity mismatch")
    return {"rows": len(rows), "distinct_physical_graphs": len(graph_keys),
        "distinct_abstract_graphs": len(canonical_keys), "query_count": q,
        "active_query_positions_in_view": active_positions,
        "query_label_support": [counts[i] for i in range(4)],
        "premise_phrase_utf8_bytes_in_this_view": sum(phrase_bytes_by_graph.values()),
        "source_graph_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(graph_keys))),
        "canonical_graph_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(canonical_keys))),
        "feature_whitelist_and_capacity_verified": True}


def validate_views(train_source, dev_source, train_views, dev_views, excluded_transfer):
    train_keys = {row["canonical_graph_key"] for row in train_source}
    dev_keys = {row["canonical_graph_key"] for row in dev_source}
    if train_keys & dev_keys:
        raise ValueError("source train/dev split overlap")
    if train_keys & excluded_transfer:
        raise ValueError("one of the 16 transfer graphs appears in train")
    if len(train_keys) != 1024 or len(dev_keys) != 64:
        raise ValueError("source graph counts changed")
    if any(len(train_views[q]) != 1024 for q in QUERY_COUNTS) or any(len(dev_views[q]) != 64 for q in QUERY_COUNTS):
        raise ValueError("nested view row counts mismatch")
    train_summary = {str(q): _validate_view(train_views[q], "train", q, train_source) for q in QUERY_COUNTS}
    dev_summary = {str(q): _validate_view(dev_views[q], "dev", q, dev_source) for q in QUERY_COUNTS}
    # Preserve the original 16-query data row payload except for explicit view metadata.
    for split, source, views in (("train", train_source, train_views), ("dev", dev_source, dev_views)):
        for index, row in enumerate(views[16]):
            original = source[index]
            projected = {key: value for key, value in row.items()
                if key not in {"source_graph_index", "source_split", "query_count", "query_view_of_original_16"}}
            if projected != original:
                raise ValueError(f"Q16 source row changed beyond new metadata: {split}[{index}]")
    return {"train": train_summary, "dev": dev_summary,
        "train_source_graphs": 1024, "dev_source_graphs": 64,
        "train_views": 4096, "dev_views": 256,
        "train_distinct_base_queries": 1024 * 16,
        "dev_distinct_base_queries": 64 * 16,
        "train_active_query_positions_across_prefix_views": 1024 * sum(QUERY_COUNTS),
        "dev_active_query_positions_across_prefix_views": 64 * sum(QUERY_COUNTS),
        "train_physical_premise_phrase_utf8_bytes_once": train_summary["16"]["premise_phrase_utf8_bytes_in_this_view"],
        "dev_physical_premise_phrase_utf8_bytes_once": dev_summary["16"]["premise_phrase_utf8_bytes_in_this_view"],
        "train_view_repeated_premise_phrase_utf8_bytes": sum(v["premise_phrase_utf8_bytes_in_this_view"] for v in train_summary.values()),
        "dev_view_repeated_premise_phrase_utf8_bytes": sum(v["premise_phrase_utf8_bytes_in_this_view"] for v in dev_summary.values()),
        "transfer_graphs_excluded_from_train": len(excluded_transfer),
        "independence_claimed": False}


class BalancedQueryCountBatchSampler:
    """Finite deterministic epoch: 8 unique source graphs, two rows per Q count."""
    schema = "fdt_querycount_alignment_sampler_state_v1"

    def __init__(self, train_views: dict[int, list[dict]], *, seed: int = SAMPLER_SEED, state=None):
        self.views = train_views
        self.seed = int(seed)
        self.graph_ids = sorted({row["source_graph_index"] for row in train_views[16]})
        if len(self.graph_ids) != 1024:
            raise ValueError("sampler requires the 1024 fixed train source graphs")
        self.lookup = {q: {row["source_graph_index"]: row for row in train_views[q]} for q in QUERY_COUNTS}
        if any(set(table) != set(self.graph_ids) for table in self.lookup.values()):
            raise ValueError("query-count view graph indexes are not identical")
        if state is None:
            self.rng = random.Random(self.seed)
            self.graph_order = list(self.graph_ids)
            self.rng.shuffle(self.graph_order)
            self.cursor = 0
        else:
            self._restore(state)

    def next_batch(self):
        if self.cursor >= len(self.graph_order):
            return None
        if self.cursor + BATCH_SIZE > len(self.graph_order):
            raise RuntimeError("sampler ended with an incomplete batch")
        graph_ids = self.graph_order[self.cursor:self.cursor + BATCH_SIZE]
        self.cursor += BATCH_SIZE
        q_counts = [1, 1, 4, 4, 8, 8, 16, 16]
        self.rng.shuffle(q_counts)
        batch = [self.lookup[q][graph_id] for graph_id, q in zip(graph_ids, q_counts)]
        if len({row["source_graph_index"] for row in batch}) != 8:
            raise AssertionError("sampler emitted duplicate source graph in a batch")
        if Counter(row["query_count"] for row in batch) != Counter({1: 2, 4: 2, 8: 2, 16: 2}):
            raise AssertionError("sampler emitted unbalanced query-count contexts")
        return batch

    def state_dict(self):
        return {"schema": self.schema, "seed": self.seed, "source_graph_count": len(self.graph_ids),
            "source_graph_indices_sha256": sha256_bytes(canonical_json_bytes(self.graph_ids)),
            "graph_order": list(self.graph_order), "cursor": self.cursor,
            "rng_state": _json_tree(self.rng.getstate())}

    @classmethod
    def from_state(cls, train_views, state):
        return cls(train_views, seed=int(state["seed"]), state=state)

    def _restore(self, state):
        if state.get("schema") != self.schema or int(state.get("seed", -1)) != self.seed:
            raise ValueError("sampler state schema/seed mismatch")
        if int(state.get("source_graph_count", -1)) != len(self.graph_ids):
            raise ValueError("sampler state source graph count mismatch")
        if state.get("source_graph_indices_sha256") != sha256_bytes(canonical_json_bytes(self.graph_ids)):
            raise ValueError("sampler state source graph identity mismatch")
        order = [int(i) for i in state.get("graph_order", [])]
        if len(order) != len(self.graph_ids) or set(order) != set(self.graph_ids):
            raise ValueError("sampler graph order is not a full source-graph permutation")
        cursor = int(state.get("cursor", -1))
        if cursor < 0 or cursor > len(order) or cursor % BATCH_SIZE:
            raise ValueError("sampler cursor must be a batch boundary")
        self.graph_order = order
        self.cursor = cursor
        self.rng = random.Random(self.seed)
        self.rng.setstate(_tuple_tree(state["rng_state"]))


def _json_tree(value):
    if isinstance(value, tuple):
        return [_json_tree(v) for v in value]
    if isinstance(value, list):
        return [_json_tree(v) for v in value]
    return value


def _tuple_tree(value):
    if isinstance(value, list):
        return tuple(_tuple_tree(v) for v in value)
    return value


def write_candidate(output: Path):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing query-count artifact: {output}")
    train_source, dev_source, excluded, source_evidence = read_sources()
    train_views = _make_views(train_source, "train", excluded)
    dev_views = _make_views(dev_source, "dev", excluded)
    summary = validate_views(train_source, dev_source, train_views, dev_views, excluded)
    output.mkdir(parents=True)
    files = {}
    for split, views in (("train", train_views), ("dev", dev_views)):
        for q in QUERY_COUNTS:
            name = f"{split}_q{q:02d}.json"
            path = output / name
            path.write_bytes(canonical_json_bytes(views[q]) + b"\n")
            files[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    source_files = {"builder_py": Path(__file__),
        "query_mix_builder_py": ROOT / "research/role8_query_mix_v1.py",
        "v2_builder_py": ROOT / "research/role8_active_balanced_v2.py",
        "data_source_py": ROOT / "src/fdt_local/data.py",
        "model_config_py": ROOT / "src/fdt_local/model_masked.py",
        "query_mix_manifest_json": SOURCE_DIR / "manifest.json",
        "transfer_registry_json": TRANSFER_DIR / "selection_registry.json",
        "transfer_manifest_json": TRANSFER_DIR / "manifest.json"}
    manifest = {"schema": SCHEMA, "status": STATUS,
        "construction": {"source_split_graphs_unchanged": True,
            "new_abstract_graphs": 0, "query_count_views": list(QUERY_COUNTS),
            "train_source_graphs": 1024, "train_view_rows": 4096,
            "dev_source_graphs": 64, "dev_view_rows": 256,
            "views_are_prefixes_of_original_uniform_16_query_order": True,
            "correlated_views_not_independent": True, "independence_claimed": False,
            "feature_whitelist": ["tokens", "roles", "fact_mask", "queries", "query_mask"],
            "transfer_graphs_excluded_from_train": len(excluded),
            "transfer_registry_facts_or_gold_copied": False},
        "sampler": {"seed": SAMPLER_SEED, "batch_size": BATCH_SIZE,
            "views_per_query_count_per_batch": 2,
            "graphs_per_batch": 8, "source_graphs_unique_within_batch": True,
            "graph_visits_per_epoch": 1, "batches_per_epoch": 128,
            "gold_driven_selection": False, "state_serializable": True},
        "inputs": source_evidence, "summary": summary,
        "files": files, "source_sha256": {key: sha256_file(path) for key, path in source_files.items()},
        "optimizer_updates": 0, "model_or_checkpoint_loaded": False, "e2_final_files_accessed": False,
        "nonclaims": ["4096 train rows are four correlated views over 1024 source graphs, not 4096 independent graphs.",
            "256 dev rows are four correlated views over 64 source graphs.",
            "Query-count views inherit source query order and are not class-balanced by construction.",
            "The 16 transfer diagnostic graph facts and gold are not copied into training rows.",
            "UTF-8 byte counts cover premise phrases and query entity IDs as reported; they are not LM token counts."]}
    manifest["manifest_payload_sha256_without_self"] = sha256_bytes(canonical_json_bytes(manifest))
    manifest_path = output / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    # Read back the committed candidate and validate the complete stored artifact.
    stored_train = {q: json.loads((output / f"train_q{q:02d}.json").read_text(encoding="utf-8")) for q in QUERY_COUNTS}
    stored_dev = {q: json.loads((output / f"dev_q{q:02d}.json").read_text(encoding="utf-8")) for q in QUERY_COUNTS}
    stored_summary = validate_views(train_source, dev_source, stored_train, stored_dev, excluded)
    if stored_summary != summary:
        raise ValueError("stored candidate summaries differ from assembled rows")
    return {"status": STATUS, "manifest_payload_sha256_without_self": manifest["manifest_payload_sha256_without_self"],
        "manifest_file_sha256": sha256_file(manifest_path), "files": files,
        "summary": stored_summary, "source_sha256": manifest["source_sha256"],
        "transfer_registry_graphs_excluded_from_train": len(excluded)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(write_candidate(args.output.resolve()), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
