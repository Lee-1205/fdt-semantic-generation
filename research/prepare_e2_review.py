"""Read-only CPU validation of sealed E2 rows and a crossed development grid."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

from fdt_local.data import collate
from fdt_local.model import Config
from fdt_local.runner import source_digest

spec = importlib.util.spec_from_file_location("e2_candidate", Path(__file__).with_name("e2_data_v1.py"))
e2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2)


def path_labels(n, facts, queries):
    """Independent path enumeration, rather than Floyd-Warshall closure."""
    edges = {i: set() for i in range(n)}
    for s, o, kind, _ in facts:
        if kind == 0:
            edges[s].add(o)
        elif kind == 1:
            edges[o].add(s)
    def reachable(a, b):
        pending, visited = list(edges[a]), set()
        while pending:
            node = pending.pop()
            if node == b:
                return True
            if node not in visited:
                visited.add(node)
                pending.extend(edges[node])
        return False
    conflict = any(reachable(i, i) for i in range(n))
    conflict |= any((k == 2 and reachable(s, o)) or (k == 3 and reachable(o, s))
                    for s, o, k, _ in facts)
    return [3 if conflict else 1 if reachable(s, o) else 2 if reachable(o, s) else 0
            for s, o in queries]


def render(rows, target):
    result = copy.deepcopy(rows)
    for i, row in enumerate(result):
        language = row["family_id"].split("_")[0]
        family = next(f for f in e2.FAMILIES[target] if f.startswith(language + "_"))
        row["family_id"] = family
        row["facts"] = [(s, o, kind, e2.FAMILIES[target][family][kind])
                        for s, o, kind, _ in row["facts"]]
    return result


def crossed_grid(splits, count=16):
    if count % 8 or count > min(len(splits["train"]), len(splits["dev"])):
        raise ValueError("grid must fit full eight-row family/label blocks")
    return {
        "seen_graph_seen_family": copy.deepcopy(splits["train"][:count]),
        "seen_graph_dev_family": render(splits["train"][:count], "dev"),
        "dev_graph_seen_family": render(splits["dev"][:count], "train"),
        "dev_graph_dev_family": copy.deepcopy(splits["dev"][:count]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("review output must be new")
    manifest_path = args.candidate / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["builder_sha256"] != hashlib.sha256(Path(e2.__file__).read_bytes()).hexdigest():
        raise ValueError("builder changed since candidate construction")
    for name, expected in manifest["files"].items():
        path = args.candidate / name
        if path.name != name or path.stat().st_size != expected["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ValueError("candidate file identity mismatch")
    splits = {key: json.loads((args.candidate / f"{key}.json").read_text(encoding="utf-8"))
              for key in ("train", "dev", "final")}
    e2.validate(splits)
    panels = crossed_grid(splits)
    checked = 0
    for rows in [*splits.values(), *panels.values()]:
        for row in rows:
            if path_labels(row["n"], row["facts"], row["queries"]) != row["labels"]:
                raise ValueError("independent oracle disagreement")
            checked += 1
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            features, targets, size = collate(batch, Config())
            if set(features) != {"tokens", "roles", "fact_mask", "queries", "query_mask"}:
                raise ValueError("unexpected neural input")
            if features["fact_mask"].sum().item() != 8 * len(batch) or features["query_mask"].sum().item() != len(batch):
                raise ValueError("collated exposure mismatch")
            if targets["labels"][:, 0].tolist() != [r["labels"][0] for r in batch]:
                raise ValueError("target collation mismatch")
            if size != sum(len(p.encode("utf-8")) for row in batch for *_, p in row["facts"]):
                raise ValueError("actual byte exposure mismatch")
    # Deliberate pairing only in this evaluation grid; never train on its dev
    # wording. Final rows/families are excluded entirely from crossed panels.
    for left, right in (("seen_graph_seen_family", "seen_graph_dev_family"),
                        ("dev_graph_seen_family", "dev_graph_dev_family")):
        for a, b in zip(panels[left], panels[right]):
            if (a["isomorphic_key"], a["queries"], a["labels"]) != (b["isomorphic_key"], b["queries"], b["labels"]):
                raise ValueError("crossed panel lost its controlled graph/query")
    base = Path(__file__).resolve().parents[1]
    frozen = json.loads((base / "runs/smoke_102m/contract.json").read_text(encoding="utf-8"))["source"]
    if source_digest() != frozen:
        raise ValueError("frozen runtime source changed")
    report = {"status": "CPU_E2_CONTRACT_PASS_NOT_TRAINED", "optimizer_attempts": 0,
              "candidate_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "runtime_source_sha256": frozen, "core_rows": sum(map(len, splits.values())),
              "independent_path_label_checks": checked, "crossed_panel_rows": {k: len(v) for k, v in panels.items()},
              "final_model_evaluation": False, "final_used_for_selection": False,
              "training_runner_integration": False, "split_stats": {k: e2.summarize(v) for k, v in splits.items()},
              "crossed_grid_notice": "Evaluation-only deliberate paired graph/family reuse; not independent samples or extra training rows."}
    args.output.mkdir(parents=True)
    panel_path = args.output / "crossed_development_panels.json"
    panel_path.write_bytes(e2.canonical_bytes(panels))
    report["crossed_panel_sha256"] = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    (args.output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "core_rows": report["core_rows"],
                      "independent_path_label_checks": checked, "runtime_source_unchanged": True}))


if __name__ == "__main__":
    main()
