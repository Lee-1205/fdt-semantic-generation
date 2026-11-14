"""Paired identity-renaming diagnosis; no optimizer and no final holdout use."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import copy
import json
from pathlib import Path
import time
import torch
from fdt_local.data import graph_key, make_split, oracle
from fdt_local.model_masked import Config, FDT100
from fdt_local.runner import configure_determinism, source_digest
from fdt_local.storage import CheckpointStore, ensure_space, sha256
from learning_io_v2 import atomic_json
from probe_e1_readonly import Budget, panel

ROOT = Path(__file__).resolve().parents[1]


def rename(rows, mapping):
    result = copy.deepcopy(rows)
    for row in result:
        row["original_n"] = row["n"]
        row["n"] = 8  # Unused entities add no facts or queries.
        row["facts"] = [(mapping[s], mapping[o], k, text) for s, o, k, text in row["facts"]]
        row["queries"] = [(mapping[s], mapping[o]) for s, o in row["queries"]]
        if oracle(8, row["facts"], row["queries"]) != (row["labels"], row["hops"]):
            raise ValueError("renaming changed typed truth")
        row["graph_key"] = graph_key(8, row["facts"])
    return result


def prepare(output):
    if output.exists():
        raise FileExistsError("fresh proof directory required")
    rows = make_split("test", 16, seed=60001)
    maps = {"identity": list(range(8)), "known_swap": [1, 0, 2, 3, 4, 5, 6, 7],
            "previously_untrained_ids": [6, 7, 2, 3, 4, 5, 0, 1]}
    for mapping in maps.values():
        rename(rows, mapping)
    marker = ROOT / "runs/e2_lexical_grounding_20261026_v1/model/complete_000129.json"
    plan = {"script_sha256": sha256(Path(__file__)), "core_sha256": source_digest(),
            "common_sha256": sha256(ROOT / "scripts/probe_e1_readonly.py"),
            "io_sha256": sha256(ROOT / "scripts/learning_io_v2.py"),
            "checkpoint_marker": str(marker), "checkpoint_sha256": "938b15e963ad4f9a120aed333653e02f588594c4235482960cc4164bb1d1bac4",
            "maps": maps, "graphs": 16, "graph_seed": 60001, "noise_seeds": [17, 29, 43],
            "wall_seconds": 180, "max_forward_attempts": 200, "optimizer_attempts": 0,
            "output_dir": str(output), "e2_final_used": False,
            "same_noise_query_order_tokens_and_masks": True, "oracle_invariance_cpu_pass": True}
    output.mkdir()
    atomic_json(output / "plan.json", plan)
    print(json.dumps({"status": "PREPARED_NOT_EXECUTED", "plan_sha256": sha256(output / "plan.json")}))


def run(plan_path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output = Path(plan["output_dir"])
    if (output / "clock.json").exists():
        raise FileExistsError("one fixed proof execution only")
    for path, expected in ((Path(__file__), plan["script_sha256"]),
        (ROOT / "scripts/probe_e1_readonly.py", plan["common_sha256"]),
        (ROOT / "scripts/learning_io_v2.py", plan["io_sha256"])):
        if sha256(path) != expected:
            raise ValueError("proof code changed")
    if source_digest() != plan["core_sha256"]:
        raise ValueError("core changed")
    ensure_space(output, 64 * 1024**2)
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("CUDA and12GiBfree required")
    configure_determinism()
    budget = Budget(plan["wall_seconds"], plan["max_forward_attempts"])
    atomic_json(output / "clock.json", {"started_unix": time.time(), "wall_seconds": 180, "optimizer_attempts": 0})
    marker = Path(plan["checkpoint_marker"])
    meta, payload = CheckpointStore(marker.parent).verify(marker)
    if meta["sha256"] != plan["checkpoint_sha256"]:
        raise ValueError("model identity changed")
    model = FDT100(Config(**payload["contract"]["config"]))
    model.load_state_dict(payload["model"], strict=True)
    model.requires_grad_(False).cuda().eval()
    del payload
    rows = make_split("test", plan["graphs"], seed=plan["graph_seed"])
    results = {}
    for name, mapping in plan["maps"].items():
        renamed = rename(rows, mapping)
        result = panel(model, renamed, plan["noise_seeds"], [4], budget)
        atomic_json(output / f"{name}.json", {"checkpoint_sha256": meta["sha256"], "mapping": mapping, **result})
        results[name] = {key: result[key] for key in ("graphs", "ground_accuracy", "results")}
    record = {"status": "COMPLETE", "checkpoint_sha256": meta["sha256"], "optimizer_attempts": 0,
              "forward_attempts": budget.attempted, "seconds": time.monotonic() - budget.start,
              "panels": results, "e2_final_used": False,
              "interpretation": "Paired development probe of entity-ID coding, not three independent datasets."}
    atomic_json(output / "summary.json", record)
    print(json.dumps({"status": "COMPLETE", "forwards": budget.attempted, "seconds": record["seconds"],
                      "diffusion": {k: v["results"]["ddim_4"]["accuracy"] for k, v in results.items()}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", type=Path)
    group.add_argument("--run-plan", type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.prepare.resolve())
    else:
        run(args.run_plan.resolve())
