"""Bounded query-count context alignment, one finite epoch from QueryMix weights."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from collections import Counter
from itertools import combinations
import copy
import ctypes
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
import traceback
import numpy as np
import torch
import yaml
from fdt_local.data import collate, graph_key, make_split, manifest, oracle
from fdt_local.model_masked import Config
from fdt_local.runner import Engine, configure_determinism, environment, parameter_digest, source_digest
from fdt_local.storage import CheckpointStore, ensure_space
from fdt_local.objectives import diffusion_loss
import learning_io_v2 as io
import run_e2_grounding_20261026 as common
from probe_e1_readonly import Budget, panel, compute_metrics
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
from query_bound_refiner_v1 import QueryBoundRefiner

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import role8_query_mix_v1 as query_mix_data
import querycount_alignment_v1 as querycount_data
TRAINABLE_PREFIXES = ("slot_router.", "memory_project.", "rel_left", "rel_right",
    "rel_score.", "rel_message.", "time.", "refiner.", "refine_norm.", "velocity.")
TARGET_UPDATES, MAX_ATTEMPTS, SEED = 128, 128, 25
DIAGNOSTIC_STEPS = (32, 64, 96, 128)
FEATURE_KEYS = {"tokens", "roles", "fact_mask", "queries", "query_mask"}
EXPOSURE_KEYS = ("fact_source", "fact_object", "query_source", "query_object")
CONTROL_RETENTION_PANELS = ("uniform_dev64", "old_role_dev128", "old_normal16", "old_large16")
RELATIVE_RETENTION_PANELS = ("primary_querymix_dev64", "paired_new_ids16")
PARENT_SHA256 = "a877f316d0d97f2c497497fd9b941184463238055906d7109d3300db4c253b36"
sha256 = common.sha256
state_digest = common.state_digest


def is_trainable(name):
    return name.startswith(TRAINABLE_PREFIXES)


def configure_trainables(model):
    chosen = []
    for name, value in model.named_parameters():
        value.grad = None
        value.requires_grad_(is_trainable(name))
        if value.requires_grad:
            chosen.append((name, value))
    if not all(any(name.startswith(p) for name, _ in chosen) for p in TRAINABLE_PREFIXES):
        raise ValueError("incomplete phase2 trainable whitelist")
    return chosen


def frozen_digest(model):
    return state_digest({k: v for k, v in model.state_dict().items() if not is_trainable(k)})


def validate_fresh_runtime_state(optimizer_state, optimizer_ids, expected_ids, base_lrs, scaler_state):
    """Enabled GradScaler has default state even before its first update."""
    if optimizer_state or optimizer_ids != expected_ids or any(lr != 3e-5 for lr in base_lrs):
        raise ValueError("fresh phase2 AdamW/ownership/LR mismatch")
    expected_scaler = {"scale": 1024.0, "growth_factor": 2.0, "backoff_factor": .5,
                       "growth_interval": 2000, "_growth_tracker": 0}
    if scaler_state != expected_scaler:
        raise ValueError("fresh enabled GradScaler defaults mismatch")


def count_batch_exposure(rows, batches):
    result = {key: [0] * 8 for key in EXPOSURE_KEYS}
    for batch in batches:
        for index in batch:
            row = rows[int(index)]
            for s, o, *_ in row["facts"]:
                result["fact_source"][s] += 1; result["fact_object"][o] += 1
            for s, o in row["queries"]:
                result["query_source"][s] += 1; result["query_object"][o] += 1
    return result


def flatten_batch_indices(batch_indices):
    """Engine.step stores one index list per microbatch."""
    return [int(index) for microbatch in batch_indices for index in microbatch]


def count_training_exposure(rows, batch_indices):
    indices = flatten_batch_indices(batch_indices)
    id_counts = count_batch_exposure(rows, [indices])
    class_counts = [0] * 4
    graph_counts = Counter(int(rows[index]["source_graph_index"]) for index in indices)
    for index in indices:
        for label in rows[index]["labels"]:
            class_counts[int(label)] += 1
    return id_counts, class_counts, graph_counts


def update_training_exposure(runtime, rows, batch_indices, successful):
    id_counts, class_counts, graph_counts = count_training_exposure(rows, batch_indices)
    query_counts = Counter(str(int(rows[index]["query_count"])) for index in flatten_batch_indices(batch_indices))
    for key, counts in id_counts.items():
        for index, count in enumerate(counts):
            runtime["attempted_id_exposure"][key][index] += count
            if successful:
                runtime["successful_id_exposure"][key][index] += count
    for index, count in enumerate(class_counts):
        runtime["attempted_query_class_exposure"][index] += count
        if successful:
            runtime["successful_query_class_exposure"][index] += count
    if successful:
        for index, count in graph_counts.items():
            runtime["unique_graph_exposure"][index] += count
    for q, count in query_counts.items():
        runtime["attempted_query_count_exposure"][q] += count
        if successful: runtime["successful_query_count_exposure"][q] += count
    return {"id_counts": id_counts, "query_class_counts": class_counts,
            "graph_counts": dict(graph_counts), "query_count_exposure": dict(query_counts)}


def metric_errors(metrics):
    """Reconstruct integer errors from the frozen compute_metrics confusion matrix."""
    confusion = metrics.get("confusion")
    queries = int(metrics.get("queries", 0))
    if not isinstance(confusion, list) or len(confusion) != 4 or any(len(row) != 4 for row in confusion):
        raise ValueError("compute_metrics confusion matrix is required")
    return queries - sum(int(confusion[i][i]) for i in range(4))


def count_balanced_q1_q4_error_rate(panel_results):
    counts = [(metric_errors(panel_results[f"querycount_dev_q{q:02d}"]["results"]["ddim_4"]),
               int(panel_results[f"querycount_dev_q{q:02d}"]["results"]["ddim_4"]["queries"])) for q in (1,4)]
    rates = [errors / max(1, queries) for errors, queries in counts]
    return sum(rates) / 2, {"q01": {"errors": counts[0][0], "queries": counts[0][1], "error_rate": rates[0]},
                          "q04": {"errors": counts[1][0], "queries": counts[1][1], "error_rate": rates[1]}}


def development_goal(baseline, step128, final_three_seed):
    baseline_rate, baseline_counts = count_balanced_q1_q4_error_rate(baseline)
    step_rate, step_counts = count_balanced_q1_q4_error_rate(step128)
    return {"baseline_seed17_count_balanced_q1_q4_error_rate": baseline_rate,
        "step128_seed17_count_balanced_q1_q4_error_rate": step_rate,
        "count_balanced_q1_q4_error_rate_halved": (step_rate <= baseline_rate / 2) if baseline_rate > 0 else None,
        "zero_error_baseline_ceiling": baseline_rate == 0,
        "baseline_q1_q4": baseline_counts, "step128_q1_q4": step_counts,
        "qdev_accuracy_at_least_99_percent": all(step128[f"querycount_dev_q{q:02d}"]["results"]["ddim_4"]["accuracy"] >= .99 for q in (1,4,8,16)),
        "step128_seed17_qdev_errors": {f"q{q:02d}": metric_errors(step128[f"querycount_dev_q{q:02d}"]["results"]["ddim_4"]) for q in (1,4,8,16)},
        "final_three_seed_querycount_panels": {f"q{q:02d}": final_three_seed[f"querycount_dev_q{q:02d}"]["results"]["ddim_4"]["accuracy"] for q in (1,4,8,16)}}


def query_class_weights(rows, device="cpu"):
    counts = Counter(label for row in rows for label in row["labels"])
    inverse = torch.tensor([counts.get(i, 0) for i in range(4)], dtype=torch.float32,
                           device=device).clamp_min(1).reciprocal()
    return (inverse / inverse.mean()).clamp(.25, 4)


def validate_query_mix_rows(rows, expected_count):
    cfg = Config()
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("query-mix row count mismatch")
    keys, abstract_keys = set(), set()
    query_classes = Counter()
    for row in rows:
        if row.get("n") != 8 or not 1 <= len(row["facts"]) <= cfg.facts:
            raise ValueError("query-mix fact/entity capacity mismatch")
        q = int(row.get("query_count", 16))
        if q not in (1,4,8,16) or len(row.get("queries", [])) != q or len(row.get("labels", [])) != q or len(row.get("hops", [])) != q:
            raise ValueError("query-count prefix length/targets mismatch")
        if len(set(map(tuple, row["queries"]))) != q or "target_class" in row:
            raise ValueError("query pairs must be unique and class targets must not be row targets")
        if row.get("original_sampling_target_class") not in range(4):
            raise ValueError("renamed original sampling metadata missing or invalid")
        if row.get("graph_key") in keys or row.get("canonical_graph_key") in abstract_keys:
            raise ValueError("duplicate query-mix physical or abstract graph")
        keys.add(row["graph_key"]); abstract_keys.add(row["canonical_graph_key"])
        if row.get("source_family_id") != row["canonical_graph_key"]:
            raise ValueError("stable query-mix abstract graph identity mismatch")
        if any(int(v) not in range(4) for v in row["labels"]):
            raise ValueError("query-mix oracle label out of range")
        query_classes.update(map(int, row["labels"]))
        active = set()
        for s, o, kind, phrase in row["facts"]:
            if not (0 <= s < 8 and 0 <= o < 8 and s != o and kind in range(4)
                    and isinstance(phrase, str) and len(phrase.encode("utf-8")) <= cfg.phrase_bytes):
                raise ValueError("invalid query-mix premise/phrase capacity")
            active.update((s, o))
        for s, o in row["queries"]:
            if not (0 <= s < 8 and 0 <= o < 8 and s != o):
                raise ValueError("invalid query-mix pair")
        if active != set(range(8)):
            raise ValueError("each of eight entities must occur in query-mix premise facts")
        if oracle(8, row["facts"], row["queries"]) != (row["labels"], row["hops"]):
            raise ValueError("stored query-mix truth/hops disagree with frozen oracle")
        if query_mix_data.v2.independent_oracle(8, row["facts"], row["queries"]) != (row["labels"], row["hops"]):
            raise ValueError("stored query-mix truth/hops disagree with independent BFS")
    features, targets, _ = collate(rows, cfg, "cpu")
    if set(features) != FEATURE_KEYS or features["query_mask"].sum(-1).tolist() != [int(r.get("query_count",16)) for r in rows]:
        raise ValueError("query-count model inputs must be exactly five features and prefix-masked")
    if targets["labels"].shape[1] != cfg.slots:
        raise ValueError("query-mix supervision slot capacity mismatch")
    exposure = count_batch_exposure(rows, [range(len(rows))])
    if any(value == 0 for key in exposure for value in exposure[key]):
        raise ValueError("query-mix dataset lacks an entity ID in exposure")
    return {"rows": len(rows), "abstract_graphs": len(abstract_keys),
            "query_class_counts": [query_classes[k] for k in range(4)],
            "entity_id_exposure": exposure}


def validate_phase2_contract(plan):
    if (plan.get("phase"), plan.get("target_successful_updates"), plan.get("max_attempts"),
            plan.get("seed")) != (2, TARGET_UPDATES, MAX_ATTEMPTS, SEED):
        raise ValueError("query-bound phase2 budget/seed mismatch")
    if (plan.get("parent_updates"), plan.get("parent_attempts")) != (256, 256):
        raise ValueError("QueryMix parent counters mismatch")
    if plan.get("parent_sha256") != PARENT_SHA256:
        raise ValueError("weights-only active-balanced parent digest mismatch")
    if (plan.get("parent_query_mix_updates"), plan.get("ancestor_active_balanced_updates"),
            plan.get("ancestor_role8_updates"), plan.get("ancestor_e2_updates"),
            plan.get("ancestor_e1_updates")) != (256, 128, 128, 128, 1440):
        raise ValueError("query-bound ancestry counters must stay distinct and exact")
    if plan.get("phase_steps") != [480, 320, 256]:
        raise ValueError("preserved phase2 cosine horizon mismatch")
    if tuple(plan.get("trainable_prefixes", ())) != TRAINABLE_PREFIXES:
        raise ValueError("phase2 trainable whitelist mismatch")
    if plan.get("parent_optimizer_imported") is not False or plan.get("parent_rng_imported") is not False:
        raise ValueError("weights-only warmstart required")
    if plan.get("e2_final_split_evaluated") is not False:
        raise ValueError("E2 final holdout forbidden")


def validate_authorization(plan, *, actual_run=False):
    if plan.get("authorized") not in (True, False):
        raise ValueError("authorized must be an explicit boolean")
    if actual_run and plan["authorized"] is not True:
        raise ValueError("actual run requires explicit plan authorization")


def preflight(plan_path, output, worker_path):
    plan = io.load_json(plan_path)
    validate_phase2_contract(plan)
    scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
    validate_fresh_runtime_state({}, set(), set(), [3e-5], scaler.state_dict())
    if Path(plan["output_dir"]).resolve() != output.resolve() or Path(plan["run_dir"]).resolve() != (output / "model").resolve():
        raise ValueError("output paths do not match the draft plan")
    common.validate_new_stage_paths(output, Path(plan["run_dir"]))
    candidate = io.load_json(plan["candidate_manifest_path"])
    expected_deps = {
        "worker_sha256": worker_path, "supervisor_sha256": ROOT / "scripts/supervise_querycount_alignment_20261027.py",
        "io_sha256": Path(io.__file__), "common_worker_sha256": Path(common.__file__),
        "probe_common_sha256": ROOT / "scripts/probe_e1_readonly.py",
        "querycount_builder_sha256": ROOT / "research/querycount_alignment_v1.py",
        "query_bound_helper_sha256": ROOT / "research/query_bound_refiner_v1.py",
        "config_sha256": Path(plan["config_path"]), "train_data_sha256": Path(plan["train_data_path"]),
        "dev_data_sha256": Path(plan["dev_data_path"]), "candidate_manifest_sha256": Path(plan["candidate_manifest_path"]),
        "parent_marker_sha256": Path(plan["parent_marker"]),
    }
    if any(plan.get(k) != sha256(path) for k, path in expected_deps.items()):
        raise ValueError("draft dependency digest mismatch")
    if candidate["schema"] != "fdt_querycount_alignment_v1" or candidate["status"] != "CPU_QUERYCOUNT_ALIGNMENT_V1_CANDIDATE_NOT_TRAINED":
        raise ValueError("unexpected query-count dataset contract")
    manifest_payload = dict(candidate)
    declared = manifest_payload.pop("manifest_payload_sha256_without_self", None)
    canonical = json.dumps(manifest_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(canonical).hexdigest() != declared or sha256(Path(plan["candidate_manifest_path"])) != plan["candidate_manifest_file_sha256"]:
        raise ValueError("query-count manifest digest mismatch")
    for split in ("train", "dev"):
        for q in (1,4,8,16):
            name = f"{split}_q{q:02d}.json"
            data_path = Path(plan["artifact_dir"]) / name
            if candidate["files"][name]["sha256"] != sha256(data_path) or candidate["files"][name]["bytes"] != data_path.stat().st_size:
                raise ValueError(f"manifest does not bind {name}")
    if source_digest() != plan["runtime_source_sha256"] or source_digest() != plan["core_sha256"]:
        raise ValueError("frozen runtime/core changed")
    if (plan["wall_seconds"], plan["reserve_seconds"], plan["target_successful_updates"], plan["max_attempts"],
        plan["seed"], plan["phase_steps"], plan["max_forward_attempts"]) != (900, 120, 128, 128, 25, [480, 320, 256], 8192):
        raise ValueError("fixed QueryCount resource/update contract mismatch")
    if plan["max_dataset_rows"] != 1024 or plan["max_dataset_queries"] != 7424 or plan["max_dataset_facts"] != 12288:
        raise ValueError("QueryCount exposure caps mismatch")
    validate_authorization(plan)
    if plan.get("e2_final_split_evaluated") is not False:
        raise ValueError("E2 final holdout forbidden")
    rows = {"train_views": {q: io.load_json(Path(plan["artifact_dir"]) / f"train_q{q:02d}.json") for q in (1,4,8,16)},
            "dev_views": {q: io.load_json(Path(plan["artifact_dir"]) / f"dev_q{q:02d}.json") for q in (1,4,8,16)}}
    source_train, source_dev, excluded, source_hashes = querycount_data.read_sources()
    rows["train"] = [row for q in (1,4,8,16) for row in rows["train_views"][q]]
    rows["dev"] = rows["dev_views"][16]
    if any(len(rows["train_views"][q]) != 1024 or len(rows["dev_views"][q]) != 64 for q in (1,4,8,16)):
        raise ValueError("stored query-count view sizes mismatch")
    for split in ("train", "dev"):
        for q in (1,4,8,16):
            validate_query_mix_rows(rows[f"{split}_views"][q], len(rows[f"{split}_views"][q]))
        base_rows = {int(r["source_graph_index"]): r for r in rows[f"{split}_views"][16]}
        for q in (1,4,8):
            q_rows = {int(r["source_graph_index"]): r for r in rows[f"{split}_views"][q]}
            if set(q_rows) != set(base_rows): raise ValueError("query-count source graph set changed")
            for graph_id, view in q_rows.items():
                base = base_rows[graph_id]
                if (view["facts"] != base["facts"] or view["queries"] != base["queries"][:q]
                        or view["labels"] != base["labels"][:q] or view["hops"] != base["hops"][:q]
                        or view["canonical_graph_key"] != base["canonical_graph_key"]):
                    raise ValueError("stored Q-prefix differs from original QueryMix order/targets")
    querycount_data.validate_views(source_train, source_dev, rows["train_views"], rows["dev_views"], excluded)
    rows["uniform_dev"] = io.load_json(Path(plan["uniform_dev_data_path"]))
    rows["old_role_dev"] = io.load_json(Path(plan["old_role_dev_path"]))
    if len(rows["uniform_dev"]) != 64 or len(rows["old_role_dev"]) != 128:
        raise ValueError("uniform and old-role retention panels must be present")
    for panel_rows, expected_q, expected_n in ((rows["uniform_dev"], 8, 64), (rows["old_role_dev"], None, 128)):
        features, _, _ = collate(panel_rows, Config(), "cpu")
        expected_masks = ([expected_q] * expected_n if expected_q is not None
                          else [len(row["queries"]) for row in panel_rows])
        if set(features) != FEATURE_KEYS or features["query_mask"].sum(-1).tolist() != expected_masks:
            raise ValueError("retention panel input contract mismatch")
    rows["transfer_views"] = {q: io.load_json(Path(plan["transfer_data_paths"][str(q)])) for q in (1,4,8,16)}
    if [len(rows["transfer_views"][q]) for q in (1,4,8,16)] != [16]*4:
        raise ValueError("transfer diagnostic panel sizes mismatch")
    transfer_paths = plan["transfer_data_paths"]
    if any(sha256(Path(transfer_paths[str(q)])) != plan[f"transfer_q{q}_sha256"] for q in (1,4,8,16)):
        raise ValueError("transfer diagnostic source digest mismatch")
    for i in range(16):
        base = rows["transfer_views"][16][i]
        for q in (1,4,8,16):
            view = rows["transfer_views"][q][i]
            if (view["facts"] != base["facts"] or view["queries"] != base["queries"][:q]
                    or view["labels"] != base["labels"][:q] or view["hops"] != base["hops"][:q]
                    or oracle(8, view["facts"], view["queries"]) != (view["labels"], view["hops"])
                    or query_mix_data.v2.independent_oracle(8, view["facts"], view["queries"])
                        != (view["labels"], view["hops"])):
                raise ValueError("transfer query-count prefix/oracle mismatch")
    for key, ref_path in plan["parent_full_reference_paths"].items():
        ref_path = Path(ref_path)
        reference = io.load_json(ref_path)
        if reference.get("noise_seeds") != [17] or "diagnostics/step256" not in ref_path.as_posix():
            raise ValueError(f"parent full reference {key} must be same-seed17 step256")
        if sha256(ref_path) != plan["parent_full_reference_sha256"][key]:
            raise ValueError(f"parent full reference digest mismatch: {key}")
    meta, payload = CheckpointStore(Path(plan["parent_marker"]).parent).verify(Path(plan["parent_marker"]))
    if (meta["sha256"] != plan["parent_sha256"]
            or meta["sha256"] != PARENT_SHA256
            or (payload["updates"], payload["attempts"]) != (256, 256)):
        raise ValueError("QueryMix parent fullstate mismatch")
    if payload["contract"]["source"] != source_digest() or payload["contract"]["variant"] != "masked":
        raise ValueError("parent source/model mismatch")
    parent_exec = payload["contract"].get("query_mix_execution", {})
    if (parent_exec.get("weights_only_warmstart"), parent_exec.get("ancestor_role8_updates"),
            parent_exec.get("ancestor_e2_updates"), parent_exec.get("ancestor_e1_updates")) != (True, 128, 128, 1440):
        raise ValueError("QueryMix parent ancestry metadata mismatch")
    del payload
    spec = yaml.safe_load(Path(plan["config_path"]).read_text(encoding="utf-8"))
    if (spec["profile"], spec["variant"], spec["start_phase"], spec["plan"]["phase_steps"],
            spec["plan"]["diffusion_lr"], spec["plan"]["seed"], spec["plan"]["micro_batch"],
            spec["plan"]["accumulation"], spec["plan"]["train_graphs"], spec["plan"]["dev_graphs"]) != ("full", "masked", 2, [480, 320, 256], 3e-5, 25, 4, 2, 1024, 64):
        raise ValueError("actual engine config mismatch")
    return plan, rows, spec


def retention_failures(baseline, current, parent_full):
    failures = []
    for key in CONTROL_RETENTION_PANELS:
        before = baseline[key]["results"]["ddim_4"]
        after = current[key]["results"]["ddim_4"]
        accuracy_floor = max(.94, before["accuracy"] - .02)
        if after["accuracy"] < accuracy_floor:
            failures.append({"panel": key, "metric": "accuracy", "value": after["accuracy"],
                "floor": accuracy_floor, "guard": "qualified_control"})
        macro_floor = before["macro_f1"] - .05
        if after["macro_f1"] < macro_floor:
            failures.append({"panel": key, "metric": "macro_f1", "value": after["macro_f1"],
                "floor": macro_floor, "guard": "qualified_control"})
        if current[key]["ground_accuracy"] < .95:
            failures.append({"panel": key, "metric": "ground_accuracy",
                "value": current[key]["ground_accuracy"], "floor": .95, "guard": "qualified_control"})
        if current[key]["results"]["initializer"]["accuracy"] < .95:
            failures.append({"panel": key, "metric": "initializer_accuracy",
                "value": current[key]["results"]["initializer"]["accuracy"],
                "floor": .95, "guard": "qualified_control"})
    for key in RELATIVE_RETENTION_PANELS:
        before = baseline[key]["results"]["ddim_4"]
        after = current[key]["results"]["ddim_4"]
        if after["accuracy"] < before["accuracy"] - .02:
            failures.append({"panel": key, "metric": "accuracy", "value": after["accuracy"],
                "floor": before["accuracy"] - .02, "guard": "relative_only"})
        if after["macro_f1"] < before["macro_f1"] - .05:
            failures.append({"panel": key, "metric": "macro_f1", "value": after["macro_f1"],
                "floor": before["macro_f1"] - .05, "guard": "relative_only"})
        if current[key]["ground_accuracy"] < .95:
            failures.append({"panel": key, "metric": "ground_accuracy",
                "value": current[key]["ground_accuracy"], "floor": .95, "guard": "relative_only"})
        if current[key]["results"]["initializer"]["accuracy"] < .95:
            failures.append({"panel": key, "metric": "initializer_accuracy",
                "value": current[key]["results"]["initializer"]["accuracy"],
                "floor": .95, "guard": "relative_only"})
    return failures


def querycount_relative_failures(baseline, current):
    failures = []
    for key in tuple(f"querycount_dev_q{q:02d}" for q in (1,4,8,16)) + tuple(f"transfer_q{q:02d}" for q in (1,4,8,16)):
        before, after = baseline[key]["results"]["ddim_4"], current[key]["results"]["ddim_4"]
        if after["accuracy"] < before["accuracy"] - .02:
            failures.append({"panel": key, "metric": "accuracy", "value": after["accuracy"], "floor": before["accuracy"]-.02, "guard": "relative_only"})
        if after["macro_f1"] < before["macro_f1"] - .05:
            failures.append({"panel": key, "metric": "macro_f1", "value": after["macro_f1"], "floor": before["macro_f1"]-.05, "guard": "relative_only"})
        if current[key]["ground_accuracy"] < .95:
            failures.append({"panel": key, "metric": "ground_accuracy", "value": current[key]["ground_accuracy"], "floor": .95, "guard": "relative_only"})
        if current[key]["results"]["initializer"]["accuracy"] < .95:
            failures.append({"panel": key, "metric": "initializer_accuracy", "value": current[key]["results"]["initializer"]["accuracy"], "floor": .95, "guard": "relative_only"})
        if current[key]["ground_accuracy"] < .95:
            failures.append({"panel": key, "metric": "ground_accuracy", "value": current[key]["ground_accuracy"], "floor": .95, "guard": "relative_only"})
        if current[key]["results"]["initializer"]["accuracy"] < .95:
            failures.append({"panel": key, "metric": "initializer_accuracy", "value": current[key]["results"]["initializer"]["accuracy"], "floor": .95, "guard": "relative_only"})
    return failures


def retention_streak(previous, baseline, current, parent_full):
    failures = retention_failures(baseline, current, parent_full) + querycount_relative_failures(baseline, current)
    return (previous + 1 if failures else 0), failures


def baseline_alignment_action(baseline, parent_full):
    """Retain legacy guards; skip SGD only if all new Q contexts have zero errors."""
    failures = retention_failures(baseline, baseline, parent_full) + querycount_relative_failures(baseline, baseline)
    if failures:
        return "STOP_BEFORE_OPTIMIZER_LEGACY_GUARD", failures
    objective = [f"querycount_dev_q{q:02d}" for q in (1,4,8,16)] + [f"transfer_q{q:02d}" for q in (1,4,8,16)]
    if all(metric_errors(baseline[key]["results"]["ddim_4"]) == 0 for key in objective):
        return "PURPOSE_ALREADY_MET_NO_OPTIMIZER", []
    return "TRAIN_BOUNDED", []


def structural_change_vs_parent(bound_baseline, parent_full):
    result = {}
    for key in set(bound_baseline) & set(parent_full):
        bound = bound_baseline[key]["results"]["ddim_4"]
        parent = parent_full[key]["results"]["ddim_4"]
        result[key] = {"bound_baseline_seed17_accuracy": bound["accuracy"],
            "parent_full_step256_seed17_accuracy": parent["accuracy"],
            "accuracy_delta": bound["accuracy"] - parent["accuracy"],
            "bound_baseline_seed17_macro_f1": bound["macro_f1"],
            "parent_full_step256_seed17_macro_f1": parent["macro_f1"],
            "macro_f1_delta": bound["macro_f1"] - parent["macro_f1"]}
    return result


def save_weights_only_baseline(model, path: Path, expected_digest: str, min_free_bytes: int):
    """Persist and re-read a weights-only proof without creating Adam/scaler state."""
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu_state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    actual = state_digest(cpu_state)
    if actual != expected_digest:
        raise ValueError("weights-only baseline digest changed before persistence")
    fd, tmp_name = tempfile.mkstemp(prefix=".baseline-", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        model_bytes = sum(t.numel() * t.element_size() for t in cpu_state.values())
        ensure_space(path.parent, model_bytes, min_free_bytes)
        torch.save({"model": cpu_state, "weights_sha256": actual, "updates": 0,
                    "optimizer_present": False, "schema": "query_bound_weights_only_v1"}, tmp_name)
        with open(tmp_name, "r+b") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_name, path)
        check = torch.load(path, map_location="cpu", weights_only=True)
        if state_digest(check["model"]) != actual or check["optimizer_present"] is not False:
            raise ValueError("weights-only baseline re-read failed verification")
        return actual, path.stat().st_size, sha256(path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def renamed(rows):
    mapping = [6, 7, 2, 3, 4, 5, 0, 1]
    items = copy.deepcopy(rows)
    for row in items:
        row["n"] = 8
        row["facts"] = [(mapping[s], mapping[o], k, text) for s, o, k, text in row["facts"]]
        row["queries"] = [(mapping[s], mapping[o]) for s, o in row["queries"]]
        if oracle(8, row["facts"], row["queries"]) != (row["labels"], row["hops"]):
            raise ValueError("renaming changed truth")
        row["graph_key"] = graph_key(8, row["facts"])
    return items


def initial_stage_manifest(train_rows, dev_rows):
    """Build the initial contract from actual pinned rows, never empty placeholders."""
    if not train_rows or not dev_rows:
        raise ValueError("actual nonempty train/dev rows are required before engine initialization")
    return manifest({"train": train_rows, "dev": dev_rows})


class QueryBoundEngine(Engine):
    """Runner-compatible engine that instantiates the bound model directly."""
    def __init__(self, spec, directory, *, train_rows, dev_rows):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU training fallback is disabled")
        if tuple(torch.cuda.get_device_capability()) not in ((7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (12, 0)):
            raise RuntimeError("CUDA device capability has not been selected for this local branch")
        configure_determinism()
        self.spec = spec
        self.out = Path(directory).resolve()
        self.store = common.ResilientCheckpointStore(self.out)
        self.run_id = self.out.name
        from fdt_local.objectives import Plan
        self.plan = Plan(**{**spec["plan"], "phase_steps": tuple(spec["plan"]["phase_steps"])})
        self.cfg = Config()
        overrides = spec.get("model_overrides") or {}
        self.cfg = Config(**{**asdict(self.cfg), **overrides})
        random.seed(self.plan.seed); np.random.seed(self.plan.seed)
        torch.manual_seed(self.plan.seed); torch.cuda.manual_seed_all(self.plan.seed)
        self.model = QueryBoundRefiner(self.cfg).cuda()
        self.train_data, self.dev = train_rows, dev_rows
        self.data_manifest = initial_stage_manifest(self.train_data, self.dev)
        self.contract = {"source": source_digest(), "config": asdict(self.cfg),
            "variant": "masked-query-bound", "plan": asdict(self.plan),
            "data": self.data_manifest["fingerprint"], "environment": environment(),
            "local_branch": "fdt-query-bound-alignment-20261027", "fresh_initialization": False}
        self.weights = torch.ones(4, device="cuda")
        self.ground_weights = torch.ones(4, device="cuda")
        self.phase = 2
        self.phase_step = self.updates = self.attempts = self.exposure = 0
        self.query_exposure, self.ground_exposure = [0] * 4, [0] * 4
        self.gates, self.history = [], []
        self.sampler = torch.Generator().manual_seed(self.plan.seed + 1)
        self.noise = torch.Generator(device="cuda").manual_seed(self.plan.seed + 2)

    def payload(self):
        value = super().payload()
        value["query_bound_runtime"] = copy.deepcopy(self.query_bound_runtime)
        value["querycount_sampler_state"] = self.qc_sampler.state_dict()
        return value

    def restore(self, payload):
        """Restore parent Engine state plus the finite data sampler and exposure ledger."""
        super().restore(payload)
        if "querycount_sampler_state" not in payload or "query_bound_runtime" not in payload:
            raise ValueError("checkpoint lacks QueryCount sampler/runtime state")
        self.qc_sampler = querycount_data.BalancedQueryCountBatchSampler.from_state(
            self.querycount_train_views, payload["querycount_sampler_state"])
        self.row_index = {(int(row["query_count"]), int(row["source_graph_index"])): i
                          for i, row in enumerate(self.train_data)}
        self.query_bound_runtime = copy.deepcopy(payload["query_bound_runtime"])

    def step(self):
        """One attempt consumes exactly one finite, balanced eight-row sampler batch."""
        ensure_space(self.out, self.estimate_checkpoint())
        self.attempts += 1
        started = time.monotonic()
        self.model.train(); self.optimizer.zero_grad(set_to_none=True)
        rows = self.qc_sampler.next_batch()
        if rows is None:
            raise RuntimeError("FINITE_QUERYCOUNT_EPOCH_EXHAUSTED")
        indices = [self.row_index[(int(r["query_count"]), int(r["source_graph_index"]))] for r in rows]
        parts, batch_indices = {}, []
        for start in (0, 4):
            ids = indices[start:start + 4]; batch_indices.append(ids)
            features, gold, exposure = collate([self.train_data[i] for i in ids], self.cfg, "cuda")
            self.exposure += exposure
            for cls in range(4):
                self.query_exposure[cls] += int(((gold["labels"] == cls) * features["query_mask"]).sum())
            with torch.autocast("cuda", dtype=torch.float16):
                with torch.no_grad(): encoded = self.model.encode(features)
                target = self.model.target(features, gold["labels"])
                loss, components = diffusion_loss(self.model, encoded, target, gold["labels"],
                                                   features["query_mask"], self.noise, self.weights)
                if not bool(torch.isfinite(loss)): raise FloatingPointError("nonfinite QueryCount loss")
                self.scaler.scale(loss / self.plan.accumulation).backward()
            components["loss"] = loss.detach()
            for key, value in components.items(): parts[key] = parts.get(key, 0.) + float(value) / self.plan.accumulation
        self.scaler.unscale_(self.optimizer)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        missing = [n for n,p in self.model.named_parameters() if p.requires_grad and p.grad is None]
        frozen_grad = [n for n,p in self.model.named_parameters() if not p.requires_grad and p.grad is not None]
        if missing or frozen_grad: raise RuntimeError(f"gradient ownership failure missing={missing}, frozen={frozen_grad}")
        finite = all(bool(torch.isfinite(p.grad).all()) for p in trainable)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
        old_scale = self.scaler.get_scale(); self.scaler.step(self.optimizer); self.scaler.update()
        successful = self.scaler.get_scale() >= old_scale
        if successful: self.updates += 1; self.phase_step += 1
        if not all(bool(torch.isfinite(p).all()) for p in self.model.parameters()): raise FloatingPointError("nonfinite parameter")
        for group in self.optimizer.param_groups:
            ratio = min(1, self.phase_step / self.plan.phase_steps[self.phase])
            group["lr"] = group["base_lr"] * (.1 + .9 * .5 * (1 + math.cos(math.pi * ratio)))
        torch.cuda.synchronize()
        entry = {"attempt": self.attempts, "updates": self.updates, "phase": "diffusion",
            "phase_step": self.phase_step, "successful": successful, "amp_skipped": not successful,
            "amp_scale": self.scaler.get_scale(), "finite_gradients": finite,
            "gradient_norm_before_clip": float(norm), "gradient_tensor_count": len(trainable),
            "missing_gradients": missing, "frozen_gradients": frozen_grad,
            "batch_indices": batch_indices, "batch_query_counts": [int(r["query_count"]) for r in rows],
            "batch_source_graph_indices": [int(r["source_graph_index"]) for r in rows],
            "sampler_state": self.qc_sampler.state_dict(), "input_utf8_bytes": self.exposure,
            "step_seconds": time.monotonic()-started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(), **parts}
        self.history.append(entry)
        return entry


class Experiment:
    def __init__(self, plan, plan_path, output, rows, spec):
        self.plan, self.plan_path, self.out = plan, plan_path, output
        self.run_dir = Path(plan["run_dir"])
        self.last_meta = self.last_marker = self.last_durable = None
        self.retention_fail_streak = 0
        state = io.load_json(output / "supervisor/state.json")
        self.deadline = float(state["deadline_unix"])
        self.training_deadline = self.deadline - plan["reserve_seconds"]
        io.atomic_json(output / "clock.json", {"started_unix": time.time(),
            "deadline_unix": self.deadline, "training_deadline_unix": self.training_deadline,
            "reserve_seconds": plan["reserve_seconds"], "immutable_budget": True})
        self.budget = Budget(max(0, self.deadline - time.time()), plan["max_forward_attempts"])
        self.progress("initializing")
        self.train_rows = rows["train"]
        self.dev_rows = [row for q in (1,4,8,16) for row in rows["dev_views"][q]]
        self.engine = QueryBoundEngine(spec, self.run_dir, train_rows=self.train_rows, dev_rows=self.dev_rows)
        engine = self.engine
        if engine.model.counts()["parameters"] != 102237197:
            raise ValueError("actual 102M parameter count mismatch")
        meta, parent = CheckpointStore(Path(plan["parent_marker"]).parent).verify(Path(plan["parent_marker"]))
        before = state_digest(parent["model"])
        engine.model.load_state_dict(parent["model"], strict=True)
        if parameter_digest(engine.model) != before:
            raise ValueError("strict parent weight import failed")
        del parent
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        engine.sampler = torch.Generator().manual_seed(SEED + 1)
        engine.querycount_train_views = rows["train_views"]
        engine.qc_sampler = querycount_data.BalancedQueryCountBatchSampler(rows["train_views"], seed=SEED)
        engine.row_index = {(int(row["query_count"]), int(row["source_graph_index"])): i
                            for i, row in enumerate(rows["train"])}
        engine.noise = torch.Generator(device="cuda").manual_seed(SEED + 2)
        engine.phase, engine.phase_step, engine.updates, engine.attempts, engine.exposure = 2, 0, 0, 0, 0
        engine.history, engine.gates = [], []
        engine.query_exposure, engine.ground_exposure = [0] * 4, [0] * 4
        engine.train_data, engine.dev = self.train_rows, self.dev_rows
        engine.data_manifest = manifest({"train": self.train_rows, "dev": self.dev_rows})
        expected_names = [k for k, _ in configure_trainables(engine.model)]
        # The declared class weighting contract counts every stored Q-prefix target.
        engine.weights = query_class_weights([r for q in (1,4,8,16) for r in rows["train_views"][q]], "cuda")
        self.frozen_before = frozen_digest(engine.model)
        engine.run_id = output.name
        engine.contract.update({"plan": asdict(engine.plan), "data": engine.data_manifest["fingerprint"],
            "fresh_initialization": False, "local_branch": "fdt-querycount-alignment-20261027",
            "query_bound_execution": {"plan_sha256": sha256(plan_path), "dependencies": {
                k: v for k, v in plan.items() if k.endswith("sha256")},
                "parent_query_mix_updates": 256, "ancestor_active_balanced_updates": 128,
                "ancestor_role8_updates": 128,
                "ancestor_e2_updates": 128, "ancestor_e1_updates": 1440,
                "weights_only_warmstart": True, "fresh_optimizer_scaler_rng": "created only if training is required",
                "trainable_names": expected_names, "query_class_weights": engine.weights.tolist(),
                "frozen_subset_sha256": self.frozen_before}})
        engine.query_bound_runtime = {"attempted_id_exposure": {k: [0]*8 for k in EXPOSURE_KEYS},
            "successful_id_exposure": {k: [0]*8 for k in EXPOSURE_KEYS},
            "attempted_query_class_exposure": [0]*4, "successful_query_class_exposure": [0]*4,
            "unique_graph_exposure": [0]*1024, "attempted_query_count_exposure": {str(q): 0 for q in (1,4,8,16)},
            "successful_query_count_exposure": {str(q): 0 for q in (1,4,8,16)},
            "forward_attempts": 0, "retention_fail_streak": 0,
            "retention_failure_panels": [], "query_class_counts": [0]*4}
        for name, value in (("contract.json", engine.contract), ("data_manifest.json", engine.data_manifest),
                            ("model_manifest.json", engine.model.counts())):
            io.atomic_json(self.run_dir / name, value)
        self.parent_full = {key: io.load_json(path) for key, path in plan["parent_full_reference_paths"].items()}
        normal = make_split("test", 16, seed=60001)
        self.panels = {"primary_querymix_dev64": rows["dev_views"][16],
            "uniform_dev64": rows["uniform_dev"], "old_role_dev128": rows["old_role_dev"],
            "old_normal16": normal,
            "old_large16": make_split("test", 16, seed=80001, n_range=(7, 8)),
            "paired_new_ids16": renamed(normal)}
        for q in (1,4,8,16):
            self.panels[f"querycount_dev_q{q:02d}"] = rows["dev_views"][q]
            self.panels[f"transfer_q{q:02d}"] = rows["transfer_views"][q]
        self.forward_usage, self.evaluation_seconds = {}, {}
        self.baseline = self.evaluate("baseline", (17,))
        engine.query_bound_runtime["baseline"] = self.baseline
        action, failures = baseline_alignment_action(self.baseline, self.parent_full)
        engine.query_bound_runtime["baseline_decision"] = {"action": action, "failures": failures}
        io.atomic_json(output / "baseline_alignment_decision.json", {
            "action": action, "failures": failures, "parent_querymix_updates": 256,
            "optimizer_created": False, "baseline_primary_seed17_accuracy":
                self.baseline["primary_querymix_dev64"]["results"]["ddim_4"]["accuracy"],
            "baseline_paired_seed17_accuracy":
                self.baseline["paired_new_ids16"]["results"]["ddim_4"]["accuracy"],
            "structural_change_vs_parent_seed17": structural_change_vs_parent(self.baseline, self.parent_full),
            "parent_full_references": plan["parent_full_reference_sha256"]})
        if action == "TRAIN_BOUNDED":
            engine._optimizer()
            optimizer_ids = {id(parameter) for group in engine.optimizer.param_groups for parameter in group["params"]}
            expected_ids = {id(parameter) for name, parameter in engine.model.named_parameters() if is_trainable(name)}
            validate_fresh_runtime_state(engine.optimizer.state, optimizer_ids, expected_ids,
                                         [g["base_lr"] for g in engine.optimizer.param_groups],
                                         engine.scaler.state_dict())
            io.atomic_json(output / "warmstart_verification.json", {"parent_sha256": meta["sha256"],
                "model_weight_sha256_before_after": before, "exact_weights": True,
                "parent_query_mix_updates": 256, "ancestor_active_balanced_updates": 128,
                "ancestor_role8_updates": 128, "ancestor_e2_updates": 128, "ancestor_e1_updates": 1440,
                "stage_updates_start": 0, "fresh_optimizer": not engine.optimizer.state,
                "fresh_rng_seed": SEED, "frozen_subset_sha256": self.frozen_before,
                "class_weights": engine.weights.tolist()})
            self.save("weights_only_warmstart")
        elif action == "PURPOSE_ALREADY_MET_NO_OPTIMIZER":
            self.final_proof = self.evaluate("final_evaluation", (17, 29, 43))
            after_proof = parameter_digest(engine.model)
            if after_proof != before:
                raise RuntimeError("weights changed during zero-optimizer final proof")
            digest, size, artifact_sha = save_weights_only_baseline(engine.model,
                self.run_dir / "baseline_alignment_weights.pt", before, plan["min_free_bytes"])
            self.weights_only_proof = {"weights_sha256": digest, "bytes": size,
                "artifact_sha256": artifact_sha,
                "artifact_path": str(self.run_dir / "baseline_alignment_weights.pt")}
            io.atomic_json(output / "weights_only_baseline_verification.json", {
                **self.weights_only_proof, "updates": 0,
                "optimizer_created": False, "scaler_created": False,
                "parent_sha256": meta["sha256"], "strict_parent_weights": True,
                "model_digest_before_final_proof": before, "model_digest_after_final_proof": after_proof,
                "same_seed17_baseline_and_final_seeds": [17, 29, 43],
                "baseline_forward_usage": self.forward_usage.get("baseline", 0),
                "final_proof_forward_usage": self.forward_usage.get("final_evaluation", 0),
                "training_gain": 0, "training_gain_claimed": False})

    def progress(self, stage, **extra):
        engine = getattr(self, "engine", None)
        io.atomic_json(self.out / "worker_progress.json", {"stage": stage, "observed_unix": time.time(),
            "worker_pid": os.getpid(), "phase": 2, "updates": engine.updates if engine else 0,
            "attempts": engine.attempts if engine else 0, "phase_step": engine.phase_step if engine else 0,
            "target_successful_updates": TARGET_UPDATES, "max_attempts": MAX_ATTEMPTS,
            "last_durable_unix": self.last_durable, "last_marker": str(self.last_marker) if self.last_marker else None,
            "deadline_unix": getattr(self, "deadline", None), "anomaly": None, **extra})

    def save(self, tag=None):
        ensure_space(self.run_dir, self.engine.estimate_checkpoint())
        self.engine.query_bound_runtime["forward_attempts"] = self.budget.attempted
        self.progress("saving")
        self.last_meta, self.last_marker = self.engine.save(tag)
        self.last_durable = time.time()
        io.atomic_json(self.out / "recovery.json", {"saved_unix": self.last_durable,
            "marker": str(self.last_marker), "sha256": self.last_meta["sha256"], "bytes": self.last_meta["bytes"],
            "stage_attempts": self.engine.attempts, "stage_updates": self.engine.updates,
            "phase": 2, "phase_step": self.engine.phase_step, "full_model_adam_scaler_all_rng": True,
            "clean_boundary": True, "save_seconds": self.last_meta["save_seconds"],
            "plan_sha256": sha256(self.plan_path), "parent_sha256": self.plan["parent_sha256"]})

    def evaluate(self, name, seeds):
        if frozen_digest(self.engine.model) != self.frozen_before:
            raise RuntimeError("FROZEN_SUBSET_CHANGED")
        self.progress("evaluating", evaluation=name)
        result = {}
        forward_start, time_start = self.budget.attempted, time.monotonic()
        # Evaluation has its own noise generators; preserve global Torch RNG as well.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.engine.model.eval()
            for key, rows in self.panels.items():
                value = panel(self.engine.model, rows, seeds, [4], self.budget)
                io.atomic_json(self.out / name / f"{key}.json", value)
                result[key] = {k: v for k, v in value.items() if k != "raw"}
        if frozen_digest(self.engine.model) != self.frozen_before:
            raise RuntimeError("FROZEN_SUBSET_CHANGED")
        self.forward_usage[name] = self.budget.attempted - forward_start
        self.evaluation_seconds[name] = time.monotonic() - time_start
        io.atomic_json(self.out / name / "summary.json", result)
        return result

    def train(self):
        done = set()
        status, reason = "STOPPED", "UNKNOWN"
        baseline_action = self.engine.query_bound_runtime["baseline_decision"]["action"]
        if baseline_action == "PURPOSE_ALREADY_MET_NO_OPTIMIZER":
            status, reason = baseline_action, "ALL_QUERYCOUNT_AND_TRANSFER_VIEWS_ERROR_FREE_NO_SGD"
            self.progress("terminal", final_status=status, reason=reason)
            io.atomic_json(self.out / "evaluation_summary.json", {
                "status": status, "reason": reason, "updates": 0, "attempts": 0,
                "optimizer_created": False, "baseline_only": True,
                "baseline_seed17": self.baseline, "parent_full_step256_seed17": self.parent_full,
                "panels": self.final_proof, "three_seed_training_completion": False,
                "structural_change_vs_parent_seed17": structural_change_vs_parent(self.baseline, self.parent_full),
                "forward_attempts": self.budget.attempted,
                "logical_forward_attempts_by_evaluation": dict(self.forward_usage),
                "evaluation_wall_seconds_by_evaluation": dict(self.evaluation_seconds),
                "training_gain": 0, "training_gain_claimed": False,
                "weights_only_proof": self.weights_only_proof,
                "logical_forward_cap": self.plan["max_forward_attempts"],
                "no_claim_of_training_gain": True})
            # This boolean describes completed read-only proof, never SGD completion.
            return status, reason, True
        if baseline_action == "STOP_BEFORE_OPTIMIZER_LEGACY_GUARD":
            status, reason = "STOPPED", baseline_action
            self.progress("terminal", final_status=status, reason=reason)
            io.atomic_json(self.out / "evaluation_summary.json", {
                "status": status, "reason": reason, "updates": 0, "attempts": 0,
                "optimizer_created": False,
                "baseline_failures": self.engine.query_bound_runtime["baseline_decision"]["failures"],
                "baseline_seed17": self.baseline, "parent_full_step256_seed17": self.parent_full})
            return status, reason, False
        while True:
            if (self.run_dir / "STOP").exists(): reason = "COOPERATIVE_STOP"; break
            if self.engine.updates >= TARGET_UPDATES: status, reason = "COMPLETE", "TARGET_UPDATES_REACHED"; break
            if self.engine.attempts >= MAX_ATTEMPTS: reason = "ATTEMPT_CAP"; break
            if time.time() >= self.training_deadline: reason = "TRAINING_DEADLINE"; break
            self.budget.reserve(2 * self.engine.plan.accumulation)  # one encode+denoise per microbatch
            self.progress("training")
            entry = self.engine.step()
            exposure_update = update_training_exposure(self.engine.query_bound_runtime, self.train_rows,
                                     entry["batch_indices"], entry["successful"])
            self.engine.query_bound_runtime["query_class_counts"] = copy.deepcopy(
                self.engine.query_bound_runtime["attempted_query_class_exposure"])
            entry["query_bound_exposure_update"] = exposure_update
            entry["query_bound_runtime"] = copy.deepcopy(self.engine.query_bound_runtime)
            self.save()
            io.atomic_json(self.out / "steps" / f"attempt_{self.engine.attempts:04d}.json", {
                **entry, "checkpoint_sha256": self.last_meta["sha256"], "saved_unix": self.last_durable})
            anomaly = common.anomaly_reason(self.engine.history)
            if anomaly: reason = anomaly; break
            step = self.engine.updates
            if step in DIAGNOSTIC_STEPS and step not in done:
                done.add(step)
                current = self.evaluate(f"diagnostics/step{step:03d}", (17,))
                self.retention_fail_streak, failures = retention_streak(
                    self.retention_fail_streak, self.baseline, current, self.parent_full)
                self.engine.query_bound_runtime["retention_fail_streak"] = self.retention_fail_streak
                self.engine.query_bound_runtime["retention_failure_panels"] = failures
                self.progress("validated", retention_fail_streak=self.retention_fail_streak,
                              retention_failures=failures)
                self.save(f"milestone_{step:04d}")
                if self.retention_fail_streak >= 2: reason = "RETENTION_REGRESSION_FAILED_TWICE"; break
        if frozen_digest(self.engine.model) != self.frozen_before:
            raise RuntimeError("FROZEN_SUBSET_CHANGED")
        self.save("final")
        complete = False
        if status == "COMPLETE" and time.time() < self.deadline - 30:
            previous_deadline = self.budget.deadline
            self.budget.deadline = min(previous_deadline, time.monotonic() + self.plan["final_eval_budget_seconds"])
            try:
                final = self.evaluate("final_evaluation", (17, 29, 43))
            finally:
                self.budget.deadline = previous_deadline
            goals = {}
            step128_panels = io.load_json(self.out / "diagnostics/step128/summary.json")
            goals["count_balanced_q1_q4"] = development_goal(self.baseline, step128_panels, final)
            goals["qdev_accuracy_by_count"] = {f"q{q:02d}": {
                "baseline_seed17_accuracy": self.baseline[f"querycount_dev_q{q:02d}"]["results"]["ddim_4"]["accuracy"],
                "step128_seed17_accuracy": step128_panels[f"querycount_dev_q{q:02d}"]["results"]["ddim_4"]["accuracy"],
                "baseline_errors": metric_errors(self.baseline[f"querycount_dev_q{q:02d}"]["results"]["ddim_4"]),
                "step128_errors": metric_errors(step128_panels[f"querycount_dev_q{q:02d}"]["results"]["ddim_4"]),
                "final_three_seed_accuracy": final[f"querycount_dev_q{q:02d}"]["results"]["ddim_4"]["accuracy"]}
                for q in (1,4,8,16)}
            paired = "paired_new_ids16"
            paired_baseline = self.baseline[paired]["results"]["ddim_4"]["accuracy"]
            paired_step128 = io.load_json(self.out / "diagnostics/step128" / f"{paired}.json")["results"]["ddim_4"]["accuracy"]
            goals[paired] = {"baseline_seed17_accuracy": paired_baseline,
                "step128_seed17_accuracy": paired_step128,
                "relative_accuracy_change": paired_step128 - paired_baseline,
                "retention_floor": paired_baseline - .02,
                "final_three_seed_accuracy": final[paired]["results"]["ddim_4"]["accuracy"]}
            io.atomic_json(self.out / "evaluation_summary.json", {"checkpoint_sha256": self.last_meta["sha256"],
                "panels": final, "forward_attempts": self.budget.attempted,
                "bound_baseline_seed17": self.baseline,
                "parent_full_step256_seed17": self.parent_full,
                "structural_change_vs_parent_seed17": structural_change_vs_parent(self.baseline, self.parent_full),
                "training_gain_is_same_seed17_vs_bound_baseline": True,
                "final_three_seed_averages_are_reported_separately": True,
                "e2_final_holdout_used": False, "independent_graph_count_claimed": False,
                "frozen_subset_sha256": self.frozen_before,
                "actual_id_exposure": self.engine.query_bound_runtime["attempted_id_exposure"],
                "query_class_exposure": self.engine.query_bound_runtime["attempted_query_class_exposure"],
                "unique_graphs_sampled": sum(v > 0 for v in self.engine.query_bound_runtime["unique_graph_exposure"]),
                "logical_forward_cap": self.plan["max_forward_attempts"],
                "logical_forward_attempts_including_baseline_diagnostics_final": self.budget.attempted,
                "logical_forward_attempts_by_evaluation": dict(self.forward_usage),
                "evaluation_wall_seconds_by_evaluation": dict(self.evaluation_seconds),
                "final_evaluation_budget_seconds": self.plan["final_eval_budget_seconds"],
                "querycount_dev_qualification_floor": .99,
                "querycount_dev_qualification_met_at_step128_seed17": all(v["step128_seed17_accuracy"] >= .99 for v in goals["qdev_accuracy_by_count"].values()),
                "count_balanced_q1_q4_error_halving_met_at_step128_seed17": goals["count_balanced_q1_q4"]["count_balanced_q1_q4_error_rate_halved"],
                "finite_sampler_epoch_complete": self.engine.qc_sampler.state_dict()["cursor"] == 1024,
                "finite_sampler_cursor": self.engine.qc_sampler.state_dict()["cursor"],
                "attempted_query_count_exposure": self.engine.query_bound_runtime["attempted_query_count_exposure"],
                "successful_query_count_exposure": self.engine.query_bound_runtime["successful_query_count_exposure"],
                "uniform_dev64_same_facts_correlated_view": True,
                "development_goals_vs_parent_baseline": goals})
            self.save("final_evaluated")
            complete = True
        self.progress("terminal", final_status=status, reason=reason)
        return status, reason, complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    plan_path, output = args.plan.resolve(), args.output.resolve()
    plan, rows, spec = preflight(plan_path, output, Path(__file__).resolve())
    if args.preflight:
        print("CPU_PREFLIGHT_PASS"); return 0
    validate_authorization(plan, actual_run=True)
    common.install_io_backend()
    experiment = None
    status, reason, complete = "FAILED", "INITIALIZATION_FAILURE", False
    power = None
    try:
        if os.name == "nt":
            power = ctypes.WinDLL("kernel32", use_last_error=True).SetThreadExecutionState
            power.argtypes, power.restype = [ctypes.c_uint32], ctypes.c_uint32
            if not power(0x80000001): raise OSError("system sleep prevention request failed")
        with common.ResilientCheckpointStore(plan["run_dir"]).lock():
            common.validate_locked_run_dir(Path(plan["run_dir"]))
            signal.signal(signal.SIGINT, lambda *_: io.atomic_text(Path(plan["run_dir"]) / "STOP", "USER_SIGNAL"))
            signal.signal(signal.SIGTERM, lambda *_: io.atomic_text(Path(plan["run_dir"]) / "STOP", "USER_SIGNAL"))
            experiment = Experiment(plan, plan_path, output, rows, spec)
            status, reason, complete = experiment.train()
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        # A half-failed step must never replace the last verified clean state.
        io.atomic_json(output / "incident.json", {"error": reason, "last_clean_marker":
            str(experiment.last_marker) if experiment and experiment.last_marker else None,
            "failed_partial_state_claimed_clean": False})
    finally:
        if power is not None: power(0x80000000)
        result = {"status": status, "reason": reason, "observed_unix": time.time(),
            "stage_updates": experiment.engine.updates if experiment else 0,
            "stage_attempts": experiment.engine.attempts if experiment else 0,
            "last_marker": str(experiment.last_marker) if experiment and experiment.last_marker else None,
            "last_checkpoint_sha256": experiment.last_meta["sha256"] if experiment and experiment.last_meta else None,
            "readonly_summary_complete": complete, "no_automatic_restart": True,
            "baseline_only": status == "PURPOSE_ALREADY_MET_NO_OPTIMIZER",
            "weights_only_proof": getattr(experiment, "weights_only_proof", None),
            "parent_query_mix_updates": 256, "ancestor_active_balanced_updates": 128,
            "ancestor_role8_updates": 128,
            "ancestor_e2_updates": 128, "ancestor_e1_updates": 1440,
            "weights_only_warmstart": True, "e2_final_holdout_used": False,
            "forward_attempts": experiment.budget.attempted if experiment else 0,
            "temporary_sleep_prevention_released": True}
        io.atomic_json(output / "final_status.json", result)
    return 0 if status in ("COMPLETE", "STOPPED", "PURPOSE_ALREADY_MET_NO_OPTIMIZER") else 1


if __name__ == "__main__":
    raise SystemExit(main())
