"""Bounded query-mix phase2 continuation from the preserved active-balanced checkpoint."""
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
from fdt_local.runner import Engine, parameter_digest, source_digest
from fdt_local.storage import CheckpointStore, ensure_space
import learning_io_v2 as io
import run_e2_grounding_20261026 as common
from probe_e1_readonly import Budget, panel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
import role8_query_mix_v1 as query_mix_data
TRAINABLE_PREFIXES = ("slot_router.", "memory_project.", "rel_left", "rel_right",
    "rel_score.", "rel_message.", "time.", "refiner.", "refine_norm.", "velocity.")
TARGET_UPDATES, MAX_ATTEMPTS, SEED = 256, 280, 23
DIAGNOSTIC_STEPS = (64, 128, 192, 256)
FEATURE_KEYS = {"tokens", "roles", "fact_mask", "queries", "query_mask"}
EXPOSURE_KEYS = ("fact_source", "fact_object", "query_source", "query_object")
CONTROL_RETENTION_PANELS = ("uniform_dev64", "old_role_dev128", "old_normal16", "old_large16")
RELATIVE_RETENTION_PANELS = ("primary_querymix_dev64", "paired_new_ids16")
PARENT_SHA256 = "cf85d259de333b25a05f52e5ac0ad3711e02154bdf176d58adf028a07ddcaa5f"
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
        raise ValueError("fresh phase2 Adam/ownership/LR mismatch")
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
    graph_counts = Counter(indices)
    for index in indices:
        for label in rows[index]["labels"]:
            class_counts[int(label)] += 1
    return id_counts, class_counts, graph_counts


def update_training_exposure(runtime, rows, batch_indices, successful):
    id_counts, class_counts, graph_counts = count_training_exposure(rows, batch_indices)
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
    return {"id_counts": id_counts, "query_class_counts": class_counts,
            "graph_counts": dict(graph_counts)}


def development_goal(baseline_seed17, step256_seed17, final_three_seed_accuracy):
    gain = step256_seed17 - baseline_seed17
    return {"baseline_seed17_accuracy": baseline_seed17,
        "step256_seed17_accuracy": step256_seed17, "seed17_absolute_gain": gain,
        "relative_gain_goal": .03, "relative_gain_goal_met": gain >= .03,
        "qualification_accuracy_floor": .94,
        "qualification_met": step256_seed17 >= .94,
        "final_three_seed_accuracy": final_three_seed_accuracy}


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
        if len(row.get("queries", [])) != 16 or len(row.get("labels", [])) != 16 or len(row.get("hops", [])) != 16:
            raise ValueError("query-mix rows require exactly sixteen queries/targets")
        if len(set(map(tuple, row["queries"]))) != 16 or "target_class" in row:
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
    if set(features) != FEATURE_KEYS or features["query_mask"].sum(-1).tolist() != [16] * len(rows):
        raise ValueError("query-mix model inputs must be exactly five features and 16 queries")
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
            plan.get("seed")) != (2, 256, 280, 23):
        raise ValueError("query-mix phase2 budget/seed mismatch")
    if (plan.get("parent_updates"), plan.get("parent_attempts")) != (128, 128):
        raise ValueError("active-balanced parent counters mismatch")
    if plan.get("parent_sha256") != PARENT_SHA256:
        raise ValueError("weights-only active-balanced parent digest mismatch")
    if (plan.get("parent_active_balanced_updates"), plan.get("ancestor_role8_updates"),
            plan.get("ancestor_e2_updates"), plan.get("ancestor_e1_updates")) != (128, 128, 128, 1440):
        raise ValueError("query-mix ancestry counters must stay distinct and exact")
    if plan.get("phase_steps") != [480, 320, 256]:
        raise ValueError("phase2 cosine horizon mismatch")
    if tuple(plan.get("trainable_prefixes", ())) != TRAINABLE_PREFIXES:
        raise ValueError("phase2 trainable whitelist mismatch")
    if plan.get("parent_optimizer_imported") is not False or plan.get("parent_rng_imported") is not False:
        raise ValueError("weights-only warmstart required")
    if plan.get("e2_final_split_evaluated") is not False:
        raise ValueError("E2 final holdout forbidden")


def preflight(plan_path, output, worker_path):
    plan = io.load_json(plan_path)
    validate_phase2_contract(plan)
    # Verify installed AMP defaults without allocating CUDA state or updating anything.
    cpu_scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0)
    validate_fresh_runtime_state({}, set(), set(), [3e-5], cpu_scaler.state_dict())
    if Path(plan["output_dir"]).resolve() != output.resolve():
        raise ValueError("matching output required")
    if Path(plan["run_dir"]).resolve() != (output / "model").resolve():
        raise ValueError("model output must be a new output/model child")
    common.validate_new_stage_paths(output, Path(plan["run_dir"]))
    dependencies = {"worker_sha256": worker_path,
        "supervisor_sha256": ROOT / "scripts/supervise_query_mix_diffusion.py",
        "io_sha256": Path(io.__file__), "common_worker_sha256": Path(common.__file__),
        "probe_common_sha256": ROOT / "scripts/probe_e1_readonly.py",
        "query_mix_builder_sha256": ROOT / "research/role8_query_mix_v1.py",
        "config_sha256": Path(plan["config_path"]),
        "train_data_sha256": Path(plan["train_data_path"]),
        "dev_data_sha256": Path(plan["dev_data_path"]),
        "candidate_manifest_sha256": Path(plan["candidate_manifest_path"]),
        "uniform_dev_data_sha256": Path(plan["uniform_dev_data_path"]),
        "old_role_dev_sha256": Path(plan["old_role_dev_path"]),
        "baseline_reference_sha256": Path(plan["baseline_reference_path"]),
        "parent_marker_sha256": Path(plan["parent_marker"])}
    if any(plan.get(key) != sha256(path) for key, path in dependencies.items()):
        raise ValueError("sealed code/data/parent marker digest mismatch")
    if source_digest() != plan["runtime_source_sha256"] or source_digest() != plan["core_sha256"]:
        raise ValueError("frozen core source changed")
    if (plan["wall_seconds"] != 1800 or plan["max_attempts"] != 280 or plan["reserve_seconds"] != 180
            or plan["min_free_bytes"] < 20 * 1024**3 or plan["min_gpu_free_mib"] < 12288
            or plan["max_forward_attempts"] != 6144 or plan["learning_rate"] != 3e-5
            or plan["final_eval_budget_seconds"] != 150 or plan.get("train_query_seed") != 33017
            or plan.get("dev_query_seed") != 22017 or plan.get("queries_per_graph") != 16
            or plan.get("max_dataset_rows") != 2240 or plan.get("max_dataset_queries") != 35840
            or plan.get("max_dataset_facts") != 26880 or plan.get("primary_qualification_floor") != .94):
        raise ValueError("fixed resource/time/forward/LR contract mismatch")
    candidate = io.load_json(plan["candidate_manifest_path"])
    if (candidate["schema"] != "fdt_role8_query_mix_v1"
            or candidate["status"] != "CPU_ROLE8_QUERY_MIX_V1_CANDIDATE_NOT_TRAINED"
            or candidate["construction"]["new_abstract_graphs_added"] != 0
            or candidate["construction"]["query_count_per_row"] != 16
            or candidate["construction"]["uniform_without_replacement"] is not True
            or candidate["construction"]["independence_claimed"] is not False):
        raise ValueError("QueryMix v1 manifest/schema contract mismatch")
    candidate_payload = dict(candidate)
    payload_sha = candidate_payload.pop("manifest_payload_sha256_without_self", None)
    canonical = json.dumps(candidate_payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != payload_sha:
        raise ValueError("candidate manifest payload digest mismatch")
    if candidate["inputs"]["source_sha256"]["builder_py"] != sha256(ROOT / "research/role8_query_mix_v1.py"):
        raise ValueError("QueryMix v1 builder changed")
    if candidate["inputs"]["source_sha256"]["v2_builder_py"] != sha256(ROOT / "research/role8_active_balanced_v2.py"):
        raise ValueError("pinned V2 builder changed")
    if candidate["inputs"]["source_sha256"]["frozen_data_py"] != sha256(ROOT / "src/fdt_local/data.py"):
        raise ValueError("frozen data source changed")
    if candidate["inputs"]["source_sha256"]["model_config_py"] != sha256(ROOT / "src/fdt_local/model_masked.py"):
        raise ValueError("frozen model config source changed")
    if candidate["inputs"]["v3_baseline_raw_sha256"] != sha256(Path(plan["baseline_reference_path"])):
        raise ValueError("prior mixed-dev baseline reference digest mismatch")
    rows = {}
    for split, count in (("train", 1024), ("dev", 64)):
        path = Path(plan[f"{split}_data_path"])
        if candidate["files"][f"{split}.json"]["sha256"] != sha256(path):
            raise ValueError("manifest does not bind data")
        rows[split] = io.load_json(path)
        query_mix_data.validate_rows(rows[split], split=split)
        validate_query_mix_rows(rows[split], count)
    if {r["canonical_graph_key"] for r in rows["train"]} & {r["canonical_graph_key"] for r in rows["dev"]}:
        raise ValueError("source-family split leak")
    if candidate["files"]["train.json"]["bytes"] != Path(plan["train_data_path"]).stat().st_size:
        raise ValueError("manifest train byte count mismatch")
    if candidate["files"]["dev.json"]["bytes"] != Path(plan["dev_data_path"]).stat().st_size:
        raise ValueError("manifest dev byte count mismatch")
    v2_train, v2_dev, v2_manifest, v2_hashes, baseline = query_mix_data.read_v2_inputs()
    expected_train = query_mix_data.sample_split(v2_train, seed=plan["train_query_seed"], split="train")
    expected_dev = query_mix_data.sample_split(v2_dev, seed=plan["dev_query_seed"], split="dev",
                                               baseline_raw=baseline["raw"])
    if (rows["train"] != expected_train or rows["dev"] != expected_dev
            or v2_hashes["train.json"] != candidate["inputs"]["v2_files"]["train.json"]
            or v2_hashes["dev.json"] != candidate["inputs"]["v2_files"]["dev.json"]):
        raise ValueError("QueryMix rows are not the pinned uniform samples over unchanged V2 graphs")
    rows["uniform_dev"] = io.load_json(plan["uniform_dev_data_path"])
    if len(rows["uniform_dev"]) != 64:
        raise ValueError("uniform V2 dev panel must have 64 same-fact rows")
    uniform_features, uniform_targets, _ = collate(rows["uniform_dev"], Config(), "cpu")
    if set(uniform_features) != FEATURE_KEYS or uniform_features["query_mask"].sum(-1).tolist() != [8] * 64:
        raise ValueError("uniform dev control must expose exactly the five inputs and eight queries")
    for primary, uniform in zip(rows["dev"], rows["uniform_dev"]):
        if (primary["facts"] != uniform["facts"]
                or primary["graph_key"] != uniform["graph_key"]
                or primary["canonical_graph_key"] != uniform["canonical_graph_key"]
                or len(uniform["queries"]) != 8
                or oracle(8, uniform["facts"], uniform["queries"]) != (uniform["labels"], uniform["hops"])
                or query_mix_data.v2.independent_oracle(8, uniform["facts"], uniform["queries"])
                    != (uniform["labels"], uniform["hops"])):
            raise ValueError("uniform dev panel must be the V2 8-query view over the same facts")
    rows["old_role_dev"] = io.load_json(plan["old_role_dev_path"])
    if len(rows["old_role_dev"]) != 128:
        raise ValueError("old Role8 dev control must contain 128 rows")
    meta, payload = CheckpointStore(Path(plan["parent_marker"]).parent).verify(Path(plan["parent_marker"]))
    if (meta["sha256"] != plan["parent_sha256"]
            or meta["sha256"] != PARENT_SHA256
            or (payload["updates"], payload["attempts"]) != (128, 128)):
        raise ValueError("active-balanced parent fullstate mismatch")
    if payload["contract"]["source"] != source_digest() or payload["contract"]["variant"] != "masked":
        raise ValueError("parent source/model mismatch")
    parent_exec = payload["contract"].get("active_balanced_execution", {})
    if (parent_exec.get("parent_role8_updates"), parent_exec.get("ancestor_e2_updates"),
            parent_exec.get("ancestor_e1_updates")) != (128, 128, 1440):
        raise ValueError("parent ancestry metadata mismatch")
    del payload
    spec = yaml.safe_load(Path(plan["config_path"]).read_text(encoding="utf-8"))
    if (spec["profile"], spec["variant"], spec["start_phase"], spec["plan"]["phase_steps"],
            spec["plan"]["diffusion_lr"], spec["plan"]["seed"], spec["plan"]["micro_batch"],
            spec["plan"]["accumulation"], spec["plan"]["train_graphs"], spec["plan"]["dev_graphs"]) != ("full", "masked", 2, [480, 320, 256], 3e-5, 23, 4, 2, 1024, 64):
        raise ValueError("actual engine config mismatch")
    return plan, rows, spec


def retention_failures(baseline, current):
    failures = []
    for key in CONTROL_RETENTION_PANELS:
        before = baseline[key]["results"]["ddim_4"]
        after = current[key]["results"]["ddim_4"]
        accuracy_floor = max(.94, before["accuracy"] - .02)
        if after["accuracy"] < accuracy_floor:
            failures.append({"panel": key, "metric": "accuracy", "value": after["accuracy"],
                "floor": accuracy_floor, "guard": "qualified_control"})
        if after["macro_f1"] < before["macro_f1"] - .05:
            failures.append({"panel": key, "metric": "macro_f1", "value": after["macro_f1"],
                "floor": before["macro_f1"] - .05, "guard": "qualified_control"})
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


def retention_streak(previous, baseline, current):
    failures = retention_failures(baseline, current)
    return (previous + 1 if failures else 0), failures


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


class QueryMixEngine(Engine):
    def payload(self):
        value = super().payload()
        value["query_mix_runtime"] = copy.deepcopy(self.query_mix_runtime)
        return value


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
        self.train_rows, self.dev_rows = rows["train"], rows["dev"]
        self.engine = QueryMixEngine(spec, self.run_dir)
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
        engine.noise = torch.Generator(device="cuda").manual_seed(SEED + 2)
        engine.phase, engine.phase_step, engine.updates, engine.attempts, engine.exposure = 2, 0, 0, 0, 0
        engine.history, engine.gates = [], []
        engine.query_exposure, engine.ground_exposure = [0] * 4, [0] * 4
        engine.train_data, engine.dev = self.train_rows, self.dev_rows
        engine.data_manifest = manifest({"train": self.train_rows, "dev": self.dev_rows})
        engine._optimizer()
        expected_names = [k for k, _ in configure_trainables(engine.model)]
        optimizer_ids = {id(parameter) for group in engine.optimizer.param_groups for parameter in group["params"]}
        expected_ids = {id(parameter) for name, parameter in engine.model.named_parameters() if is_trainable(name)}
        validate_fresh_runtime_state(engine.optimizer.state, optimizer_ids, expected_ids,
                                     [g["base_lr"] for g in engine.optimizer.param_groups],
                                     engine.scaler.state_dict())
        engine.weights = query_class_weights(self.train_rows, "cuda")
        self.frozen_before = frozen_digest(engine.model)
        engine.run_id = output.name
        engine.contract.update({"plan": asdict(engine.plan), "data": engine.data_manifest["fingerprint"],
            "fresh_initialization": False, "local_branch": "fdt-query-mix-phase2-20261027",
            "query_mix_execution": {"plan_sha256": sha256(plan_path), "dependencies": {
                k: v for k, v in plan.items() if k.endswith("sha256")},
                "parent_active_balanced_updates": 128, "ancestor_active_balanced_updates": 128,
                "ancestor_role8_updates": 128,
                "ancestor_e2_updates": 128, "ancestor_e1_updates": 1440,
                "weights_only_warmstart": True, "fresh_optimizer_scaler_rng": True,
                "trainable_names": expected_names, "query_class_weights": engine.weights.tolist(),
                "frozen_subset_sha256": self.frozen_before}})
        engine.query_mix_runtime = {"attempted_id_exposure": {k: [0]*8 for k in EXPOSURE_KEYS},
            "successful_id_exposure": {k: [0]*8 for k in EXPOSURE_KEYS},
            "attempted_query_class_exposure": [0]*4, "successful_query_class_exposure": [0]*4,
            "unique_graph_exposure": [0]*len(self.train_rows),
            "forward_attempts": 0, "retention_fail_streak": 0,
            "retention_failure_panels": [], "query_class_counts": [0]*4}
        for name, value in (("contract.json", engine.contract), ("data_manifest.json", engine.data_manifest),
                            ("model_manifest.json", engine.model.counts())):
            io.atomic_json(self.run_dir / name, value)
        io.atomic_json(output / "warmstart_verification.json", {"parent_sha256": meta["sha256"],
            "model_weight_sha256_before_after": before, "exact_weights": True,
            "parent_active_balanced_updates": 128, "ancestor_role8_updates": 128,
            "ancestor_e2_updates": 128, "ancestor_e1_updates": 1440, "stage_updates_start": 0,
            "fresh_optimizer": not engine.optimizer.state, "fresh_rng_seed": SEED,
            "frozen_subset_sha256": self.frozen_before, "class_weights": engine.weights.tolist()})
        normal = make_split("test", 16, seed=60001)
        self.panels = {"primary_querymix_dev64": self.dev_rows,
            "uniform_dev64": rows["uniform_dev"], "old_role_dev128": rows["old_role_dev"],
            "old_normal16": normal,
            "old_large16": make_split("test", 16, seed=80001, n_range=(7, 8)),
            "paired_new_ids16": renamed(normal)}
        self.save("weights_only_warmstart")
        self.forward_usage, self.evaluation_seconds = {}, {}
        self.baseline = self.evaluate("baseline", (17,))
        engine.query_mix_runtime["baseline"] = self.baseline

    def progress(self, stage, **extra):
        engine = getattr(self, "engine", None)
        io.atomic_json(self.out / "worker_progress.json", {"stage": stage, "observed_unix": time.time(),
            "worker_pid": os.getpid(), "phase": 2, "updates": engine.updates if engine else 0,
            "attempts": engine.attempts if engine else 0, "phase_step": engine.phase_step if engine else 0,
            "target_successful_updates": 256, "max_attempts": 280,
            "last_durable_unix": self.last_durable, "last_marker": str(self.last_marker) if self.last_marker else None,
            "deadline_unix": getattr(self, "deadline", None), "anomaly": None, **extra})

    def save(self, tag=None):
        ensure_space(self.run_dir, self.engine.estimate_checkpoint())
        self.engine.query_mix_runtime["forward_attempts"] = self.budget.attempted
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
        while True:
            if (self.run_dir / "STOP").exists(): reason = "COOPERATIVE_STOP"; break
            if self.engine.updates >= 256: status, reason = "COMPLETE", "TARGET_UPDATES_REACHED"; break
            if self.engine.attempts >= 280: reason = "ATTEMPT_CAP"; break
            if time.time() >= self.training_deadline: reason = "TRAINING_DEADLINE"; break
            self.budget.reserve(2 * self.engine.plan.accumulation)  # one encode+denoise per microbatch
            self.progress("training")
            entry = self.engine.step()
            exposure_update = update_training_exposure(self.engine.query_mix_runtime, self.train_rows,
                                     entry["batch_indices"], entry["successful"])
            self.engine.query_mix_runtime["query_class_counts"] = copy.deepcopy(
                self.engine.query_mix_runtime["attempted_query_class_exposure"])
            entry["query_mix_exposure_update"] = exposure_update
            entry["query_mix_runtime"] = copy.deepcopy(self.engine.query_mix_runtime)
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
                    self.retention_fail_streak, self.baseline, current)
                self.engine.query_mix_runtime["retention_fail_streak"] = self.retention_fail_streak
                self.engine.query_mix_runtime["retention_failure_panels"] = failures
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
            for key in ("primary_querymix_dev64",):
                b = self.baseline[key]["results"]["ddim_4"]["accuracy"]
                step256 = io.load_json(self.out / "diagnostics/step256" / f"{key}.json")["results"]["ddim_4"]["accuracy"]
                final_three_seed = final[key]["results"]["ddim_4"]["accuracy"]
                goals[key] = development_goal(b, step256, final_three_seed)
            paired = "paired_new_ids16"
            paired_baseline = self.baseline[paired]["results"]["ddim_4"]["accuracy"]
            paired_step256 = io.load_json(self.out / "diagnostics/step256" / f"{paired}.json")["results"]["ddim_4"]["accuracy"]
            goals[paired] = {"baseline_seed17_accuracy": paired_baseline,
                "step256_seed17_accuracy": paired_step256,
                "relative_accuracy_change": paired_step256 - paired_baseline,
                "retention_floor": paired_baseline - .02,
                "final_three_seed_accuracy": final[paired]["results"]["ddim_4"]["accuracy"]}
            io.atomic_json(self.out / "evaluation_summary.json", {"checkpoint_sha256": self.last_meta["sha256"],
                "panels": final, "forward_attempts": self.budget.attempted,
                "e2_final_holdout_used": False, "independent_graph_count_claimed": False,
                "frozen_subset_sha256": self.frozen_before,
                "actual_id_exposure": self.engine.query_mix_runtime["attempted_id_exposure"],
                "query_class_exposure": self.engine.query_mix_runtime["attempted_query_class_exposure"],
                "unique_graphs_sampled": sum(v > 0 for v in self.engine.query_mix_runtime["unique_graph_exposure"]),
                "logical_forward_cap": self.plan["max_forward_attempts"],
                "logical_forward_attempts_including_baseline_diagnostics_final": self.budget.attempted,
                "logical_forward_attempts_by_evaluation": dict(self.forward_usage),
                "evaluation_wall_seconds_by_evaluation": dict(self.evaluation_seconds),
                "final_evaluation_budget_seconds": self.plan["final_eval_budget_seconds"],
                "primary_qualification_floor": .94,
                "primary_qualification_met_at_step256_seed17": goals["primary_querymix_dev64"]["qualification_met"],
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
    if plan.get("authorized") is not True:
        raise ValueError("launch requires explicit authorized plan")
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
            "parent_active_balanced_updates": 128, "ancestor_role8_updates": 128,
            "ancestor_e2_updates": 128, "ancestor_e1_updates": 1440,
            "weights_only_warmstart": True, "e2_final_holdout_used": False,
            "forward_attempts": experiment.budget.attempted if experiment else 0,
            "temporary_sleep_prevention_released": True}
        io.atomic_json(output / "final_status.json", result)
    return 0 if status in ("COMPLETE", "STOPPED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
