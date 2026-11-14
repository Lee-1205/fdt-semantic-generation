"""One bounded role-ID coverage intervention; original phase-2 objective only."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from collections import Counter
import copy
import ctypes
from dataclasses import asdict
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
TRAINABLE_PREFIXES = ("slot_router.", "memory_project.", "rel_left", "rel_right",
    "rel_score.", "rel_message.", "time.", "refiner.", "refine_norm.", "velocity.")
TARGET_UPDATES, MAX_ATTEMPTS, SEED = 128, 140, 21
DIAGNOSTIC_STEPS = (32, 64, 96, 128)
FEATURE_KEYS = {"tokens", "roles", "fact_mask", "queries", "query_mask"}
EXPOSURE_KEYS = ("fact_source", "fact_object", "query_source", "query_object")
sha256 = common.sha256
state_digest = common.state_digest


def is_role8_trainable(name):
    return name.startswith(TRAINABLE_PREFIXES)


def configure_role8_trainables(model):
    chosen = []
    for name, value in model.named_parameters():
        value.grad = None
        value.requires_grad_(is_role8_trainable(name))
        if value.requires_grad:
            chosen.append((name, value))
    if not all(any(name.startswith(p) for name, _ in chosen) for p in TRAINABLE_PREFIXES):
        raise ValueError("incomplete phase2 trainable whitelist")
    return chosen


def frozen_digest(model):
    return state_digest({k: v for k, v in model.state_dict().items() if not is_role8_trainable(k)})


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


def query_class_weights(rows, device="cpu"):
    counts = Counter(label for row in rows for label in row["labels"])
    inverse = torch.tensor([counts.get(i, 0) for i in range(4)], dtype=torch.float32,
                           device=device).clamp_min(1).reciprocal()
    return (inverse / inverse.mean()).clamp(.25, 4)


def validate_role8_rows(rows, expected_count):
    cfg = Config()
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("role8 row count mismatch")
    keys, families = set(), Counter()
    for row in rows:
        if row.get("n") != 8 or not 1 <= len(row["facts"]) <= cfg.facts:
            raise ValueError("role8 fact/entity capacity mismatch")
        if not 1 <= len(row["queries"]) <= cfg.slots:
            raise ValueError("role8 query capacity mismatch")
        if row["graph_key"] in keys:
            raise ValueError("duplicate physical graph key")
        keys.add(row["graph_key"])
        families[row.get("source_graph_key", "unspecified")] += 1
        for s, o, kind, phrase in row["facts"]:
            if not (0 <= s < 8 and 0 <= o < 8 and s != o and kind in range(4)
                    and isinstance(phrase, str) and len(phrase.encode("utf-8")) <= cfg.phrase_bytes):
                raise ValueError("invalid premise or phrase capacity")
        for s, o in row["queries"]:
            if not (0 <= s < 8 and 0 <= o < 8 and s != o):
                raise ValueError("invalid query")
        if oracle(8, row["facts"], row["queries"]) != (row["labels"], row["hops"]):
            raise ValueError("stored truth/hops disagree with oracle")
    features, _, _ = collate(rows[:4], cfg, "cpu")
    if set(features) != FEATURE_KEYS:
        raise ValueError("supervision leaked into neural input")
    return {"rows": len(rows), "source_families": len(families),
            "family_rotation_counts": sorted(families.values()),
            "entity_id_exposure": count_batch_exposure(rows, [range(len(rows))])}


def validate_phase2_contract(plan):
    if (plan.get("phase"), plan.get("target_successful_updates"), plan.get("max_attempts"),
            plan.get("seed")) != (2, 128, 140, 21):
        raise ValueError("fixed phase2 budget/seed mismatch")
    if (plan.get("parent_updates"), plan.get("parent_attempts")) != (128, 128):
        raise ValueError("E2 parent counters mismatch")
    if tuple(plan.get("trainable_prefixes", ())) != TRAINABLE_PREFIXES:
        raise ValueError("trainable whitelist mismatch")
    if plan.get("parent_optimizer_imported") is not False or plan.get("parent_rng_imported") is not False:
        raise ValueError("weights-only warmstart required")
    if plan.get("e2_final_split_evaluated") is not False:
        raise ValueError("final E2 holdout forbidden")


def preflight(plan_path, output, worker_path):
    plan = io.load_json(plan_path)
    validate_phase2_contract(plan)
    if plan.get("authorized") is not True or Path(plan["output_dir"]).resolve() != output.resolve():
        raise ValueError("authorized matching output required")
    if Path(plan["run_dir"]).resolve() != (output / "model").resolve():
        raise ValueError("model output must be a new output/model child")
    common.validate_new_stage_paths(output, Path(plan["run_dir"]))
    dependencies = {"worker_sha256": worker_path,
        "supervisor_sha256": ROOT / "scripts/supervise_role8_diffusion.py",
        "io_sha256": Path(io.__file__), "common_worker_sha256": Path(common.__file__),
        "probe_common_sha256": ROOT / "scripts/probe_e1_readonly.py",
        "config_sha256": Path(plan["config_path"]),
        "train_data_sha256": Path(plan["train_data_path"]),
        "dev_data_sha256": Path(plan["dev_data_path"]),
        "candidate_manifest_sha256": Path(plan["candidate_manifest_path"]),
        "parent_marker_sha256": Path(plan["parent_marker"])}
    if any(plan.get(key) != sha256(path) for key, path in dependencies.items()):
        raise ValueError("sealed code/data/parent marker digest mismatch")
    if source_digest() != plan["runtime_source_sha256"] or source_digest() != plan["core_sha256"]:
        raise ValueError("frozen core source changed")
    if (plan["wall_seconds"] != 1800 or plan["reserve_seconds"] != 180
            or plan["min_free_bytes"] < 20 * 1024**3 or plan["min_gpu_free_mib"] < 12288
            or plan["max_forward_attempts"] != 4096 or plan["learning_rate"] != 3e-5
            or plan["final_eval_budget_seconds"] != 150):
        raise ValueError("fixed resource/time/forward/LR contract mismatch")
    candidate = io.load_json(plan["candidate_manifest_path"])
    if candidate["schema"] != "fdt_role8_coverage_v1" or candidate["runtime_source_sha256"] != source_digest():
        raise ValueError("candidate schema/source mismatch")
    rows = {}
    for split, count in (("train", 512), ("dev", 128)):
        path = Path(plan[f"{split}_data_path"])
        if candidate["files"][f"{split}.json"]["sha256"] != sha256(path):
            raise ValueError("manifest does not bind data")
        rows[split] = io.load_json(path)
        info = validate_role8_rows(rows[split], count)
        if info["family_rotation_counts"] != [8] * (count // 8):
            raise ValueError("incomplete source-family rotations")
        if any(len(set(v)) != 1 for v in info["entity_id_exposure"].values()):
            raise ValueError("candidate ID exposure not balanced")
    if {r["source_graph_key"] for r in rows["train"]} & {r["source_graph_key"] for r in rows["dev"]}:
        raise ValueError("source-family split leak")
    meta, payload = CheckpointStore(Path(plan["parent_marker"]).parent).verify(Path(plan["parent_marker"]))
    if meta["sha256"] != plan["parent_sha256"] or (payload["updates"], payload["attempts"]) != (128, 128):
        raise ValueError("E2 parent fullstate mismatch")
    if payload["contract"]["source"] != source_digest() or payload["contract"]["variant"] != "masked":
        raise ValueError("parent source/model mismatch")
    del payload
    spec = yaml.safe_load(Path(plan["config_path"]).read_text(encoding="utf-8"))
    if (spec["profile"], spec["variant"], spec["start_phase"], spec["plan"]["phase_steps"][2],
            spec["plan"]["diffusion_lr"], spec["plan"]["seed"], spec["plan"]["micro_batch"],
            spec["plan"]["accumulation"]) != ("full", "masked", 2, 128, 3e-5, 21, 4, 2):
        raise ValueError("actual engine config mismatch")
    return plan, rows, spec


def retention_streak(previous, baseline, current):
    failed = False
    for key in ("old_normal16", "old_large16"):
        a, b = baseline[key]["results"]["ddim_4"], current[key]["results"]["ddim_4"]
        failed |= b["accuracy"] < max(.94, a["accuracy"] - .02)
        failed |= b["macro_f1"] < a["macro_f1"] - .05
        failed |= current[key]["ground_accuracy"] < .95
        failed |= current[key]["results"]["initializer"]["accuracy"] < .95
    return previous + 1 if failed else 0


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


class Role8Engine(Engine):
    def payload(self):
        value = super().payload()
        value["role8_runtime"] = copy.deepcopy(self.role8_runtime)
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
        self.engine = Role8Engine(spec, self.run_dir)
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
        expected_names = [k for k, _ in configure_role8_trainables(engine.model)]
        if engine.optimizer.state or any(g["base_lr"] != 3e-5 for g in engine.optimizer.param_groups):
            raise ValueError("fresh phase2 Adam/LR mismatch")
        engine.weights = query_class_weights(self.train_rows, "cuda")
        self.frozen_before = frozen_digest(engine.model)
        engine.run_id = output.name
        engine.contract.update({"plan": asdict(engine.plan), "data": engine.data_manifest["fingerprint"],
            "fresh_initialization": False, "local_branch": "fdt-role8-phase2-20261027",
            "role8_execution": {"plan_sha256": sha256(plan_path), "dependencies": {
                k: v for k, v in plan.items() if k.endswith("sha256")},
                "parent_updates": 128, "ancestor_e1_updates": 1440,
                "weights_only_warmstart": True, "fresh_optimizer_scaler_rng": True,
                "trainable_names": expected_names, "query_class_weights": engine.weights.tolist(),
                "frozen_subset_sha256": self.frozen_before}})
        engine.role8_runtime = {"attempted_id_exposure": {k: [0]*8 for k in EXPOSURE_KEYS},
            "successful_id_exposure": {k: [0]*8 for k in EXPOSURE_KEYS},
            "forward_attempts": 0, "retention_fail_streak": 0}
        for name, value in (("contract.json", engine.contract), ("data_manifest.json", engine.data_manifest),
                            ("model_manifest.json", engine.model.counts())):
            io.atomic_json(self.run_dir / name, value)
        io.atomic_json(output / "warmstart_verification.json", {"parent_sha256": meta["sha256"],
            "model_weight_sha256_before_after": before, "exact_weights": True,
            "parent_e2_updates": 128, "ancestor_e1_updates": 1440, "stage_updates_start": 0,
            "fresh_optimizer": not engine.optimizer.state, "fresh_rng_seed": SEED,
            "frozen_subset_sha256": self.frozen_before, "class_weights": engine.weights.tolist()})
        normal = make_split("test", 16, seed=60001)
        self.panels = {"role8_dev128": self.dev_rows, "old_normal16": normal,
            "old_large16": make_split("test", 16, seed=80001, n_range=(7, 8)),
            "paired_new_ids16": renamed(normal)}
        self.save("weights_only_warmstart")
        self.baseline = self.evaluate("baseline", (17,))
        engine.role8_runtime["baseline"] = self.baseline

    def progress(self, stage, **extra):
        engine = getattr(self, "engine", None)
        io.atomic_json(self.out / "worker_progress.json", {"stage": stage, "observed_unix": time.time(),
            "worker_pid": os.getpid(), "phase": 2, "updates": engine.updates if engine else 0,
            "attempts": engine.attempts if engine else 0, "phase_step": engine.phase_step if engine else 0,
            "target_successful_updates": 128, "max_attempts": 140,
            "last_durable_unix": self.last_durable, "last_marker": str(self.last_marker) if self.last_marker else None,
            "deadline_unix": getattr(self, "deadline", None), "anomaly": None, **extra})

    def save(self, tag=None):
        ensure_space(self.run_dir, self.engine.estimate_checkpoint())
        self.engine.role8_runtime["forward_attempts"] = self.budget.attempted
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
        # Evaluation has its own noise generators; preserve global Torch RNG as well.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            self.engine.model.eval()
            for key, rows in self.panels.items():
                value = panel(self.engine.model, rows, seeds, [4], self.budget)
                io.atomic_json(self.out / name / f"{key}.json", value)
                result[key] = {k: v for k, v in value.items() if k != "raw"}
        if frozen_digest(self.engine.model) != self.frozen_before:
            raise RuntimeError("FROZEN_SUBSET_CHANGED")
        io.atomic_json(self.out / name / "summary.json", result)
        return result

    def train(self):
        done = set()
        status, reason = "STOPPED", "UNKNOWN"
        while True:
            if (self.run_dir / "STOP").exists(): reason = "COOPERATIVE_STOP"; break
            if self.engine.updates >= 128: status, reason = "COMPLETE", "TARGET_UPDATES_REACHED"; break
            if self.engine.attempts >= 140: reason = "ATTEMPT_CAP"; break
            if time.time() >= self.training_deadline: reason = "TRAINING_DEADLINE"; break
            self.budget.reserve(2 * self.engine.plan.accumulation)  # one encode+denoise per microbatch
            self.progress("training")
            entry = self.engine.step()
            exposure = count_batch_exposure(self.train_rows, entry["batch_indices"])
            for kind in ("attempted_id_exposure", "successful_id_exposure"):
                if kind.startswith("successful") and not entry["successful"]: continue
                for key, values in exposure.items():
                    for index, value in enumerate(values):
                        self.engine.role8_runtime[kind][key][index] += value
            entry["role8_id_exposure"] = copy.deepcopy(self.engine.role8_runtime)
            self.save()
            io.atomic_json(self.out / "steps" / f"attempt_{self.engine.attempts:04d}.json", {
                **entry, "checkpoint_sha256": self.last_meta["sha256"], "saved_unix": self.last_durable})
            anomaly = common.anomaly_reason(self.engine.history)
            if anomaly: reason = anomaly; break
            step = self.engine.updates
            if step in DIAGNOSTIC_STEPS and step not in done:
                done.add(step)
                current = self.evaluate(f"diagnostics/step{step:03d}", (17,))
                self.retention_fail_streak = retention_streak(self.retention_fail_streak, self.baseline, current)
                self.engine.role8_runtime["retention_fail_streak"] = self.retention_fail_streak
                self.progress("validated", retention_fail_streak=self.retention_fail_streak)
                if self.retention_fail_streak >= 2: reason = "OLD_RETENTION_FAILED_TWICE"; break
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
            io.atomic_json(self.out / "evaluation_summary.json", {"checkpoint_sha256": self.last_meta["sha256"],
                "panels": final, "forward_attempts": self.budget.attempted,
                "e2_final_holdout_used": False, "independent_graph_count_claimed": False,
                "frozen_subset_sha256": self.frozen_before,
                "actual_id_exposure": self.engine.role8_runtime["attempted_id_exposure"]})
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
            "parent_e2_updates": 128, "ancestor_e1_updates": 1440,
            "weights_only_warmstart": True, "e2_final_holdout_used": False,
            "forward_attempts": experiment.budget.attempted if experiment else 0,
            "temporary_sleep_prevention_released": True}
        io.atomic_json(output / "final_status.json", result)
    return 0 if status in ("COMPLETE", "STOPPED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
