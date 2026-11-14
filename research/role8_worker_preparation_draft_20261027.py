"""Role8 phase-2 diffusion worker; --preflight is CPU-only and never launches CUDA."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np
import torch
import yaml

from fdt_local.data import collate, make_split, manifest
from fdt_local.objectives import compute_metrics
from fdt_local.runner import Engine, source_digest
from fdt_local.storage import CheckpointStore, ensure_space
import learning_io_v2 as safe_io

TRAINABLE_PREFIXES = ("slot_router.", "memory_project.", "rel_left", "rel_right",
    "rel_score.", "rel_message.", "time.", "refiner.", "refine_norm.", "velocity.")
PHASE_STEPS = (128, 320, 128)
TARGET_UPDATES, MAX_ATTEMPTS, SEED = 128, 140, 21
DIAGNOSTIC_STEPS = (32, 64, 96, 128)
PARENT_UPDATES = PARENT_ATTEMPTS = 128
FEATURE_KEYS = {"tokens", "roles", "fact_mask", "queries", "query_mask"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_state(state) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode()); h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def is_trainable(name: str) -> bool:
    return any(name.startswith(p) for p in TRAINABLE_PREFIXES)


def configure_trainables(model):
    selected = []
    for name, p in model.named_parameters():
        p.grad = None
        p.requires_grad_(is_trainable(name))
        if p.requires_grad: selected.append((name, p))
    names = [n for n, _ in selected]
    missing = [p for p in TRAINABLE_PREFIXES if not any(n.startswith(p) for n in names)]
    if not selected or missing or any(not is_trainable(n) for n in names):
        raise ValueError(f"trainable allowlist mismatch; missing={missing}")
    return selected


def frozen_digest(model) -> str:
    return digest_state({n: t for n, t in model.state_dict().items() if not is_trainable(n)})


def validate_rows(rows, expected, config):
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"row count mismatch: expected {expected}")
    ids, families = set(), set()
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or r.get("n") != 8:
            raise ValueError(f"row {i} must have n=8")
        if not 2 <= len(r.get("facts", [])) <= config.facts or len(r.get("queries", [])) != 1:
            raise ValueError(f"row {i} violates masked Config capacities")
        if len(r.get("labels", [])) != 1 or len(r.get("hops", [])) != 1:
            raise ValueError(f"row {i} label/hop cardinality mismatch")
        key, family = r.get("graph_key"), r.get("source_graph_key")
        if not isinstance(key, str) or key in ids or not isinstance(family, str):
            raise ValueError(f"row {i} duplicate/missing identity")
        ids.add(key); families.add(family)
        if any(len(str(f[3]).encode("utf-8")) > config.phrase_bytes for f in r["facts"]):
            raise ValueError(f"row {i} phrase exceeds model capacity")
    return {"rows": len(rows), "physical_graphs": len(ids), "source_families": len(families)}


def contract_check(plan: dict, plan_path: Path, worker_path: Path):
    if plan.get("authorized") is not True:
        raise ValueError("authorized role8 plan required")
    expected_values = {"parent_updates": PARENT_UPDATES, "parent_attempts": PARENT_ATTEMPTS,
        "seed": SEED, "phase_steps": list(PHASE_STEPS), "target_successful_updates": TARGET_UPDATES,
        "max_attempts": MAX_ATTEMPTS, "attempts_at_start": 0, "successful_at_start": 0}
    for k, v in expected_values.items():
        if plan.get(k) != v: raise ValueError(f"plan {k} mismatch: {plan.get(k)!r}")
    if tuple(plan.get("trainable_prefixes", ())) != TRAINABLE_PREFIXES:
        raise ValueError("plan trainable prefixes mismatch")
    if plan.get("parent_optimizer_imported") or plan.get("parent_rng_imported"):
        raise ValueError("parent optimizer/RNG import prohibited")
    if plan.get("e2_final_split_evaluated") is not False:
        raise ValueError("E2 final split access is prohibited")
    if plan.get("final_eval_budget_seconds", 0) <= 0 or plan.get("logical_forward_examples_max", 0) <= 0:
        raise ValueError("finite final evaluation budget and forward cap must be sealed")
    root = Path(__file__).resolve().parents[1]
    config_path = Path(plan["config_path"])
    parent_marker = Path(plan["parent_marker"])
    checks = {"worker_sha256": sha256(worker_path),
        "supervisor_sha256": sha256(Path(__file__).with_name("supervise_role8_diffusion.py")),
        "io_sha256": sha256(Path(__file__).with_name("learning_io_v2.py")),
        "runtime_source_sha256": source_digest(), "config_sha256": sha256(config_path),
        "parent_marker_sha256": sha256(parent_marker),
        "train_data_sha256": sha256(Path(plan["train_data_path"])),
        "dev_data_sha256": sha256(Path(plan["dev_data_path"])),
        "candidate_manifest_sha256": sha256(Path(plan["candidate_manifest_path"]))}
    for key, relpath in (("readonly_probe_sha256", "probe_e1_readonly.py"),
                         ("rename_probe_sha256", "probe_role_coverage.py")):
        checks[key] = sha256(Path(__file__).with_name(relpath))
    mismatches = {k: (plan.get(k), v) for k, v in checks.items() if plan.get(k) != v}
    if mismatches: raise ValueError(f"plan SHA mismatch: {mismatches}")
    if plan.get("parent_sha256") != json.loads(parent_marker.read_text(encoding="utf-8")).get("sha256"):
        raise ValueError("plan parent checkpoint digest mismatch")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if cfg.get("variant") != "masked" or cfg.get("profile") != "full":
        raise ValueError("role8 requires full masked config")
    return cfg


def preflight(plan_path: Path, output: Path, worker_path: Path):
    plan = safe_io.load_json(plan_path)
    cfg_spec = contract_check(plan, plan_path, worker_path)
    from fdt_local.model_masked import Config
    config = Config(**{**asdict(Config()), **(cfg_spec.get("model_overrides") or {})})
    train = safe_io.load_json(Path(plan["train_data_path"]))
    dev = safe_io.load_json(Path(plan["dev_data_path"]))
    train_info = validate_rows(train, 512, config)
    dev_info = validate_rows(dev, 128, config)
    train_families = {r["source_graph_key"] for r in train}
    dev_families = {r["source_graph_key"] for r in dev}
    if train_families & dev_families: raise ValueError("train/dev family overlap")
    candidate = safe_io.load_json(Path(plan["candidate_manifest_path"]))
    if candidate.get("status") != "CPU_ROLE8_COVERAGE_CANDIDATE_NOT_TRAINED":
        raise ValueError("role8 data manifest status mismatch")
    if candidate.get("construction", {}).get("rendered_rows_are_independent_examples") is not False:
        raise ValueError("correlated-rotation declaration missing")
    parent_store = CheckpointStore(Path(plan["parent_marker"]).parent)
    meta, payload = parent_store.verify(Path(plan["parent_marker"]))
    if meta["sha256"] != plan["parent_sha256"] or meta["updates"] != 128:
        raise ValueError("E2 parent marker identity/counter mismatch")
    if payload.get("updates") != 128 or payload.get("attempts") != 128:
        raise ValueError("E2 parent payload counter mismatch")
    if meta.get("contract", {}).get("source") != source_digest():
        raise ValueError("parent frozen source SHA mismatch")
    from fdt_local.model_masked import FDT100
    model = FDT100(config)
    selected = configure_trainables(model)
    if not all(p.requires_grad for _, p in selected): raise ValueError("trainable allowlist failure")
    # CPU class weights that phase-2 Engine uses; counts are from the new training rows.
    from collections import Counter
    counts = Counter(int(r["labels"][0]) for r in train)
    weights = torch.tensor([counts[i] for i in range(4)], dtype=torch.float32).clamp_min(1).reciprocal()
    weights = (weights / weights.mean()).clamp(.25, 4)
    result = {"status": "PREFLIGHT_OK_NO_GPU_NO_LAUNCH", "parent_updates": meta["updates"],
        "parent_attempts": payload["attempts"], "train": train_info, "dev": dev_info,
        "train_label_counts": [counts[i] for i in range(4)], "phase2_class_weights": weights.tolist(),
        "config_phase_steps": plan["phase_steps"], "trainable_parameter_count": sum(p.numel() for _, p in selected),
        "all_rng_from_parent_imported": False, "final_split_read": False,
        "estimated_final_eval_budget_seconds": plan.get("final_eval_budget_seconds"),
        "plan_sha256": sha256(plan_path)}
    safe_io.atomic_json(output / "preflight.json", result)
    return result


@torch.no_grad()
def eval_diffusion(model, rows, *, seeds=(17,), batch_size=4):
    """DDIM4 diagnostics using only local generators; no training RNG is consumed."""
    device = "cuda"
    model.eval()
    out = {"initializer": {}, "diffusion": {}}
    for seed in seeds:
        y, h, m, init_preds, diff_preds = [], [], [], [], []
        started = time.monotonic(); forward_batches = 0
        for start in range(0, len(rows), batch_size):
            features, gold, _ = collate(rows[start:start + batch_size], model.c, device)
            if set(features) != FEATURE_KEYS: raise ValueError("feature whitelist mismatch")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                encoded = model.encode(features)
                init = model.readout(encoded["init"]).argmax(-1)
                gen = torch.Generator(device="cuda").manual_seed(int(seed) + start)
                noise = torch.randn(encoded["init"].shape, device="cuda", generator=gen)
                sampled = model.rollout(encoded, noise, schedule=(1., .75, .5, .25, 0.))
                diff = model.readout(sampled).argmax(-1)
            forward_batches += 1 + 4
            init_preds.append(init.cpu()); diff_preds.append(diff.cpu())
            y.append(gold["labels"].cpu()); h.append(gold["hops"].cpu()); m.append(features["query_mask"].cpu())
        labels, hops, mask = torch.cat(y), torch.cat(h), torch.cat(m)
        init_metrics = compute_metrics(torch.cat(init_preds), labels, mask, hops)
        diff_metrics = compute_metrics(torch.cat(diff_preds), labels, mask, hops)
        out["initializer"][str(seed)] = init_metrics
        out["diffusion"][str(seed)] = diff_metrics
        out.setdefault("instrumentation", {})[str(seed)] = {
            "rows": len(rows), "forward_batches": forward_batches,
            "estimated_model_forward_examples": len(rows) * 5,
            "elapsed_seconds": time.monotonic() - started, "ddim_steps": 4}
    model.train()
    return out


class ResilientCheckpointStore(CheckpointStore):
    def _rotate(self): safe_io.rotate_verified_generations(self)


def install_io_backend():
    from fdt_local import runner as core_runner, storage as core_storage
    core_storage.atomic_json = safe_io.atomic_json
    core_runner.atomic_json = safe_io.atomic_json
    core_runner.CheckpointStore = ResilientCheckpointStore


class Experiment:
    def __init__(self, plan, plan_path, output):
        self.plan, self.plan_path, self.out = plan, plan_path, output
        self.run_dir = Path(plan["run_dir"])
        self.train_rows = safe_io.load_json(Path(plan["train_data_path"]))
        self.dev_rows = safe_io.load_json(Path(plan["dev_data_path"]))
        self.deadline = float(safe_io.load_json(output / "supervisor" / "state.json")["deadline_unix"])
        self.training_deadline = self.deadline - int(plan["reserve_seconds"])
        safe_io.atomic_json(output / "clock.json", {"deadline_unix": self.deadline,
            "training_deadline_unix": self.training_deadline, "reserve_seconds": plan["reserve_seconds"],
            "immutable_budget": True})
        self.engine = self._build_engine()
        self.frozen_before = frozen_digest(self.engine.model)
        self.last_meta = self.last_marker = None
        self.last_saved = None
        self.history = []
        self.id_exposure = {k: [0] * 8 for k in
            ("fact_source", "fact_object", "query_source", "query_object")}
        self.baseline = eval_diffusion(self.engine.model, self.dev_rows, seeds=(17,))
        self.baseline_seconds = self.baseline["instrumentation"]["17"]["elapsed_seconds"]
        safe_io.atomic_json(self.out / "baseline_role8_dev.json", self.baseline)
        self.progress("ready", baseline_seconds=self.baseline_seconds)

    def _build_engine(self):
        config_path = Path(self.plan["config_path"])
        spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        spec["start_phase"] = 2
        spec["plan"] = {**spec["plan"], "seed": SEED, "phase_steps": list(PHASE_STEPS),
                        "diffusion_lr": float(self.plan["learning_rate"]),
                        "train_graphs": len(self.train_rows), "dev_graphs": len(self.dev_rows)}
        engine = Engine(spec, self.run_dir)
        if engine.model.counts()["parameters"] != 102237197: raise ValueError("masked model size mismatch")
        marker = Path(self.plan["parent_marker"])
        meta, payload = CheckpointStore(marker.parent).verify(marker)
        if meta["sha256"] != self.plan["parent_sha256"] or meta["updates"] != PARENT_UPDATES:
            raise ValueError("parent E2 marker mismatch")
        if payload["updates"] != PARENT_UPDATES or payload["attempts"] != PARENT_ATTEMPTS:
            raise ValueError("parent E2 counters mismatch")
        if meta["contract"].get("source") != source_digest(): raise ValueError("parent source digest mismatch")
        engine.model.load_state_dict(payload["model"], strict=True)
        del payload
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        engine.sampler = torch.Generator().manual_seed(SEED + 1)
        engine.noise = torch.Generator(device="cuda").manual_seed(SEED + 2)
        engine.phase, engine.phase_step, engine.updates, engine.attempts, engine.exposure = 2, 0, 0, 0, 0
        engine.query_exposure, engine.ground_exposure = [0] * 4, [0] * 4
        engine.gates, engine.history = [], []
        engine.train_data, engine.dev = self.train_rows, self.dev_rows
        engine.data_manifest = manifest({"train": self.train_rows, "dev": self.dev_rows})
        engine.plan = replace(engine.plan, seed=SEED, phase_steps=PHASE_STEPS,
            diffusion_lr=float(self.plan["learning_rate"]), train_graphs=len(self.train_rows), dev_graphs=len(self.dev_rows))
        configure_trainables(engine.model)
        engine._optimizer()
        if engine.optimizer.state: raise ValueError("fresh phase2 optimizer has nonempty state")
        counts = torch.tensor(engine.data_manifest["splits"]["train"]["query_labels"], dtype=torch.float32).clamp_min(1)
        engine.weights = (counts.reciprocal() / counts.reciprocal().mean()).clamp(.25, 4).to("cuda")
        engine.run_id = self.out.name
        engine.contract = {**engine.contract, "data": engine.data_manifest["fingerprint"],
            "plan": asdict(engine.plan), "local_branch": "role8-phase2-diffusion-20261027",
            "role8": {"parent_checkpoint_sha256": meta["sha256"], "weights_only_warmstart": True,
                "fresh_optimizer": True, "rng_seed": SEED, "trainable_prefixes": list(TRAINABLE_PREFIXES),
                "train_rows_sha256": self.plan["train_data_sha256"], "dev_rows_sha256": self.plan["dev_data_sha256"],
                "candidate_manifest_sha256": self.plan["candidate_manifest_sha256"],
                "worker_sha256": self.plan["worker_sha256"], "supervisor_sha256": self.plan["supervisor_sha256"],
                "io_sha256": self.plan["io_sha256"], "runtime_source_sha256": source_digest()}}
        engine.spec = spec
        safe_io.atomic_json(self.run_dir / "contract.json", engine.contract)
        safe_io.atomic_json(self.run_dir / "lineage.json", {"parent_checkpoint_sha256": meta["sha256"],
            "parent_updates": PARENT_UPDATES, "parent_attempts": PARENT_ATTEMPTS,
            "strict_old_optimizer_resume": False, "weights_only_warmstart": True})
        return engine

    def progress(self, stage, **extra):
        safe_io.atomic_json(self.out / "worker_progress.json", {"stage": stage,
            "observed_unix": time.time(), "worker_pid": os.getpid(), "phase": 2,
            "phase_step": self.engine.phase_step if hasattr(self, "engine") else 0,
            "attempts": self.engine.attempts if hasattr(self, "engine") else 0,
            "updates": self.engine.updates if hasattr(self, "engine") else 0,
            "target_successful_updates": TARGET_UPDATES, "max_attempts": MAX_ATTEMPTS,
            "last_durable_unix": self.last_saved, **extra})

    def save(self, tag=None, *, clean_boundary=True):
        self.last_meta, self.last_marker = self.engine.save(tag)
        self.last_saved = time.time()
        safe_io.atomic_json(self.out / "recovery.json", {"marker": str(self.last_marker),
            "sha256": self.last_meta["sha256"], "bytes": self.last_meta["bytes"],
            "stage_attempts": self.engine.attempts, "stage_updates": self.engine.updates,
            "phase": 2, "phase_step": self.engine.phase_step, "full_model_adam_scaler_all_rng": True,
            "clean_boundary": bool(clean_boundary), "save_seconds": self.last_meta["save_seconds"]})

    def train(self):
        self.save("weights_only_e2_warmstart_fresh_phase2_optimizer")
        diagnosed = set()
        while True:
            if (self.run_dir / "STOP").exists(): return "STOPPED", "COOPERATIVE_STOP", False
            if self.engine.updates >= TARGET_UPDATES: break
            if self.engine.attempts >= MAX_ATTEMPTS: return "STOPPED", "ATTEMPT_CAP", False
            if time.time() >= self.training_deadline: return "STOPPED", "TRAINING_DEADLINE", False
            entry = self.engine.step()
            self.history.append(entry)
            for batch in entry["batch_indices"]:
                for row_index in batch:
                    row = self.train_rows[int(row_index)]
                    for s, o, *_ in row["facts"]:
                        self.id_exposure["fact_source"][s] += 1
                        self.id_exposure["fact_object"][o] += 1
                    for s, o in row["queries"]:
                        self.id_exposure["query_source"][s] += 1
                        self.id_exposure["query_object"][o] += 1
            entry["actual_entity_id_exposure"] = json.loads(json.dumps(self.id_exposure))
            self.save()
            safe_io.atomic_json(self.out / "steps" / f"attempt_{self.engine.attempts:04d}.json",
                {**entry, "checkpoint_sha256": self.last_meta["sha256"], "phase": 2})
            if self.engine.phase_step in DIAGNOSTIC_STEPS and self.engine.phase_step not in diagnosed:
                diagnosed.add(self.engine.phase_step)
                if frozen_digest(self.engine.model) != self.frozen_before:
                    return "FAILED", "FROZEN_PARAMETERS_CHANGED", False
                start = time.monotonic(); score = eval_diffusion(self.engine.model, self.dev_rows, seeds=(17,))
                duration = time.monotonic() - start
                safe_io.atomic_json(self.out / "diagnostics" / f"role8_step{self.engine.phase_step:03d}.json",
                    {"updates": self.engine.updates, "attempts": self.engine.attempts,
                     "eval_seconds": duration, "panels": score, "final_holdout_used": False})
                self.progress("diagnosed", diagnostic_step=self.engine.phase_step,
                              diagnostic_seconds=duration)
        final_eval_start = time.monotonic()
        final = eval_diffusion(self.engine.model, self.dev_rows, seeds=(17, 29, 43))
        final_eval_seconds = time.monotonic() - final_eval_start
        self.save("final")
        safe_io.atomic_json(self.out / "final_evaluation.json", {"checkpoint_sha256": self.last_meta["sha256"],
            "panels": final, "elapsed_seconds": final_eval_seconds,
            "planned_budget_seconds": self.plan["final_eval_budget_seconds"],
            "final_holdout_used": False, "model_selection": False})
        return "COMPLETE", "TARGET_UPDATES_REACHED", True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true", help="CPU-only validation; never launches CUDA")
    args = parser.parse_args()
    plan_path, output, worker = args.plan.resolve(), args.output.resolve(), Path(__file__).resolve()
    if args.preflight:
        try:
            result = preflight(plan_path, output, worker)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:
            print(f"preflight failed: {exc}", file=sys.stderr)
            return 2
    install_io_backend()
    experiment = None; status, reason, readonly_complete = "FAILED", "INITIALIZATION_FAILURE", False
    plan = safe_io.load_json(plan_path)
    try:
        contract_check(plan, plan_path, worker)
        with ResilientCheckpointStore(Path(plan["run_dir"])).lock():
            experiment = Experiment(plan, plan_path, output)
            def request_stop(*_): safe_io.atomic_text(Path(plan["run_dir"]) / "STOP", "USER_SIGNAL\n")
            signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
            status, reason, readonly_complete = experiment.train()
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        if experiment:
            try: safe_io.atomic_text(Path(plan["run_dir"]) / "STOP", reason + "\n")
            except Exception: pass
            try: experiment.save("failed_preserve", clean_boundary=False)
            except Exception: traceback.print_exc()
        status = "FAILED"
    result = {"status": status, "reason": reason, "observed_unix": time.time(),
        "stage_attempts": experiment.engine.attempts if experiment else 0,
        "stage_updates": experiment.engine.updates if experiment else 0,
        "lineage_parent_updates": PARENT_UPDATES,
        "last_marker": str(experiment.last_marker) if experiment and experiment.last_marker else None,
        "last_checkpoint_sha256": experiment.last_meta["sha256"] if experiment and experiment.last_meta else None,
        "readonly_summary_complete": readonly_complete, "final_holdout_used": False,
        "strict_optimizer_resume": False, "weights_only_warmstart": True,
        "no_automatic_restart": True}
    safe_io.atomic_json(output / "final_status.json", result)
    return 0 if status in ("COMPLETE", "STOPPED") else 1


if __name__ == "__main__":
    raise SystemExit(main())



def eval_diffusion(model, rows, *, seeds=(17,), batch_size=4):
    """DDIM4 diagnostics using only local generators; no training RNG is consumed."""
    device = "cuda"
    model.eval()
    out = {"initializer": {}, "diffusion": {}}
    for seed in seeds:
        y, h, m, init_preds, diff_preds = [], [], [], [], []
        started = time.monotonic(); forward_batches = 0
        for start in range(0, len(rows), batch_size):
            features, gold, _ = collate(rows[start:start + batch_size], model.c, device)
            if set(features) != FEATURE_KEYS: raise ValueError("feature whitelist mismatch")
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                encoded = model.encode(features)
                init = model.readout(encoded["init"]).argmax(-1)
                gen = torch.Generator(device="cuda").manual_seed(int(seed) + start)
                noise = torch.randn(encoded["init"].shape, device="cuda", generator=gen)
                sampled = model.rollout(encoded, noise, schedule=(1., .75, .5, .25, 0.))
                diff = model.readout(sampled).argmax(-1)
            forward_batches += 1 + 4
            init_preds.append(init.cpu()); diff_preds.append(diff.cpu())
            y.append(gold["labels"].cpu()); h.append(gold["hops"].cpu()); m.append(features["query_mask"].cpu())
        labels, hops, mask = torch.cat(y), torch.cat(h), torch.cat(m)
        init_metrics = compute_metrics(torch.cat(init_preds), labels, mask, hops)
        diff_metrics = compute_metrics(torch.cat(diff_preds), labels, mask, hops)
        out["initializer"][str(seed)] = init_metrics
        out["diffusion"][str(seed)] = diff_metrics
        out.setdefault("instrumentation", {})[str(seed)] = {
            "rows": len(rows), "forward_batches": forward_batches,
            "estimated_model_forward_examples": len(rows) * 5,
            "elapsed_seconds": time.monotonic() - started, "ddim_steps": 4}
    model.train()
    return out


class ResilientCheckpointStore(CheckpointStore):
    def _rotate(self): safe_io.rotate_verified_generations(self)


def install_io_backend():
    from fdt_local import runner as core_runner, storage as core_storage
    core_storage.atomic_json = safe_io.atomic_json
    core_runner.atomic_json = safe_io.atomic_json
    core_runner.CheckpointStore = ResilientCheckpointStore


class Experiment:
    def __init__(self, plan, plan_path, output):
        self.plan, self.plan_path, self.out = plan, plan_path, output
        self.run_dir = Path(plan["run_dir"])
        self.train_rows = safe_io.load_json(Path(plan["train_data_path"]))
        self.dev_rows = safe_io.load_json(Path(plan["dev_data_path"]))
        self.deadline = float(safe_io.load_json(output / "supervisor" / "state.json")["deadline_unix"])
        self.training_deadline = self.deadline - int(plan["reserve_seconds"])
        safe_io.atomic_json(output / "clock.json", {"deadline_unix": self.deadline,
            "training_deadline_unix": self.training_deadline, "reserve_seconds": plan["reserve_seconds"],
            "immutable_budget": True})
        self.engine = self._build_engine()
        self.frozen_before = frozen_digest(self.engine.model)
        self.last_meta = self.last_marker = None
        self.last_saved = None
        self.history = []
        self.baseline = eval_diffusion(self.engine.model, self.dev_rows, seeds=(17,))
        self.baseline_seconds = self.baseline["instrumentation"]["17"]["elapsed_seconds"]
        safe_io.atomic_json(self.out / "baseline_role8_dev.json", self.baseline)
        self.progress("ready", baseline_seconds=self.baseline_seconds)

    def _build_engine(self):
        config_path = Path(self.plan["config_path"])
        spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        spec["start_phase"] = 2
        spec["plan"] = {**spec["plan"], "seed": SEED, "phase_steps": list(PHASE_STEPS),
                        "diffusion_lr": float(self.plan["learning_rate"]),
                        "train_graphs": len(self.train_rows), "dev_graphs": len(self.dev_rows)}
        engine = Engine(spec, self.run_dir)
        if engine.model.counts()["parameters"] != 102237197: raise ValueError("masked model size mismatch")
        marker = Path(self.plan["parent_marker"])
        meta, payload = CheckpointStore(marker.parent).verify(marker)
        if meta["sha256"] != self.plan["parent_sha256"] or meta["updates"] != PARENT_UPDATES:
            raise ValueError("parent E2 marker mismatch")
        if payload["updates"] != PARENT_UPDATES or payload["attempts"] != PARENT_ATTEMPTS:
            raise ValueError("parent E2 counters mismatch")
        if meta["contract"].get("source") != source_digest(): raise ValueError("parent source digest mismatch")
        engine.model.load_state_dict(payload["model"], strict=True)
        del payload
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        engine.sampler = torch.Generator().manual_seed(SEED + 1)
        engine.noise = torch.Generator(device="cuda").manual_seed(SEED + 2)
        engine.phase, engine.phase_step, engine.updates, engine.attempts, engine.exposure = 2, 0, 0, 0, 0
        engine.query_exposure, engine.ground_exposure = [0] * 4, [0] * 4
        engine.gates, engine.history = [], []
        engine.train_data, engine.dev = self.train_rows, self.dev_rows
        engine.data_manifest = manifest({"train": self.train_rows, "dev": self.dev_rows})
        engine.plan = replace(engine.plan, seed=SEED, phase_steps=PHASE_STEPS,
            diffusion_lr=float(self.plan["learning_rate"]), train_graphs=len(self.train_rows), dev_graphs=len(self.dev_rows))
        configure_trainables(engine.model)
        engine._optimizer()
        if engine.optimizer.state: raise ValueError("fresh phase2 optimizer has nonempty state")
        labels = torch.tensor([int(r["labels"][0]) for r in self.train_rows], dtype=torch.long)
        counts = torch.bincount(labels, minlength=4).float().clamp_min(1)
        engine.weights = counts.reciprocal(); engine.weights = (engine.weights / engine.weights.mean()).clamp(.25, 4).to("cuda")
        engine.run_id = self.out.name
        engine.contract = {**engine.contract, "data": engine.data_manifest["fingerprint"],
            "plan": asdict(engine.plan), "local_branch": "role8-phase2-diffusion-20261027",
            "role8": {"parent_checkpoint_sha256": meta["sha256"], "weights_only_warmstart": True,
                "fresh_optimizer": True, "rng_seed": SEED, "trainable_prefixes": list(TRAINABLE_PREFIXES),
                "train_rows_sha256": self.plan["train_data_sha256"], "dev_rows_sha256": self.plan["dev_data_sha256"],
                "candidate_manifest_sha256": self.plan["candidate_manifest_sha256"],
                "worker_sha256": self.plan["worker_sha256"], "supervisor_sha256": self.plan["supervisor_sha256"],
                "io_sha256": self.plan["io_sha256"], "runtime_source_sha256": source_digest()}}
        engine.spec = spec
        safe_io.atomic_json(self.run_dir / "contract.json", engine.contract)
        safe_io.atomic_json(self.run_dir / "lineage.json", {"parent_checkpoint_sha256": meta["sha256"],
            "parent_updates": PARENT_UPDATES, "parent_attempts": PARENT_ATTEMPTS,
            "strict_old_optimizer_resume": False, "weights_only_warmstart": True})
        return engine

    def progress(self, stage, **extra):
        safe_io.atomic_json(self.out / "worker_progress.json", {"stage": stage,
            "observed_unix": time.time(), "worker_pid": os.getpid(), "phase": 2,
            "phase_step": self.engine.phase_step if hasattr(self, "engine") else 0,
            "attempts": self.engine.attempts if hasattr(self, "engine") else 0,
            "updates": self.engine.updates if hasattr(self, "engine") else 0,
            "target_successful_updates": TARGET_UPDATES, "max_attempts": MAX_ATTEMPTS,
            "last_durable_unix": self.last_saved, **extra})

    def save(self, tag=None):
        self.last_meta, self.last_marker = self.engine.save(tag)
        self.last_saved = time.time()
        safe_io.atomic_json(self.out / "recovery.json", {"marker": str(self.last_marker),
            "sha256": self.last_meta["sha256"], "bytes": self.last_meta["bytes"],
            "stage_attempts": self.engine.attempts, "stage_updates": self.engine.updates,
            "phase": 2, "phase_step": self.engine.phase_step, "full_model_adam_scaler_all_rng": True,
            "clean_boundary": True, "save_seconds": self.last_meta["save_seconds"]})

    def train(self):
        self.save("weights_only_e2_warmstart_fresh_phase2_optimizer")
        diagnosed = set()
        while True:
            if (self.run_dir / "STOP").exists(): return "STOPPED", "COOPERATIVE_STOP", False
            if self.engine.updates >= TARGET_UPDATES: break
            if self.engine.attempts >= MAX_ATTEMPTS: return "STOPPED", "ATTEMPT_CAP", False
            if time.time() >= self.training_deadline: return "STOPPED", "TRAINING_DEADLINE", False
            entry = self.engine.step()
            self.history.append(entry)
            self.save()
            safe_io.atomic_json(self.out / "steps" / f"attempt_{self.engine.attempts:04d}.json",
                {**entry, "checkpoint_sha256": self.last_meta["sha256"], "phase": 2})
            if self.engine.phase_step in DIAGNOSTIC_STEPS and self.engine.phase_step not in diagnosed:
                diagnosed.add(self.engine.phase_step)
                if frozen_digest(self.engine.model) != self.frozen_before:
                    return "FAILED", "FROZEN_PARAMETERS_CHANGED", False
                start = time.monotonic(); score = eval_diffusion(self.engine.model, self.dev_rows, seeds=(17,))
                duration = time.monotonic() - start
                safe_io.atomic_json(self.out / "diagnostics" / f"role8_step{self.engine.phase_step:03d}.json",
                    {"updates": self.engine.updates, "attempts": self.engine.attempts,
                     "eval_seconds": duration, "panels": score, "final_holdout_used": False})
                self.progress("diagnosed", diagnostic_step=self.engine.phase_step,
                              diagnostic_seconds=duration)
        final_eval_start = time.monotonic()
        final = eval_diffusion(self.engine.model, self.dev_rows, seeds=(17, 29, 43))
        final_eval_seconds = time.monotonic() - final_eval_start
        self.save("final")
        safe_io.atomic_json(self.out / "final_evaluation.json", {"checkpoint_sha256": self.last_meta["sha256"],
            "panels": final, "elapsed_seconds": final_eval_seconds,
            "planned_budget_seconds": self.plan["final_eval_budget_seconds"],
            "final_holdout_used": False, "model_selection": False})
        return "COMPLETE", "TARGET_UPDATES_REACHED", True


