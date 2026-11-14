"""Explicit full-state continuation after Windows metadata replace failure.

The frozen package implements the model/optimizer. This separate experiment
controller adds periodic diagnostics and preserves its own sealed provenance.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import ctypes
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

import torch
import yaml

from fdt_local.data import collate, make_split
from fdt_local.objectives import PHASES, evaluate
from fdt_local.runner import Engine, source_digest
from fdt_local.storage import CheckpointStore, atomic_json, ensure_space, sha256
from fdt_local import storage as core_storage, runner as core_runner
import learning_io_v2 as safe_io


class ResilientCheckpointStore(CheckpointStore):
    def _rotate(self):
        safe_io.rotate_verified_generations(self)


# Explicit operational backend migration. The frozen source files themselves
# remain intact; the helper SHA and this controller are part of the NEW full
# execution contract below. No model/data/loss/optimizer function is patched.
def install_io_backend():
    global atomic_json
    core_storage.atomic_json = safe_io.atomic_json
    core_runner.atomic_json = safe_io.atomic_json
    core_runner.CheckpointStore = ResilientCheckpointStore
    atomic_json = safe_io.atomic_json

STATE_FIELDS = ("model", "optimizer", "scaler", "sampler_rng", "noise_rng", "torch_rng", "cuda_rng",
                "python_rng", "numpy_rng", "phase", "phase_step", "updates", "attempts", "exposure",
                "ground_exposure", "query_exposure", "history", "gates", "trainable_names")


def resume_state_digest(payload):
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            digest.update(b"tensor" + str(value.dtype).encode())
            digest.update(json.dumps(list(value.shape)).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict" + str(len(value)).encode())
            for key in sorted(value, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(value[key])
        elif isinstance(value, (tuple, list)):
            digest.update(type(value).__name__.encode() + str(len(value)).encode())
            for child in value:
                visit(child)
        else:
            digest.update(type(value).__name__.encode() + repr(value).encode() + b";")
    for name in STATE_FIELDS:
        visit(name)
        visit(payload[name])
    return digest.hexdigest()


def anomaly_reason(history):
    """Stop on numerical/ownership failures, not ordinary early underfitting."""
    if not history:
        return None
    row = history[-1]
    for field in ("loss",):
        if not math.isfinite(row[field]):
            return "NONFINITE_LOSS"
    if row.get("missing_gradients") or row.get("frozen_gradients"):
        return "GRADIENT_OWNERSHIP"
    if len(history) >= 3 and all(r["amp_skipped"] for r in history[-3:]):
        return "THREE_CONSECUTIVE_AMP_SKIPS"
    if len(history) >= 20 and sum(r["amp_skipped"] for r in history[-20:]) > 4:
        return "AMP_SKIP_RATE_ABOVE_20_PERCENT"
    if row["successful"] and (not row["finite_gradients"] or not math.isfinite(row["gradient_norm_before_clip"])):
        return "NONFINITE_SUCCESSFUL_GRADIENT"
    if len(history) >= 3 and all(r["gradient_norm_before_clip"] > 1000 for r in history[-3:]):
        return "REPEATED_EXTREME_PRECLIP_GRADIENT"
    if len(history) >= 23:
        before = sorted(r["loss"] for r in history[-23:-3])[10]
        if all(r["loss"] > max(10 * before, 100.) for r in history[-3:]):
            return "REPEATED_LOSS_EXPLOSION"
    return None


def interval_due(step):
    return step in (1, 64, 128, 256, 384, 480, 512, 640)


def raw_examples(model, items, noise_seed=17):
    rows = []
    model.eval()
    with torch.no_grad():
        for item in items:
            features, gold, _ = collate([item], model.c, "cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                encoded = model.encode(features)
                generator = torch.Generator(device="cuda").manual_seed(noise_seed)
                noise = torch.randn(encoded["init"].shape, device="cuda", generator=generator)
                refined = model.rollout(encoded, noise)
                count = len(item["queries"])
                initial = model.readout(encoded["init"]).argmax(-1)[0, :count].cpu().tolist()
                final = model.readout(refined).argmax(-1)[0, :count].cpu().tolist()
                ground = encoded["ground"].argmax(-1)[0, :len(item["facts"])].cpu().tolist()
            rows.append({"graph_key": item["graph_key"], "facts": item["facts"], "queries": item["queries"],
                         "gold": item["labels"], "hops": item["hops"], "ground_predictions": ground,
                         "initializer": initial, "diffusion": final, "noise_seed": noise_seed,
                         "gold_never_in_neural_input": True})
    return rows


@torch.no_grad()
def grounding_detail(model, items, batch_size):
    model.eval()
    confusion = torch.zeros(4, 4, dtype=torch.long)
    for start in range(0, len(items), batch_size):
        features, gold, _ = collate(items[start:start + batch_size], model.c, "cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            predictions = model.encode(features)["ground"].argmax(-1)
        mask = features["fact_mask"].bool()
        target, predicted = gold["ground"][mask].cpu(), predictions[mask].cpu()
        confusion += torch.bincount(4 * target + predicted, minlength=16).reshape(4, 4)
    support = confusion.sum(1)
    denominator = support + confusion.sum(0)
    f1 = 2 * confusion.diag().double() / denominator.clamp_min(1)
    return {"label_order": ["GREATER", "LESS", "NOT_GREATER", "NOT_LESS"],
            "confusion": confusion.tolist(), "support": support.tolist(),
            "class_f1": f1.tolist(), "macro_f1": float(f1.mean())}


class Experiment:
    def __init__(self, plan, plan_path, output):
        self.plan, self.plan_path, self.out = plan, plan_path, output
        self.run_dir = Path(plan["run_dir"])
        self.engine = None
        self.last_marker, self.last_meta, self.last_durable = None, None, None
        self.phase_history = []
        self.reason = None
        self.start = time.time()
        state_path = output / "supervisor/state.json"
        for _ in range(100):
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("deadline_unix"):
                    break
            time.sleep(.05)
        else:
            raise RuntimeError("owned supervisor deadline is required")
        self.deadline = min(float(state["deadline_unix"]), self.start + plan["wall_seconds"],
                            plan["absolute_deadline_unix"])
        self.training_deadline = self.deadline - plan["reserve_seconds"]
        atomic_json(output / "clock.json", {"started_unix": self.start, "deadline_unix": self.deadline,
                                            "training_deadline_unix": self.training_deadline,
                                            "immutable_budget": True})

    def progress(self, stage, **extra):
        record = {"observed_unix": time.time(), "stage": stage, "worker_pid": os.getpid(),
                  "phase": self.engine.phase if self.engine else None,
                  "phase_step": self.engine.phase_step if self.engine else 0,
                  "attempts": self.engine.attempts if self.engine else 0,
                  "updates": self.engine.updates if self.engine else 0,
                  "target_updates": sum(self.plan["phase_steps"]), "max_attempts": self.plan["max_attempts"],
                  "attempts_at_start": self.plan["attempts_at_start"],
                  "new_updates": self.engine.updates - self.plan["successful_at_start"] if self.engine else 0,
                  "global_max_attempts": self.plan["global_max_attempts"],
                  "deadline_unix": self.deadline, "last_durable_unix": self.last_durable,
                  "last_marker": str(self.last_marker) if self.last_marker else None,
                  "last_checkpoint_bytes": self.last_meta["bytes"] if self.last_meta else None,
                  "anomaly": None, **extra}
        atomic_json(self.out / "worker_progress.json", record)
        return record

    def save(self, tag=None):
        self.progress("saving")
        self.last_meta, self.last_marker = self.engine.save(tag)
        self.last_durable = time.time()
        atomic_json(self.out / "recovery.json", {"saved_unix": self.last_durable,
            "marker": str(self.last_marker), "sha256": self.last_meta["sha256"], "bytes": self.last_meta["bytes"],
            "attempts": self.engine.attempts, "updates": self.engine.updates,
            "phase": self.engine.phase, "phase_step": self.engine.phase_step,
            "full_model_adam_scaler_all_rng": True, "clean_boundary": True,
            "worker_sha256": self.plan["worker_sha256"], "plan_sha256": sha256(self.plan_path),
            "save_seconds": self.last_meta["save_seconds"]})

    def stop_reason(self):
        if (self.run_dir / "STOP").exists():
            return "COOPERATIVE_STOP"
        if time.time() >= self.training_deadline:
            return "FIXED_TRAINING_DEADLINE"
        if self.engine.attempts - self.plan["attempts_at_start"] >= self.plan["max_attempts"]:
            return "ATTEMPT_CAP"
        return None

    def diagnostic(self, complete_phase=False):
        if source_digest() != self.plan["runtime_source_sha256"] or sha256(Path(__file__)) != self.plan["worker_sha256"]:
            raise ValueError("sealed source changed during experiment")
        if sha256(Path(safe_io.__file__)) != self.plan["io_sha256"]:
            raise ValueError("operational I/O backend changed during experiment")
        self.progress("validating")
        phase = self.engine.phase
        items = self.engine.dev if complete_phase else self.engine.dev[:16]
        full = phase == 2
        before = time.monotonic()
        result = evaluate(self.engine.model, items, "cuda", self.engine.plan.micro_batch,
                          seeds=(17,), full=full)
        result["grounding_detail"] = grounding_detail(self.engine.model, items, self.engine.plan.micro_batch)
        record = {"observed_unix": time.time(), "phase": PHASES[phase],
                  "phase_step": self.engine.phase_step, "updates": self.engine.updates,
                  "checkpoint_sha256": self.last_meta["sha256"], "phase_gate": complete_phase,
                  "seconds": time.monotonic() - before, "metrics": result}
        path = self.out / "diagnostics" / f"phase{phase}_step{self.engine.phase_step:04d}.json"
        atomic_json(path, record)
        self.phase_history.append(record)
        self.progress("validated", latest_diagnostic=str(path))
        if complete_phase:
            score = result["ground_accuracy"] if phase == 0 else result["initializer"]["accuracy"]
            threshold = (self.engine.plan.ground_gate, self.engine.plan.semantic_gate, 0.)[phase]
            self.engine.gates.append({"phase": PHASES[phase], "measured": score,
                                      "required": threshold, "passed": score >= threshold})
            return score >= threshold
        return True

    def train(self):
        self.progress("initializing")
        spec = yaml.safe_load(Path(self.plan["config_path"]).read_text(encoding="utf-8"))
        self.engine = Engine(spec, self.run_dir)
        if self.engine.model.counts()["parameters"] != 102237197:
            raise ValueError("this authorized plan requires the actual full 102M model")
        parent_marker = Path(self.plan["parent_marker"])
        parent_store = CheckpointStore(parent_marker.parent)
        meta, parent = parent_store.verify(parent_marker, self.engine.contract)
        if meta["sha256"] != self.plan["parent_sha256"] or parent["updates"] != self.plan["imported_updates"]:
            raise ValueError("unexpected continuation checkpoint")
        expected = resume_state_digest(parent)
        self.engine.restore(parent)
        restored = resume_state_digest(self.engine.payload())
        if restored != expected:
            raise ValueError("model/Adam/scaler/RNG/counter import is not exact")
        # Importing the populated optimizer, LR progress, phase_step and all RNG
        # preserves the next batch. Identity and operational provenance change.
        self.engine.run_id = self.out.name
        self.engine.contract = {**self.engine.contract, "fresh_initialization": False,
            "local_branch": "fdt-local-v1-operational-v2", "operational_migration_v2": {
            "controller_sha256": self.plan["worker_sha256"], "io_sha256": self.plan["io_sha256"],
            "parent_checkpoint_sha256": meta["sha256"], "parent_updates": parent["updates"],
            "new_execution_plan_sha256": sha256(self.plan_path),
            "original_absolute_deadline_unix": self.plan["absolute_deadline_unix"]}}
        atomic_json(self.run_dir / "contract.json", self.engine.contract)
        generator = torch.Generator().set_state(parent["sampler_rng"])
        next_indices = [torch.randint(len(self.engine.train_data), (self.engine.plan.micro_batch,), generator=generator).tolist()
                        for _ in range(self.engine.plan.accumulation)]
        atomic_json(self.out / "import_verification.json", {"parent_sha256": meta["sha256"],
            "imported_updates": parent["updates"], "imported_attempts": parent["attempts"],
            "state_before_sha256": expected, "state_after_sha256": restored,
            "full_numeric_state_exact": True, "next_expected_batch_indices": next_indices,
            "original_core_contract_exact": True, "new_operational_contract": self.engine.contract,
            "strict_old_full_execution_contract_claimed": False, "first110_updates_retrained": False})
        self.next_expected_indices = next_indices
        self.imported_attempts = parent["attempts"]
        del parent
        atomic_json(self.out / "experiment_contract.json", {
            "plan_sha256": sha256(self.plan_path), "worker_sha256": sha256(Path(__file__)),
            "supervisor_sha256": self.plan["supervisor_sha256"], "core_contract": self.engine.contract,
            "continued_full_state_from_update110": True, "e2_candidate_used_for_training": False,
            "io_sha256": self.plan["io_sha256"], "operational_migration": True})
        self.save("imported_full_state")
        if time.time() < self.training_deadline:
            self.diagnostic()
        self.progress("training_ready")
        while self.engine.phase < 3:
            # Final phase checks are processed even when the successful-update
            # target was reached on the immediately preceding attempt.
            if self.engine.phase_step >= self.engine.plan.phase_steps[self.engine.phase]:
                if time.time() >= self.deadline - 120:
                    return "FIXED_BUDGET_BEFORE_PHASE_GATE"
                if not self.diagnostic(complete_phase=True):
                    self.save("gate_failed")
                    return "FUNCTIONAL_GATE"
                self.save(f"phase_{self.engine.phase}_passed")
                self.engine.phase += 1
                self.engine.phase_step = 0
                if self.engine.phase == 3:
                    self.save("final")
                    return "TRAINING_COMPLETE"
                self.engine._optimizer()
                self.phase_history = []
                self.save(f"phase_{self.engine.phase}_initial")
            reason = self.stop_reason()
            if reason:
                return reason
            self.progress("training")
            entry = self.engine.step()
            self.save()
            if self.next_expected_indices is not None:
                if entry["batch_indices"] != self.next_expected_indices:
                    raise ValueError("first continued batch differs from imported sampler state")
                atomic_json(self.out / "first_continued_update.json", {"updates": self.engine.updates,
                    "attempts": self.engine.attempts, "batch_indices": entry["batch_indices"],
                    "expected_batch_indices": self.next_expected_indices, "batch_match": True,
                    "checkpoint_sha256": self.last_meta["sha256"], "loss": entry["loss"]})
                self.next_expected_indices = None
            phase_rows = [r for r in self.engine.history if r["phase"] == PHASES[self.engine.phase]]
            anomaly = anomaly_reason(phase_rows)
            atomic_json(self.out / "steps" / f"attempt_{self.engine.attempts:05d}.json", {
                **entry, "observed_unix": time.time(), "checkpoint_sha256": self.last_meta["sha256"],
                "save_seconds": self.last_meta["save_seconds"], "last_durable_unix": self.last_durable})
            self.progress("training", successful=entry["successful"], loss=entry["loss"],
                          gradient_norm=entry["gradient_norm_before_clip"], anomaly=anomaly)
            print(json.dumps({"attempts": self.engine.attempts, "updates": self.engine.updates,
                              "phase": entry["phase"], "phase_step": entry["phase_step"],
                              "loss": entry["loss"], "save_seconds": self.last_meta["save_seconds"]}), flush=True)
            if anomaly:
                (self.run_dir / "STOP").write_text(anomaly, encoding="utf-8")
                return anomaly
            if entry["successful"] and interval_due(self.engine.phase_step) and self.engine.phase_step < self.engine.plan.phase_steps[self.engine.phase]:
                if not self.stop_reason():
                    self.diagnostic()
        return "TRAINING_COMPLETE"

    def final_evaluation(self):
        results = {}
        suite = self.plan["evaluation_suites"]
        for name, values in suite.items():
            if time.time() >= self.deadline - 60 or (self.run_dir / "STOP").exists():
                break
            self.progress("final_evaluation", evaluation_panel=name)
            items = make_split("test", values["graphs"], seed=values["seed"],
                               heldout=values.get("heldout", False), n_range=tuple(values.get("n_range", [4, 6])))
            measured = evaluate(self.engine.model, items, "cuda", self.engine.plan.micro_batch,
                                seeds=tuple(self.plan["evaluation_noise_seeds"]), full=True)
            raw = raw_examples(self.engine.model, items[:4])
            record = {"checkpoint_sha256": self.last_meta["sha256"], "suite": values,
                      "metrics": measured, "raw_first_four_graphs": raw,
                      "model_selection_final_holdout": False, "neural_before_verifier": True}
            atomic_json(self.out / "evaluation" / f"{name}.json", record)
            results[name] = measured
        atomic_json(self.out / "evaluation_summary.json", {"checkpoint_sha256": self.last_meta["sha256"],
            "complete": len(results) == len(suite), "completed_panels": results,
            "role_or_mask_causal_gain_claimed": False, "baseline_control_not_yet_trained_locally": True})
        return len(results) == len(suite)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path, output = args.plan.resolve(), args.output.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if output != Path(plan["output_dir"]).resolve() or output.joinpath("clock.json").exists():
        raise ValueError("new matching execution directory required")
    for path, expected in ((Path(__file__), plan["worker_sha256"]),
                           (Path(plan["config_path"]), plan["config_sha256"]),
                           (Path(safe_io.__file__), plan["io_sha256"])):
        if sha256(path) != expected:
            raise ValueError("worker or config changed after sealing")
    if source_digest() != plan["runtime_source_sha256"]:
        raise ValueError("frozen runtime changed")
    if plan["attempts_at_start"] + plan["max_attempts"] != plan["global_max_attempts"]:
        raise ValueError("remaining and global attempt caps disagree")
    if plan["successful_at_start"] + plan["target_successful_updates"] != sum(plan["phase_steps"]):
        raise ValueError("remaining and global successful-update targets disagree")
    install_io_backend()
    if Path(plan["run_dir"]).exists():
        raise FileExistsError("training must not overwrite or restart a run")
    ensure_space(output, 6 * 1024**3)
    held_awake = False
    experiment = None
    status, reason, evaluation_complete = "FAILED", "INITIALIZATION_FAILURE", False
    try:
        if os.name == "nt":
            held_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
        experiment = Experiment(plan, plan_path, output)
        def request_stop(*_):
            Path(plan["run_dir"]).mkdir(parents=True, exist_ok=True)
            Path(plan["run_dir"], "STOP").write_text("USER_SIGNAL", encoding="utf-8")
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)
        with CheckpointStore(plan["run_dir"]).lock():
            reason = experiment.train()
            status = "COMPLETE" if reason == "TRAINING_COMPLETE" else "STOPPED"
            if reason in ("TRAINING_COMPLETE", "FUNCTIONAL_GATE") and time.time() < experiment.deadline - 120:
                evaluation_complete = experiment.final_evaluation()
            experiment.progress("terminal", terminal_status=status, reason=reason)
    except BaseException as exc:
        reason = type(exc).__name__ + ": " + str(exc)
        traceback.print_exc()
        if experiment:
            try:
                experiment.progress("failed", anomaly=reason)
            except OSError:
                pass  # Final incident uses its own new destination below.
        status = "FAILED"
    finally:
        if held_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        atomic_json(output / "final_status.json", {"status": status, "reason": reason,
            "observed_unix": time.time(), "evaluation_complete": evaluation_complete,
            "attempts": experiment.engine.attempts if experiment and experiment.engine else 0,
            "updates": experiment.engine.updates if experiment and experiment.engine else 0,
            "last_marker": str(experiment.last_marker) if experiment and experiment.last_marker else None,
            "last_durable_unix": experiment.last_durable if experiment else None,
            "prior_verified_checkpoints_preserved": True, "no_automatic_restart": True})
    return 0 if status in ("COMPLETE", "STOPPED") else 1


if __name__ == "__main__":
    sys.exit(main())
