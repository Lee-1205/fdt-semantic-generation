"""E2 lexical grounding warm-start experiment; weights-only import, fresh optimizer."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
import math
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
from fdt_local.model_masked import Config
from fdt_local.objectives import masked_ce
from fdt_local.runner import Engine, source_digest, parameter_digest
from fdt_local.storage import CheckpointStore, ensure_space
from fdt_local import runner as core_runner, storage as core_storage
import learning_io_v2 as safe_io

FEATURE_KEYS = {"tokens", "roles", "fact_mask", "queries", "query_mask"}
GROUND_PREFIXES = ("language.", "phrase_router.", "ground_head.")
FROZEN_PREFIXES = ("anchors", "prior")
GOLD_PHASE_STEPS = (128, 320, 640)
DIAGNOSTIC_STEPS = (32, 64, 96, 128)
PARENT_E1_UPDATES = 1440


class ResilientCheckpointStore(CheckpointStore):
    def _rotate(self):
        safe_io.rotate_verified_generations(self)


def install_io_backend():
    core_storage.atomic_json = safe_io.atomic_json
    core_runner.atomic_json = safe_io.atomic_json
    core_runner.CheckpointStore = ResilientCheckpointStore


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith(FROZEN_PREFIXES):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configure_grounding_trainables(model):
    """Freeze every tensor except language, phrase_router and ground_head."""
    selected = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith(GROUND_PREFIXES)
        parameter.grad = None
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append((name, parameter))
    names = [name for name, _ in selected]
    if not selected or not all(any(name.startswith(prefix) for name in names) for prefix in GROUND_PREFIXES):
        raise ValueError(f"grounding trainable whitelist incomplete: {names[:8]}")
    unexpected = [name for name in names if not name.startswith(GROUND_PREFIXES)]
    if unexpected:
        raise ValueError(f"grounding trainable whitelist violation: {unexpected}")
    return selected


def fresh_grounding_optimizer(model, *, encoder_lr=1e-5, head_lr=3e-4):
    selected = configure_grounding_trainables(model)
    language = [parameter for name, parameter in selected if name.startswith("language.")]
    heads = [parameter for name, parameter in selected if not name.startswith("language.")]
    optimizer = torch.optim.AdamW([
        {"params": language, "lr": encoder_lr, "base_lr": encoder_lr},
        {"params": heads, "lr": head_lr, "base_lr": head_lr},
    ], weight_decay=0.01, foreach=False)
    if optimizer.state:
        raise AssertionError("fresh grounding optimizer unexpectedly has state")
    return optimizer, [name for name, _ in selected]


def validate_rows(rows, *, expected_count: int | None = None) -> dict:
    if not isinstance(rows, list) or (expected_count is not None and len(rows) != expected_count):
        raise ValueError(f"data row count mismatch: expected {expected_count}, got {len(rows) if isinstance(rows, list) else type(rows)}")
    ground = Counter()
    graph_ids = set()
    families = set()
    for index, row in enumerate(rows):
        required = {"n", "facts", "queries", "labels", "hops", "graph_key", "family_id"}
        if not isinstance(row, dict) or not required <= row.keys():
            raise ValueError(f"row {index} has an invalid schema")
        n, facts, queries, labels, hops = row["n"], row["facts"], row["queries"], row["labels"], row["hops"]
        if not isinstance(n, int) or not 4 <= n <= 8 or len(facts) != 8 or len(queries) != 1:
            raise ValueError(f"row {index} violates E2 8-fact/1-query capacity")
        if len(labels) != 1 or len(hops) != 1 or labels[0] not in (0, 1, 2, 3):
            raise ValueError(f"row {index} label shape is invalid")
        kinds = Counter()
        for fact in facts:
            if not isinstance(fact, (list, tuple)) or len(fact) != 4:
                raise ValueError(f"row {index} premise shape is invalid")
            source, target, kind, phrase = fact
            if not (isinstance(source, int) and isinstance(target, int) and source != target
                    and 0 <= source < n and 0 <= target < n and kind in (0, 1, 2, 3)
                    and isinstance(phrase, str)):
                raise ValueError(f"row {index} premise value is invalid")
            if len(phrase.encode("utf-8")) > Config().phrase_bytes:
                raise ValueError(f"row {index} phrase exceeds model capacity")
            kinds[kind] += 1
        if [kinds[kind] for kind in range(4)] != [2, 2, 2, 2]:
            raise ValueError(f"row {index} grounding kinds are not balanced")
        for source, target in queries:
            if not (isinstance(source, int) and isinstance(target, int) and source != target
                    and 0 <= source < n and 0 <= target < n):
                raise ValueError(f"row {index} query value is invalid")
        graph = row["graph_key"]
        if not isinstance(graph, str) or graph in graph_ids:
            raise ValueError(f"duplicate/invalid graph key at row {index}")
        graph_ids.add(graph)
        families.add(row["family_id"])
        ground.update(kinds)
    return {"rows": len(rows), "facts": len(rows) * 8,
            "grounding_counts": [ground[kind] for kind in range(4)],
            "families": sorted(str(value) for value in families),
            "graph_keys": graph_ids}


def neural_inputs(rows, config, device="cpu"):
    features, targets, byte_count = collate(rows, config, device)
    if set(features) != FEATURE_KEYS:
        raise ValueError(f"neural feature whitelist mismatch: {set(features)}")
    if any(key in features for key in ("labels", "ground", "hops", "family_id", "graph_key")):
        raise ValueError("supervision/metadata leaked into neural features")
    return features, targets, byte_count


def next_retention_streak(previous: int, old_ground_accuracy: float, threshold: float = 0.95,
                          old_initializer_accuracy: float | None = None,
                          initializer_threshold: float = 0.95) -> int:
    failed = old_ground_accuracy < threshold
    if old_initializer_accuracy is not None:
        failed = failed or old_initializer_accuracy < initializer_threshold
    return previous + 1 if failed else 0


def retention_stop(streak: int) -> bool:
    return streak >= 2


def training_stop_reason(*, attempts: int, updates: int, max_attempts: int, target_updates: int,
                         now: float, training_deadline: float) -> str | None:
    if updates >= target_updates:
        return "TARGET_UPDATES_REACHED"
    if attempts >= max_attempts:
        return "ATTEMPT_CAP"
    if now >= training_deadline:
        return "TRAINING_DEADLINE"
    return None


def stage_lineage(parent_updates: int, parent_attempts: int) -> dict:
    return {"parent_updates": int(parent_updates), "parent_attempts": int(parent_attempts),
            "stage_updates": 0, "stage_attempts": 0,
            "strict_optimizer_resume": False, "weights_only_warmstart": True}


def stage_plan(plan, *, seed: int, phase_steps: tuple, encoder_lr: float, head_lr: float,
               train_graphs: int, dev_graphs: int):
    return replace(plan, seed=seed, phase_steps=tuple(phase_steps), encoder_lr=encoder_lr,
                   ground_lr=head_lr, train_graphs=train_graphs, dev_graphs=dev_graphs)


def stage_spec(spec: dict, plan) -> dict:
    resolved = dict(spec)
    resolved["plan"] = asdict(plan)
    return resolved


def _file_spec(plan: dict, key: str) -> tuple[Path, str]:
    aliases = {"train_data": ("train_data_path", "train_data"),
               "dev_data": ("dev_data_path", "dev_data"),
               "candidate_manifest": ("candidate_manifest_path", "candidate_manifest")}
    raw = next((plan[name] for name in aliases[key] if name in plan), None)
    digest = plan.get(f"{key}_sha256") or plan.get(f"{key}_hash")
    if isinstance(raw, dict):
        digest = digest or raw.get("sha256")
        raw = raw.get("path")
    if not isinstance(raw, str) or not isinstance(digest, str):
        raise ValueError(f"plan lacks {key} absolute path/hash")
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"plan {key} path must be absolute")
    return path, digest.lower()


def load_candidate_inputs(plan: dict):
    paths, expected = {}, {}
    for key in ("train_data", "dev_data", "candidate_manifest"):
        paths[key], expected[key] = _file_spec(plan, key)
        if sha256(paths[key]) != expected[key]:
            raise ValueError(f"sealed {key} SHA256 mismatch")
    candidate = safe_io.load_json(paths["candidate_manifest"])
    if not isinstance(candidate, dict) or candidate.get("status") != "CPU_LEXICAL_CANDIDATE_NOT_TRAINED":
        raise ValueError("unexpected lexical candidate manifest")
    files = candidate.get("files", {})
    for key, filename in (("train_data", "train.json"), ("dev_data", "dev.json")):
        entry = files.get(filename)
        if not isinstance(entry, dict) or entry.get("sha256", "").lower() != expected[key]:
            raise ValueError(f"candidate manifest does not bind {filename}")
        if entry.get("bytes") != paths[key].stat().st_size:
            raise ValueError(f"candidate manifest {filename} size mismatch")
    train_rows = safe_io.load_json(paths["train_data"])
    dev_rows = safe_io.load_json(paths["dev_data"])
    train_info = validate_rows(train_rows, expected_count=int(plan.get("train_rows", 512)))
    dev_info = validate_rows(dev_rows, expected_count=int(plan.get("dev_rows", 64)))
    overlap = train_info["graph_keys"] & dev_info["graph_keys"]
    if overlap:
        raise ValueError("train/dev graph identity overlap")
    if set(train_info["families"]) & set(dev_info["families"]):
        raise ValueError("train/dev expression family overlap")
    return train_rows, dev_rows, candidate, paths, expected, train_info, dev_info


def _tensor_confusion_update(confusion: torch.Tensor, gold: torch.Tensor, pred: torch.Tensor):
    for expected, measured in zip(gold.reshape(-1).tolist(), pred.reshape(-1).tolist()):
        confusion[int(expected), int(measured)] += 1


def raw_row_identity(row: dict) -> dict:
    return {"family_id": row.get("family_id", "legacy_mixed"),
            "graph_key": row.get("graph_key"), "facts": row["facts"], "queries": row["queries"]}


@torch.no_grad()
def limited_grounding_eval(model, rows, config, *, include_initializer: bool, keep_raw: bool = False,
                           batch_size: int = 8) -> dict:
    """Grounding and optional initializer only; deliberately never runs diffusion."""
    model.eval()
    ground_conf = torch.zeros(4, 4, dtype=torch.long)
    init_conf = torch.zeros(4, 4, dtype=torch.long)
    raw = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        features, targets, _ = neural_inputs(batch, config, "cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            encoded = model.encode(features)
            ground_pred = encoded["ground"].argmax(-1)
            init_pred = model.readout(encoded["init"]).argmax(-1) if include_initializer else None
        valid_facts = features["fact_mask"].bool()
        _tensor_confusion_update(ground_conf, targets["ground"][valid_facts].cpu(), ground_pred[valid_facts].cpu())
        if include_initializer:
            valid_queries = features["query_mask"].bool()
            _tensor_confusion_update(init_conf, targets["labels"][valid_queries].cpu(), init_pred[valid_queries].cpu())
        if keep_raw:
            for offset, row in enumerate(batch):
                fact_mask = valid_facts[offset]
                row_out = {**raw_row_identity(row),
                           "ground_gold": targets["ground"][offset][fact_mask].cpu().tolist(),
                           "ground_pred": ground_pred[offset][fact_mask].cpu().tolist()}
                if include_initializer:
                    query_mask = features["query_mask"][offset].bool()
                    row_out["initializer_gold"] = targets["labels"][offset][query_mask].cpu().tolist()
                    row_out["initializer_pred"] = init_pred[offset][query_mask].cpu().tolist()
                raw.append(row_out)
    def metrics(confusion):
        support = confusion.sum(1).tolist()
        denom = confusion.sum(1) + confusion.sum(0)
        f1 = [float(2 * confusion[i, i] / denom[i]) if denom[i] else 0. for i in range(4)]
        total = int(confusion.sum())
        return {"count": total, "accuracy": float(confusion.diag().sum() / max(1, total)),
                "confusion": confusion.tolist(), "support": support,
                "class_f1": f1, "macro_f1": sum(f1) / 4}
    result = {"rows": len(rows), "grounding": metrics(ground_conf),
              "diffusion_evaluated": False, "initializer_evaluated": include_initializer}
    if include_initializer:
        result["initializer"] = metrics(init_conf)
    if keep_raw:
        result["raw_predictions"] = raw
    return result


def anomaly_reason(history):
    if not history:
        return None
    row = history[-1]
    if not math.isfinite(row.get("loss", float("nan"))):
        return "NONFINITE_LOSS"
    if row.get("missing_gradients") or row.get("frozen_gradients"):
        return "GRADIENT_OWNERSHIP"
    if len(history) >= 3 and all(item.get("amp_skipped") for item in history[-3:]):
        return "THREE_CONSECUTIVE_AMP_SKIPS"
    if len(history) >= 20 and sum(bool(item.get("amp_skipped")) for item in history[-20:]) > 4:
        return "AMP_SKIP_RATE_ABOVE_20_PERCENT"
    if row.get("successful") and (not row.get("finite_gradients") or not math.isfinite(row.get("gradient_norm_before_clip", float("nan")))):
        return "NONFINITE_SUCCESSFUL_GRADIENT"
    if len(history) >= 3 and all(item.get("gradient_norm_before_clip", 0) > 1000 for item in history[-3:]):
        return "REPEATED_EXTREME_PRECLIP_GRADIENT"
    if len(history) >= 23:
        before = sorted(item["loss"] for item in history[-23:-3])[10]
        if all(item["loss"] > max(10 * before, 100.) for item in history[-3:]):
            return "REPEATED_LOSS_EXPLOSION"
    return None


class Experiment:
    def __init__(self, plan, plan_path: Path, output: Path):
        self.plan, self.plan_path, self.out = plan, plan_path, output
        self.run_dir = Path(plan["run_dir"])
        self.train_rows, self.dev_rows, self.candidate_manifest, self.data_paths, self.data_hashes, self.train_info, self.dev_info = load_candidate_inputs(plan)
        self.original_dev = make_split("dev", 96)
        self.engine = self._build_engine()
        self.retention_streak = 0
        self.last_durable = None
        self.reason = None
        self.deadline, self.training_deadline = self._deadlines()
        self.baseline_retention = limited_grounding_eval(self.engine.model, self.original_dev,
            self.engine.cfg, include_initializer=True)
        self.retention_streak = next_retention_streak(0,
            self.baseline_retention["grounding"]["accuracy"],
            threshold=float(self.plan.get("old_retention_floor", 0.95)),
            old_initializer_accuracy=self.baseline_retention["initializer"]["accuracy"],
            initializer_threshold=float(self.plan.get("old_initializer_retention_floor", 0.95)))
        self.anchor_prior_sha256 = frozen_digest(self.engine.model)
        self.lineage = stage_lineage(plan["parent_updates"], plan["parent_attempts"])
        self.progress("ready", old_retention_ground_accuracy=self.baseline_retention["grounding"]["accuracy"],
                      old_retention_initializer_accuracy=self.baseline_retention["initializer"]["accuracy"],
                      frozen_anchor_prior_sha256=self.anchor_prior_sha256)
        safe_io.atomic_json(self.out / "baseline_retention.json", self.baseline_retention)

    def _deadlines(self):
        state = safe_io.load_json(self.out / "supervisor" / "state.json")
        if not isinstance(state, dict) or not isinstance(state.get("deadline_unix"), (int, float)):
            raise RuntimeError("owned supervisor deadline is required")
        launched = time.time()
        deadline = min(float(state["deadline_unix"]), launched + int(self.plan["wall_seconds"]))
        if self.plan.get("absolute_deadline_unix") is not None:
            deadline = min(deadline, float(self.plan["absolute_deadline_unix"]))
        training_deadline = deadline - int(self.plan["reserve_seconds"])
        safe_io.atomic_json(self.out / "clock.json", {"started_unix": launched,
            "deadline_unix": deadline, "training_deadline_unix": training_deadline,
            "reserve_seconds": self.plan["reserve_seconds"], "immutable_budget": True})
        return deadline, training_deadline

    def _build_engine(self):
        validate_locked_run_dir(self.run_dir)
        spec = yaml.safe_load(Path(self.plan["config_path"]).read_text(encoding="utf-8"))
        if spec.get("variant") != "masked" or spec.get("profile") != "full":
            raise ValueError("E2 requires the actual full masked model config")
        stage_steps = tuple(self.plan.get("phase_steps", [128, *spec["plan"]["phase_steps"][1:]]))
        if len(stage_steps) != 3 or stage_steps[0] != 128:
            raise ValueError("grounding stage must use phase_steps[0] == 128")
        spec["plan"] = {**spec["plan"], "seed": int(self.plan.get("seed", 20)),
                        "phase_steps": list(stage_steps), "encoder_lr": float(self.plan["encoder_lr"]),
                        "ground_lr": float(self.plan["head_router_lr"])}
        engine = Engine(spec, self.run_dir)
        if engine.model.counts()["parameters"] != 102237197:
            raise ValueError("E2 plan requires actual 102M parameter count")
        # Keep the E1 validation split exactly; only replace optimizer training/dev rows.
        original_dev = engine.dev
        parent_marker = Path(self.plan["parent_marker"])
        parent_store = CheckpointStore(parent_marker.parent)
        meta, payload = parent_store.verify(parent_marker)
        if meta["sha256"] != self.plan["parent_sha256"] or meta["updates"] != PARENT_E1_UPDATES:
            raise ValueError("E1 final marker identity/update mismatch")
        if payload["updates"] != PARENT_E1_UPDATES or payload["attempts"] != PARENT_E1_UPDATES:
            raise ValueError("E1 parent payload counters mismatch")
        parent_contract = meta["contract"]
        if parent_contract.get("variant") != "masked" or parent_contract.get("source") != source_digest():
            raise ValueError("E1 parent model/source contract mismatch")
        expected_weights = state_digest(payload["model"])
        engine.model.load_state_dict(payload["model"], strict=True)
        if state_digest(engine.model.state_dict()) != expected_weights:
            raise ValueError("weights-only warmstart did not strictly reproduce parent model")
        self.parent_weights_sha256 = expected_weights
        self.parent_model_counts = engine.model.counts()
        del payload

        # New stage counters and RNG. Do not import parent optimizer, scaler or RNG.
        seed = int(self.plan.get("seed", 20))
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        engine.sampler = torch.Generator().manual_seed(seed + 1)
        engine.noise = torch.Generator(device="cuda").manual_seed(seed + 2)
        engine.phase = 0
        engine.phase_step = engine.updates = engine.attempts = engine.exposure = 0
        engine.query_exposure = [0] * 4; engine.ground_exposure = [0] * 4
        engine.gates = []; engine.history = []
        engine.train_data = self.train_rows
        engine.dev = self.dev_rows
        engine.data_manifest = manifest({"train": self.train_rows, "dev": self.dev_rows})
        engine.plan = stage_plan(engine.plan, seed=seed, phase_steps=stage_steps,
            encoder_lr=float(self.plan["encoder_lr"]), head_lr=float(self.plan["head_router_lr"]),
            train_graphs=len(self.train_rows), dev_graphs=len(self.dev_rows))
        optimizer, trainable_names = fresh_grounding_optimizer(engine.model,
            encoder_lr=float(self.plan["encoder_lr"]), head_lr=float(self.plan["head_router_lr"]))
        engine.optimizer = optimizer
        engine.scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.)
        engine.ground_weights = torch.ones(4, device="cuda", dtype=torch.float32)
        if engine.optimizer.state:
            raise ValueError("fresh phase-0 AdamW state must be empty")
        self.trainable_names = trainable_names
        self.original_dev = original_dev

        migration = {"kind": "weights_only_grounding_warmstart", "parent_checkpoint_sha256": meta["sha256"],
            "parent_updates": meta["updates"], "parent_attempts": meta["attempts"],
            "parent_model_weight_sha256": self.parent_weights_sha256, "strict_model_load": True,
            "optimizer": "fresh_AdamW_foreach_false", "scaler": "fresh_GradScaler",
            "rng_seed": seed, "stage_counters_start_at_zero": True,
            "trainable_prefixes": list(GROUND_PREFIXES), "frozen_prefixes": list(FROZEN_PREFIXES),
            "core_contract_claim": "model/data/loss implementation reused; E2 data and optimizer contract are new; not strict E1 optimizer resume"}
        engine.run_id = self.out.name
        engine.contract = {**engine.contract, "data": engine.data_manifest["fingerprint"],
            "variant": "masked", "fresh_initialization": False,
            "plan": asdict(engine.plan),
            "local_branch": "fdt-local-v1-e2-lexical-grounding",
            "e2_execution": {"plan_sha256": sha256(self.plan_path),
                "worker_sha256": self.plan["worker_sha256"], "supervisor_sha256": self.plan["supervisor_sha256"],
                "io_sha256": self.plan["io_sha256"], "core_sha256": self.plan.get("core_sha256", source_digest()),
                "candidate_manifest_sha256": self.data_hashes["candidate_manifest"],
                "train_data_sha256": self.data_hashes["train_data"], "dev_data_sha256": self.data_hashes["dev_data"],
                "trainable_names": trainable_names, "fresh_optimizer": True, "migration": migration,
                "prospective_old_retention_floors": {"grounding": float(self.plan.get("old_retention_floor", 0.95)),
                    "initializer": float(self.plan.get("old_initializer_retention_floor", 0.95)),
                    "consecutive_failed_observations_to_stop": 2},
                "retention_dev_graphs": len(original_dev),
                "old_development_families_promoted_to_training": self.candidate_manifest.get("old_development_families_explicitly_promoted_to_training", [])}}
        spec = stage_spec(spec, engine.plan)
        engine.spec = spec
        for name, value in (("resolved_config.json", {**spec,
                "base_config_sha256": self.plan["config_sha256"],
                "stage_phase_steps": list(stage_steps), "optimizer": "fresh_grounding_only"}),
                ("contract.json", engine.contract), ("data_manifest.json", engine.data_manifest),
                ("e2_data_provenance.json", {"candidate_manifest_sha256": self.data_hashes["candidate_manifest"],
                    "train_data_sha256": self.data_hashes["train_data"], "dev_data_sha256": self.data_hashes["dev_data"],
                    "train_rows": self.train_info["rows"], "dev_rows": self.dev_info["rows"],
                    "old_development_families_promoted_to_training": self.candidate_manifest.get(
                        "old_development_families_explicitly_promoted_to_training", [])}),
                ("model_manifest.json", engine.model.counts()), ("lineage.json", {
                    "parent_checkpoint_sha256": meta["sha256"], "parent_updates": meta["updates"],
                    "parent_attempts": meta["attempts"], "parent_model_weight_sha256": self.parent_weights_sha256,
                    "stage_updates_start": 0, "stage_attempts_start": 0,
                    "strict_old_optimizer_resume": False, "weights_only_warmstart": True})):
            safe_io.atomic_json(self.run_dir / name, value)
        return engine

    def progress(self, stage: str, **extra):
        engine = getattr(self, "engine", None)
        record = {"observed_unix": time.time(), "stage": stage,
            "worker_pid": os.getpid(), "phase": 0,
            "phase_step": engine.phase_step if engine else 0,
            "attempts": engine.attempts if engine else 0,
            "updates": engine.updates if engine else 0,
            "target_successful_updates": int(self.plan["target_successful_updates"]),
            "max_attempts": int(self.plan["max_attempts"]),
            "lineage_parent_updates": int(self.plan["parent_updates"]),
            "lineage_parent_attempts": int(self.plan["parent_attempts"]),
            "new_updates": engine.updates if engine else 0,
            "deadline_unix": getattr(self, "deadline", None),
            "training_deadline_unix": getattr(self, "training_deadline", None),
            "last_durable_unix": self.last_durable,
            "last_marker": str(self.last_marker) if getattr(self, "last_marker", None) else None,
            "last_checkpoint_bytes": self.last_meta["bytes"] if getattr(self, "last_meta", None) else None,
            "anomaly": None, **extra}
        safe_io.atomic_json(self.out / "worker_progress.json", record)
        return record

    def save(self, tag=None):
        self.progress("saving")
        self.last_meta, self.last_marker = self.engine.save(tag)
        self.last_durable = time.time()
        safe_io.atomic_json(self.out / "recovery.json", {"saved_unix": self.last_durable,
            "marker": str(self.last_marker), "sha256": self.last_meta["sha256"], "bytes": self.last_meta["bytes"],
            "stage_attempts": self.engine.attempts, "stage_updates": self.engine.updates,
            "phase": 0, "phase_step": self.engine.phase_step, "full_model_adam_scaler_all_rng": True,
            "clean_boundary": True, "worker_sha256": self.plan["worker_sha256"],
            "plan_sha256": sha256(self.plan_path), "save_seconds": self.last_meta["save_seconds"],
            "parent_checkpoint_sha256": self.plan["parent_sha256"], "optimizer_fresh": True})

    def stop_reason(self):
        if (self.run_dir / "STOP").exists():
            return "COOPERATIVE_STOP"
        return training_stop_reason(attempts=self.engine.attempts, updates=self.engine.updates,
            max_attempts=int(self.plan["max_attempts"]), target_updates=int(self.plan["target_successful_updates"]),
            now=time.time(), training_deadline=self.training_deadline)

    def check_frozen(self):
        current = frozen_digest(self.engine.model)
        if current != self.anchor_prior_sha256:
            raise RuntimeError("FROZEN_ANCHOR_OR_PRIOR_CHANGED")

    def diagnose(self, step: int):
        self.check_frozen()
        new_dev = limited_grounding_eval(self.engine.model, self.dev_rows, self.engine.cfg,
                                         include_initializer=False)
        old_dev = limited_grounding_eval(self.engine.model, self.original_dev, self.engine.cfg,
                                         include_initializer=True)
        self.retention_streak = next_retention_streak(self.retention_streak,
            old_dev["grounding"]["accuracy"], threshold=float(self.plan.get("old_retention_floor", 0.95)),
            old_initializer_accuracy=old_dev["initializer"]["accuracy"],
            initializer_threshold=float(self.plan.get("old_initializer_retention_floor", 0.95)))
        record = {"stage_step": step, "stage_updates": self.engine.updates,
            "stage_attempts": self.engine.attempts, "observed_unix": time.time(),
            "candidate_manifest_sha256": self.data_hashes["candidate_manifest"],
            "new_dev_grounding": new_dev["grounding"],
            "old_e1_dev96_retention": old_dev,
            "old_retention_floor": float(self.plan.get("old_retention_floor", 0.95)),
            "old_initializer_retention_floor": float(self.plan.get("old_initializer_retention_floor", 0.95)),
            "old_initializer_retention_accuracy": old_dev["initializer"]["accuracy"],
            "old_retention_fail_streak": self.retention_streak,
            "frozen_anchor_prior_sha256": self.anchor_prior_sha256,
            "diffusion_evaluated": False, "e2_final_model_evaluation": False,
            "e2_final_split_used": False}
        safe_io.atomic_json(self.out / "diagnostics" / f"grounding_step{step:03d}.json", record)
        self.progress("validated", latest_diagnostic=str(self.out / "diagnostics" / f"grounding_step{step:03d}.json"),
                      old_retention_fail_streak=self.retention_streak,
                      new_dev_ground_accuracy=new_dev["grounding"]["accuracy"])
        return record

    def final_readonly(self):
        self.check_frozen()
        # Small old/seen snapshots and the complete new dev split. No diffusion or E2 final split.
        panels = {
            "old_e1_dev16": limited_grounding_eval(self.engine.model, self.original_dev[:16], self.engine.cfg,
                include_initializer=True, keep_raw=True),
            "e2_train16_seen": limited_grounding_eval(self.engine.model, self.train_rows[:16], self.engine.cfg,
                include_initializer=True, keep_raw=True),
            "e2_dev64_unseen": limited_grounding_eval(self.engine.model, self.dev_rows, self.engine.cfg,
                include_initializer=True, keep_raw=True),
        }
        safe_io.atomic_json(self.out / "posttraining_readonly.json", {
            "checkpoint_sha256": self.last_meta["sha256"], "panels": panels,
            "diffusion_evaluated": False, "e2_final_split_evaluated": False,
            "final_model_selection": False, "causal_data_or_mask_gain_claimed": False,
            "old_e1_dev96_previously_used_for_development": True,
            "old_development_families_promoted_to_training": self.candidate_manifest.get("old_development_families_explicitly_promoted_to_training", [])})
        return panels

    def train(self):
        self.save("weights_only_warmstart_fresh_optimizer")
        self.progress("training")
        evaluated_steps = set()
        status, reason = "STOPPED", "UNKNOWN"
        while True:
            stop = self.stop_reason()
            if stop:
                reason = stop
                status = "COMPLETE" if stop == "TARGET_UPDATES_REACHED" else "STOPPED"
                break
            self.progress("training")
            entry = self.engine.step()
            self.save()
            safe_io.atomic_json(self.out / "steps" / f"attempt_{self.engine.attempts:04d}.json", {
                **entry, "observed_unix": time.time(), "checkpoint_sha256": self.last_meta["sha256"],
                "lineage_parent_updates": self.plan["parent_updates"],
                "local_stage_update": self.engine.updates, "save_seconds": self.last_meta["save_seconds"]})
            anomaly = anomaly_reason(self.engine.history)
            if anomaly:
                reason = anomaly
                (self.run_dir / "STOP").write_text(anomaly + "\n", encoding="utf-8")
                status = "STOPPED"
                break
            if self.engine.phase_step in DIAGNOSTIC_STEPS and self.engine.phase_step not in evaluated_steps:
                evaluated_steps.add(self.engine.phase_step)
                diagnostic = self.diagnose(self.engine.phase_step)
                if retention_stop(self.retention_streak):
                    reason, status = "OLD_RETENTION_BELOW_95_TWICE", "STOPPED"
                    (self.run_dir / "STOP").write_text(reason + "\n", encoding="utf-8")
                    break
                if self.engine.phase_step >= 128:
                    reason, status = "TARGET_UPDATES_REACHED", "COMPLETE"
                    break
            if not all(torch.isfinite(parameter).all().item() for parameter in self.engine.model.parameters()):
                reason, status = "NONFINITE_PARAMETER", "FAILED"
                (self.run_dir / "STOP").write_text(reason + "\n", encoding="utf-8")
                break
        self.reason = reason
        # Preserve a terminal full-state marker, including fresh optimizer/scaler/RNG.
        self.save("final")
        readonly_complete = False
        if status == "COMPLETE" and time.time() < self.deadline - 30:
            self.final_readonly()
            readonly_complete = True
        return status, reason, readonly_complete


def validate_new_stage_paths(output: Path, run_dir: Path) -> None:
    if run_dir.exists():
        raise FileExistsError("E2 model run directory must be new and nonexistent")
    if output.exists() and any((output / name).exists() for name in
                               ("STOP", "clock.json", "final.json", "final_status.json")):
        raise FileExistsError("E2 output already contains terminal markers")


def validate_locked_run_dir(run_dir: Path) -> None:
    """Accept only the directory and writer.lock created by our held lock."""
    if not run_dir.is_dir():
        raise FileNotFoundError("E2 writer lock did not create its run directory")
    entries = list(run_dir.iterdir())
    if len(entries) != 1 or entries[0].name != "writer.lock" or not entries[0].is_file():
        raise FileExistsError("E2 run directory contains data beyond its newly acquired writer lock")


def verify_plan(plan_path: Path, worker_path: Path, output: Path) -> dict:
    plan = safe_io.load_json(plan_path)
    if not isinstance(plan, dict) or plan.get("authorized") is not True:
        raise ValueError("authorized E2 plan required")
    if Path(plan["output_dir"]).resolve() != output.resolve():
        raise ValueError("output path does not match sealed plan")
    if Path(plan["run_dir"]).resolve() != (output / "model").resolve():
        raise ValueError("E2 run_dir must be output/model")
    validate_new_stage_paths(output, Path(plan["run_dir"]))
    checks = {
        "worker_sha256": sha256(worker_path),
        "supervisor_sha256": sha256(Path(__file__).with_name("supervise_e2_grounding.py")),
        "io_sha256": sha256(Path(safe_io.__file__)),
        "runtime_source_sha256": source_digest(),
        "config_sha256": sha256(Path(plan["config_path"])),
    }
    if plan.get("core_sha256", checks["runtime_source_sha256"]) != checks["runtime_source_sha256"]:
        raise ValueError("frozen core SHA mismatch")
    mismatches = {key: (plan.get(key), actual) for key, actual in checks.items() if plan.get(key) != actual}
    if mismatches:
        raise ValueError(f"sealed execution digest mismatch: {mismatches}")
    if (plan.get("max_attempts") != 140 or plan.get("target_successful_updates") != 128
            or plan.get("attempts_at_start") != 0 or plan.get("successful_at_start") != 0):
        raise ValueError("E2 relative stage budget must be 140 attempts/128 updates from local zero")
    if not 0 < int(plan["wall_seconds"]) <= 1800 or int(plan["reserve_seconds"]) != 180:
        raise ValueError("E2 time budget must be <=1800s with 180s reserve")
    if (float(plan.get("old_retention_floor", -1)) != 0.95
            or float(plan.get("old_initializer_retention_floor", -1)) != 0.95):
        raise ValueError("E2 plan must seal both old-dev grounding and initializer retention floors at 0.95")
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_path, output = args.plan.resolve(), args.output.resolve()
    install_io_backend()
    experiment = None
    status, reason, readonly_complete = "FAILED", "INITIALIZATION_FAILURE", False
    try:
        plan = verify_plan(plan_path, Path(__file__).resolve(), output)
        with ResilientCheckpointStore(plan["run_dir"]).lock():
            experiment = Experiment(plan, plan_path, output)
            def request_stop(*_):
                safe_io.atomic_text(Path(plan["run_dir"]) / "STOP", "USER_SIGNAL\n")
            signal.signal(signal.SIGINT, request_stop)
            signal.signal(signal.SIGTERM, request_stop)
            status, reason, readonly_complete = experiment.train()
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        if experiment:
            try:
                safe_io.atomic_text(Path(plan["run_dir"]) / "STOP", reason + "\n")
            except Exception:
                pass
            try:
                experiment.save("failed_preserve")
            except Exception:
                traceback.print_exc()
        status = "FAILED"
    finally:
        result = {"status": status, "reason": reason, "observed_unix": time.time(),
            "stage_attempts": experiment.engine.attempts if experiment else 0,
            "stage_updates": experiment.engine.updates if experiment else 0,
            "lineage_parent_updates": plan.get("parent_updates", PARENT_E1_UPDATES) if "plan" in locals() else PARENT_E1_UPDATES,
            "last_marker": str(experiment.last_marker) if experiment and experiment.last_marker else None,
            "last_checkpoint_sha256": experiment.last_meta["sha256"] if experiment and experiment.last_meta else None,
            "readonly_summary_complete": readonly_complete,
            "diffusion_evaluated": False, "e2_final_split_evaluated": False,
            "final_model_selection": False, "strict_e1_optimizer_resume": False,
            "weights_only_warmstart": True, "no_automatic_restart": True}
        try:
            safe_io.atomic_json(output / "final_status.json", result)
        except Exception as exc:
            incident = output / f"worker_incident_{time.time_ns()}.json"
            try:
                incident.parent.mkdir(parents=True, exist_ok=True)
                with incident.open("x", encoding="utf-8") as stream:
                    json.dump({"event": "worker_final_status_write_error", "error": repr(exc), "final": result}, stream, indent=2)
                    stream.flush(); os.fsync(stream.fileno())
            except Exception:
                print(f"final-status and incident persistence failed: {exc!r}", file=sys.stderr)
    return 0 if status in ("COMPLETE", "STOPPED") else 1


if __name__ == "__main__":
    raise SystemExit(main())
