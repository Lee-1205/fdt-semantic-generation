"""CUDA-only local execution; original objectives, new durable contract."""
from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

from . import model as baseline
from . import model_masked as masked
from .data import LABELS, collate, make_split, manifest
from .objectives import Plan, PHASES, diffusion_loss, evaluate, masked_ce, masked_mse
from .storage import CheckpointStore, atomic_json, canonical, ensure_space


def source_digest():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def configure_determinism():
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def environment():
    return {"python": platform.python_version(), "os": platform.system(),
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "compute_capability": list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None,
            "deterministic_algorithms": True, "sdp_backend": "math", "tf32": False}


def numpy_state():
    name, keys, position, gaussian, cached = np.random.get_state()
    return {"name": name, "keys": keys.tolist(), "position": position,
            "gaussian": gaussian, "cached": cached}


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class Engine:
    def __init__(self, spec, directory, resume=False, marker=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU training fallback is disabled")
        if tuple(torch.cuda.get_device_capability()) not in ((7, 5), (8, 0), (8, 6), (8, 9), (9, 0), (12, 0)):
            raise RuntimeError("CUDA device capability has not been selected for this local branch")
        configure_determinism()
        self.spec = spec
        self.out = Path(directory).resolve()
        self.store = CheckpointStore(self.out)
        self.run_id = self.out.name
        self.plan = Plan(**{**spec["plan"], "phase_steps": tuple(spec["plan"]["phase_steps"])})
        module = masked if spec.get("variant") == "masked" else baseline
        self.cfg = module.Config.smoke() if spec.get("profile") == "tiny" else module.Config()
        overrides = spec.get("model_overrides") or {}
        self.cfg = module.Config(**{**asdict(self.cfg), **overrides})
        random.seed(self.plan.seed)
        np.random.seed(self.plan.seed)
        torch.manual_seed(self.plan.seed)
        torch.cuda.manual_seed_all(self.plan.seed)
        self.model = module.FDT100(self.cfg).cuda()
        self.train_data = make_split("train", self.plan.train_graphs)
        self.dev = make_split("dev", self.plan.dev_graphs)
        self.data_manifest = manifest({"train": self.train_data, "dev": self.dev})
        self.contract = {"source": source_digest(), "config": asdict(self.cfg),
                         "variant": spec.get("variant", "baseline"), "plan": asdict(self.plan),
                         "data": self.data_manifest["fingerprint"], "environment": environment(),
                         "local_branch": "fdt-local-v1", "fresh_initialization": True}
        self.weights = torch.tensor(self.data_manifest["splits"]["train"]["query_labels"], device="cuda", dtype=torch.float32).clamp_min(1).reciprocal()
        self.weights = (self.weights / self.weights.mean()).clamp(.25, 4)
        self.ground_weights = torch.tensor(self.data_manifest["splits"]["train"]["ground_labels"], device="cuda", dtype=torch.float32).clamp_min(1).reciprocal()
        self.ground_weights /= self.ground_weights.mean()
        self.phase = int(spec.get("start_phase", 0))
        self.phase_step = self.updates = self.attempts = self.exposure = 0
        self.query_exposure, self.ground_exposure = [0] * 4, [0] * 4
        self.gates, self.history = [], []
        self.sampler = torch.Generator().manual_seed(self.plan.seed + 1)
        self.noise = torch.Generator(device="cuda").manual_seed(self.plan.seed + 2)
        self._optimizer()
        if resume:
            if marker:
                _, payload = self.store.verify(marker, self.contract)
            else:
                _, payload, _ = self.store.recover(self.contract)
            self.restore(payload)
        elif self.store.markers():
            raise FileExistsError("existing run cannot be overwritten")
        atomic_json(self.out / "resolved_config.json", spec)
        atomic_json(self.out / "contract.json", self.contract)
        atomic_json(self.out / "data_manifest.json", self.data_manifest)
        atomic_json(self.out / "model_manifest.json", self.model.counts())

    def _optimizer(self):
        prefixes = (
            ("language.", "phrase_router.", "ground_head.", "anchors", "prior"),
            ("language.", "phrase_router.", "ground_head.", "anchors", "prior", "value.", "condition.", "role_project.", "initializer.", "initial_out."),
            ("slot_router.", "memory_project.", "rel_left", "rel_right", "rel_score.", "rel_message.", "time.", "refiner.", "refine_norm.", "velocity."),
        )[min(self.phase, 2)]
        encoder, rest = [], []
        for name, parameter in self.model.named_parameters():
            parameter.grad = None
            parameter.requires_grad_(name.startswith(prefixes))
            if parameter.requires_grad:
                (encoder if name.startswith("language.") else rest).append(parameter)
        rate = (self.plan.ground_lr, self.plan.semantic_lr, self.plan.diffusion_lr)[min(self.phase, 2)]
        groups = []
        if encoder:
            groups.append({"params": encoder, "lr": self.plan.encoder_lr, "base_lr": self.plan.encoder_lr})
        if rest:
            groups.append({"params": rest, "lr": rate, "base_lr": rate})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=.01, foreach=False)
        self.scaler = torch.amp.GradScaler("cuda", enabled=True, init_scale=1024.)

    def estimate_checkpoint(self):
        model = sum(value.numel() * value.element_size() for value in self.model.state_dict().values())
        adam = sum(parameter.numel() * 8 for parameter in self.model.parameters() if parameter.requires_grad)
        return model + adam + 64 * 1024**2

    def restore(self, payload):
        if canonical(payload["contract"]) != canonical(self.contract):
            raise ValueError("restore contract mismatch")
        self.run_id = payload["run_id"]
        self.model.load_state_dict(payload["model"], strict=True)
        for name in ("phase", "phase_step", "updates", "attempts", "exposure", "query_exposure", "ground_exposure", "gates", "history"):
            setattr(self, name, payload[name])
        self._optimizer()
        if payload["trainable_names"] != [name for name, value in self.model.named_parameters() if value.requires_grad]:
            raise ValueError("trainable parameter set mismatch")
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.sampler.set_state(payload["sampler_rng"])
        self.noise.set_state(payload["noise_rng"])
        torch.set_rng_state(payload["torch_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        random.setstate(payload["python_rng"])
        state = payload["numpy_rng"]
        np.random.set_state((state["name"], np.array(state["keys"], dtype=np.uint32), state["position"], state["gaussian"], state["cached"]))

    def payload(self):
        return {"schema": CheckpointStore.schema, "contract": self.contract, "run_id": self.run_id,
                "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(), "trainable_names": [name for name, value in self.model.named_parameters() if value.requires_grad],
                "phase": self.phase, "phase_step": self.phase_step, "updates": self.updates,
                "attempts": self.attempts, "exposure": self.exposure,
                "query_exposure": self.query_exposure, "ground_exposure": self.ground_exposure,
                "gates": self.gates, "history": self.history, "sampler_rng": self.sampler.get_state(),
                "noise_rng": self.noise.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate(),
                "numpy_rng": numpy_state(), "scheduler": "manual cosine by successful phase_step; optimizer stores current/base LR"}

    def save(self, tag=None):
        meta, marker = self.store.commit(self.payload(), self.estimate_checkpoint(), tag)
        atomic_json(self.out / "status.json", {"phase": self.phase, "phase_step": self.phase_step,
                    "attempts": self.attempts, "updates": self.updates, "last_marker": marker.name,
                    "checkpoint_sha256": meta["sha256"], "local_checkpoint_verified": True,
                    "backup_to_drive_verified": False, "evaluation_complete": False})
        return meta, marker

    def step(self):
        ensure_space(self.out, self.estimate_checkpoint())
        self.attempts += 1
        start = time.monotonic()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        parts, batch_indices = {}, []
        for _ in range(self.plan.accumulation):
            ids = torch.randint(len(self.train_data), (self.plan.micro_batch,), generator=self.sampler).tolist()
            batch_indices.append(ids)
            features, gold, exposure = collate([self.train_data[index] for index in ids], self.cfg, "cuda")
            self.exposure += exposure
            for index in range(4):
                if self.phase < 2:
                    self.ground_exposure[index] += int(((gold["ground"] == index) * features["fact_mask"]).sum())
                if self.phase > 0:
                    self.query_exposure[index] += int(((gold["labels"] == index) * features["query_mask"]).sum())
            with torch.autocast("cuda", dtype=torch.float16):
                if self.phase == 2:
                    with torch.no_grad():
                        encoded = self.model.encode(features)
                else:
                    encoded = self.model.encode(features)
                if self.phase == 0:
                    loss = masked_ce(encoded["ground"], gold["ground"], features["fact_mask"], self.ground_weights)
                    components = {"ground_ce": loss.detach()}
                elif self.phase == 1:
                    target = self.model.target(features, gold["labels"])
                    ce = masked_ce(self.model.readout(encoded["init"]), gold["labels"], features["query_mask"], self.weights)
                    mse = masked_mse(encoded["init"], target, features["query_mask"])
                    ground = masked_ce(encoded["ground"], gold["ground"], features["fact_mask"], self.ground_weights)
                    loss = ce + .25 * mse + .2 * ground
                    components = {"semantic_ce": ce.detach(), "semantic_mse": mse.detach(), "ground_ce": ground.detach()}
                else:
                    target = self.model.target(features, gold["labels"])
                    loss, components = diffusion_loss(self.model, encoded, target, gold["labels"], features["query_mask"], self.noise, self.weights)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite loss; previous verified checkpoint retained")
                self.scaler.scale(loss / self.plan.accumulation).backward()
            components["loss"] = loss.detach()
            for name, value in components.items():
                parts[name] = parts.get(name, 0.) + float(value) / self.plan.accumulation
        self.scaler.unscale_(self.optimizer)
        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        missing = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad and parameter.grad is None]
        frozen_grad = [name for name, parameter in self.model.named_parameters() if not parameter.requires_grad and parameter.grad is not None]
        if missing or frozen_grad:
            raise RuntimeError(f"gradient ownership failure missing={missing}, frozen={frozen_grad}")
        finite = all(bool(torch.isfinite(parameter.grad).all()) for parameter in trainable)
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1.)
        old_scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        successful = self.scaler.get_scale() >= old_scale
        if successful:
            if not finite:
                raise FloatingPointError("nonfinite gradients were not skipped")
            self.updates += 1
            self.phase_step += 1
        if not all(bool(torch.isfinite(parameter).all()) for parameter in self.model.parameters()):
            raise FloatingPointError("nonfinite model parameter")
        for group in self.optimizer.param_groups:
            progress = min(1, self.phase_step / max(1, self.plan.phase_steps[self.phase]))
            group["lr"] = group["base_lr"] * (.1 + .9 * .5 * (1 + math.cos(math.pi * progress)))
        torch.cuda.synchronize()
        entry = {"attempt": self.attempts, "updates": self.updates, "phase": PHASES[self.phase],
                 "phase_step": self.phase_step, "successful": successful, "amp_skipped": not successful,
                 "amp_scale": self.scaler.get_scale(), "finite_gradients": finite,
                 "gradient_norm_before_clip": float(norm), "gradient_tensor_count": len(trainable),
                 "missing_gradients": missing, "frozen_gradients": frozen_grad,
                 "batch_indices": batch_indices, "query_exposure": self.query_exposure.copy(),
                 "ground_exposure": self.ground_exposure.copy(), "input_utf8_bytes": self.exposure,
                 "step_seconds": time.monotonic() - start,
                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                 "peak_reserved_bytes": torch.cuda.max_memory_reserved(), **parts}
        self.history.append(entry)
        return entry

    def run(self, max_attempts, deadline):
        # Caller starts the fixed deadline before constructing this Engine.
        self.save("initial" if not self.attempts else None)
        starting = self.attempts
        result = "ATTEMPT_CAP"
        while self.phase < 3 and self.attempts - starting < max_attempts:
            if time.monotonic() >= deadline:
                result = "TIME_BUDGET"
                break
            if (self.out / "STOP").exists():
                result = "USER_STOP"
                break
            if self.phase_step >= self.plan.phase_steps[self.phase]:
                measured = evaluate(self.model, self.dev, "cuda", self.plan.micro_batch, full=False)
                score = measured["ground_accuracy"] if self.phase == 0 else measured["initializer"]["accuracy"]
                required = (self.plan.ground_gate, self.plan.semantic_gate, 0.)[self.phase]
                self.gates.append({"phase": PHASES[self.phase], "measured": score, "required": required, "passed": score >= required})
                if score < required:
                    self.save("gate_failed")
                    result = "FUNCTIONAL_GATE"
                    break
                self.phase += 1
                self.phase_step = 0
                if self.phase == 3:
                    self.save("final")  # preserve final optimizer/scaler
                    result = "TRAINING_COMPLETE_EVALUATION_PENDING"
                    break
                self._optimizer()
                self.save()
                continue
            entry = self.step()
            self.save()
            print(json.dumps(entry, ensure_ascii=False), flush=True)
        atomic_json(self.out / "exit.json", {"reason": result, "attempts": self.attempts, "updates": self.updates,
                    "evaluation_complete": False, "source": source_digest()})
        return result
