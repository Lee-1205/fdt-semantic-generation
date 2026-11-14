"""Five attempts maximum, one 600-second shared suite, never auto-retry."""
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from .budget import SmokeBudget
from .runner import Engine
from .storage import CheckpointStore, atomic_json, canonical, ensure_space


def tensor_tree_digest(value):
    digest = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            cpu = item.detach().cpu().contiguous()
            digest.update(str((str(cpu.dtype), tuple(cpu.shape))).encode())
            digest.update(cpu.numpy().tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                digest.update(str(key).encode())
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            for element in item:
                visit(element)
        else:
            digest.update(canonical(item).encode())
    visit(value)
    return digest.hexdigest()


def trajectory(engine):
    payload = engine.payload()
    return tensor_tree_digest({key: payload[key] for key in (
        "model", "optimizer", "scaler", "phase", "phase_step", "updates", "attempts",
        "sampler_rng", "noise_rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng",
        "exposure", "query_exposure", "ground_exposure")})


def resume_probe(directory, marker, budget_path, expected_path):
    out = Path(directory)
    spec = json.loads((out / "resolved_config.json").read_text(encoding="utf-8"))
    expected = json.loads(Path(expected_path).read_text(encoding="utf-8"))
    with CheckpointStore(out).lock():
        engine = Engine(spec, out, resume=True, marker=marker)
        SmokeBudget(budget_path).reserve_attempt("102M fresh-process repeated next update")
        actual = engine.step()
        digest = trajectory(engine)
        # A verification fork has an explicit separate storage directory; it never
        # publishes over the uninterrupted branch's verified generations.
        engine.out = out / "verification_fork"
        engine.store = CheckpointStore(engine.out)
        atomic_json(engine.out / "resolved_config.json", spec)
        meta, saved_marker = engine.save("resume_proof")
        result = {"trajectory_equal": digest == expected["trajectory_sha256"],
                  "loss_equal": actual["loss"] == expected["entry"]["loss"],
                  "input_indices_equal": actual["batch_indices"] == expected["entry"]["batch_indices"],
                  "actual_sha256": digest, "expected_sha256": expected["trajectory_sha256"],
                  "entry": actual, "checkpoint": meta, "marker": str(saved_marker),
                  "fresh_process": True}
        atomic_json(out / "resume_proof.json", result)
        if not all(result[key] for key in ("trajectory_equal", "loss_equal", "input_indices_equal")):
            raise RuntimeError("fresh-process next-update mismatch")


def run_suite(spec, project, max_attempts, wall_seconds):
    if max_attempts > 5 or wall_seconds > 600 or max_attempts < 5 or wall_seconds < 60:
        raise ValueError("this sealed proof suite uses exactly five attempt slots and 60..600 wall seconds")
    project = Path(project)
    budget = SmokeBudget(project / "runs" / "smoke_budget.json")
    suite = project / "runs" / "smoke_102m"
    if suite.exists():
        raise FileExistsError("smoke run already exists; no automatic restart")
    ensure_space(project, 6 * 2**30)
    budget.begin()
    start = time.monotonic()
    success = False
    report = {"optimizer_attempt_limit": 5, "gpu_wall_limit_seconds": wall_seconds,
              "past_2240_updates_recovered": False, "profile": "actual_102237197_plus_tiny_phase_checks"}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; no CPU fallback")
        a = torch.tensor([2., 3.], device="cuda", requires_grad=True)
        (a.square().sum()).backward()
        torch.cuda.synchronize()
        report["cuda_tensor_check"] = bool(torch.equal(a.grad, torch.tensor([4., 6.], device="cuda")))
        del a
        torch.cuda.reset_peak_memory_stats()
        with CheckpointStore(suite).lock():
            engine = Engine(spec, suite)
            if engine.model.counts()["parameters"] != 102237197:
                raise ValueError("actual 102M parameter count mismatch")
            engine.save("initial")
            budget.reserve_attempt("102M grounding first update")
            first = engine.step()
            first_meta, first_marker = engine.save("resume_parent")
            budget.reserve_attempt("102M uninterrupted next update")
            second = engine.step()
            expected = {"entry": second, "trajectory_sha256": trajectory(engine)}
            expected_path = suite / "expected_next.json"
            atomic_json(expected_path, expected)
            engine.save("uninterrupted_reference")
            report["actual_102m"] = {"parameters": engine.model.counts()["parameters"],
                                    "first": first, "second": second, "parent_marker": str(first_marker)}
            del engine
            gc.collect()
            torch.cuda.empty_cache()
        remaining = wall_seconds - (time.monotonic() - start)
        if remaining < 20:
            raise TimeoutError("insufficient budget for independent reload")
        command = [sys.executable, "-B", "-X", "utf8", "-m", "fdt_local", "_resume-probe", "--run-dir", str(suite),
                   "--marker", str(first_marker), "--budget", str(budget.path), "--expected", str(expected_path)]
        child = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=remaining)
        (suite / "reload.stdout.txt").write_text(child.stdout, encoding="utf-8")
        (suite / "reload.stderr.txt").write_text(child.stderr, encoding="utf-8")
        if child.returncode:
            raise RuntimeError(f"fresh reload failed: {child.stderr[-1200:]}")
        report["resume_proof"] = json.loads((suite / "resume_proof.json").read_text(encoding="utf-8"))
        for phase in (1, 2):
            if time.monotonic() - start >= wall_seconds:
                raise TimeoutError("suite wall cap reached")
            tiny_spec = {**spec, "profile": "tiny", "start_phase": phase}
            directory = project / "runs" / f"smoke_tiny_phase{phase}"
            with CheckpointStore(directory).lock():
                engine = Engine(tiny_spec, directory)
                engine.save("initial")
                budget.reserve_attempt(f"tiny phase {phase} gradient check")
                entry = engine.step()
                engine.save("phase_check")
                report[f"tiny_phase{phase}"] = {"parameters": engine.model.counts()["parameters"], "entry": entry}
                del engine
                gc.collect()
                torch.cuda.empty_cache()
        success = True
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.monotonic() - start
        budget.finish(elapsed, success)
        report.update(completed=success, total_gpu_window_seconds=elapsed, budget=budget.read())
        atomic_json(project / "runs" / "smoke_report.json", report)
    return report
