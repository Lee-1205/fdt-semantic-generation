"""Passive, bounded DDIM trajectory capture for a fixed query-count alignment case."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "research"), str(ROOT / "scripts")]
from fdt_local.data import collate
from fdt_local.model_masked import Config, FDT100
from fdt_local.runner import configure_determinism, source_digest
import alignment_transfer_querycount_v1 as data_builder
from query_bound_refiner_v1 import QueryBoundRefiner
import run_e2_grounding_20261026 as common
import learning_io_v2 as io

PARENT_SHA256 = "a877f316d0d97f2c497497fd9b941184463238055906d7109d3300db4c253b36"
SOURCE_TRANSFER_PLAN_SHA256 = "732efc57fb07b5524604762d94748601c1b259b582241f11112614179e589f63"
SOURCE_TRANSFER_RESULT_SHA256 = "4ca65f7ce7384a40304160468b041fb5ace619af310c3e65181f28134c624a4b"
SOURCE_NOISE_SHA256 = "254688fce9a8471e8e2251f26fbed5031f8b0af7310616766d1099a14f56b55c"
GRAPH_INDICES = (8, 9, 10, 11)
QUERY_COUNTS = (4, 8, 16)
SCHEDULE = (1.0, 0.75, 0.5, 0.25, 0.0)
FEATURE_KEYS = {"tokens", "roles", "fact_mask", "queries", "query_mask"}
MAX_TRACE_BYTES = 128 * 1024 * 1024


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha256(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def expected_call_counts():
    return {"actual_encode_denoise_calls": len(QUERY_COUNTS) * (2 + 4 + 4),
        "readout_head_calls_separate": len(QUERY_COUNTS) * 2,
        "max_actual_encode_denoise_calls": 40,
        "graph_count": 4, "query_views": list(QUERY_COUNTS),
        "actual_batched_model_calls": len(QUERY_COUNTS) * (2 + 4 + 4)}


def trace_step_values(z, velocity, t, t_next):
    """Use the frozen rollout's exact arithmetic order and constants."""
    a, s = math.cos(math.pi * t / 2), math.sin(math.pi * t / 2)
    if t == 1.0:
        a, s = 0.0, 1.0
    clean = a * z - s * velocity
    eps = s * z + a * velocity
    next_z = math.cos(math.pi * t_next / 2) * clean + math.sin(math.pi * t_next / 2) * eps
    return clean, eps, next_z


def clone_float_cpu(tensor):
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def require_finite(tensor, label):
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
        raise FloatingPointError(f"nonfinite trajectory value: {label}")


def validate_fixed_case(rows_by_q, noise, plan):
    if tuple(plan.get("graph_indices", ())) != GRAPH_INDICES or tuple(plan.get("query_counts", ())) != QUERY_COUNTS:
        raise ValueError("fixed trajectory slice changed")
    if not isinstance(noise, torch.Tensor) or tuple(noise.shape) != (16, 16, 512) or noise.dtype != torch.float32:
        raise ValueError("saved seed-17 base noise must have exact float32 [16,16,512] shape")
    by_q = {}
    keys = None
    for q in QUERY_COUNTS:
        rows = rows_by_q[q]
        if len(rows) != 16:
            raise ValueError(f"q{q} source file must retain 16 fixed ordered graphs")
        chosen = rows[8:12]
        current = [row["canonical_graph_key"] for row in chosen]
        if keys is None:
            keys = current
        elif current != keys:
            raise ValueError("fixed graph indices do not refer to same registered graphs across views")
        for row in chosen:
            if row["query_count"] != q or len(row["queries"]) != q:
                raise ValueError("query-count source view row mismatch")
            if data_builder.oracle(8, row["facts"], row["queries"]) != (row["labels"], row["hops"]):
                raise ValueError("frozen oracle mismatch in selected trajectory rows")
            if data_builder.v2.independent_oracle(8, row["facts"], row["queries"]) != (row["labels"], row["hops"]):
                raise ValueError("independent BFS mismatch in selected trajectory rows")
        by_q[q] = chosen
    for graph_slot in range(4):
        full = by_q[16][graph_slot]
        for q in (4, 8):
            row = by_q[q][graph_slot]
            if row["facts"] != full["facts"] or row["queries"] != full["queries"][:q]:
                raise ValueError("fixed selected views are not same-facts ordered query prefixes")
            if row["labels"] != full["labels"][:q] or row["hops"] != full["hops"][:q]:
                raise ValueError("selected view targets are not original full-view prefixes")
    target = by_q[4][0]
    if target["queries"][3] != [4, 6] or int(target["labels"][3]) != 1:
        raise ValueError("the preregistered graph-index-8 slot-3 GREATER case changed")
    if by_q[8][0]["queries"][3] != target["queries"][3] or by_q[16][0]["queries"][3] != target["queries"][3]:
        raise ValueError("the fixed target query must persist as the same query prefix")
    if not bool(torch.isfinite(noise).all()):
        raise ValueError("saved paired noise contains nonfinite values")
    return {"graph_indices": list(GRAPH_INDICES), "query_counts": list(QUERY_COUNTS),
        "graphs_per_view": 4, "target_case": {"graph_index": 8, "slot": 3,
            "query": [4, 6], "gold_class": 1, "gold_name": "GREATER"},
        "noise_sha256": SOURCE_NOISE_SHA256, "noise_shape": list(noise.shape),
        "oracle_and_bfs_verified": True, "same_fact_query_prefix_verified": True}


def output_name_contract(names, *, execution=False):
    allowed = {"plan.json", "readiness.json", "seal.json", "code_snapshot", "supervisor.stdout.log",
        "supervisor.stderr.log", "supervisor", "worker.stdout.log", "worker.stderr.log", "control"}
    if execution:
        allowed |= {"worker_progress.json", "trajectory_trace.pt", "diagnostic_result.json", "incident.json"}
    forbidden = {"model", "checkpoint.pt", "optimizer.pt", "final_status.json", "clock.json"}
    if set(names) - allowed or set(names) & forbidden:
        raise FileExistsError(f"trajectory output entries conflict: unexpected={sorted(set(names)-allowed)} forbidden={sorted(set(names)&forbidden)}")


def preflight(plan_path: Path, output: Path, worker_path: Path, *, strict_load=True):
    plan = io.load_json(plan_path)
    if not isinstance(plan, dict) or plan.get("schema") != "fdt_alignment_trajectory_readonly_v1":
        raise ValueError("trajectory plan schema mismatch")
    if plan.get("authorized") not in (False, True) or Path(plan["output_dir"]).resolve() != output.resolve():
        raise ValueError("trajectory plan authorization/output mismatch")
    if (plan.get("wall_seconds"), plan.get("shutdown_reserve_seconds"), plan.get("max_actual_encode_denoise_calls"),
            plan.get("trace_cap_bytes"), plan.get("graph_indices"), plan.get("query_counts"), plan.get("seed")) != (
            90, 20, 40, MAX_TRACE_BYTES, [8, 9, 10, 11], [4, 8, 16], 17):
        raise ValueError("fixed bounded trajectory contract mismatch")
    if plan.get("optimizer_allowed") is not False or plan.get("checkpoint_write_allowed") is not False or plan.get("e2_final_files_accessed") is not False:
        raise ValueError("trajectory diagnostic must be readonly and avoid E2 final data")
    if plan.get("worker_sha256") != sha256(worker_path):
        raise ValueError("trajectory worker SHA mismatch")
    root = ROOT
    dataset = Path(plan["dataset_dir"]).resolve()
    deps = {"worker": worker_path, "supervisor": root / "scripts/supervise_alignment_trajectory_20261027.py",
        "config": Path(plan["config_path"]), "io": root / "scripts/learning_io_v2.py",
        "common_worker": root / "scripts/run_e2_grounding_20261026.py",
        "builder": root / "research/alignment_transfer_querycount_v1.py",
        "query_bound_helper": root / "research/query_bound_refiner_v1.py",
        "model_config": root / "src/fdt_local/model_masked.py", "data_source": root / "src/fdt_local/data.py",
        "parent_marker": Path(plan["parent_marker"]), "dataset_manifest": dataset / "manifest.json",
        "selection_registry": dataset / "selection_registry.json",
        "source_transfer_plan": Path(plan["source_transfer_plan_path"]),
        "source_transfer_result": Path(plan["source_transfer_result_path"]),
        "source_noise": Path(plan["source_noise_path"])}
    for key, path in deps.items():
        if plan.get("dependency_sha256", {}).get(key) != sha256(path):
            raise ValueError(f"pinned source/data/parent SHA mismatch: {key}")
    if source_digest() != plan["runtime_source_sha256"] or source_digest() != plan["core_sha256"]:
        raise ValueError("frozen FDT core SHA mismatch")
    if plan.get("parent_payload_sha256") != PARENT_SHA256 or plan.get("source_transfer_plan_sha256") != SOURCE_TRANSFER_PLAN_SHA256:
        raise ValueError("frozen parent/transfer plan identity mismatch")
    if plan.get("source_transfer_result_sha256") != SOURCE_TRANSFER_RESULT_SHA256 or plan.get("source_noise_sha256") != SOURCE_NOISE_SHA256:
        raise ValueError("frozen transfer result or saved noise identity mismatch")
    manifest = io.load_json(dataset / "manifest.json")
    payload = dict(manifest)
    manifest_payload_hash = payload.pop("manifest_payload_sha256_without_self", None)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest_payload_hash:
        raise ValueError("query-count dataset manifest payload hash mismatch")
    if plan.get("dataset_manifest_sha256") != sha256(dataset / "manifest.json"):
        raise ValueError("query-count dataset manifest file hash mismatch")
    rows_by_q = {}
    for q in QUERY_COUNTS:
        path = dataset / f"q{q:02d}.json"
        if plan.get("data_sha256", {}).get(path.name) != sha256(path):
            raise ValueError(f"query-count data SHA mismatch: {path.name}")
        rows_by_q[q] = io.load_json(path)
    if plan.get("selection_registry_sha256") != sha256(dataset / "selection_registry.json"):
        raise ValueError("selection registry SHA mismatch")
    noise_path = Path(plan["source_noise_path"])
    if sha256(noise_path) != SOURCE_NOISE_SHA256:
        raise ValueError("saved matched-noise artifact SHA mismatch")
    noise = torch.load(noise_path, map_location="cpu", weights_only=True)
    selected = validate_fixed_case(rows_by_q, noise, plan)
    marker, parent = common.ResilientCheckpointStore(Path(plan["parent_marker"]).parent).verify(Path(plan["parent_marker"]))
    if marker["sha256"] != PARENT_SHA256 or (parent.get("updates"), parent.get("attempts")) != (256, 256):
        raise ValueError("parent checkpoint final status/payload hash mismatch")
    if parent.get("contract", {}).get("source") != source_digest() or parent.get("contract", {}).get("variant") != "masked":
        raise ValueError("parent model/core contract mismatch")
    cfg = Config(**parent["contract"]["config"])
    configure_determinism()
    if strict_load:
        original, bound = FDT100(cfg), QueryBoundRefiner(cfg)
        if original.counts()["parameters"] != 102237197 or bound.counts()["parameters"] != 102237197:
            raise ValueError("model parameter count mismatch")
        if set(original.state_dict()) != set(bound.state_dict()):
            raise ValueError("original and query-bound model state keys differ")
        original.load_state_dict(parent["model"], strict=True)
        bound.load_state_dict(parent["model"], strict=True)
        expected_state_hash = plan["parent_model_state_sha256"]
        if state_sha256(original.state_dict()) != expected_state_hash or state_sha256(bound.state_dict()) != expected_state_hash:
            raise ValueError("strict model load does not match parent state digest")
        del original, bound
    del parent, noise
    if output.exists():
        output_name_contract({p.name for p in output.iterdir()}, execution=True)
    return {"plan": plan, "selected": selected, "status": "CPU_PREFLIGHT_PASS"}


class CallCounter:
    def __init__(self, cap, deadline, stop_path):
        self.cap, self.deadline, self.stop_path = cap, deadline, stop_path
        self.calls = 0
        self.by_kind = {"encode": 0, "denoise": 0}

    def charge(self, kind):
        if time.time() >= self.deadline:
            raise TimeoutError("trajectory worker deadline reached")
        if self.stop_path.exists():
            raise InterruptedError("owned supervisor requested cooperative stop")
        if self.calls >= self.cap:
            raise RuntimeError("trajectory encode/denoise call cap exceeded")
        self.calls += 1
        self.by_kind[kind] += 1


def rollout_trace(model, encoded, noise, counter, *, autocast_cuda=True):
    mask = encoded["query_mask"][..., None]
    z = noise.float() * mask
    trace = {"init": clone_float_cpu(encoded["init"]), "active_mask": encoded["query_mask"].detach().cpu().bool().clone(), "steps": []}
    autocast = torch.autocast("cuda", dtype=torch.float16) if autocast_cuda else nullcontext()
    with torch.no_grad(), autocast:
        for t, t_next in zip(SCHEDULE, SCHEDULE[1:]):
            x_t = z
            times = z.new_full((len(z),), t)
            counter.charge("denoise")
            velocity = model.denoise(x_t, times, encoded).float()
            require_finite(velocity, "velocity")
            clean, eps, next_z = trace_step_values(x_t, velocity, t, t_next)
            for label, value in (("x_t", x_t), ("predicted_clean", clean), ("predicted_eps", eps), ("x_next", next_z)):
                require_finite(value, label)
            trace["steps"].append({"t": t, "t_next": t_next,
                "x_t": clone_float_cpu(x_t), "velocity": clone_float_cpu(velocity),
                "predicted_clean": clone_float_cpu(clean), "predicted_eps": clone_float_cpu(eps),
                "x_next": clone_float_cpu(next_z)})
            z = next_z
    z = z * mask
    trace["final"] = clone_float_cpu(z)
    return z, trace


def logits_and_margins(logits):
    sorted_logits, order = logits.sort(dim=-1, descending=True)
    return order[..., 0], sorted_logits[..., 0] - sorted_logits[..., 1]


def run_diagnostic(plan_path: Path, output: Path, worker_path: Path):
    checked = preflight(plan_path, output, worker_path, strict_load=False)
    plan = checked["plan"]
    if plan.get("authorized") is not True:
        raise ValueError("draft is not authorized for GPU inference")
    state = io.load_json(output / "supervisor/state.json")
    if not isinstance(state, dict):
        raise ValueError("owned supervisor launch state is missing")
    launch_unix = float(state["invocation_started_unix"])
    deadline = float(state["worker_deadline_unix"])
    counter = CallCounter(plan["max_actual_encode_denoise_calls"], deadline, Path(plan["control_dir"]) / "STOP")
    progress_path = output / "worker_progress.json"
    def progress(stage, anomaly=None):
        io.atomic_json(progress_path, {"stage": stage, "observed_unix": time.time(), "worker_pid": os.getpid(),
            "actual_encode_denoise_calls": counter.calls, "calls_by_kind": counter.by_kind,
            "call_cap": counter.cap, "updates": 0, "optimizer_used": False, "anomaly": anomaly})
    for name in ("trajectory_trace.pt", "diagnostic_result.json", "worker_progress.json", "incident.json"):
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite readonly evidence: {name}")
    marker, parent = common.ResilientCheckpointStore(Path(plan["parent_marker"]).parent).verify(Path(plan["parent_marker"]))
    configure_determinism()
    cfg = Config(**parent["contract"]["config"])
    cpu_rng_before = torch.random.get_rng_state().clone()
    cuda_device = torch.cuda.current_device()
    cuda_rng_before = torch.cuda.get_rng_state(cuda_device).clone()
    with torch.random.fork_rng(devices=[cuda_device]):
        original = FDT100(cfg).to("cuda")
        bound = QueryBoundRefiner(cfg).to("cuda")
    if not torch.equal(torch.random.get_rng_state(), cpu_rng_before) or not torch.equal(torch.cuda.get_rng_state(cuda_device), cuda_rng_before):
        raise ValueError("model construction changed global RNG state")
    original.load_state_dict(parent["model"], strict=True)
    bound.load_state_dict(parent["model"], strict=True)
    parent_state_sha = plan["parent_model_state_sha256"]
    if state_sha256(original.state_dict()) != parent_state_sha or state_sha256(bound.state_dict()) != parent_state_sha:
        raise ValueError("runtime model state does not match parent including buffers")
    del parent
    for model in (original, bound):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    source_noise = torch.load(plan["source_noise_path"], map_location="cpu", weights_only=True)
    rows_by_q = {q: io.load_json(Path(plan["dataset_dir"]) / f"q{q:02d}.json") for q in QUERY_COUNTS}
    trace_payload = {"schema": "fdt_alignment_trajectory_trace_tensors_v1",
        "source_noise_sha256": SOURCE_NOISE_SHA256,
        "source_noise_graph_indices": list(GRAPH_INDICES),
        "noise_fp32": source_noise[list(GRAPH_INDICES)].contiguous().clone(),
        "views": {}}
    scoring = {"query_counts": {}, "raw_active_query_rows": []}
    readout_calls = 0
    initializer_max_delta = 0.0
    progress("trace_start")
    try:
        for q in QUERY_COUNTS:
            rows = [rows_by_q[q][idx] for idx in GRAPH_INDICES]
            features, _, _ = collate(rows, cfg, "cuda")
            if set(features) != FEATURE_KEYS or features["query_mask"].sum(-1).tolist() != [q] * 4:
                raise ValueError("selected active mask or five-feature input contract mismatch")
            noise = source_noise[list(GRAPH_INDICES)].to("cuda")
            traces, logits_by_model = {}, {}
            for variant, model in (("original", original), ("query_bound", bound)):
                counter.charge("encode")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    encoded = model.encode(features)
                for name, tensor in encoded.items():
                    if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
                        raise FloatingPointError(f"nonfinite encoder output {variant}.{name}")
                final, trace = rollout_trace(model, encoded, noise, counter)
                # One batched projection call per model/view: initializer and each clean estimate.
                readout_states = torch.stack([encoded["init"]] + [step["predicted_clean"].to("cuda")
                    for step in trace["steps"]], dim=1)
                with torch.no_grad():
                    readouts = model.readout(readout_states).float()
                readout_calls += 1
                if not bool(torch.isfinite(readouts).all()) or not bool(torch.isfinite(final).all()):
                    raise FloatingPointError(f"nonfinite trajectory projection: {variant} q={q}")
                predicted, margins = logits_and_margins(readouts)
                trace["readout_logits_init_then_clean_steps"] = clone_float_cpu(readouts)
                trace["predicted_class_init_then_clean_steps"] = predicted.detach().cpu().long().clone()
                trace["top1_minus_top2_margin_init_then_clean_steps"] = clone_float_cpu(margins)
                trace["query_count"] = q
                traces[variant] = trace
                logits_by_model[variant] = {"logits": readouts.detach().cpu(),
                    "prediction": predicted.detach().cpu(), "margin": margins.detach().cpu()}
            if not torch.equal(logits_by_model["original"]["logits"][:, 0], logits_by_model["query_bound"]["logits"][:, 0]):
                raise ValueError("original and bound initializer projection must be identical")
            initializer_max_delta = max(initializer_max_delta, float((logits_by_model["original"]["logits"][:,0]
                - logits_by_model["query_bound"]["logits"][:,0]).abs().max()))
            trace_payload["views"][str(q)] = {"query_mask": features["query_mask"].detach().cpu().bool().clone(),
                "original": traces["original"], "query_bound": traces["query_bound"]}
            for bi, row in enumerate(rows):
                for slot in range(q):
                    scoring["raw_active_query_rows"].append({"graph_index": GRAPH_INDICES[bi],
                        "canonical_graph_key": row["canonical_graph_key"], "query_count": q, "slot": slot,
                        "query": row["queries"][slot], "gold_class": int(row["labels"][slot]),
                        "gold_hops": int(row["hops"][slot]),
                        "original_logits_init_then_clean_steps": logits_by_model["original"]["logits"][bi,:,slot].tolist(),
                        "original_pred_init_then_clean_steps": logits_by_model["original"]["prediction"][bi,:,slot].tolist(),
                        "original_margin_init_then_clean_steps": logits_by_model["original"]["margin"][bi,:,slot].tolist(),
                        "query_bound_logits_init_then_clean_steps": logits_by_model["query_bound"]["logits"][bi,:,slot].tolist(),
                        "query_bound_pred_init_then_clean_steps": logits_by_model["query_bound"]["prediction"][bi,:,slot].tolist(),
                        "query_bound_margin_init_then_clean_steps": logits_by_model["query_bound"]["margin"][bi,:,slot].tolist()})
            progress(f"q{q}_complete")
        if counter.calls != 30 or counter.by_kind != {"encode": 6, "denoise": 24} or readout_calls != 6:
            raise RuntimeError(f"trajectory model-call counts mismatch calls={counter.calls} by_kind={counter.by_kind} readout={readout_calls}")
        original_final = state_sha256(original.state_dict())
        bound_final = state_sha256(bound.state_dict())
        if original_final != parent_state_sha or bound_final != parent_state_sha:
            raise ValueError("trajectory forward altered model weights or buffers")
        rng_unchanged = torch.equal(torch.random.get_rng_state(), cpu_rng_before) and torch.equal(
            torch.cuda.get_rng_state(cuda_device), cuda_rng_before)
        if not rng_unchanged:
            raise ValueError("passive trajectory capture changed global torch RNG state")
        trace_path = output / "trajectory_trace.pt"
        temp_trace = trace_path.with_suffix(".pt.tmp")
        torch.save(trace_payload, temp_trace)
        io.replace_with_retry(temp_trace, trace_path)
        total_evidence = trace_path.stat().st_size
        if total_evidence >= MAX_TRACE_BYTES:
            raise RuntimeError(f"trajectory artifact exceeds 128 MiB cap: {total_evidence}")
        result = {"schema": "fdt_alignment_trajectory_readonly_result_v1", "status": "PASSIVE_TRACE_COMPLETE",
            "source_transfer_plan_sha256": SOURCE_TRANSFER_PLAN_SHA256,
            "source_transfer_result_sha256": SOURCE_TRANSFER_RESULT_SHA256,
            "source_noise_sha256": SOURCE_NOISE_SHA256, "source_noise_rows": list(GRAPH_INDICES),
            "fixed_graph_indices": list(GRAPH_INDICES), "query_counts": list(QUERY_COUNTS),
            "trace_file": str(trace_path), "trace_sha256": sha256(trace_path), "trace_bytes": total_evidence,
            "trace_cap_bytes": MAX_TRACE_BYTES, "readout_head_calls_separate": readout_calls,
            "actual_encode_denoise_calls": counter.calls, "calls_by_kind": counter.by_kind,
            "initializer_max_abs_logit_delta": initializer_max_delta,
            "parent_model_state_sha256": parent_state_sha,
            "original_final_state_sha256": original_final, "query_bound_final_state_sha256": bound_final,
            "both_model_and_buffer_digests_unchanged": True,
            "global_torch_rng_unchanged": rng_unchanged,
            "optimizer_used": False, "checkpoint_written": False, "e2_final_files_accessed": False,
            "active_masks_saved": True, "gold_used_only_after_model_calls_for_scoring": True,
            "latent_projection_caveat": "Readout of predicted-clean estimates at positive noise is a diagnostic projection, not the model's normal answer at that diffusion time.",
            "purpose_limit": "Post-observation trace of one fixed miss across Q=4/8/16; descriptive timing/noise/latent evidence, not causal proof or a fresh benchmark.",
            "raw_active_query_rows": scoring["raw_active_query_rows"],
            "elapsed_seconds": time.time() - launch_unix, "wall_seconds_cap": 90}
        io.atomic_json(output / "diagnostic_result.json", result)
        progress("terminal")
        return result
    except BaseException as exc:
        progress("terminal", f"{type(exc).__name__}: {exc}")
        io.atomic_json(output / "incident.json", {"error": f"{type(exc).__name__}: {exc}",
            "optimizer_used": False, "checkpoint_written": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight(args.plan.resolve(), args.output.resolve(), Path(__file__).resolve(), strict_load=True)
        print("CPU_PREFLIGHT_PASS")
        return 0
    run_diagnostic(args.plan.resolve(), args.output.resolve(), Path(__file__).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
