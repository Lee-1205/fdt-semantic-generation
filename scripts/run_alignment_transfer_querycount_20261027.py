"""Bounded read-only query-count transfer diagnostic for original/bound refiners."""
from __future__ import annotations
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
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

FEATURE_KEYS = {"tokens", "roles", "fact_mask", "queries", "query_mask"}
QUERY_COUNTS = (1, 4, 8, 16)
PARENT_SHA256 = "a877f316d0d97f2c497497fd9b941184463238055906d7109d3300db4c253b36"
MAX_ACTUAL_CALLS = 200
WALL_SECONDS = 120
SEED = 17
ROLLOUT_SCHEDULE = (1.0, 0.75, 0.5, 0.25, 0.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_finite_tensors(values, context: str) -> None:
    items = values.items() if isinstance(values, dict) else values
    for name, tensor in items:
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"nonfinite tensor in {context}: {name}")


def expected_call_counts() -> dict:
    batches_per_view = 4
    views = len(QUERY_COUNTS)
    return {"actual_batched_model_calls": views * batches_per_view * (2 + 4 + 4),
        "max_actual_batched_model_calls": MAX_ACTUAL_CALLS,
        "views": views, "graphs": 16, "graphs_per_view": 16,
        "active_query_positions_per_variant": 16 * sum(QUERY_COUNTS),
        "distinct_base_queries": 16 * 16,
        "actual_encoder_graph_exposures_per_variant": views * 16,
        "actual_encoder_graph_exposures_both_variants": views * 16 * 2,
        "actual_premise_fact_instances_encoder_exposures_both_variants": 145 * views * 2,
        "noise_batched_calls": views * batches_per_view,
        "readout_calls_not_in_approved_encode_denoise_call_counter": views * batches_per_view * 2 * 2}


def validate_rows_by_view(rows_by_q: dict[int, list[dict]], selected_registry: dict) -> dict:
    expected_keys = [r["canonical_graph_key"] for r in selected_registry["graphs"]]
    if len(expected_keys) != 16 or len(set(expected_keys)) != 16:
        raise ValueError("registry must contain 16 distinct canonical graphs")
    all_features = set(FEATURE_KEYS)
    active_positions = 0
    for q in QUERY_COUNTS:
        rows = rows_by_q[q]
        if len(rows) != 16:
            raise ValueError(f"q{q:02d} must contain exactly 16 graphs")
        if [r["canonical_graph_key"] for r in rows] != expected_keys:
            raise ValueError("query-count rows do not follow registered graph order")
        for index, row in enumerate(rows):
            if row["query_count"] != q or len(row["queries"]) != q or len(set(map(tuple, row["queries"]))) != q:
                raise ValueError("query count or unique query pair contract mismatch")
            if row["canonical_graph_key"] != expected_keys[index]:
                raise ValueError("view graph registration order changed")
            original = data_builder.oracle(8, row["facts"], row["queries"])
            bfs = data_builder.v2.independent_oracle(8, row["facts"], row["queries"])
            if original != bfs or original != (row["labels"], row["hops"]):
                raise ValueError("oracle/BFS labels or shortest hops mismatch")
            features, targets, _ = collate([row], Config(), "cpu")
            if set(features) != all_features or int(features["query_mask"].sum()) != q:
                raise ValueError("feature whitelist or active query mask mismatch")
            if targets["labels"].shape != (1, 16) or targets["hops"].shape != (1, 16):
                raise ValueError("target capacity contract mismatch")
            active_positions += int(features["query_mask"].sum())
    q16 = {r["canonical_graph_key"]: r for r in rows_by_q[16]}
    for q in (1, 4, 8):
        for row in rows_by_q[q]:
            full = q16[row["canonical_graph_key"]]
            if row["facts"] != full["facts"] or row["queries"] != full["queries"][:q]:
                raise ValueError("views must share facts and ordered query prefixes")
            if row["labels"] != full["labels"][:q] or row["hops"] != full["hops"][:q]:
                raise ValueError("oracle labels/hops do not match full-view prefixes")
    expected = 16 * sum(QUERY_COUNTS)
    if active_positions != expected:
        raise ValueError("active query-position total mismatch")
    return {"graphs": 16, "query_count_views": 64,
        "active_query_positions_per_variant": active_positions,
        "distinct_base_queries": 256, "independence_claimed": False,
        "all_original_oracle_and_bfs_verified": True,
        "all_facts_and_query_prefixes_verified": True,
        "five_feature_input_contract_verified": True}


def output_name_contract(names, *, allow_execution=False):
    allowed = {"plan.json", "readiness.json", "seal.json", "code_snapshot", "supervisor.stdout.log",
        "supervisor.stderr.log", "supervisor", "worker.stdout.log", "worker.stderr.log", "control"}
    if allow_execution:
        allowed |= {"worker_progress.json", "diagnostic_result.json", "paired_seed17_noise.pt", "incident.json",
            "supervisor"}
    forbidden = {"model", "checkpoint.pt", "optimizer.pt", "clock.json", "final_status.json"}
    unexpected = set(names) - allowed
    if unexpected or (set(names) & forbidden):
        raise FileExistsError(f"readonly output entries invalid: unexpected={sorted(unexpected)} forbidden={sorted(set(names)&forbidden)}")


def preflight(plan_path: Path, output: Path, worker_path: Path, *, verify_strict_load=True) -> dict:
    plan = io.load_json(plan_path)
    if not isinstance(plan, dict) or plan.get("schema") != "fdt_alignment_transfer_querycount_readonly_v1":
        raise ValueError("readonly transfer diagnostic schema mismatch")
    if plan.get("authorized") not in (False, True):
        raise ValueError("authorized marker must be boolean")
    if Path(plan["output_dir"]).resolve() != output.resolve():
        raise ValueError("output directory path mismatch")
    if plan.get("worker_sha256") != sha256(worker_path):
        raise ValueError("worker digest mismatch")
    if (plan.get("wall_seconds"), plan.get("wall_shutdown_reserve_seconds"), plan.get("max_actual_batched_model_calls"),
            plan.get("seed"), plan.get("query_counts"), plan.get("graph_count"), plan.get("batch_size")) != (120, 30, 200, 17, [1, 4, 8, 16], 16, 4):
        raise ValueError("fixed bounded diagnostic contract mismatch")
    if plan.get("optimizer_allowed") is not False or plan.get("checkpoint_write_allowed") is not False:
        raise ValueError("optimizer and model checkpoint writes must be prohibited")
    if plan.get("e2_final_files_accessed") is not False:
        raise ValueError("E2 final holdout is prohibited")
    if plan.get("expected_actual_batched_model_calls") != expected_call_counts()["actual_batched_model_calls"]:
        raise ValueError("actual batched model call contract mismatch")
    if Path(plan["parent_marker"]).resolve().name != "complete_000262.json" or plan.get("parent_payload_sha256") != PARENT_SHA256:
        raise ValueError("wrong frozen QueryMix weights-only parent")
    artifact = Path(plan["dataset_dir"]).resolve()
    if artifact != (ROOT / "research/artifacts/alignment_transfer_querycount_20261027_v1").resolve():
        raise ValueError("unexpected query-count artifact directory")
    expected_files = {"builder_py": ROOT / "research/alignment_transfer_querycount_v1.py",
        "query_bound_helper": ROOT / "research/query_bound_refiner_v1.py",
        "v2_builder": ROOT / "research/role8_active_balanced_v2.py",
        "frozen_data": ROOT / "src/fdt_local/data.py", "model_config": ROOT / "src/fdt_local/model_masked.py",
        "io": ROOT / "scripts/learning_io_v2.py", "common_worker": ROOT / "scripts/run_e2_grounding_20261026.py",
        "parent_marker": Path(plan["parent_marker"]), "manifest": artifact / "manifest.json",
        "selection_registry": artifact / "selection_registry.json",
        "worker": worker_path, "supervisor": ROOT / "scripts/supervise_alignment_transfer_querycount_20261027.py",
        "config": Path(plan["config_path"])}
    for key, path in expected_files.items():
        if plan.get("dependency_sha256", {}).get(key) != sha256(path):
            raise ValueError(f"pinned dependency digest mismatch: {key}")
    if source_digest() != plan["runtime_source_sha256"] or source_digest() != plan["core_sha256"]:
        raise ValueError("frozen FDT core source digest mismatch")
    config_spec = io.load_json(plan["config_path"])
    if (config_spec.get("schema"), config_spec.get("models"), config_spec.get("query_counts"),
            config_spec.get("paired_noise_seed"), config_spec.get("batch_size"),
            config_spec.get("rollout_steps"), config_spec.get("optimizer_allowed"),
            config_spec.get("checkpoint_write_allowed")) != (
            "fdt_alignment_transfer_querycount_readonly_v1", ["FDT100", "QueryBoundRefiner"],
            [1, 4, 8, 16], 17, 4, 4, False, False):
        raise ValueError("fixed readonly diagnostic config mismatch")
    if plan.get("bound_helper_sha256") != sha256(ROOT / "research/query_bound_refiner_v1.py"):
        raise ValueError("QueryBoundRefiner helper digest mismatch")
    if plan.get("dataset_manifest_sha256") != sha256(artifact / "manifest.json"):
        raise ValueError("query-count manifest file digest mismatch")
    manifest = io.load_json(artifact / "manifest.json")
    if manifest.get("status") != "CPU_ALIGNMENT_TRANSFER_QUERYCOUNT_V1_NOT_TRAINED" or manifest.get("optimizer_updates") != 0:
        raise ValueError("unexpected query-count artifact state")
    rows_by_q = {}
    for q in QUERY_COUNTS:
        path = artifact / f"q{q:02d}.json"
        if plan["data_sha256"].get(path.name) != sha256(path):
            raise ValueError(f"query-count data file hash mismatch: {path.name}")
        rows_by_q[q] = io.load_json(path)
    registry_path = artifact / "selection_registry.json"
    if plan.get("selection_registry_sha256") != sha256(registry_path):
        raise ValueError("selection registry digest mismatch")
    registry = io.load_json(registry_path)
    checked_data = validate_rows_by_view(rows_by_q, registry)
    marker_meta, payload = common.ResilientCheckpointStore(Path(plan["parent_marker"]).parent).verify(Path(plan["parent_marker"]))
    if marker_meta["sha256"] != PARENT_SHA256 or marker_meta["sha256"] != plan["parent_payload_sha256"]:
        raise ValueError("parent checkpoint payload digest mismatch")
    if (payload.get("updates"), payload.get("attempts")) != (256, 256) or payload.get("phase") != 2:
        raise ValueError("parent checkpoint must be QueryMix final 256/256 phase2")
    if payload.get("contract", {}).get("source") != source_digest() or payload["contract"].get("variant") != "masked":
        raise ValueError("parent model/core contract mismatch")
    config = Config(**payload["contract"]["config"])
    configure_determinism()
    if verify_strict_load:
        original = FDT100(config)
        bound = QueryBoundRefiner(config)
        if original.counts()["parameters"] != 102237197 or bound.counts()["parameters"] != 102237197:
            raise ValueError("both implementations must retain exactly 102,237,197 parameters")
        if set(original.state_dict()) != set(bound.state_dict()):
            raise ValueError("bound candidate state-dict keys differ from original FDT100")
        original.load_state_dict(payload["model"], strict=True)
        bound.load_state_dict(payload["model"], strict=True)
        original_state = state_sha256(original.state_dict())
        bound_state = state_sha256(bound.state_dict())
        payload_state = state_sha256(payload["model"])
        if original_state != bound_state or original_state != payload_state or original_state != plan["parent_model_state_sha256"]:
            raise ValueError("strict-loaded implementations differ from the pinned parent weights")
        del original, bound
    del payload
    if output.exists():
        output_name_contract({p.name for p in output.iterdir()}, allow_execution=True)
    return {"plan": plan, "data_validation": checked_data, "status": "CPU_PREFLIGHT_PASS"}


class CallBudget:
    def __init__(self, cap, deadline, stop_path):
        self.cap, self.deadline, self.stop_path = cap, deadline, stop_path
        self.calls = 0
        self.graph_exposures = {"original": 0, "bound": 0}
        self.active_query_position_exposures = {"original": 0, "bound": 0}
        self.encoder_rows = {"original": 0, "bound": 0}
        self.fact_instances = {"original": 0, "bound": 0}

    def charge(self, variant, rows, active_q, fact_count, kind):
        if time.time() >= self.deadline:
            raise TimeoutError("readonly diagnostic worker deadline reached")
        if self.stop_path.exists():
            raise InterruptedError("owned supervisor requested cooperative stop")
        if self.calls >= self.cap:
            raise RuntimeError("actual batched model call cap exceeded")
        self.calls += 1
        self.graph_exposures[variant] += rows
        self.active_query_position_exposures[variant] += rows * active_q
        if kind == "encode":
            self.encoder_rows[variant] += rows
            self.fact_instances[variant] += fact_count


def paired_noise(batch_start, z_dim, device):
    generator = torch.Generator(device=device).manual_seed(SEED + batch_start)
    return torch.randn((4, 16, z_dim), device=device, generator=generator, dtype=torch.float32)


def rollout_counted(model, encoded, noise, variant, budget, q_count, fact_count):
    assert_finite_tensors(encoded, f"{variant}_encoded")
    assert_finite_tensors({"noise": noise}, f"{variant}_noise")
    z = noise.float() * encoded["query_mask"][..., None]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for t, t_next in zip(ROLLOUT_SCHEDULE, ROLLOUT_SCHEDULE[1:]):
            budget.charge(variant, len(z), q_count, fact_count, "denoise")
            times = z.new_full((len(z),), t)
            velocity = model.denoise(z, times, encoded).float()
            assert_finite_tensors({"velocity": velocity}, f"{variant}_denoise")
            a, s = math.cos(math.pi * t / 2), math.sin(math.pi * t / 2)
            if t == 1.0:
                a, s = 0.0, 1.0
            clean = a * z - s * velocity
            eps = s * z + a * velocity
            z = math.cos(math.pi * t_next / 2) * clean + math.sin(math.pi * t_next / 2) * eps
            assert_finite_tensors({"state": z}, f"{variant}_rollout_state")
    return z * encoded["query_mask"][..., None]


def summarize(rows):
    cm = [[0] * 4 for _ in range(4)]
    correct = 0
    active = 0
    direction_errors = 0
    unknown_errors = 0
    conflict_errors = 0
    for row in rows:
        for gold, pred in zip(row["gold"], row["pred"]):
            gold, pred = int(gold), int(pred)
            cm[gold][pred] += 1
            active += 1
            correct += int(gold == pred)
            if gold in (1, 2) and pred in (1, 2) and gold != pred:
                direction_errors += 1
            if (gold == 0) != (pred == 0):
                unknown_errors += 1
            if (gold == 3) != (pred == 3):
                conflict_errors += 1
    gold_support = [sum(row_count) for row_count in cm]
    predicted_support = [sum(cm[g][p] for g in range(4)) for p in range(4)]
    return {"correct": correct, "active_queries": active, "accuracy": correct / active if active else None,
        "confusion_gold_rows_pred_columns": cm, "direction_inversions_1_vs_2": direction_errors,
        "unknown_boundary_errors": unknown_errors, "conflict_boundary_errors": conflict_errors,
        "gold_class_support": gold_support, "predicted_class_support": predicted_support}


def run_diagnostic(plan_path: Path, output: Path, worker_path: Path):
    checked = preflight(plan_path, output, worker_path, verify_strict_load=False)
    plan = checked["plan"]
    if plan.get("authorized") is not True:
        raise ValueError("GPU inference requires a separately sealed/authorized plan")
    state = io.load_json(output / "supervisor/state.json")
    if not isinstance(state, dict):
        raise ValueError("owned supervisor launch state missing")
    launch_unix = float(state["invocation_started_unix"])
    worker_deadline = float(state["worker_deadline_unix"])
    budget = CallBudget(plan["max_actual_batched_model_calls"], worker_deadline,
        Path(plan["control_dir"]) / "STOP")
    progress_path = output / "worker_progress.json"
    def progress(stage, anomaly=None):
        io.atomic_json(progress_path, {"stage": stage, "observed_unix": time.time(), "worker_pid": os.getpid(),
            "actual_batched_model_calls": budget.calls, "max_actual_batched_model_calls": budget.cap,
            "graph_forward_exposures": budget.graph_exposures,
            "active_query_position_forward_exposures": budget.active_query_position_exposures,
            "updates": 0, "optimizer_used": False, "anomaly": anomaly})
    for reserved in ("diagnostic_result.json", "paired_seed17_noise.pt", "worker_progress.json", "incident.json"):
        if (output / reserved).exists():
            raise FileExistsError(f"refusing to overwrite readonly evidence {reserved}")
    marker_meta, payload = common.ResilientCheckpointStore(Path(plan["parent_marker"]).parent).verify(Path(plan["parent_marker"]))
    configure_determinism()
    cfg = Config(**payload["contract"]["config"])
    original = FDT100(cfg).to("cuda")
    bound = QueryBoundRefiner(cfg).to("cuda")
    if original.counts()["parameters"] != 102237197 or bound.counts()["parameters"] != 102237197:
        raise ValueError("model parameter count mismatch")
    original.load_state_dict(payload["model"], strict=True)
    bound.load_state_dict(payload["model"], strict=True)
    original_state = state_sha256(original.state_dict())
    bound_state = state_sha256(bound.state_dict())
    if original_state != bound_state or original_state != plan["parent_model_state_sha256"]:
        raise ValueError("runtime model weights do not match the frozen parent")
    del payload
    for model in (original, bound):
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    rows_by_q = {q: io.load_json(Path(plan["dataset_dir"]) / f"q{q:02d}.json") for q in QUERY_COUNTS}
    base_noise = []
    per_query_results = {q: [] for q in QUERY_COUNTS}
    initializer_equal_batches = 0
    initializer_max_abs_delta = 0.0
    progress("inference_start")
    try:
        for q in QUERY_COUNTS:
            rows = rows_by_q[q]
            for batch_start in range(0, 16, 4):
                batch = rows[batch_start:batch_start + 4]
                features, _, _ = collate(batch, cfg, "cuda")
                if set(features) != FEATURE_KEYS or features["query_mask"].sum(-1).tolist() != [q] * 4:
                    raise ValueError("actual model input contract/mask mismatch")
                # Labels and metadata are only read after both models finish all calls for this batch.
                facts_per_row = [len(row["facts"]) for row in batch]
                noise = paired_noise(batch_start, cfg.z_dim, "cuda")
                if q == 16:
                    base_noise.append(noise.detach().cpu())
                encoded = {}
                for name, model in (("original", original), ("bound", bound)):
                    budget.charge(name, len(batch), q, sum(facts_per_row), "encode")
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        encoded[name] = model.encode(features)
                    assert_finite_tensors(encoded[name], f"{name}_encode")
                init_logits = {name: model.readout(encoded[name]["init"]).float()
                    for name, model in (("original", original), ("bound", bound))}
                assert_finite_tensors(init_logits, "initializer_readout")
                init_delta = float((init_logits["original"] - init_logits["bound"]).abs().max())
                initializer_max_abs_delta = max(initializer_max_abs_delta, init_delta)
                if not torch.equal(init_logits["original"], init_logits["bound"]):
                    raise ValueError("both candidate classes must preserve identical original initializer outputs")
                initializer_equal_batches += 1
                states = {}
                for name, model in (("original", original), ("bound", bound)):
                    states[name] = rollout_counted(model, encoded[name], noise, name, budget, q, sum(facts_per_row))
                logits = {name: model.readout(states[name]).float()
                    for name, model in (("original", original), ("bound", bound))}
                assert_finite_tensors(states, "rollout_output")
                assert_finite_tensors(logits, "final_readout")
                for bi, row in enumerate(batch):
                    active_queries = row["queries"]
                    labels = [int(x) for x in row["labels"]]
                    raw = {"canonical_graph_key": row["canonical_graph_key"],
                        "transfer_graph_index": row["transfer_graph_index"], "query_count": q,
                        "queries": active_queries, "gold": labels,
                        "query_roles": [[int(x) for x in (s, o)] for s, o in active_queries],
                        "noise_sha256_full16_base": tensor_sha256(noise[bi:bi+1]),
                        "initializer_logits": init_logits["original"][bi, :q].cpu().tolist(),
                        "original_full_logits": logits["original"][bi, :q].cpu().tolist(),
                        "bound_logits": logits["bound"][bi, :q].cpu().tolist(),
                        "original_pred": logits["original"][bi, :q].argmax(-1).cpu().tolist(),
                        "bound_pred": logits["bound"][bi, :q].argmax(-1).cpu().tolist()}
                    per_query_results[q].append(raw)
                progress(f"q{q:02d}_batch_{batch_start + 4}_of_16")
        expected = expected_call_counts()
        if budget.calls != expected["actual_batched_model_calls"]:
            raise RuntimeError(f"actual call count mismatch: {budget.calls}")
        noise_path = output / "paired_seed17_noise.pt"
        tmp = noise_path.with_suffix(".pt.tmp")
        torch.save(torch.cat(base_noise, dim=0), tmp)
        io.replace_with_retry(tmp, noise_path)
        panels = {}
        for q in QUERY_COUNTS:
            rows = per_query_results[q]
            panels[str(q)] = {"initializer": summarize([dict(r, pred=[max(range(4), key=r["initializer_logits"][i].__getitem__)
                    for i in range(q)]) for r in rows]),
                "original_refiner": summarize([dict(r, pred=r["original_pred"]) for r in rows]),
                "query_bound_refiner": summarize([dict(r, pred=r["bound_pred"]) for r in rows]),
                "graph_count": len(rows), "active_query_positions": 16 * q,
                "correlated_with_other_query_count_views": True}
        byq = {q: {row["canonical_graph_key"]: row for row in per_query_results[q]} for q in QUERY_COUNTS}
        paired = {}
        for small, large in ((1, 4), (4, 8), (8, 16), (1, 16), (4, 16), (8, 16)):
            common_q = small
            comparisons = []
            aggregate = {variant: {"small_correct": 0, "large_shared_correct": 0,
                "prediction_agreements": 0, "shared_query_positions": 0}
                for variant in ("original", "bound")}
            for key in byq[small]:
                a, b = byq[small][key], byq[large][key]
                if a["queries"] != b["queries"][:common_q] or a["noise_sha256_full16_base"] != b["noise_sha256_full16_base"]:
                    raise ValueError("cross-count paired query/noise alignment mismatch")
                for variant, pred_key in (("original", "original_pred"), ("bound", "bound_pred")):
                    equal = sum(int(x == y) for x, y in zip(a[pred_key], b[pred_key][:common_q]))
                    small_correct = sum(int(x == y) for x, y in zip(a[pred_key], a["gold"]))
                    large_correct = sum(int(x == y) for x, y in zip(b[pred_key][:common_q], b["gold"][:common_q]))
                    aggregate[variant]["small_correct"] += small_correct
                    aggregate[variant]["large_shared_correct"] += large_correct
                    aggregate[variant]["prediction_agreements"] += equal
                    aggregate[variant]["shared_query_positions"] += common_q
                    comparisons.append({"graph": key, "variant": variant, "shared_queries": common_q,
                        "small_view_correct": small_correct, "large_view_shared_correct": large_correct,
                        "predictions_agree": equal})
            summary = {}
            for variant, totals in aggregate.items():
                denominator = totals["shared_query_positions"]
                small_acc = totals["small_correct"] / denominator
                large_acc = totals["large_shared_correct"] / denominator
                summary[variant] = {**totals, "small_accuracy": small_acc,
                    "large_view_shared_accuracy": large_acc,
                    "paired_accuracy_delta_large_minus_small": large_acc - small_acc,
                    "prediction_agreement_rate": totals["prediction_agreements"] / denominator}
            paired[f"q{small}_vs_q{large}"] = {"shared_query_count_per_graph": common_q,
                "graph_count": 16, "summary_over_only_shared_query_prefix": summary,
                "per_graph": comparisons}
        final_original_state = state_sha256(original.state_dict())
        final_bound_state = state_sha256(bound.state_dict())
        if final_original_state != plan["parent_model_state_sha256"] or final_bound_state != plan["parent_model_state_sha256"]:
            raise ValueError("read-only diagnostic modified model weights/buffers")
        physical_fact_phrase_bytes = sum(sum(len(f[3].encode("utf-8")) for f in row["facts"])
            for row in rows_by_q[16])
        physical_query_id_bytes = sum(len(str(entity).encode("utf-8")) for row in rows_by_q[16]
            for query in row["queries"] for entity in query)
        result = {"schema": "fdt_alignment_transfer_querycount_readonly_result_v1",
            "status": "READONLY_ALIGNMENT_TRANSFER_COMPLETE", "parent_payload_sha256": marker_meta["sha256"],
            "dataset_manifest_sha256": plan["dataset_manifest_sha256"],
            "selection_registry_sha256": plan["selection_registry_sha256"],
            "query_count_views": panels, "paired_shared_query_comparisons": paired,
            "raw_per_graph_per_q": {str(q): per_query_results[q] for q in QUERY_COUNTS},
            "paired_noise_file": {"path": str(noise_path), "sha256": sha256(noise_path),
                "seed": SEED, "shape": [16, 16, cfg.z_dim], "reused_for_all_query_counts_and_both_models": True},
            "budget": {**expected, "actual_batched_model_calls": budget.calls,
                "graph_forward_exposures_by_variant": budget.graph_exposures,
                "active_query_position_forward_exposures_by_variant": budget.active_query_position_exposures,
                "encoder_graph_row_exposures_by_variant": budget.encoder_rows,
                "encoder_fact_instance_exposures_by_variant": budget.fact_instances,
                "physical_premise_phrase_utf8_bytes_once": physical_fact_phrase_bytes,
                "physical_query_entity_id_utf8_bytes_once": physical_query_id_bytes,
                "actual_encoder_fact_instance_exposures_are_separately_counted": True},
            "integrity": {"parent_model_state_sha256": plan["parent_model_state_sha256"],
                "original_final_state_sha256": final_original_state,
                "bound_final_state_sha256": final_bound_state,
                "both_final_states_match_parent_including_buffers": True,
                "initializer_outputs_equal_for_every_batch": initializer_equal_batches == 16,
                "initializer_batches_compared": initializer_equal_batches,
                "initializer_max_abs_logit_delta": initializer_max_abs_delta,
                "all_encoded_velocity_state_and_logit_tensors_finite": True},
            "elapsed_seconds": time.time() - launch_unix, "wall_seconds_cap": WALL_SECONDS,
            "optimizer_used": False, "checkpoint_written": False, "e2_final_files_accessed": False,
            "interpretation_limit": "Descriptive transfer of the binding change across query-count views on 16 fixed new graphs; 64 views are correlated, not independent. No benchmark promotion or causal generalization claim.",
            "independence_claimed": False}
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
        preflight(args.plan.resolve(), args.output.resolve(), Path(__file__).resolve())
        print("CPU_PREFLIGHT_PASS")
        return 0
    run_diagnostic(args.plan.resolve(), args.output.resolve(), Path(__file__).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
