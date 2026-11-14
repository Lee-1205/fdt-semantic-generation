"""Read-only query-interference diagnostic for one frozen QueryMix dev slice."""
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
import traceback
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "scripts"))
from fdt_local.data import collate
from fdt_local.model_masked import Config, FDT100
from fdt_local.runner import configure_determinism, source_digest
from fdt_local.storage import CheckpointStore
import learning_io_v2 as io

EXPECTED_PARENT = "a877f316d0d97f2c497497fd9b941184463238055906d7109d3300db4c253b36"
LOGICAL_CAP = 200
WALL_CAP_SECONDS = 120
QUERY_COUNT = 16
SEED = 17
MODES = ("full16", "reversed16", "per_query_single")
DEV_INDICES = (0, 8, 16, 24, 32, 40, 48, 56)
FEATURE_KEYS = {"tokens", "roles", "fact_mask", "queries", "query_mask"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def reverse_queries(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError("query-axis tensor must have a query dimension")
    return value.flip(1)


def undo_reverse_queries(value: torch.Tensor) -> torch.Tensor:
    return reverse_queries(value)


def reverse_encoded_queries(encoded: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = dict(encoded)
    for key in ("init", "query_roles", "query_mask"):
        result[key] = reverse_queries(encoded[key])
    return result


def singleton_views(encoded: dict[str, torch.Tensor], noise: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Make B*Q independent one-query views from already encoded full16 rows."""
    b, q, _ = encoded["init"].shape
    if q != QUERY_COUNT or noise.shape[:2] != (b, q):
        raise ValueError("full16 query/noise shape mismatch")
    result: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if key in ("init", "query_roles", "query_mask"):
            result[key] = torch.stack([value[:, i] for i in range(q)], dim=1).reshape(b * q, 1, *value.shape[2:])
        else:
            expanded = value[:, None].expand(b, q, *value.shape[1:])
            result[key] = expanded.reshape(b * q, *value.shape[1:])
    one_noise = torch.stack([noise[:, i] for i in range(q)], dim=1).reshape(b * q, 1, noise.shape[-1])
    return result, one_noise


class ForwardBudget:
    def __init__(self, logical_cap: int, deadline_unix: float, stop_path: Path | None = None):
        if logical_cap <= 0:
            raise ValueError("logical forward cap must be positive")
        self.logical_cap = logical_cap
        self.deadline_unix = deadline_unix
        self.stop_path = stop_path
        self.logical_forwards = 0
        self.actual_batched_calls = 0

    def charge(self, logical_examples: int) -> None:
        if time.time() >= self.deadline_unix:
            raise TimeoutError("readonly diagnostic wall deadline reached")
        if self.stop_path is not None and self.stop_path.exists():
            raise InterruptedError("owned supervisor requested cooperative stop")
        if logical_examples <= 0 or self.logical_forwards + logical_examples > self.logical_cap:
            raise RuntimeError("readonly diagnostic logical forward cap exceeded")
        self.logical_forwards += logical_examples
        self.actual_batched_calls += 1


def rollout_counted(model, encoded, noise, *, logical_examples: int, budget: ForwardBudget):
    z = noise.float() * encoded["query_mask"][..., None]
    schedule = (1.0, .75, .5, .25, 0.0)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for t, t_next in zip(schedule, schedule[1:]):
            budget.charge(logical_examples)
            times = z.new_full((len(z),), t)
            velocity = model.denoise(z, times, encoded).float()
            a, s = math.cos(math.pi * t / 2), math.sin(math.pi * t / 2)
            if t == 1.0:
                a, s = 0.0, 1.0
            clean = a * z - s * velocity
            eps = s * z + a * velocity
            z = math.cos(math.pi * t_next / 2) * clean + math.sin(math.pi * t_next / 2) * eps
    return z * encoded["query_mask"][..., None]


def expected_forward_counts(graph_count: int) -> dict:
    # Logical count is per graph per model forward, following the fixed 3-view contract.
    logical = graph_count * (1 + 4 + 4 + 4)
    actual_batched_calls = (graph_count // 4) * (1 + 4 * len(MODES))
    return {"logical_forwards": logical, "actual_batched_calls": actual_batched_calls}


def paired_noise_for_indices(indices, *, z_dim: int, device="cpu", seed: int = SEED) -> torch.Tensor:
    """Recreate panel() noise for each original dev row's four-row batch and retain its row slice."""
    by_batch = {}
    pieces = []
    for index in indices:
        batch_start = (int(index) // 4) * 4
        if batch_start not in by_batch:
            generator = torch.Generator(device=device).manual_seed(seed + batch_start)
            by_batch[batch_start] = torch.randn((4, QUERY_COUNT, z_dim), device=device, generator=generator)
        pieces.append(by_batch[batch_start][int(index) - batch_start:int(index) - batch_start + 1])
    return torch.cat(pieces, dim=0)


def validate_rows(rows: list[dict], selected_hash: str | None = None) -> dict:
    if len(rows) != 64:
        raise ValueError("QueryMix dev split must contain exactly 64 rows")
    chosen = [rows[i] for i in DEV_INDICES]
    if any(row.get("n") != 8 or len(row.get("queries", [])) != 16
           or len(row.get("labels", [])) != 16 or len(row.get("hops", [])) != 16 for row in chosen):
        raise ValueError("fixed systematic sample rows must be full16 QueryMix examples")
    groups = [int(row["original_sampling_target_class"]) for row in chosen]
    if groups != [0, 0, 1, 1, 2, 2, 3, 3]:
        raise ValueError("systematic dev slice must cover two stored rows from each source group")
    features, _, _ = collate(chosen[:4], Config(), "cpu")
    if set(features) != FEATURE_KEYS or features["query_mask"].sum(-1).tolist() != [16] * 4:
        raise ValueError("model input feature whitelist or full16 mask mismatch")
    projection = [[row["graph_key"], row["facts"], row["queries"], row["labels"], row["hops"]]
                  for row in chosen]
    encoded = json.dumps(projection, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    if selected_hash is not None and digest != selected_hash:
        raise ValueError("fixed first-eight QueryMix dev projection changed")
    return {"count": len(chosen), "indices": list(DEV_INDICES), "projection_sha256": digest,
            "graph_keys": [row["graph_key"] for row in chosen]}


def validate_output_entries(output: Path) -> None:
    allowed = {"plan.json", "seal.json", "readiness.json", "code_snapshot",
        "supervisor.stdout.log", "supervisor.stderr.log", "supervisor", "worker.stdout.log",
        "worker.stderr.log", "control"}
    forbidden = {"clock.json", "diagnostic_result.json", "paired_seed17_noise.pt",
        "incident.json", "final_status.json", "model", "worker_progress.json"}
    names = {path.name for path in output.iterdir()} if output.exists() else set()
    validate_output_names(names, allowed, forbidden)


def validate_output_names(names, allowed=None, forbidden=None) -> None:
    if allowed is None:
        allowed = {"plan.json", "seal.json", "readiness.json", "code_snapshot",
            "supervisor.stdout.log", "supervisor.stderr.log", "supervisor", "worker.stdout.log",
            "worker.stderr.log", "control"}
    if forbidden is None:
        forbidden = {"clock.json", "diagnostic_result.json", "paired_seed17_noise.pt",
            "incident.json", "final_status.json", "model", "worker_progress.json"}
    unexpected = names - allowed
    present_forbidden = names & forbidden
    if unexpected or present_forbidden:
        raise FileExistsError(f"diagnostic output conflict: unexpected={sorted(unexpected)}, reserved={sorted(present_forbidden)}")


def preflight(plan_path: Path, output: Path, worker_path: Path) -> dict:
    plan = io.load_json(plan_path)
    if not isinstance(plan, dict) or plan.get("schema") != "fdt_query_interference_readonly_v1":
        raise ValueError("diagnostic plan schema mismatch")
    if plan.get("output_dir") != str(output.resolve()) or plan.get("worker_sha256") != sha256(worker_path):
        raise ValueError("output or worker digest mismatch")
    if plan.get("authorized") not in (False, True):
        raise ValueError("authorization marker must be boolean")
    if (plan.get("max_logical_forwards"), plan.get("wall_seconds"), plan.get("seed"),
            plan.get("query_count"), plan.get("graph_count")) != (200, 120, 17, 16, 8):
        raise ValueError("bounded fixed-slice diagnostic contract mismatch")
    if plan.get("optimizer_allowed") is not False or plan.get("checkpoint_write_allowed") is not False:
        raise ValueError("diagnostic must remain weights-only/read-only")
    if plan.get("e2_final_split_accessed") is not False:
        raise ValueError("E2 final split is prohibited")
    if Path(plan["output_dir"]).resolve() != output.resolve() or Path(plan["control_dir"]).resolve() != (output / "control").resolve():
        raise ValueError("diagnostic output/control paths mismatch")
    if Path(plan["parent_marker"]).resolve().name != "complete_000262.json":
        raise ValueError("diagnostic parent must be final QueryMix generation 262")
    if Path(plan["dev_data_path"]).resolve() != (ROOT / "research/artifacts/role8_query_mix_20261027_v1/dev.json").resolve():
        raise ValueError("unexpected QueryMix dev data source")
    deps = {
        "worker_sha256": worker_path,
        "supervisor_sha256": ROOT / "scripts/supervise_query_interference_diagnostic_20261027.py",
        "io_sha256": Path(io.__file__),
        "parent_marker_sha256": Path(plan["parent_marker"]),
        "dev_data_sha256": Path(plan["dev_data_path"]),
        "candidate_manifest_sha256": Path(plan["candidate_manifest_path"]),
        "data_source_sha256": ROOT / "src/fdt_local/data.py",
        "model_source_sha256": ROOT / "src/fdt_local/model_masked.py",
    }
    if any(plan.get(key) != sha256(path) for key, path in deps.items()):
        raise ValueError("diagnostic source/data/checkpoint digest mismatch")
    if source_digest() != plan["runtime_source_sha256"] or source_digest() != plan["core_sha256"]:
        raise ValueError("frozen FDT runtime source digest mismatch")
    marker_meta, payload = CheckpointStore(Path(plan["parent_marker"]).parent).verify(Path(plan["parent_marker"]))
    if marker_meta["sha256"] != plan["parent_payload_sha256"] or marker_meta["sha256"] != EXPECTED_PARENT:
        raise ValueError("final QueryMix checkpoint payload digest mismatch")
    if (payload.get("updates"), payload.get("attempts")) != (256, 256):
        raise ValueError("diagnostic parent must be terminal 256-success/256-attempt checkpoint")
    if payload.get("phase") != 2 or payload["contract"].get("variant") != "masked":
        raise ValueError("diagnostic parent phase/variant mismatch")
    if payload["contract"].get("source") != source_digest():
        raise ValueError("diagnostic parent frozen core mismatch")
    rows = io.load_json(plan["dev_data_path"])
    if len(rows) != 64:
        raise ValueError("QueryMix dev row count changed")
    selected = validate_rows(rows, plan["selected_projection_sha256"])
    if tuple(plan.get("selected_dev_indices", ())) != DEV_INDICES:
        raise ValueError("fixed systematic QueryMix dev selection changed")
    plan_caps = expected_forward_counts(8)
    if plan.get("expected_logical_forwards") != plan_caps["logical_forwards"] or plan.get("expected_actual_batched_calls") != plan_caps["actual_batched_calls"]:
        raise ValueError("expected diagnostic forward count mismatch")
    del payload
    return {"plan": plan, "selected": selected, "rows": [rows[i] for i in DEV_INDICES]}


def _repeat_static(encoded: dict[str, torch.Tensor], q: int) -> dict[str, torch.Tensor]:
    b = encoded["init"].shape[0]
    result = {}
    for key, value in encoded.items():
        if key in ("init", "query_roles", "query_mask"):
            continue
        result[key] = value[:, None].expand(b, q, *value.shape[1:]).reshape(b * q, *value.shape[1:])
    return result


def run_diagnostic(plan_path: Path, output: Path, worker_path: Path) -> dict:
    checked = preflight(plan_path, output, worker_path)
    plan, rows = checked["plan"], checked["rows"]
    if plan.get("authorized") is not True:
        raise ValueError("readonly GPU diagnostic requires separately authorized plan")
    validate_output_entries(output)
    supervisor_state = io.load_json(output / "supervisor/state.json")
    if not isinstance(supervisor_state, dict):
        raise ValueError("owned supervisor launch state is missing")
    launch_unix = float(supervisor_state["launched_unix"])
    deadline = float(supervisor_state["worker_deadline_unix"])
    stop_path = Path(plan["control_dir"]) / "STOP"
    budget = ForwardBudget(plan["max_logical_forwards"], deadline, stop_path)
    progress_path = output / "worker_progress.json"
    output.mkdir(parents=True, exist_ok=True)
    def progress(stage, anomaly=None):
        io.atomic_json(progress_path, {"stage": stage, "observed_unix": time.time(),
            "worker_pid": os.getpid(), "attempts": budget.logical_forwards, "updates": 0,
            "logical_forwards": budget.logical_forwards, "actual_batched_calls": budget.actual_batched_calls,
            "logical_cap": budget.logical_cap, "anomaly": anomaly})
    try:
        for reserved in ("diagnostic_result.json", "paired_seed17_noise.pt", "worker_progress.json", "incident.json"):
            if (output / reserved).exists():
                raise FileExistsError(f"refusing to overwrite diagnostic evidence: {reserved}")
        marker_meta, payload = CheckpointStore(Path(plan["parent_marker"]).parent).verify(Path(plan["parent_marker"]))
        configure_determinism()
        cfg = Config(**payload["contract"]["config"])
        model = FDT100(cfg).to("cuda")
        if model.counts()["parameters"] != 102237197:
            raise ValueError("masked model parameter count mismatch")
        model.load_state_dict(payload["model"], strict=True)
        del payload
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        raw_results = []
        noise_retained = []
        progress("inference_start")
        selected_indices = list(DEV_INDICES)
        for start in range(0, 8, 4):
            batch = rows[start:start + 4]
            original_indices = selected_indices[start:start + 4]
            features, targets, _ = collate(batch, cfg, "cuda")
            if set(features) != FEATURE_KEYS:
                raise ValueError("unexpected model input feature set")
            # Gold/labels/hops remain outside all model calls.
            budget.charge(len(batch))
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                encoded = model.encode(features)
            noise = paired_noise_for_indices(original_indices, z_dim=cfg.z_dim, device="cuda", seed=SEED)
            noise_retained.append(noise.detach().cpu())
            noise_sha = [tensor_sha256(noise[i:i+1]) for i in range(len(batch))]
            full_state = rollout_counted(model, encoded, noise, logical_examples=len(batch), budget=budget)
            reversed_enc = reverse_encoded_queries(encoded)
            reversed_noise = reverse_queries(noise)
            reversed_state = rollout_counted(model, reversed_enc, reversed_noise,
                logical_examples=len(batch), budget=budget)
            reversed_aligned = undo_reverse_queries(reversed_state)
            single_enc, single_noise = singleton_views(encoded, noise)
            single_state = rollout_counted(model, single_enc, single_noise,
                logical_examples=len(batch), budget=budget).reshape(len(batch), QUERY_COUNT, cfg.z_dim)
            logits = {
                "full16": model.readout(full_state).float(),
                "reversed16": model.readout(reversed_aligned).float(),
                "per_query_single": model.readout(single_state).float(),
                "initializer": model.readout(encoded["init"]).float(),
            }
            # Targets are consulted only now, after every inference view is complete.
            for bi, row in enumerate(batch):
                gold = [int(x) for x in row["labels"]]
                raw_results.append({"graph_key": row["graph_key"], "dev_row_index": original_indices[bi],
                    "queries": row["queries"], "gold": gold, "noise_seed": SEED + (original_indices[bi] // 4) * 4,
                    "noise_shape": [1, QUERY_COUNT, cfg.z_dim], "noise_sha256": noise_sha[bi],
                    "initializer_logits": logits["initializer"][bi].cpu().tolist(),
                    "full16_logits": logits["full16"][bi].cpu().tolist(),
                    "reversed16_logits_aligned": logits["reversed16"][bi].cpu().tolist(),
                    "per_query_single_logits": logits["per_query_single"][bi].cpu().tolist(),
                    "full_vs_reversed_argmax_disagreement": int((logits["full16"][bi].argmax(-1)
                        != logits["reversed16"][bi].argmax(-1)).sum()),
                    "full_vs_single_argmax_disagreement": int((logits["full16"][bi].argmax(-1)
                        != logits["per_query_single"][bi].argmax(-1)).sum()),
                    "full_vs_reversed_max_abs_logit_delta": float((logits["full16"][bi]
                        - logits["reversed16"][bi]).abs().max()),
                    "full_vs_single_max_abs_logit_delta": float((logits["full16"][bi]
                        - logits["per_query_single"][bi]).abs().max())})
            progress("inference_batch_complete")
        # Preserve the exact float32 noise tensors for reproducible paired inspection.
        noise_path = output / "paired_seed17_noise.pt"
        tmp = noise_path.with_suffix(".pt.tmp")
        torch.save(torch.cat(noise_retained, dim=0), tmp)
        os.replace(tmp, noise_path)
        score = {}
        for mode, field in (("initializer", "initializer_logits"), ("full16", "full16_logits"),
                            ("reversed16", "reversed16_logits_aligned"),
                            ("per_query_single", "per_query_single_logits")):
            correct = total = 0
            for row in raw_results:
                predicted = [max(range(4), key=rowlog.__getitem__) for rowlog in row[field]]
                correct += sum(int(a == b) for a, b in zip(predicted, row["gold"]))
                total += len(row["gold"])
            score[mode] = {"correct": correct, "queries": total, "accuracy": correct / total}
        result = {"schema": "fdt_query_interference_readonly_result_v1",
            "status": "READONLY_DIAGNOSTIC_COMPLETE", "parent_payload_sha256": marker_meta["sha256"],
            "selected_dev_indices": list(DEV_INDICES),
            "selected_projection_sha256": plan["selected_projection_sha256"],
            "graphs": len(raw_results), "queries_per_graph": QUERY_COUNT,
            "modes": {k: {"accuracy": v["accuracy"], "correct": v["correct"], "queries": v["queries"]}
                      for k, v in score.items()},
            "paired_noise": {"seed": SEED, "shape_per_graph": [1, QUERY_COUNT, cfg.z_dim],
                "file": str(noise_path), "sha256": sha256(noise_path),
                "same_noise_across_full_reverse_single": True},
            "logical_forwards": budget.logical_forwards,
            "expected_logical_forwards": plan["expected_logical_forwards"],
            "actual_batched_model_calls": budget.actual_batched_calls,
            "expected_actual_batched_calls": plan["expected_actual_batched_calls"],
            "logical_forward_cap": plan["max_logical_forwards"],
            "elapsed_seconds": time.time() - launch_unix,
            "wall_seconds_cap": plan["wall_seconds"],
            "optimizer_used": False, "checkpoint_written": False,
            "e2_final_split_accessed": False,
            "interpretation_limit": "full-vs-single bundles query-context/self-attention removal; it is not an isolated causal estimate",
            "raw": raw_results}
        if budget.logical_forwards != plan["expected_logical_forwards"] or budget.actual_batched_calls != plan["expected_actual_batched_calls"]:
            raise RuntimeError("actual forward calls differ from approved expected budget")
        io.atomic_json(output / "diagnostic_result.json", result)
        progress("terminal", anomaly=None)
        return result
    except BaseException as exc:
        progress("terminal", anomaly=f"{type(exc).__name__}: {exc}")
        io.atomic_json(output / "incident.json", {"error": f"{type(exc).__name__}: {exc}",
            "optimizer_used": False, "checkpoint_written": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
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
