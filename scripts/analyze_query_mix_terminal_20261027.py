"""Read existing terminal artifacts on CPU; no model forward or optimizer."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import torch

from fdt_local.storage import CheckpointStore
from learning_io_v2 import atomic_json
from run_query_mix_diffusion_20261027 import (
    count_training_exposure, flatten_batch_indices, is_trainable,
    retention_streak, state_digest,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/query_mix_diffusion_20261027_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    native_cmd = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python(w)?\\.exe$' } | "
                  "Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3")
    native = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", native_cmd],
                            check=True, capture_output=True, text=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    processes = json.loads(native.stdout) if native.stdout.strip() else []
    if isinstance(processes, dict):
        processes = [processes]
    active = [p for p in processes if any(name in (p.get("CommandLine") or "") for name in
              ("run_query_mix_diffusion_20261027.py", "supervise_query_mix_diffusion.py"))]
    if active:
        raise RuntimeError("training owner is still alive; root analysis must wait")
    final, supervisor, plan = read(OUT / "final_status.json"), read(OUT / "supervisor/final.json"), read(OUT / "plan.json")
    assert final["status"] == supervisor["status"] == "COMPLETE"
    assert supervisor["exit_code"] == 0 and supervisor["launch_count"] == 1
    marker = Path(final["last_marker"])
    meta, payload = CheckpointStore(marker.parent).verify(marker)
    assert meta["sha256"] == final["last_checkpoint_sha256"] == "a877f316d0d97f2c497497fd9b941184463238055906d7109d3300db4c253b36"
    parent_marker = Path(plan["parent_marker"])
    _, parent = CheckpointStore(parent_marker.parent).verify(parent_marker)
    frozen_changes = [k for k, value in parent["model"].items()
                      if not is_trainable(k) and not torch.equal(value, payload["model"][k])]
    assert not frozen_changes
    assert (payload["updates"], payload["attempts"], payload["phase"], payload["phase_step"]) == (256, 256, 2, 256)
    required = {"optimizer", "scaler", "sampler_rng", "noise_rng", "torch_rng", "cuda_rng", "python_rng", "numpy_rng"}
    assert required <= payload.keys() and payload["optimizer"]["state"]
    history, runtime = payload["history"], payload["query_mix_runtime"]
    assert [x["attempt"] for x in history] == list(range(1, 257))
    assert not any(x["amp_skipped"] for x in history)
    rows = read(Path(plan["train_data_path"]))
    batches = [batch for entry in history for batch in entry["batch_indices"]]
    id_counts, class_counts, graph_counts = count_training_exposure(rows, batches)
    assert id_counts == runtime["attempted_id_exposure"] == runtime["successful_id_exposure"]
    assert class_counts == runtime["attempted_query_class_exposure"] == runtime["successful_query_class_exposure"]
    assert [graph_counts[i] for i in range(len(rows))] == runtime["unique_graph_exposure"]
    model_hash = state_digest(payload["model"])
    evaluation = read(OUT / "evaluation_summary.json")
    eval_markers = [(path, read(path)) for path in marker.parent.glob("complete_*.json")]
    eval_marker = next(path for path, m in eval_markers if m["sha256"] == evaluation["checkpoint_sha256"])
    _, eval_payload = CheckpointStore(marker.parent).verify(eval_marker)
    assert model_hash == state_digest(eval_payload["model"])
    milestone_hashes = {}
    for step in (64, 128, 192, 256):
        tagged = next(path for path, m in eval_markers if m.get("tag") == f"milestone_{step:04d}")
        milestone_hashes[str(step)] = CheckpointStore(marker.parent).verify(tagged, load=False)[0]["sha256"]
    baseline = read(OUT / "baseline/summary.json")
    comparisons, all_errors, diagnostics = {}, {}, {}
    for step in (64, 128, 192, 256):
        current = read(OUT / f"diagnostics/step{step:03d}/summary.json")
        streak, failures = retention_streak(0, baseline, current)
        assert not failures and streak == 0
        diagnostics[str(step)] = {name: value["results"]["ddim_4"]["accuracy"] for name, value in current.items()}
    current = read(OUT / "diagnostics/step256/summary.json")
    for name, before in baseline.items():
        after = current[name]
        pre, post = before["results"]["ddim_4"], after["results"]["ddim_4"]
        comparisons[name] = {"before_accuracy": pre["accuracy"], "after_accuracy": post["accuracy"],
            "absolute_gain": post["accuracy"] - pre["accuracy"],
            "before_macro_f1": pre["macro_f1"], "after_macro_f1": post["macro_f1"],
            "ground_accuracy": after["ground_accuracy"],
            "initializer_accuracy": after["results"]["initializer"]["accuracy"],
            "after_class_f1": post["class_f1"], "after_confusion": post["confusion"],
            "final_three_seed_accuracy": evaluation["panels"][name]["results"]["ddim_4"]["accuracy"]}
        raw = read(OUT / f"diagnostics/step256/{name}.json")["raw"]
        before_raw = read(OUT / f"baseline/{name}.json")["raw"]
        assert len(raw) == len(before_raw)
        confusion, example, corrections, regressions, total = Counter(), [], 0, 0, 0
        for row, old in zip(raw, before_raw):
            assert all(row[k] == old[k] for k in ("facts", "queries", "gold", "seed", "graph_key"))
            for q, gold, pred, initial, oldpred in zip(row["queries"], row["gold"], row["diffusion"]["4"], row["initializer"], old["diffusion"]["4"]):
                total += 1
                corrections += oldpred != gold and pred == gold
                regressions += oldpred == gold and pred != gold
                if pred != gold:
                    confusion[f"{gold}->{pred}"] += 1
                    if len(example) < 4:
                        example.append({"query": q, "gold": gold, "prediction": pred,
                            "initializer": initial, "facts": row["facts"], "seed": row["seed"], "graph_key": row["graph_key"]})
        all_errors[name] = {"count": sum(confusion.values()), "queries": total,
            "direction_inversions": confusion["1->2"] + confusion["2->1"],
            "false_positive": confusion["0->1"] + confusion["0->2"],
            "omission": confusion["1->0"] + confusion["2->0"], "confusions": dict(confusion),
            "corrected": corrections, "new_errors": regressions, "examples": example, "raw_graph_traces_read": len(raw)}
    flat = flatten_batch_indices(batches)
    source_digest = __import__("hashlib").sha256()
    for path in sorted((ROOT / "src/fdt_local").glob("*.py")):
        source_digest.update(path.name.encode()); source_digest.update(path.read_bytes())
    assert source_digest.hexdigest() == plan["core_sha256"]
    result = {"status": "ROOT_QUERYMIX_TERMINAL_CPU_REVIEW_COMPLETE",
        "observed_utc": datetime.now(timezone.utc).isoformat(), "owned_native_processes_still_running": active,
        "worker_status": final["status"], "supervisor_exit_code": supervisor["exit_code"],
        "fullstate_hash_and_safe_load_verified": True, "full_recovery_state_present": True,
        "checkpoint_sha256": meta["sha256"], "checkpoint_bytes": meta["bytes"],
        "evaluation_checkpoint_sha256": evaluation["checkpoint_sha256"], "evaluation_and_final_model_state_equal": True,
        "model_state_sha256": model_hash, "frozen_tensor_changes": frozen_changes,
        "updates": 256, "attempts": 256, "amp_skips": 0, "optimizer_updates_in_review": 0,
        "model_forwards_in_review": 0, "core_source_unchanged": True,
        "milestone_hashes": milestone_hashes, "diagnostic_accuracy": diagnostics,
        "same_seed17_comparisons": comparisons, "diagnostic_raw_errors": all_errors,
        "development_goals": evaluation["development_goals_vs_parent_baseline"], "all_retention_checks_pass": True,
        "actual_exposure_reconstructed_exactly": True, "sampled_rows": len(flat),
        "distinct_training_graphs_sampled": len(graph_counts), "input_utf8_bytes": payload["exposure"],
        "query_class_exposure": class_counts, "ID_exposure": id_counts,
        "step_seconds_sum": sum(x["step_seconds"] for x in history),
        "max_preclip_gradient_norm": max(x["gradient_norm_before_clip"] for x in history),
        "logical_forward_attempts": final["forward_attempts"],
        "worker_wall_seconds": final["observed_unix"] - read(OUT / "clock.json")["started_unix"],
        "E2_final_holdout_used": False, "auto_promoted": False,
        "next_step": "One bounded readonly paired query-context intervention; no repeated QueryMix optimizer extension."}
    atomic_json(OUT / "root_result.json", result)
    compact = {k: v for k, v in result.items() if k not in ("ID_exposure", "diagnostic_raw_errors", "same_seed17_comparisons")}
    compact["primary_errors"] = {k: v for k, v in all_errors["primary_querymix_dev64"].items() if k != "examples"}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
