from pathlib import Path
import hashlib
import json
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_query_mix_diffusion_20261027 as worker

DATA = ROOT / "research/artifacts/role8_query_mix_20261027_v1"


def test_querymix_train_dev_contract_and_all_query_class_weights():
    train = json.loads((DATA / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((DATA / "dev.json").read_text(encoding="utf-8"))
    assert worker.validate_query_mix_rows(train, 1024)["rows"] == 1024
    assert worker.validate_query_mix_rows(dev, 64)["rows"] == 64
    assert all("target_class" not in row and len(row["queries"]) == len(row["labels"]) == 16
               for row in train + dev)
    counts = [sum(label == cls for row in train for label in row["labels"]) for cls in range(4)]
    expected = torch.tensor(counts, dtype=torch.float32).clamp_min(1).reciprocal()
    expected = (expected / expected.mean()).clamp(.25, 4)
    assert torch.equal(worker.query_class_weights(train), expected)
    assert len(set(counts)) > 1


def test_new_parent_seed_phase_and_attempt_lineage_contract():
    spec = yaml.safe_load((ROOT / "configs/query_mix_diffusion_20261027.yaml").read_text())
    assert spec["start_phase"] == 2
    assert spec["plan"]["phase_steps"] == [480, 320, 256]
    assert spec["plan"]["seed"] == 23 and spec["plan"]["diffusion_lr"] == 3e-5
    assert spec["plan"]["train_graphs"] == 1024 and spec["plan"]["dev_graphs"] == 64
    plan = json.loads((ROOT / "research/artifacts/query_mix_diffusion_20261027_v1/cpu_preflight_draft.json")
                      .read_text(encoding="utf-8"))
    worker.validate_phase2_contract(plan)
    assert plan["authorized"] is False
    assert plan["parent_sha256"] == worker.PARENT_SHA256
    assert [plan[k] for k in ("parent_active_balanced_updates", "ancestor_role8_updates",
                               "ancestor_e2_updates", "ancestor_e1_updates")] == [128, 128, 128, 1440]
    assert [plan[k] for k in ("target_successful_updates", "max_attempts", "max_forward_attempts")] == [256, 280, 6144]
    assert [plan[k] for k in ("wall_seconds", "reserve_seconds", "final_eval_budget_seconds")] == [1800, 180, 150]
    assert [plan[k] for k in ("train_query_seed", "dev_query_seed", "queries_per_graph")] == [33017, 22017, 16]


def test_uniform_and_querymix_dev_are_same_facts_but_correlated_query_views():
    mixed = json.loads((DATA / "dev.json").read_text(encoding="utf-8"))
    uniform = json.loads((ROOT / "research/artifacts/role8_active_balanced_20261027_v2/dev.json")
                         .read_text(encoding="utf-8"))
    baseline = json.loads((ROOT / "runs/active_balanced_diffusion_20261027_runtime_v3/baseline/new_dev_mixed64.json")
                          .read_text(encoding="utf-8"))
    for qmix, v2, raw in zip(mixed, uniform, baseline["raw"]):
        assert qmix["facts"] == v2["facts"] == raw["facts"]
        assert qmix["queries"] == raw["queries"]
        assert len(qmix["queries"]) == 16 and len(v2["queries"]) == 8
        assert qmix["labels"] == raw["gold"]
        assert worker.query_mix_data.v2.independent_oracle(8, qmix["facts"], qmix["queries"]) == (
            qmix["labels"], qmix["hops"])


def _panel_result(acc, f1, ground=1., init=1.):
    return {"ground_accuracy": ground, "results": {
        "initializer": {"accuracy": init}, "ddim_4": {"accuracy": acc, "macro_f1": f1}}}


def test_relative_retention_is_distinct_from_absolute_primary_qualification():
    panels = {key: _panel_result(.96, .94) for key in
              worker.CONTROL_RETENTION_PANELS + worker.RELATIVE_RETENTION_PANELS}
    baseline = {key: _panel_result(.90, .90) for key in panels}
    current = dict(panels)
    current["primary_querymix_dev64"] = _panel_result(.93, .88)
    assert worker.retention_failures(baseline, current) == []
    assert worker.development_goal(.90, .93, .92)["qualification_met"] is False
    assert worker.development_goal(.90, .93, .92)["relative_gain_goal_met"] is True
    assert worker.retention_streak(0, baseline, current) == (0, [])
    current["uniform_dev64"] = _panel_result(.93, .94)
    streak, failures = worker.retention_streak(0, baseline, current)
    assert streak == 1 and failures[0]["panel"] == "uniform_dev64"
    assert failures[0]["guard"] == "qualified_control"


def test_nested_microbatch_exposure_reconstructs_all_queries_and_graphs():
    rows = json.loads((DATA / "train.json").read_text(encoding="utf-8"))[:4]
    nested = [[0, 1, 0, 2], [3, 1, 2, 0]]
    assert worker.flatten_batch_indices(nested) == [0, 1, 0, 2, 3, 1, 2, 0]
    id_counts, classes, graphs = worker.count_training_exposure(rows, nested)
    assert sum(classes) == 8 * 16 and sum(id_counts["query_source"]) == 8 * 16
    assert graphs == {0: 3, 1: 2, 2: 2, 3: 1}
    runtime = {"attempted_id_exposure": {k: [0] * 8 for k in worker.EXPOSURE_KEYS},
        "successful_id_exposure": {k: [0] * 8 for k in worker.EXPOSURE_KEYS},
        "attempted_query_class_exposure": [0] * 4,
        "successful_query_class_exposure": [0] * 4, "unique_graph_exposure": [0] * len(rows)}
    worker.update_training_exposure(runtime, rows, nested, successful=False)
    assert sum(runtime["attempted_query_class_exposure"]) == 128
    assert sum(runtime["successful_query_class_exposure"]) == 0
    worker.update_training_exposure(runtime, rows, nested, successful=True)
    assert sum(runtime["successful_query_class_exposure"]) == 128
    assert sum(runtime["unique_graph_exposure"]) == 8


def test_same_seed17_gain_and_sealed_predecessors_are_preserved():
    result = worker.development_goal(.50, .53, .80)
    assert result["seed17_absolute_gain"] == pytest.approx(.03)
    assert result["relative_gain_goal_met"] is True and result["qualification_met"] is False
    expected = {
        "scripts/run_active_balanced_diffusion_20261027_v3.py": "debc560f01a2f65da291e9a82ded2934187cd8017647794d18ab81331561eec4",
        "scripts/supervise_active_balanced_diffusion_v3.py": "5d30601d88c5023806efa450c47d8d7a0589f64a82936bc6176f67f9d8ea5c88",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
