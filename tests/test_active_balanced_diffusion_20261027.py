from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_active_balanced_diffusion_20261027 as active


ARTIFACT = ROOT / "research/artifacts/role8_active_balanced_20261027_v2"


def test_v2_dataset_quotas_and_all_query_class_weights():
    train = json.loads((ARTIFACT / "train.json").read_text(encoding="utf-8"))
    dev = json.loads((ARTIFACT / "dev.json").read_text(encoding="utf-8"))
    assert active.validate_active_balanced_rows(train, 1024, 256)["query_class_graph_counts"] == [256] * 4
    assert active.validate_active_balanced_rows(dev, 64, 16)["query_class_graph_counts"] == [16] * 4
    assert torch.equal(active.query_class_weights(train), torch.ones(4))
    assert {len(row["labels"]) for row in train} == {8}
    assert all(set(row["labels"]) == {row["target_class"]} for row in train)


def test_config_phase_parent_and_new_plan_bounds():
    spec = yaml.safe_load((ROOT / "configs/active_balanced_diffusion_20261027.yaml").read_text())
    assert spec["start_phase"] == 2
    assert spec["plan"]["phase_steps"] == [480, 320, 256]
    assert spec["plan"]["seed"] == 22
    assert spec["plan"]["train_graphs"] == 1024 and spec["plan"]["dev_graphs"] == 64
    assert spec["plan"]["diffusion_lr"] == 3e-5
    plan_path = ROOT / "research/artifacts/active_balanced_diffusion_20261027_v1/cpu_preflight_draft.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    active.validate_phase2_contract(plan)
    assert plan["authorized"] is False
    assert (plan["target_successful_updates"], plan["max_attempts"], plan["max_forward_attempts"]) == (256, 280, 6144)
    assert (plan["wall_seconds"], plan["reserve_seconds"], plan["final_eval_budget_seconds"]) == (1800, 180, 150)
    assert (plan["mixed_query_seed"], plan["mixed_queries_per_graph"]) == (22017, 16)


def test_mixed_dev_queries_reuse_same_facts_and_match_oracle():
    dev = json.loads((ARTIFACT / "dev.json").read_text(encoding="utf-8"))
    mixed = active.validate_mixed_queries(dev)
    again = active.make_mixed_queries(dev)
    assert len(mixed) == len(dev) == 64
    for original, row, repeated in zip(dev, mixed, again):
        assert row["facts"] == original["facts"]
        assert row["queries"] == repeated["queries"]
        assert len(row["queries"]) == 16 and len(set(map(tuple, row["queries"]))) == 16
        assert (row["labels"], row["hops"]) == active.oracle(8, row["facts"], row["queries"])
        assert row["canonical_graph_key"] == original["canonical_graph_key"]


def test_nested_engine_microbatches_reconstruct_graph_class_and_id_exposure():
    rows = json.loads((ARTIFACT / "train.json").read_text(encoding="utf-8"))[:4]
    nested = [[0, 1, 0, 2], [3, 1, 2, 0]]
    assert active.flatten_batch_indices(nested) == [0, 1, 0, 2, 3, 1, 2, 0]
    ids, classes, graphs = active.count_training_exposure(rows, nested)
    assert sum(classes) == 8 * 8
    assert sum(ids["query_source"]) == 8 * 8
    assert graphs == {0: 3, 1: 2, 2: 2, 3: 1}
    assert classes[rows[0]["target_class"]] >= 3 * 8
    runtime = {"attempted_id_exposure": {k: [0] * 8 for k in active.EXPOSURE_KEYS},
        "successful_id_exposure": {k: [0] * 8 for k in active.EXPOSURE_KEYS},
        "attempted_query_class_exposure": [0] * 4,
        "successful_query_class_exposure": [0] * 4,
        "unique_graph_exposure": [0] * len(rows)}
    active.update_training_exposure(runtime, rows, nested, successful=False)
    assert sum(runtime["attempted_query_class_exposure"]) == 64
    assert sum(runtime["successful_query_class_exposure"]) == 0
    assert sum(runtime["successful_id_exposure"]["query_source"]) == 0
    active.update_training_exposure(runtime, rows, nested, successful=True)
    assert sum(runtime["attempted_query_class_exposure"]) == 128
    assert sum(runtime["successful_query_class_exposure"]) == 64
    assert sum(runtime["successful_id_exposure"]["query_source"]) == 64
    assert sum(value > 0 for value in runtime["unique_graph_exposure"]) == 4
    assert sum(runtime["unique_graph_exposure"]) == 8


def test_mixed_view_and_old_role_dev_are_in_retention_guard():
    names = ("old_normal16", "old_large16", "old_role_dev128", "new_dev_mixed64")

    def result(acc, f1):
        return {name: {"ground_accuracy": 1., "results": {
            "initializer": {"accuracy": 1.},
            "ddim_4": {"accuracy": acc, "macro_f1": f1}}} for name in names}

    baseline = result(.98, .96)
    good = result(.98, .96)
    assert active.retention_streak(0, baseline, good) == 0
    degraded = result(.98, .96)
    degraded["new_dev_mixed64"]["results"]["ddim_4"]["accuracy"] = .93
    assert active.retention_streak(0, baseline, degraded) == 1
    assert active.retention_streak(1, baseline, degraded) == 2


def test_prospective_gain_uses_same_seed17_not_final_three_seed_mean():
    summary = active.development_goal(.50, .54, .80)
    assert summary["seed17_absolute_gain"] == pytest.approx(.04)
    assert summary["goal_met"] is True
    assert summary["final_three_seed_accuracy"] == .80


def test_sealed_role8_worker_and_supervisor_remain_byte_identical():
    expected = {
        "scripts/run_role8_diffusion_20261027.py": "d65cc3ffc2b34d5cb043355c3b3a8e25ad5a69160167dabd7e5ac96dbf926e17",
        "scripts/supervise_role8_diffusion.py": "c6e99105342f81418bb2143f47ca47c20c94cfbe36aaec178b2411e26c1a4914",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
