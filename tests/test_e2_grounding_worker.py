import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from fdt_local.objectives import Plan

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

spec = importlib.util.spec_from_file_location("e2_worker", ROOT / "scripts" / "run_e2_grounding_20261026.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def test_candidate_features_exclude_labels_and_metadata():
    path = ROOT / "research" / "artifacts" / "e2_lexical_20261026_v2" / "train.json"
    rows = json.loads(path.read_text(encoding="utf-8"))[:2]
    info = worker.validate_rows(rows, expected_count=2)
    features, targets, _ = worker.neural_inputs(rows, worker.Config(), "cpu")
    assert info["facts"] == 16
    assert set(features) == worker.FEATURE_KEYS
    assert "ground" in targets and "labels" in targets
    assert not (set(features) & {"ground", "labels", "hops", "family_id", "graph_key"})


def test_grounding_optimizer_is_fresh_and_whitelist_only():
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language = torch.nn.Linear(2, 2)
            self.phrase_router = torch.nn.Linear(2, 2)
            self.ground_head = torch.nn.Linear(2, 2)
            self.anchors = torch.nn.Linear(2, 2)
            self.prior = torch.nn.Linear(2, 2)

    model = Tiny()
    optimizer, names = worker.fresh_grounding_optimizer(model)
    assert optimizer.state == {}
    assert names and all(name.startswith(worker.GROUND_PREFIXES) for name in names)
    assert all(p.requires_grad for name, p in model.named_parameters() if name.startswith(worker.GROUND_PREFIXES))
    assert all(not p.requires_grad for name, p in model.named_parameters() if name.startswith(worker.FROZEN_PREFIXES))
    assert [group["lr"] for group in optimizer.param_groups] == [1e-5, 3e-4]


def test_old_retention_stops_after_two_consecutive_failures():
    streak = worker.next_retention_streak(0, 0.94)
    assert streak == 1 and not worker.retention_stop(streak)
    streak = worker.next_retention_streak(streak, 0.90)
    assert streak == 2 and worker.retention_stop(streak)
    assert worker.next_retention_streak(streak, 0.95) == 0


def test_initializer_retention_failure_also_stops_after_two_observations():
    streak = worker.next_retention_streak(0, 1.0, old_initializer_accuracy=0.50)
    assert streak == 1 and not worker.retention_stop(streak)
    streak = worker.next_retention_streak(streak, 1.0, old_initializer_accuracy=0.50)
    assert streak == 2 and worker.retention_stop(streak)
    assert worker.next_retention_streak(streak, 1.0, old_initializer_accuracy=0.95) == 0


def test_stage_budget_and_parent_lineage_are_separate():
    assert worker.training_stop_reason(attempts=139, updates=127, max_attempts=140,
        target_updates=128, now=100, training_deadline=200) is None
    assert worker.training_stop_reason(attempts=139, updates=128, max_attempts=140,
        target_updates=128, now=100, training_deadline=200) == "TARGET_UPDATES_REACHED"
    assert worker.training_stop_reason(attempts=140, updates=127, max_attempts=140,
        target_updates=128, now=100, training_deadline=200) == "ATTEMPT_CAP"
    lineage = worker.stage_lineage(1440, 1440)
    assert lineage["parent_updates"] == 1440 and lineage["stage_updates"] == 0
    assert lineage["strict_optimizer_resume"] is False and lineage["weights_only_warmstart"] is True


def test_data_validator_rejects_duplicate_graph_and_unbalanced_grounding():
    rows = json.loads((ROOT / "research" / "artifacts" / "e2_lexical_20261026_v2" / "train.json").read_text(encoding="utf-8"))[:2]
    rows[1]["graph_key"] = rows[0]["graph_key"]
    with pytest.raises(ValueError, match="duplicate/invalid graph key"):
        worker.validate_rows(rows)
    rows = json.loads((ROOT / "research" / "artifacts" / "e2_lexical_20261026_v2" / "train.json").read_text(encoding="utf-8"))[:1]
    rows[0]["facts"][0][2] = rows[0]["facts"][1][2]
    with pytest.raises(ValueError, match="not balanced"):
        worker.validate_rows(rows)


def test_new_run_lock_creation_is_allowed_but_old_marker_is_rejected(tmp_path):
    output = tmp_path / "output"
    run_dir = output / "model"
    output.mkdir()
    worker.validate_new_stage_paths(output, run_dir)
    with worker.ResilientCheckpointStore(run_dir).lock():
        worker.validate_locked_run_dir(run_dir)

    stale_output = tmp_path / "stale"
    stale_run = stale_output / "model"
    stale_run.mkdir(parents=True)
    (stale_run / "complete_000001.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be new"):
        worker.validate_new_stage_paths(stale_output, stale_run)
    with pytest.raises(FileExistsError, match="beyond its newly acquired writer lock"):
        worker.validate_locked_run_dir(stale_run)


def test_legacy_development_rows_keep_raw_identity_without_family_id():
    row = {"graph_key": "legacy-graph", "facts": [[1, 2, 0, "is greater than"]], "queries": [[1, 2]]}
    raw = worker.raw_row_identity(row)
    assert raw == {"family_id": "legacy_mixed", "graph_key": "legacy-graph",
                   "facts": row["facts"], "queries": row["queries"]}


def test_resolved_stage_plan_records_new_rows_and_keeps_retention_split_separate():
    base_spec = {"variant": "masked", "profile": "full",
                 "plan": {"train_graphs": 512, "dev_graphs": 96}}
    stage = worker.stage_plan(Plan(train_graphs=512, dev_graphs=96), seed=20,
        phase_steps=(128, 320, 640), encoder_lr=1e-5, head_lr=3e-4,
        train_graphs=512, dev_graphs=64)
    resolved = worker.stage_spec(base_spec, stage)
    assert stage.train_graphs == 512 and stage.dev_graphs == 64
    assert resolved["plan"]["train_graphs"] == 512 and resolved["plan"]["dev_graphs"] == 64
    assert resolved["variant"] == "masked" and resolved["profile"] == "full"
    assert base_spec["plan"]["dev_graphs"] == 96
