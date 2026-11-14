from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_role8_diffusion_20261027 as role8
from fdt_local.data import oracle


def valid_plan():
    return {"phase": 2, "target_successful_updates": 128, "max_attempts": 140,
            "seed": 21, "trainable_prefixes": list(role8.TRAINABLE_PREFIXES),
            "parent_updates": 128, "parent_attempts": 128,
            "parent_optimizer_imported": False, "parent_rng_imported": False,
            "e2_final_split_evaluated": False}


def row(key, family="a", rotation=0):
    facts = [[rotation, (rotation + 1) % 8, 0, "is greater than"]]
    queries = [[rotation, (rotation + 2) % 8]]
    labels, hops = oracle(8, facts, queries)
    return {"n": 8, "facts": facts, "queries": queries, "labels": labels, "hops": hops,
            "graph_key": key, "source_graph_key": family, "rotation": rotation}


def test_contract_accepts_only_expected_phase_and_parent_counters():
    role8.validate_phase2_contract(valid_plan())
    invalid = valid_plan() | {"parent_attempts": 1440}
    with pytest.raises(ValueError, match="parent counters"):
        role8.validate_phase2_contract(invalid)


def test_trainable_allowlist_and_frozen_digest():
    model = torch.nn.Module()
    for prefix in role8.TRAINABLE_PREFIXES:
        setattr(model, prefix.rstrip("."), torch.nn.Linear(2, 2))
    model.language = torch.nn.Linear(2, 2)
    before = role8.frozen_digest(model)
    selected = role8.configure_role8_trainables(model)
    assert all(role8.is_role8_trainable(name) for name, _ in selected)
    assert model.language.weight.requires_grad is False
    assert role8.frozen_digest(model) == before


def test_role8_row_validation_and_actual_exposure():
    rows = [row("k0", rotation=0), row("k1", rotation=1)]
    summary = role8.validate_role8_rows(rows, 2)
    assert summary["source_families"] == 1
    exposure = role8.count_batch_exposure(rows, [[0, 1], [1]])
    assert exposure["fact_source"] == [1, 2, 0, 0, 0, 0, 0, 0]
    assert exposure["query_object"] == [0, 0, 1, 2, 0, 0, 0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        role8.validate_role8_rows([row("same"), row("same")], 2)


def test_class_weighting_uses_all_queries_not_first_query_only():
    rows = [{"labels": [0, 1, 1, 2, 3, 3, 3, 3]}]
    inverse = torch.tensor([1., .5, 1., .25])
    assert torch.equal(role8.query_class_weights(rows), (inverse / inverse.mean()).clamp(.25, 4))


def test_saved_labels_cannot_silently_change_training_truth():
    altered = row("a")
    altered["labels"][0] = (altered["labels"][0] + 1) % 4
    with pytest.raises(ValueError, match="oracle"):
        role8.validate_role8_rows([altered], 1)


def test_old_retention_requires_two_failures_and_resets_on_recovery():
    def measurement(acc, f1):
        return {name: {"ground_accuracy": 1., "results": {
            "initializer": {"accuracy": 1.}, "ddim_4": {"accuracy": acc, "macro_f1": f1}}}
                for name in ("old_normal16", "old_large16")}
    baseline = measurement(.98, .95)
    assert role8.retention_streak(0, baseline, measurement(.95, .95)) == 1
    assert role8.retention_streak(1, baseline, measurement(.95, .95)) == 2
    assert role8.retention_streak(1, baseline, measurement(.98, .89)) == 2
    assert role8.retention_streak(1, baseline, measurement(.98, .95)) == 0


def test_all_source_rows_fit_actual_model_capacities():
    import json
    rows = json.loads((ROOT / "research/artifacts/role8_coverage_20261026_v1/train.json").read_text(encoding="utf-8"))
    info = role8.validate_role8_rows(rows, 512)
    assert info["source_families"] == 64
    assert info["family_rotation_counts"] == [8] * 64
    assert all(len(set(values)) == 1 for values in info["entity_id_exposure"].values())
