"""Regression for the actual pre-step initialization assertion failure."""
from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_active_balanced_diffusion_20261027_v3 as worker

DEFAULTS = {"scale": 1024.0, "growth_factor": 2.0, "backoff_factor": .5,
            "growth_interval": 2000, "_growth_tracker": 0}


def test_fresh_enabled_scaler_is_nonempty_and_is_accepted():
    actual = torch.amp.GradScaler("cpu", enabled=True, init_scale=1024.0).state_dict()
    assert actual == DEFAULTS and actual
    worker.validate_fresh_runtime_state({}, {1, 2}, {1, 2}, [3e-5], actual)


@pytest.mark.parametrize("state", [{}, DEFAULTS | {"_growth_tracker": 128}, DEFAULTS | {"scale": 512.0}])
def test_disabled_or_previously_used_scaler_is_rejected(state):
    with pytest.raises(ValueError, match="GradScaler"):
        worker.validate_fresh_runtime_state({}, {1}, {1}, [3e-5], state)


def test_populated_adam_wrong_ownership_or_lr_still_rejected():
    for optimizer, actual, expected, rates in (({1: {"step": 1}}, {1}, {1}, [3e-5]),
            ({}, {1}, {1, 2}, [3e-5]), ({}, {1}, {1}, [3e-4])):
        with pytest.raises(ValueError, match="Adam/ownership/LR"):
            worker.validate_fresh_runtime_state(optimizer, actual, expected, rates, DEFAULTS)


def test_corrected_module_keeps_nested_exposure_and_same_seed_gain_contracts():
    assert worker.flatten_batch_indices([[0, 1, 0], [2]]) == [0, 1, 0, 2]
    result = worker.development_goal(.80, .84, .99)
    assert result["goal_met"] and result["seed17_absolute_gain"] == pytest.approx(.04)
