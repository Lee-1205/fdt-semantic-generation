from __future__ import annotations

import random

import torch

from scripts.prepare_capability_completion_v16 import CATEGORY_NAMES
from scripts.prepare_fdt_v21_knowledge_scale import (
    CATEGORY_COUNTS,
    CATEGORY_WEIGHTS,
    FACTUAL_LENGTH_QUOTAS,
    NATURAL_SOURCE_QUOTAS,
    V21Writer,
)
from scripts.train_capability_completion_v20 import capture_rng_state, restore_rng_state


def test_v21_objective_is_natural_language_and_knowledge_first() -> None:
    assert set(CATEGORY_WEIGHTS) == set(CATEGORY_NAMES)
    assert abs(sum(CATEGORY_WEIGHTS.values()) - 1.0) < 1e-9
    assert CATEGORY_WEIGHTS["natural_lm"] == 0.55
    assert CATEGORY_WEIGHTS["factual_qa"] == 0.25
    assert CATEGORY_WEIGHTS["retrieval"] == 0.15
    assert CATEGORY_WEIGHTS["python_code"] == 0.02
    assert CATEGORY_WEIGHTS["json"] == 0.03
    assert sum(CATEGORY_COUNTS.values()) == 100_000
    assert sum(FACTUAL_LENGTH_QUOTAS.values()) == CATEGORY_COUNTS["factual_qa"]
    assert sum(NATURAL_SOURCE_QUOTAS.values()) == CATEGORY_COUNTS["natural_lm"]


def test_v21_nonce_is_provenance_not_training_text() -> None:
    writer = V21Writer.__new__(V21Writer)
    writer.serial = 0
    assert writer.marker("natural_lm") == ""
    assert writer.serial == 1


def test_rng_checkpoint_state_restores_python_and_torch() -> None:
    random.seed(31)
    torch.manual_seed(31)
    state = capture_rng_state()
    expected_python = random.random()
    expected_torch = torch.rand(3)
    random.random()
    torch.rand(3)
    restore_rng_state(state)
    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)
