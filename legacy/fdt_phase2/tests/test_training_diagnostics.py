from pathlib import Path

import importlib.util
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_training_module():
    path = ROOT / "scripts" / "train_next_model.py"
    spec = importlib.util.spec_from_file_location("train_next_model_diagnostics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_explicit_decay_keeps_resume_lr_independent_of_extended_target():
    module = load_training_module()
    settings = {
        "learning_rate": 4e-5,
        "lr_schedule": "cosine",
        "lr_schedule_start_step": 6104,
        "lr_warmup_steps": 750,
        "lr_decay_steps": 12207,
        "lr_min_ratio": 0.15,
        "lr_resume_max_ratio": 1.25,
    }
    current_final = 18311
    extended_final = 61036
    current = module.scheduled_learning_rate(settings, 16832, 16831, current_final)
    extended = module.scheduled_learning_rate(settings, 16832, 16831, extended_final)

    assert current == pytest.approx(extended)
    event = module.validate_resume_learning_rate(settings, 7.3826433e-6, 16831, extended_final)
    assert event["ratio"] < 1.01
    assert event["explicit_decay_steps"] == 12207


def test_resume_lr_guard_rejects_target_dependent_rebound():
    module = load_training_module()
    unsafe = {
        "learning_rate": 4e-5,
        "lr_schedule": "cosine",
        "lr_schedule_start_step": 6104,
        "lr_warmup_steps": 750,
        "lr_min_ratio": 0.15,
        "lr_resume_max_ratio": 1.25,
    }
    with pytest.raises(RuntimeError, match="Refusing discontinuous resume learning rate"):
        module.validate_resume_learning_rate(unsafe, 7.3826433e-6, 16831, 61036)


def test_500m_stage_schedule_resumes_smoothly_and_ends_at_floor():
    module = load_training_module()
    settings = {
        "learning_rate": 1.2e-5,
        "lr_schedule": "cosine",
        "lr_schedule_start_step": 18311,
        "lr_warmup_steps": 500,
        "lr_warmup_start": 6e-6,
        "lr_decay_steps": 12207,
        "lr_min_ratio": 0.5,
        "lr_resume_max_ratio": 1.25,
    }

    first = module.scheduled_learning_rate(settings, 18312, 18311, 30518)
    peak = module.scheduled_learning_rate(settings, 18811, 18311, 30518)
    final = module.scheduled_learning_rate(settings, 30518, 18311, 30518)
    event = module.validate_resume_learning_rate(settings, 6e-6, 18311, 29660)

    assert first == pytest.approx(6.012e-6)
    assert peak == pytest.approx(1.2e-5)
    assert final == pytest.approx(6e-6)
    assert event["ratio"] < 1.01


def test_evenly_spaced_validation_covers_full_split_without_duplicates():
    module = load_training_module()
    indices = module.validation_batch_indices(
        total_rows=702,
        batch_size=1,
        eval_batches=20,
        sampling="evenly_spaced",
    )

    assert len(indices) == len(set(indices)) == 20
    assert indices[0] == 0
    assert indices[-1] == 701
    assert sum(index >= 568 for index in indices) >= 3


def test_indirect_training_order_matches_materialized_interleave():
    module = load_training_module()
    ids = torch.arange(24, dtype=torch.int32).view(6, 4)
    mask = torch.ones_like(ids, dtype=torch.bool)
    order = torch.tensor([4, 1, 5, 0, 3, 2], dtype=torch.long)

    batch_ids, batch_mask, next_index = module.select_training_batch(
        ids,
        mask,
        sequence_index=1,
        batch_size=3,
        device=torch.device("cpu"),
        train_order=order,
    )

    assert torch.equal(batch_ids, ids.index_select(0, order)[1:4].long())
    assert torch.equal(batch_mask, mask.index_select(0, order)[1:4].long())
    assert batch_ids.dtype == torch.long
    assert batch_mask.dtype == torch.long
    assert next_index == 4


def test_indirect_training_order_wraps_without_changing_stream_order():
    module = load_training_module()
    ids = torch.arange(20, dtype=torch.int32).view(5, 4)
    mask = torch.ones_like(ids, dtype=torch.bool)
    order = torch.tensor([3, 1, 4, 0, 2], dtype=torch.long)

    batch_ids, _, next_index = module.select_training_batch(
        ids,
        mask,
        sequence_index=4,
        batch_size=2,
        device=torch.device("cpu"),
        train_order=order,
    )

    assert torch.equal(batch_ids, ids.index_select(0, order)[:2].long())
    assert next_index == 2
