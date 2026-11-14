from pathlib import Path

import torch

from scripts.prepare_capability_completion_v16 import CATEGORY_NAMES
from scripts.prepare_capability_completion_v20_t2 import (
    CATEGORY_COUNTS,
    CATEGORY_WEIGHTS,
    FACTUAL_LENGTH_QUOTAS,
    GROUNDING_LOSS_WEIGHTS,
    NATURAL_SOURCE_QUOTAS,
)
from scripts.train_capability_completion_v20_t2 import (
    STAGE,
    copy_grounding_loss,
    move_optimizer_state,
    optimizer_state_devices,
    validate_resume_checkpoint,
)


def loss_for_indices(logits, input_ids, labels, attention, categories, grounding, signature, indices):
    return copy_grounding_loss(
        logits.index_select(0, indices),
        input_ids.index_select(0, indices),
        labels.index_select(0, indices),
        attention.index_select(0, indices),
        categories.index_select(0, indices),
        grounding.index_select(0, indices),
        signature.index_select(0, indices),
        eos_id=1,
        eos_loss_weight=2.0,
        grounding_loss_weights=GROUNDING_LOSS_WEIGHTS,
        signature_loss_weight=1.25,
        category_weights=CATEGORY_WEIGHTS,
    )[0]


def test_two_balanced_microbatches_equal_effective_batch_objective():
    generator = torch.Generator().manual_seed(19)
    batch = 2 * len(CATEGORY_NAMES)
    logits = torch.randn(batch, 4, 11, generator=generator)
    input_ids = torch.randint(2, 11, (batch, 4), generator=generator)
    labels = torch.full((batch, 4), -100, dtype=torch.long)
    labels[:, 1:] = torch.randint(2, 11, (batch, 3), generator=generator)
    attention = torch.ones(batch, 4, dtype=torch.long)
    categories = torch.arange(len(CATEGORY_NAMES)).repeat_interleave(2)
    grounding = torch.zeros(batch, 4, dtype=torch.long)
    grounding[:, 1] = 1
    signature = torch.zeros(batch, 4, dtype=torch.long)
    signature[categories.eq(2), 2] = 1

    full = loss_for_indices(
        logits, input_ids, labels, attention, categories, grounding, signature,
        torch.arange(batch),
    )
    first = torch.arange(0, batch, 2)
    second = torch.arange(1, batch, 2)
    assert set(categories.index_select(0, first).tolist()) == set(range(len(CATEGORY_NAMES)))
    assert set(categories.index_select(0, second).tolist()) == set(range(len(CATEGORY_NAMES)))
    split = (
        loss_for_indices(logits, input_ids, labels, attention, categories, grounding, signature, first)
        + loss_for_indices(logits, input_ids, labels, attention, categories, grounding, signature, second)
    ) / 2.0
    assert torch.allclose(full, split, atol=1e-6)


def test_optimizer_state_can_be_offloaded_before_evaluation():
    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(2, 4)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    move_optimizer_state(optimizer, torch.device("cpu"))
    assert optimizer_state_devices(optimizer) == {"cpu"}
    assert all(parameter.grad is None for parameter in model.parameters())


def test_v20_resume_pins_microbatch_cursor_and_checkpoint_interval(tmp_path):
    dataset = tmp_path / "dataset"
    checkpoint = {
        "train_config": {
            "stage": STAGE,
            "dataset_dir": str(dataset),
            "manifest_sha256": "ABCD",
            "batch_size": 10,
            "micro_batch_size": 5,
            "learning_rate": 2.5e-8,
            "weight_decay": 0.01,
            "prompt_lm_weight": 0.35,
            "eos_loss_weight": 2.0,
            "category_grounding_loss_weights": GROUNDING_LOSS_WEIGHTS,
            "signature_loss_weight": 1.25,
            "checkpoint_interval": 3000,
            "seed": 20260821,
            "lr_schedule": "constant",
            "lr_warmup_steps": 0,
            "freeze_routing": True,
        },
        "stage_status": "PAUSED",
        "optimizer_state_saved": True,
        "optimizer_state_dict": {"state": {}, "param_groups": [{"weight_decay": 0.01}]},
        "optimizer_step": 1500,
        "sample_cursor": 15000,
        "tokens_seen": 523978299,
    }
    assert validate_resume_checkpoint(
        checkpoint, dataset, "ABCD", 10, 5, 2.5e-8, 0.01, 0.35, 2.0,
        GROUNDING_LOSS_WEIGHTS["factual_qa"], GROUNDING_LOSS_WEIGHTS["retrieval"],
        GROUNDING_LOSS_WEIGHTS["json"], 1.25, 3000, 20260821, 22500,
    ) == (1500, 15000, 523978299)


def test_v20_stage_and_scale_plan_are_predeclared():
    assert STAGE == "capability_completion_v20_balanced_scale_tranche_2"
    assert set(CATEGORY_COUNTS) == set(CATEGORY_NAMES)
    assert set(CATEGORY_COUNTS.values()) == {4500}
    assert sum(CATEGORY_COUNTS.values()) == 22500
    assert sum(FACTUAL_LENGTH_QUOTAS.values()) == CATEGORY_COUNTS["factual_qa"]
    assert sum(NATURAL_SOURCE_QUOTAS.values()) == CATEGORY_COUNTS["natural_lm"]
    assert abs(sum(CATEGORY_WEIGHTS.values()) - 1.0) < 1e-9
    assert CATEGORY_WEIGHTS["natural_lm"] == 0.25
    assert CATEGORY_WEIGHTS["retrieval"] == 0.25
    assert CATEGORY_WEIGHTS["json"] == 0.20
    assert GROUNDING_LOSS_WEIGHTS == {"factual_qa": 1.25, "retrieval": 2.25, "json": 2.25}
    assert Path(r"C:\safe").drive.upper() == "C:"
