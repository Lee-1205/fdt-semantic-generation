from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from scripts.prepare_capability_completion_v16 import CATEGORY_NAMES, CATEGORY_TO_ID
from scripts.prepare_capability_completion_v18 import (
    CATEGORY_WEIGHTS,
    V18Writer,
    factual_length_bucket,
    require_c_path,
    repeated_ngram_rate,
)
from scripts.train_capability_completion_v18 import (
    STAGE,
    category_normalized_loss,
    validate_category_weights,
    validate_resume_checkpoint,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [2 + (ord(character) % 29) for character in text]


def test_v18_writer_uses_new_nonce_and_supervises_eos(tmp_path):
    writer = V18Writer(tmp_path, TinyTokenizer(), max_length=128, reference=set())
    assert writer.add("Question:", "answer", "factual_qa", max_answer_tokens=32)
    labels = writer.labels[0]
    assert labels[sum(writer.masks[0]) - 1] == writer.tokenizer.eos_token_id
    assert writer.row_categories == [CATEGORY_TO_ID["factual_qa"]]
    active = writer.ids[0][: sum(writer.masks[0])]
    marker = writer.tokenizer.encode("Source record v18-calibrated-retention-")
    assert active[: len(marker)] == marker


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [(1, "short_1_4"), (4, "short_1_4"), (5, "medium_5_7"), (7, "medium_5_7"), (8, "long_8_plus")],
)
def test_factual_length_bucket(tokens, expected):
    assert factual_length_bucket(tokens) == expected


def test_repeated_ngram_rate_detects_repeated_completion():
    assert repeated_ngram_rate([1, 2, 3, 4, 5, 6], 4) == 0.0
    assert repeated_ngram_rate([1, 2, 3, 4, 1, 2, 3, 4], 4) > 0.0


def test_v18_objective_weights_are_complete_positive_and_normalized():
    validate_category_weights(CATEGORY_WEIGHTS)
    with pytest.raises(ValueError, match="sum to one"):
        validate_category_weights({**CATEGORY_WEIGHTS, "natural_lm": 0.31})


def test_category_normalized_loss_applies_predeclared_weights():
    vocab = 8
    batch = len(CATEGORY_NAMES)
    logits = torch.zeros(batch, 2, vocab)
    labels = torch.full((batch, 2), -100, dtype=torch.long)
    labels[:, 1] = 2
    input_ids = torch.full((batch, 2), 2, dtype=torch.long)
    attention_mask = torch.ones(batch, 2, dtype=torch.long)
    categories = torch.arange(batch)
    for index in range(batch):
        logits[index, 0, 2] = float(index)

    completion, prompt, category_losses = category_normalized_loss(
        logits,
        input_ids,
        labels,
        attention_mask,
        categories,
        eos_id=1,
        eos_loss_weight=1.25,
        category_weights=CATEGORY_WEIGHTS,
    )
    expected_losses = torch.stack([
        F.cross_entropy(logits[index, 0].unsqueeze(0), torch.tensor([2]))
        for index in range(batch)
    ])
    expected = sum(
        expected_losses[index] * CATEGORY_WEIGHTS[name]
        for index, name in enumerate(CATEGORY_NAMES)
    )
    assert torch.allclose(torch.stack(category_losses), expected_losses)
    assert torch.allclose(completion, expected)
    assert float(prompt) == 0.0


def test_v18_rejects_non_c_path():
    with pytest.raises(ValueError, match="only C"):
        require_c_path(Path(r"D:\forbidden"))


def test_v18_resume_requires_exact_optimizer_and_cursor_continuity(tmp_path):
    dataset = tmp_path / "dataset"
    settings = {
        "stage": STAGE,
        "dataset_dir": str(dataset),
        "manifest_sha256": "ABCD",
        "batch_size": 10,
        "learning_rate": 5e-8,
        "weight_decay": 0.01,
        "prompt_lm_weight": 0.25,
        "eos_loss_weight": 1.25,
        "seed": 20260818,
        "lr_schedule": "constant",
        "lr_warmup_steps": 0,
        "freeze_routing": True,
    }
    checkpoint = {
        "train_config": settings,
        "stage_status": "PAUSED",
        "optimizer_state_saved": True,
        "optimizer_state_dict": {"state": {}, "param_groups": [{"weight_decay": 0.01}]},
        "optimizer_step": 411,
        "sample_cursor": 4110,
        "tokens_seen": 523544867,
    }
    assert validate_resume_checkpoint(
        checkpoint, dataset, "ABCD", 10, 5e-8, 0.01, 0.25, 1.25, 20260818, 5000
    ) == (411, 4110, 523544867)

    checkpoint["sample_cursor"] = 4100
    with pytest.raises(ValueError, match="cursor"):
        validate_resume_checkpoint(
            checkpoint, dataset, "ABCD", 10, 5e-8, 0.01, 0.25, 1.25, 20260818, 5000
        )
