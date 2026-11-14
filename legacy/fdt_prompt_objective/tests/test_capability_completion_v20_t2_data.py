from pathlib import Path

import pytest
import torch

from scripts.prepare_capability_completion_v16 import CATEGORY_NAMES, CATEGORY_TO_ID
from scripts.prepare_capability_completion_v20_t2 import (
    CATEGORY_WEIGHTS,
    GROUNDING_LOSS_WEIGHTS,
    PROMPT_LM_WEIGHT,
    V20Writer,
    hotpot_support_context,
    json_value_spans,
    python_signature_span,
    require_c_path,
    token_span_mask,
)
from scripts.train_capability_completion_v20_t2 import (
    STAGE,
    copy_grounding_loss,
    validate_category_weights,
    validate_resume_checkpoint,
)


class CharTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [2 + ord(character) for character in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        del add_special_tokens
        payload = {"input_ids": self.encode(text)}
        if return_offsets_mapping:
            payload["offset_mapping"] = [(index, index + 1) for index in range(len(text))]
        return payload


def test_span_mask_marks_only_overlapping_tokens():
    offsets = [(0, 1), (1, 3), (3, 4), (4, 8)]
    assert token_span_mask(offsets, [(1, 4)]) == [0, 1, 1, 0]


def test_json_value_spans_exclude_keys_and_include_nested_values():
    value = {"name": "Ada", "location": {"city": "Seoul"}, "scores": [3, 8]}
    serialized = '{"name":"Ada","location":{"city":"Seoul"},"scores":[3,8]}'
    selected = [serialized[start:end] for start, end in json_value_spans(serialized, value)]
    assert '"Ada"' in selected
    assert '"Seoul"' in selected
    assert "3" in selected and "8" in selected
    assert '"name"' not in selected and '"city"' not in selected


def test_python_signature_span_is_limited_to_definition_line():
    code = "def add(left: int, right: int) -> int:\n    return left + right\n"
    span = python_signature_span(code)
    assert span is not None
    assert code[slice(*span)] == "def add(left: int, right: int) -> int:"


def test_hotpot_support_context_uses_only_declared_support_sentences():
    row = {
        "context": {
            "title": ["Alpha", "Beta"],
            "sentences": [["keep alpha", "ignore alpha"], ["ignore beta", "keep beta"]],
        },
        "supporting_facts": {"title": ["Alpha", "Beta"], "sent_id": [0, 1]},
    }
    context = hotpot_support_context(row)
    assert "keep alpha" in context and "keep beta" in context
    assert "ignore alpha" not in context and "ignore beta" not in context


def test_v20_writer_aligns_grounding_signature_and_eos(tmp_path):
    writer = V20Writer(tmp_path, CharTokenizer(), max_length=256, reference=set())
    answer = "def add(x, y):\n    return x + y"
    signature = python_signature_span(answer)
    assert signature is not None
    assert writer.add_grounded(
        "Task: add two values. Required signature: def add(x, y):\nPython:\n",
        answer,
        "python_code",
        96,
        signature_spans=[signature],
    )
    active = sum(writer.masks[0])
    assert writer.labels[0][active - 1] == writer.tokenizer.eos_token_id
    assert writer.signature_masks[0][active - 1] == 0
    assert sum(writer.signature_masks[0]) == len("def add(x, y):")
    assert writer.row_categories == [CATEGORY_TO_ID["python_code"]]


def test_copy_grounding_loss_keeps_category_totals_normalized():
    batch = len(CATEGORY_NAMES)
    vocab = 8
    logits = torch.zeros(batch, 3, vocab)
    input_ids = torch.full((batch, 3), 2, dtype=torch.long)
    labels = torch.full((batch, 3), -100, dtype=torch.long)
    labels[:, 1:] = torch.tensor([2, 1])
    attention = torch.ones(batch, 3, dtype=torch.long)
    categories = torch.arange(batch)
    grounding = torch.zeros(batch, 3, dtype=torch.long)
    signature = torch.zeros(batch, 3, dtype=torch.long)
    grounding[0, 1] = 1
    signature[2, 1] = 1
    completion, prompt, category_losses = copy_grounding_loss(
        logits,
        input_ids,
        labels,
        attention,
        categories,
        grounding,
        signature,
        eos_id=1,
        eos_loss_weight=2.0,
        grounding_loss_weights=GROUNDING_LOSS_WEIGHTS,
        signature_loss_weight=1.25,
        category_weights=CATEGORY_WEIGHTS,
    )
    assert torch.isfinite(completion)
    assert len(category_losses) == len(CATEGORY_NAMES)
    assert float(prompt) == 0.0


def test_v20_objective_and_c_only_paths():
    validate_category_weights(CATEGORY_WEIGHTS)
    assert PROMPT_LM_WEIGHT == 0.35
    assert GROUNDING_LOSS_WEIGHTS["retrieval"] > GROUNDING_LOSS_WEIGHTS["factual_qa"]
    assert GROUNDING_LOSS_WEIGHTS["json"] > GROUNDING_LOSS_WEIGHTS["factual_qa"]
    with pytest.raises(ValueError, match="sum to one"):
        validate_category_weights({**CATEGORY_WEIGHTS, "natural_lm": 0.31})
    with pytest.raises(ValueError, match="only C"):
        require_c_path(Path(r"D:\forbidden"))


def test_v20_resume_requires_exact_identity_and_cursor(tmp_path):
    dataset = tmp_path / "dataset"
    settings = {
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
    }
    checkpoint = {
        "train_config": settings,
        "stage_status": "PAUSED",
        "optimizer_state_saved": True,
        "optimizer_state_dict": {"state": {}, "param_groups": [{"weight_decay": 0.01}]},
        "optimizer_step": 1500,
        "sample_cursor": 15000,
        "tokens_seen": 524000000,
    }
    assert validate_resume_checkpoint(
        checkpoint, dataset, "ABCD", 10, 5, 2.5e-8, 0.01, 0.35, 2.0,
        GROUNDING_LOSS_WEIGHTS["factual_qa"], GROUNDING_LOSS_WEIGHTS["retrieval"],
        GROUNDING_LOSS_WEIGHTS["json"], 1.25, 3000, 20260821, 22500,
    ) == (1500, 15000, 524000000)
    checkpoint["sample_cursor"] = 14990
    with pytest.raises(ValueError, match="cursor"):
        validate_resume_checkpoint(
            checkpoint, dataset, "ABCD", 10, 5, 2.5e-8, 0.01, 0.35, 2.0,
            GROUNDING_LOSS_WEIGHTS["factual_qa"], GROUNDING_LOSS_WEIGHTS["retrieval"],
            GROUNDING_LOSS_WEIGHTS["json"], 1.25, 3000, 20260821, 22500,
        )
