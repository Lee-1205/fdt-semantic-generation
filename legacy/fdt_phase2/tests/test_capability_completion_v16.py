from pathlib import Path

import pytest
import torch

from scripts.prepare_capability_completion_v16 import (
    CATEGORY_NAMES,
    CATEGORY_TO_ID,
    V16Writer,
    profile_indices,
)
from scripts.select_capability_completion_v16 import require_c_path
from scripts.train_capability_completion_v16 import (
    STAGE,
    unexpected_output_files,
    validate_resume_checkpoint,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [2 + (ord(character) % 29) for character in text]


def test_v16_writer_masks_prompt_supervises_eos_and_records_category(tmp_path):
    writer = V16Writer(tmp_path, TinyTokenizer(), max_length=64, reference=set())
    assert writer.add("Question:", "answer", "factual_qa", max_answer_tokens=16)
    prompt_length = len(writer.tokenizer.encode("Question:"))
    labels = writer.labels[0]
    assert labels[:prompt_length] == [-100] * prompt_length
    assert labels[sum(writer.masks[0]) - 1] == writer.tokenizer.eos_token_id
    assert writer.row_categories == [CATEGORY_TO_ID["factual_qa"]]


def test_v16_tokenized_natural_row_has_completion_and_eos(tmp_path):
    writer = V16Writer(tmp_path, TinyTokenizer(), max_length=16, reference=set())
    assert writer.add_ids([2, 3, 4, 5], [6, 7, 8], "natural_lm")
    labels = writer.labels[0]
    assert labels[:4] == [-100] * 4
    assert labels[4:8] == [6, 7, 8, 1]
    assert writer.row_categories == [CATEGORY_TO_ID["natural_lm"]]


def test_profile_indices_are_exact_and_deterministic():
    category_ids = torch.tensor(
        [CATEGORY_TO_ID[name] for name in CATEGORY_NAMES for _ in range(10)], dtype=torch.int16
    )
    profile = {
        "counts": {"factual_qa": 4, "retrieval": 3, "natural_lm": 2},
        "prompt_lm_weight": 0.15,
    }
    first = profile_indices(category_ids, profile, seed=17)
    second = profile_indices(category_ids, profile, seed=17)
    assert torch.equal(first, second)
    selected = category_ids.index_select(0, first)
    assert len(first) == 9
    assert int((selected == CATEGORY_TO_ID["factual_qa"]).sum()) == 4
    assert int((selected == CATEGORY_TO_ID["retrieval"]).sum()) == 3
    assert int((selected == CATEGORY_TO_ID["natural_lm"]).sum()) == 2


def test_v16_rejects_d_drive():
    with pytest.raises(ValueError, match="only C"):
        require_c_path(Path(r"D:\unstable\v16"))


def test_v16_output_guard_allows_only_launcher_bootstrap(tmp_path):
    for name in (
        "active_logs.json",
        "train.pid",
        "train.20260812_220000.stdout.log",
        "train.20260812_220000.stderr.log",
    ):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert unexpected_output_files(tmp_path) == []
    (tmp_path / "latest.pt").write_bytes(b"stale")
    assert unexpected_output_files(tmp_path) == ["latest.pt"]


def test_v16_resume_requires_paused_matching_identity(tmp_path):
    output_dir = tmp_path / "run"
    dataset_dir = tmp_path / "data"
    selection = tmp_path / "selection.json"
    checkpoint = {
        "stage_status": "PAUSED",
        "optimizer_step": 125,
        "sample_cursor": 1000,
        "train_config": {
            "stage": STAGE,
            "output_dir": str(output_dir),
            "dataset_dir": str(dataset_dir),
            "selection_artifact": str(selection),
            "selected_profile": "balanced",
        },
    }
    validate_resume_checkpoint(checkpoint, output_dir, dataset_dir, selection, "balanced")
    checkpoint["stage_status"] = "COMPLETE"
    with pytest.raises(ValueError, match="PAUSED"):
        validate_resume_checkpoint(checkpoint, output_dir, dataset_dir, selection, "balanced")
