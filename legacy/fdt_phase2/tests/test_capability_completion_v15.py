from pathlib import Path

import pytest

from scripts.prepare_capability_completion_v15 import CompletionWriter, require_c_path, token_hash
from scripts.train_capability_completion_v15 import (
    unexpected_output_files,
    validate_resume_checkpoint,
    verify_sha256,
)


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [2 + (ord(character) % 29) for character in text]


def test_completion_labels_mask_prompt_and_supervise_answer_eos(tmp_path):
    tokenizer = TinyTokenizer()
    writer = CompletionWriter(tmp_path, tokenizer, max_length=64, reference=set())
    assert writer.add("Question:", "answer", "factual_qa", max_answer_tokens=16)
    prompt_length = len(tokenizer.encode("Question:"))
    answer = tokenizer.encode(" answer") + [tokenizer.eos_token_id]
    labels = writer.labels[0]
    mask = writer.masks[0]
    assert labels[:prompt_length] == [-100] * prompt_length
    assert labels[prompt_length : prompt_length + len(answer)] == answer
    assert labels[prompt_length + len(answer) :] == [-100] * (64 - prompt_length - len(answer))
    assert labels[sum(mask) - 1] == tokenizer.eos_token_id


def test_completion_writer_rejects_prior_and_local_exact_rows(tmp_path):
    tokenizer = TinyTokenizer()
    prompt = "Record:"
    answer = "ZX123456"
    active = tokenizer.encode(prompt) + tokenizer.encode(" " + answer) + [tokenizer.eos_token_id]
    writer = CompletionWriter(tmp_path, tokenizer, max_length=64, reference={token_hash(active)})
    assert not writer.add(prompt, answer, "retrieval", max_answer_tokens=16)
    writer.reference.clear()
    assert writer.add(prompt, answer, "retrieval", max_answer_tokens=16)
    assert not writer.add(prompt, answer, "retrieval", max_answer_tokens=16)


def test_v15_paths_reject_d_drive():
    with pytest.raises(ValueError, match="only C"):
        require_c_path(Path(r"D:\unstable\v15"))


def test_checkpoint_identity_verification(tmp_path):
    path = tmp_path / "checkpoint.pt"
    path.write_bytes(b"immutable-parent")
    expected = "06270d2aa072ed532d899f73b97caed39cfc96bac157dd5858cfd3c73baaed8b"
    assert verify_sha256(path, expected, "Parent") == expected
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_sha256(path, "0" * 64, "Parent")


def test_dedicated_output_allows_only_launcher_bootstrap_files(tmp_path):
    for name in (
        "active_logs.json",
        "train.pid",
        "train.20260812_190000.stdout.log",
        "train.20260812_190000.stderr.log",
    ):
        (tmp_path / name).write_text("", encoding="utf-8")
    assert unexpected_output_files(tmp_path) == []
    (tmp_path / "latest.pt").write_bytes(b"stale")
    assert unexpected_output_files(tmp_path) == ["latest.pt"]


def test_resume_checkpoint_requires_paused_matching_stage_and_paths(tmp_path):
    output_dir = tmp_path / "run"
    dataset_dir = tmp_path / "data"
    checkpoint = {
        "stage_status": "PAUSED",
        "optimizer_step": 219,
        "sample_cursor": 1752,
        "train_config": {
            "stage": "capability_completion_v15",
            "output_dir": str(output_dir),
            "dataset_dir": str(dataset_dir),
        },
    }
    validate_resume_checkpoint(checkpoint, output_dir, dataset_dir)
    checkpoint["stage_status"] = "COMPLETE"
    with pytest.raises(ValueError, match="PAUSED"):
        validate_resume_checkpoint(checkpoint, output_dir, dataset_dir)
