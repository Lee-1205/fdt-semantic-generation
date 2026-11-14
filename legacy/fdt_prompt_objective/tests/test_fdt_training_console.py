import json
import os
from pathlib import Path

import pytest
import torch

os.environ["FDT_CONSOLE_NO_MANAGER"] = "1"

from apps.fdt_training_console.server import (  # noqa: E402
    FDTTrainingManager,
    atomic_write_json,
    build_evaluation_prompt,
    build_report,
    c_only,
    inspect_checkpoint,
    terminal_training_event,
    validate_dataset,
    validate_model_test_request,
    within,
)
from scripts.test_fdt_checkpoints import fidelity_metrics  # noqa: E402


def checkpoint_payload(status="PAUSED", optimizer=True):
    payload = {
        "model_state_dict": {"weight": torch.ones(2, 2)},
        "model_config": {"model_type": "fdt_v3"},
        "train_config": {
            "stage": "capability_completion_v20_balanced_scale_tranche_2",
            "dataset_dir": r"C:\dataset",
            "manifest_sha256": "ABCD",
        },
        "optimizer_step": 123,
        "sample_cursor": 1230,
        "tokens_seen": 9000,
        "stage_status": status,
        "optimizer_state_saved": optimizer,
    }
    if optimizer:
        payload["optimizer_state_dict"] = {"state": {}, "param_groups": []}
    return payload


def test_c_only_rejects_d_drive():
    with pytest.raises(ValueError, match="C: only"):
        c_only(Path(r"D:\forbidden"), "test")


def test_output_must_remain_under_runs(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    assert within(root / "fresh", root, "output") == root / "fresh"
    with pytest.raises(ValueError, match="must stay under"):
        within(tmp_path / "outside", root, "output")


def test_atomic_json_has_no_temporary_residue(tmp_path):
    path = tmp_path / "state.json"
    atomic_write_json(path, {"status": "READY"})
    assert json.loads(path.read_text()) == {"status": "READY"}
    assert not list(tmp_path.glob("*.tmp"))


def test_paused_checkpoint_requires_optimizer_and_verifies(tmp_path):
    path = tmp_path / "latest.pt"
    torch.save(checkpoint_payload(), path)
    result = inspect_checkpoint(path)
    assert result["stage_status"] == "PAUSED"
    assert result["optimizer_state_saved"] is True
    assert result["optimizer_step"] == 123
    assert len(result["sha256"]) == 64


def test_paused_checkpoint_without_optimizer_is_rejected(tmp_path):
    path = tmp_path / "latest.pt"
    torch.save(checkpoint_payload(status="PAUSED", optimizer=False), path)
    with pytest.raises(RuntimeError, match="not exactly resumable"):
        inspect_checkpoint(path)


def test_complete_checkpoint_is_model_only(tmp_path):
    path = tmp_path / "latest.pt"
    torch.save(checkpoint_payload(status="COMPLETE", optimizer=False), path)
    result = inspect_checkpoint(path)
    assert result["stage_status"] == "COMPLETE"
    assert result["optimizer_state_saved"] is False


def test_dataset_contract_requires_five_domains_and_zero_overlap(tmp_path):
    dataset = tmp_path / "dataset"
    train = dataset / "shards" / "train" / "shard_000000.pt"
    validation = dataset / "shards" / "validation" / "shard_000000.pt"
    train.parent.mkdir(parents=True)
    validation.parent.mkdir(parents=True)
    train.write_bytes(b"train")
    validation.write_bytes(b"validation")
    manifest = {
        "post_build_exact_overlap": 0,
        "category_objective_weights": {
            "factual_qa": 0.15,
            "retrieval": 0.25,
            "python_code": 0.15,
            "json": 0.20,
            "natural_lm": 0.25,
        },
        "category_grounding_loss_weights": {"factual_qa": 1.25, "retrieval": 2.25, "json": 2.25},
    }
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_dataset(dataset)
    assert result["manifest_sha256"]
    manifest["post_build_exact_overlap"] = 1
    (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="exact-overlap"):
        validate_dataset(dataset)


def test_terminal_event_and_report_are_derived_from_training_log(tmp_path):
    log = tmp_path / "training_log.jsonl"
    rows = [
        {"event": "train", "step": 25, "loss": 3.0, "category_losses": {"factual_qa": 4.0}, "grad_norm": 1.0, "tokens_per_sec": 700},
        {"event": "train", "step": 50, "loss": 2.5, "category_losses": {"factual_qa": 3.5}, "grad_norm": 1.2, "tokens_per_sec": 710},
        {"event": "paused", "status": "PAUSED", "step": 50, "tokens_seen": 1000},
    ]
    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert terminal_training_event(log)["status"] == "PAUSED"
    metrics, report = build_report(tmp_path, None)
    assert metrics.exists() and report.exists()
    assert "2.5" in report.read_text(encoding="utf-8")
    prompt = tmp_path / "analysis" / "EVALUATION_PROMPT.md"
    assert prompt.exists()
    assert "exact-copy/retrieval probe" in prompt.read_text(encoding="utf-8")


def test_evaluation_prompt_marks_weight_only_abort_as_not_resumable(tmp_path):
    path = build_evaluation_prompt(
        tmp_path,
        {"path": "latest.pt", "sha256": "A" * 64, "stage_status": "ABORTED_SMOKE_GATE", "optimizer_state_saved": False},
        {"terminal": {"status": "ABORTED_SMOKE_GATE"}, "smoke_gates": [], "category_loss": {}},
        None,
    )
    text = path.read_text(encoding="utf-8")
    assert "not an exact continuation point" in text
    assert "4/8/16/32/64-character" in text


def test_model_test_request_and_fidelity_metrics(tmp_path):
    checkpoint = tmp_path / "runs" / "candidate" / "latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    from apps.fdt_training_console import server
    original_root = server.ROOT
    server.ROOT = tmp_path
    try:
        payload = validate_model_test_request({
            "checkpoints": [str(checkpoint)],
            "prompt": "Repeat this",
            "expected": "ABC123",
            "mode": "greedy",
        })
        assert payload["expected"] == "ABC123"
        with pytest.raises(ValueError, match="duplicate"):
            validate_model_test_request({"checkpoints": [str(checkpoint), str(checkpoint)], "prompt": "x"})
    finally:
        server.ROOT = original_root

    class Tokenizer:
        @staticmethod
        def encode(value, add_special_tokens=False):
            return list(value)

    fidelity = fidelity_metrics("ABC123", "ABC124", Tokenizer())
    assert fidelity["exact_match"] is False
    assert fidelity["first_character_divergence"] == 5
    assert fidelity["character_accuracy"] == pytest.approx(5 / 6)


def test_paused_result_prepares_fresh_resume_defaults(tmp_path):
    checkpoint = tmp_path / "paused_run" / "latest.pt"
    checkpoint.parent.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manager = object.__new__(FDTTrainingManager)
    manager.state = {
        "defaults": {
            "parent_checkpoint": str(tmp_path / "old.pt"),
            "output_dir": str(tmp_path / "used_output"),
            "evaluation_mode": "smoke",
        },
        "launch": {"evaluation_mode": "standard"},
    }
    changed = manager._prepare_resume_defaults({
        "path": str(checkpoint),
        "dataset_dir": str(dataset),
        "stage_status": "PAUSED",
        "optimizer_state_saved": True,
    })
    assert changed is True
    assert Path(manager.state["defaults"]["parent_checkpoint"]) == checkpoint
    assert Path(manager.state["defaults"]["output_dir"]).parent.name == "runs"
    assert manager.state["defaults"]["evaluation_mode"] == "standard"
