import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_assistant_only_loss_and_prompt_retention_are_finite():
    module = load_script("train_chat_sft", "train_chat_sft.py")
    input_ids = torch.tensor([[0, 1, 2, 3]])
    labels = torch.tensor([[-100, -100, 2, 3]])
    mask = torch.ones_like(input_ids)
    logits = torch.zeros(1, 4, 5)
    logits[0, 0, 1] = 8
    logits[0, 1, 2] = 8
    logits[0, 2, 3] = 8

    assistant, prompt = module.masked_chat_losses(logits, input_ids, labels, mask)

    assert torch.isfinite(assistant)
    assert torch.isfinite(prompt)
    assert assistant.item() < 0.01
    assert prompt.item() < 0.01


def test_language_gate_repetition_metric_detects_loops():
    module = load_script("check_language_gate", "check_language_gate.py")
    assert module.repetition_rate([1, 2, 3, 1, 2, 3]) > 0
    assert module.repetition_rate([1, 2, 3, 4, 5, 6]) == 0


def test_language_gate_uses_latest_validation_not_after_checkpoint(tmp_path):
    module = load_script("check_language_gate_metrics", "check_language_gate.py")
    checkpoint_path = tmp_path / "latest.pt"
    (tmp_path / "onk_runtime.json").write_text(
        json.dumps({"promotion_metrics": {"tokens_seen": 100, "validation_loss": 9.0}}),
        encoding="utf-8",
    )
    rows = (
        {"event": "validation", "tokens_seen": 200, "validation_loss": 5.0},
        {"event": "validation", "tokens_seen": 300, "validation_loss": 4.0},
        {"event": "validation", "tokens_seen": 400, "validation_loss": 3.0},
    )
    (tmp_path / "training_log.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )

    metrics = module.runtime_metrics(checkpoint_path, checkpoint_tokens=350)

    assert metrics["tokens_seen"] == 300
    assert metrics["validation_loss"] == 4.0


def test_language_gate_finds_metrics_for_copied_checkpoint(tmp_path):
    module = load_script("check_language_gate_snapshot", "check_language_gate.py")
    snapshot_dir = tmp_path / "artifacts"
    run_dir = tmp_path / "run"
    snapshot_dir.mkdir()
    run_dir.mkdir()
    (run_dir / "training_log.jsonl").write_text(
        json.dumps({"event": "validation", "tokens_seen": 300, "validation_loss": 4.5}),
        encoding="utf-8",
    )

    metrics = module.runtime_metrics(
        snapshot_dir / "snapshot.pt", checkpoint_tokens=300, metrics_dir=run_dir
    )

    assert metrics["validation_loss"] == 4.5


def test_pilot_quality_profile_is_normalized_and_excludes_tulu():
    module = load_script("prepare_chat_sft", "prepare_chat_sft.py")
    sources = module.TRAIN_SOURCE_PROFILES["pilot_quality"]

    assert abs(sum(source[3] for source in sources) - 1.0) < 1e-9
    assert all("tulu" not in source[0].lower() for source in sources)
