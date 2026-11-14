import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "prepare_pointer_gate_calibration.py"
    spec = importlib.util.spec_from_file_location("prepare_pointer_gate_calibration", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_negative_labels_only_supervise_the_active_tail():
    module = load_module()
    ids = torch.tensor([[1, 2, 3, 4, 5, 0], [7, 8, 9, 10, 11, 12]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]])

    labels = module.negative_labels(ids, mask, tokens=2)

    assert labels.tolist() == [
        [-100, -100, -100, 4, 5, -100],
        [-100, -100, -100, -100, 11, 12],
    ]


def test_combined_payload_uses_compact_storage_types():
    module = load_module()
    positive = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
        "labels": torch.tensor([[-100, 2]], dtype=torch.long),
        "attention_mask": torch.ones(1, 2, dtype=torch.long),
    }
    negative = {
        "input_ids": torch.tensor([[3, 4]], dtype=torch.int32),
        "labels": torch.tensor([[-100, 4]], dtype=torch.int32),
        "attention_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    combined = module.combine_and_shuffle(positive, negative, seed=1)

    assert combined["input_ids"].dtype == torch.int32
    assert combined["labels"].dtype == torch.int32
    assert combined["attention_mask"].dtype == torch.bool
