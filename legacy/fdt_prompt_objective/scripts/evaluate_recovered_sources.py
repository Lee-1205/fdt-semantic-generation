from __future__ import annotations

import json
from pathlib import Path

import torch

from forensic_staged_regression import evaluate_checkpoint, labels_for_split, load_split


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dataset = ROOT / "prepared_data" / "fdt_rlm_100m"
    ids, mask = load_split(dataset, "validation")
    labels = labels_for_split(dataset, "validation", ids.size(0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {}
    for model in ("transformer", "fdt"):
        checkpoint = ROOT / "runs" / "staged_100m_interleaved" / model / "latest.pt"
        models[model] = evaluate_checkpoint(checkpoint, ids, mask, labels, device)
    output = {
        "validation_dataset": str(dataset.relative_to(ROOT)),
        "sequences": int(ids.size(0)),
        "models": models,
    }
    path = ROOT / "artifacts" / "recovered_100m_source_specific_eval.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({model: result["losses"] for model, result in models.items()}, indent=2))


if __name__ == "__main__":
    main()
