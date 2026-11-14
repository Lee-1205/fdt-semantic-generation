from __future__ import annotations

import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "artifacts" / "rlm_uncertainty_dataset.jsonl"
OUTPUT = ROOT / "artifacts" / "rlm_uncertainty_analysis.json"
CONVENTIONAL = ["token_entropy", "mean_logprob", "confidence"]
FDT_SIGNALS = [
    "anchor_entropy",
    "normalized_anchor_entropy",
    "effective_anchor_count",
    "top1_membership",
    "top1_top2_margin",
    "state_conflict",
]


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    positives = int(labels.sum().item())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float32)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float32)
    positive_rank_sum = ranks[labels.bool()].sum().item()
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def pr_auc(scores: torch.Tensor, labels: torch.Tensor) -> float | None:
    positives = int(labels.sum().item())
    if positives == 0:
        return None
    order = torch.argsort(scores, descending=True)
    y = labels[order].float()
    tp = y.cumsum(0)
    precision = tp / torch.arange(1, len(y) + 1, dtype=torch.float32)
    return float((precision * y).sum().item() / positives)


def ece(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (probabilities >= low) & (probabilities < high if index < bins - 1 else probabilities <= high)
        if mask.any():
            value += float(mask.float().mean().item()) * abs(
                float(probabilities[mask].mean().item()) - float(labels[mask].float().mean().item())
            )
    return value


def fit_logistic(x: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor):
    mean = x[train_mask].mean(0)
    std = x[train_mask].std(0).clamp_min(1e-5)
    z = (x - mean) / std
    weight = torch.zeros(x.size(1), requires_grad=True)
    bias = torch.zeros((), requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=80, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = z[train_mask] @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y[train_mask].float())
        loss = loss + 1e-3 * weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return torch.sigmoid(z @ weight.detach() + bias.detach()), weight.detach(), bias.detach()


def evaluate(name: str, features: list[str], rows, labels, train_mask, test_mask):
    x = torch.tensor([[float(row.get("signals", {}).get(key, 0.0)) for key in features] for row in rows])
    probabilities, weight, bias = fit_logistic(x, labels, train_mask)
    test_prob, test_labels = probabilities[test_mask], labels[test_mask]
    return {
        "name": name,
        "features": features,
        "test_records": int(test_mask.sum().item()),
        "roc_auc": roc_auc(test_prob, test_labels),
        "pr_auc": pr_auc(test_prob, test_labels),
        "brier": float(((test_prob - test_labels.float()) ** 2).mean().item()),
        "ece": ece(test_prob, test_labels),
        "weights": {key: float(value) for key, value in zip(features, weight)},
        "bias": float(bias.item()),
    }


def main():
    rows = [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = torch.tensor(
        [bool(row["additional_read_helpful"] or row["recursive_call_helpful"]) for row in rows],
        dtype=torch.long,
    )
    # Split by task id, not by event, to prevent trace leakage.
    train_mask = torch.tensor([sum(row["task_id"].encode("utf-8")) % 5 != 0 for row in rows])
    test_mask = ~train_mask
    payload = {
        "records": len(rows),
        "positive_rate": float(labels.float().mean().item()) if len(labels) else 0.0,
        "split": "deterministic task-level 80/20",
        "models": [],
        "claim_boundary": "Exploratory small-sample calibration; FDT contribution requires held-out improvement.",
    }
    if len(rows) and labels[train_mask].unique().numel() == 2 and test_mask.any():
        payload["models"].append(evaluate("conventional", CONVENTIONAL, rows, labels, train_mask, test_mask))
        payload["models"].append(evaluate("conventional_plus_fdt", CONVENTIONAL + FDT_SIGNALS, rows, labels, train_mask, test_mask))
    else:
        payload["status"] = "insufficient class variation for calibration"
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
