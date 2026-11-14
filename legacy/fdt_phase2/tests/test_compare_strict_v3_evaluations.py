from scripts.compare_strict_v3_evaluations import (
    mean_interval,
    not_credibly_higher,
    not_credibly_lower,
    paired_binary_outcome,
    paired_category,
    protocol_compatibility,
)
import pytest


def test_mean_interval_is_deterministic() -> None:
    values = [-1.0, -0.5, 0.0, 0.5]
    assert mean_interval(values, 200, 7) == mean_interval(values, 200, 7)


def test_paired_category_uses_per_row_metrics() -> None:
    baseline = [
        {"category": "json", "row_id": 0, "tokens": 2, "loss_sum": 4.0, "top1": 1},
        {"category": "json", "row_id": 1, "tokens": 4, "loss_sum": 8.0, "top1": 2},
    ]
    candidate = [
        {"category": "json", "row_id": 0, "tokens": 2, "loss_sum": 2.0, "top1": 2},
        {"category": "json", "row_id": 1, "tokens": 4, "loss_sum": 4.0, "top1": 4},
    ]
    result = paired_category(baseline, candidate, samples=200, seed=3)
    assert result["rows"] == 2
    assert result["mean_row_loss_delta"] == -1.0
    assert result["mean_row_top1_delta"] == 0.5


def test_credible_direction_helpers_reject_one_sided_regressions() -> None:
    assert not_credibly_higher([-0.02, 0.01])
    assert not not_credibly_higher([0.01, 0.03])
    assert not_credibly_lower([-0.01, 0.02])
    assert not not_credibly_lower([-0.03, -0.01])


def test_paired_binary_outcome_counts_gains_and_losses() -> None:
    baseline = [
        {"bucket": "a", "row_id": 0, "loop_free": True},
        {"bucket": "a", "row_id": 1, "loop_free": True},
        {"bucket": "b", "row_id": 0, "loop_free": False},
        {"bucket": "b", "row_id": 1, "loop_free": False},
    ]
    candidate = [
        {"bucket": "a", "row_id": 0, "loop_free": False},
        {"bucket": "a", "row_id": 1, "loop_free": True},
        {"bucket": "b", "row_id": 0, "loop_free": True},
        {"bucket": "b", "row_id": 1, "loop_free": False},
    ]
    result = paired_binary_outcome(baseline, candidate, "loop_free")
    assert result["rows"] == 4
    assert result["baseline_rate"] == result["candidate_rate"] == 0.5
    assert result["delta"] == 0.0
    assert result["gains"] == result["losses"] == 1
    assert result["unchanged"] == 2
    assert result["delta_bootstrap_95"][0] <= 0.0 <= result["delta_bootstrap_95"][1]


def test_pairing_rejects_missing_or_duplicate_rows() -> None:
    baseline = [{"category": "json", "row_id": 0, "tokens": 1, "loss_sum": 1.0, "top1": 1}]
    with pytest.raises(ValueError, match="Unpaired"):
        paired_category(baseline, [], samples=10, seed=1)
    with pytest.raises(ValueError, match="Duplicate"):
        paired_category(baseline * 2, baseline, samples=10, seed=1)


def test_protocol_requires_audited_matching_fingerprints() -> None:
    payload = {
        "evaluator": {
            "version": "strict_base_v3_audited_1",
            "script_sha256": "script",
            "mode": "standard",
            "settings": {"rows": 1},
        },
        "metadata": {
            "checkpoint_sha256": "checkpoint",
            "tokenizer": "tokenizer",
            "parameters": 10,
        },
        "datasets": {
            "broad_validation_fingerprint": {"aggregate_sha256": "broad"},
            "capability_validation_fingerprint": {"aggregate_sha256": "capability"},
            "unbiased_sealed_final_test": False,
        },
    }
    result = protocol_compatibility(payload, payload)
    assert result["compatible"]
    assert not result["sealed_final_test"]

    payload["evaluator"]["version"] = "strict_base_v3_audited_2"
    repaired = protocol_compatibility(payload, payload)
    assert repaired["compatible"]
