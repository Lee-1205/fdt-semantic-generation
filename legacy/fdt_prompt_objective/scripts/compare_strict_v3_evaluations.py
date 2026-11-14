from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


CATEGORIES = ("factual_qa", "retrieval", "python_code", "json")
BUCKETS = ("fineweb_edu_dedup", "cosmopedia", "fineweb", "tinystories", "code_python")
PREFIX_LOOP_FREE_NONINFERIORITY_MARGIN = 0.05
PAIRED_ROW_LOSS_NONINFERIORITY_MARGIN = 0.01
PAIRED_ROW_TOP1_NONINFERIORITY_MARGIN = 0.005


def unique_row_map(rows: list[dict], fields: tuple[str, ...], label: str) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        if key in result:
            raise ValueError(f"Duplicate {label} row key: {key}")
        result[key] = row
    return result


def require_same_keys(base: dict, candidate: dict, label: str) -> list[tuple]:
    if set(base) != set(candidate):
        missing = sorted(set(base) - set(candidate))[:10]
        extra = sorted(set(candidate) - set(base))[:10]
        raise ValueError(f"Unpaired {label} rows; missing={missing}, extra={extra}")
    if not base:
        raise ValueError(f"No paired {label} rows")
    return sorted(base)


def mean_interval(values: list[float], samples: int, seed: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    count = len(values)
    means = sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count for _ in range(samples)
    )
    return [means[int(0.025 * (samples - 1))], means[int(0.975 * (samples - 1))]]


def paired_category(base_rows: list[dict], candidate_rows: list[dict], samples: int, seed: int) -> dict:
    base = unique_row_map(base_rows, ("category", "row_id"), "capability")
    candidate = unique_row_map(candidate_rows, ("category", "row_id"), "capability")
    keys = require_same_keys(base, candidate, "capability")
    loss_deltas = []
    top1_deltas = []
    for key in keys:
        old = base[key]
        new = candidate[key]
        if old["tokens"] != new["tokens"]:
            raise ValueError(f"Token count changed for paired row {key}")
        tokens = old["tokens"]
        loss_deltas.append(new["loss_sum"] / tokens - old["loss_sum"] / tokens)
        top1_deltas.append(new["top1"] / tokens - old["top1"] / tokens)
    return {
        "rows": len(keys),
        "mean_row_loss_delta": sum(loss_deltas) / len(loss_deltas),
        "mean_row_loss_delta_bootstrap_95": mean_interval(loss_deltas, samples, seed),
        "mean_row_top1_delta": sum(top1_deltas) / len(top1_deltas),
        "mean_row_top1_delta_bootstrap_95": mean_interval(top1_deltas, samples, seed + 1),
    }


def paired_binary_outcome(
    base_rows: list[dict],
    candidate_rows: list[dict],
    field: str,
    key_fields: tuple[str, ...] = ("bucket", "row_id"),
    samples: int = 20_000,
    seed: int = 20260810,
) -> dict:
    base = unique_row_map(base_rows, key_fields, field)
    candidate = unique_row_map(candidate_rows, key_fields, field)
    keys = require_same_keys(base, candidate, field)
    old_values = [bool(base[key][field]) for key in keys]
    new_values = [bool(candidate[key][field]) for key in keys]
    gains = sum((not old) and new for old, new in zip(old_values, new_values))
    losses = sum(old and (not new) for old, new in zip(old_values, new_values))
    baseline_rate = sum(old_values) / len(keys)
    candidate_rate = sum(new_values) / len(keys)
    deltas = [int(new) - int(old) for old, new in zip(old_values, new_values)]
    return {
        "rows": len(keys),
        "baseline_rate": baseline_rate,
        "candidate_rate": candidate_rate,
        "delta": candidate_rate - baseline_rate,
        "gains": gains,
        "losses": losses,
        "unchanged": len(keys) - gains - losses,
        "delta_bootstrap_95": mean_interval(deltas, samples, seed),
    }


def protocol_compatibility(base: dict, candidate: dict) -> dict:
    old_eval = base.get("evaluator", {})
    new_eval = candidate.get("evaluator", {})
    old_data = base.get("datasets", {})
    new_data = candidate.get("datasets", {})
    checks = {
        "audited_evaluator": old_eval.get("version")
        in {"strict_base_v3_audited_1", "strict_base_v3_audited_2"}
        and new_eval.get("version")
        in {"strict_base_v3_audited_1", "strict_base_v3_audited_2"},
        "same_evaluator_script": bool(old_eval.get("script_sha256"))
        and old_eval.get("script_sha256") == new_eval.get("script_sha256"),
        "same_mode": old_eval.get("mode") == new_eval.get("mode"),
        "same_settings": old_eval.get("settings") == new_eval.get("settings"),
        "same_broad_validation": old_data.get("broad_validation_fingerprint", {}).get(
            "aggregate_sha256"
        )
        == new_data.get("broad_validation_fingerprint", {}).get("aggregate_sha256")
        and bool(old_data.get("broad_validation_fingerprint", {}).get("aggregate_sha256")),
        "same_capability_validation": old_data.get("capability_validation_fingerprint", {}).get(
            "aggregate_sha256"
        )
        == new_data.get("capability_validation_fingerprint", {}).get("aggregate_sha256")
        and bool(old_data.get("capability_validation_fingerprint", {}).get("aggregate_sha256")),
        "same_tokenizer": base.get("metadata", {}).get("tokenizer")
        == candidate.get("metadata", {}).get("tokenizer"),
        "same_model_shape": base.get("metadata", {}).get("parameters")
        == candidate.get("metadata", {}).get("parameters"),
        "distinct_checkpoint_hashes_present": bool(
            base.get("metadata", {}).get("checkpoint_sha256")
        )
        and bool(candidate.get("metadata", {}).get("checkpoint_sha256")),
    }
    return {
        "compatible": all(checks.values()),
        "checks": checks,
        "sealed_final_test": bool(old_data.get("unbiased_sealed_final_test"))
        and bool(new_data.get("unbiased_sealed_final_test")),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_delta(base: float, candidate: float) -> dict:
    return {
        "baseline": base,
        "candidate": candidate,
        "delta": candidate - base,
        "relative_delta": candidate / base - 1.0 if base else None,
    }


def not_credibly_higher(interval: list[float]) -> bool:
    return interval[0] <= 0.0


def not_credibly_lower(interval: list[float]) -> bool:
    return interval[1] >= 0.0


def compare(base: dict, candidate: dict, samples: int, seed: int) -> dict:
    protocol = protocol_compatibility(base, candidate)
    old = base["evaluation"]
    new = candidate["evaluation"]
    old_heldout = old["heldout"]
    new_heldout = new["heldout"]
    old_tf = old["capability_teacher_forced"]
    new_tf = new["capability_teacher_forced"]
    paired = {}
    for offset, category in enumerate(CATEGORIES):
        old_rows = [row for row in old_tf["rows"] if row["category"] == category]
        new_rows = [row for row in new_tf["rows"] if row["category"] == category]
        paired[category] = paired_category(old_rows, new_rows, samples, seed + offset * 10)

    bucket_deltas = {
        bucket: {
            "loss": metric_delta(old_heldout[bucket]["loss"], new_heldout[bucket]["loss"]),
            "top1": metric_delta(
                old_heldout[bucket]["top1_accuracy"], new_heldout[bucket]["top1_accuracy"]
            ),
        }
        for bucket in BUCKETS
    }
    old_gen = old["capability_generation"]
    new_gen = new["capability_generation"]
    old_retrieval = old["retrieval_base_continuation"]
    new_retrieval = new["retrieval_base_continuation"]
    old_prefix = old["heldout_prefix_generation"]["overall"]
    new_prefix = new["heldout_prefix_generation"]["overall"]
    paired_prefix = paired_binary_outcome(
        old["heldout_prefix_generation"]["rows"],
        new["heldout_prefix_generation"]["rows"],
        "loop_free",
        samples=samples,
        seed=seed + 100,
    )
    generation_pairs = {}
    for offset, category in enumerate(CATEGORIES):
        fields = ["correct"]
        if category == "python_code":
            fields.extend(["compilable", "usable_structure"])
        if category == "json":
            fields.append("valid")
        generation_pairs[category] = {
            field: paired_binary_outcome(
                old_gen[category]["rows"],
                new_gen[category]["rows"],
                field,
                key_fields=("row_id",),
                samples=samples,
                seed=seed + 200 + offset * 10 + len(field),
            )
            for field in fields
        }
    retrieval_rank_pair = paired_binary_outcome(
        old_retrieval["rows"],
        new_retrieval["rows"],
        "rank_correct",
        key_fields=("distance", "code"),
        samples=samples,
        seed=seed + 300,
    )
    retrieval_exact_pair = paired_binary_outcome(
        old_retrieval["rows"],
        new_retrieval["rows"],
        "greedy_exact",
        key_fields=("distance", "code"),
        samples=samples,
        seed=seed + 301,
    )

    checks = {
        "broad_loss_non_regression": new_heldout["overall"]["loss"]
        <= old_heldout["overall"]["loss"] * 1.003,
        "python_loss_non_regression": new_tf["by_category"]["python_code"]["loss"]
        <= old_tf["by_category"]["python_code"]["loss"] * 1.005,
        "python_loss_not_credibly_higher": not_credibly_higher(
            paired["python_code"]["mean_row_loss_delta_bootstrap_95"]
        ),
        "python_top1_not_credibly_lower": paired["python_code"][
            "mean_row_top1_delta_bootstrap_95"
        ][1]
        >= 0.0,
        "factual_loss_not_credibly_higher": not_credibly_higher(
            paired["factual_qa"]["mean_row_loss_delta_bootstrap_95"]
        ),
        "factual_top1_not_credibly_lower": not_credibly_lower(
            paired["factual_qa"]["mean_row_top1_delta_bootstrap_95"]
        ),
        "retrieval_loss_not_credibly_higher": not_credibly_higher(
            paired["retrieval"]["mean_row_loss_delta_bootstrap_95"]
        ),
        "retrieval_top1_not_credibly_lower": not_credibly_lower(
            paired["retrieval"]["mean_row_top1_delta_bootstrap_95"]
        ),
        "json_top1_non_regression": new_tf["by_category"]["json"]["top1"]["rate"]
        >= old_tf["by_category"]["json"]["top1"]["rate"] - 0.005,
        "json_validity_preserved": new_gen["json"]["valid"]["rate"]
        >= old_gen["json"]["valid"]["rate"],
        "distance_1400_retrieval_non_regression": new_retrieval["by_distance"]["1400"][
            "rank_accuracy"
        ]
        >= old_retrieval["by_distance"]["1400"]["rank_accuracy"] - 0.025,
        "routing_healthy": new_heldout["overall"]["routing"]["dead_anchor_ratio"] <= 0.01,
        "prefix_generation_not_materially_worse": paired_prefix["delta"]
        >= -PREFIX_LOOP_FREE_NONINFERIORITY_MARGIN,
        "prefix_mean_repetition_non_regression": new_prefix["mean_repetition_rate"]
        <= old_prefix["mean_repetition_rate"] + 0.01,
        "factual_generation_non_regression": generation_pairs["factual_qa"]["correct"][
            "candidate_rate"
        ]
        >= generation_pairs["factual_qa"]["correct"]["baseline_rate"],
        "retrieval_generation_non_regression": generation_pairs["retrieval"]["correct"][
            "candidate_rate"
        ]
        >= generation_pairs["retrieval"]["correct"]["baseline_rate"],
        "python_compilation_non_regression": generation_pairs["python_code"]["compilable"][
            "candidate_rate"
        ]
        >= generation_pairs["python_code"]["compilable"]["baseline_rate"],
        "python_structure_non_regression": generation_pairs["python_code"]["usable_structure"][
            "candidate_rate"
        ]
        >= generation_pairs["python_code"]["usable_structure"]["baseline_rate"],
        "json_semantic_exact_non_regression": generation_pairs["json"]["correct"][
            "candidate_rate"
        ]
        >= generation_pairs["json"]["correct"]["baseline_rate"],
        "retrieval_rank_paired_non_regression": retrieval_rank_pair["delta"] >= -0.025,
        "retrieval_exact_paired_non_regression": retrieval_exact_pair["candidate_rate"]
        >= retrieval_exact_pair["baseline_rate"],
    }
    evidence_checks = {
        "protocol_compatible_and_audited": protocol["compatible"],
        "prefix_loop_free_confidently_noninferior": paired_prefix["delta_bootstrap_95"][0]
        >= -PREFIX_LOOP_FREE_NONINFERIORITY_MARGIN,
        "retrieval_rank_confidently_noninferior": retrieval_rank_pair["delta_bootstrap_95"][0]
        >= -0.025,
    }
    for category in ("factual_qa", "retrieval", "python_code", "json"):
        evidence_checks[f"{category}_loss_confidently_noninferior"] = paired[category][
            "mean_row_loss_delta_bootstrap_95"
        ][1] <= PAIRED_ROW_LOSS_NONINFERIORITY_MARGIN
        evidence_checks[f"{category}_top1_confidently_noninferior"] = paired[category][
            "mean_row_top1_delta_bootstrap_95"
        ][0] >= -PAIRED_ROW_TOP1_NONINFERIORITY_MARGIN
    legacy_gate_status = "PASS" if all(checks.values()) else "FAIL"
    promotion_status = (
        "REQUIRES_SEALED_HOLDOUT_CONFIRMATION"
        if legacy_gate_status == "PASS" and all(evidence_checks.values())
        else "PROMOTION_BLOCKED"
    )
    return {
        "status": promotion_status,
        "legacy_gate_status": legacy_gate_status,
        "protocol": protocol,
        "baseline": {
            "path": str(base["metadata"]["checkpoint"]),
            "tokens_seen": base["metadata"]["tokens_seen"],
        },
        "candidate": {
            "path": str(candidate["metadata"]["checkpoint"]),
            "tokens_seen": candidate["metadata"]["tokens_seen"],
        },
        "broad": {
            "loss": metric_delta(
                old_heldout["overall"]["loss"], new_heldout["overall"]["loss"]
            ),
            "top1": metric_delta(
                old_heldout["overall"]["top1_accuracy"],
                new_heldout["overall"]["top1_accuracy"],
            ),
            "buckets": bucket_deltas,
        },
        "paired_capability": paired,
        "generation": {
            "json_valid_rate": metric_delta(
                old_gen["json"]["valid"]["rate"], new_gen["json"]["valid"]["rate"]
            ),
            "json_exact_rate": metric_delta(
                old_gen["json"]["correct"]["rate"], new_gen["json"]["correct"]["rate"]
            ),
            "prefix_loop_free_rate": metric_delta(
                old_prefix["loop_free"]["rate"], new_prefix["loop_free"]["rate"]
            ),
            "prefix_mean_repetition_rate": metric_delta(
                old_prefix["mean_repetition_rate"], new_prefix["mean_repetition_rate"]
            ),
        },
        "paired_prefix_generation": {
            "loop_free": paired_prefix,
            "noninferiority_margin": PREFIX_LOOP_FREE_NONINFERIORITY_MARGIN,
        },
        "paired_capability_generation": generation_pairs,
        "paired_retrieval": {
            "rank": retrieval_rank_pair,
            "greedy_exact": retrieval_exact_pair,
        },
        "retrieval_rank": {
            "overall": metric_delta(old_retrieval["rank"]["rate"], new_retrieval["rank"]["rate"]),
            "by_distance": {
                distance: metric_delta(
                    old_retrieval["by_distance"][distance]["rank_accuracy"],
                    new_retrieval["by_distance"][distance]["rank_accuracy"],
                )
                for distance in ("64", "512", "1400")
            },
        },
        "checks": checks,
        "evidence_checks": evidence_checks,
        "confidence_aware_noninferiority_margins": {
            "paired_row_loss": PAIRED_ROW_LOSS_NONINFERIORITY_MARGIN,
            "paired_row_top1": PAIRED_ROW_TOP1_NONINFERIORITY_MARGIN,
            "prefix_loop_free": PREFIX_LOOP_FREE_NONINFERIORITY_MARGIN,
            "retrieval_rank": 0.025,
        },
        "interpretation": {
            "python_correct_metric": (
                "structural usability, not semantic or functional exactness"
            ),
            "fixed_capability_split": (
                "reused development gate; an independent sealed holdout is required for promotion"
            ),
            "non_significant_regression": (
                "absence of evidence is not proof of equivalence; confidence bounds are reported separately"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired strict-v3 FDT evaluations.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = compare(baseline, candidate, args.samples, args.seed)
    report["comparator"] = {
        "version": "strict_v3_paired_audited_1",
        "script_sha256": sha256(Path(__file__)),
    }
    report["inputs"] = {
        "baseline": str(args.baseline.resolve()),
        "baseline_sha256": sha256(args.baseline),
        "candidate": str(args.candidate.resolve()),
        "candidate_sha256": sha256(args.candidate),
        "bootstrap_samples": args.samples,
        "seed": args.seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
