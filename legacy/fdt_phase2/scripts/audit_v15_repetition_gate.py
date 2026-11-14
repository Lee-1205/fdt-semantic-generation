from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[int(quantile * (len(ordered) - 1))]


def paired_bootstrap(deltas: list[float], samples: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    count = len(deltas)
    draws = [
        sum(deltas[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(samples)
    ]
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def exact_sign_test_two_sided(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(0, min(gains, losses) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def paired_rows(payload: dict) -> dict[tuple[str, int], dict]:
    rows = payload["evaluation"]["heldout_prefix_generation"]["rows"]
    return {(row["bucket"], int(row["row_id"])): row for row in rows}


def threshold_report(base: dict, candidate: dict, thresholds: tuple[float, ...]) -> dict:
    report = {}
    for threshold in thresholds:
        old = [row["repetition_rate"] <= threshold for row in base.values()]
        new = [candidate[key]["repetition_rate"] <= threshold for key in base]
        gains = sum((not before) and after for before, after in zip(old, new))
        losses = sum(before and (not after) for before, after in zip(old, new))
        report[str(threshold)] = {
            "baseline_rate": sum(old) / len(old),
            "candidate_rate": sum(new) / len(new),
            "delta": (sum(new) - sum(old)) / len(old),
            "gains": gains,
            "losses": losses,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the V15 strict prefix repetition gate.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate_payload = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline = paired_rows(baseline_payload)
    candidate = paired_rows(candidate_payload)
    keys = sorted(baseline.keys() & candidate.keys())
    if len(keys) != 50:
        raise ValueError(f"Expected 50 paired rows, found {len(keys)}")

    repetition_deltas = [
        candidate[key]["repetition_rate"] - baseline[key]["repetition_rate"] for key in keys
    ]
    binary_deltas = [
        int(candidate[key]["loop_free"]) - int(baseline[key]["loop_free"]) for key in keys
    ]
    gains = sum(delta == 1 for delta in binary_deltas)
    losses = sum(delta == -1 for delta in binary_deltas)
    changed = []
    for key, repetition_delta, binary_delta in zip(keys, repetition_deltas, binary_deltas):
        if binary_delta or abs(repetition_delta) >= 0.05:
            old = baseline[key]
            new = candidate[key]
            changed.append(
                {
                    "bucket": key[0],
                    "row_id": key[1],
                    "baseline_repetition": old["repetition_rate"],
                    "candidate_repetition": new["repetition_rate"],
                    "delta": repetition_delta,
                    "baseline_loop_free": old["loop_free"],
                    "candidate_loop_free": new["loop_free"],
                    "baseline_completion": old["completion"],
                    "candidate_completion": new["completion"],
                }
            )

    mean_delta = sum(repetition_deltas) / len(repetition_deltas)
    loop_delta = sum(binary_deltas) / len(binary_deltas)
    report = {
        "status": "REPETITION_REGRESSION_NOT_STATISTICALLY_CONFIRMED",
        "inputs": {
            "baseline": str(args.baseline.resolve()),
            "baseline_sha256": sha256(args.baseline),
            "candidate": str(args.candidate.resolve()),
            "candidate_sha256": sha256(args.candidate),
            "bootstrap_samples": args.samples,
            "seed": args.seed,
        },
        "paired_rows": len(keys),
        "mean_repetition_delta": mean_delta,
        "mean_repetition_delta_bootstrap_95": paired_bootstrap(
            repetition_deltas, args.samples, args.seed
        ),
        "loop_free_delta": loop_delta,
        "loop_free_delta_bootstrap_95": paired_bootstrap(
            binary_deltas, args.samples, args.seed + 1
        ),
        "discordant": {
            "gains": gains,
            "losses": losses,
            "exact_two_sided_sign_test_p": exact_sign_test_two_sided(gains, losses),
        },
        "threshold_sensitivity": threshold_report(
            baseline, candidate, (0.15, 0.20, 0.25, 0.30, 0.35)
        ),
        "changed_rows": changed,
        "interpretation": [
            "The original comparator used fixed point thresholds, not a confidence-aware paired repetition gate.",
            "A result is not a confirmed regression when both paired bootstrap intervals include zero and the discordant-row sign test is non-significant.",
            "The V15 checkpoint should be preserved as a capability candidate, while the 520M checkpoint remains the conservative general-pretraining base.",
        ],
    }
    if report["mean_repetition_delta_bootstrap_95"][0] > 0.0:
        report["status"] = "MEAN_REPETITION_REGRESSION_CONFIRMED"
    if report["loop_free_delta_bootstrap_95"][1] < 0.0:
        report["status"] = "LOOP_FREE_REGRESSION_CONFIRMED"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "paired_rows": report["paired_rows"],
                "mean_repetition_delta": mean_delta,
                "mean_repetition_delta_bootstrap_95": report[
                    "mean_repetition_delta_bootstrap_95"
                ],
                "loop_free_delta": loop_delta,
                "loop_free_delta_bootstrap_95": report["loop_free_delta_bootstrap_95"],
                "discordant": report["discordant"],
                "changed_rows": len(changed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
