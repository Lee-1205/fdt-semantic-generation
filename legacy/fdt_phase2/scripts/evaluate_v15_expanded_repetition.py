from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for entry in (SRC, SCRIPTS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import evaluate_base_model_strict as v2
import evaluate_base_model_strict_v3 as v3
from audit_v15_repetition_gate import exact_sign_test_two_sided, paired_bootstrap


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def require_c_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"Expanded V15 repetition evaluation permits only C: paths: {resolved}")
    return resolved


def disjoint_rows(start: int, end: int, count: int, excluded: set[int]) -> list[int]:
    available = [index for index in range(start, end) if index not in excluded]
    if len(available) < count:
        raise ValueError(f"Only {len(available)} disjoint rows available for requested {count}")
    if count == 1:
        return [available[len(available) // 2]]
    return [available[index * (len(available) - 1) // (count - 1)] for index in range(count)]


def fixed_selection(
    ranges: list[tuple[str, int, int]], rows_per_bucket: int, excluded_per_bucket: int
) -> dict[str, list[int]]:
    selection = {}
    for bucket, start, end in ranges:
        excluded = set(v2.evenly_spaced(start, end, excluded_per_bucket))
        selection[bucket] = disjoint_rows(start, end, rows_per_bucket, excluded)
    return selection


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint: Path,
    ids: torch.Tensor,
    masks: torch.Tensor,
    selection: dict[str, list[int]],
    device: torch.device,
    prefix_tokens: int,
    continuation_tokens: int,
    progress_callback=None,
) -> dict:
    model, tokenizer, config, metadata = v2.load_checkpoint_model(checkpoint, device)
    rows = []
    started = time.perf_counter()
    total = sum(len(values) for values in selection.values())
    completed = 0
    for bucket, row_ids in selection.items():
        for row_id in row_ids:
            active = ids[row_id][masks[row_id].bool()].tolist()
            if len(active) < prefix_tokens + continuation_tokens:
                raise ValueError(f"Validation row {bucket}:{row_id} is too short")
            prompt_ids = active[:prefix_tokens]
            reference = active[prefix_tokens : prefix_tokens + continuation_tokens]
            prompt = tokenizer.decode(prompt_ids, skip_special_tokens=True)
            completion, generated = v2.greedy_completion(
                model, tokenizer, config, prompt, device, continuation_tokens
            )
            prefix_match = 0
            for generated_id, reference_id in zip(generated, reference):
                if generated_id != reference_id:
                    break
                prefix_match += 1
            repetition = v2.repetition_rate(generated)
            rows.append(
                {
                    "bucket": bucket,
                    "row_id": row_id,
                    "prompt": prompt,
                    "reference": tokenizer.decode(reference, skip_special_tokens=True),
                    "completion": completion,
                    "generated_tokens": len(generated),
                    "repetition_rate": repetition,
                    "unique_token_rate": len(set(generated)) / max(len(generated), 1),
                    "reference_prefix_match_tokens": prefix_match,
                    "loop_free": repetition <= 0.20,
                    "non_whitespace": bool(completion.strip()),
                }
            )
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(
                    json.dumps(
                        {
                            "event": "expanded_prefix_generation",
                            "checkpoint": str(checkpoint),
                            "rows": completed,
                            "total": total,
                        }
                    ),
                    flush=True,
                )
                if progress_callback is not None:
                    progress_callback(metadata, rows)
    report = {
        "metadata": metadata,
        "checkpoint_sha256": sha256(checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": v3.summarize_heldout_generation(rows),
        "rows": rows,
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def compare(base: dict, candidate: dict, samples: int, seed: int) -> dict:
    old = {(row["bucket"], row["row_id"]): row for row in base["rows"]}
    new = {(row["bucket"], row["row_id"]): row for row in candidate["rows"]}
    keys = sorted(old.keys() & new.keys())
    repetition_deltas = [new[key]["repetition_rate"] - old[key]["repetition_rate"] for key in keys]
    binary_deltas = [int(new[key]["loop_free"]) - int(old[key]["loop_free"]) for key in keys]
    gains = sum(delta == 1 for delta in binary_deltas)
    losses = sum(delta == -1 for delta in binary_deltas)
    mean_delta = sum(repetition_deltas) / len(keys)
    loop_delta = sum(binary_deltas) / len(keys)
    mean_interval = paired_bootstrap(repetition_deltas, samples, seed)
    loop_interval = paired_bootstrap(binary_deltas, samples, seed + 1)
    by_bucket = {}
    for bucket in sorted({key[0] for key in keys}):
        bucket_keys = [key for key in keys if key[0] == bucket]
        deltas = [new[key]["repetition_rate"] - old[key]["repetition_rate"] for key in bucket_keys]
        loop_deltas = [int(new[key]["loop_free"]) - int(old[key]["loop_free"]) for key in bucket_keys]
        by_bucket[bucket] = {
            "rows": len(bucket_keys),
            "mean_repetition_delta": sum(deltas) / len(deltas),
            "loop_free_delta": sum(loop_deltas) / len(loop_deltas),
        }
    confirmed = mean_interval[0] > 0.0 or loop_interval[1] < 0.0
    return {
        "status": "REPETITION_REGRESSION_CONFIRMED" if confirmed else "REPETITION_REGRESSION_NOT_CONFIRMED",
        "paired_rows": len(keys),
        "mean_repetition_delta": mean_delta,
        "mean_repetition_delta_bootstrap_95": mean_interval,
        "loop_free_delta": loop_delta,
        "loop_free_delta_bootstrap_95": loop_interval,
        "discordant": {
            "gains": gains,
            "losses": losses,
            "exact_two_sided_sign_test_p": exact_sign_test_two_sided(gains, losses),
        },
        "by_bucket": by_bucket,
        "changed_rows": [
            {
                "bucket": key[0],
                "row_id": key[1],
                "baseline_repetition": old[key]["repetition_rate"],
                "candidate_repetition": new[key]["repetition_rate"],
                "baseline_loop_free": old[key]["loop_free"],
                "candidate_loop_free": new[key]["loop_free"],
                "baseline_completion": old[key]["completion"],
                "candidate_completion": new[key]["completion"],
            }
            for key in keys
            if old[key]["loop_free"] != new[key]["loop_free"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanded disjoint V15 repetition comparison.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-per-bucket", type=int, default=20)
    parser.add_argument("--exclude-standard-rows", type=int, default=10)
    parser.add_argument("--prefix-tokens", type=int, default=96)
    parser.add_argument("--continuation-tokens", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    for path in (args.baseline, args.candidate, args.dataset_dir, args.output):
        require_c_path(path)
    if args.output.exists():
        raise FileExistsError(f"Expanded evaluation output already exists: {args.output}")
    manifest = json.loads((args.dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    ids, masks = v2.split_rows(args.dataset_dir, "validation")
    ranges = v2.bucket_ranges(manifest, "validation")
    selection = fixed_selection(ranges, args.rows_per_bucket, args.exclude_standard_rows)
    device = torch.device(args.device)
    partial = args.output.with_suffix(args.output.suffix + ".partial")

    baseline = evaluate_checkpoint(
        args.baseline,
        ids,
        masks,
        selection,
        device,
        args.prefix_tokens,
        args.continuation_tokens,
        progress_callback=lambda metadata, rows: atomic_write_json(
            partial,
            {
                "status": "BASELINE_IN_PROGRESS",
                "baseline": {"metadata": metadata, "rows": rows},
            },
        ),
    )
    atomic_write_json(partial, {"status": "BASELINE_COMPLETE", "baseline": baseline})
    candidate = evaluate_checkpoint(
        args.candidate,
        ids,
        masks,
        selection,
        device,
        args.prefix_tokens,
        args.continuation_tokens,
        progress_callback=lambda metadata, rows: atomic_write_json(
            partial,
            {
                "status": "CANDIDATE_IN_PROGRESS",
                "baseline": baseline,
                "candidate": {"metadata": metadata, "rows": rows},
            },
        ),
    )
    comparison = compare(baseline, candidate, args.bootstrap_samples, args.seed)
    payload = {
        "status": comparison["status"],
        "settings": {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "manifest_sha256": sha256(args.dataset_dir / "manifest.json"),
            "rows_per_bucket": args.rows_per_bucket,
            "excluded_standard_rows_per_bucket": args.exclude_standard_rows,
            "prefix_tokens": args.prefix_tokens,
            "continuation_tokens": args.continuation_tokens,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "selection": selection,
        },
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
    }
    atomic_write_json(args.output, payload)
    partial.unlink(missing_ok=True)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == "__main__":
    main()
