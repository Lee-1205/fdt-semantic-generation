from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import evaluate_base_model_strict as v2
import evaluate_base_model_strict_v3 as v3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def require_c_path(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.drive.upper() != "C:":
        raise ValueError(f"Strict evaluation audit permits only C: paths: {resolved}")
    return resolved


def recorded_evaluator_source(recorded_hash: str | None) -> Path | None:
    if not recorded_hash:
        return None
    current = ROOT / "scripts" / "evaluate_base_model_strict_v3.py"
    if current.is_file() and sha256(current) == recorded_hash:
        return current
    snapshot = ROOT / "artifacts" / "evaluator_snapshots" / (
        f"evaluate_base_model_strict_v3_{recorded_hash}.py"
    )
    if snapshot.is_file() and sha256(snapshot) == recorded_hash:
        return snapshot
    return None


def walk_nonfinite(value, path: str = "root") -> list[str]:
    errors = []
    if isinstance(value, float) and not math.isfinite(value):
        errors.append(f"nonfinite value at {path}: {value}")
    elif isinstance(value, dict):
        for key, child in value.items():
            errors.extend(walk_nonfinite(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(walk_nonfinite(child, f"{path}[{index}]"))
    return errors


def unique_keys(rows: list[dict], fields: tuple[str, ...], label: str, errors: list[str]) -> None:
    keys = [tuple(row.get(field) for field in fields) for row in rows]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate {label} keys: {duplicates[:10]}")


def close(actual: float, expected: float, label: str, errors: list[str], tolerance=1e-9) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance):
        errors.append(f"{label} mismatch: artifact={actual}, recomputed={expected}")


def audit_generation(report: dict, settings: dict, errors: list[str], warnings: list[str]) -> dict:
    summary = {}
    for category, expected_count in settings["generation"].items():
        category_report = report[category]
        rows = category_report["rows"]
        if len(rows) != expected_count:
            errors.append(f"{category} generation count {len(rows)} != {expected_count}")
        unique_keys(rows, ("row_id",), f"{category} generation", errors)
        mismatches = []
        for row in rows:
            rescored = v3.score_capability_generation(
                category, row["gold"], row["completion"]
            )
            for field in (
                "correct",
                "valid",
                "parseable",
                "compilable",
                "has_function",
                "usable_structure",
                "ast_exact",
                "function_signature_match",
            ):
                if field in row and row[field] != rescored.get(field):
                    mismatches.append({"row_id": row["row_id"], "field": field})
        if mismatches:
            errors.append(f"{category} row rescoring mismatches: {mismatches[:20]}")
        correct = sum(bool(row["correct"]) for row in rows)
        if category_report["correct"]["successes"] != correct:
            errors.append(f"{category} aggregate correct count mismatch")
        close(
            category_report["mean_repetition_rate"],
            sum(float(row["repetition_rate"]) for row in rows) / max(len(rows), 1),
            f"{category} mean repetition",
            errors,
        )
        summary[category] = {
            "rows": len(rows),
            "rescoring_mismatches": len(mismatches),
            "correct": correct,
        }
    warnings.append(
        "Python 'correct' is structural usability only; compilability does not establish functional correctness."
    )
    warnings.append(
        "Greedy exact generation measures one deterministic trajectory and can undercount valid paraphrases."
    )
    return summary


def audit_teacher_forced(report: dict, expected_rows: int, errors: list[str]) -> dict:
    rows = report["rows"]
    if len(rows) != expected_rows:
        errors.append(f"teacher-forced rows {len(rows)} != {expected_rows}")
    unique_keys(rows, ("category", "row_id"), "teacher-forced", errors)
    for row in rows:
        if int(row["tokens"]) <= 0:
            errors.append(f"teacher-forced row has no supervised tokens: {row['row_id']}")
        if not 0 <= int(row["top1"]) <= int(row["tokens"]):
            errors.append(f"teacher-forced top1 out of range: {row['row_id']}")
        if not 0 <= int(row["top5"]) <= int(row["tokens"]):
            errors.append(f"teacher-forced top5 out of range: {row['row_id']}")
    categories = Counter(row["category"] for row in rows)
    return {"rows": len(rows), "categories": dict(categories)}


def audit_prefix(report: dict, settings: dict, errors: list[str]) -> dict:
    rows = report["rows"]
    expected = 5 * int(settings["general_rows_per_bucket"])
    if len(rows) != expected:
        errors.append(f"prefix generation rows {len(rows)} != {expected}")
    unique_keys(rows, ("bucket", "row_id"), "prefix generation", errors)
    for row in rows:
        repetition = float(row["repetition_rate"])
        if not 0.0 <= repetition <= 1.0:
            errors.append(f"prefix repetition out of range: {row['bucket']}:{row['row_id']}")
        if bool(row["loop_free"]) != (repetition <= 0.20):
            errors.append(f"prefix loop-free mismatch: {row['bucket']}:{row['row_id']}")
    overall = report["overall"]
    loop_free = sum(bool(row["loop_free"]) for row in rows)
    if int(overall["loop_free"]["successes"]) != loop_free:
        errors.append("prefix aggregate loop-free count mismatch")
    close(
        overall["mean_repetition_rate"],
        sum(float(row["repetition_rate"]) for row in rows) / max(len(rows), 1),
        "prefix mean repetition",
        errors,
    )
    return {"rows": len(rows), "loop_free": loop_free}


def audit_retrieval(report: dict, settings: dict, errors: list[str]) -> dict:
    rows = report["rows"]
    expected = int(settings["retrieval_cases"])
    if len(rows) != expected:
        errors.append(f"retrieval rows {len(rows)} != {expected}")
    unique_keys(rows, ("distance", "code"), "retrieval", errors)
    mismatches = 0
    for row in rows:
        scores = row["scores"]
        predicted = max(range(len(scores)), key=lambda index: scores[index]["mean"])
        rank_correct = predicted == 0
        greedy_exact = v2.normalize_code(row["completion"]).startswith(
            v2.normalize_code(row["code"])
        )
        if bool(row["rank_correct"]) != rank_correct:
            mismatches += 1
        if bool(row["greedy_exact"]) != greedy_exact:
            mismatches += 1
    if mismatches:
        errors.append(f"retrieval rescoring mismatches: {mismatches}")
    return {
        "rows": len(rows),
        "distance_counts": dict(Counter(str(row["distance"]) for row in rows)),
        "rescoring_mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict-v3 artifact integrity and semantics.")
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--require-standard", action="store_true")
    args = parser.parse_args()

    artifact_path = require_c_path(args.artifact)
    output_path = require_c_path(args.output)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    errors = walk_nonfinite(payload)
    warnings = []
    evaluator = payload.get("evaluator", {})
    metadata = payload.get("metadata", {})
    datasets = payload.get("datasets", {})
    settings = evaluator.get("settings", {})

    if payload.get("status") != "EVALUATION_COMPLETED":
        errors.append("artifact status is not EVALUATION_COMPLETED")
    if payload.get("capability_verdict") != "NOT_ASSIGNED_BY_EVALUATOR":
        errors.append("artifact conflates evaluation completion with a capability verdict")
    supported_versions = {"strict_base_v3_audited_1", "strict_base_v3_audited_2"}
    if evaluator.get("version") not in supported_versions:
        errors.append(
            f"unsupported audited evaluator version: {evaluator.get('version')!r}"
        )
    evaluator_source = recorded_evaluator_source(evaluator.get("script_sha256"))
    if evaluator_source is None:
        errors.append("recorded evaluator source is neither current nor preserved as an exact snapshot")
    if args.require_standard and evaluator.get("mode") != "standard":
        errors.append("artifact is not a standard-mode evaluation")

    checkpoint = Path(metadata.get("checkpoint", ""))
    if not checkpoint.is_file():
        errors.append(f"checkpoint missing: {checkpoint}")
        checkpoint_hash = None
    else:
        require_c_path(checkpoint)
        checkpoint_hash = sha256(checkpoint)
        if checkpoint_hash != metadata.get("checkpoint_sha256"):
            errors.append("checkpoint hash does not match artifact metadata")
        if args.expected_checkpoint_sha256 and checkpoint_hash.upper() != args.expected_checkpoint_sha256.upper():
            errors.append("checkpoint hash does not match the expected pinned hash")

    fingerprints = {}
    for label, path_key, fingerprint_key in (
        ("broad", "broad_validation", "broad_validation_fingerprint"),
        ("capability", "capability_validation", "capability_validation_fingerprint"),
    ):
        dataset_path = Path(datasets.get(path_key, ""))
        try:
            require_c_path(dataset_path)
            actual = v3.dataset_split_fingerprint(dataset_path, "validation")
            recorded = datasets.get(fingerprint_key, {})
            fingerprints[label] = actual
            if actual != recorded:
                errors.append(f"{label} validation fingerprint mismatch")
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))

    evaluation = payload.get("evaluation", {})
    generation = audit_generation(
        evaluation.get("capability_generation", {}), settings, errors, warnings
    )
    teacher_forced = audit_teacher_forced(
        evaluation.get("capability_teacher_forced", {}),
        int(datasets.get("capability_validation_rows", -1)),
        errors,
    )
    prefix = audit_prefix(
        evaluation.get("heldout_prefix_generation", {}), settings, errors
    )
    retrieval = audit_retrieval(
        evaluation.get("retrieval_base_continuation", {}), settings, errors
    )

    if datasets.get("unbiased_sealed_final_test"):
        split_role = "sealed_final_test"
    else:
        split_role = "reused_development_gate"
        warnings.append(
            "The capability split has informed repeated lineage decisions; it is not an unbiased final test."
        )

    report = {
        "status": "AUDIT_PASS" if not errors else "AUDIT_FAIL",
        "auditor": {
            "version": "strict_v3_integrity_auditor_1",
            "script_sha256": sha256(Path(__file__)),
        },
        "promotion_decision_allowed": False,
        "promotion_blockers": (
            ["independent sealed holdout confirmation required"] if not errors else errors
        ),
        "artifact": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "evaluator_source": str(evaluator_source) if evaluator_source else None,
        "checkpoint_sha256": checkpoint_hash,
        "split_role": split_role,
        "checks": {
            "generation": generation,
            "teacher_forced": teacher_forced,
            "prefix_generation": prefix,
            "retrieval": retrieval,
            "dataset_fingerprints": fingerprints,
        },
        "errors": errors,
        "warnings": warnings,
        "interpretation_rules": [
            "Evaluation completion is not a capability pass.",
            "Teacher-forced improvement is not free-generation capability.",
            "Python compilability/structure is not semantic or functional correctness.",
            "A zero exact score may miss valid paraphrases; a prefix score may accept irrelevant suffixes.",
            "A non-significant regression is not proof of equivalence; use predeclared noninferiority bounds.",
            "Repeatedly consulted validation is a development gate, not a sealed final test.",
        ],
    }
    atomic_write_json(output_path, report)
    print(json.dumps({key: report[key] for key in ("status", "artifact", "errors", "warnings")}, indent=2))
    raise SystemExit(0 if not errors else 2)


if __name__ == "__main__":
    main()
