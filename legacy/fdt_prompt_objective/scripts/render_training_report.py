from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"event": "invalid_jsonl", "raw": line[:500]})
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def training_summary(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    trains = [row for row in rows if row.get("event") == "train"]
    smoke = [row for row in rows if row.get("event") == "smoke_gate"]
    terminal = next(
        (row for row in reversed(rows) if row.get("event") in {"complete", "paused", "aborted"}),
        None,
    )
    losses = [value for row in trains if (value := finite(row.get("loss"))) is not None]
    throughput = [value for row in trains if (value := finite(row.get("tokens_per_sec"))) is not None]
    gradients = [value for row in trains if (value := finite(row.get("grad_norm"))) is not None]
    categories: dict[str, list[float]] = {}
    for row in trains:
        for name, raw in row.get("category_losses", {}).items():
            if (value := finite(raw)) is not None:
                categories.setdefault(name, []).append(value)
    category_summary = {
        name: {
            "first": values[0],
            "last": values[-1],
            "minimum": min(values),
            "change": values[-1] - values[0],
        }
        for name, values in categories.items()
        if values
    }
    return {
        "path": str(path),
        "sha256": sha256(path) if path.exists() else None,
        "events": len(rows),
        "train_points": len(trains),
        "terminal": terminal,
        "smoke_gates": smoke,
        "loss": {
            "first": losses[0] if losses else None,
            "last": losses[-1] if losses else None,
            "minimum": min(losses) if losses else None,
            "median": statistics.median(losses) if losses else None,
        },
        "tokens_per_second": {
            "last": throughput[-1] if throughput else None,
            "median": statistics.median(throughput) if throughput else None,
        },
        "maximum_gradient_norm": max(gradients) if gradients else None,
        "category_loss": category_summary,
    }


def rate(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    return {
        "successes": payload.get("successes"),
        "count": payload.get("count"),
        "rate": payload.get("rate"),
        "wilson_95": payload.get("wilson_95"),
    }


def generation_examples(category_name: str, category: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    examples = []
    for row in category.get("rows", [])[:limit]:
        examples.append({
            "row_id": row.get("row_id"),
            "prompt": str(row.get("prompt", ""))[-500:],
            "gold": row.get("gold"),
            "completion": str(row.get("completion", ""))[:500],
            "correct": None if category_name == "python_code" else row.get("correct"),
            "parseable": row.get("parseable"),
            "compilable": row.get("compilable"),
            "usable_structure": row.get("usable_structure"),
            "valid_json": row.get("valid_json"),
            "semantic_exact": row.get("correct") if category_name == "json" else row.get("semantic_exact"),
            "repetition_rate": row.get("repetition_rate"),
        })
    return examples


def strict_summary(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    evaluation = payload.get("evaluation", {})
    heldout = evaluation.get("heldout", {}).get("overall", {})
    teacher = evaluation.get("capability_teacher_forced", {})
    generation = evaluation.get("capability_generation", {})
    python = generation.get("python_code", {})
    json_generation = generation.get("json", {})
    prefix = evaluation.get("heldout_prefix_generation", {})
    result = {
        "path": str(path),
        "sha256": sha256(path),
        "status": payload.get("status"),
        "capability_verdict": payload.get("capability_verdict"),
        "checkpoint_sha256": payload.get("metadata", {}).get("checkpoint_sha256"),
        "heldout": {
            "loss": heldout.get("loss"),
            "top1_accuracy": heldout.get("top1_accuracy"),
            "dead_anchor_ratio": heldout.get("routing", {}).get("dead_anchor_ratio"),
            "effective_k": heldout.get("routing", {}).get("effective_k"),
        },
        "teacher_forced": {
            "overall_loss": teacher.get("overall", {}).get("loss"),
            "overall_top1": rate(teacher.get("overall", {}).get("top1")),
            "by_category": {
                name: {
                    "loss": row.get("loss"),
                    "top1": rate(row.get("top1")),
                    "sequence_exact": rate(row.get("sequence_exact")),
                }
                for name, row in teacher.get("by_category", {}).items()
            },
        },
        "free_generation": {
            "factual_exact": rate(generation.get("factual_qa", {}).get("correct")),
            "retrieval_exact": rate(generation.get("retrieval", {}).get("correct")),
            "python_parseable": rate(python.get("parseable")),
            "python_compilable": rate(python.get("compilable")),
            "python_structural": rate(python.get("usable_structure")),
            "python_ast_exact": rate(python.get("ast_exact")),
            "python_signature_exact": rate(python.get("function_signature_match")),
            "json_valid": rate(json_generation.get("valid")),
            "json_semantic_exact": rate(json_generation.get("correct")),
        },
        "repetition": {
            "loop_free": rate(prefix.get("loop_free")),
            "mean_repetition_rate": prefix.get("mean_repetition_rate"),
        },
        "examples": {
            name: generation_examples(name, row)
            for name, row in generation.items()
            if isinstance(row, dict)
        },
        "interpretation_warnings": [
            "Evaluator completion is not a capability pass.",
            "Python structural usability is not semantic correctness.",
            "Teacher-forced loss improvement is not free-generation capability.",
            "A non-significant regression is not proof of equivalence.",
        ],
    }
    return result


def comparison_summary(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    keep = {
        "path": str(path),
        "sha256": sha256(path),
        "status": payload.get("status"),
        "decision": payload.get("decision"),
        "legacy_gate_pass": payload.get("legacy_gate_pass"),
        "promotion_eligible": payload.get("promotion_eligible"),
    }
    for key in (
        "broad_relative_delta",
        "broad_relative_change",
        "factual_mean_row_loss_delta",
        "factual_mean_row_top1_delta",
        "retrieval_mean_row_loss_delta",
        "retrieval_mean_row_top1_delta",
        "json_mean_row_loss_delta",
        "loop_free_delta",
        "mean_repetition_delta",
    ):
        if key in payload:
            keep[key] = payload[key]
    keep["gates"] = payload.get("gates")
    return keep


def md_rate(value: dict[str, Any] | None) -> str:
    if not value:
        return "n/a"
    successes, count, fraction = value.get("successes"), value.get("count"), value.get("rate")
    if successes is not None and count is not None:
        return f"{successes}/{count} ({100.0 * float(fraction or 0):.2f}%)"
    return "n/a"


def markdown(metrics: dict[str, Any]) -> str:
    lines = [f"# Automated Training Report: {metrics['job_id']}", ""]
    lines += [
        f"- Controller status: `{metrics.get('controller_status')}`",
        f"- Generated: `{metrics.get('generated_at')}`",
        f"- Job SHA-256: `{metrics.get('job_sha256')}`",
        "",
    ]
    train = metrics.get("training")
    if train:
        terminal = train.get("terminal") or {}
        lines += ["## Training", ""]
        lines += [
            f"- Status: `{terminal.get('status', terminal.get('event', 'in_progress'))}`",
            f"- Step: `{terminal.get('step', 'n/a')}`; tokens: `{terminal.get('tokens_seen', 'n/a')}`",
            f"- Logged loss: first `{train['loss']['first']}`, last `{train['loss']['last']}`, minimum `{train['loss']['minimum']}`",
            f"- Median throughput: `{train['tokens_per_second']['median']}` tokens/s",
            f"- Maximum gradient norm: `{train.get('maximum_gradient_norm')}`",
            "",
            "### Category Loss",
            "",
            "| Category | First | Last | Change | Minimum |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, row in train.get("category_loss", {}).items():
            lines.append(f"| {name} | {row['first']:.6f} | {row['last']:.6f} | {row['change']:+.6f} | {row['minimum']:.6f} |")
        lines.append("")
    for strict in metrics.get("strict_evaluations", []):
        free = strict["free_generation"]
        lines += ["## Strict Evaluation", ""]
        lines += [
            f"- Artifact: `{strict['path']}`",
            f"- Status: `{strict.get('status')}` / verdict `{strict.get('capability_verdict')}`",
            f"- Broad heldout loss: `{strict['heldout'].get('loss')}`",
            f"- Factual exact: {md_rate(free.get('factual_exact'))}",
            f"- Retrieval exact: {md_rate(free.get('retrieval_exact'))}",
            f"- Python parseable: {md_rate(free.get('python_parseable'))}",
            f"- Python compilable: {md_rate(free.get('python_compilable'))}",
            f"- Python structurally usable: {md_rate(free.get('python_structural'))}",
            f"- Python AST exact: {md_rate(free.get('python_ast_exact'))}",
            f"- Python signature exact: {md_rate(free.get('python_signature_exact'))}",
            f"- JSON valid: {md_rate(free.get('json_valid'))}",
            f"- JSON semantic exact: {md_rate(free.get('json_semantic_exact'))}",
            f"- Loop-free: {md_rate(strict['repetition'].get('loop_free'))}",
            "",
            "### Generated Examples",
            "",
        ]
        for category, examples in strict.get("examples", {}).items():
            lines.append(f"#### {category}")
            lines.append("")
            for index, example in enumerate(examples, 1):
                lines += [
                    f"{index}. Prompt: `{example.get('prompt', '').replace(chr(10), ' ')[:220]}`",
                    f"   Gold: `{str(example.get('gold'))[:180]}`",
                    f"   Output: `{example.get('completion', '').replace(chr(10), ' ')[:220]}`",
                    f"   Flags: correct={example.get('correct')}, compilable={example.get('compilable')}, structural={example.get('usable_structure')}, valid_json={example.get('valid_json')}, semantic_exact={example.get('semantic_exact')}",
                ]
            lines.append("")
    if metrics.get("comparisons"):
        lines += ["## Comparisons", ""]
        for row in metrics["comparisons"]:
            lines.append(f"- `{row['path']}`: status `{row.get('status')}`, legacy gate `{row.get('legacy_gate_pass')}`, promotion `{row.get('promotion_eligible')}`")
        lines.append("")
    lines += [
        "## Review Contract",
        "",
        "Ask an LLM to inspect this report together with `metrics.json`, the job specification, and raw artifacts. Any proposed change must be written into a new immutable job file. Never overwrite the completed checkpoint or weaken a gate after seeing its result.",
        "",
    ]
    return "\n".join(lines)


def render(job_path: Path, state_path: Path | None = None) -> tuple[Path, Path]:
    job = load_json(job_path)
    workspace = Path(job["workspace"]).resolve()
    state_dir = (workspace / job["state_dir"]).resolve()
    state = load_json(state_path or state_dir / "state.json") if (state_path or state_dir / "state.json").exists() else {}
    report = job.get("report", {})
    training_path = workspace / report["training_log"] if report.get("training_log") else None
    strict_paths = [workspace / value for value in report.get("strict_evaluations", [])]
    comparison_paths = [workspace / value for value in report.get("comparisons", [])]
    metrics = {
        "schema_version": 1,
        "job_id": job["job_id"],
        "job_sha256": sha256(job_path),
        "generated_at": state.get("updated_at"),
        "controller_status": state.get("status"),
        "training": training_summary(training_path) if training_path and training_path.exists() else None,
        "strict_evaluations": [strict_summary(path) for path in strict_paths if path.exists()],
        "comparisons": [comparison_summary(path) for path in comparison_paths if path.exists()],
        "missing_report_inputs": [
            str(path)
            for path in ([training_path] if training_path else []) + strict_paths + comparison_paths
            if path is not None and not path.exists()
        ],
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = state_dir / "metrics.json"
    report_path = state_dir / "REPORT.md"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(markdown(metrics), encoding="utf-8")
    return metrics_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an immutable training/evaluation review bundle.")
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--state", type=Path)
    args = parser.parse_args()
    metrics, report = render(args.job.resolve(), args.state.resolve() if args.state else None)
    print(json.dumps({"metrics": str(metrics), "report": str(report)}, indent=2))


if __name__ == "__main__":
    main()
