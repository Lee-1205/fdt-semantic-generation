from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_model(model: dict, training: dict) -> dict:
    code_rows = model["code"]
    code_parseability = sum(bool(row["parseable"]) for row in code_rows) / max(len(code_rows), 1)
    json_rows = [row for row in code_rows if row["language"] == "json"]
    json_parseability = sum(bool(row["parseable"]) for row in json_rows) / max(len(json_rows), 1)
    overall = model["language"]["overall"]
    action = model["action"]["metrics"]
    return {
        "effective_tokens_seen": int(training["tokens_seen"]),
        "validation_loss": overall["loss"],
        "perplexity": overall["perplexity"],
        "top1": overall["top1_accuracy"],
        "top5": overall["top5_accuracy"],
        "top10": overall["top10_accuracy"],
        "code_parseability": code_parseability,
        "json_parseability": json_parseability,
        "valid_action_rate": action["valid_action_rate"],
        "action_accuracy": action["action_accuracy"],
        "short_retrieval": model["retrieval"]["contains_answer"],
        "external_context_action_accuracy": action["action_accuracy"],
        "readiness": model["readiness"],
        "tokens_per_second": training["tokens_per_second"],
        "peak_allocated_vram_bytes": training["peak_allocated_vram_bytes"],
        "peak_reserved_vram_bytes": training["peak_reserved_vram_bytes"],
        "wall_time_seconds": training["wall_time_seconds"],
        "anchor_health": model.get("anchor_health"),
    }


def update_emergence(comparison: dict) -> dict:
    capabilities = ["code_parseability", "json_parseability", "valid_action_rate", "short_retrieval"]
    result = {model: {} for model in ("transformer", "fdt")}
    for model in result:
        for capability in capabilities:
            observed = []
            for stage in ("25m", "50m", "100m"):
                row = comparison.get(stage, {}).get(model)
                if row:
                    observed.append((stage, float(row.get(capability, 0.0))))
            nonzero = next((stage for stage, value in observed if value > 0), None)
            meaningful = next((stage for stage, value in observed if value >= 0.05), None)
            best = max(observed, key=lambda item: item[1]) if observed else (None, None)
            result[model][capability] = {
                "first_nonzero_milestone": nonzero,
                "first_meaningful_milestone": meaningful,
                "best_milestone": best[0],
                "best_value": best[1],
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--milestone", choices=["25m", "50m", "100m"], required=True)
    args = parser.parse_args()
    capability = read(ROOT / "artifacts" / f"stage_{args.milestone}_capability_results.json")
    stage = {}
    for model in ("transformer", "fdt"):
        training = read(ROOT / "runs" / f"staged_{args.milestone}" / model / "stage_summary.json")
        stage[model] = summarize_model(capability["models"][model], training)
    output = {
        "milestone": args.milestone,
        "fixed_validation_sequences": capability["fixed_validation_sequences"],
        "models": stage,
    }
    (ROOT / "artifacts" / f"stage_{args.milestone}_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")

    comparison_path = ROOT / "artifacts" / "staged_pretraining_comparison.json"
    comparison = read(comparison_path) if comparison_path.exists() else {"25m": {}, "50m": {}, "100m": {}}
    comparison[args.milestone] = stage
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    emergence = update_emergence(comparison)
    (ROOT / "artifacts" / "capability_emergence.json").write_text(json.dumps(emergence, indent=2), encoding="utf-8")

    lines = [f"# Stage {args.milestone.upper()} Report", ""]
    for model, row in stage.items():
        lines.extend([
            f"## {model.title()}", "",
            f"- effective tokens: {row['effective_tokens_seen']:,}",
            f"- validation loss: {row['validation_loss']:.6f}",
            f"- top-1 / top-5 / top-10: {row['top1']:.4f} / {row['top5']:.4f} / {row['top10']:.4f}",
            f"- code / JSON parseability: {row['code_parseability']:.4f} / {row['json_parseability']:.4f}",
            f"- valid action / retrieval: {row['valid_action_rate']:.4f} / {row['short_retrieval']:.4f}",
            f"- readiness: {row['readiness']}", "",
        ])
    (ROOT / f"STAGE_{args.milestone.upper()}_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({model: {key: row[key] for key in ("effective_tokens_seen", "validation_loss", "valid_action_rate", "short_retrieval", "readiness")} for model, row in stage.items()}, indent=2))


if __name__ == "__main__":
    main()
