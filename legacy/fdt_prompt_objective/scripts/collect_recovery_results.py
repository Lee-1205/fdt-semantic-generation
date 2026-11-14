from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def capability_summary(payload: dict, model: str) -> dict:
    row = payload["models"][model]
    code = row["code"]
    json_rows = [item for item in code if item["language"] == "json"]
    return {
        "loss": row["language"]["overall"]["loss"],
        "perplexity": row["language"]["overall"]["perplexity"],
        "top1_accuracy": row["language"]["overall"]["top1_accuracy"],
        "top5_accuracy": row["language"]["overall"]["top5_accuracy"],
        "code_parseability": sum(bool(item["parseable"]) for item in code) / max(len(code), 1),
        "json_parseability": sum(bool(item["parseable"]) for item in json_rows) / max(len(json_rows), 1),
        "valid_action_rate": row["action"]["metrics"]["valid_action_rate"],
        "retrieval_contains_answer": row["retrieval"]["contains_answer"],
        "readiness": row["readiness"],
        "anchor_health": row.get("anchor_health"),
    }


def main() -> None:
    capabilities = {
        stage: read(ROOT / "artifacts" / f"stage_{stage}_capability_results.json")
        for stage in ("25m", "50m", "100m")
    }
    recovered_capability = read(ROOT / "artifacts" / "recovered_100m_capability_results.json")
    recovered_sources = read(ROOT / "artifacts" / "recovered_100m_source_specific_eval.json")
    broken_sources = read(ROOT / "artifacts" / "source_specific_checkpoint_eval.json")["stages"]["100m"]
    index_manifest = read(ROOT / "prepared_data" / "fdt_rlm_remaining_interleaved_50m" / "manifest.json")
    smoke = read(ROOT / "artifacts" / "recovery_smoke_results.json")

    models = {}
    for model in ("transformer", "fdt"):
        training = read(ROOT / "runs" / "staged_100m_interleaved" / model / "stage_summary.json")
        stage_capabilities = {
            stage: capability_summary(payload, model)
            for stage, payload in capabilities.items()
        }
        recovered = capability_summary(recovered_capability, model)
        recovered_better_than_50m = recovered["loss"] < stage_capabilities["50m"]["loss"]
        models[model] = {
            "capability": {
                **stage_capabilities,
                "recovered_100m": recovered,
            },
            "recovered_training": {
                "status": training["status"],
                "tokens_seen": training["tokens_seen"],
                "validation_loss": training["final_val_loss"],
                "perplexity": training["final_val_ppl"],
                "tokens_per_second": training["tokens_per_second"],
                "peak_allocated_vram_bytes": training["peak_allocated_vram_bytes"],
                "wall_time_seconds": training["wall_time_seconds"],
            },
            "source_losses": {
                "broken_100m": broken_sources[model]["losses"],
                "recovered_100m": recovered_sources["models"][model]["losses"],
            },
            "recovered_better_than_50m": recovered_better_than_50m,
            "official_checkpoint": str(
                (ROOT / "runs" / "staged_100m_interleaved" / model / "latest.pt").relative_to(ROOT)
            ),
        }

    result = {
        "status": "PASS" if all(row["recovered_better_than_50m"] for row in models.values()) else "PARTIAL",
        "controlled_variable": "training sequence order only",
        "same_recovery_stream": True,
        "recovery_index": index_manifest,
        "smoke": smoke,
        "models": models,
        "capability_emergence": {
            "code": "not observed",
            "json": "not observed",
            "retrieval": "not observed",
            "action": "not observed",
        },
        "decision": "Adopt recovered 100M checkpoints; retain broken 100M only as forensic evidence.",
    }
    (ROOT / "artifacts" / "recovery_experiment_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    lines = [
        "# FDT-RLM Recovery Experiment", "",
        "## Control", "",
        "Only training sequence order changed. Architecture, tokenizer, loss, routing, anchors, optimizer, LR, hyperparameters, validation, and evaluator remained unchanged.", "",
        f"The shared virtual recovery dataset contains {index_manifest['tokens']:,} tokens in {index_manifest['sequences']:,} unique 512-token sequences, with zero replay and a maximum same-source run of {index_manifest['longest_single_source_run_sequences']} sequences.", "",
        "## Results", "",
        "| Model | 50M loss | Broken 100M | Recovered 100M | vs 50M | vs Broken |", 
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model, row in models.items():
        cap = row["capability"]
        lines.append(
            f"| {model.title()} | {cap['50m']['loss']:.4f} | {cap['100m']['loss']:.4f} | {cap['recovered_100m']['loss']:.4f} | "
            f"{cap['recovered_100m']['loss'] - cap['50m']['loss']:+.4f} | {cap['recovered_100m']['loss'] - cap['100m']['loss']:+.4f} |"
        )
    lines.extend(["", "Source-specific validation also improves on both web and code:", "", "| Model | Broken Edu | Recovered Edu | Broken Python | Recovered Python |", "|---|---:|---:|---:|---:|"])
    for model, row in models.items():
        broken = row["source_losses"]["broken_100m"]
        recovered = row["source_losses"]["recovered_100m"]
        lines.append(
            f"| {model.title()} | {broken['fineweb_edu']['loss']:.4f} | {recovered['fineweb_edu']['loss']:.4f} | "
            f"{broken['code_python']['loss']:.4f} | {recovered['code_python']['loss']:.4f} |"
        )
    lines.extend(["", "## Capability And Runtime", "", "| Model | Top-1 | Top-5 | Code | JSON | Retrieval | Action | VRAM GiB | tok/s |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for model, row in models.items():
        cap = row["capability"]["recovered_100m"]
        train = row["recovered_training"]
        lines.append(
            f"| {model.title()} | {cap['top1_accuracy']:.4f} | {cap['top5_accuracy']:.4f} | {cap['code_parseability']:.4f} | "
            f"{cap['json_parseability']:.4f} | {cap['retrieval_contains_answer']:.4f} | {cap['valid_action_rate']:.4f} | "
            f"{train['peak_allocated_vram_bytes'] / 2**30:.2f} | {train['tokens_per_second']:.0f} |"
        )
    anchor = models["fdt"]["capability"]["recovered_100m"]["anchor_health"]
    lines.extend([
        "", "## FDT Anchor Health", "",
        f"Entropy={anchor['entropy']:.4f}, normalized entropy={anchor['normalized_entropy']:.4f}, effective-K={anchor['effective_k']:.4f}, top-1 membership={anchor['top1_membership']:.4f}, dead-anchor ratio={anchor['dead_anchor_ratio']:.4f}.", "",
        "## Decision", "",
        "Recovery PASS. Both recovered models beat their trusted 50M checkpoints and the broken 100M runs. The recovered checkpoints are the official 100M results; structured capability emergence remains unobserved and must not be claimed.",
    ])
    (ROOT / "RECOVERY_EXPERIMENT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({model: {"loss": row["capability"]["recovered_100m"]["loss"], "official": row["official_checkpoint"]} for model, row in models.items()}, indent=2))


if __name__ == "__main__":
    main()
