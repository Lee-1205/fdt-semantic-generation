from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read(path: Path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main() -> None:
    state = read(ROOT / "artifacts" / "current_effective_token_state.json", {})
    manifest = read(ROOT / "prepared_data" / "fdt_rlm_100m" / "manifest.json", {})
    comparison = read(ROOT / "artifacts" / "staged_pretraining_comparison.json", {"25m": {}, "50m": {}, "100m": {}})
    emergence = read(ROOT / "artifacts" / "capability_emergence.json", {})
    lines = [
        "# Staged Pretraining Final Report", "",
        "## Starting state", "",
        f"- Transformer best: {state.get('transformer', {}).get('effective_tokens_seen', 'unknown')} effective tokens",
        f"- FDT corrected best: {state.get('fdt', {}).get('effective_tokens_seen', 'unknown')} effective tokens",
        f"- Tokenizer verified: {state.get('tokenizer', {}).get('verified', False)}", "",
        "## Corpus", "",
        f"- requested source tokens: {manifest.get('requested_tokens', 0):,}",
        f"- actual source tokens: {sum(int(value) for value in manifest.get('tokens_by_bucket', {}).values()):,}",
        f"- packed tokens: {manifest.get('packed_tokens', 0):,}",
        f"- tokenizer vocab: {manifest.get('tokenizer_vocab_size', 0):,}",
        f"- validation: {read(ROOT / 'prepared_data' / 'fdt_rlm_100m' / 'validation_report.json', {}).get('status', 'unknown')}", "",
        "## Milestones", "",
    ]
    for milestone in ("25m", "50m", "100m"):
        rows = comparison.get(milestone, {})
        if not rows:
            lines.extend([f"### {milestone.upper()}", "", "Not reached.", ""])
            continue
        lines.extend([f"### {milestone.upper()}", ""])
        for model in ("transformer", "fdt"):
            row = rows[model]
            anchor = row.get("anchor_health") or {}
            lines.append(
                f"- {model}: val={row['validation_loss']:.6f}, action={row['valid_action_rate']:.4f}, "
                f"retrieval={row['short_retrieval']:.4f}, readiness={row['readiness']}, speed={row['tokens_per_second']:.0f} tok/s"
            )
            if anchor:
                lines.append(
                    f"  anchors: entropy={anchor['entropy']:.4f}, effective-K={anchor['effective_k']:.4f}, "
                    f"top1={anchor['top1_membership']:.4f}, dead={anchor['dead_anchor_ratio']:.4f}"
                )
        lines.append("")
    lines.extend([
        "## Capability emergence", "",
        "Detailed first-nonzero and first-meaningful milestones are stored in `artifacts/capability_emergence.json`.", "",
        "## RLM gate", "",
    ])
    reached = [comparison[key] for key in ("25m", "50m", "100m") if comparison.get(key)]
    latest = reached[-1] if reached else {}
    gate = any(row.get("valid_action_rate", 0) >= 0.5 and row.get("short_retrieval", 0) >= 0.3 for row in latest.values())
    lines.append(f"- reactivated: {'YES' if gate else 'NO'}")
    lines.append("- criterion: valid actions >= 50% and short retrieval >= 30%")
    lines.extend(["", "## Notes", "", "The aborted high-LR stage attempt was archived and was not used as a source checkpoint."])
    (ROOT / "STAGED_PRETRAINING_FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": "STAGED_PRETRAINING_FINAL_REPORT.md", "rlm_reactivated": gate, "stages": [key for key in comparison if comparison[key]]}, indent=2))


if __name__ == "__main__":
    main()
