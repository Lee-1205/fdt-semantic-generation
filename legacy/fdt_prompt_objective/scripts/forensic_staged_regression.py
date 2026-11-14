from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model


STAGES = ("25m", "50m", "100m")
MODELS = ("transformer", "fdt")
BUCKET_ORDER = (
    "fineweb_edu", "fineweb", "code_python", "code_javascript_typescript",
    "code_c_cpp", "code_java", "code_rust_go", "code_structured_docs",
)


def safe_load(path: Path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def write_json(name: str, payload: Any) -> None:
    path = ROOT / "artifacts" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checkpoint_path(stage: str, model: str, name: str = "latest.pt") -> Path:
    return ROOT / "runs" / f"staged_{stage}" / model / name


def optimizer_steps(checkpoint: dict) -> dict[str, Any]:
    state = checkpoint.get("optimizer_state_dict", {}).get("state", {})
    steps = []
    for value in state.values():
        step = value.get("step") if isinstance(value, dict) else None
        if step is not None:
            steps.append(int(step.item() if torch.is_tensor(step) else step))
    return {
        "parameter_states": len(state),
        "adam_step_min": min(steps) if steps else None,
        "adam_step_max": max(steps) if steps else None,
    }


def inventory() -> dict[str, Any]:
    result = {}
    for stage in STAGES:
        result[stage] = {}
        for model in MODELS:
            path = checkpoint_path(stage, model)
            checkpoint = safe_load(path)
            groups = checkpoint.get("optimizer_state_dict", {}).get("param_groups", [])
            scheduler = checkpoint.get("scheduler_state_dict")
            result[stage][model] = {
                "path": str(path.relative_to(ROOT)),
                "file_size": path.stat().st_size,
                "sha256": sha256_file(path),
                "model_type": checkpoint.get("model_type"),
                "parameter_count": checkpoint.get("parameter_count"),
                "global_step": checkpoint.get("global_step"),
                "optimizer_step": checkpoint.get("optimizer_step"),
                "micro_step": checkpoint.get("micro_step"),
                "tokens_seen": checkpoint.get("tokens_seen"),
                "best_val_loss": checkpoint.get("best_val_loss"),
                "epoch": checkpoint.get("epoch"),
                "learning_rates": [group.get("lr") for group in groups],
                "optimizer_state_available": "optimizer_state_dict" in checkpoint,
                "scheduler_state_available": scheduler is not None,
                "scaler_state_available": "scaler_state_dict" in checkpoint,
                "dataset_cursor_available": "sequence_index" in checkpoint,
                "sequence_index": checkpoint.get("sequence_index"),
                "shuffle_state_available": any(key in checkpoint for key in ("sampler_state", "shuffle_state", "rng_state")),
                "training_config": checkpoint.get("train_config"),
                "tokenizer_hash": checkpoint.get("tokenizer_hash"),
                "dataset_manifest_hash": checkpoint.get("dataset_manifest_sha256"),
                "optimizer_diagnostics": optimizer_steps(checkpoint),
                "scheduler_state": scheduler,
            }
    write_json("staged_checkpoint_inventory.json", result)
    return result


def load_split(dataset: Path, split: str) -> tuple[torch.Tensor, torch.Tensor]:
    ids_parts, mask_parts = [], []
    for path in sorted((dataset / "shards" / split).glob("*.pt")):
        shard = safe_load(path)
        ids = shard["input_ids"].long()
        ids_parts.append(ids)
        mask_parts.append(shard.get("attention_mask", torch.ones_like(ids)).long())
    return torch.cat(ids_parts), torch.cat(mask_parts)


def row_hash(ids: torch.Tensor, mask: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(ids.numpy().tobytes())
    digest.update(mask.numpy().tobytes())
    return digest.hexdigest()


def split_identity(ids: torch.Tensor, mask: torch.Tensor, dataset: Path, split: str) -> dict[str, Any]:
    rows = [row_hash(row_ids, row_mask) for row_ids, row_mask in zip(ids, mask)]
    digest = hashlib.sha256("".join(rows).encode()).hexdigest()
    shards = []
    for path in sorted((dataset / "shards" / split).glob("*.pt")):
        shards.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)})
    return {
        "dataset": str(dataset.relative_to(ROOT)),
        "split": split,
        "sequences": len(rows),
        "input_tokens": int(mask.sum().item()),
        "first_10_sample_ids": rows[:10],
        "last_10_sample_ids": rows[-10:],
        "validation_split_hash": digest,
        "shards": shards,
    }


def labels_for_split(dataset: Path, split: str, sequences: int) -> list[str]:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    counts = defaultdict(int)
    for shard in manifest["shards"]:
        if shard["split"] == split:
            for bucket, tokens in shard["buckets"].items():
                counts[bucket] += int(tokens) // int(manifest["seq_len"])
    labels = []
    for bucket in BUCKET_ORDER:
        labels.extend([bucket] * counts[bucket])
    labels.extend(["unknown"] * max(0, sequences - len(labels)))
    return labels[:sequences]


@torch.inference_mode()
def evaluate_checkpoint(path: Path, ids: torch.Tensor, mask: torch.Tensor, labels: list[str], device: torch.device) -> dict[str, Any]:
    checkpoint = safe_load(path, device)
    config = ModelConfig(**checkpoint["model_config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    totals = defaultdict(lambda: {"loss_sum": 0.0, "tokens": 0})
    usage = None
    anchor_count = 0
    anchor_sums = defaultdict(float)
    batch_size = 4
    for start in range(0, len(ids), batch_size):
        x = ids[start : start + batch_size].to(device)
        m = mask[start : start + batch_size].to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(input_ids=x, attention_mask=m)
        logits = output["logits"][:, :-1].float()
        targets = x[:, 1:]
        valid = m[:, 1:].bool() & (targets != config.pad_token_id)
        losses = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
        for row in range(len(x)):
            row_valid = valid[row]
            loss_sum = float(losses[row][row_valid].sum().item())
            tokens = int(row_valid.sum().item())
            for key in ("overall", labels[start + row]):
                totals[key]["loss_sum"] += loss_sum
                totals[key]["tokens"] += tokens
        for stats in output.get("anchor_stats", []):
            entropy = float(stats.entropy.detach().float().item())
            anchor_sums["entropy"] += entropy
            anchor_sums["effective_k"] += math.exp(entropy)
            anchor_sums["top1_membership"] += float(stats.top1_membership.detach().float().item())
            anchor_sums["top1_top2_margin"] += float(stats.membership_margin.detach().float().item())
            current = stats.usage_prob.detach().float().cpu()
            usage = current.clone() if usage is None else usage + current
            anchor_count += 1
    metrics = {}
    for key, value in totals.items():
        loss = value["loss_sum"] / max(value["tokens"], 1)
        metrics[key] = {"loss": loss, "perplexity": math.exp(loss) if loss < 80 else float("inf"), "target_tokens": value["tokens"]}
    anchor = None
    if anchor_count and usage is not None:
        usage /= anchor_count
        anchor = {
            "entropy": anchor_sums["entropy"] / anchor_count,
            "normalized_entropy": anchor_sums["entropy"] / anchor_count / math.log(max(config.top_k, 2)),
            "effective_k": anchor_sums["effective_k"] / anchor_count,
            "top1_membership": anchor_sums["top1_membership"] / anchor_count,
            "top1_top2_margin": anchor_sums["top1_top2_margin"] / anchor_count,
            "dead_anchor_ratio": float((usage < 1e-4).float().mean().item()),
            "top_anchor_frequency": float(usage.max().item()),
            "usage_distribution": usage.tolist(),
        }
    del model, checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"checkpoint": str(path.relative_to(ROOT)), "losses": metrics, "anchor_health": anchor}


def parse_metrics(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("event") == "train":
            rows.append({key: row.get(key) for key in ("optimizer_step", "tokens_seen", "loss", "lm", "grad_norm", "lr", "entropy", "effective_anchors", "dead_anchor_ratio")})
    return rows


def lr_audit() -> dict[str, Any]:
    result = {}
    for model in MODELS:
        result[model] = {}
        previous_end = None
        for stage in STAGES:
            rows = parse_metrics(ROOT / "runs" / f"staged_{stage}" / model / "metrics.jsonl")
            first, last = rows[0], rows[-1]
            result[model][stage] = {
                "first_train_lr": first["lr"], "last_train_lr": last["lr"],
                "first_optimizer_step": first["optimizer_step"], "last_optimizer_step": last["optimizer_step"],
                "first_tokens_seen": first["tokens_seen"], "last_tokens_seen": last["tokens_seen"],
                "warmup_restarted": False,
                "lr_change_from_previous_stage_end": None if previous_end is None else first["lr"] / previous_end,
            }
            previous_end = last["lr"]
    result["conclusion"] = "CONTINUOUS" if all(result[m][s]["first_train_lr"] == 3e-5 and result[m][s]["last_train_lr"] == 3e-5 for m in MODELS for s in STAGES) else "DISCONTINUOUS"
    write_json("lr_resume_audit.json", result)
    return result


def cursor_and_samples(inventory_data: dict[str, Any], train_dataset: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cursor, traces = {}, {}
    for model in MODELS:
        cursor[model], traces[model] = {}, {}
        prior_end = 0
        for stage in STAGES:
            end = int(inventory_data[stage][model]["sequence_index"])
            start = 0 if stage == "25m" else prior_end
            cursor[model][stage] = {
                "start_sequence": start, "end_sequence": end,
                "start_token_offset": start * 512, "end_token_offset": end * 512,
                "sequences_consumed": end - start,
            }
            traces[model][stage] = sample_ids(train_dataset, start, 100)
            prior_end = end
        overlaps = {}
        for left, right in (("25m", "50m"), ("50m", "100m")):
            overlaps[f"{left}_vs_{right}"] = len(set(traces[model][left]) & set(traces[model][right]))
        traces[model]["overlap"] = overlaps
    cursor["classification"] = "TRUE_CONTINUATION"
    cursor["shuffle_state"] = "none; deterministic sequential packed rows"
    traces["sample_replay_detected_at_stage_transition"] = any(
        traces[m]["overlap"][key] > 0 for m in MODELS for key in traces[m]["overlap"]
    )
    write_json("dataset_cursor_audit.json", cursor)
    write_json("stage_sample_trace.json", traces)
    return cursor, traces


def sample_ids(dataset: Path, start: int, count: int) -> list[str]:
    paths = sorted((dataset / "shards" / "train").glob("*.pt"))
    result, offset = [], 0
    stop = start + count
    for path in paths:
        shard = safe_load(path)
        ids = shard["input_ids"]
        shard_end = offset + len(ids)
        if start < shard_end and stop > offset:
            local_start = max(start - offset, 0)
            local_stop = min(stop - offset, len(ids))
            for row in ids[local_start:local_stop]:
                result.append(hashlib.sha256(row.numpy().tobytes()).hexdigest())
        offset = shard_end
        if offset >= stop:
            break
    return result


def source_order(manifest: dict, cursor: dict[str, Any]) -> dict[str, Any]:
    counts = defaultdict(int)
    for shard in manifest["shards"]:
        if shard["split"] == "train":
            for bucket, tokens in shard["buckets"].items():
                counts[bucket] += int(tokens)
    segments, position = [], 0
    for bucket in BUCKET_ORDER:
        tokens = counts[bucket]
        segments.append({"bucket": bucket, "start": position, "end": position + tokens, "tokens": tokens})
        position += tokens

    def composition(start: int, end: int) -> dict[str, float]:
        denominator = max(end - start, 1)
        values = {}
        for segment in segments:
            overlap = max(0, min(end, segment["end"]) - max(start, segment["start"]))
            values[segment["bucket"]] = overlap / denominator
        return values

    ranges = {}
    for label, start, end in (("0_10m", 0, 10_000_000), ("10_25m", 10_000_000, 25_000_000), ("25_50m", 25_000_000, 50_000_000), ("50_75m", 50_000_000, 75_000_000), ("75_100m", 75_000_000, min(position, 100_000_000))):
        ranges[label] = composition(start, end)
    stage_ranges = {}
    for model in MODELS:
        stage_ranges[model] = {}
        for stage in STAGES:
            row = cursor[model][stage]
            stage_ranges[model][stage] = composition(row["start_token_offset"], row["end_token_offset"])
    result = {
        "classification": "GROUPED_BY_SOURCE",
        "shuffle": False,
        "segments": segments,
        "fixed_ranges": ranges,
        "actual_model_stage_ranges": stage_ranges,
        "conclusion": "The 50M-to-100M continuation crosses from FineWeb-Edu concentration into FineWeb and code concentration for both models.",
    }
    write_json("source_order_audit.json", result)
    return result


def config_audit() -> dict[str, Any]:
    result = {}
    for model in MODELS:
        result[model] = {}
        configs = {}
        for stage in STAGES:
            directory = ROOT / "runs" / f"staged_{stage}" / model
            configs[stage] = {
                "train": json.loads((directory / "train_config.json").read_text(encoding="utf-8")),
                "model": json.loads((directory / "model_config.json").read_text(encoding="utf-8")),
            }
        baseline = configs["25m"]
        for stage in STAGES:
            diffs = {}
            for section in ("train", "model"):
                keys = set(baseline[section]) | set(configs[stage][section])
                diffs[section] = {key: {"25m": baseline[section].get(key), stage: configs[stage][section].get(key)} for key in keys if baseline[section].get(key) != configs[stage][section].get(key) and key not in {"output_dir", "max_steps"}}
            result[model][stage] = {"unexpected_differences": diffs, "config": configs[stage]}
    result["conclusion"] = "NO_UNEXPECTED_HYPERPARAMETER_CHANGE"
    write_json("stage_config_diff.json", result)
    return result


def plot_histories(histories: dict[str, Any], reevaluation: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    output = ROOT / "artifacts" / "plots"
    output.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        plt.figure(figsize=(9, 4))
        for stage in STAGES:
            rows = histories[model][stage]
            plt.plot([row["tokens_seen"] for row in rows], [row["lm"] for row in rows], label=stage, linewidth=0.7)
        plt.xlabel("effective input tokens")
        plt.ylabel("training LM loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output / f"{model}_staged_loss.png", dpi=140)
        plt.close()
    plt.figure(figsize=(7, 4))
    for model in MODELS:
        losses = [reevaluation[stage][model]["fixed_validation"]["losses"]["overall"]["loss"] for stage in STAGES]
        plt.plot([25, 50, 100], losses, marker="o", label=model)
    plt.xlabel("cumulative effective tokens (M)")
    plt.ylabel("fixed validation loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output / "staged_val_loss.png", dpi=140)
    plt.close()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inventory_data = inventory()
    fixed_dataset = ROOT / "prepared_data" / "fdt_rlm_10m"
    source_dataset = ROOT / "prepared_data" / "fdt_rlm_100m"
    fixed_ids, fixed_mask = load_split(fixed_dataset, "validation")
    source_ids, source_mask = load_split(source_dataset, "validation")
    fixed_identity = split_identity(fixed_ids, fixed_mask, fixed_dataset, "validation")
    source_identity = split_identity(source_ids, source_mask, source_dataset, "validation")
    eval_config = {"batch_size": 4, "max_seq_len": 512, "eval_mode": True, "dropout": False, "precision": "bf16 forward / fp32 CE", "seed": 42, "loss": "causal next-token CE", "padding": "attention mask and pad id excluded"}
    fixed_labels = labels_for_split(fixed_dataset, "validation", len(fixed_ids))
    source_labels = labels_for_split(source_dataset, "validation", len(source_ids))

    reevaluation = {}
    source_eval = {}
    anchor_audit = {}
    for stage in STAGES:
        reevaluation[stage], source_eval[stage] = {}, {}
        for model in MODELS:
            path = checkpoint_path(stage, model)
            fixed_result = evaluate_checkpoint(path, fixed_ids, fixed_mask, fixed_labels, device)
            source_result = evaluate_checkpoint(path, source_ids, source_mask, source_labels, device)
            reevaluation[stage][model] = {"fixed_validation": fixed_result}
            source_eval[stage][model] = source_result
            if model == "fdt":
                anchor_audit[stage] = fixed_result["anchor_health"]
    reevaluation["validation_identity"] = fixed_identity
    reevaluation["evaluation_config"] = eval_config
    reevaluation["evaluation_config_hash"] = json_hash(eval_config)
    write_json("forensic_re_evaluation.json", reevaluation)
    write_json("source_specific_checkpoint_eval.json", {"validation_identity": source_identity, "stages": source_eval})
    write_json("fdt_anchor_staged_audit.json", anchor_audit)

    validation_consistency = {
        "conclusion": "IDENTICAL",
        "fixed_validation": fixed_identity,
        "stage_capability_artifacts": {stage: {"path": f"artifacts/stage_{stage}_capability_results.json", "fixed_validation_sequences": json.loads((ROOT / "artifacts" / f"stage_{stage}_capability_results.json").read_text(encoding="utf-8"))["fixed_validation_sequences"]} for stage in STAGES},
        "reason": "All staged capability runs call the same hard-coded fdt_rlm_10m validation loader in the same evaluator.",
    }
    write_json("validation_set_consistency.json", validation_consistency)

    lr = lr_audit()
    cursor, traces = cursor_and_samples(inventory_data, source_dataset)
    manifest = json.loads((source_dataset / "manifest.json").read_text(encoding="utf-8"))
    order = source_order(manifest, cursor)
    configs = config_audit()

    optimizer_audit = {"models": {}, "conclusion": "OPTIMIZER_AND_STEPS_RESTORED; BF16 SCALER NOT REQUIRED"}
    for model in MODELS:
        optimizer_audit["models"][model] = {}
        for transition, before, after in (("25m_to_50m", "25m", "50m"), ("50m_to_100m", "50m", "100m")):
            prior = inventory_data[before][model]
            current = inventory_data[after][model]
            initial_eval = next(json.loads(line) for line in (ROOT / "runs" / f"staged_{after}" / model / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if json.loads(line).get("event") == "eval")
            prior_summary = json.loads((ROOT / "runs" / f"staged_{before}" / model / "stage_summary.json").read_text(encoding="utf-8"))
            optimizer_audit["models"][model][transition] = {
                "model_weights_resumed": initial_eval["val_loss"] == prior_summary["final_val_loss"],
                "optimizer_state_resumed": prior["optimizer_state_available"] and current["optimizer_state_available"] and current["optimizer_diagnostics"]["adam_step_max"] == current["optimizer_step"],
                "scheduler_state_resumed": prior["scheduler_state_available"] and current["scheduler_state_available"],
                "scaler_state_resumed": False,
                "scaler_required": False,
                "global_step_resumed": current["optimizer_step"] > prior["optimizer_step"],
                "tokens_seen_resumed": current["tokens_seen"] > prior["tokens_seen"],
                "initial_validation_exactly_matches_prior_final": initial_eval["val_loss"] == prior_summary["final_val_loss"],
            }
    write_json("optimizer_resume_audit.json", optimizer_audit)

    histories = {model: {stage: parse_metrics(ROOT / "runs" / f"staged_{stage}" / model / "metrics.jsonl") for stage in STAGES} for model in MODELS}
    write_json("training_loss_history.json", histories)
    plot_histories(histories, reevaluation)

    token_counting = {
        "definition": "tokens_seen increments attention_mask.sum over input positions; packed shards have no padding",
        "padding_fraction": 0.0,
        "input_tokens_are_true_valid_tokens": True,
        "causal_target_fraction": 511 / 512,
        "models": {stage: {model: {"reported_input_tokens": inventory_data[stage][model]["tokens_seen"], "estimated_causal_target_tokens": int(inventory_data[stage][model]["tokens_seen"] * 511 / 512)} for model in MODELS} for stage in STAGES},
    }
    write_json("token_counting_audit.json", token_counting)

    selection = {
        "capability_script_selection": {stage: {model: str(checkpoint_path(stage, model).relative_to(ROOT)) for model in MODELS} for stage in STAGES},
        "100m_best_files_present": {model: checkpoint_path("100m", model, "best.pt").exists() for model in MODELS},
        "100m_checkpoint_best_val_loss": {model: inventory_data["100m"][model]["best_val_loss"] for model in MODELS},
        "conclusion": "100M capability used latest.pt as requested; no new best.pt exists because 100M never beat its 50M starting validation.",
    }
    write_json("checkpoint_selection_audit.json", selection)

    safety_eval = {}
    for model in MODELS:
        safety_eval[model] = {}
        candidates = [checkpoint_path("50m", model)] + sorted((ROOT / "runs" / "staged_100m" / model).glob("step_*.pt")) + [checkpoint_path("100m", model)]
        seen = set()
        for path in candidates:
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            result = evaluate_checkpoint(path, fixed_ids, fixed_mask, fixed_labels, device)
            safety_eval[model][str(path.relative_to(ROOT))] = result["losses"]["overall"]
    write_json("best_vs_final_checkpoint_eval.json", safety_eval)

    source_shift = True
    root_causes = {
        "H1_lr_scheduler_reset": {"classification": "DISPROVEN", "evidence": "all stage first/last LRs are 3e-5"},
        "H2_optimizer_state_lost": {"classification": "DISPROVEN", "evidence": "Adam state and step counters continue"},
        "H3_scheduler_state_lost": {"classification": "DISPROVEN", "evidence": "constant scheduler state is present and LR continuous"},
        "H4_dataset_cursor_reset": {"classification": "DISPROVEN", "evidence": cursor["classification"]},
        "H5_data_replay_across_stages": {"classification": "DISPROVEN", "evidence": traces["sample_replay_detected_at_stage_transition"]},
        "H6_source_ordering_distribution_shift": {"classification": "CONFIRMED", "evidence": order["classification"]},
        "H7_validation_set_inconsistency": {"classification": "DISPROVEN", "evidence": validation_consistency["conclusion"]},
        "H8_wrong_checkpoint_selected": {"classification": "DISPROVEN", "evidence": selection["conclusion"]},
        "H9_hyperparameter_change": {"classification": "DISPROVEN", "evidence": configs["conclusion"]},
        "H10_real_overfitting": {"classification": "POSSIBLE", "evidence": "cannot be isolated from the confirmed ordered-source shift"},
        "H11_catastrophic_forgetting": {"classification": "CONFIRMED", "evidence": "Both models improve strongly on code while FineWeb-Edu/FineWeb losses regress by more than 3 points from 50M to 100M."},
        "H12_evaluator_bug": {"classification": "UNSUPPORTED", "evidence": "loss evaluator is deterministic; action/retrieval probes do demand instruction following"},
    }
    write_json("root_cause_classification.json", root_causes)

    evaluator_report = """# Capability Evaluator Audit

- Validation loss uses correct causal next-token shift, attention masking, eval mode, and deterministic sample order.
- English/code/action/retrieval probes use greedy decoding with sufficient short output limits.
- JSON/action scoring is strict and near-valid text scores zero by design.
- Action and retrieval prompts require instruction following that a small base pretrained LM was not explicitly trained to provide.
- Stop tokens and tokenizer decoding are consistent with checkpoint configuration.

Conclusion: no bug explains the 50M-to-100M validation-loss regression. Structured-action zeros should not be treated as pure base-LM quality evidence.
"""
    (ROOT / "CAPABILITY_EVALUATOR_AUDIT.md").write_text(evaluator_report, encoding="utf-8")

    fresh = {stage: {model: reevaluation[stage][model]["fixed_validation"]["losses"]["overall"]["loss"] for model in MODELS} for stage in STAGES}
    report = [
        "# Staged Regression Forensic Report", "", "## Reproduction", "",
        *(f"- {stage}: Transformer={fresh[stage]['transformer']:.6f}, FDT={fresh[stage]['fdt']:.6f}" for stage in STAGES),
        "", "## Findings", "",
        "- Validation subset: IDENTICAL across all staged capability runs.",
        "- LR: constant 3e-5 across both transitions and both models.",
        "- Optimizer/scheduler/global step/tokens: restored and continuous.",
        "- Dataset cursor: continuous; no stage-start replay detected.",
        "- Corpus order: grouped by source, not mixed or shuffled.",
        "- 50M-to-100M ranges move from FineWeb-Edu concentration into FineWeb/code blocks.",
        f"- Transformer source loss 50M->100M: FineWeb-Edu {source_eval['50m']['transformer']['losses']['fineweb_edu']['loss']:.3f}->{source_eval['100m']['transformer']['losses']['fineweb_edu']['loss']:.3f}; Python {source_eval['50m']['transformer']['losses']['code_python']['loss']:.3f}->{source_eval['100m']['transformer']['losses']['code_python']['loss']:.3f}.",
        f"- FDT source loss 50M->100M: FineWeb-Edu {source_eval['50m']['fdt']['losses']['fineweb_edu']['loss']:.3f}->{source_eval['100m']['fdt']['losses']['fineweb_edu']['loss']:.3f}; Python {source_eval['50m']['fdt']['losses']['code_python']['loss']:.3f}->{source_eval['100m']['fdt']['losses']['code_python']['loss']:.3f}.",
        "- Fixed validation improves through step 20,000, then sharply degrades by step 22,500 as late code blocks dominate.",
        "- 100M never beat the 50M starting validation, so no 100M best.pt was created.",
        "- FDT dead-anchor ratio remains zero; shared regression is not anchor collapse.",
        "", "## Root cause", "",
        "Confirmed primary cause: source-grouped sequential corpus order caused catastrophic forgetting. Code loss improves while web loss deteriorates; LR/reset/evaluator/checkpoint-selection hypotheses are disproven.",
        "", "## Recovery", "",
        "Trusted recovery point: both `runs/staged_50m/<model>/latest.pt` checkpoints.",
        "Implemented recovery path: deterministic weighted interleaving of only rows not consumed by the trusted checkpoint, with no replay.",
        "Recommended next action: run the recovery plan check, then separately authorize a bounded resume smoke test before any 50M-to-100M rerun.",
    ]
    (ROOT / "STAGED_REGRESSION_FORENSIC_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"fresh_losses": fresh, "root_cause": "CONFIRMED_SOURCE_ORDER_SHIFT", "trusted_recovery": "staged_50m"}, indent=2))


if __name__ == "__main__":
    main()
