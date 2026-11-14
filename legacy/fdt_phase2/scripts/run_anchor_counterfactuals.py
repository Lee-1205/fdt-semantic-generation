from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model
from fdt_rlm.models.causal_lm import CausalFuzzyAnchorLayer
from fdt_rlm.tokenization import load_tokenizer


MODES = (
    "learned",
    "uniform_topk",
    "shuffle_membership",
    "onehot_top1",
    "random_topk",
    "zero_anchor",
)
CHECKPOINTS = {
    "pre_repair": ROOT / "runs" / "full_fdt" / "pre_entropy_repair_step_2444.pt",
    "best": ROOT / "runs" / "full_fdt" / "best.pt",
    "latest": ROOT / "runs" / "full_fdt" / "latest.pt",
}


def safe_load(path: Path, device: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_model(path: Path, device: torch.device):
    checkpoint = safe_load(path, device)
    config = ModelConfig(**checkpoint["model_config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config, checkpoint


def set_mode(model: torch.nn.Module, mode: str) -> None:
    for module in model.modules():
        if isinstance(module, CausalFuzzyAnchorLayer):
            module.set_probe_mode(mode)


def load_validation(limit: int) -> Tuple[torch.Tensor, torch.Tensor]:
    shard = safe_load(
        ROOT / "prepared_data" / "fdt_rlm_10m" / "shards" / "validation" / "shard_000000.pt"
    )
    ids = shard["input_ids"].long()
    mask = shard.get("attention_mask", torch.ones_like(ids)).long()
    if limit > 0:
        ids = ids[:limit]
        mask = mask[:limit]
    return ids, mask


def token_loss(logits: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor, pad_id: int):
    labels = ids[:, 1:]
    valid = mask[:, 1:].bool() & (labels != pad_id)
    losses = F.cross_entropy(
        logits[:, :-1].float().transpose(1, 2),
        labels,
        reduction="none",
    )
    return (losses * valid).sum(), valid


def empty_total() -> Dict[str, float]:
    return {
        "loss_sum": 0.0,
        "tokens": 0.0,
        "kl_sum": 0.0,
        "abs_logit_sum": 0.0,
        "logit_elements": 0.0,
        "hidden_mse_sum": 0.0,
        "hidden_cosine_sum": 0.0,
        "positions": 0.0,
    }


@torch.inference_mode()
def run_counterfactuals(
    model: torch.nn.Module,
    config: ModelConfig,
    ids: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    totals = {mode: empty_total() for mode in MODES}
    for row_ids, row_mask in zip(ids, masks):
        batch_ids = row_ids.unsqueeze(0).to(device)
        batch_mask = row_mask.unsqueeze(0).to(device)
        set_mode(model, "learned")
        base = model(input_ids=batch_ids, attention_mask=batch_mask)
        base_loss, valid_next = token_loss(
            base["logits"], batch_ids, batch_mask, config.pad_token_id
        )
        valid_hidden = batch_mask.bool()
        totals["learned"]["loss_sum"] += float(base_loss.item())
        totals["learned"]["tokens"] += float(valid_next.sum().item())
        base_logits = base["logits"][:, :-1].float()
        base_logp = F.log_softmax(base_logits, dim=-1)
        base_prob = base_logp.exp()

        for mode in MODES[1:]:
            set_mode(model, mode)
            variant = model(input_ids=batch_ids, attention_mask=batch_mask)
            loss, valid = token_loss(
                variant["logits"], batch_ids, batch_mask, config.pad_token_id
            )
            variant_logits = variant["logits"][:, :-1].float()
            variant_logp = F.log_softmax(variant_logits, dim=-1)
            kl = (base_prob * (base_logp - variant_logp)).sum(dim=-1)[valid]
            abs_diff = (base_logits - variant_logits).abs()[valid]
            hidden_mse = (
                (base["hidden"].float() - variant["hidden"].float())
                .square()
                .mean(dim=-1)[valid_hidden]
            )
            hidden_cos = F.cosine_similarity(
                base["hidden"].float(), variant["hidden"].float(), dim=-1
            )[valid_hidden]
            total = totals[mode]
            total["loss_sum"] += float(loss.item())
            total["tokens"] += float(valid.sum().item())
            total["kl_sum"] += float(kl.sum().item())
            total["abs_logit_sum"] += float(abs_diff.sum().item())
            total["logit_elements"] += float(abs_diff.numel())
            total["hidden_mse_sum"] += float(hidden_mse.sum().item())
            total["hidden_cosine_sum"] += float(hidden_cos.sum().item())
            total["positions"] += float(valid_hidden.sum().item())

    set_mode(model, "learned")
    learned_loss = totals["learned"]["loss_sum"] / max(totals["learned"]["tokens"], 1.0)
    report: Dict[str, Any] = {}
    for mode, total in totals.items():
        loss = total["loss_sum"] / max(total["tokens"], 1.0)
        positions = max(total["positions"], 1.0)
        report[mode] = {
            "validation_loss": loss,
            "perplexity": math.exp(loss) if loss < 80 else float("inf"),
            "loss_delta_vs_learned": loss - learned_loss,
            "mean_logit_kl_vs_learned": total["kl_sum"] / positions if mode != "learned" else 0.0,
            "mean_absolute_logit_difference": total["abs_logit_sum"] / max(total["logit_elements"], 1.0),
            "mean_hidden_mse_vs_learned": total["hidden_mse_sum"] / positions if mode != "learned" else 0.0,
            "mean_hidden_cosine_with_learned": total["hidden_cosine_sum"] / positions if mode != "learned" else 1.0,
            "tokens": int(total["tokens"]),
        }
    return report


@torch.inference_mode()
def checkpoint_routing_metrics(
    model: torch.nn.Module,
    config: ModelConfig,
    ids: torch.Tensor,
    masks: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    loss_sum = 0.0
    token_count = 0
    entropy_sum = 0.0
    top1_sum = 0.0
    margin_sum = 0.0
    routing_tokens = 0
    usage = torch.zeros(config.num_anchors, dtype=torch.float64)
    set_mode(model, "learned")
    for row_ids, row_mask in zip(ids, masks):
        batch_ids = row_ids.unsqueeze(0).to(device)
        batch_mask = row_mask.unsqueeze(0).to(device)
        out = model(input_ids=batch_ids, attention_mask=batch_mask)
        loss, valid = token_loss(out["logits"], batch_ids, batch_mask, config.pad_token_id)
        loss_sum += float(loss.item())
        token_count += int(valid.sum().item())
        valid_tokens = batch_mask.bool()
        for stats in out["anchor_stats"]:
            membership = stats.membership.float()
            entropy = -(membership * membership.clamp_min(1e-8).log()).sum(dim=-1)
            entropy_sum += float(entropy[valid_tokens].sum().item())
            top1_sum += float(membership[..., 0][valid_tokens].sum().item())
            margin_sum += float(
                (membership[..., 0] - membership[..., 1])[valid_tokens].sum().item()
            )
            routing_tokens += int(valid_tokens.sum().item())
            usage += stats.usage_prob.detach().double().cpu()
    loss = loss_sum / max(token_count, 1)
    mean_entropy = entropy_sum / max(routing_tokens, 1)
    usage = usage / usage.sum().clamp_min(1e-12)
    return {
        "validation_loss": loss,
        "perplexity": math.exp(loss) if loss < 80 else float("inf"),
        "entropy": mean_entropy,
        "normalized_entropy": mean_entropy / math.log(max(config.top_k, 2)),
        "effective_anchor_count": math.exp(mean_entropy),
        "top1_membership": top1_sum / max(routing_tokens, 1),
        "top1_top2_margin": margin_sum / max(routing_tokens, 1),
        "dead_anchor_ratio": float((usage < 1e-4).double().mean().item()),
        "membership_mass_share_max": float(usage.max().item()),
        "tokens": token_count,
    }


@torch.inference_mode()
def greedy_tokens(
    model: torch.nn.Module,
    config: ModelConfig,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
) -> List[int]:
    ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    generated: List[int] = []
    for _ in range(max_new_tokens):
        context = ids[-config.max_seq_len :]
        input_ids = torch.tensor([context], device=device, dtype=torch.long)
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        next_id = int(out["logits"][0, -1].argmax().item())
        if next_id in {config.eos_token_id, config.pad_token_id}:
            break
        ids.append(next_id)
        generated.append(next_id)
    return generated


def generation_counterfactuals(
    model: torch.nn.Module,
    config: ModelConfig,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> Dict[str, Any]:
    prompts = [
        "The main reason neural networks require optimization is",
        "def calculate_average(values):\n",
        '{"action":"SEARCH","query":',
    ]
    rows = []
    for prompt in prompts:
        mode_tokens: Dict[str, List[int]] = {}
        for mode in MODES:
            set_mode(model, mode)
            mode_tokens[mode] = greedy_tokens(
                model, config, tokenizer, prompt, device, max_new_tokens
            )
        learned = mode_tokens["learned"]
        for mode in MODES:
            candidate = mode_tokens[mode]
            overlap = sum(a == b for a, b in zip(learned, candidate))
            rows.append(
                {
                    "prompt": prompt,
                    "mode": mode,
                    "completion": tokenizer.decode(candidate, skip_special_tokens=True),
                    "exact_token_match": candidate == learned,
                    "position_token_agreement": overlap / max(len(learned), len(candidate), 1),
                    "sequence_similarity": difflib.SequenceMatcher(
                        a=learned, b=candidate
                    ).ratio(),
                }
            )
    set_mode(model, "learned")
    return {"max_new_tokens": max_new_tokens, "decoding": "greedy", "rows": rows}


def correlation(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    x = torch.tensor(list(xs), dtype=torch.float64)
    y = torch.tensor(list(ys), dtype=torch.float64)
    if x.numel() < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return None
    return float(torch.corrcoef(torch.stack([x, y]))[0, 1].item())


def write_report(payload: Dict[str, Any]) -> None:
    result = payload["counterfactuals"]
    comparison = payload["repair_comparison"]
    fuzzy_delta = result["uniform_topk"]["loss_delta_vs_learned"]
    selection_delta = result["random_topk"]["loss_delta_vs_learned"]
    disabled_delta = result["zero_anchor"]["loss_delta_vs_learned"]
    gate = "PASS" if max(selection_delta, disabled_delta) > 0.01 else "INCONCLUSIVE"
    fuzzy = "YES" if fuzzy_delta > 0.005 else "NO" if fuzzy_delta < -0.005 else "INCONCLUSIVE"
    lines = [
        "# Anchor Counterfactual Report",
        "",
        f"Checkpoint: `{payload['primary_checkpoint']['path']}`",
        f"Fixed validation sequences: {payload['fixed_subset']['sequences']}",
        "",
        "## Counterfactuals",
        "",
        "| Mode | Loss | Delta | Logit KL | Hidden MSE |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        row = result[mode]
        lines.append(
            f"| {mode} | {row['validation_loss']:.6f} | {row['loss_delta_vs_learned']:+.6f} | "
            f"{row['mean_logit_kl_vs_learned']:.6f} | {row['mean_hidden_mse_vs_learned']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Repair Comparison",
            "",
            "| Checkpoint | Loss | Entropy | Normalized | Effective K | Top-1 | Dead |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, row in comparison.items():
        lines.append(
            f"| {name} | {row['validation_loss']:.6f} | {row['entropy']:.6f} | "
            f"{row['normalized_entropy']:.6f} | {row['effective_anchor_count']:.4f} | "
            f"{row['top1_membership']:.4f} | {row['dead_anchor_ratio']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Gate A, anchor path matters: **{gate}**",
            f"- Learned fuzzy weights add value over uniform Top-K: **{fuzzy}**",
            "- This distinction prevents selection/output importance from being misreported as fuzzy-weight importance.",
            "- Correlations across three checkpoints are descriptive only and are not causal evidence.",
        ]
    )
    (ROOT / "ANCHOR_COUNTERFACTUAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINTS["best"]))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sequences", type=int, default=0, help="0 uses the full validation shard")
    parser.add_argument("--generation-tokens", type=int, default=16)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(20260711)
    ids, masks = load_validation(args.sequences)
    tokenizer = load_tokenizer(str(ROOT / "tokenizers" / "fdt_bpe"))
    primary_path = Path(args.checkpoint)
    model, config, _ = load_model(primary_path, device)
    counterfactuals = run_counterfactuals(model, config, ids, masks, device)
    generations = (
        {"status": "skipped"}
        if args.skip_generation
        else generation_counterfactuals(
            model, config, tokenizer, device, args.generation_tokens
        )
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    repair_comparison: Dict[str, Any] = {}
    checkpoint_meta: Dict[str, Any] = {}
    for name, path in CHECKPOINTS.items():
        model, checkpoint_config, checkpoint = load_model(path, device)
        repair_comparison[name] = checkpoint_routing_metrics(
            model, checkpoint_config, ids, masks, device
        )
        checkpoint_meta[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "optimizer_step": checkpoint.get("optimizer_step", checkpoint.get("step")),
            "tokens_seen": checkpoint.get("tokens_seen"),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ordered = [repair_comparison[name] for name in CHECKPOINTS]
    correlations = {
        "entropy_vs_validation_loss": correlation(
            [row["entropy"] for row in ordered],
            [row["validation_loss"] for row in ordered],
        ),
        "top1_vs_validation_loss": correlation(
            [row["top1_membership"] for row in ordered],
            [row["validation_loss"] for row in ordered],
        ),
        "effective_anchors_vs_validation_loss": correlation(
            [row["effective_anchor_count"] for row in ordered],
            [row["validation_loss"] for row in ordered],
        ),
        "warning": "Only three checkpoints; descriptive, not causal.",
    }
    payload = {
        "schema_version": 1,
        "primary_checkpoint": {
            "path": str(primary_path),
            "sha256": sha256(primary_path),
        },
        "fixed_subset": {
            "source": "prepared_data/fdt_rlm_10m validation shard",
            "selection": "first N in stored order; N=all when sequences=0",
            "sequences": int(ids.size(0)),
            "tokens": int(masks.sum().item()),
        },
        "counterfactuals": counterfactuals,
        "generation_counterfactuals": generations,
        "repair_comparison": repair_comparison,
        "checkpoint_metadata": checkpoint_meta,
        "correlations": correlations,
    }
    write_json(ROOT / "artifacts" / "anchor_counterfactual_results.json", payload)
    write_report(payload)
    print(
        json.dumps(
            {
                "sequences": int(ids.size(0)),
                "counterfactuals": counterfactuals,
                "repair_comparison": repair_comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
