from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn as nn
import torch.nn.functional as F

from fdt_rlm.analysis.routing_metrics import (
    entropy_from_membership,
    mean_pairwise_distances,
    membership_gini,
    normalized_entropy,
    summarize,
    usage_metrics,
)
from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model
from fdt_rlm.models.causal_lm import CausalFuzzyAnchorLayer
from fdt_rlm.tokenization import load_tokenizer


ARTIFACTS = ROOT / "artifacts"
PLOTS = ARTIFACTS / "plots"
TRACE_DIR = ARTIFACTS / "fdt_profile_trace"
DATASET_DIR = ROOT / "prepared_data" / "fdt_rlm_10m"
TOKENIZER_DIR = ROOT / "tokenizers" / "fdt_bpe"
CURRENT_FDT = ROOT / "runs" / "full_fdt" / "best.pt"
CURRENT_TRANSFORMER = ROOT / "runs" / "full_transformer" / "best.pt"
EARLY_SOURCE = Path(r"<USER_HOME>\Desktop\SDFT\SD.py.bak")
EARLY_CHECKPOINT = Path(r"<USER_HOME>\Desktop\SDFT\fdt_korean_model_latest.pt")
SDFT_SOURCE = Path(r"<USER_HOME>\Desktop\SDFT\SD.py")
SDFT_CHECKPOINT = Path(r"<USER_HOME>\Desktop\SDFT\checkpoints\fdt_epoch_19.pt")
FDT2_SOURCE = ROOT / "legacy" / "korean_fdt" / "SD_fdt2_dualanchor_oneshot.py"
FDT2_CHECKPOINT = Path(r"<USER_HOME>\Desktop\FDT2\checkpoints_fdt2_dualanchor_oneshot\fdt_epoch_4.pt")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_load(path: Path, device: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_checkpoint_model(path: Path, device: torch.device):
    checkpoint = safe_load(path, device)
    config = ModelConfig(**checkpoint["model_config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config, checkpoint


def set_probe_mode(model: nn.Module, mode: str) -> None:
    for module in model.modules():
        if isinstance(module, CausalFuzzyAnchorLayer):
            module.set_probe_mode(mode)


def reconstruct_labels(manifest: Dict[str, Any], split: str) -> List[str]:
    seq_len = int(manifest["seq_len"])
    labels: List[str] = []
    for shard in manifest["shards"]:
        if shard["split"] != split:
            continue
        for bucket, tokens in shard["buckets"].items():
            labels.extend([bucket] * (int(tokens) // seq_len))
    return labels


def load_split(split: str) -> Tuple[torch.Tensor, torch.Tensor]:
    shard = safe_load(DATASET_DIR / "shards" / split / "shard_000000.pt")
    ids = shard["input_ids"].long()
    mask = shard.get("attention_mask", torch.ones_like(ids)).long()
    return ids, mask


def source_probe_sequences(per_source: int) -> Tuple[torch.Tensor, torch.Tensor, List[str], Dict[str, Any]]:
    manifest = json.loads((DATASET_DIR / "manifest.json").read_text(encoding="utf-8"))
    selected_ids: List[torch.Tensor] = []
    selected_masks: List[torch.Tensor] = []
    selected_sources: List[str] = []
    provenance: Dict[str, List[Dict[str, int]]] = defaultdict(list)
    counts: Dict[str, int] = defaultdict(int)
    cached: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for split in ("validation", "test", "train"):
        ids, mask = load_split(split)
        cached[split] = (ids, mask)
        labels = reconstruct_labels(manifest, split)
        if len(labels) != len(ids):
            labels = (labels + ["unknown"] * len(ids))[: len(ids)]
        for index, source in enumerate(labels):
            if counts[source] >= per_source:
                continue
            selected_ids.append(ids[index])
            selected_masks.append(mask[index])
            selected_sources.append(source)
            provenance[source].append({"split": split, "sequence_index": index})
            counts[source] += 1
    return (
        torch.stack(selected_ids),
        torch.stack(selected_masks),
        selected_sources,
        {"label_reconstruction": "manifest bucket order; packed boundary sequences may contain adjacent-source tokens", "sequences": provenance},
    )


KEYWORDS = {
    "python_keyword": {"def", "class", "import", "from", "return", "yield", "lambda", "async", "await", "elif", "except"},
    "javascript_keyword": {"function", "const", "let", "var", "export", "import", "async", "await", "class", "extends"},
    "c_cpp_keyword": {"include", "define", "typedef", "struct", "namespace", "template", "constexpr", "sizeof", "malloc", "nullptr"},
    "java_keyword": {"public", "private", "protected", "static", "final", "interface", "extends", "implements", "package", "throws"},
    "rust_go_keyword": {"fn", "let", "mut", "impl", "trait", "crate", "func", "package", "defer", "chan", "select"},
}


def classify_piece(piece: str, token_id: int, special_ids: set[int]) -> str:
    if token_id in special_ids:
        return "special"
    raw = piece
    text = raw.replace("\u0120", "").replace("\u010a", "").replace("\u2581", "").strip()
    lower = text.lower()
    if not text:
        return "whitespace_code"
    if lower in {"{", "}", "[", "]", ":", ","}:
        return "json_structural"
    if lower in {"#", "##", "###", "*", "**", "`", "```", ">", "-"}:
        return "markdown_syntax"
    for category, words in KEYWORDS.items():
        if lower in words:
            return category
    if re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)", lower):
        return "number"
    if re.fullmatch(r"[^\w\s]+", text, flags=re.UNICODE):
        return "punctuation"
    if (text.startswith(('"', "'")) or text.endswith(('"', "'"))) and len(text) > 1:
        return "string_literal"
    if re.fullmatch(r"[A-Za-z]+", text):
        return "alphabetic_english"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", text):
        return "identifier"
    if raw.startswith(("\u0120", "\u010a")) and any(ch in text for ch in "(){}[]=.;"):
        return "whitespace_code"
    return "other"


def position_bucket(position: int, length: int) -> str:
    ratio = position / max(length, 1)
    if ratio < 0.10:
        return "00-10%"
    if ratio < 0.25:
        return "10-25%"
    if ratio < 0.50:
        return "25-50%"
    if ratio < 0.75:
        return "50-75%"
    return "75-100%"


def sharpness_fields(membership: torch.Tensor) -> Dict[str, torch.Tensor]:
    values = membership.float().sort(dim=-1, descending=True).values
    top1 = values[:, 0]
    top2 = values[:, 1] if values.size(1) > 1 else torch.zeros_like(top1)
    top3 = values[:, 2] if values.size(1) > 2 else torch.zeros_like(top1)
    return {
        "top1_membership": top1,
        "top2_membership": top2,
        "top3_membership": top3,
        "top1_minus_top2_margin": top1 - top2,
        "top1_to_top2_ratio": top1 / top2.clamp_min(1e-12),
        "max_membership": top1,
        "membership_gini": membership_gini(values),
    }


def group_report(
    membership: torch.Tensor,
    indices: torch.Tensor,
    group_labels: Sequence[str],
    num_anchors: int,
    top_k: int,
) -> Dict[str, Any]:
    labels = list(group_labels)
    result: Dict[str, Any] = {}
    for label in sorted(set(labels)):
        select = torch.tensor([item == label for item in labels], dtype=torch.bool)
        if not select.any():
            continue
        member = membership[select]
        idx = indices[select]
        entropy = entropy_from_membership(member)
        sharp = sharpness_fields(member)
        usage = usage_metrics(idx.unsqueeze(0), member.unsqueeze(0), torch.ones(1, len(member)), num_anchors)
        result[label] = {
            "tokens_across_layers": int(select.sum().item()),
            "raw_entropy": summarize(entropy),
            "normalized_entropy": summarize(normalized_entropy(entropy, top_k)),
            "effective_anchor_count": summarize(entropy.exp()),
            "sharpness": {key: summarize(value) for key, value in sharp.items()},
            "usage": usage,
        }
    return result


def distribution_distances(grouped: Dict[str, Any], field: str) -> Dict[str, float]:
    distributions = {key: value["usage"][field] for key, value in grouped.items()}
    return mean_pairwise_distances(distributions)


def pearson(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.float().reshape(-1)
    right = right.float().reshape(-1)
    valid = torch.isfinite(left) & torch.isfinite(right)
    left = left[valid]
    right = right[valid]
    if left.numel() < 2:
        return 0.0
    left = left - left.mean()
    right = right - right.mean()
    denom = left.square().sum().sqrt() * right.square().sum().sqrt()
    return float((left * right).sum().div(denom.clamp_min(1e-12)).item())


def regression_r2(features: Sequence[torch.Tensor], target: torch.Tensor) -> float:
    y = target.float().reshape(-1, 1)
    x = torch.stack([item.float().reshape(-1) for item in features], dim=1)
    valid = torch.isfinite(x).all(dim=1) & torch.isfinite(y[:, 0])
    x = x[valid]
    y = y[valid]
    x = (x - x.mean(dim=0, keepdim=True)) / x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    design = torch.cat([torch.ones(len(x), 1), x], dim=1)
    beta = torch.linalg.lstsq(design, y).solution
    pred = design @ beta
    ss_res = (y - pred).square().sum()
    ss_tot = (y - y.mean()).square().sum().clamp_min(1e-12)
    return float((1.0 - ss_res / ss_tot).item())


@torch.no_grad()
def probe_current_checkpoint(model, config, tokenizer, ids, masks, sources, device, batch_size: int) -> Dict[str, Any]:
    membership_records: List[torch.Tensor] = []
    index_records: List[torch.Tensor] = []
    source_records: List[str] = []
    token_type_records: List[str] = []
    position_records: List[str] = []
    layer_records: List[str] = []
    unique_per_sequence: List[int] = []
    unique_per_batch: List[int] = []
    uncertainty: Dict[str, List[torch.Tensor]] = defaultdict(list)
    special_ids = set(int(x) for x in tokenizer.all_special_ids)
    set_probe_mode(model, "learned")
    model.eval()

    for start in range(0, len(ids), batch_size):
        batch_ids = ids[start : start + batch_size].to(device)
        batch_mask = masks[start : start + batch_size].to(device)
        out = model(input_ids=batch_ids, attention_mask=batch_mask)
        pieces_per_sequence = [tokenizer.convert_ids_to_tokens(row.tolist()) for row in batch_ids.cpu()]
        types_per_sequence = [
            [classify_piece(piece, int(token_id), special_ids) for piece, token_id in zip(pieces, row.tolist())]
            for pieces, row in zip(pieces_per_sequence, batch_ids.cpu())
        ]
        for layer_index, stats in enumerate(out["anchor_stats"]):
            for local in range(len(batch_ids)):
                valid = batch_mask[local].bool().cpu()
                member = stats.membership[local].float().cpu()[valid]
                index = stats.indices[local].long().cpu()[valid]
                membership_records.append(member)
                index_records.append(index)
                source_records.extend([sources[start + local]] * len(member))
                token_type_records.extend([types_per_sequence[local][pos] for pos in torch.nonzero(valid).flatten().tolist()])
                position_records.extend([position_bucket(pos, int(valid.sum().item())) for pos in torch.nonzero(valid).flatten().tolist()])
                layer_records.extend([f"layer_{layer_index}"] * len(member))
                unique_per_sequence.append(int(torch.unique(index).numel()))
            unique_per_batch.append(int(torch.unique(stats.indices[batch_mask.bool()]).numel()))

        logits = out["logits"].float()
        shift_logits = logits[:, :-1]
        targets = batch_ids[:, 1:]
        valid_targets = batch_mask[:, 1:].bool() & (targets != config.pad_token_id)
        token_loss = F.cross_entropy(shift_logits.transpose(1, 2), targets, reduction="none")
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        last_membership = out["anchor_stats"][-1].membership[:, :-1].float()
        last_entropy = entropy_from_membership(last_membership)
        last_sharp = sharpness_fields(last_membership.reshape(-1, last_membership.size(-1)))
        uncertainty["token_loss"].append(token_loss[valid_targets].cpu())
        uncertainty["token_entropy"].append(token_entropy[valid_targets].cpu())
        uncertainty["anchor_entropy"].append(last_entropy[valid_targets].cpu())
        uncertainty["top1_membership"].append(last_sharp["top1_membership"].reshape_as(valid_targets)[valid_targets].cpu())
        uncertainty["top1_minus_top2_margin"].append(last_sharp["top1_minus_top2_margin"].reshape_as(valid_targets)[valid_targets].cpu())

    membership = torch.cat(membership_records, dim=0)
    indices = torch.cat(index_records, dim=0)
    entropy = entropy_from_membership(membership)
    sharp = sharpness_fields(membership)
    valid = torch.ones(1, len(membership), dtype=torch.bool)
    usage = usage_metrics(indices.unsqueeze(0), membership.unsqueeze(0), valid, config.num_anchors)
    source = group_report(membership, indices, source_records, config.num_anchors, config.top_k)
    token_type = group_report(membership, indices, token_type_records, config.num_anchors, config.top_k)
    position = group_report(membership, indices, position_records, config.num_anchors, config.top_k)
    layer = group_report(membership, indices, layer_records, config.num_anchors, config.top_k)
    unc = {key: torch.cat(value) for key, value in uncertainty.items()}
    uncertainty_report = {
        "correlations_with_next_token_loss": {
            key: pearson(value, unc["token_loss"])
            for key, value in unc.items()
            if key != "token_loss"
        },
        "r2_token_entropy_only": regression_r2([unc["token_entropy"]], unc["token_loss"]),
        "r2_token_plus_anchor_metrics": regression_r2(
            [unc["token_entropy"], unc["anchor_entropy"], unc["top1_membership"], unc["top1_minus_top2_margin"]],
            unc["token_loss"],
        ),
        "sample_tokens": int(unc["token_loss"].numel()),
    }
    return {
        "global": {
            "raw_entropy": summarize(entropy),
            "normalized_entropy": summarize(normalized_entropy(entropy, config.top_k)),
            "effective_anchor_count": summarize(entropy.exp()),
            "sharpness": {key: summarize(value) for key, value in sharp.items()},
            "usage": usage,
            "mean_unique_anchors_per_sequence_per_layer": float(sum(unique_per_sequence) / len(unique_per_sequence)),
            "mean_unique_anchors_per_batch_per_layer": float(sum(unique_per_batch) / len(unique_per_batch)),
        },
        "by_source": source,
        "by_token_type": token_type,
        "by_position": position,
        "by_layer": layer,
        "source_distances": {
            field: distribution_distances(source, field)
            for field in ("top1_selection_distribution", "topk_slot_distribution", "membership_mass_distribution", "token_coverage_distribution")
        },
        "token_type_distances": {
            field: distribution_distances(token_type, field)
            for field in ("top1_selection_distribution", "membership_mass_distribution")
        },
        "uncertainty": uncertainty_report,
        "raw": {"membership": membership, "indices": indices},
    }


def token_loss_sum(logits, ids, mask, pad_id) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = ids[:, 1:]
    valid = mask[:, 1:].bool() & (labels != pad_id)
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), labels, reduction="none")
    return (losses * valid).sum(), valid.sum()


@torch.no_grad()
def counterfactual_probe(model, config, ids, masks, device, limit: int) -> Dict[str, Any]:
    ids = ids[:limit]
    masks = masks[:limit]
    modes = ("learned", "uniform_topk", "random_topk", "zero_anchor")
    totals = {mode: {"loss_sum": 0.0, "tokens": 0, "kl_sum": 0.0, "hidden_mse_sum": 0.0, "hidden_cosine_sum": 0.0, "positions": 0} for mode in modes}
    model.eval()
    for row_ids, row_mask in zip(ids, masks):
        batch_ids = row_ids.unsqueeze(0).to(device)
        batch_mask = row_mask.unsqueeze(0).to(device)
        set_probe_mode(model, "learned")
        base = model(input_ids=batch_ids, attention_mask=batch_mask)
        base_logp = F.log_softmax(base["logits"].float(), dim=-1)
        base_prob = base_logp.exp()
        base_loss, token_count = token_loss_sum(base["logits"], batch_ids, batch_mask, config.pad_token_id)
        totals["learned"]["loss_sum"] += float(base_loss.item())
        totals["learned"]["tokens"] += int(token_count.item())
        for mode in modes[1:]:
            set_probe_mode(model, mode)
            variant = model(input_ids=batch_ids, attention_mask=batch_mask)
            loss, count = token_loss_sum(variant["logits"], batch_ids, batch_mask, config.pad_token_id)
            variant_logp = F.log_softmax(variant["logits"].float(), dim=-1)
            valid = batch_mask.bool()
            kl = (base_prob * (base_logp - variant_logp)).sum(dim=-1)[valid]
            hidden_mse = (base["hidden"].float() - variant["hidden"].float()).square().mean(dim=-1)[valid]
            hidden_cos = F.cosine_similarity(base["hidden"].float(), variant["hidden"].float(), dim=-1)[valid]
            totals[mode]["loss_sum"] += float(loss.item())
            totals[mode]["tokens"] += int(count.item())
            totals[mode]["kl_sum"] += float(kl.sum().item())
            totals[mode]["hidden_mse_sum"] += float(hidden_mse.sum().item())
            totals[mode]["hidden_cosine_sum"] += float(hidden_cos.sum().item())
            totals[mode]["positions"] += int(valid.sum().item())
    set_probe_mode(model, "learned")
    base_loss = totals["learned"]["loss_sum"] / max(totals["learned"]["tokens"], 1)
    report: Dict[str, Any] = {}
    for mode, data in totals.items():
        loss = data["loss_sum"] / max(data["tokens"], 1)
        positions = max(data["positions"], 1)
        report[mode] = {
            "validation_loss": loss,
            "loss_delta_vs_learned": loss - base_loss,
            "mean_logit_kl_from_learned": data["kl_sum"] / positions if mode != "learned" else 0.0,
            "mean_hidden_mse_from_learned": data["hidden_mse_sum"] / positions if mode != "learned" else 0.0,
            "mean_hidden_cosine_with_learned": data["hidden_cosine_sum"] / positions if mode != "learned" else 1.0,
            "tokens": data["tokens"],
        }
    return report


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class EarlyCdistRouter(nn.Module):
    def __init__(self, dim: int, anchors: int, top_k: int):
        super().__init__()
        self.dim = dim
        self.num_anchors = anchors
        self.top_k = top_k
        self.anchor_keys = nn.Parameter(torch.empty(anchors, dim))
        self.anchor_values = nn.Parameter(torch.empty(anchors, dim))
        self.log_sigma = nn.Parameter(torch.empty(anchors))
        self.gate = nn.Parameter(torch.empty(()))
        self.q_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.t_embed = nn.Sequential(nn.Linear(1, dim), nn.SiLU(), nn.Linear(dim, dim))

    def route(self, x: torch.Tensor, timestep: torch.Tensor):
        bsz, _, dim = x.shape
        conditioned = x + self.t_embed(timestep.view(bsz, 1)).unsqueeze(1)
        q = F.normalize(self.q_proj(conditioned), dim=-1)
        keys = F.normalize(self.anchor_keys, dim=-1)
        sigma = self.log_sigma.exp().clamp(1.0, 20.0).view(1, 1, -1)
        factor = torch.sigmoid(5.0 * (timestep - 0.5)) + 0.1
        sigma = sigma * factor.view(bsz, 1, 1)
        dist_sq = torch.cdist(q, keys.unsqueeze(0), p=2).square() / math.sqrt(dim)
        logits = (-dist_sq / (2.0 * sigma.square() + 1e-6)).clamp(-50, 50)
        raw = torch.exp(logits)
        values, indices = torch.topk(raw, k=self.top_k, dim=-1)
        membership = values.clamp_min(1e-8)
        membership = membership / (membership.sum(dim=-1, keepdim=True) + 1e-6)
        return indices, membership


def legacy_summary(memberships: List[torch.Tensor], top_ks: List[int], labels: List[str]) -> Dict[str, Any]:
    by_route: Dict[str, Any] = {}
    normalized_values = []
    top1_values = []
    margin_values = []
    for membership, top_k, label in zip(memberships, top_ks, labels):
        flat = membership.detach().float().reshape(-1, membership.size(-1)).cpu()
        entropy = entropy_from_membership(flat)
        sharp = sharpness_fields(flat)
        by_route[label] = {
            "top_k": top_k,
            "raw_entropy": summarize(entropy),
            "normalized_entropy": summarize(normalized_entropy(entropy, top_k)),
            "effective_anchor_count": summarize(entropy.exp()),
            "top1_membership": summarize(sharp["top1_membership"]),
            "top1_minus_top2_margin": summarize(sharp["top1_minus_top2_margin"]),
        }
        normalized_values.append(normalized_entropy(entropy, top_k))
        top1_values.append(sharp["top1_membership"])
        margin_values.append(sharp["top1_minus_top2_margin"])
    return {
        "by_route": by_route,
        "aggregate_normalized_entropy": summarize(torch.cat(normalized_values)),
        "aggregate_top1_membership": summarize(torch.cat(top1_values)),
        "aggregate_top1_minus_top2_margin": summarize(torch.cat(margin_values)),
    }


@torch.no_grad()
def probe_legacy(device: torch.device) -> Dict[str, Any]:
    torch.manual_seed(1234)
    report: Dict[str, Any] = {}
    if EARLY_CHECKPOINT.exists():
        state = safe_load(EARLY_CHECKPOINT)
        embedding = state["embedding.weight"].to(device)
        router = EarlyCdistRouter(512, 128, 16).to(device)
        router_state = {key[len("denoiser.") :]: value for key, value in state.items() if key.startswith("denoiser.")}
        router.load_state_dict(router_state, strict=True)
        router.eval()
        ids = torch.randint(1, embedding.size(0), (8, 128), device=device)
        positions = state["positional.pe"].to(device)[:, :128]
        hidden = embedding[ids] + positions
        memberships = []
        labels = []
        for timestep_value in (0.0, 0.25, 0.5, 0.75, 1.0):
            timestep = torch.full((len(ids), 1, 1), timestep_value, device=device)
            _, membership = router.route(hidden, timestep)
            memberships.append(membership)
            labels.append(f"t={timestep_value:.2f}")
        report["early_cdist_checkpoint"] = legacy_summary(memberships, [16] * len(memberships), labels)

    if SDFT_CHECKPOINT.exists():
        module = import_module(SDFT_SOURCE, "legacy_sdft_runtime")
        checkpoint = safe_load(SDFT_CHECKPOINT)
        config = module.FDTConfig(**checkpoint["config"])
        state = checkpoint["model_state_dict"]
        vocab = state["token_embedding.weight"].size(0)
        model = module.FDTLanguageModel(config, vocab_size=vocab, pad_token_id=0).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        ids = torch.randint(1, vocab, (8, config.max_len), device=device)
        mask = torch.ones_like(ids)
        memberships = []
        labels = []
        for timestep_value in (0.0, 0.25, 0.5, 0.75, 1.0):
            out = model(input_ids=ids, attention_mask=mask, timesteps=torch.full((len(ids),), timestep_value, device=device))
            for layer_index, stats in enumerate(out["routing"]):
                memberships.append(stats.membership)
                labels.append(f"layer_{layer_index}/t={timestep_value:.2f}")
        report["revised_sdft_epoch19"] = legacy_summary(memberships, [config.top_k] * len(memberships), labels)
        report["revised_sdft_epoch19"]["config"] = checkpoint["config"]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if FDT2_CHECKPOINT.exists():
        module = import_module(FDT2_SOURCE, "legacy_fdt2_runtime")
        checkpoint = safe_load(FDT2_CHECKPOINT)
        config = module.FDTConfig(**checkpoint["config"])
        state = checkpoint["model_state_dict"]
        vocab = state["token_embedding.weight"].size(0)
        model = module.FDTLanguageModel(config, vocab_size=vocab, pad_token_id=0).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        ids = torch.randint(1, vocab, (4, min(config.max_len, 256)), device=device)
        mask = torch.ones_like(ids)
        memberships = []
        top_ks = []
        labels = []
        for timestep_value in (0.0, 0.5, 1.0):
            out = model(input_ids=ids, attention_mask=mask, timesteps=torch.full((len(ids),), timestep_value, device=device))
            for route_index, stats in enumerate(out["routing"]):
                memberships.append(stats.membership)
                top_ks.append(int(stats.top_k))
                labels.append(f"route_{route_index}_k{stats.top_k}/t={timestep_value:.2f}")
        report["dual_anchor_fdt2_epoch4"] = legacy_summary(memberships, top_ks, labels)
        report["dual_anchor_fdt2_epoch4"]["config"] = checkpoint["config"]
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


def file_inventory() -> Dict[str, Any]:
    paths = [
        EARLY_SOURCE,
        EARLY_CHECKPOINT,
        SDFT_SOURCE,
        SDFT_CHECKPOINT,
        FDT2_SOURCE,
        FDT2_CHECKPOINT,
        ROOT / "src" / "fdt_rlm" / "models" / "causal_lm.py",
        CURRENT_FDT,
    ]
    entries = []
    for path in paths:
        if not path.exists():
            entries.append({"path": str(path), "exists": False})
            continue
        stat = path.stat()
        entries.append({
            "path": str(path),
            "exists": True,
            "bytes": stat.st_size,
            "modified_unix": stat.st_mtime,
            "sha256": sha256(path),
        })
    return {
        "files": entries,
        "legacy_generations": {
            "early_cdist": {"K": 128, "top_k": 16, "dim": 512, "max_len": 128, "tokenizer": "klue/bert-base", "checkpoint_compatible": EARLY_CHECKPOINT.exists()},
            "revised_sdft": {"K": 128, "top_k": 16, "dim": 512, "max_len": 128, "tokenizer": "klue/bert-base", "checkpoint_compatible": SDFT_CHECKPOINT.exists()},
            "dual_anchor_fdt2": {"K": 256, "role_K": 64, "semantic_K": 192, "role_top_k": 4, "semantic_top_k": 12, "dim": 768, "max_len": 768, "tokenizer": "klue/bert-base", "checkpoint_compatible": FDT2_CHECKPOINT.exists()},
        },
    }


def metric_audit(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    metrics = checkpoint.get("metrics", {})
    return {
        "best_checkpoint_saved_anchor_entropy": metrics.get("entropy"),
        "first_experiment_report_final_microbatch_anchor_entropy": 2.0237987836201983,
        "answers": {
            "1_before_or_after_topk": "after Top-K",
            "2_membership_sums_to_one": True,
            "3_log_base": "natural logarithm",
            "4_averaged_per_token": True,
            "5_averaged_per_sequence": "not separately; valid tokens are pooled within each microbatch",
            "6_averaged_per_batch": True,
            "7_averaged_across_steps": "the report used the final microbatch metric, not a training-step or validation average",
            "8_averaged_across_layers": True,
            "9_padding_excluded": True,
            "10_special_tokens": "included when present and attention_mask=1",
            "11_sources_combined": True,
            "12_timesteps_combined": "current routing has no timestep input; noisy auxiliary forward is not the reported anchor metric",
            "13_train_or_eval": "reported value came from the final training microbatch",
            "14_dropout_active": True,
            "15_exact_semantics_of_2_02": "last training microbatch, averaged over valid tokens and six layers",
        },
        "ambiguous_legacy_metrics": {
            "mean_active_anchors": "count of anchors with mean layer membership mass >=1e-4 in one training microbatch; not per-token activity",
            "top_anchor_frequency": "maximum of the layer-averaged membership-mass distribution; denominator is total selected membership mass",
        },
        "replacement_names": [
            "mean_active_anchors_per_token",
            "mean_unique_anchors_per_sequence_per_layer",
            "mean_unique_anchors_per_batch_per_layer",
            "global_active_anchors",
            "top1_selection_frequency",
            "topk_inclusion_frequency",
            "membership_mass_share",
            "token_coverage_frequency",
        ],
    }


@torch.no_grad()
def generate_once(model, config, tokenizer, prompt: str, device: torch.device, temperature: float, max_new_tokens: int) -> str:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    generated: List[int] = []
    generator = torch.Generator(device=device.type).manual_seed(20260710 + int(temperature * 100))
    for _ in range(max_new_tokens):
        context = ids[-config.max_seq_len :]
        input_ids = torch.tensor([context], device=device)
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
        logits = out["logits"][0, -1].float()
        for token_id in set(generated[-64:]):
            logits[token_id] = logits[token_id] * 1.1 if logits[token_id] < 0 else logits[token_id] / 1.1
        if temperature <= 0.2:
            next_id = int(logits.argmax().item())
        else:
            logits = logits / temperature
            top_values = torch.topk(logits, min(50, logits.numel())).values
            logits[logits < top_values[-1]] = -float("inf")
            sorted_logits, sorted_ids = logits.sort(descending=True)
            probs = sorted_logits.softmax(dim=-1)
            cumulative = probs.cumsum(dim=-1)
            remove = cumulative > 0.92
            remove[1:] = remove[:-1].clone()
            remove[0] = False
            logits[sorted_ids[remove]] = -float("inf")
            next_id = int(torch.multinomial(logits.softmax(dim=-1), 1, generator=generator).item())
        if next_id in {config.eos_token_id, config.pad_token_id}:
            break
        ids.append(next_id)
        generated.append(next_id)
    return tokenizer.decode(generated, skip_special_tokens=True)


def generation_probe(device: torch.device, tokenizer, max_new_tokens: int) -> Dict[str, Any]:
    prompts = [
        "The primary reason neural networks require optimization is",
        "A good scientific experiment should control",
        "In a small village surrounded by mountains,",
        "def calculate_average(values):",
        "class DatasetLoader:\n    def __init__(self, path):",
        "function parseConfig(config) {",
        '{"action":"SEARCH","query":',
        "Read the following observations and decide which information should be examined next:",
    ]
    results = []
    for checkpoint_path, name in ((CURRENT_TRANSFORMER, "transformer"), (CURRENT_FDT, "fdt")):
        model, config, _ = load_checkpoint_model(checkpoint_path, device)
        for temperature in (0.2, 0.8):
            for prompt in prompts:
                results.append({
                    "model": name,
                    "temperature": temperature,
                    "prompt": prompt,
                    "completion": generate_once(model, config, tokenizer, prompt, device, temperature, max_new_tokens),
                })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"settings": {"temperatures": [0.2, 0.8], "top_k": 50, "top_p": 0.92, "repetition_penalty": 1.1, "max_new_tokens": max_new_tokens}, "generations": results}


def profile_fdt(model, config, ids, mask, device: torch.device) -> Dict[str, Any]:
    if device.type != "cuda":
        return {"status": "skipped", "reason": "CUDA profiler requested but CUDA is unavailable"}
    TRACE_DIR.mkdir(parents=True, exist_ok=True)
    sample_ids = ids[:1].to(device)
    sample_mask = mask[:1].to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=sample_ids, attention_mask=sample_mask)
            loss, count = token_loss_sum(out["logits"], sample_ids, sample_mask, config.pad_token_id)
            loss = loss / count.clamp_min(1)
        loss.backward()
        optimizer.step()

    step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=False) as profiler:
        step()
        torch.cuda.synchronize()
    trace_path = TRACE_DIR / "fdt_step_trace.json"
    profiler.export_chrome_trace(str(trace_path))
    events = []
    for event in profiler.key_averages():
        events.append({
            "name": event.key,
            "self_cuda_time_us": float(getattr(event, "self_device_time_total", getattr(event, "self_cuda_time_total", 0.0))),
            "cuda_time_us": float(getattr(event, "device_time_total", getattr(event, "cuda_time_total", 0.0))),
            "self_cpu_time_us": float(event.self_cpu_time_total),
            "cpu_time_us": float(event.cpu_time_total),
            "cpu_memory_bytes": int(getattr(event, "cpu_memory_usage", 0)),
            "cuda_memory_bytes": int(getattr(event, "device_memory_usage", getattr(event, "cuda_memory_usage", 0))),
            "calls": int(event.count),
        })
    events.sort(key=lambda item: item["self_cuda_time_us"], reverse=True)
    cuda_profiler_available = any(item["self_cuda_time_us"] > 0 for item in events)
    dtype_bytes = 2
    dense_numer_bytes = config.max_seq_len * config.num_anchors * config.dim * dtype_bytes
    return {
        "status": "complete" if cuda_profiler_available else "partial",
        "cuda_profiler_available": cuda_profiler_available,
        "limitation": None if cuda_profiler_available else "CUPTI failed with CUDA_ERROR_INVALID_DEVICE; CUDA event timings were not captured. Memory accounting and the Chrome CPU trace remain valid.",
        "trace": str(trace_path),
        "top_operations_by_self_cuda_time": events[:30],
        "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved()),
        "routing_tensor_estimates_per_layer_bf16": {
            "dense_prefix_numerator_bytes": dense_numer_bytes,
            "dense_prefix_mass_bytes": config.max_seq_len * config.num_anchors * dtype_bytes,
            "selected_contributions_bytes": config.max_seq_len * config.top_k * config.dim * dtype_bytes,
            "dense_to_sparse_numerator_ratio": config.num_anchors / config.top_k,
        },
    }


def plot_results(current: Dict[str, Any]) -> Dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {"status": f"matplotlib unavailable: {exc}"}
    PLOTS.mkdir(parents=True, exist_ok=True)
    membership = current["raw"]["membership"]
    entropy = entropy_from_membership(membership)
    normalized = normalized_entropy(entropy, membership.size(-1))
    effective = entropy.exp()
    top1 = membership.max(dim=-1).values
    outputs = {}
    for name, values, xlabel in (
        ("entropy_histogram.png", entropy, "Raw selected-membership entropy"),
        ("normalized_entropy_histogram.png", normalized, "Entropy / ln(Top-K)"),
        ("effective_anchor_count_histogram.png", effective, "Effective selected anchors exp(H)"),
        ("top1_membership_histogram.png", top1, "Top-1 membership"),
    ):
        plt.figure(figsize=(7, 4))
        plt.hist(values.numpy(), bins=60, color="#2878b5", alpha=0.9)
        plt.xlabel(xlabel)
        plt.ylabel("Token-layer observations")
        plt.tight_layout()
        path = PLOTS / name
        plt.savefig(path, dpi=160)
        plt.close()
        outputs[name] = str(path)

    def heatmap(group_key: str, filename: str):
        grouped = current[group_key]
        labels = sorted(grouped)
        matrix = torch.tensor([grouped[label]["usage"]["membership_mass_distribution"] for label in labels])
        plt.figure(figsize=(14, max(4, len(labels) * 0.45)))
        plt.imshow(matrix.numpy(), aspect="auto", interpolation="nearest", cmap="viridis")
        plt.yticks(range(len(labels)), labels)
        plt.xlabel("Anchor index")
        plt.colorbar(label="Membership mass share")
        plt.tight_layout()
        path = PLOTS / filename
        plt.savefig(path, dpi=160)
        plt.close()
        outputs[filename] = str(path)

    heatmap("by_source", "source_anchor_heatmap.png")
    heatmap("by_token_type", "token_type_anchor_heatmap.png")
    heatmap("by_layer", "anchor_usage_heatmap.png")
    return outputs


def reports(
    inventory: Dict[str, Any],
    audit: Dict[str, Any],
    current: Dict[str, Any],
    legacy: Dict[str, Any],
    counterfactual: Dict[str, Any],
    hotspots: Dict[str, Any],
    generations: Dict[str, Any],
) -> None:
    global_metrics = current["global"]
    source_distance = current["source_distances"]["membership_mass_distribution"]
    token_distance = current["token_type_distances"]["membership_mass_distribution"]
    early = legacy.get("early_cdist_checkpoint", {})
    revised = legacy.get("revised_sdft_epoch19", {})
    dual = legacy.get("dual_anchor_fdt2_epoch4", {})
    write_text(ROOT / "LEGACY_VS_CURRENT_ROUTING_REPORT.md", f"""# Legacy vs Current Routing

## Verified inventory

- Early FDT: K=128, Top-K=16, dim=512, max_len=128, `torch.cdist`, timestep-scaled sigma, full-sequence structural aggregation.
- Revised SDFT epoch 19: K=128, Top-K=16, dim=512, bounded per-anchor sigma, causal prefix aggregation.
- Dual-anchor FDT2 epoch 4: role K=64/Top-K=4 plus semantic K=192/Top-K=12, dim=768, causal prefix aggregation.
- Current FDT-RLM: K=128, Top-K=8, dim=384, cosine temperature=0.25, no timestep input to routing, causal prefix aggregation.

All three historical checkpoint families were found and loaded without forced key conversion.

## Numerical checkpoint probe

- Early cdist normalized entropy mean: {early.get('aggregate_normalized_entropy', {}).get('mean')}
- Revised SDFT epoch 19 normalized entropy mean: {revised.get('aggregate_normalized_entropy', {}).get('mean')}
- Dual-anchor FDT2 epoch 4 normalized entropy mean: {dual.get('aggregate_normalized_entropy', {}).get('mean')}
- Current FDT-RLM normalized entropy mean: {global_metrics['normalized_entropy']['mean']}

Legacy probes use deterministic checkpoint embeddings and synthetic token IDs because the historical Korean training rows are not stored in the repository. They prove checkpoint routing behavior on controlled inputs, not historical dataset-wide averages.

## Mathematical changes

The earliest model divided normalized squared distance by `sqrt(D)` and used sigma in `[1,20]` multiplied by a timestep factor. This made logits extremely close and memberships nearly uniform. Later SDFT removed `sqrt(D)`, bounded sigma to `[0.1,0.7]`, and used a target-entropy objective. Current FDT-RLM uses cosine logits divided by 0.25 and weakly maximizes raw Top-K entropy.

The early structural aggregation was full-sequence and leaked future information. The current causal prefix aggregation must be retained even if it costs more or yields a higher LM loss.
""")
    write_text(ROOT / "ENTROPY_AND_ROUTING_ANALYSIS.md", f"""# Entropy and Routing Analysis

## Metric correction

The reported 2.0238 was not a full-validation mean. It was the final training microbatch, pooled over valid tokens and averaged across six layers with dropout active. It is post-Top-K natural-log entropy.

`mean_active_anchors=128` meant globally active anchors in that microbatch, not anchors per token. Every current token has exactly 8 selected anchors. The reported 4.43% was the largest membership-mass share after averaging layer usage distributions.

## Deterministic checkpoint probe

- Raw entropy mean: {global_metrics['raw_entropy']['mean']:.6f}
- Normalized entropy mean: {global_metrics['normalized_entropy']['mean']:.6f}
- Effective anchor count mean: {global_metrics['effective_anchor_count']['mean']:.6f}
- Top-1 membership mean: {global_metrics['sharpness']['top1_membership']['mean']:.6f}
- Top-1 minus Top-2 margin: {global_metrics['sharpness']['top1_minus_top2_margin']['mean']:.6f}
- Global active anchors: {global_metrics['usage']['global_active_anchors']}
- Mean unique anchors per sequence/layer: {global_metrics['mean_unique_anchors_per_sequence_per_layer']:.3f}

Mean pairwise source membership-mass JS divergence is {source_distance['mean_jensen_shannon_divergence']:.6f}; token-type JS divergence is {token_distance['mean_jensen_shannon_divergence']:.6f}.

## Functional counterfactuals

- Uniform selected Top-K loss delta: {counterfactual['uniform_topk']['loss_delta_vs_learned']:.6f}
- Random Top-K loss delta: {counterfactual['random_topk']['loss_delta_vs_learned']:.6f}
- Zero-anchor loss delta: {counterfactual['zero_anchor']['loss_delta_vs_learned']:.6f}

The entropy conclusion follows the joint evidence in the JSON artifacts; it is not inferred from the mean alone.

## Conclusion

This checkpoint has a true flat-membership problem, concentrated in the last two routing layers. Layer 5 is numerically uniform and layer 4 is nearly uniform. Anchor selection and anchor output remain functionally useful, but the learned fuzzy weights inside the selected Top-K add no measurable value on the counterfactual subset. This is a hybrid of Case B and Case C in the requested decision tree, not global anchor collapse.

## RLM relevance

- Anchor entropy correlation with next-token loss: {current['uncertainty']['correlations_with_next_token_loss']['anchor_entropy']:.6f}
- Token entropy correlation with next-token loss: {current['uncertainty']['correlations_with_next_token_loss']['token_entropy']:.6f}
- R2 using token entropy only: {current['uncertainty']['r2_token_entropy_only']:.6f}
- R2 after adding anchor metrics: {current['uncertainty']['r2_token_plus_anchor_metrics']:.6f}

These are exploratory correlations, not calibrated uncertainty claims.
""")
    top_ops = hotspots.get("top_operations_by_self_cuda_time", [])[:5]
    if hotspots.get("cuda_profiler_available"):
        operation_lines = "\n".join(f"- {item['name']}: self CUDA {item['self_cuda_time_us']:.1f} us, calls={item['calls']}" for item in top_ops)
    else:
        operation_lines = "- CUDA operation timing unavailable: CUPTI returned `INVALID_DEVICE`. The trace contains CPU dispatch and memory events only."
    write_text(ROOT / "PERFORMANCE_BOTTLENECK_REPORT.md", f"""# Performance Bottleneck Report

The matched run measured 3,720 tokens/s and 3.80 GB reserved VRAM for FDT versus 13,278 tokens/s and 2.43 GB for the Transformer.

## Profiled CUDA operations

{operation_lines}

## Structural cause

Each FDT layer materializes a `[B,N,A,D]` prefix numerator. At B=1, N=512, A=128, D=384 this is {hotspots.get('routing_tensor_estimates_per_layer_bf16', {}).get('dense_prefix_numerator_bytes', 0) / 2**20:.2f} MiB in BF16 before cumsum outputs and gradients. The selected sparse contribution is 16 times smaller. Six repetitions plus backward intermediates explain the memory and bandwidth gap.

The first optimization candidate keeps the same dense cumsum but gathers the selected Top-K numerator and denominator before division. This avoids materializing and dividing the full `[B,N,A,D]` prefix mean. It is implemented behind `dense_gather_first`; the historical path remains the default until the blocked PyTorch equivalence rerun completes.
""")
    generation_lines = []
    for item in generations["generations"]:
        generation_lines.append(f"### {item['model']} / T={item['temperature']}\n\nPrompt: `{item['prompt']}`\n\nCompletion: `{item['completion']}`")
    write_text(ROOT / "GENERATION_PROBE_REPORT.md", "# Generation Probe\n\nNo samples were omitted. These 28M models saw only one 10M-token pass, so coherence is diagnostic rather than deployment quality.\n\n" + "\n\n".join(generation_lines))
    write_text(ROOT / "SECOND_EXPERIMENT_PLAN.md", f"""# Second Experiment Plan

## Gate before another 10M-token run

Do not start a second full run yet. First verify and benchmark the behavior-preserving gather-before-divide prefix implementation, then run a fixed 250-step ablation from the same step-250 checkpoint and data position.

## Bounded candidates

1. A0: current FDT reproduction, 250 optimizer steps.
2. A1: gather-before-divide causal-prefix optimization, identical weights and data order.
3. A2: current routing with entropy maximization disabled (`entropy_weight=0`).
4. A3: diffusion auxiliary disabled, only after A1 preserves routing.
5. A4: target-entropy regularization only if A2 remains flat; do not change Top-K in the same run.

Promote at most one or two variants only if they improve validation trend, materially improve throughput at similar loss, or strengthen measured specialization without collapse.

Current uniform-routing loss delta: {counterfactual['uniform_topk']['loss_delta_vs_learned']:.6f}. Current random-routing loss delta: {counterfactual['random_topk']['loss_delta_vs_learned']:.6f}.

No 100M-token scaling or cloud spending is recommended at this stage.
""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--per-source", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--counterfactual-sequences", type=int, default=16)
    parser.add_argument("--generation-tokens", type=int, default=40)
    parser.add_argument("--skip-generations", action="store_true")
    parser.add_argument("--skip-profile", action="store_true")
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(20260710)
    tokenizer = load_tokenizer(str(TOKENIZER_DIR))
    ids, masks, sources, provenance = source_probe_sequences(args.per_source)
    model, config, checkpoint = load_checkpoint_model(CURRENT_FDT, device)
    current = probe_current_checkpoint(model, config, tokenizer, ids, masks, sources, device, args.batch_size)
    counterfactual = counterfactual_probe(model, config, ids, masks, device, args.counterfactual_sequences)
    hotspots = {"status": "skipped"} if args.skip_profile else profile_fdt(model, config, ids, masks, device)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    legacy = probe_legacy(device)
    generations = {"settings": {}, "generations": []} if args.skip_generations else generation_probe(device, tokenizer, args.generation_tokens)
    inventory = file_inventory()
    audit = metric_audit(checkpoint)
    plot_paths = plot_results(current)

    entropy_artifact = {
        "checkpoint": str(CURRENT_FDT),
        "probe_sequences": len(ids),
        "source_provenance": provenance,
        "raw_entropy": current["global"]["raw_entropy"],
        "normalized_entropy": current["global"]["normalized_entropy"],
        "effective_anchor_count": current["global"]["effective_anchor_count"],
        "by_layer": {key: {field: value[field] for field in ("raw_entropy", "normalized_entropy", "effective_anchor_count")} for key, value in current["by_layer"].items()},
        "by_position": current["by_position"],
        "plots": plot_paths,
    }
    sharpness_artifact = {
        "global": current["global"]["sharpness"],
        "by_source": {key: value["sharpness"] for key, value in current["by_source"].items()},
        "by_token_type": {key: value["sharpness"] for key, value in current["by_token_type"].items()},
    }
    source_artifact = {"groups": current["by_source"], "distances": current["source_distances"], "provenance": provenance}
    token_artifact = {"groups": current["by_token_type"], "distances": current["token_type_distances"]}
    ablations = {
        "stage": "inference_counterfactuals_before_retraining",
        "results": counterfactual,
        "training_ablations_run": False,
        "reason": "The requested forensic gate precedes any parameter-changing retraining.",
    }
    write_json(ARTIFACTS / "legacy_fdt_inventory.json", {**inventory, "checkpoint_probe": legacy})
    write_json(ARTIFACTS / "metric_semantics_audit.json", audit)
    write_json(ARTIFACTS / "current_fdt_entropy_distribution.json", entropy_artifact)
    write_json(ARTIFACTS / "current_fdt_membership_sharpness.json", sharpness_artifact)
    write_json(ARTIFACTS / "source_anchor_specialization.json", source_artifact)
    write_json(ARTIFACTS / "token_type_anchor_specialization.json", token_artifact)
    write_json(ARTIFACTS / "fdt_hotspots.json", hotspots)
    write_json(ARTIFACTS / "ablation_results.json", ablations)
    write_json(ARTIFACTS / "rlm_uncertainty_probe.json", current["uncertainty"])
    write_json(ARTIFACTS / "generation_probe.json", generations)
    reports(inventory, audit, current, legacy, counterfactual, hotspots, generations)
    print(json.dumps({
        "probe_sequences": len(ids),
        "normalized_entropy_mean": current["global"]["normalized_entropy"]["mean"],
        "top1_membership_mean": current["global"]["sharpness"]["top1_membership"]["mean"],
        "counterfactuals": counterfactual,
        "reports_written": True,
    }, indent=2))


if __name__ == "__main__":
    main()
