from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from fdt_rlm.config import load_yaml_like
from fdt_rlm.data.code_language_map import map_language
from fdt_rlm.data.dedup import build_deduper, normalized_text_hash
from fdt_rlm.data.filters import FilterStats, filter_code_text, filter_web_text, normalize_code_text, normalize_web_text
from fdt_rlm.data.manifest import file_sha256, write_json
from fdt_rlm.data.sharding import TokenShardWriter
from fdt_rlm.data.source_adapters import stream_hf_documents, stream_local_code, stream_local_tar_texts, stream_local_text_file, stream_stack_smol_xl_group
from fdt_rlm.data.splitting import assign_split
from fdt_rlm.data.state import DatasetState
from fdt_rlm.data.streaming_mix import build_token_plan, ratio_report
from fdt_rlm.tokenization import load_tokenizer


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def iter_bucket_documents(cfg: dict, bucket: str, local_code_dir: str = "", local_fineweb_edu_file: str = "", local_fineweb_file: str = "", local_text_tar_dir: str = ""):
    if bucket in ("fineweb_edu", "fineweb"):
        local_path = local_fineweb_edu_file if bucket == "fineweb_edu" else local_fineweb_file
        if local_path:
            yield from stream_local_text_file(local_path, bucket=bucket, source_name=f"local_{bucket}")
            return
        spec = cfg["sources"][bucket]
        yield from stream_hf_documents(
            dataset=spec["dataset"],
            source_name=bucket,
            bucket=bucket,
            config=spec.get("preferred_config", ""),
            split=spec.get("split", "train"),
            content_field=spec.get("content_field", "text"),
            streaming=spec.get("streaming", True),
        )
        return

    spec = cfg.get("sources", {}).get(bucket)
    if spec and spec.get("kind") == "local_tar_text":
        tar_dir = local_text_tar_dir or spec.get("path", "")
        yield from stream_local_tar_texts(
            tar_dir,
            bucket=bucket,
            source_name=spec.get("source_name", bucket),
        )
        return
    if spec and spec.get("kind") == "hf_text":
        yield from stream_hf_documents(
            dataset=spec["dataset"],
            source_name=bucket,
            bucket=bucket,
            config=spec.get("preferred_config", spec.get("config", "")),
            split=spec.get("split", "train"),
            content_field=spec.get("content_field", "text"),
            streaming=spec.get("streaming", True),
        )
        return

    if bucket.startswith("code_") and local_code_dir:
        for doc in stream_local_code(local_code_dir):
            if doc.bucket == bucket:
                yield doc
        return

    code = cfg["sources"].get("code")
    if not code:
        raise ValueError(f"No source configuration for bucket {bucket!r}")
    candidates = [code.get("preferred", {})] + list(code.get("fallbacks", []))
    wanted_group = bucket.removeprefix("code_")
    for spec in candidates:
        dataset = spec.get("dataset")
        if not dataset:
            continue
        if dataset == "bigcode/the-stack-smol-xl" and spec.get("access_method") == "language_json_files":
            yield from stream_stack_smol_xl_group(wanted_group)
            continue
        try:
            stream = stream_hf_documents(
                dataset=dataset,
                source_name=dataset,
                bucket="code",
                config=spec.get("preferred_config", spec.get("config", "")),
                split=spec.get("split", "train"),
                content_field=spec.get("content_field", ""),
                streaming=spec.get("streaming", True),
            )
            for doc in stream:
                raw_group = doc.bucket.removeprefix("code_")
                if raw_group == wanted_group:
                    yield doc
        except Exception as exc:
            print(f"[fallback] {dataset} skipped: {type(exc).__name__}: {str(exc)[:160]}")
            continue


def process_document(doc, tokenizer, cfg, filter_stats: dict):
    is_code = doc.bucket.startswith("code_")
    if is_code:
        text = normalize_code_text(doc.text)
    else:
        text = normalize_web_text(doc.text)
    ids = tokenizer.encode(text, add_special_tokens=False)
    fcfg = cfg["filters"]["code" if is_code else "web"]
    ok, reason = filter_code_text(text, len(ids), fcfg, doc.metadata.get("path", "")) if is_code else filter_web_text(text, len(ids), fcfg)
    stats = filter_stats[doc.bucket]
    stats.seen += 1
    if not ok:
        stats.reject(reason, len(ids))
        return None
    stats.accept(len(ids))
    return text, ids


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "datasets.yaml"))
    p.add_argument("--total-tokens", type=int, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tokenizer-name", default="gpt2")
    p.add_argument("--local-code-dir", default="")
    p.add_argument("--local-fineweb-edu-file", default="")
    p.add_argument("--local-fineweb-file", default="")
    p.add_argument("--local-text-tar-dir", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-documents", type=int, default=0)
    args = p.parse_args()

    cfg = load_yaml_like(args.config)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if free_gb(out) < float(cfg.get("limits", {}).get("min_free_gb", 20)):
        raise RuntimeError("Insufficient free disk space safety margin.")

    tokenizer = load_tokenizer(args.tokenizer_name)
    plan = build_token_plan(args.total_tokens, cfg.get("target_mix"), cfg.get("code_mix"))
    seq_len = int(cfg["packing"].get("max_seq_len", 512))
    shard_tokens = int(cfg["packing"].get("shard_tokens", 25_000_000))
    writer = TokenShardWriter(out / "shards", seq_len=seq_len, shard_tokens=shard_tokens)
    state_path = out / "dataset_state.json"
    state = DatasetState.load(state_path) if args.resume else DatasetState()
    dedup_path = out / cfg.get("dedup", {}).get("sqlite_path", "artifacts/dedup.sqlite3")
    deduper = build_deduper(dedup_path)
    split_cfg = cfg.get("splits", {"train": 0.99, "validation": 0.005, "test": 0.005})
    filter_stats = defaultdict(FilterStats)
    split_seen = {}
    duplicate_count = 0
    rejected_duplicates = 0
    token_buffers = defaultdict(list)
    language_stats = defaultdict(int)

    for bucket, quota in plan.quotas.items():
        have = int(state.tokens_by_bucket.get(bucket, 0))
        if have >= quota:
            continue
        print(f"[bucket] {bucket}: {have:,}/{quota:,}")
        for doc in iter_bucket_documents(cfg, bucket, args.local_code_dir, args.local_fineweb_edu_file, args.local_fineweb_file, args.local_text_tar_dir):
            if args.max_documents and state.documents_seen >= args.max_documents:
                break
            state.documents_seen += 1
            processed = process_document(doc, tokenizer, cfg, filter_stats)
            if processed is None:
                continue
            text, ids = processed
            current = int(state.tokens_by_bucket.get(bucket, 0))
            tolerance = float(cfg.get("quota", {}).get("relative_tolerance", 0.001))
            max_allowed = int(quota * (1.0 + tolerance))
            if current >= quota:
                break
            if current + len(ids) + 1 > max_allowed:
                filter_stats[doc.bucket].reject("quota_overshoot", len(ids))
                continue
            digest = normalized_text_hash(text)
            split_key = str(doc.metadata.get("repo_name") or doc.metadata.get("directory_id") or doc.doc_id or digest)
            split = assign_split(split_key, split_cfg, seed=int(cfg.get("seed", 42)))
            if not deduper.check_and_add(digest, doc.source, split):
                duplicate_count += 1
                rejected_duplicates += 1
                continue
            if digest in split_seen and split_seen[digest] != split:
                raise RuntimeError("duplicate hash crossed splits")
            split_seen[digest] = split
            ids = [int(x) for x in ids] + [int(tokenizer.eos_token_id)]
            combined = token_buffers[(split, bucket)] + ids
            full_len = (len(combined) // seq_len) * seq_len
            for start in range(0, full_len, seq_len):
                writer.add(split, combined[start : start + seq_len], bucket)
            token_buffers[(split, bucket)] = combined[full_len:]
            state.tokens_by_bucket[bucket] = int(state.tokens_by_bucket.get(bucket, 0)) + len(ids)
            state.documents_kept += 1
            if bucket.startswith("code_"):
                language_stats[bucket] += len(ids)
            if state.tokens_by_bucket[bucket] >= quota:
                break
        state.save(state_path)

    shards = writer.close()
    packed_tokens = sum(int(s.tokens) for s in shards)
    discarded_tail_tokens = sum(len(v) for v in token_buffers.values())
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "requested_tokens": args.total_tokens,
        "tokenizer": args.tokenizer_name,
        "tokenizer_vocab_size": len(tokenizer),
        "plan": plan.quotas,
        "tokens_by_bucket": state.tokens_by_bucket,
        "packed_tokens": packed_tokens,
        "discarded_tail_tokens": discarded_tail_tokens,
        "ratio_report": ratio_report(state.tokens_by_bucket, plan),
        "seq_len": seq_len,
        "shards": [s.__dict__ | {"sha256": file_sha256(s.path)} for s in shards],
        "duplicates": {"exact_duplicates": duplicate_count, "rejected_duplicates": rejected_duplicates},
        "source_config": cfg.get("sources", {}),
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "dataset_statistics.json", {"tokens_by_bucket": state.tokens_by_bucket, "documents_seen": state.documents_seen, "documents_kept": state.documents_kept})
    write_json(out / "filter_stats.json", {k: {"seen": v.seen, "kept": v.kept, "rejected": v.rejected, "reasons": dict(v.reasons), "tokens_kept": v.tokens_kept, "tokens_rejected": v.tokens_rejected} for k, v in filter_stats.items()})
    write_json(out / "dedup_stats.json", {"exact_duplicates": duplicate_count, "rejected_duplicates": rejected_duplicates})
    write_json(out / "language_stats.json", dict(language_stats))
    report = [
        "# DATASET REPORT",
        "",
        f"created_at: {manifest['created_at']}",
        f"requested_tokens: {args.total_tokens}",
        f"tokenizer: {args.tokenizer_name}",
        "",
        "## Requested / Achieved",
    ]
    for key, row in manifest["ratio_report"].items():
        report.append(f"- {key}: expected={int(row['expected_tokens']):,}, actual={int(row['actual_tokens']):,}")
    report.extend(["", "## Known Limitations", "- Streaming upstream order/revisions may change unless revision is pinned.", "- PII filtering is heuristic, not a guarantee."])
    (out / "DATASET_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    if hasattr(deduper, "close"):
        deduper.close()
    print(f"prepared: {out}")


if __name__ == "__main__":
    main()
