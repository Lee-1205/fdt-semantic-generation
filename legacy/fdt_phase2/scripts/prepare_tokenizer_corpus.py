from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from fdt_rlm.config import load_yaml_like
from fdt_rlm.data.filters import normalize_web_text
from fdt_rlm.data.source_adapters import stream_hf_documents
from fdt_rlm.tokenization import load_tokenizer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(ROOT / "configs" / "datasets.yaml"))
    p.add_argument("--target-tokens", type=int, default=10_000_000)
    p.add_argument("--output", default=str(ROOT / "artifacts" / "tokenizer_corpus.txt"))
    p.add_argument("--bootstrap-tokenizer", default="gpt2")
    p.add_argument("--max-documents", type=int, default=0)
    args = p.parse_args()

    cfg = load_yaml_like(args.config)
    tok = load_tokenizer(args.bootstrap_tokenizer)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    written_by_bucket = {}
    docs = 0

    def write_text(handle, text: str, bucket: str, quota: int) -> bool:
        nonlocal docs
        current = int(written_by_bucket.get(bucket, 0))
        if current >= quota:
            return True
        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            return current >= quota
        handle.write(text.rstrip() + "\n\n")
        written_by_bucket[bucket] = current + len(ids)
        docs += 1
        return written_by_bucket[bucket] >= quota or (args.max_documents and docs >= args.max_documents)

    if out.exists():
        out.unlink()

    target_mix = cfg.get("target_mix", {})
    if not target_mix:
        raise ValueError("config must define a non-empty target_mix")
    ratio_sum = sum(float(value) for value in target_mix.values())
    if ratio_sum <= 0:
        raise ValueError("target_mix ratios must sum to a positive value")
    quotas = {
        name: int(args.target_tokens * float(ratio) / ratio_sum)
        for name, ratio in target_mix.items()
    }
    final_source = next(reversed(quotas))
    quotas[final_source] += args.target_tokens - sum(quotas.values())

    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for source_name in target_mix:
            spec = cfg["sources"][source_name]
            if spec.get("kind", "hf_text") != "hf_text":
                raise ValueError(f"Tokenizer corpus only accepts hf_text sources, got {source_name}")
            stream = stream_hf_documents(
                dataset=spec["dataset"],
                config=spec.get("preferred_config", ""),
                split=spec.get("split", "train"),
                streaming=spec.get("streaming", True),
                content_field=spec.get("content_field", "text"),
                source_name=source_name,
                bucket=source_name,
            )
            for doc in stream:
                if write_text(handle, normalize_web_text(doc.text), source_name, quotas[source_name]):
                    break
    total = sum(written_by_bucket.values())
    print(f"saved {total:,} bootstrap tokens to {out}")
    print(written_by_bucket)


if __name__ == "__main__":
    main()
