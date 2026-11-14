from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _bootstrap import add_src_to_path

add_src_to_path()

from fdt_rlm.data.manifest import write_json
from fdt_rlm.tokenization import SPECIAL_TOKENS, load_tokenizer


def tokenizer_hash(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    for file in sorted(path.glob("*")):
        if file.is_file() and file.name != "tokenizer_validation.json":
            h.update(file.read_bytes())
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    args = p.parse_args()

    tok = load_tokenizer(args.tokenizer)
    samples = [
        "The model routes tokens through fuzzy anchors.",
        "def calculate_loss(logits, targets):\n    return F.cross_entropy(logits, targets)",
        '{"action":"SEARCH","query":"optimizer step"}',
        "## Training\nRun `python train.py --config config.yaml`.",
        "src/fdt_rlm/models/causal_lm.py",
    ]
    failures = []
    for text in samples:
        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            failures.append("empty_encoding")
        if tok.decode(ids) != text:
            failures.append("roundtrip_failed")
    for token in SPECIAL_TOKENS:
        token_id = tok.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0:
            failures.append(f"missing_special:{token}")
    report = {
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "vocab_size": len(tok),
        "hash": tokenizer_hash(args.tokenizer) if Path(args.tokenizer).exists() else "",
        "special_tokens": SPECIAL_TOKENS,
    }
    out = Path(args.tokenizer) / "tokenizer_validation.json" if Path(args.tokenizer).exists() else Path("tokenizer_validation.json")
    write_json(out, report)
    print(report["status"])
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
