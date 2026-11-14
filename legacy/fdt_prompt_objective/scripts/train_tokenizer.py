from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()

from fdt_rlm.data.manifest import write_json
from fdt_rlm.tokenization import SPECIAL_TOKENS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(ROOT / "artifacts" / "tokenizer_corpus.txt"))
    p.add_argument("--output-dir", default=str(ROOT / "tokenizers" / "fdt_bpe"))
    p.add_argument("--vocab-size", type=int, default=49152)
    p.add_argument("--min-frequency", type=int, default=2)
    args = p.parse_args()

    from tokenizers import ByteLevelBPETokenizer

    inp = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = ByteLevelBPETokenizer()
    tok.train(files=[str(inp)], vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=SPECIAL_TOKENS)
    tok.save_model(str(out))
    tok.save(str(out / "tokenizer.json"))
    h = hashlib.sha256()
    for file in sorted(out.glob("*")):
        h.update(file.read_bytes())
    write_json(out / "tokenizer_report.json", {"vocab_size": args.vocab_size, "special_tokens": SPECIAL_TOKENS, "hash": h.hexdigest()})
    print(f"saved tokenizer: {out}")


if __name__ == "__main__":
    main()
