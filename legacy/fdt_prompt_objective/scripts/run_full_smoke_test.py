from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from _bootstrap import add_src_to_path

ROOT = add_src_to_path()


def run(cmd):
    print("RUN", " ".join(str(x) for x in cmd))
    subprocess.run([sys.executable, *map(str, cmd)], check=True)


def main():
    dataset = ROOT / "prepared_data" / "fdt_rlm_10m"
    tok = ROOT / "tokenizers" / "fdt_bpe"
    run([ROOT / "scripts" / "validate_tokenizer.py", "--tokenizer", tok])
    run([ROOT / "scripts" / "validate_dataset.py", "--dataset-dir", dataset])
    run([ROOT / "scripts" / "train_base.py", "--dataset_dir", dataset, "--tokenizer_name", tok, "--model_type", "transformer", "--max_steps", "1", "--batch_size", "1", "--grad_accum_steps", "1", "--dim", "64", "--n_layers", "1", "--n_heads", "4", "--output_dir", ROOT / "runs" / "full_smoke_transformer", "--device", "cpu"])
    run([ROOT / "scripts" / "train_base.py", "--dataset_dir", dataset, "--tokenizer_name", tok, "--model_type", "fdt", "--max_steps", "1", "--batch_size", "1", "--grad_accum_steps", "1", "--dim", "64", "--n_layers", "1", "--n_heads", "4", "--num_anchors", "16", "--top_k", "4", "--output_dir", ROOT / "runs" / "full_smoke_fdt", "--device", "cpu"])
    run([ROOT / "scripts" / "check_training_readiness.py", "--dataset-dir", dataset, "--tokenizer", tok, "--device", "cpu"])


if __name__ == "__main__":
    main()
