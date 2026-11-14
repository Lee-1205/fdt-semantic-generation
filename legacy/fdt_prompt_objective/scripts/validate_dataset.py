from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import add_src_to_path

add_src_to_path()

from fdt_rlm.data.manifest import file_sha256, read_json, write_json
from fdt_rlm.data.streaming_mix import TokenPlan, assert_within_tolerance


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--tolerance", type=float, default=0.001)
    args = p.parse_args()

    root = Path(args.dataset_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise SystemExit("FAIL: manifest.json missing")
    manifest = read_json(manifest_path)
    failures = []
    total_tokens = int(manifest["requested_tokens"])
    plan = TokenPlan(total_tokens=total_tokens, quotas={k: int(v) for k, v in manifest["plan"].items()})

    for shard in manifest.get("shards", []):
        path = Path(shard["path"])
        if not path.exists():
            failures.append(f"missing shard {path}")
            continue
        if file_sha256(path) != shard.get("sha256"):
            failures.append(f"checksum mismatch {path}")
            continue
        data = torch.load(path, map_location="cpu")
        ids = data["input_ids"]
        if ids.ndim != 2 or ids.numel() == 0:
            failures.append(f"bad shape {path}")
        if int(ids.min()) < 0:
            failures.append(f"negative token id {path}")
        if int(ids.max()) >= int(manifest.get("tokenizer_vocab_size", 10**12)):
            failures.append(f"token id outside vocab {path}")

    try:
        assert_within_tolerance({k: int(v) for k, v in manifest.get("tokens_by_bucket", {}).items()}, plan, args.tolerance)
    except Exception as exc:
        failures.append(str(exc))

    report = {"status": "FAIL" if failures else "PASS", "failures": failures}
    write_json(root / "validation_report.json", report)
    if failures:
        print("FAIL")
        for item in failures:
            print("-", item)
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()

