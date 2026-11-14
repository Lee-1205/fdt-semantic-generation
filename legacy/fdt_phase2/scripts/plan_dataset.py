from __future__ import annotations

import argparse
import json

from _bootstrap import add_src_to_path

add_src_to_path()

from fdt_rlm.data.streaming_mix import build_token_plan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--total-tokens", type=int, required=True)
    p.add_argument("--shard-tokens", type=int, default=25_000_000)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    plan = build_token_plan(args.total_tokens)
    disk_gb = args.total_tokens * 2.2 / (1024**3)
    shards = max(1, (args.total_tokens + args.shard_tokens - 1) // args.shard_tokens)
    data = {
        "total_tokens": args.total_tokens,
        "quotas": plan.quotas,
        "estimated_disk_gb": round(disk_gb, 3),
        "estimated_shards": shards,
    }
    if args.json:
        print(json.dumps(data, indent=2))
        return
    print(f"TOTAL TARGET: {args.total_tokens:,}")
    for key, value in plan.quotas.items():
        print(f"{key:32s} {value:15,}")
    print(f"estimated_disk_gb: {disk_gb:.2f}")
    print(f"estimated_shards: {shards}")


if __name__ == "__main__":
    main()

