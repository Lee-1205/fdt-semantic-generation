from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.data.rlm_task_generator import generate_rlm_tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--output", default=str(ROOT / "artifacts" / "rlm_tasks_500.jsonl"))
    args = parser.parse_args()
    tasks = list(generate_rlm_tasks(args.per_family, args.seed))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task.to_dict(), ensure_ascii=False) + "\n")
    summary = {
        "output": str(output),
        "tasks": len(tasks),
        "families": dict(Counter(task.family for task in tasks)),
        "seed": args.seed,
        "training_tokens_used": 0,
    }
    (ROOT / "artifacts" / "rlm_task_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
