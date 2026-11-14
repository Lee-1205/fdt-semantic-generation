"""Checkpoint/suite-scoped results; resume only missing completed tasks."""
import hashlib
import json
from pathlib import Path
import time
from .storage import atomic_json, canonical


def complete_missing(output, checkpoint_sha256, suite, evaluate_task):
    output = Path(output)
    suite_sha = hashlib.sha256(canonical(suite).encode()).hexdigest()
    result = {"checkpoint_sha256": checkpoint_sha256, "suite_sha256": suite_sha,
              "scope": "neural_before_verifier", "suites": {}}
    if output.exists():
        saved = json.loads(output.read_text(encoding="utf-8"))
        if saved.get("checkpoint_sha256") != checkpoint_sha256 or saved.get("suite_sha256") != suite_sha:
            raise ValueError("evaluation checkpoint/suite scope mismatch")
        result = saved
    start = time.monotonic()
    for name, task in suite["suites"].items():
        if name in result["suites"]:
            continue
        if time.monotonic() - start >= suite["wall_seconds"]:
            break
        result["suites"][name] = evaluate_task(task)
        atomic_json(output, result)
    complete = set(result["suites"]) == set(suite["suites"])
    atomic_json(output.with_suffix(".complete.json"), {"evaluation_complete": complete,
                "checkpoint_sha256": checkpoint_sha256, "suite_sha256": suite_sha})
    return complete
