import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from astra_evidence_io import ensure_run_identity  # noqa: E402


def test_prior_clock_and_failed_exit_remain_when_identity_is_created(tmp_path):
    execution = tmp_path / "execution"
    execution.mkdir()
    clock = execution / "clock.json"
    failed_exit = execution / "budget_exit_failed.json"
    clock.write_text(json.dumps({"deadline_unix": 12345}), encoding="utf-8")
    failed_exit.write_text(json.dumps({"charged_seconds": 64}), encoding="utf-8")

    identity_dir = execution / "identity"
    identity = {"plan_sha256": "frozen", "runner_sha256": "patched"}
    ensure_run_identity(identity_dir, identity)
    ensure_run_identity(identity_dir, identity)
    assert json.loads(clock.read_text(encoding="utf-8")) == {"deadline_unix": 12345}
    assert json.loads(failed_exit.read_text(encoding="utf-8")) == {"charged_seconds": 64}
    assert json.loads((identity_dir / "run_identity.json").read_text(encoding="utf-8")) == identity

    with pytest.raises(RuntimeError, match="identity changed"):
        ensure_run_identity(identity_dir, {**identity, "runner_sha256": "drift"})
