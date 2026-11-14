from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "guard_fdt_v4_1_retention_smoke.py"
SPEC = importlib.util.spec_from_file_location("fdt_v4_1_guardian_test", PATH)
guardian = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(guardian)


def test_guardian_only_flags_declared_metric_failures():
    assert guardian.event_stop_reason(
        {"event": "train", "finite_loss": True, "loss": 2.0, "base_grad_norm": 1.0}
    ) is None
    assert guardian.event_stop_reason(
        {"event": "train", "finite_loss": False, "loss": float("nan")}
    ) == "nonfinite_train"
    assert guardian.event_stop_reason(
        {"event": "fixed52_validation", "gate_triggered": True}
    ) == "fixed52_regression_gate"


def test_guardian_has_no_process_termination_primitive():
    source = PATH.read_text(encoding="utf-8")
    assert "STOP_REQUESTED" in source
    assert ".kill(" not in source
    assert ".terminate(" not in source
    assert "taskkill" not in source.lower()
