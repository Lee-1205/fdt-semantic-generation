import math
from pathlib import Path

from scripts.audit_strict_v3_evaluation_integrity import (
    close,
    unique_keys,
    walk_nonfinite,
)


ROOT = Path(__file__).resolve().parents[1]


def test_nonfinite_walker_reports_nested_values() -> None:
    errors = walk_nonfinite({"ok": 1.0, "bad": [math.inf, {"nan": math.nan}]})
    assert len(errors) == 2
    assert any("root.bad[0]" in error for error in errors)
    assert any("root.bad[1].nan" in error for error in errors)


def test_unique_key_and_close_checks_are_strict() -> None:
    errors = []
    unique_keys([{"row_id": 1}, {"row_id": 1}], ("row_id",), "rows", errors)
    close(1.0, 1.1, "metric", errors)
    assert any("duplicate rows" in error for error in errors)
    assert any("metric mismatch" in error for error in errors)


def test_auditor_supports_python_whitespace_repair_version() -> None:
    source = (
        ROOT / "scripts" / "audit_strict_v3_evaluation_integrity.py"
    ).read_text(encoding="utf-8")
    assert '"strict_base_v3_audited_1"' in source
    assert '"strict_base_v3_audited_2"' in source
