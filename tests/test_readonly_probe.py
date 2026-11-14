import importlib.util
from pathlib import Path
import sys

import pytest

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("proof", scripts / "probe_e1_readonly.py")
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


def test_inference_attempt_cap_and_clock_never_reset(monkeypatch):
    now = [0.]
    monkeypatch.setattr(proof.time, "monotonic", lambda: now[0])
    budget = proof.Budget(5, 3)
    budget.reserve(2)
    with pytest.raises(RuntimeError):
        budget.reserve(2)
    assert budget.attempted == 2 and budget.deadline == 5
    now[0] = 4
    budget.reserve(1)
    assert budget.deadline == 5
    now[0] = 5
    with pytest.raises(RuntimeError):
        budget.reserve(1)
    assert budget.attempted == 3
