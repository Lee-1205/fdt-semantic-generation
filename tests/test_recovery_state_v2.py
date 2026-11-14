import copy
import importlib.util
from pathlib import Path
import sys

import torch

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("recovery_v2", scripts / "run_learning_102m_recovery_v2.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def payload():
    value = {key: None for key in recovery.STATE_FIELDS}
    value.update({"model": {"weight": torch.tensor([1., 2.])},
                  "optimizer": {"state": {0: {"step": torch.tensor(110.), "exp_avg": torch.tensor([.1, .2])}},
                                "param_groups": [{"lr": .0008, "params": [0]}]},
                  "sampler_rng": torch.Generator().manual_seed(20).get_state(),
                  "phase": 0, "phase_step": 110, "attempts": 110, "updates": 110})
    return value


def test_state_digest_tracks_weights_adam_rng_and_lr():
    before = payload()
    same = copy.deepcopy(before)
    assert recovery.resume_state_digest(before) == recovery.resume_state_digest(same)
    for field in ("model", "optimizer", "sampler_rng", "phase_step"):
        changed = copy.deepcopy(before)
        if field == "model":
            changed[field]["weight"][0] += 1
        elif field == "optimizer":
            changed[field]["param_groups"][0]["lr"] *= .5
        elif field == "sampler_rng":
            changed[field] = torch.Generator().manual_seed(21).get_state()
        else:
            changed[field] += 1
        assert recovery.resume_state_digest(before) != recovery.resume_state_digest(changed)


def test_restored_sampler_starts_with_next_batch_not_first_batch():
    source = torch.Generator().manual_seed(20)
    first = torch.randint(4096, (4,), generator=source)
    state = source.get_state()
    uninterrupted_next = torch.randint(4096, (4,), generator=source)
    restored = torch.Generator().set_state(state)
    assert torch.equal(torch.randint(4096, (4,), generator=restored), uninterrupted_next)
    assert not torch.equal(first, uninterrupted_next)


def test_recovery_budget_subtracts_imported_attempts(tmp_path):
    exp = recovery.Experiment.__new__(recovery.Experiment)
    exp.run_dir, exp.training_deadline = tmp_path, float("inf")
    exp.plan = {"attempts_at_start": 110, "max_attempts": 1390}
    exp.engine = type("Engine", (), {"attempts": 1390})()
    assert exp.stop_reason() is None  # 1,280 new attempts, not the 1,390 limit.
    exp.engine.attempts = 1500
    assert exp.stop_reason() == "ATTEMPT_CAP"
