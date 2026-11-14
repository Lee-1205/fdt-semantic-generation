from dataclasses import asdict, FrozenInstanceError
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from filelock import Timeout

from fdt_local import model, model_masked
from fdt_local.budget import SmokeBudget
from fdt_local.data import collate, make_split, oracle
from fdt_local.objectives import masked_ce, masked_mse
from fdt_local.runtime import Commit, realize
from fdt_local.storage import CheckpointStore, ensure_space
from fdt_local.evaluation import complete_missing

ROOT = Path(__file__).resolve().parents[1]


def fixture_payload(attempt=0):
    return {"schema": CheckpointStore.schema, "contract": {"source": "fixture", "env": "cpu-serialization-only"},
            "run_id": "test", "attempts": attempt, "updates": attempt,
            "phase": 0, "phase_step": attempt, "history": [], "model": {"weight": torch.ones(2)},
            "optimizer": {"state": {}, "param_groups": []}, "scaler": {}, "sampler_rng": torch.zeros(2, dtype=torch.uint8),
            "noise_rng": torch.zeros(2, dtype=torch.uint8), "torch_rng": torch.zeros(2, dtype=torch.uint8),
            "cuda_rng": [], "python_rng": (), "numpy_rng": {}}


def test_original_forward_equivalence():
    source = ROOT / "archive" / "recovery_v2" / "original_source" / "model.py"
    spec = importlib.util.spec_from_file_location("preserved_model_fixture", source)
    original = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = original
    spec.loader.exec_module(original)
    cfg = model.Config.smoke()
    torch.manual_seed(19)
    port = model.FDT100(cfg).eval()
    old = original.FDT100(original.Config(**asdict(cfg))).eval()
    old.load_state_dict(port.state_dict())
    features, _, _ = collate(make_split("train", 1), cfg)
    with torch.no_grad():
        left, right = port.encode(features), old.encode(features)
    for key in ("init", "facts", "memory", "ground"):
        assert torch.equal(left[key], right[key])


def test_objective_denominators_and_masks():
    logits = torch.tensor([[[1., 0.], [0., 2.], [8., -8.]]])
    labels = torch.tensor([[0, 1, 1]])
    mask = torch.tensor([[1., 1., 0.]])
    weights = torch.tensor([1., 3.])
    terms = torch.nn.functional.cross_entropy(logits[0, :2], labels[0, :2], weight=weights, reduction="none")
    assert torch.allclose(masked_ce(logits, labels, mask, weights), terms.sum() / 4)
    assert masked_mse(torch.ones(1, 3, 2), torch.zeros(1, 3, 2), mask) == 1
    assert masked_mse(torch.ones(1, 3, 2), torch.zeros(1, 3, 2), mask * 0) == 0


@pytest.mark.parametrize("t", [0., .25, .5, 1.])
def test_velocity_inverse(t):
    import math
    a, s = math.cos(math.pi * t / 2), math.sin(math.pi * t / 2)
    if t == 1:
        a, s = 0., 1.
    z0, noise = torch.randn(2, 3), torch.randn(2, 3)
    z = a * z0 + s * noise
    velocity = a * noise - s * z0
    assert abs(a * a + s * s - 1) < 1e-12
    assert torch.allclose(a * z - s * velocity, z0, atol=1e-6)
    assert torch.allclose(s * z + a * velocity, noise, atol=1e-6)


def test_router_zero_invalid_and_ties():
    cfg = model.Config.smoke()
    router = model.Router(cfg.lang_dim, cfg)
    mu, rho, gamma = router(torch.zeros(1, 1, cfg.lang_dim), torch.zeros(cfg.anchors, cfg.key_dim))
    assert not mu.any() and not rho.any() and not gamma.any()
    assert torch.isfinite(mu).all()


@pytest.mark.parametrize("facts,expected", [
    ([(0, 1, 2, "not greater")], 0),
    ([(0, 1, 0, "greater")], 1),
    ([(0, 1, 1, "less")], 2),
    ([(0, 1, 0, "greater"), (0, 1, 2, "not greater")], 3),
    ([(0, 1, 0, "greater"), (1, 2, 0, "greater")], 1),
])
def test_typed_oracle(facts, expected):
    query = (0, 2) if len(facts) == 2 and facts[1][1] == 2 else (0, 1)
    assert oracle(3, facts, [query])[0] == [expected]


def test_supervision_not_in_features():
    features, targets, exposure = collate(make_split("train", 1), model.Config.smoke())
    assert set(features) == {"tokens", "roles", "fact_mask", "queries", "query_mask"}
    assert set(targets) == {"labels", "ground", "hops"}
    assert exposure > 0


def test_masked_candidate_padding_invariance():
    cfg = model_masked.Config.smoke()
    small = model_masked.Config(**{**asdict(cfg), "facts": 6, "slots": 8})
    item = {"n": 3, "facts": [(0, 1, 0, "is greater than")], "queries": [(0, 1)], "labels": [1], "hops": [1]}
    large_features, _, _ = collate([item], cfg)
    small_features, _, _ = collate([item], small)
    torch.manual_seed(31)
    large = model_masked.FDT100(cfg).eval()
    smaller = model_masked.FDT100(small).eval()
    smaller.load_state_dict(large.state_dict())
    with torch.no_grad():
        a = large.encode(large_features)
        b = smaller.encode(small_features)
    assert torch.allclose(a["init"][:, :1], b["init"][:, :1], atol=1e-6, rtol=1e-5)
    assert not a["init"][:, 1:].any()


def test_all_invalid_empty_fact_and_query_are_finite():
    cfg = model_masked.Config.smoke()
    features, _, _ = collate([{"n": 2, "facts": [], "queries": [], "labels": [], "hops": []}], cfg)
    instance = model_masked.FDT100(cfg).eval()
    with torch.no_grad():
        enc = instance.encode(features)
        velocity = instance.denoise(torch.zeros_like(enc["init"]), torch.ones(1), enc)
    assert torch.isfinite(velocity).all() and not velocity.any()
    assert torch.isfinite(enc["memory"]).all() and not enc["memory"].any()


def test_utf8_not_normalized_and_byte_capacity_enforced():
    decomposed, composed = "e\u0301", "é"
    commit = Commit("A", "B", "UNRESOLVED", "UNRESOLVED", [0, 1], [decomposed, composed], 0.)
    assert commit.source_texts[0].encode("utf-8") != commit.source_texts[1].encode("utf-8")
    cfg = model.Config.smoke()
    item = {"n": 2, "facts": [(0, 1, 0, "가" * 17)], "queries": [], "labels": [], "hops": []}
    with pytest.raises(ValueError, match="bytes exceeded"):
        collate([item], cfg)


def test_commit_has_no_mutable_alias():
    ids, texts = [0], ["A 더 크다 B"]
    commit = Commit("A", "B", "GREATER", "SOURCE_RELATIVE_VERIFIED", ids, texts, .5)
    before = realize(commit)
    ids.append(1)
    texts[0] = "changed"
    assert commit.source_ids == (0,) and commit.source_texts == ("A 더 크다 B",)
    assert realize(commit) == before
    with pytest.raises(FrozenInstanceError):
        commit.subject = "C"
    with pytest.raises(TypeError):
        Commit("A", "B", "GREATER", "x", [0], [["mutable"]], .5)


def test_store_corrupt_latest_falls_back(tmp_path):
    store = CheckpointStore(tmp_path)
    _, marker0 = store.commit(fixture_payload(), 100000)
    meta1, marker1 = store.commit(fixture_payload(1), 100000)
    (tmp_path / meta1["file"]).write_bytes(b"corrupt")
    meta, payload, marker = store.recover({"source": "fixture", "env": "cpu-serialization-only"})
    assert marker == marker0 and payload["attempts"] == 0


def test_contract_mismatch_and_marker_missing_field(tmp_path):
    store = CheckpointStore(tmp_path)
    _, marker = store.commit(fixture_payload(), 100000)
    with pytest.raises(ValueError, match="mismatch"):
        store.verify(marker, {"source": "different"})
    meta = json.loads(marker.read_text())
    del meta["bytes"]
    marker.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="incomplete"):
        store.verify(marker)


def test_save_failure_preserves_previous(tmp_path):
    store = CheckpointStore(tmp_path)
    _, previous = store.commit(fixture_payload(), 100000)
    with patch("fdt_local.storage.os.replace", side_effect=PermissionError("injected Windows sharing violation")):
        with pytest.raises(PermissionError):
            store.commit(fixture_payload(1), 100000)
    assert store.verify(previous)[1]["attempts"] == 0


def test_disk_floor_and_writer_lock(tmp_path):
    with patch("fdt_local.storage.shutil.disk_usage") as usage:
        usage.return_value.free = 20 * 2**30 + 10
        with pytest.raises(OSError, match="reserve"):
            ensure_space(tmp_path, 11)
    store = CheckpointStore(tmp_path)
    with store.lock():
        with pytest.raises(Timeout):
            with store.lock():
                pass
    with store.lock():
        pass


def test_shared_budget_never_resets(tmp_path):
    budget = SmokeBudget(tmp_path / "budget.json")
    budget.begin()
    for index in range(5):
        budget.reserve_attempt(str(index))
    with pytest.raises(RuntimeError):
        budget.reserve_attempt("six")
    with pytest.raises(ValueError):
        budget.begin()
    budget.finish(.01, True)
    assert budget.read()["attempts"] == 5


def test_partial_evaluation_resumes_only_missing(tmp_path):
    suite = {"wall_seconds": 20, "suites": {"a": 1, "b": 2}}
    output = tmp_path / "eval.json"
    calls = []
    def interrupted(task):
        calls.append(task)
        if task == 2:
            raise RuntimeError("interrupted evaluation")
        return {"fixture": True}
    with pytest.raises(RuntimeError):
        complete_missing(output, "fixture-checkpoint", suite, interrupted)
    assert complete_missing(output, "fixture-checkpoint", suite, lambda task: calls.append(task) or {"fixture": True})
    assert calls == [1, 2, 2]
    with pytest.raises(ValueError, match="scope"):
        complete_missing(output, "other-checkpoint", suite, lambda _: {})


def test_initial_save_time_counts_against_fixed_deadline(tmp_path):
    from fdt_local.runner import Engine
    from types import SimpleNamespace
    instance = Engine.__new__(Engine)
    instance.phase = instance.phase_step = instance.attempts = instance.updates = 0
    instance.out = tmp_path
    instance.plan = SimpleNamespace(phase_steps=(10, 10, 10))
    clock = [0.]
    instance.save = lambda *args: clock.__setitem__(0, clock[0] + 6.)
    instance.step = lambda: pytest.fail("must not update after initial save exhausts deadline")
    with patch("fdt_local.runner.time.monotonic", side_effect=lambda: clock[0]):
        assert instance.run(5, 5.) == "TIME_BUDGET"


def fake_boundary_engine(tmp_path, phase):
    from fdt_local.runner import Engine
    from types import SimpleNamespace
    instance = Engine.__new__(Engine)
    instance.phase = phase
    instance.phase_step = 1
    instance.attempts = instance.updates = 0
    instance.out, instance.model, instance.dev = tmp_path, object(), []
    instance.gates = []
    instance.plan = SimpleNamespace(phase_steps=(1, 1, 1), micro_batch=1, ground_gate=.9, semantic_gate=.8)
    instance.save = lambda *args: None
    instance.step = lambda: pytest.fail("no optimizer update allowed at terminal/gate failure")
    instance._optimizer = lambda: pytest.fail("final optimizer must be retained")
    return instance


def test_functional_gate_blocks_next_phase(tmp_path):
    instance = fake_boundary_engine(tmp_path, 0)
    with patch("fdt_local.runner.evaluate", return_value={"ground_accuracy": 0.}):
        assert instance.run(1, float("inf")) == "FUNCTIONAL_GATE"
    assert instance.phase == 0 and not instance.gates[0]["passed"]


def test_terminal_phase_preserves_final_optimizer(tmp_path):
    instance = fake_boundary_engine(tmp_path, 2)
    instance.optimizer = object()
    original_optimizer = instance.optimizer
    with patch("fdt_local.runner.evaluate", return_value={"initializer": {"accuracy": .1}}):
        assert instance.run(1, float("inf")) == "TRAINING_COMPLETE_EVALUATION_PENDING"
    assert instance.phase == 3 and instance.optimizer is original_optimizer


def test_save_failure_prevents_next_attempt(tmp_path):
    from fdt_local.runner import Engine
    from types import SimpleNamespace
    instance = Engine.__new__(Engine)
    instance.phase = instance.phase_step = instance.attempts = instance.updates = 0
    instance.out = tmp_path
    instance.plan = SimpleNamespace(phase_steps=(10, 10, 10))
    saves, steps = [], []
    def save(*args):
        saves.append(1)
        if len(saves) > 1:
            raise OSError("injected checkpoint failure")
    def step():
        steps.append(1)
        instance.attempts += 1
        return {"fixture": True}
    instance.save, instance.step = save, step
    with pytest.raises(OSError, match="checkpoint"):
        instance.run(5, float("inf"))
    assert len(steps) == 1
