import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("learning", Path(__file__).resolve().parents[1] / "scripts/run_learning_102m.py")
learning = importlib.util.module_from_spec(spec)
spec.loader.exec_module(learning)


def row(**change):
    return {"loss": 1., "amp_skipped": False, "successful": True,
            "finite_gradients": True, "gradient_norm_before_clip": 1., **change}


def test_numerical_anomalies_and_normal_learning():
    assert learning.anomaly_reason([row(loss=1.4), row(loss=1.5)]) is None
    assert learning.anomaly_reason([row(loss=float("nan"))]) == "NONFINITE_LOSS"
    assert learning.anomaly_reason([row(amp_skipped=True, successful=False)] * 3) == "THREE_CONSECUTIVE_AMP_SKIPS"
    assert learning.anomaly_reason([row(gradient_norm_before_clip=2000)] * 3) == "REPEATED_EXTREME_PRECLIP_GRADIENT"
    assert learning.anomaly_reason([row()] * 20 + [row(loss=200)] * 3) == "REPEATED_LOSS_EXPLOSION"


def test_deadline_and_completed_success_cap_are_distinct(tmp_path):
    exp = learning.Experiment.__new__(learning.Experiment)
    exp.run_dir = tmp_path
    exp.training_deadline = 0
    exp.engine = type("Engine", (), {"attempts": 1440})()
    exp.plan = {"max_attempts": 1500}
    assert exp.stop_reason() == "FIXED_TRAINING_DEADLINE"
    exp.training_deadline = float("inf")
    assert exp.stop_reason() is None
    exp.engine.attempts = 1500
    assert exp.stop_reason() == "ATTEMPT_CAP"
    (tmp_path / "STOP").write_text("user")
    assert exp.stop_reason() == "COOPERATIVE_STOP"


def fixture_experiment(tmp_path, monkeypatch, phase, step, passed):
    exp = learning.Experiment.__new__(learning.Experiment)
    config = tmp_path / "config.yaml"
    config.write_text("{}", encoding="utf-8")
    exp.plan_path = tmp_path / "plan.json"
    exp.plan_path.write_text("{}", encoding="utf-8")
    exp.plan = {"config_path": str(config), "worker_sha256": "test", "supervisor_sha256": "test"}
    exp.out, exp.run_dir, exp.deadline = tmp_path, tmp_path / "model", float("inf")
    exp.training_deadline = float("inf")
    exp.progress = lambda *_a, **_kw: None
    saved = []
    class Engine:
        plan = type("Plan", (), {"phase_steps": [480, 320, 640]})()
        model = type("Model", (), {"counts": lambda _s: {"parameters": 102237197}})()
        contract = {}
        attempts = 1440
        optimizer = object()
        def _optimizer(self):
            raise AssertionError("must not create another optimizer in this case")
        def step(self):
            raise AssertionError("must not perform another optimizer attempt")
    engine = Engine()
    engine.phase, engine.phase_step = phase, step
    monkeypatch.setattr(learning, "Engine", lambda *_a, **_kw: engine)
    exp.save = lambda tag=None: saved.append(tag)
    exp.diagnostic = lambda **_kw: passed
    return exp, saved, engine


def test_gate_failure_prevents_next_phase(tmp_path, monkeypatch):
    exp, saved, engine = fixture_experiment(tmp_path, monkeypatch, 0, 480, False)
    assert exp.train() == "FUNCTIONAL_GATE"
    assert saved == ["initial", "gate_failed"]
    assert engine.phase == 0


def test_last_update_processes_final_gate_without_extra_attempt(tmp_path, monkeypatch):
    exp, saved, engine = fixture_experiment(tmp_path, monkeypatch, 2, 640, True)
    optimizer = engine.optimizer
    assert exp.train() == "TRAINING_COMPLETE"
    assert saved == ["initial", "phase_2_passed", "final"]
    assert engine.phase == 3 and engine.optimizer is optimizer
