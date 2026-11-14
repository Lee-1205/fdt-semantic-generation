import json
from pathlib import Path

import torch

from research import alignment_transfer_querycount_v1 as data_builder
from scripts import run_alignment_trajectory_20261027 as worker
from scripts import supervise_alignment_trajectory_20261027 as supervisor


class TinyCounter:
    def __init__(self): self.calls = 0
    def charge(self, kind): self.calls += 1


class TinyDenoiser:
    def __init__(self): self.calls = 0
    def denoise(self, z, times, enc, ablate=False):
        self.calls += 1
        return torch.tanh(z) * 0.1 + times[:, None, None] * 0.01 * enc["query_mask"][..., None]


def test_traced_schedule_final_matches_original_expression_and_exact_call_count():
    model = TinyDenoiser()
    counter = TinyCounter()
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    encoded = {"init": torch.zeros(2, 3, 5), "query_mask": mask}
    noise = torch.arange(30, dtype=torch.float32).reshape(2, 3, 5) / 31
    final, trace = worker.rollout_trace(model, encoded, noise, counter, autocast_cuda=False)
    reference = worker.FDT100.rollout(model, encoded, noise, schedule=worker.SCHEDULE)
    assert torch.equal(final, reference)
    assert torch.equal(trace["final"], reference)
    assert len(trace["steps"]) == 4
    assert counter.calls == 4 and model.calls == 8  # trace is four calls; reference adds no calls to traced budget
    for step in trace["steps"]:
        assert all(torch.isfinite(step[k]).all() for k in ("x_t", "velocity", "predicted_clean", "predicted_eps", "x_next"))


def test_nonfinite_latent_trace_fails_closed():
    class BadDenoiser(TinyDenoiser):
        def denoise(self, z, times, enc):
            return torch.full_like(z, float("nan"))
    encoded = {"init": torch.zeros(1, 1, 2), "query_mask": torch.ones(1, 1, dtype=torch.bool)}
    try:
        worker.rollout_trace(BadDenoiser(), encoded, torch.zeros(1, 1, 2), TinyCounter(), autocast_cuda=False)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("nonfinite denoiser output must stop the trace")


def test_fixed_graph_indices_target_query_and_saved_noise_are_pinned():
    root = data_builder.ROOT
    dataset = root / "research/artifacts/alignment_transfer_querycount_20261027_v1"
    rows = {q: json.loads((dataset / f"q{q:02d}.json").read_text(encoding="utf-8")) for q in worker.QUERY_COUNTS}
    noise_path = root / "runs/alignment_transfer_querycount_20261027_v1/paired_seed17_noise.pt"
    noise = torch.load(noise_path, map_location="cpu", weights_only=True)
    plan = {"graph_indices": [8, 9, 10, 11], "query_counts": [4, 8, 16]}
    contract = worker.validate_fixed_case(rows, noise, plan)
    assert contract["target_case"] == {"graph_index": 8, "slot": 3, "query": [4, 6],
        "gold_class": 1, "gold_name": "GREATER"}
    assert contract["noise_sha256"] == worker.SOURCE_NOISE_SHA256
    assert noise.shape == (16, 16, 512) and noise.dtype == torch.float32


def test_trace_call_budget_is_30_plus_separate_six_readout_batches():
    counts = worker.expected_call_counts()
    assert counts["actual_encode_denoise_calls"] == 30
    assert counts["max_actual_encode_denoise_calls"] == 40
    assert counts["readout_head_calls_separate"] == 6
    assert counts["graph_count"] == 4


def test_collated_gold_and_metadata_are_not_model_inputs():
    q4 = json.loads((data_builder.ROOT / "research/artifacts/alignment_transfer_querycount_20261027_v1/q04.json").read_text(encoding="utf-8"))[8:12]
    features, targets, _ = data_builder.collate(q4, data_builder.Config(), "cpu")
    assert set(features) == worker.FEATURE_KEYS
    assert features["query_mask"].sum(-1).tolist() == [4, 4, 4, 4]
    assert targets["labels"].shape == (4, 16)
    assert "labels" not in features and "hops" not in features


def test_trace_cap_and_output_conflicts_are_enforced():
    assert worker.MAX_TRACE_BYTES == 128 * 1024 * 1024
    worker.output_name_contract({"plan.json", "readiness.json", "supervisor", "worker.stderr.log"}, execution=True)
    for names in ({"plan.json", "optimizer.pt"}, {"plan.json", "final_status.json"}):
        try:
            worker.output_name_contract(names, execution=True)
        except FileExistsError:
            pass
        else:
            raise AssertionError("reserved model/status output must fail closed")


def test_duplicate_guard_uses_first_invoked_script_argument():
    command = (r'python.exe -B C:\proj\scripts\supervise_alignment_trajectory_20261027.py '
        r'--worker C:\proj\scripts\run_alignment_trajectory_20261027.py --plan p --output o')
    worker_command = r'python.exe C:\proj\scripts\run_alignment_trajectory_20261027.py --plan p'
    active = {"c:/proj/scripts/run_alignment_trajectory_20261027.py"}
    assert not supervisor.is_active_worker_command(command, active)
    assert supervisor.is_active_worker_command(worker_command, active)


def test_deadline_includes_startup_and_proxy_exit_requires_owned_child_exit():
    deadlines = supervisor.fixed_deadlines(2500.0)
    assert deadlines == (2590.0, 2570.0)
    alive = {"all_owned_processes_gone": False, "actual_process_after_exit": {"ProcessId": 888},
        "lingering_owned_processes": [{"ProcessId": 888}]}
    assert supervisor.natural_terminal_status(alive, 0, True, None) == "UNCONFIRMED_EXIT"
    clean = {"all_owned_processes_gone": True, "actual_process_after_exit": None, "lingering_owned_processes": []}
    assert supervisor.natural_terminal_status(clean, 0, True, None) == "COMPLETE"


def test_monitor_exception_cleanup_stops_only_owned_child_and_rechecks_exit(monkeypatch):
    class Child:
        pid = 6789
        returncode = None
        def poll(self): return self.returncode
        def wait(self, timeout=None):
            self.returncode = -9
            return self.returncode
    child = Child()
    stopped, commands = [], []
    def fake_run(args, **kwargs):
        commands.append(args); child.returncode = -9
        class Result: pass
        return Result()
    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    monkeypatch.setattr(supervisor, "write_stop", lambda control, reason: stopped.append(reason))
    monkeypatch.setattr(supervisor, "process_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(supervisor, "process_inventory", lambda root: [])
    result = supervisor.cleanup_owned_after_exception(child, None, Path("worker.py"), Path("plan.json"),
        Path("out"), Path("root"), {"ProcessId": child.pid}, "process_identity_error")
    assert stopped == ["process_identity_error"]
    assert commands[0][:4] == ["taskkill.exe", "/PID", str(child.pid), "/T"]
    assert result["status"] == "STOPPED"
    assert result["exit_verification"]["all_owned_processes_gone"]
