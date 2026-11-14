import json
from pathlib import Path

import torch

from research import alignment_transfer_querycount_v1 as dataset
from scripts import run_alignment_transfer_querycount_20261027 as worker
from scripts import supervise_alignment_transfer_querycount_20261027 as supervisor


def test_budget_is_exact_for_four_nested_views_and_two_variants():
    expected = worker.expected_call_counts()
    assert expected["actual_batched_model_calls"] == 160
    assert expected["actual_batched_model_calls"] <= expected["max_actual_batched_model_calls"] == 200
    assert expected["active_query_positions_per_variant"] == 464
    assert expected["distinct_base_queries"] == 256
    assert expected["actual_encoder_graph_exposures_per_variant"] == 64
    assert expected["actual_encoder_graph_exposures_both_variants"] == 128
    assert expected["actual_premise_fact_instances_encoder_exposures_both_variants"] == 1160
    assert expected["readout_calls_not_in_approved_encode_denoise_call_counter"] == 64


def test_dataset_contract_and_prefixes_match_reviewed_artifact():
    root = dataset.ROOT / "research/artifacts/alignment_transfer_querycount_20261027_v1"
    registry = json.loads((root / "selection_registry.json").read_text(encoding="utf-8"))
    rows_by_q = {q: json.loads((root / f"q{q:02d}.json").read_text(encoding="utf-8"))
                 for q in dataset.QUERY_COUNTS}
    report = worker.validate_rows_by_view(rows_by_q, registry)
    assert report["graphs"] == 16 and report["query_count_views"] == 64
    assert report["active_query_positions_per_variant"] == 464
    assert report["distinct_base_queries"] == 256
    assert report["independence_claimed"] is False


def test_paired_noise_uses_original_batch_start_and_repeats_exactly():
    a = worker.paired_noise(8, 512, "cpu")
    b = worker.paired_noise(8, 512, "cpu")
    c = worker.paired_noise(12, 512, "cpu")
    assert a.shape == (4, 16, 512)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert worker.tensor_sha256(a[0:1]) == worker.tensor_sha256(b[0:1])


def test_scoring_uses_only_active_queries_and_reports_error_categories():
    rows = [{"gold": [1, 0, 3, 2], "pred": [2, 1, 0, 2]}]
    score = worker.summarize(rows)
    assert score["active_queries"] == 4
    assert score["confusion_gold_rows_pred_columns"] == [
        [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]]
    assert score["direction_inversions_1_vs_2"] == 1
    assert score["unknown_boundary_errors"] == 2
    assert score["conflict_boundary_errors"] == 1


def test_owner_output_allowlist_and_reserved_execution_markers():
    supervisor.validate_plan  # API exists for the draft-plan integration check.
    worker.output_name_contract({"plan.json", "readiness.json", "supervisor", "worker.stdout.log"},
        allow_execution=True)
    try:
        worker.output_name_contract({"plan.json", "diagnostic_result.json"})
    except FileExistsError:
        pass
    else:
        raise AssertionError("a pre-existing result must block the one-shot output")
    try:
        worker.output_name_contract({"plan.json", "optimizer.pt"}, allow_execution=True)
    except FileExistsError:
        pass
    else:
        raise AssertionError("model/optimizer artifacts are forbidden")


def test_duplicate_detection_uses_invoked_script_not_later_worker_argument():
    own_supervisor = (r'python.exe -B -X utf8 C:\proj\scripts\supervise_alignment_transfer_querycount_20261027.py '
        r'--worker C:\proj\scripts\run_alignment_transfer_querycount_20261027.py --plan x --output y')
    other_training = r'python.exe C:\proj\scripts\run_query_mix_diffusion_20261027.py --plan x'
    active = {"c:/proj/scripts/run_alignment_transfer_querycount_20261027.py",
        "c:/proj/scripts/run_query_mix_diffusion_20261027.py"}
    assert not supervisor.is_active_worker_command(own_supervisor, active)
    assert supervisor.is_active_worker_command(other_training, active)
    assert supervisor.invoked_script(own_supervisor).endswith("supervise_alignment_transfer_querycount_20261027.py")


def test_invocation_deadline_includes_preflight_and_startup_delay():
    started = 1000.0
    absolute, worker_deadline = supervisor.fixed_deadlines(started)
    assert absolute == 1120.0
    assert worker_deadline == 1090.0
    # A delayed preflight does not move either immutable deadline.
    assert supervisor.fixed_deadlines(started) == (absolute, worker_deadline)


def test_live_owned_child_after_proxy_exit_cannot_be_reported_complete():
    exit_info = {"all_owned_processes_gone": False, "actual_process_after_exit": {"ProcessId": 1234},
        "lingering_owned_processes": [{"ProcessId": 1234}]}
    assert supervisor.natural_terminal_status(exit_info, 0, True, None) == "UNCONFIRMED_EXIT"
    clean = {"all_owned_processes_gone": True, "actual_process_after_exit": None, "lingering_owned_processes": []}
    assert supervisor.natural_terminal_status(clean, 0, True, None) == "COMPLETE"


def test_monitor_exception_cooperatively_stops_owned_child_and_records_exit(monkeypatch):
    class Child:
        pid = 4321
        returncode = None
        def poll(self): return self.returncode
        def wait(self, timeout=None):
            self.returncode = -9
            return self.returncode

    child = Child()
    commands = []
    stopped = []
    def fake_run(args, **kwargs):
        commands.append(args)
        child.returncode = -9
        class Result: pass
        return Result()
    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    monkeypatch.setattr(supervisor, "write_stop", lambda control, reason: stopped.append(reason))
    monkeypatch.setattr(supervisor, "process_identity", lambda *args, **kwargs: None)
    monkeypatch.setattr(supervisor, "process_inventory", lambda root: [])
    result = supervisor.cleanup_owned_after_exception(child, None, Path("worker.py"), Path("plan.json"),
        Path("out"), Path("root"), {"ProcessId": child.pid}, "monitor_metadata_error")
    assert stopped == ["monitor_metadata_error"]
    assert commands and commands[0][:4] == ["taskkill.exe", "/PID", str(child.pid), "/T"]
    assert result["status"] == "STOPPED"
    assert result["exit_verification"]["all_owned_processes_gone"] is True

