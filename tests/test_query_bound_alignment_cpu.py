from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "research"))
import run_query_bound_alignment_20261027 as worker
import supervise_query_bound_alignment_20261027 as supervisor


PANELS = ("primary_querymix_dev64", "uniform_dev64", "old_role_dev128",
          "old_normal16", "old_large16", "paired_new_ids16")


def panel(accuracy=.95, macro=.95, ground=.96, initializer=.96):
    return {"ground_accuracy": ground,
            "results": {"ddim_4": {"accuracy": accuracy, "macro_f1": macro},
                        "initializer": {"accuracy": initializer}}}


def test_phase2_budget_and_parent_counter_contract_is_distinct():
    plan = {"phase": 2, "target_successful_updates": 64, "max_attempts": 70, "seed": 24,
            "parent_updates": 256, "parent_attempts": 256, "parent_sha256": worker.PARENT_SHA256,
            "parent_query_mix_updates": 256, "ancestor_active_balanced_updates": 128,
            "ancestor_role8_updates": 128, "ancestor_e2_updates": 128, "ancestor_e1_updates": 1440,
            "phase_steps": [480, 320, 256], "trainable_prefixes": worker.TRAINABLE_PREFIXES,
            "parent_optimizer_imported": False, "parent_rng_imported": False,
            "e2_final_split_evaluated": False}
    worker.validate_phase2_contract(plan)
    plan["target_successful_updates"] = 256
    try:
        worker.validate_phase2_contract(plan)
    except ValueError:
        pass
    else:
        raise AssertionError("must reject inherited 256-update target")


def test_baseline_alignment_qualification_skips_all_optimizer_steps():
    baseline = {key: panel(.96) for key in PANELS}
    parent = {key: panel(.96) for key in PANELS}
    assert worker.baseline_alignment_action(baseline, parent) == ("BASELINE_ALIGNMENT_MET_NO_OPTIMIZER", [])


def test_unqualified_baseline_with_controls_preserved_enters_bounded_stage():
    baseline = {key: panel(.95) for key in PANELS}
    baseline["primary_querymix_dev64"] = panel(.93)
    baseline["paired_new_ids16"] = panel(.93)
    parent = {key: panel(.95) for key in PANELS}
    action, failures = worker.baseline_alignment_action(baseline, parent)
    assert action == "TRAIN_BOUNDED" and failures == []


def test_baseline_parent_control_damage_stops_before_optimizer():
    baseline = {key: panel(.92, .90) for key in PANELS}
    parent = {key: panel(.98, .98) for key in PANELS}
    action, failures = worker.baseline_alignment_action(baseline, parent)
    assert action == "STOP_BEFORE_OPTIMIZER_LEGACY_GUARD"
    assert failures
    assert all(f["guard"] == "qualified_control" for f in failures)


def test_control_retention_floors_use_parent_and_candidate_baselines():
    baseline = {key: panel(.95, .95) for key in PANELS}
    parent = {key: panel(.98, .98) for key in PANELS}
    current = {key: panel(.955, .955) for key in PANELS}
    fails = worker.retention_failures(baseline, current, parent)
    controls = [f for f in fails if f["panel"] in worker.CONTROL_RETENTION_PANELS]
    assert controls and all(f["floor"] == .96 for f in controls if f["metric"] == "accuracy")
    assert all(f["guard"] == "qualified_control" for f in controls)


def test_query_bound_candidate_still_has_exact_fdt_state_contract():
    from fdt_local.model_masked import Config, FDT100
    from query_bound_refiner_v1 import QueryBoundRefiner
    import torch
    torch.manual_seed(89)
    base = FDT100(Config.smoke())
    bound = QueryBoundRefiner(Config.smoke())
    assert list(base.state_dict()) == list(bound.state_dict())
    assert base.counts()["parameters"] == bound.counts()["parameters"]
    bound.load_state_dict(base.state_dict(), strict=True)


def test_duplicate_scan_does_not_mistake_own_supervisor_worker_argument_for_worker():
    import json
    from types import SimpleNamespace
    supervisor_path = Path(supervisor.__file__).resolve()
    worker_path = Path(worker.__file__).resolve()
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    item = {"ProcessId": 43210, "ExecutablePath": str(venv_python),
            "CommandLine": f'"{venv_python}" "{supervisor_path}" --worker "{worker_path}" --validate'}
    old_run = supervisor.subprocess.run
    supervisor.subprocess.run = lambda *a, **k: SimpleNamespace(stdout=json.dumps(item), stderr="", returncode=0)
    try:
        assert supervisor.find_existing_training(ROOT, 12345) == []
    finally:
        supervisor.subprocess.run = old_run


def test_duplicate_scan_flags_actual_worker_invocation():
    import json
    from types import SimpleNamespace
    worker_path = Path(worker.__file__).resolve()
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    item = {"ProcessId": 54321, "ExecutablePath": str(venv_python),
            "CommandLine": f'"{venv_python}" "{worker_path}" --plan draft.json --output new-run'}
    old_run = supervisor.subprocess.run
    supervisor.subprocess.run = lambda *a, **k: SimpleNamespace(stdout=json.dumps(item), stderr="", returncode=0)
    try:
        found = supervisor.find_existing_training(ROOT, 12345)
        assert len(found) == 1 and found[0]["pid"] == 54321
    finally:
        supervisor.subprocess.run = old_run


def test_duplicate_scan_keeps_prior_querymix_and_readonly_workers_visible():
    import json
    from types import SimpleNamespace
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    old_run = supervisor.subprocess.run
    try:
        for script_name, pid in (("run_query_mix_diffusion_20261027.py", 60101),
                                 ("run_query_interference_diagnostic_20261027.py", 60102)):
            script = ROOT / "scripts" / script_name
            item = {"ProcessId": pid, "ExecutablePath": str(venv_python),
                    "CommandLine": f'"{venv_python}" "{script}" --plan plan.json'}
            supervisor.subprocess.run = lambda *a, _item=item, **k: SimpleNamespace(
                stdout=json.dumps(_item), stderr="", returncode=0)
            found = supervisor.find_existing_training(ROOT, 12345)
            assert len(found) == 1 and found[0]["pid"] == pid
    finally:
        supervisor.subprocess.run = old_run


def test_draft_pins_same_seed17_step256_parent_references_not_final_averages():
    import json
    plan_path = ROOT / "research/artifacts/query_bound_alignment_20261027_v1/cpu_preflight_draft.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["authorized"] is False
    assert set(plan["parent_full_reference_paths"]) == set(PANELS)
    for path in plan["parent_full_reference_paths"].values():
        assert "diagnostics/step256" in path.replace("\\", "/")
        assert "final_evaluation" not in path.replace("\\", "/")
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        assert raw["noise_seeds"] == [17]
