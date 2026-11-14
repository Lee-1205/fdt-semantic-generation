from pathlib import Path

import importlib.util


ROOT = Path(__file__).resolve().parents[1]


def load_training_module():
    path = ROOT / "scripts" / "train_next_model.py"
    spec = importlib.util.spec_from_file_location("train_next_model_stop", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_resolve_stop_file_defaults_to_run_directory(tmp_path):
    module = load_training_module()
    assert module.resolve_stop_file({}, tmp_path) == tmp_path / "STOP_REQUESTED"


def test_resolve_stop_file_supports_relative_and_absolute_paths(tmp_path):
    module = load_training_module()
    assert module.resolve_stop_file({"stop_file": "control/stop"}, tmp_path) == tmp_path / "control/stop"

    absolute = tmp_path / "absolute-stop"
    assert module.resolve_stop_file({"stop_file": str(absolute)}, tmp_path) == absolute
