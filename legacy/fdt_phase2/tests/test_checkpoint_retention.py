import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_training_script():
    path = ROOT / "scripts" / "train_next_model.py"
    spec = importlib.util.spec_from_file_location("train_next_model_retention", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prune_step_checkpoints_keeps_newest_and_named_outputs(tmp_path):
    module = load_training_script()
    for step in (10, 30, 20, 40):
        (tmp_path / f"step_{step}.pt").write_bytes(b"checkpoint")
    (tmp_path / "latest.pt").write_bytes(b"latest")
    (tmp_path / "onk_ready.pt").write_bytes(b"best")

    removed = module.prune_step_checkpoints(tmp_path, keep=2)

    assert sorted(removed) == ["step_10.pt", "step_20.pt"]
    assert sorted(path.name for path in tmp_path.glob("step_*.pt")) == [
        "step_30.pt",
        "step_40.pt",
    ]
    assert (tmp_path / "latest.pt").is_file()
    assert (tmp_path / "onk_ready.pt").is_file()


def test_link_checkpoint_reuses_the_same_file_bytes(tmp_path):
    module = load_training_script()
    source = tmp_path / "latest.pt"
    destination = tmp_path / "step_10.pt"
    source.write_bytes(b"checkpoint")

    module.link_checkpoint(source, destination)

    assert destination.read_bytes() == b"checkpoint"
    assert source.stat().st_ino == destination.stat().st_ino
