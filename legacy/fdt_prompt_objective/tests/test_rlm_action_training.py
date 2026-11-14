import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_rlm_actions.py"
SPEC = importlib.util.spec_from_file_location("train_rlm_actions", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_balanced_task_order_round_robins_families():
    tasks = [
        {"family": "a", "task_id": "a0"},
        {"family": "a", "task_id": "a1"},
        {"family": "b", "task_id": "b0"},
        {"family": "b", "task_id": "b1"},
    ]

    ordered = MODULE.balanced_task_order(tasks)

    assert [task["task_id"] for task in ordered] == ["a0", "b0", "a1", "b1"]
