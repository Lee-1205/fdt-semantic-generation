import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_capability_v2 import CATEGORIES, score_case, summarize


def test_capability_v2_has_separate_categories_and_scores():
    assert set(CATEGORIES) == {"pure_continuation", "code_prefix_completion", "json_prefix_completion", "instruction_action_generation", "retrieval_continuation", "retrieval_instruction"}
    score = score_case("instruction_action_generation", {"gold": "STOP"}, '{"action":"STOP"}')
    assert score == {"nonempty": True, "valid_json": True, "correct": True}
    summary = summarize([{"score": score}])
    assert summary["metrics"]["correct_rate"] == 1.0
