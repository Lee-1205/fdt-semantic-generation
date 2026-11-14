import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "scripts" / "evaluate_lexical_pointer_v2.py"
    spec = importlib.util.spec_from_file_location("evaluate_lexical_pointer_v2", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]


def test_long_distance_prompt_adds_exact_filler_budget():
    module = load_module()
    tokenizer = CharacterTokenizer()
    code = "ABCD2345"

    length_64 = len(module.retrieval_prompt_ids(tokenizer, code, "distance_64"))
    length_512 = len(module.retrieval_prompt_ids(tokenizer, code, "distance_512"))
    length_1400 = len(module.retrieval_prompt_ids(tokenizer, code, "distance_1400"))

    assert length_512 - length_64 == 512 - 64
    assert length_1400 - length_512 == 1400 - 512


def test_proportion_reports_rate_and_bounded_wilson_interval():
    module = load_module()
    result = module.proportion(9, 10)

    assert result["rate"] == 0.9
    assert 0.0 <= result["wilson_95"][0] < result["rate"]
    assert result["rate"] < result["wilson_95"][1] <= 1.0
