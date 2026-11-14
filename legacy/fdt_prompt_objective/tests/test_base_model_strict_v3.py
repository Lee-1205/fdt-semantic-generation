from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]


def load_eval_module():
    path = ROOT / "scripts" / "evaluate_base_model_strict_v3.py"
    spec = importlib.util.spec_from_file_location("evaluate_base_model_strict_v3_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_wilson_interval_handles_bounds_and_center():
    module = load_eval_module()
    assert module.wilson_interval(0, 0) == [0.0, 0.0]
    low, high = module.wilson_interval(5, 10)
    assert 0.2 < low < 0.3
    assert 0.7 < high < 0.8
    assert module.wilson_interval(0, 10)[0] == 0.0
    assert module.wilson_interval(10, 10)[1] == 1.0


def test_capability_ranges_preserve_manifest_order():
    module = load_eval_module()
    manifest = {
        "validation": {"categories": {"facts": 3, "retrieval": 2, "json": 1}}
    }
    assert module.capability_ranges(manifest) == [
        ("facts", 0, 3),
        ("retrieval", 3, 5),
        ("json", 5, 6),
    ]


def test_teacher_forced_summary_reports_row_and_token_statistics():
    module = load_eval_module()
    rows = [
        {"tokens": 2, "top1": 2, "top5": 2, "sequence_exact": True, "loss_sum": 1.0},
        {"tokens": 2, "top1": 0, "top5": 1, "sequence_exact": False, "loss_sum": 3.0},
    ]
    summary = module.summarize_teacher_forced_rows(rows)
    assert summary["loss"] == 1.0
    assert summary["top1"]["rate"] == 0.5
    assert summary["top5"]["rate"] == 0.75
    assert summary["sequence_exact"]["rate"] == 0.5
    assert summary["mean_row_top1"] == 0.5


def test_generation_scoring_is_category_specific():
    module = load_eval_module()
    assert module.score_capability_generation("retrieval", " AB-12", "AB12 trailing")["correct"]
    json_score = module.score_capability_generation("json", '{"ok":true}', ' {"ok":true} trailing')
    assert json_score["valid"] and json_score["correct"]
    python_score = module.score_capability_generation(
        "python_code", "def f():\n    return 1", "def f():\n    return 2"
    )
    assert python_score["parseable"]
    assert python_score["compilable"]
    assert python_score["usable_structure"]
    assert python_score["function_signature_match"]


def test_factual_scoring_uses_whole_word_boundaries():
    module = load_eval_module()
    assert module.score_capability_generation("factual_qa", "Paris", "Paris is correct")[
        "correct"
    ]
    assert not module.score_capability_generation("factual_qa", "Paris", "Parisian culture")[
        "correct"
    ]


def test_python_scoring_separates_structure_from_reference_ast():
    module = load_eval_module()
    gold = "def add(a, b):\n    return a + b"
    same = module.score_python_structure(gold, gold)
    wrong_body = module.score_python_structure("def add(a, b):\n    return a - b", gold)
    wrong_signature = module.score_python_structure("def subtract(a, b):\n    return a - b", gold)
    assert same["usable_structure"] and same["ast_exact"] and same["function_signature_match"]
    assert wrong_body["usable_structure"] and not wrong_body["ast_exact"]
    assert wrong_body["function_signature_match"]
    assert wrong_signature["usable_structure"] and not wrong_signature["function_signature_match"]


def test_python_scoring_ignores_outer_generation_whitespace():
    module = load_eval_module()
    generated = "  \n def add(a, b):\n    return a + b\n  "
    score = module.score_python_structure(generated, "def add(a, b):\n    return a + b")
    assert score["parseable"]
    assert score["compilable"]
    assert score["usable_structure"]
    assert score["ast_exact"]
    assert score["function_signature_match"]


def test_python_generation_scoring_rejects_non_compilable_and_repeated_functions():
    module = load_eval_module()

    duplicate_argument = module.score_python_structure("def f(a, a):\n    return a")
    assert duplicate_argument["parseable"]
    assert not duplicate_argument["compilable"]
    assert not duplicate_argument["usable_structure"]

    repeated = module.score_python_structure(
        "def f():\n    return 1\n\ndef f():\n    return 2"
    )
    assert repeated["compilable"]
    assert not repeated["usable_structure"]

    self_recursive = module.score_python_structure("def f():\n    return f()")
    assert self_recursive["compilable"]
    assert not self_recursive["usable_structure"]
