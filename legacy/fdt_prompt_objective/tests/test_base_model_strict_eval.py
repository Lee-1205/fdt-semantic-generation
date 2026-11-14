from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]


def load_eval_module():
    path = ROOT / "scripts" / "evaluate_base_model_strict.py"
    spec = importlib.util.spec_from_file_location("evaluate_base_model_strict_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_evenly_spaced_covers_bucket_edges():
    module = load_eval_module()
    assert module.evenly_spaced(10, 20, 4) == [10, 13, 16, 19]
    assert module.evenly_spaced(3, 3, 4) == []


def test_bucket_ranges_follow_manifest_order():
    module = load_eval_module()
    manifest = {
        "seq_len": 4,
        "bucket_order": ["text", "code"],
        "shards": [
            {"split": "validation", "sequences": 3, "buckets": {"text": 8, "code": 4}},
        ],
    }
    assert module.bucket_ranges(manifest, "validation") == [
        ("text", 0, 2),
        ("code", 2, 3),
    ]


def test_json_prefix_and_repetition_metrics():
    module = load_eval_module()
    value, end = module.extract_json_prefix('{"ok":true} trailing')
    assert value == {"ok": True}
    assert end == len('{"ok":true}')
    assert module.repetition_rate([1, 2, 3, 1, 2, 3], width=3) > 0
    assert module.normalize_code(" AB-12 ") == "ab12"


def test_code_cloze_cases_have_distinct_valid_gold_options():
    module = load_eval_module()
    assert len(module.CODE_CLOZE_CASES) >= 8
    for prompt, options, gold in module.CODE_CLOZE_CASES:
        assert prompt.strip()
        assert len(set(options)) == 4
        assert gold == 0
        ast_source = prompt + options[gold]
        compile(ast_source, "<code-cloze>", "exec")
