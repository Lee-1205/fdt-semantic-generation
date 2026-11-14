from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_next_generation_files_exist():
    required = [
        "scripts/train_next_model.py",
        "scripts/benchmark_inference.py",
        "scripts/evaluate_capability_v2.py",
        "scripts/generate_probe.py",
        "scripts/profile_next_models.py",
        "scripts/benchmark_context_scaling.py",
        "RUN_NEXT_SMOKE.ps1",
        "RUN_NEXT_TRAIN.ps1",
        "RUN_NEXT_EVAL.ps1",
        "RUN_NEXT_INFER_BENCH.ps1",
        "RUN_NEXT_PROFILE.ps1",
    ]
    missing = [name for name in required if not (ROOT / name).is_file()]
    assert not missing
