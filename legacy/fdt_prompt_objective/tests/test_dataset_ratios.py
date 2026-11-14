import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.data.streaming_mix import build_token_plan


def test_100m_default_budget():
    plan = build_token_plan(100_000_000)
    assert plan.quotas["fineweb_edu"] == 60_000_000
    assert plan.quotas["fineweb"] == 15_000_000
    assert plan.quotas["code_python"] == 11_250_000
    assert plan.quotas["code_javascript_typescript"] == 5_000_000
    assert sum(plan.quotas.values()) == 100_000_000


def test_custom_quality_text_mix_is_supported():
    plan = build_token_plan(
        1_000_000,
        {"tinystories": 0.25, "cosmopedia": 0.35, "fineweb_edu_dedup": 0.40},
    )
    assert plan.quotas == {
        "tinystories": 250_000,
        "cosmopedia": 350_000,
        "fineweb_edu_dedup": 400_000,
    }
