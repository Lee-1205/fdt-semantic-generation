import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.data.splitting import assign_split


def test_split_deterministic():
    ratios = {"train": 0.9, "validation": 0.05, "test": 0.05}
    assert assign_split("repo/file.py", ratios, seed=1) == assign_split("repo/file.py", ratios, seed=1)

