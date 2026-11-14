import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.data.dedup import MemoryDeduper, normalized_text_hash


def test_normalized_hash_exact_duplicate():
    assert normalized_text_hash("Hello   world") == normalized_text_hash("Hello world")


def test_memory_deduper():
    d = MemoryDeduper()
    h = "abc"
    assert d.check_and_add(h, "a")
    assert not d.check_and_add(h, "b")

