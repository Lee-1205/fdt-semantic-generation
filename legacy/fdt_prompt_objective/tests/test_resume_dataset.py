import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.data.state import DatasetState


def test_dataset_state_roundtrip(tmp_path):
    path = tmp_path / "state.json"
    s = DatasetState(documents_seen=3, documents_kept=2, tokens_by_bucket={"fineweb": 10})
    s.save(path)
    loaded = DatasetState.load(path)
    assert loaded.documents_seen == 3
    assert loaded.tokens_by_bucket["fineweb"] == 10

