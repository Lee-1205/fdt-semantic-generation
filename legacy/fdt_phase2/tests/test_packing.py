import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.data.packing import PackedTextDataset, pack_token_stream


def test_pack_token_stream():
    blocks = list(pack_token_stream(range(10), 4))
    assert blocks == [[0, 1, 2, 3], [4, 5, 6, 7]]


def test_packed_dataset_can_restart_from_factory():
    class Tokenizer:
        eos_token_id = 9

        @staticmethod
        def encode(text, add_special_tokens=False):
            return [int(part) for part in text.split()]

    calls = []

    def texts():
        calls.append(1)
        return iter(["1 2 3", "4 5 6"])

    dataset = PackedTextDataset(texts, Tokenizer(), seq_len=4)
    first = list(dataset)
    second = list(dataset)
    assert len(calls) == 2
    assert first[0]["input_ids"].tolist() == [1, 2, 3, 9]
    assert second[0]["input_ids"].tolist() == first[0]["input_ids"].tolist()
