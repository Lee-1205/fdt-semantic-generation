import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fdt_rlm.tokenization import SPECIAL_TOKENS, load_tokenizer


def test_gpt2_bootstrap_tokenizer_roundtrip():
    tok = load_tokenizer("gpt2")
    samples = [
        "The model routes tokens through fuzzy anchors.",
        "def calculate_loss(logits, targets):\n    return F.cross_entropy(logits, targets)",
        '{"action":"SEARCH","query":"optimizer step"}',
        "## Training\nRun `python train.py --config config.yaml`.",
        "src/fdt_rlm/models/causal_lm.py",
    ]
    for text in samples:
        ids = tok.encode(text, add_special_tokens=False)
        assert ids
        decoded = tok.decode(ids)
        assert decoded == text


def test_required_special_token_list_unique():
    assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS))
    assert "<rlm>" in SPECIAL_TOKENS
