import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model
from fdt_rlm.next_tools import choose_onk_cache
from generate_probe import load_prompts, run_probes


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(char) % 20) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(map(str, ids))


def test_generation_probe_loads_and_runs_one_bounded_prompt():
    prompts = load_prompts(ROOT / "eval_prompts" / "base_probe_prompts.jsonl", 1)
    cfg = ModelConfig(vocab_size=32, pad_token_id=0, eos_token_id=1, model_type="fdt_anchor_mixer", dim=16, n_layers=1, n_heads=4, mlp_ratio=2, max_seq_len=64, dropout=0.0, num_anchors=8, top_k=2)
    model = build_model(cfg).eval()
    args = SimpleNamespace(max_new_tokens=2, temperature=0.0, top_p=1.0, greedy=True, use_cache=True, seed=1)
    rows = run_probes(model, TinyTokenizer(), cfg, prompts, args, torch.device("cpu"))
    assert len(rows) == 1
    assert 0 <= rows[0]["new_tokens"] <= 2


def test_onk_cache_policy_uses_measured_long_bf16_regime():
    cfg = ModelConfig(model_type="fdt_optimized", use_self_attention=False, max_seq_len=512, use_incremental_anchor_state=True)
    assert not choose_onk_cache(cfg, 128, "bf16")
    assert choose_onk_cache(cfg, 384, "bf16")
    assert not choose_onk_cache(cfg, 384, "fp32")
    assert choose_onk_cache(cfg, 128, "fp32", memory_priority=True)
