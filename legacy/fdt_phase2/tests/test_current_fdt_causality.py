import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
FDT2_PATH = ROOT / "FDT2" / "SD.py"


def load_fdt2_module():
    spec = importlib.util.spec_from_file_location("fdt2_sd", FDT2_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tiny_config(module):
    return module.FDTConfig(
        model_name="tiny-test",
        max_len=12,
        batch_size=2,
        dim=16,
        num_anchors=8,
        top_k=3,
        use_dual_routing=False,
        role_num_anchors=4,
        semantic_num_anchors=4,
        role_top_k=2,
        semantic_top_k=2,
        num_layers=1,
        ffn_mult=2,
        dropout=0.0,
        init_from_pretrained_embeddings=False,
        tie_lm_head=False,
        freeze_token_embedding=False,
        amp=False,
    )


def test_causal_shift_uses_next_token_labels():
    module = load_fdt2_module()
    config = tiny_config(module)
    model = module.FDTLanguageModel(config, vocab_size=32, pad_token_id=0)
    trainer = module.FDTTrainer(model, config, torch.device("cpu"))

    batch = {
        "input_ids": torch.tensor([[10, 11, 12, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "label_weights": torch.tensor([[0.0, 0.1, 1.0, 0.0]]),
    }
    model_input, labels, model_mask, weights = trainer._shift_batch(batch)

    assert model_input.tolist() == [[10, 11, 12]]
    assert labels.tolist() == [[11, 12, -100]]
    assert model_mask.tolist() == [[1, 1, 1]]
    assert torch.allclose(weights, torch.tensor([[0.1, 1.0, 0.0]]))


@torch.no_grad()
def test_no_future_leakage_in_prefix_logits():
    module = load_fdt2_module()
    torch.manual_seed(7)
    config = tiny_config(module)
    model = module.FDTLanguageModel(config, vocab_size=64, pad_token_id=0)
    model.eval()

    prefix = [1, 2, 3, 4]
    seq_a = torch.tensor([prefix + [5, 6, 7]], dtype=torch.long)
    seq_b = torch.tensor([prefix + [31, 32, 33]], dtype=torch.long)
    mask = torch.ones_like(seq_a)
    t = torch.zeros(1)

    logits_a = model(input_ids=seq_a, timesteps=t, attention_mask=mask)["logits"]
    logits_b = model(input_ids=seq_b, timesteps=t, attention_mask=mask)["logits"]

    assert torch.allclose(logits_a[:, : len(prefix), :], logits_b[:, : len(prefix), :], atol=1e-5)


@torch.no_grad()
def test_sparse_membership_is_finite_and_sums_to_one():
    module = load_fdt2_module()
    torch.manual_seed(11)
    config = tiny_config(module)
    layer = module.FuzzySemanticRoutingLayer(config, num_anchors=8, top_k=3)
    x = torch.randn(2, 6, config.dim)
    mask = torch.ones(2, 6, dtype=torch.long)

    _routed, stats = layer(x, attention_mask=mask)
    sums = stats.membership.sum(dim=-1)

    assert torch.isfinite(stats.membership).all()
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
    assert stats.indices.shape[-1] == 3
