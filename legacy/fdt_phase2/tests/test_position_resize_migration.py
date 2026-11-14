import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from train_next_model import load_resume_state
from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model


def make_config(max_seq_len: int) -> ModelConfig:
    return ModelConfig(
        vocab_size=41,
        pad_token_id=0,
        eos_token_id=1,
        model_type="transformer",
        dim=16,
        n_layers=1,
        n_heads=4,
        mlp_ratio=2,
        max_seq_len=max_seq_len,
        dropout=0.0,
        use_rope=False,
    )


def checkpoint_for(model, optimizer):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {"tie_embeddings": True},
    }


def test_position_resize_preserves_existing_rows_and_resets_optimizer():
    torch.manual_seed(52)
    old_model = build_model(make_config(8))
    old_optimizer = torch.optim.AdamW(old_model.parameters())
    checkpoint = checkpoint_for(old_model, old_optimizer)
    old_positions = old_model.position_embedding.weight.detach().clone()

    new_model = build_model(make_config(20))
    new_optimizer = torch.optim.AdamW(new_model.parameters())
    new_tail = new_model.position_embedding.weight.detach().clone()[8:]
    result = load_resume_state(
        new_model,
        new_optimizer,
        checkpoint,
        new_model.config,
        {
            "allow_position_embedding_resize": True,
            "reset_optimizer_on_migration": True,
        },
    )

    assert result["position_embedding_resized"] is True
    assert result["optimizer_loaded"] is False
    assert torch.equal(new_model.position_embedding.weight[:8], old_positions)
    assert torch.equal(new_model.position_embedding.weight[8:], new_tail)


def test_position_resize_requires_explicit_permission():
    old_model = build_model(make_config(8))
    old_optimizer = torch.optim.AdamW(old_model.parameters())
    new_model = build_model(make_config(20))
    new_optimizer = torch.optim.AdamW(new_model.parameters())
    with pytest.raises(RuntimeError):
        load_resume_state(
            new_model,
            new_optimizer,
            checkpoint_for(old_model, old_optimizer),
            new_model.config,
            {},
        )


def test_position_resize_can_repeat_trained_interior_rows():
    torch.manual_seed(61)
    old_model = build_model(make_config(8))
    old_optimizer = torch.optim.AdamW(old_model.parameters())
    old_positions = old_model.position_embedding.weight.detach().clone()
    new_model = build_model(make_config(20))
    new_optimizer = torch.optim.AdamW(new_model.parameters())
    load_resume_state(
        new_model,
        new_optimizer,
        checkpoint_for(old_model, old_optimizer),
        new_model.config,
        {
            "allow_position_embedding_resize": True,
            "reset_optimizer_on_migration": True,
            "position_resize_init": "repeat_interior",
        },
    )

    expected = old_positions[1:-1][torch.arange(12) % 6]
    assert torch.equal(new_model.position_embedding.weight[:8], old_positions)
    assert torch.equal(new_model.position_embedding.weight[8:], expected)
