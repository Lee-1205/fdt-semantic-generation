import torch

from fdt_rlm import next_tools
from fdt_rlm.config import ModelConfig
from fdt_rlm.models import build_model, causal_lm_loss


def config(model_type):
    return ModelConfig(vocab_size=97, pad_token_id=0, eos_token_id=1, model_type=model_type, dim=32, n_layers=2, n_heads=4, mlp_ratio=2, max_seq_len=16, dropout=0.0, num_anchors=8, top_k=2, use_rope=model_type != "fdt_v3")


def test_all_next_models_forward_and_backward():
    for model_type in ("transformer", "fdt", "fdt_optimized", "fdt_hybrid", "fdt_anchor_mixer", "fdt_v3"):
        model = build_model(config(model_type))
        ids = torch.randint(2, 97, (2, 8))
        output = model(ids, attention_mask=torch.ones_like(ids))
        assert output["logits"].shape == (2, 8, 97)
        loss = causal_lm_loss(output["logits"], ids, torch.ones_like(ids), 0)
        loss.backward()
        assert torch.isfinite(loss)


def test_next_checkpoint_round_trip(tmp_path):
    cfg = config("fdt_optimized")
    original = build_model(cfg)
    path = tmp_path / "tiny.pt"
    torch.save({"model_config": cfg.__dict__, "model_state_dict": original.state_dict()}, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    restored = build_model(ModelConfig(**payload["model_config"]))
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    assert all(torch.equal(a, b) for a, b in zip(original.state_dict().values(), restored.state_dict().values()))


def test_finetuned_checkpoint_inherits_base_tokenizer_metadata(tmp_path, monkeypatch):
    cfg = config("fdt_v3")
    model = build_model(cfg)
    base_path = tmp_path / "base.pt"
    child_path = tmp_path / "child.pt"
    torch.save({"train_config": {"tokenizer_name": "correct-tokenizer"}}, base_path)
    torch.save(
        {
            "model_config": cfg.__dict__,
            "model_state_dict": model.state_dict(),
            "base_checkpoint": str(base_path),
            "train_config": {},
        },
        child_path,
    )

    class StubTokenizer:
        def __len__(self):
            return cfg.vocab_size

    loaded_names = []

    def fake_load_tokenizer(name):
        loaded_names.append(name)
        return StubTokenizer()

    monkeypatch.setattr(next_tools, "load_tokenizer", fake_load_tokenizer)
    next_tools.load_model_and_tokenizer(child_path, "cpu")

    assert loaded_names == ["correct-tokenizer"]


def test_current_fdt_state_dict_strictly_loads_into_optimized_fdt():
    current = build_model(config("fdt"))
    optimized = build_model(config("fdt_optimized"))
    assert current.state_dict().keys() == optimized.state_dict().keys()
    optimized.load_state_dict(current.state_dict(), strict=True)


def test_learned_position_table_uses_small_initialization():
    cfg = config("fdt_optimized")
    cfg.use_rope = False
    model = build_model(cfg)
    assert model.position_embedding is not None
    assert model.position_embedding.weight.std().item() < 0.03


def test_new_token_and_untied_lm_embeddings_use_small_initialization():
    cfg = config("fdt_optimized")
    cfg.tie_embeddings = False
    model = build_model(cfg)
    assert model.token_embedding.weight.std().item() < 0.03
    assert model.lm_head.weight.std().item() < 0.03
    assert model.token_embedding.weight.data_ptr() != model.lm_head.weight.data_ptr()


@torch.no_grad()
def test_zero_initialized_local_mixer_preserves_checkpoint_function():
    old_cfg = config("fdt_optimized")
    old_cfg.use_rope = False
    old_model = build_model(old_cfg).eval()
    new_cfg = config("fdt_optimized")
    new_cfg.use_rope = False
    new_cfg.use_local_mixer = True
    new_cfg.local_mixer_kernel_size = 5
    new_model = build_model(new_cfg).eval()
    incompatible = new_model.load_state_dict(old_model.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert incompatible.missing_keys
    assert all(".local_" in key for key in incompatible.missing_keys)
    ids = torch.randint(2, 97, (2, 8))
    mask = torch.ones_like(ids)
    old_logits = old_model(ids, attention_mask=mask)["logits"]
    new_logits = new_model(ids, attention_mask=mask)["logits"]
    assert torch.equal(old_logits, new_logits)
