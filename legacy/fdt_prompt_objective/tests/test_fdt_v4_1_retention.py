from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str, filename: str):
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


trainer = load_script("fdt_v4_1_retention_trainer_test", "train_fdt_v4_1_retention.py")
views = load_script("fdt_v4_1_retention_views_test", "prepare_fdt_v4_1_retention_source_views.py")


def test_category_source_discovery_is_explicit_and_complete():
    data = {
        "natural_lm_dir": "prepared_data/natural",
        "factual_dir": "prepared_data/factual",
        "retrieval_dir": "prepared_data/retrieval",
        "python_dir": "prepared_data/python",
        "json_dir": "prepared_data/json",
        "exact_copy_dir": "prepared_data/exact",
    }
    resolved = trainer.configured_training_dataset_dirs(data)
    assert set(resolved) == {
        "natural",
        "factual",
        "retrieval",
        "python",
        "json",
        "exact_copy",
    }


def test_retention_weights_require_natural_factual_majority():
    accepted = {
        "natural": 0.45,
        "factual": 0.35,
        "retrieval": 0.10,
        "python": 0.05,
        "json": 0.05,
    }
    trainer.validate_retention_source_fractions(accepted)
    rejected = dict(accepted, natural=0.35, factual=0.30, retrieval=0.25)
    with pytest.raises(ValueError, match="at least 75%"):
        trainer.validate_retention_source_fractions(rejected)


def test_token_planner_hits_all_category_budgets_and_exactly_resumes():
    fractions = {
        "natural": 0.45,
        "factual": 0.35,
        "retrieval": 0.10,
        "python": 0.05,
        "json": 0.05,
    }
    planner = trainer.TokenBudgetSourcePlanner(fractions)
    lengths = {"natural": 493, "factual": 47, "retrieval": 291, "python": 217, "json": 91}
    for _ in range(50_000):
        source = planner.choose_source()
        planner.record(source, lengths[source])
    total = sum(planner.active_tokens.values())
    for source, expected in fractions.items():
        assert abs(planner.active_tokens[source] / total - expected) < 0.001
    restored = trainer.TokenBudgetSourcePlanner(fractions)
    restored.restore(planner.state())
    assert restored.state() == planner.state()
    assert restored.choose_source() == planner.choose_source()


def test_category_view_preserves_every_row_field():
    payload = {
        "input_ids": torch.arange(48).view(6, 8),
        "labels": torch.arange(48).view(6, 8),
        "attention_mask": torch.ones(6, 8, dtype=torch.uint8),
        "category_ids": torch.tensor([0, 1, 0, 1, 0, 1]),
        "grounding_mask": torch.zeros(6, 8, dtype=torch.uint8),
        "signature_mask": torch.ones(6, 8, dtype=torch.uint8),
    }
    selected = views.select_category(payload, 1)
    assert set(selected) == set(payload)
    assert selected["input_ids"].shape == (3, 8)
    assert selected["category_ids"].eq(1).all()


def test_manifest_row_count_accepts_category_views_and_standard_manifests():
    assert trainer.manifest_declared_rows(
        {"category": {"rows": 20_000}}, "train"
    ) == 20_000
    assert trainer.manifest_declared_rows(
        {"shards": [{"rows": 11}, {"rows": 13}]}, "train"
    ) == 24
    assert trainer.manifest_declared_rows({"train": {"rows": 31}}, "train") == 31
    assert trainer.manifest_declared_rows({"rows": 52}, "validation") == 52


def test_exact_pointer_group_can_be_frozen_at_zero_learning_rate():
    from fdt_rlm.config import ModelConfig
    from fdt_rlm.models import build_model

    config = ModelConfig(
        model_type="fdt_v4",
        vocab_size=97,
        pad_token_id=0,
        eos_token_id=1,
        dim=32,
        n_layers=2,
        n_heads=4,
        mlp_ratio=2,
        max_seq_len=32,
        dropout=0.0,
        use_rope=True,
        local_attention_window=8,
        anchor_layer_indices=[0],
        num_anchors=16,
        top_k=4,
        router_dim=16,
        exact_memory_enabled=True,
        exact_memory_mode="copy",
    )
    model = build_model(config)
    groups, _, exact = trainer.optimizer_parameter_groups(
        model,
        {
            "learning_rate": 1e-8,
            "weight_decay": 0.01,
            "exact_pointer_learning_rate": 0.0,
            "exact_pointer_weight_decay": 0.0,
        },
    )
    assert exact
    by_name = {group["group_name"]: group for group in groups}
    assert by_name["exact_pointer"]["lr"] == 0.0


def test_retention_trainer_keeps_no_training_preflight_mode():
    source = (ROOT / "scripts" / "train_fdt_v4_1_retention.py").read_text(
        encoding="utf-8"
    )
    assert 'parser.add_argument("--preflight-only", action="store_true")' in source
    assert '"training_launched": False' in source
    assert source.index("if args.preflight_only:") < source.index(
        "optimizer_groups, base_trainable, exact_trainable"
    )


def test_fixed52_dual_snapshot_keeps_row_and_token_aggregates_distinct():
    class ControlledModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))

        def forward(self, input_ids, attention_mask):
            logits = torch.zeros(input_ids.size(0), input_ids.size(1), 8)
            if int(input_ids[0, 0]) == 1:
                for position in range(input_ids.size(1) - 1):
                    logits[0, position, int(input_ids[0, position + 1])] = 4.0
            return {"logits": logits + self.marker}

    payload = {
        "input_ids": torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]]),
        "labels": torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]]),
        "attention_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
    }
    metrics = trainer.fixed52_dual_snapshot(
        ControlledModel(), payload, torch.device("cpu")
    )
    confident = math.log(math.exp(4.0) + 7.0) - 4.0
    uniform = math.log(8.0)
    assert metrics["row_mean_nll"] == pytest.approx((confident + uniform) / 2.0)
    assert metrics["token_weighted_nll"] == pytest.approx(
        (confident + 3.0 * uniform) / 4.0
    )
    assert metrics["tokens"] == 4


def test_fixed52_gate_stops_on_either_aggregate_or_nonfinite():
    base = {
        "rows": 52,
        "tokens": 6231,
        "row_mean_nll": 3.35,
        "token_weighted_nll": 3.57,
    }
    assert not trainer.fixed52_gate_exceeded(
        base,
        parent_row_mean_nll=3.3217333062336993,
        parent_token_weighted_nll=3.5506424759425697,
        max_relative_regression=0.01,
    )
    assert trainer.fixed52_gate_exceeded(
        dict(base, row_mean_nll=3.36),
        parent_row_mean_nll=3.3217333062336993,
        parent_token_weighted_nll=3.5506424759425697,
        max_relative_regression=0.01,
    )
    assert trainer.fixed52_gate_exceeded(
        dict(base, token_weighted_nll=float("nan")),
        parent_row_mean_nll=3.3217333062336993,
        parent_token_weighted_nll=3.5506424759425697,
        max_relative_regression=0.01,
    )


def test_fixed52_manifest_pins_every_row_identity():
    manifest_path = (
        ROOT
        / "prepared_data"
        / "fdt_v4_1_fixed52_dual_validation_20260903_r1"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identities = manifest["row_identity_sha256"]
    assert manifest["purpose"] == "validation_only_never_training"
    assert manifest["source_sha256"] == (
        "7F3A28AA049904A4D715ACE2307CDF7A80E7D6E2EB65A707EAAD8764AB908E9E"
    )
    assert len(identities) == len(set(identities)) == 52
    assert manifest["ordered_row_identity_sha256"] == (
        "10C85D1D6DD7CF56BF34E9AD577BA5A34E345C670E87DF5B8A67536ED915E0A7"
    )


def test_fixed52_gate_state_is_persisted_for_resume():
    source = (ROOT / "scripts" / "train_fdt_v4_1_retention.py").read_text(
        encoding="utf-8"
    )
    assert '"last_fixed52_metrics"' in source
    assert '"last_fixed52_step"' in source
    assert '"fixed52_gate_triggered"' in source
    assert '"event": "fixed52_validation"' in source
    assert '"operations": "unquantized_fp32"' in source


def test_fixed52_startup_gate_precedes_any_optimizer_update():
    source = (ROOT / "scripts" / "train_fdt_v4_1_retention.py").read_text(
        encoding="utf-8"
    )
    startup = source.index('"event": "fixed52_startup_validation"')
    first_update = source.index("optimizer.step()")
    assert startup < first_update
    assert '"optimizer_updates_before_evaluation": 0' in source
    assert '"failure": "fixed52_startup_regression_gate"' in source


def test_fixed52_failed_checkpoint_is_not_resumable():
    assert not trainer.fixed52_gate_was_triggered({})
    assert trainer.fixed52_gate_was_triggered({"fixed52_gate_triggered": True})
    assert trainer.fixed52_gate_was_triggered(
        {"startup_fixed52_gate_triggered": True}
    )


def test_alpha020_smoke_changes_only_the_predeclared_control_and_fresh_seed():
    old = trainer.load_yaml_like(
        ROOT / "configs" / "fdt_v4_1_424m_alpha024_retention_smoke_500k_20260903.yaml"
    )
    new = trainer.load_yaml_like(
        ROOT / "configs" / "fdt_v4_1_424m_alpha020_retention_smoke_500k_20260906.yaml"
    )
    assert new["model"]["rope_transition_alpha"] == 0.20
    assert new["train"]["rope_transition_alpha_start"] == 0.20
    assert new["train"]["rope_transition_alpha_end"] == 0.20
    assert new["train"]["source_token_fractions"] == old["train"]["source_token_fractions"]
    assert new["train"]["parent_checkpoint"] == old["train"]["parent_checkpoint"]
    normalized = copy.deepcopy(new)
    normalized["stage"] = old["stage"]
    normalized["model"]["rope_transition_alpha"] = old["model"]["rope_transition_alpha"]
    normalized["train"]["rope_transition_alpha_start"] = old["train"]["rope_transition_alpha_start"]
    normalized["train"]["rope_transition_alpha_end"] = old["train"]["rope_transition_alpha_end"]
    normalized["train"]["seed"] = old["train"]["seed"]
    normalized["train"]["overfit_gate_version"] = old["train"]["overfit_gate_version"]
    assert normalized == old
