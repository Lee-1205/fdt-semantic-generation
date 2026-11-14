import json

import pytest
import torch

from fdt_rlm.runtime.actions import Action, ActionName, ActionValidationError
from fdt_rlm.runtime.anchor_state import AnchorState
from fdt_rlm.runtime.budget import BudgetConfig, BudgetExceeded, BudgetTracker
from fdt_rlm.runtime.loop_detection import LoopDetector
from fdt_rlm.runtime.model_backend import LocalCausalBackend
from fdt_rlm.runtime.parser import parse_action
from fdt_rlm.runtime.recursive_controller import ModelStep, RecursiveController
from fdt_rlm.runtime.text_environment import TextEnvironment
from fdt_rlm.runtime.uncertainty_policy import UncertaintyFeatures, UncertaintyPolicy


def test_rlm_action_schema():
    action = Action.from_mapping({"action": "SEARCH", "query": "anchor"})
    assert action.name == ActionName.SEARCH
    with pytest.raises(ActionValidationError):
        Action.from_mapping({"action": "READ", "start": 3, "end": 2})


def test_rlm_parser_invalid_json():
    assert parse_action("not json").action is None
    repaired = parse_action('prefix {"action":"STOP","answer":"ok"} suffix')
    assert repaired.action is not None
    assert repaired.repaired


def test_rlm_budget():
    budget = BudgetTracker(BudgetConfig(max_steps=1))
    budget.consume_step()
    with pytest.raises(BudgetExceeded):
        budget.consume_step()


def test_rlm_loop_detection():
    detector = LoopDetector(max_repeated_action=2)
    action = Action.from_mapping({"action": "SEARCH", "query": "x"})
    assert not detector.observe(action)
    assert not detector.observe(action)
    assert detector.observe(action)


def test_rlm_depth_limit():
    class Backend:
        def generate_action(self, prompt, anchor_state):
            return ModelStep('{"action":"STOP","answer":"ok"}')

    controller = RecursiveController(Backend(), BudgetConfig(max_depth=1))
    with pytest.raises(BudgetExceeded):
        controller.run("q", TextEnvironment("context"), depth=2)


def test_anchor_state_update_and_reset():
    state = AnchorState(4, 3, alpha=0.5, norm_clip=1.0)
    metrics = state.update(torch.ones(4, 3), torch.tensor([1.0, 0.5, 0.0, 0.0]))
    assert metrics.active_anchors == 2
    assert state.state.norm() > 0
    assert torch.all(state.state.norm(dim=-1) <= 1.00001)
    state.reset()
    assert torch.equal(state.state, torch.zeros_like(state.state))
    assert torch.equal(state.confidence, torch.zeros_like(state.confidence))


def test_uncertainty_features_and_controller():
    features = UncertaintyFeatures(token_entropy=4.0, confidence=0.1)
    policy = UncertaintyPolicy()
    probability = policy.need_more_context_probability(features)
    assert 0.0 <= probability <= 1.0
    assert policy.action_hint(features) in {"STOP", "SEARCH", "CALL", "COMPARE"}


def test_stop_action():
    class Backend:
        def generate_action(self, prompt, anchor_state):
            return ModelStep('{"action":"STOP","answer":"4817"}', generated_tokens=5)

    result = RecursiveController(Backend()).run("code?", TextEnvironment("code 4817"))
    assert result.status == "stopped"
    assert result.answer == "4817"
    assert result.steps == 1


def test_recursive_call_isolation():
    class Backend:
        def __init__(self):
            self.root_calls = 0

        def generate_action(self, prompt, anchor_state):
            if "Depth: 1" in prompt:
                return ModelStep('{"action":"STOP","answer":"4817"}')
            self.root_calls += 1
            if self.root_calls == 1:
                return ModelStep('{"action":"SEARCH","query":"4817"}')
            if self.root_calls == 2:
                return ModelStep('{"action":"CALL","question":"What code?","context_ref":"result_0"}')
            return ModelStep('{"action":"STOP","answer":"4817"}')

    config = BudgetConfig(max_steps=8, max_depth=1, max_subcalls=1)
    result = RecursiveController(Backend(), config).run(
        "What code?", TextEnvironment("The code is 4817.")
    )
    assert result.status == "stopped"
    assert result.answer == "4817"
    assert result.depth == 0


def test_same_budget_across_models():
    class Backend:
        def generate_action(self, prompt, anchor_state):
            return ModelStep(json.dumps({"action": "STOP", "answer": "x"}))

    config = BudgetConfig(max_steps=4, max_depth=1, max_subcalls=2)
    transformer = RecursiveController(Backend(), config)
    fdt = RecursiveController(Backend(), config)
    assert transformer.config == fdt.config
    assert transformer.run("q", TextEnvironment("x")).budget_remaining == fdt.run(
        "q", TextEnvironment("x")
    ).budget_remaining


def test_local_backend_uses_incremental_generation_cache():
    class Config:
        max_seq_len = 16
        eos_token_id = 3
        pad_token_id = 0
        num_anchors = 4

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [1]

        def decode(self, ids, skip_special_tokens=True):
            return "x" * len(ids)

    class CachedModel:
        def __init__(self):
            self.prefill_calls = 0
            self.decode_calls = 0

        def prefill(self, input_ids, attention_mask):
            self.prefill_calls += 1
            logits = torch.zeros(1, 1, 5)
            logits[..., 2] = 10
            return {"logits": logits, "hidden": torch.zeros(1, 1, 4), "anchor_stats": []}, {"length": 1}

        def decode_step(self, input_ids, cache):
            self.decode_calls += 1
            logits = torch.zeros(1, 1, 5)
            logits[..., 3] = 10
            return {"logits": logits, "hidden": torch.zeros(1, 1, 4), "anchor_stats": []}, {"length": cache["length"] + 1}

    model = CachedModel()
    backend = LocalCausalBackend(model, Config(), Tokenizer(), torch.device("cpu"), max_new_tokens=4)
    result = backend.generate_action("prompt", None)

    assert result.text == "x"
    assert result.generated_tokens == 1
    assert model.prefill_calls == 1
    assert model.decode_calls == 1
