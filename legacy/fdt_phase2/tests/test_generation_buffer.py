from __future__ import annotations

from types import SimpleNamespace

import torch

from fdt_rlm.next_tools import apply_repetition_penalty_, generate_ids


class DeterministicModel:
    def __init__(self, max_seq_len: int = 8, eos_token_id: int = 9) -> None:
        self.config = SimpleNamespace(max_seq_len=max_seq_len, eos_token_id=eos_token_id)

    def _output(self, current: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, length = current.shape
        logits = torch.full((batch, length, 10), -10.0)
        next_ids = (current[:, -1] + 1).clamp_max(self.config.eos_token_id)
        logits[torch.arange(batch), -1, next_ids] = 10.0
        return {"logits": logits}

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self._output(input_ids)

    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self._output(input_ids), input_ids.clone()

    def decode_step(self, next_id: torch.Tensor, cache: torch.Tensor):
        updated = torch.cat((cache, next_id), dim=1)
        return self._output(updated), updated


class RepeatingFDTv4Model(DeterministicModel):
    def __init__(self) -> None:
        super().__init__(max_seq_len=20, eos_token_id=9)
        self.config.model_type = "fdt_v4"
        self.config.generation_ngram_order = 3
        self.config.generation_ngram_penalty = 8.0
        self.config.generation_ngram_window = 96
        self.config.generation_ngram_hard_block_after = 2

    def _output(self, current: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, length = current.shape
        logits = torch.full((batch, length, 10), -10.0)
        next_id = (3, 4, 5)[(length - 1) % 3]
        logits[:, -1, next_id] = 10.0
        logits[:, -1, 6] = 1.0
        return {"logits": logits}


def legacy_greedy(model: DeterministicModel, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    generated = input_ids
    output, cache = model.prefill(input_ids, torch.ones_like(input_ids))
    for _ in range(max_new_tokens):
        if generated.size(1) >= model.config.max_seq_len:
            break
        next_id = output["logits"][:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_id), dim=1)
        if bool((next_id == model.config.eos_token_id).all()):
            break
        output, cache = model.decode_step(next_id, cache)
    return generated


def test_preallocated_generation_matches_legacy_tokens() -> None:
    model = DeterministicModel()
    prompt = torch.tensor([[2, 3]])
    expected = legacy_greedy(model, prompt, max_new_tokens=6)
    actual = generate_ids(model, prompt, max_new_tokens=6, greedy=True, use_cache=True)
    assert torch.equal(actual, expected)


def test_generation_respects_context_limit_without_unused_tail() -> None:
    model = DeterministicModel(max_seq_len=5)
    prompt = torch.tensor([[1, 2, 3]])
    actual = generate_ids(model, prompt, max_new_tokens=8, greedy=True, use_cache=True)
    assert actual.shape == (1, 5)
    assert actual.tolist() == [[1, 2, 3, 4, 5]]


def test_fdt_v4_generic_generation_hard_blocks_persistent_ngram_loop() -> None:
    model = RepeatingFDTv4Model()
    actual = generate_ids(
        model,
        torch.tensor([[2]]),
        max_new_tokens=9,
        greedy=True,
        use_cache=True,
        ngram_loop_penalty=model.config.generation_ngram_penalty,
    )

    assert actual.tolist()[0][-1] == 6
    assert actual.tolist()[0][1:-1] == [3, 4, 5, 3, 4, 5, 3, 4]


def test_fdt_v4_generic_generation_leaves_loop_penalty_off_by_default() -> None:
    model = RepeatingFDTv4Model()
    actual = generate_ids(
        model,
        torch.tensor([[2]]),
        max_new_tokens=9,
        greedy=True,
        use_cache=True,
    )

    assert 6 not in actual.tolist()[0]


def test_repetition_penalty_is_sign_aware_and_batch_local() -> None:
    logits = torch.tensor([[4.0, -2.0, 3.0], [5.0, 6.0, -4.0]])
    history = torch.tensor([[0, 1, 1], [2, 2, 2]])
    result = apply_repetition_penalty_(logits.clone(), history, 2.0)

    assert torch.equal(result[0], torch.tensor([2.0, -4.0, 3.0]))
    assert torch.equal(result[1], torch.tensor([5.0, 6.0, -8.0]))


def test_repetition_penalty_rejects_values_below_one() -> None:
    try:
        apply_repetition_penalty_(torch.zeros(1, 4), torch.tensor([[1]]), 0.9)
    except ValueError:
        pass
    else:
        raise AssertionError("penalties below one would reward repetition")
