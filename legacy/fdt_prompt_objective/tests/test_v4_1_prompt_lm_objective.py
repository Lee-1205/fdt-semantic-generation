from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from train_fdt_v4_1_retention import chunked_prompt_lm_loss  # noqa: E402


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(4, 11, bias=False)
        self.config = type("Config", (), {"lm_logit_clip": 100.0})()


def fixture():
    torch.manual_seed(17)
    model = TinyLM()
    hidden = torch.randn(2, 6, 4, requires_grad=True)
    ids = torch.tensor([[1, 3, 4, 5, 6, 0], [1, 7, 8, 9, 10, 0]])
    labels = torch.tensor([[-100, -100, -100, 5, 6, -100],
                           [-100, -100, 8, 9, 10, -100]])
    active = ids.ne(0)
    return model, hidden, ids, labels, active


def test_chunked_prompt_loss_matches_v20_formula_and_gradients():
    model, hidden, ids, labels, active = fixture()
    logits = model.lm_head(hidden[:, :-1]).float()
    target = ids[:, 1:]
    valid = labels[:, 1:].eq(-100) & active[:, 1:]
    token_loss = F.cross_entropy(logits.reshape(-1, 11), target.reshape(-1),
                                 reduction="none").reshape_as(target)
    expected = ((token_loss * valid).sum(1) / valid.sum(1).clamp_min(1)).sum()
    rows = []
    actual = chunked_prompt_lm_loss(model, hidden, ids, labels, active, 0, 2,
                                    checkpoint_chunks=False, accounting=rows)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert [row["prompt_tokens"] for row in rows] == [2, 1]
    actual.backward()
    assert hidden.grad[:, 0].abs().sum() > 0
    assert hidden.grad[0, 1].abs().sum() > 0
    assert torch.count_nonzero(hidden.grad[0, 2:]).item() == 0
    assert torch.count_nonzero(hidden.grad[1, 1:]).item() == 0


def test_prompt_loss_microbatch_and_checkpoint_equivalence():
    model, hidden, ids, labels, active = fixture()
    full = chunked_prompt_lm_loss(model, hidden, ids, labels, active, 0, 2, False)
    split = sum(chunked_prompt_lm_loss(model, hidden[i:i+1], ids[i:i+1],
                                       labels[i:i+1], active[i:i+1], 0, 1, False)
                for i in range(2))
    checkpointed = chunked_prompt_lm_loss(model, hidden, ids, labels, active,
                                          0, 3, True)
    torch.testing.assert_close(full, split, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(full, checkpointed, rtol=1e-6, atol=1e-6)


def test_no_prompt_targets_have_zero_contribution():
    model, hidden, ids, labels, active = fixture()
    labels = ids.clone()
    assert chunked_prompt_lm_loss(model, hidden, ids, labels, active,
                                  0, 2, False).item() == 0.0
