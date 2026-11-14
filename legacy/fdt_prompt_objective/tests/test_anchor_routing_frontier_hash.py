from __future__ import annotations

import torch

from scripts.diagnose_fdt_anchor_routing_frontier import tensor_group_sha256


def test_tensor_group_sha256_accepts_scalar_anchor_parameters() -> None:
    state = {
        "blocks.0.anchor.gate": torch.tensor(0.0),
        "blocks.0.anchor.anchor_keys": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "blocks.0.mlp.gate.weight": torch.ones(2, 2),
    }

    first = tensor_group_sha256(state, ".anchor.")
    second = tensor_group_sha256(state, ".anchor.")

    assert len(first) == 64
    assert first == second
