"""CPU-testable query-bound cross-attention candidate for the frozen FDT100 core."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fdt_local.model_masked import FDT100


def build_query_binding_mask(query_mask: torch.Tensor, fact_mask: torch.Tensor) -> torch.Tensor:
    """Return [B,Q,2Q+F+1] access: own recalled/init, all valid facts, and null."""
    if query_mask.ndim != 2 or fact_mask.ndim != 2 or query_mask.shape[0] != fact_mask.shape[0]:
        raise ValueError("query_mask and fact_mask must be rank-2 with matching batch")
    batch, queries = query_mask.shape
    facts = fact_mask.shape[1]
    qvalid = query_mask.bool()
    fvalid = fact_mask.bool()
    own = torch.eye(queries, dtype=torch.bool, device=query_mask.device)[None].expand(batch, -1, -1)
    own = own & qvalid[:, :, None]
    fact_access = fvalid[:, None, :].expand(batch, queries, facts)
    null_access = torch.ones(batch, queries, 1, dtype=torch.bool, device=query_mask.device)
    return torch.cat((own, own, fact_access, null_access), dim=-1)


def bound_core_forward(block, x: torch.Tensor, context: torch.Tensor,
                       query_mask: torch.Tensor, binding_mask: torch.Tensor) -> torch.Tensor:
    """Core.forward with its original self-attention/FFN and per-query cross mask."""
    valid = query_mask[..., None].to(x.dtype)
    x = x * valid
    y = block.norms[0](x)
    x = (x + block.self_attn(y, y, query_mask[:, None, None, :].bool())) * valid
    x = (x + block.cross(block.norms[1](x), context, binding_mask[:, None, :, :].bool())) * valid
    return (x + block.ff(block.norms[2](x))) * valid


class QueryBoundRefiner(FDT100):
    """Keep original encoder/self-attention/relational path; bind cross-attention slots."""

    def denoise(self, z, t, enc, ablate=False):
        batch, queries, _ = z.shape
        if enc["query_mask"].shape != (batch, queries):
            raise ValueError("query_mask must match z batch and query dimensions")
        context = self.condition_at(z, enc, ablate)
        expected_context = 2 * queries + enc["facts"].shape[1] + 1
        if context.shape[1] != expected_context:
            raise ValueError("context must be [recalled Q, init Q, facts F, null]")
        binding_mask = build_query_binding_mask(enc["query_mask"], enc["fact_mask"])

        phase = t.float()[:, None] * self.frequencies[None]
        time = self.time(torch.cat((phase.sin(), phase.cos()), -1))[:, None]
        x = z + time
        if self.c.diffusion_role_condition:
            x = x + self.role_project(enc["query_roles"])
        query_mask = enc["query_mask"]
        for block in self.refiner:
            if self.c.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(
                    lambda current, ctx, block=block, query_mask=query_mask, binding_mask=binding_mask:
                        bound_core_forward(block, current, ctx, query_mask, binding_mask),
                    x, context, use_reentrant=False,
                )
            else:
                x = bound_core_forward(block, x, context, query_mask, binding_mask)
        return self.velocity(self.refine_norm(x)) * query_mask[..., None]
