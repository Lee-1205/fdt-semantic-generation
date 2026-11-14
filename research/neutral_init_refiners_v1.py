"""CPU-only neutral-initializer FDT variants for a narrow prospective E3 study."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "research"))
from fdt_local.model_masked import FDT100
from query_local_refiner_v1 import QueryLocalRefiner


_QUERY_AXIS_KEYS = ("init", "query_roles", "query_mask")


def neutral_initializer(query_roles: torch.Tensor, query_mask: torch.Tensor,
                        semantic) -> torch.Tensor:
    """Query-role-plus-constant template; predicate evidence is exactly zero."""
    if query_roles.ndim != 3 or query_roles.shape[-1] != 16:
        raise ValueError("query_roles must have shape [batch, query, 16]")
    if query_mask.shape != query_roles.shape[:2]:
        raise ValueError("query mask must match query_roles")
    zero_predicate = query_roles.new_zeros(*query_roles.shape[:2], 4)
    return semantic(query_roles, zero_predicate) * query_mask[..., None]


class NeutralInitMixin:
    """Replace only denoiser initialization after running the unchanged full encoder."""

    def encode(self, features, ablate=False):
        encoded = super().encode(features, ablate=ablate)
        result = dict(encoded)
        result["init"] = neutral_initializer(result["query_roles"], result["query_mask"], self.semantic)
        return result


class NeutralInitFullRefiner(NeutralInitMixin, FDT100):
    """Neutral query-template initializer with the original full-query denoiser."""


class NeutralInitQueryLocalRefiner(NeutralInitMixin, QueryLocalRefiner):
    """Same neutral initializer with query-local singleton denoising."""
