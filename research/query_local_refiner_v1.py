"""CPU-testable query-local denoiser wrapper for the frozen masked FDT core.

This candidate changes only how denoise() batches the original queries. It keeps
all FDT100 parameters, buffers, names, and the full-query encoder unchanged.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fdt_local.model_masked import FDT100


_QUERY_AXIS_KEYS = ("init", "query_roles", "query_mask")


def flatten_single_query_views(encoded: dict[str, torch.Tensor], z: torch.Tensor,
                                times: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Convert BxQ denoising inputs into B*Q one-query rows, repeating only graph context."""
    if z.ndim != 3:
        raise ValueError("z must have shape [batch, query, dimension]")
    batch, queries, dim = z.shape
    if queries <= 0 or encoded["init"].shape != z.shape:
        raise ValueError("encoded init and z must have matching nonempty query axes")
    if encoded["query_roles"].shape[:2] != (batch, queries):
        raise ValueError("query roles do not match z query axes")
    if encoded["query_mask"].shape != (batch, queries):
        raise ValueError("query mask does not match z query axes")
    if times.shape != (batch,):
        raise ValueError("times must retain one scalar per original graph")

    single: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if key in _QUERY_AXIS_KEYS:
            single[key] = value.reshape(batch * queries, 1, *value.shape[2:])
        else:
            if value.shape[0] != batch:
                raise ValueError(f"static encoded field {key!r} has a different batch size")
            single[key] = value[:, None].expand(batch, queries, *value.shape[1:]).reshape(
                batch * queries, *value.shape[1:])
    single_z = z.reshape(batch * queries, 1, dim)
    single_times = times[:, None].expand(batch, queries).reshape(batch * queries)
    return single, single_z, single_times


class QueryLocalRefiner(FDT100):
    """FDT100 with singleton-query calls only inside the existing denoise method."""

    def denoise(self, z, t, enc, ablate=False):
        batch, queries, dim = z.shape
        single_enc, single_z, single_t = flatten_single_query_views(enc, z, t)
        result = super().denoise(single_z, single_t, single_enc, ablate=ablate)
        return result.reshape(batch, queries, dim)
