"""KV compression helpers shared by CSA and HCA paths.

Phase-0 stubs. Real implementations land in future work alongside the kernels.
"""

from __future__ import annotations

import torch

__all__ = ["softmax_gated_pool"]


def softmax_gated_pool(
    kv: torch.Tensor,
    score: torch.Tensor,
    *,
    dim: int = -2,
) -> torch.Tensor:
    """Softmax-gated weighted pooling along ``dim``.

    Equivalent to ``(kv * score.softmax(dim)).sum(dim)`` but kept as a named
    function so the optimized kernel can swap it out without changing call sites.
    """
    return (kv * score.softmax(dim=dim)).sum(dim=dim)
