"""Triton kernel implementations of FlashSparse — future work prototypes.

Public exports:
sparse_attn_fwd: forward-only entry point (no autograd).
sparse_attn_bwd: backward-only entry point (called by the autograd Function).
sparse_attn: differentiable autograd-wrapped function — use this from
PyTorch training loops or anywhere you want grad support.
"""

from __future__ import annotations

from typing import Optional

import torch

from flash_sparse.triton.sparse_attn_fwd import sparse_attn_fwd
from flash_sparse.triton.sparse_attn_bwd import sparse_attn_bwd


class _SparseAttnFunc(torch.autograd.Function):
    """torch.autograd.Function wrapper. Saves Q, KV, attn_sink, topk_idxs, O, LSE
    for the backward pass."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_sink: torch.Tensor,
        topk_idxs: torch.Tensor,
        softmax_scale: Optional[float],
    ) -> torch.Tensor:
        o, lse = sparse_attn_fwd(q, kv, attn_sink, topk_idxs, softmax_scale)
        ctx.save_for_backward(q, kv, attn_sink, topk_idxs, o, lse)
        ctx.softmax_scale = softmax_scale
        return o

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        q, kv, attn_sink, topk_idxs, o, lse = ctx.saved_tensors
        dq, dkv, dattn_sink = sparse_attn_bwd(
            q, kv, attn_sink, topk_idxs, o, lse, do.contiguous(), ctx.softmax_scale
        )
        # Return one gradient per forward-input, plus None for non-tensor `softmax_scale`.
        return dq, dkv, dattn_sink, None, None


def sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Differentiable sparse attention. Wraps the Triton fwd + bwd kernels.

    See :func:`sparse_attn_fwd` for argument shapes/dtypes. Returns ``o`` only
    (the LSE is saved internally for the backward pass).
    """
    return _SparseAttnFunc.apply(q, kv, attn_sink, topk_idxs, softmax_scale)


__all__ = ["sparse_attn", "sparse_attn_fwd", "sparse_attn_bwd"]
