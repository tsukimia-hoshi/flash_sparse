"""Public API for Heavily Compressed Attention (HCA) — the kernel-level work integration.

HCA differs from CSA in:
1. Compression ratio ``m_prime = 128`` (vs CSA's m=4); non-overlapping windows.
2. No indexer / no top-k — *all* causally-legal compressed positions are attended to.
3. Sliding window is the same as CSA's (n_win=128).

So HCA is "dense attention over heavily-compressed KV + sliding window".
We share the underlying ``sparse_attn`` kernel; the only HCA-specific work
is constructing ``topk_idxs`` to include all legal compressed entries.
"""

from __future__ import annotations

from typing import Optional

import torch

from flash_sparse.reference import _build_sliding_window_idxs, reference_sparse_attn

__all__ = ["flash_hca_forward", "build_hca_topk_idxs"]


def build_hca_topk_idxs(
    *,
    B: int,
    seq_len: int,
    n_win: int,
    n_compressed: int,
    m_prime: int,
    device: torch.device,
) -> torch.Tensor:
    """Build [B, S, n_win + n_compressed] index tensor for HCA.

    Layout: first ``n_win`` columns point into uncompressed KV ([0, S));
    next ``n_compressed`` columns point into compressed KV ([S, S + n_compressed))
    with causal masking — query at position t can attend to compressed block i
    iff ``m_prime * (i + 1) - 1 <= t``.
    """
    win_idxs = _build_sliding_window_idxs(B, seq_len, n_win, device)  # [B, S, n_win]

    block_last = (torch.arange(n_compressed, device=device) + 1) * m_prime - 1
    q_pos = torch.arange(seq_len, device=device).unsqueeze(-1)
    legal = q_pos >= block_last  # [S, n_compressed]
    base_idx = torch.arange(n_compressed, device=device).view(1, 1, -1).expand(B, seq_len, -1)
    legal_b = legal.unsqueeze(0).expand(B, -1, -1)
    comp_idxs = torch.where(legal_b, base_idx, torch.full_like(base_idx, -1))
    minus_one = torch.full_like(comp_idxs, -1)
    comp_idxs_global = torch.where(comp_idxs >= 0, comp_idxs + seq_len, minus_one)

    return torch.cat([win_idxs.long(), comp_idxs_global.long()], dim=-1).int()


def flash_hca_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_compressed: torch.Tensor,
    attn_sink: torch.Tensor,
    *,
    n_win: int,
    m_prime: int = 128,
    softmax_scale: Optional[float] = None,
    use_triton: bool = True,
) -> torch.Tensor:
    """End-to-end HCA forward (prefill path, single layer).

    Args:
    q: ``[B, S, n_h, d]`` BF16.
    kv: ``[B, S, d]`` BF16. Uncompressed KV (sliding-window source).
    kv_compressed: ``[B, S/m_prime, d]`` BF16. HCA-compressed KV.
    attn_sink: ``[n_h]`` FP32.
    n_win: sliding window size.
    m_prime: HCA compression ratio (128).
    softmax_scale: defaults to ``1 / sqrt(d)``.
    use_triton: if True, use Triton sparse_attn; otherwise reference.
    """
    B, S, n_h, d = q.shape
    n_compressed = kv_compressed.shape[1]

    topk_idxs = build_hca_topk_idxs(
        B=B,
        seq_len=S,
        n_win=n_win,
        n_compressed=n_compressed,
        m_prime=m_prime,
        device=q.device,
    )
    kv_full = torch.cat([kv, kv_compressed], dim=1)

    if use_triton:
        from flash_sparse.triton import sparse_attn

        return sparse_attn(q, kv_full, attn_sink, topk_idxs, softmax_scale)
    else:
        return reference_sparse_attn(q, kv_full, attn_sink, topk_idxs, softmax_scale)
