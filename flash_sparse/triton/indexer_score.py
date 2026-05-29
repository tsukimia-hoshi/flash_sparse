"""Triton implementation of the lightning indexer's score computation
(eq. 16 of DeepSeek-V4):

I(t, s) = sum_h w_{t,h} * relu(q_{t,h} . K^IComp_s)

Fused kernel: matmul Q @ K^T → ReLU → multiply by per-head weights → sum over
heads. Output: [B, S, T] score matrix in FP32.

Phase-2.1.b prototype materializes the score matrix to HBM (`T · 4` bytes /
token write). The streaming-top-k version that keeps scores in registers /
shared memory and updates a heap of size k=1024 is the next iteration —
see `docs/streaming_topk.md` § 3.2 for the design.

After this kernel returns the score matrix, top-k is taken via `torch.topk`
(reference path). When we replace it with the streaming heap, the public API
(`indexer_score_topk`) doesn't change — callers just pay less HBM.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


_INDEXER_CONFIGS = [
    triton.Config({"BLOCK_T": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_T": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_T": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_T": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_T": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_T": 128}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_T": 128}, num_warps=8, num_stages=4),
    triton.Config({"BLOCK_T": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_T": 256}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_INDEXER_CONFIGS, key=["T", "H_I", "D_I"])
@triton.jit
def _indexer_score_kernel(
    Q_ptr,
    K_ptr,
    W_ptr,
    S_ptr,
    # Q strides
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    # K strides
    stride_kb,
    stride_kt,
    stride_kd,
    # W strides
    stride_wb,
    stride_ws,
    stride_wh,
    # S strides
    stride_sb,
    stride_ss,
    stride_st,
    # constants
    T: tl.constexpr,  # number of compressed keys (n_compressed)
    H_I: tl.constexpr,  # indexer heads
    D_I: tl.constexpr,  # indexer head_dim
    BLOCK_T: tl.constexpr,  # K block in this kernel
):
    """Per program: one (batch, query position, T-block) → writes BLOCK_T scores.

    Grid: (S, B, T_blocks). S in axis 0 to avoid CUDA grid-Y limit at long context.

    Pointer arithmetic uses int64. For long context (S = T = 65536) the score
    matrix offset `pid_s * stride_ss = pid_s * T` reaches 4.3e9, which
    overflows int32 (max 2.1e9). All offset terms are promoted to int64 below.
    """
    pid_s = tl.program_id(0).to(tl.int64)
    pid_b = tl.program_id(1).to(tl.int64)
    pid_tb = tl.program_id(2).to(tl.int64)

    h_offsets = tl.arange(0, H_I).to(tl.int64)
    d_offsets = tl.arange(0, D_I).to(tl.int64)

    # Load Q[pid_b, pid_s, :, :] -> [H_I, D_I]
    q_ptrs = (
        Q_ptr
        + pid_b * stride_qb
        + pid_s * stride_qs
        + h_offsets[:, None] * stride_qh
        + d_offsets[None, :] * stride_qd
    )
    q = tl.load(q_ptrs)  # [H_I, D_I]

    # Load weights[pid_b, pid_s, :] -> [H_I]
    w_ptrs = W_ptr + pid_b * stride_wb + pid_s * stride_ws + h_offsets * stride_wh
    weights = tl.load(w_ptrs).to(tl.float32)  # [H_I], FP32

    # Process this T block (int64 — see header comment)
    t_offsets = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T).to(tl.int64)
    t_mask = t_offsets < T

    # Load K[pid_b, t_offsets, :] -> [BLOCK_T, D_I]
    k_ptrs = K_ptr + pid_b * stride_kb + t_offsets[:, None] * stride_kt + d_offsets[None, :] * stride_kd
    k_block = tl.load(k_ptrs, mask=t_mask[:, None], other=0.0)  # [BLOCK_T, D_I]

    # Q @ K^T -> [H_I, BLOCK_T], FP32 accumulator
    raw_scores = tl.dot(q, tl.trans(k_block))

    # ReLU
    raw_scores = tl.maximum(raw_scores, 0.0)

    # Multiply by per-head weights and sum over heads
    weighted = raw_scores * weights[:, None]  # [H_I, BLOCK_T]
    per_t = tl.sum(weighted, axis=0)  # [BLOCK_T]

    # Store
    s_ptrs = S_ptr + pid_b * stride_sb + pid_s * stride_ss + t_offsets * stride_st
    tl.store(s_ptrs, per_t, mask=t_mask)


def indexer_score(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Compute the lightning-indexer score matrix (eq. 16 of V4 paper). Autotuned.

    Args:
    q: ``[B, S, H_I, D_I]`` BF16. Indexer queries.
    k_idx: ``[B, T, D_I]`` BF16. Indexer compressed keys.
    weights: ``[B, S, H_I]`` FP32 or BF16. Per-head weights.

    Returns:
    ``scores: [B, S, T]`` FP32. Apply causal masking and top-k externally.
    """
    assert q.is_cuda and k_idx.is_cuda and weights.is_cuda
    assert q.dim == 4 and k_idx.dim == 3 and weights.dim == 3
    B, S, H_I, D_I = q.shape
    B2, T, D_I2 = k_idx.shape
    assert B2 == B and D_I2 == D_I, f"q/k shape mismatch: q={q.shape}, k={k_idx.shape}"
    assert weights.shape == (B, S, H_I)

    if H_I & (H_I - 1) != 0:
        raise ValueError(f"H_I ({H_I}) must be a power of 2 (Triton tl.dot constraint)")
    if D_I & (D_I - 1) != 0:
        raise ValueError(f"D_I ({D_I}) must be a power of 2")
    if H_I < 16:
        raise ValueError(f"H_I ({H_I}) must be >= 16 for tl.dot. Pad heads with zeros if needed.")

        scores = torch.empty((B, S, T), device=q.device, dtype=torch.float32)
        # Grid: (S, B, T_blocks) — S in axis 0 (CUDA X dim, no practical limit) so
        # the kernel scales to S > 65535 without hitting the grid-Y limit.
        grid = lambda meta: (S, B, triton.cdiv(T, meta["BLOCK_T"]))

        _indexer_score_kernel[grid](
            q,
            k_idx,
            weights,
            scores,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k_idx.stride(0),
            k_idx.stride(1),
            k_idx.stride(2),
            weights.stride(0),
            weights.stride(1),
            weights.stride(2),
            scores.stride(0),
            scores.stride(1),
            scores.stride(2),
            T=T,
            H_I=H_I,
            D_I=D_I,
        )
        return scores


def indexer_score_topk(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
    *,
    causal_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Indexer + top-k. Returns ``(top_k_indices, top_k_scores)``.

    Args:
    q, k_idx, weights: see :func:`indexer_score`.
    top_k: number of selected compressed entries per query.
    causal_mask: optional ``[B, S, T]`` bool — True = legal entry.

    Returns:
    ``(top_k_indices: [B, S, k] int64, top_k_scores: [B, S, k] FP32)``.
    """
    scores = indexer_score(q, k_idx, weights)
    if causal_mask is not None:
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        k = min(top_k, scores.shape[-1])
        top_scores, top_idx = scores.topk(k, dim=-1)
        return top_idx.to(torch.int64), top_scores


__all__ = ["indexer_score", "indexer_score_topk"]
