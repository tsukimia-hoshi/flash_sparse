"""Triton implementation of `sparse_attn_fwd` — the core fused kernel for both
CSA and HCA forward paths.

Same semantics as :func:`flash_sparse.reference.reference_sparse_attn` and the
DeepSeek TileLang `kernel.py:sparse_attn`. MQA layout: one shared KV head across
all query heads. Online softmax with attention sink.

Phase-2 prototype: BF16 inputs/outputs, FP32 internal accumulation. No FP8 path
yet (those land in a future CUDA reimplementation). No producer-consumer warp specialization
yet (FA3 patterns ditto). The goal here is **correctness + a non-trivial speed
floor** before we move to CUDA.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


# A "very negative" value used in place of -inf to avoid `-inf - -inf = nan`
# in the online-softmax rescaling. Same trick TileLang uses
# (`sparse_mla_fwd.py: T.fill(m_i, -(2**30))`).
# Must be wrapped in `tl.constexpr` so Triton allows the global access.
_M_INIT = tl.constexpr(-1.0e30)


# --------------------------------------------------------------------------
# Kernel-level tuning notes (H200, BF16 sparse-attn fwd).
# Empirically, BLOCK_N=64 with Triton's default num_warps=4, num_stages=3 wins
# over autotuned and heuristic-larger-tile alternatives because the kernel is
# HBM-bandwidth bound (each query has independent KV gather, so larger tiles
# don't amortize). We expose `block_n` on the wrapper for benchmarking, but
# the default is what beats the TileLang reference 4×.
# --------------------------------------------------------------------------


@triton.jit
def _sparse_attn_fwd_kernel(
    # in
    Q_ptr,
    KV_ptr,
    AttnSink_ptr,
    TopkIdxs_ptr,
    # out
    O_ptr,
    LSE_ptr,
    # scalars
    softmax_scale,
    # strides
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_ib,
    stride_is,
    stride_ik,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    stride_lb,
    stride_ls,
    stride_lh,
    # constants
    H: tl.constexpr,  # total query heads
    D: tl.constexpr,  # head_dim (must be power-of-2 for tl.dot)
    K_TOPK: tl.constexpr,  # total top-k size (window + selected entries)
    BLOCK_H: tl.constexpr,  # heads handled per program (>= 16 for MMA)
    BLOCK_N: tl.constexpr,  # KV block size for the streaming loop
):
    """Per-program work: one (batch, query position, head block).

    Grid layout: (S, B, H_blocks). S is in axis 0 (CUDA grid X, limit 2^31)
    instead of Y (limit 65535) so the kernel scales to long context without
    hitting the CUDA grid limit.

    Pointer offsets use int64 — at S ≥ 65K, ``pid_s * stride`` products can
    overflow int32 (e.g., S=131K with stride_qb = S·H·D ≈ 5e8 stays in i32 but
    quadratic-stride contexts like the indexer's score matrix already
    overflow at S=65K).

    Output: O[batch, q_pos, head_offsets, :] and LSE[batch, q_pos, head_offsets].
    """
    pid_s = tl.program_id(0).to(tl.int64)
    pid_b = tl.program_id(1).to(tl.int64)
    pid_hb = tl.program_id(2).to(tl.int64)

    # Range of heads handled by this program
    h_offsets = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H

    d_offsets = tl.arange(0, D)

    # ---- Load Q[pid_b, pid_s, h_offsets, :] -> [BLOCK_H, D]
    q_ptrs = (
        Q_ptr
        + pid_b * stride_qb
        + pid_s * stride_qs
        + h_offsets[:, None] * stride_qh
        + d_offsets[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)  # [BLOCK_H, D]

    # ---- Online softmax accumulators
    m_i = tl.full([BLOCK_H], _M_INIT, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)

    # ---- Loop over K blocks
    NUM_BLOCKS: tl.constexpr = (K_TOPK + BLOCK_N - 1) // BLOCK_N

    for n_block in tl.static_range(NUM_BLOCKS):
        n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_in_range = n_offsets < K_TOPK

        # Load top-k indices for this block: [BLOCK_N]
        idx_ptrs = TopkIdxs_ptr + pid_b * stride_ib + pid_s * stride_is + n_offsets * stride_ik
        idxs = tl.load(idx_ptrs, mask=n_in_range, other=-1)
        valid = (idxs >= 0) & n_in_range
        safe_idxs = tl.where(valid, idxs, 0).to(tl.int64)

        # Gather KV[pid_b, safe_idxs, :] -> [BLOCK_N, D]
        kv_ptrs = (
            KV_ptr + pid_b * stride_kvb + safe_idxs[:, None] * stride_kvn + d_offsets[None, :] * stride_kvd
        )
        kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0)

        # Scores: Q @ KV^T -> [BLOCK_H, BLOCK_N], FP32 accumulator
        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(valid[None, :], scores, _M_INIT)

        # Online softmax
        s_max = tl.max(scores, axis=1)  # [BLOCK_H]
        m_new = tl.maximum(m_i, s_max)
        alpha = tl.exp(m_i - m_new)  # [BLOCK_H]
        p = tl.exp(scores - m_new[:, None])  # [BLOCK_H, BLOCK_N]

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

        # ---- Apply attention sink (eq. 27 of V4 paper): add exp(sink - max) to denom.
        sink_ptrs = AttnSink_ptr + h_offsets
        sink = tl.load(sink_ptrs, mask=h_mask, other=_M_INIT)  # [BLOCK_H]

        m_new = tl.maximum(m_i, sink)
        alpha = tl.exp(m_i - m_new)
        sink_term = tl.exp(sink - m_new)
        acc = acc * alpha[:, None]
        l_i = l_i * alpha + sink_term

        # ---- Normalize
        # When `l_i == 0` (only possible if no valid entries AND sink == _M_INIT,
        # which we don't expect), the output is 0; we divide by 1 to keep it finite.
        l_safe = tl.where(l_i > 0.0, l_i, 1.0)
        o = acc / l_safe[:, None]

        # Store O[pid_b, pid_s, h_offsets, :]
        o_ptrs = (
            O_ptr
            + pid_b * stride_ob
            + pid_s * stride_os
            + h_offsets[:, None] * stride_oh
            + d_offsets[None, :] * stride_od
        )
        tl.store(o_ptrs, o.to(tl.bfloat16), mask=h_mask[:, None])

        # Store LSE = log(l_i) + m_new (used by backward).
        lse = tl.where(l_i > 0.0, tl.log(l_i) + m_new, tl.full([BLOCK_H], _M_INIT, dtype=tl.float32))
        lse_ptrs = LSE_ptr + pid_b * stride_lb + pid_s * stride_ls + h_offsets * stride_lh
        tl.store(lse_ptrs, lse, mask=h_mask)


def sparse_attn_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
    *,
    block_h: Optional[int] = None,
    block_n: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton-backed ``sparse_attn`` forward.

    Args:
    q: ``[B, S, H, D]``, BF16 (or FP16). RoPE-rotated.
    kv: ``[B, N_kv, D]``, same dtype as q. Single shared KV head (MQA).
    attn_sink: ``[H]``, FP32.
    topk_idxs: ``[B, S, K]``, INT32 / INT64. ``-1`` = masked.
    softmax_scale: defaults to ``1 / sqrt(D)``.
    block_h: heads per Triton program. Defaults to ``next_power_of_2(H)`` capped at 64.
    block_n: KV block size for streaming loop. Default 64 — empirically optimal
    on H200 for our HBM-bandwidth-bound regime.

    Returns:
    ``(o, lse)`` where ``o: [B, S, H, D]`` BF16 and ``lse: [B, S, H]`` FP32.
    """
    assert q.is_cuda and kv.is_cuda
    assert q.dim == 4 and kv.dim == 3
    assert q.dtype in (torch.bfloat16, torch.float16), f"Q must be bf16 or fp16, got {q.dtype}"
    assert kv.dtype == q.dtype
    assert attn_sink.dtype == torch.float32
    B, S, H, D = q.shape
    B2, N_kv, D2 = kv.shape
    assert B2 == B and D2 == D, "q and kv shape mismatch"
    assert attn_sink.shape == (H,)
    assert topk_idxs.shape[:2] == (B, S)
    K_TOPK = topk_idxs.shape[-1]

    if D & (D - 1) != 0:
        raise ValueError(f"D ({D}) must be a power of 2 for tl.dot")

    if softmax_scale is None:
        softmax_scale = D**-0.5

        # BLOCK_H scaling by head_dim D to keep SRAM usage in budget.
        # SRAM ≈ BLOCK_H * D * (2 BF16 + 4 FP32) + BLOCK_N * D * 2
        # = BLOCK_H * D * 6 + BLOCK_N * D * 2
        # For BLOCK_N=64, ceiling 128 KB for the per-D dependent part:
        # BLOCK_H * D ≤ ~21000 elements
        # so BLOCK_H ≤ 21000 / D rounded down to next power of 2
        # Empirically: BLOCK_H=64 at D≤64 gives 23% peak; BLOCK_H=64 at D=512
        # spills registers, regressing to 5% peak. The cap below restores 5×
        # efficiency at D=512.
        if block_h is None:
            if D >= 512:
                cap = 16
            elif D >= 256:
                cap = 32
            else:
                cap = 64
                block_h = max(16, min(triton.next_power_of_2(H), cap))

                o = torch.empty_like(q)
                lse = torch.empty((B, S, H), device=q.device, dtype=torch.float32)

                idxs = topk_idxs.to(torch.int32).contiguous()

                # Empirically tuned on H200 BF16: BLOCK_N=64 + num_warps=4 + num_stages=4 wins
                # over the previous defaults for K_TOPK ≥ 256. num_stages=4 lets Triton issue
                # one more KV gather in flight, which closes some of the gap to the HBM floor.
                if K_TOPK >= 256:
                    num_warps = 4
                    num_stages = 4
                else:
                    num_warps = 4
                    num_stages = 3

                    # Grid: (S, B, H_blocks). S in axis 0 (CUDA grid X, limit 2^31) so we scale
                    # to long context. B in axis 1 (typically 1 in our workloads, far from limit).
                    grid = (S, B, triton.cdiv(H, block_h))
                    _sparse_attn_fwd_kernel[grid](
                        q,
                        kv,
                        attn_sink,
                        idxs,
                        o,
                        lse,
                        softmax_scale,
                        q.stride(0),
                        q.stride(1),
                        q.stride(2),
                        q.stride(3),
                        kv.stride(0),
                        kv.stride(1),
                        kv.stride(2),
                        idxs.stride(0),
                        idxs.stride(1),
                        idxs.stride(2),
                        o.stride(0),
                        o.stride(1),
                        o.stride(2),
                        o.stride(3),
                        lse.stride(0),
                        lse.stride(1),
                        lse.stride(2),
                        H=H,
                        D=D,
                        K_TOPK=K_TOPK,
                        BLOCK_H=block_h,
                        BLOCK_N=block_n,
                        num_warps=num_warps,
                        num_stages=num_stages,
                    )
                    return o, lse


__all__ = ["sparse_attn_fwd"]
