"""Triton implementation of `sparse_attn_bwd` — the backward pass for the
sparse attention forward kernel in :mod:`flash_sparse.triton.sparse_attn_fwd`.

Mathematics (sink-augmented softmax; see `docs/streaming_topk.md` and the V4
paper, eq. 27 for the sink semantics):

o[t,h] = Σ_j p[t,h,j] · V[t,h,j]
LSE[t,h] = log(Σ_j exp(s[t,h,j]) + exp(sink[h]))
p[t,h,j] = exp(s[t,h,j] - LSE[t,h])
p_sink[t,h] = exp(sink[h] - LSE[t,h]) (used only by dsink)
Δ[t,h] = Σ_d o[t,h,d] · do[t,h,d]
dp[t,h,j] = p[t,h,j] · (do[t,h] · V[j] − Δ[t,h])

Gradients (MQA — K and V share the same tensor, so ``dKV`` accumulates both
the score-path and value-path contributions):

dQ[t,h] = sm_scale · Σ_j dp[t,h,j] · K[j]
dKV[s_j] += Σ_t,h ( sm_scale · dp[t,h,j] · q[t,h] + p[t,h,j] · do[t,h] )
dsink[h] = −Σ_t p_sink[t,h] · Δ[t,h]

This the kernel-level work prototype uses the simple atomic-scatter dKV pattern
(matches TileLang's reference); the inverted-topk sum-reduction win from
``fusion_analysis.md`` § 6 lands in future work.b. Forward correctness must
already hold before this kernel is meaningful — see
``tests/test_triton_sparse_attn_fwd.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


_M_INIT = tl.constexpr(-1.0e30)


# Autotune config space for the main bwd kernel. Tighter than fwd because we
# also have to fit Δ + LSE + dQ accumulator + dKV partial product in SRAM.
_BWD_CONFIGS = [
    triton.Config({"BLOCK_N": 32}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=3),
    triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_N": 128}, num_warps=8, num_stages=3),
]


@triton.jit
def _delta_kernel(
    O_ptr,
    dO_ptr,
    Delta_ptr,
    stride_ob,
    stride_os,
    stride_oh,
    stride_od,
    stride_dob,
    stride_dos,
    stride_doh,
    stride_dod,
    stride_db,
    stride_ds,
    stride_dh,
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Δ[b, s, h] = Σ_d O[b, s, h, d] · dO[b, s, h, d]. FP32 result."""
    # Grid: (S, B, H_blocks). int64 pointer offsets for long-context safety.
    pid_s = tl.program_id(0).to(tl.int64)
    pid_b = tl.program_id(1).to(tl.int64)
    pid_hb = tl.program_id(2).to(tl.int64)

    h_offsets = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = h_offsets < H
    d_offsets = tl.arange(0, D).to(tl.int64)

    o_ptrs = (
        O_ptr
        + pid_b * stride_ob
        + pid_s * stride_os
        + h_offsets[:, None] * stride_oh
        + d_offsets[None, :] * stride_od
    )
    do_ptrs = (
        dO_ptr
        + pid_b * stride_dob
        + pid_s * stride_dos
        + h_offsets[:, None] * stride_doh
        + d_offsets[None, :] * stride_dod
    )
    o = tl.load(o_ptrs, mask=h_mask[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_ptrs, mask=h_mask[:, None], other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=1)  # [BLOCK_H]

    delta_ptrs = Delta_ptr + pid_b * stride_db + pid_s * stride_ds + h_offsets * stride_dh
    tl.store(delta_ptrs, delta, mask=h_mask)


@triton.autotune(
    configs=_BWD_CONFIGS,
    key=["H", "D", "K_TOPK", "BLOCK_H"],
    # Atomic-add output buffers must be zeroed between autotuner trials —
    # otherwise each trial's atomic_add stacks on the previous trial's, and
    # the "best config" gets selected against contaminated outputs.
    reset_to_zero=["dKV_ptr", "dSink_ptr"],
)
@triton.jit
def _sparse_attn_bwd_kernel(
    # in
    Q_ptr,
    KV_ptr,
    AttnSink_ptr,
    TopkIdxs_ptr,
    dO_ptr,
    LSE_ptr,
    Delta_ptr,
    # out (FP32 for atomic adds; cast to BF16 in postprocess)
    dQ_ptr,
    dKV_ptr,
    dSink_ptr,
    # scalars
    softmax_scale,
    # strides — Q
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    # KV
    stride_kvb,
    stride_kvn,
    stride_kvd,
    # idxs
    stride_ib,
    stride_is,
    stride_ik,
    # dO
    stride_dob,
    stride_dos,
    stride_doh,
    stride_dod,
    # LSE / Δ
    stride_lb,
    stride_ls,
    stride_lh,
    stride_db,
    stride_ds,
    stride_dh,
    # dQ
    stride_dqb,
    stride_dqs,
    stride_dqh,
    stride_dqd,
    # dKV (FP32 buffer)
    stride_dkvb,
    stride_dkvn,
    stride_dkvd,
    # constants
    H: tl.constexpr,
    D: tl.constexpr,
    K_TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per (batch, query position, head block): compute dQ for that query,
    atomic-scatter dKV at top-k indices, atomic-add dsink per head."""
    # Grid: (S, B, H_blocks). int64 pointer offsets for long-context safety.
    pid_s = tl.program_id(0).to(tl.int64)
    pid_b = tl.program_id(1).to(tl.int64)
    pid_hb = tl.program_id(2).to(tl.int64)

    h_offsets = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H).to(tl.int64)
    h_mask = h_offsets < H
    d_offsets = tl.arange(0, D).to(tl.int64)

    # ---- Load Q, dO, LSE, Δ, sink for this (b, s, head_block)
    q_ptrs = (
        Q_ptr
        + pid_b * stride_qb
        + pid_s * stride_qs
        + h_offsets[:, None] * stride_qh
        + d_offsets[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)  # [BLOCK_H, D]

    do_ptrs = (
        dO_ptr
        + pid_b * stride_dob
        + pid_s * stride_dos
        + h_offsets[:, None] * stride_doh
        + d_offsets[None, :] * stride_dod
    )
    do = tl.load(do_ptrs, mask=h_mask[:, None], other=0.0)  # [BLOCK_H, D]

    lse_ptrs = LSE_ptr + pid_b * stride_lb + pid_s * stride_ls + h_offsets * stride_lh
    lse = tl.load(lse_ptrs, mask=h_mask, other=0.0)  # [BLOCK_H]

    delta_ptrs = Delta_ptr + pid_b * stride_db + pid_s * stride_ds + h_offsets * stride_dh
    delta = tl.load(delta_ptrs, mask=h_mask, other=0.0)  # [BLOCK_H]

    sink = tl.load(AttnSink_ptr + h_offsets, mask=h_mask, other=_M_INIT)  # [BLOCK_H]

    # ---- dsink contribution = -p_sink * Δ
    # p_sink = exp(sink - LSE). When everything was masked (LSE = -1e30), this is
    # exp(very_negative) ≈ 0 → no spurious dsink contribution.
    p_sink = tl.exp(sink - lse)
    dsink_local = -p_sink * delta  # [BLOCK_H]

    # ---- Initialize dQ accumulator
    dq_acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)

    # ---- Loop over K blocks
    NUM_BLOCKS: tl.constexpr = (K_TOPK + BLOCK_N - 1) // BLOCK_N

    for n_block in tl.static_range(NUM_BLOCKS):
        n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_in_range = n_offsets < K_TOPK

        idx_ptrs = TopkIdxs_ptr + pid_b * stride_ib + pid_s * stride_is + n_offsets * stride_ik
        idxs = tl.load(idx_ptrs, mask=n_in_range, other=-1)
        valid = (idxs >= 0) & n_in_range
        safe_idxs = tl.where(valid, idxs, 0).to(tl.int64)

        # Gather KV
        kv_ptrs = (
            KV_ptr + pid_b * stride_kvb + safe_idxs[:, None] * stride_kvn + d_offsets[None, :] * stride_kvd
        )
        kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0)  # [BLOCK_N, D]

        # Recompute scores; mask invalids to _M_INIT.
        scores = tl.dot(q, tl.trans(kv)) * softmax_scale  # [BLOCK_H, BLOCK_N]
        scores = tl.where(valid[None, :], scores, _M_INIT)

        # P = exp(scores - LSE). For invalid entries, p = 0 (since scores=_M_INIT).
        p = tl.exp(scores - lse[:, None])  # [BLOCK_H, BLOCK_N]
        p = tl.where(valid[None, :], p, 0.0)

        # dP_scaled = p * (do · V_j − Δ) · scale [BLOCK_H, BLOCK_N]
        do_v = tl.dot(do, tl.trans(kv))  # [BLOCK_H, BLOCK_N]
        dp = p * (do_v - delta[:, None]) * softmax_scale
        dp = tl.where(valid[None, :], dp, 0.0)

        # dQ contribution: dp @ KV
        dq_acc += tl.dot(dp.to(kv.dtype), kv)

        # dKV partial:
        # dK contribution from this (b, s, head_block): dp.T @ q [BLOCK_N, D]
        # dV contribution from this (b, s, head_block): p.T @ do [BLOCK_N, D]
        # In MQA, K and V share the tensor, so dKV = dK + dV.
        dk_partial = tl.dot(tl.trans(dp.to(q.dtype)), q)  # [BLOCK_N, D]
        dv_partial = tl.dot(tl.trans(p.to(do.dtype)), do)  # [BLOCK_N, D]
        dkv_partial = (dk_partial + dv_partial).to(tl.float32)

        # Atomic-add to FP32 dKV buffer at the selected indices.
        dkv_ptrs = (
            dKV_ptr
            + pid_b * stride_dkvb
            + safe_idxs[:, None] * stride_dkvn
            + d_offsets[None, :] * stride_dkvd
        )
        tl.atomic_add(dkv_ptrs, dkv_partial, mask=valid[:, None])

        # ---- Store dQ (per-query — no race)
        dq = dq_acc.to(tl.bfloat16)
        dq_ptrs = (
            dQ_ptr
            + pid_b * stride_dqb
            + pid_s * stride_dqs
            + h_offsets[:, None] * stride_dqh
            + d_offsets[None, :] * stride_dqd
        )
        tl.store(dq_ptrs, dq, mask=h_mask[:, None])

        # ---- Atomic-add dsink
        dsink_ptrs = dSink_ptr + h_offsets
        tl.atomic_add(dsink_ptrs, dsink_local, mask=h_mask)


def sparse_attn_bwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    o: torch.Tensor,
    lse: torch.Tensor,
    do: torch.Tensor,
    softmax_scale: Optional[float] = None,
    *,
    block_h: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-backed ``sparse_attn`` backward.

    Args:
    q, kv, attn_sink, topk_idxs: same as in :func:`sparse_attn_fwd`.
    o, lse: outputs of the forward pass.
    do: grad-of-loss wrt o, same shape/dtype as o.
    softmax_scale: must match the forward pass.

    Returns:
    ``(dq, dkv, dattn_sink)`` with shapes/dtypes matching ``q``, ``kv``,
    ``attn_sink`` respectively.
    """
    assert q.is_cuda and kv.is_cuda and do.is_cuda
    assert q.shape == do.shape, "do must match q shape"
    assert q.dtype == kv.dtype == do.dtype
    assert attn_sink.dtype == torch.float32
    assert lse.dtype == torch.float32

    B, S, H, D = q.shape
    N_kv = kv.shape[1]
    K_TOPK = topk_idxs.shape[-1]

    if D & (D - 1) != 0:
        raise ValueError(f"D ({D}) must be a power of 2 for tl.dot")
    if softmax_scale is None:
        softmax_scale = D**-0.5
        if block_h is None:
            block_h = max(16, min(triton.next_power_of_2(H), 64))

            idxs = topk_idxs.to(torch.int32).contiguous()

            # 1) Compute Δ
            delta = torch.empty((B, S, H), device=q.device, dtype=torch.float32)
            grid = (S, B, triton.cdiv(H, block_h))
            _delta_kernel[grid](
                o,
                do,
                delta,
                o.stride(0),
                o.stride(1),
                o.stride(2),
                o.stride(3),
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                delta.stride(0),
                delta.stride(1),
                delta.stride(2),
                H=H,
                D=D,
                BLOCK_H=block_h,
            )

            # 2) Allocate FP32 dKV buffer (atomic adds), zero-initialized.
            dkv_fp32 = torch.zeros((B, N_kv, D), device=q.device, dtype=torch.float32)
            dq = torch.empty_like(q)
            dsink = torch.zeros_like(attn_sink)  # FP32, zero-init for atomic_add

            grid = (S, B, triton.cdiv(H, block_h))
            _sparse_attn_bwd_kernel[grid](
                q,
                kv,
                attn_sink,
                idxs,
                do,
                lse,
                delta,
                dq,
                dkv_fp32,
                dsink,
                softmax_scale,
                # Q
                q.stride(0),
                q.stride(1),
                q.stride(2),
                q.stride(3),
                # KV
                kv.stride(0),
                kv.stride(1),
                kv.stride(2),
                # idxs
                idxs.stride(0),
                idxs.stride(1),
                idxs.stride(2),
                # dO
                do.stride(0),
                do.stride(1),
                do.stride(2),
                do.stride(3),
                # LSE
                lse.stride(0),
                lse.stride(1),
                lse.stride(2),
                # Δ
                delta.stride(0),
                delta.stride(1),
                delta.stride(2),
                # dQ
                dq.stride(0),
                dq.stride(1),
                dq.stride(2),
                dq.stride(3),
                # dKV
                dkv_fp32.stride(0),
                dkv_fp32.stride(1),
                dkv_fp32.stride(2),
                H=H,
                D=D,
                K_TOPK=K_TOPK,
                BLOCK_H=block_h,
            )

            # 3) Postprocess: cast FP32 dKV → BF16 (matches kv dtype).
            dkv = dkv_fp32.to(kv.dtype)
            return dq, dkv, dsink


__all__ = ["sparse_attn_bwd"]
