"""the kernel-level work.b (research artifact) — sparse_attn backward with inverted-topk
dK/dV reduction.

Status: correct but **slower than v1 in benchmarks on H200** (see below).
Kept in-tree as a research artifact and a starting point for the CUDA port.

Replaces v1's FP32 atomic-scatter dKV write pattern with a per-K-row
sum-reduce via a precomputed inverted-topk index. The hypothesis (from
`docs/fusion_analysis.md` § 6 first draft): eliminating global atomic
contention should yield 5-6× wall-clock speedup.

**Reality (measured 2026-04-25, H200, BF16):**

| config | v1 (atomic) | v2 (inverted-topk) | speedup |
|---|---:|---:|---:|
| S=512 K=256 H=64 | 121 µs | 2411 µs | 0.05× |
| S=1024 K=512 H=64 | 229 µs | 10020 µs | 0.02× |
| S=2048 K=1024 H=64 | 898 µs | 42068 µs | 0.02× |

**Why v2 lost**: The original IO analysis missed the cost of *re-reading Q for
every (query, kv-row) pair*. In v2, per kv row, we reload Q[H, D] for every
query that selected it. Total Q reads = S · K · H · D · 2 bytes/token, which
dwarfs v1's atomic-scatter dKV writes when H is moderately large.

For our V4-Pro target (H=128, S=1M, K=1024, N_kv=250K):
- v1 dKV atomic-write = K · D · 4 = ~256 KB/token
- v2 Q re-reads = K · H · D · 2 = ~8 MB/token (32× more)

The atomic-contention-dominated regime where v2 wins requires *very* large
S/N_kv ratios (i.e., many queries selecting the same key, saturating atomic
units). Our test workloads with K_TOPK=1024 and uniform topk distributions
do not reach that regime.

**Path forward**: a correct fusion_analysis would split-K-block the v2
approach so that Q is loaded once per CTA and reused across multiple kv
rows. That's a substantial restructure, deferred to future work.c (CUDA).

For now, `flash_sparse.triton.sparse_attn_bwd` (v1) remains the production
backward. v2 is exposed as `sparse_attn_bwd_v2` for experimentation.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from flash_sparse.triton.inverted_topk import build_inverted_topk


_M_INIT = tl.constexpr(-1.0e30)


@triton.jit
def _delta_kernel_v2(
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
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_hb = tl.program_id(2)

    h_offsets = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H
    d_offsets = tl.arange(0, D)

    o = tl.load(
        O_ptr
        + pid_b * stride_ob
        + pid_s * stride_os
        + h_offsets[:, None] * stride_oh
        + d_offsets[None, :] * stride_od,
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        dO_ptr
        + pid_b * stride_dob
        + pid_s * stride_dos
        + h_offsets[:, None] * stride_doh
        + d_offsets[None, :] * stride_dod,
        mask=h_mask[:, None],
        other=0.0,
    ).to(tl.float32)

    delta = tl.sum(o * do, axis=1)
    tl.store(
        Delta_ptr + pid_b * stride_db + pid_s * stride_ds + h_offsets * stride_dh,
        delta,
        mask=h_mask,
    )


@triton.jit
def _dq_kernel_v2(
    Q_ptr,
    KV_ptr,
    AttnSink_ptr,
    TopkIdxs_ptr,
    dO_ptr,
    LSE_ptr,
    Delta_ptr,
    dQ_ptr,
    dSink_ptr,
    softmax_scale,
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
    stride_dob,
    stride_dos,
    stride_doh,
    stride_dod,
    stride_lb,
    stride_ls,
    stride_lh,
    stride_db,
    stride_ds,
    stride_dh,
    stride_dqb,
    stride_dqs,
    stride_dqh,
    stride_dqd,
    H: tl.constexpr,
    D: tl.constexpr,
    K_TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per (b, s, head_block): compute dQ for that query (no dKV path)."""
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_hb = tl.program_id(2)

    h_offsets = pid_hb * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offsets < H
    d_offsets = tl.arange(0, D)

    q = tl.load(
        Q_ptr
        + pid_b * stride_qb
        + pid_s * stride_qs
        + h_offsets[:, None] * stride_qh
        + d_offsets[None, :] * stride_qd,
        mask=h_mask[:, None],
        other=0.0,
    )
    do = tl.load(
        dO_ptr
        + pid_b * stride_dob
        + pid_s * stride_dos
        + h_offsets[:, None] * stride_doh
        + d_offsets[None, :] * stride_dod,
        mask=h_mask[:, None],
        other=0.0,
    )
    lse = tl.load(
        LSE_ptr + pid_b * stride_lb + pid_s * stride_ls + h_offsets * stride_lh,
        mask=h_mask,
        other=0.0,
    )
    delta = tl.load(
        Delta_ptr + pid_b * stride_db + pid_s * stride_ds + h_offsets * stride_dh,
        mask=h_mask,
        other=0.0,
    )
    sink = tl.load(AttnSink_ptr + h_offsets, mask=h_mask, other=_M_INIT)
    p_sink = tl.exp(sink - lse)
    dsink_local = -p_sink * delta

    dq_acc = tl.zeros([BLOCK_H, D], dtype=tl.float32)

    NUM_BLOCKS: tl.constexpr = (K_TOPK + BLOCK_N - 1) // BLOCK_N
    for n_block in tl.static_range(NUM_BLOCKS):
        n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
        n_in_range = n_offsets < K_TOPK

        idxs = tl.load(
            TopkIdxs_ptr + pid_b * stride_ib + pid_s * stride_is + n_offsets * stride_ik,
            mask=n_in_range,
            other=-1,
        )
        valid = (idxs >= 0) & n_in_range
        safe_idxs = tl.where(valid, idxs, 0).to(tl.int64)

        kv = tl.load(
            KV_ptr + pid_b * stride_kvb + safe_idxs[:, None] * stride_kvn + d_offsets[None, :] * stride_kvd,
            mask=valid[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(valid[None, :], scores, _M_INIT)
        p = tl.exp(scores - lse[:, None])
        p = tl.where(valid[None, :], p, 0.0)

        do_v = tl.dot(do, tl.trans(kv))
        dp = p * (do_v - delta[:, None]) * softmax_scale
        dp = tl.where(valid[None, :], dp, 0.0)

        dq_acc += tl.dot(dp.to(kv.dtype), kv)

        tl.store(
            dQ_ptr
            + pid_b * stride_dqb
            + pid_s * stride_dqs
            + h_offsets[:, None] * stride_dqh
            + d_offsets[None, :] * stride_dqd,
            dq_acc.to(tl.bfloat16),
            mask=h_mask[:, None],
        )

        tl.atomic_add(dSink_ptr + h_offsets, dsink_local, mask=h_mask)


@triton.jit
def _dkv_kernel_v2(
    Q_ptr,
    KV_ptr,
    dO_ptr,
    LSE_ptr,
    Delta_ptr,
    InvTopk_ptr,
    InvCount_ptr,
    dKV_ptr,
    softmax_scale,
    stride_qb,
    stride_qs,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_dob,
    stride_dos,
    stride_doh,
    stride_dod,
    stride_lb,
    stride_ls,
    stride_lh,
    stride_deb,
    stride_des,
    stride_deh,
    stride_invb,
    stride_invn,
    stride_invk,
    stride_cb,
    stride_cn,
    stride_dkvb,
    stride_dkvn,
    stride_dkvd,
    H: tl.constexpr,
    D: tl.constexpr,
    K_MAX: tl.constexpr,
    H_PADDED: tl.constexpr,  # next_pow2(H), >= 16
    BLOCK_Q: tl.constexpr,  # queries per inner iteration
):
    """Per (b, s_global): accumulate dKV[s_global, :] across all queries that
    selected s_global. Direct BF16 write (no atomics) since one CTA owns each
    K row. ALL heads handled in one CTA (so requires H ≤ H_PADDED).
    """
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)

    h_offsets = tl.arange(0, H_PADDED)
    h_mask = h_offsets < H
    d_offsets = tl.arange(0, D)

    # Load this kv row (shared K=V in MQA)
    kv_s = tl.load(KV_ptr + pid_b * stride_kvb + pid_n * stride_kvn + d_offsets * stride_kvd)  # [D]

    # Number of queries that selected this kv row, clamped to K_MAX.
    cnt = tl.load(InvCount_ptr + pid_b * stride_cb + pid_n * stride_cn)
    cnt_capped = tl.minimum(cnt, K_MAX)

    # FP32 accumulator
    dkv_acc = tl.zeros([D], dtype=tl.float32)

    # Loop over queries that selected this row, in chunks of BLOCK_Q.
    # Dynamic range — runtime-bounded by cnt_capped, so we don't pay full K_MAX
    # cost for kv rows that few queries selected. (Most kv rows have ~ S·K/N_kv
    # queries on average, well below K_MAX.)
    for q_off_start in tl.range(0, cnt_capped, BLOCK_Q):
        q_off = q_off_start + tl.arange(0, BLOCK_Q)
        q_mask = q_off < cnt_capped

        # Load query indices: [BLOCK_Q]
        query_t = tl.load(
            InvTopk_ptr + pid_b * stride_invb + pid_n * stride_invn + q_off * stride_invk,
            mask=q_mask,
            other=0,
        ).to(tl.int64)

        # Load Q[b, query_t, h_offsets, :] → [BLOCK_Q, H_PADDED, D]
        q_tile = tl.load(
            Q_ptr
            + pid_b * stride_qb
            + query_t[:, None, None] * stride_qs
            + h_offsets[None, :, None] * stride_qh
            + d_offsets[None, None, :] * stride_qd,
            mask=q_mask[:, None, None] & h_mask[None, :, None],
            other=0.0,
        ).to(tl.float32)

        # Load dO[b, query_t, h_offsets, :] same shape
        do_tile = tl.load(
            dO_ptr
            + pid_b * stride_dob
            + query_t[:, None, None] * stride_dos
            + h_offsets[None, :, None] * stride_doh
            + d_offsets[None, None, :] * stride_dod,
            mask=q_mask[:, None, None] & h_mask[None, :, None],
            other=0.0,
        ).to(tl.float32)

        # Load LSE[b, query_t, h_offsets] and Δ[b, query_t, h_offsets] → [BLOCK_Q, H_PADDED]
        lse_t = tl.load(
            LSE_ptr + pid_b * stride_lb + query_t[:, None] * stride_ls + h_offsets[None, :] * stride_lh,
            mask=q_mask[:, None] & h_mask[None, :],
            other=0.0,
        )
        delta_t = tl.load(
            Delta_ptr + pid_b * stride_deb + query_t[:, None] * stride_des + h_offsets[None, :] * stride_deh,
            mask=q_mask[:, None] & h_mask[None, :],
            other=0.0,
        )

        # Per-head scores: scale · Q[q, h] · KV[s] = Σ_d Q[q, h, d] · KV[d]
        # Implemented as element-wise mul over d, then reduce-sum.
        scores = tl.sum(q_tile * kv_s[None, None, :], axis=2) * softmax_scale  # [BLOCK_Q, H_PADDED]

        # P[q, h] = exp(scores[q, h] - LSE[q, h]). Mask invalid q & h to 0.
        p = tl.exp(scores - lse_t)
        p = tl.where(q_mask[:, None] & h_mask[None, :], p, 0.0)

        # do · v[s] per (q, h): same pattern
        do_v = tl.sum(do_tile * kv_s[None, None, :], axis=2)

        # dp = p * (do_v - Δ) * scale: [BLOCK_Q, H_PADDED]
        dp = p * (do_v - delta_t) * softmax_scale
        dp = tl.where(q_mask[:, None] & h_mask[None, :], dp, 0.0)

        # dKV contribution combining dK + dV paths:
        # dK[s, d] += sum_(q, h) dp[q, h] * Q[q, h, d]
        # dV[s, d] += sum_(q, h) p[q, h] * dO[q, h, d]
        # dKV = dK + dV (MQA-shared storage)
        dk_outer = tl.sum(dp[:, :, None] * q_tile, axis=1)  # [BLOCK_Q, D]
        dv_outer = tl.sum(p[:, :, None] * do_tile, axis=1)  # [BLOCK_Q, D]
        dkv_acc += tl.sum(dk_outer + dv_outer, axis=0)  # [D]

        # Single direct BF16 write — no atomics.
        tl.store(
            dKV_ptr + pid_b * stride_dkvb + pid_n * stride_dkvn + d_offsets * stride_dkvd,
            dkv_acc.to(tl.bfloat16),
        )


def sparse_attn_bwd_v2(
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
    block_n: int = 64,
    block_q: int = 8,
    k_max: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Backward via inverted-topk sum-reduction (atomic-free dKV).

    Args:
    q, kv, attn_sink, topk_idxs, o, lse, do, softmax_scale:
    same as :func:`sparse_attn_bwd` (v1).
    block_h: heads per program for dQ kernel (default: ≤64)
    block_n: KV block size for dQ kernel (default 64)
    block_q: query-chunk size in the dKV kernel inner loop (default 8)
    k_max: inv_topk last-dim size. If ``None``, sized as
    ``min(q.shape[1], 4 · topk_idxs.shape[-1])``, a heuristic upper
    bound on the queries-per-key distribution.

    Returns:
    ``(dq, dkv, dattn_sink)`` with the same dtypes as v1.
    """
    assert q.is_cuda and kv.is_cuda and do.is_cuda
    assert q.shape == do.shape
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
            if k_max is None:
                # Conservative upper bound — most workloads have queries-per-key well below
                # 4·K_TOPK at the 99th percentile (cf. fusion_analysis.md § 6.1).
                k_max = min(S, 4 * K_TOPK)
                H_PADDED = max(16, triton.next_power_of_2(H))

                if H > 64:
                    # The current dKV kernel processes all heads in one CTA; split-head support
                    # comes later. For now, fall back to v1 atomic-scatter on H > 64.
                    from flash_sparse.triton.sparse_attn_bwd import sparse_attn_bwd as v1

                    return v1(q, kv, attn_sink, topk_idxs, o, lse, do, softmax_scale, block_h=block_h)

                idxs = topk_idxs.to(torch.int32).contiguous()

                # 1) Build inverted topk index
                inv_topk, inv_count = build_inverted_topk(idxs, n_kv=N_kv, k_max=k_max)

                # 2) Compute Δ
                delta = torch.empty((B, S, H), device=q.device, dtype=torch.float32)
                grid = (B, S, triton.cdiv(H, block_h))
                _delta_kernel_v2[grid](
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

                # 3) Compute dQ + dsink (per-query, no atomic on dQ; per-head atomic on dsink)
                dq = torch.empty_like(q)
                dsink = torch.zeros_like(attn_sink)
                grid = (B, S, triton.cdiv(H, block_h))
                _dq_kernel_v2[grid](
                    q,
                    kv,
                    attn_sink,
                    idxs,
                    do,
                    lse,
                    delta,
                    dq,
                    dsink,
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
                    do.stride(0),
                    do.stride(1),
                    do.stride(2),
                    do.stride(3),
                    lse.stride(0),
                    lse.stride(1),
                    lse.stride(2),
                    delta.stride(0),
                    delta.stride(1),
                    delta.stride(2),
                    dq.stride(0),
                    dq.stride(1),
                    dq.stride(2),
                    dq.stride(3),
                    H=H,
                    D=D,
                    K_TOPK=K_TOPK,
                    BLOCK_H=block_h,
                    BLOCK_N=block_n,
                )

                # 4) Compute dKV via inverted-topk sum-reduce (atomic-free)
                dkv = torch.empty((B, N_kv, D), device=q.device, dtype=q.dtype)
                grid = (B, N_kv)
                _dkv_kernel_v2[grid](
                    q,
                    kv,
                    do,
                    lse,
                    delta,
                    inv_topk,
                    inv_count,
                    dkv,
                    softmax_scale,
                    q.stride(0),
                    q.stride(1),
                    q.stride(2),
                    q.stride(3),
                    kv.stride(0),
                    kv.stride(1),
                    kv.stride(2),
                    do.stride(0),
                    do.stride(1),
                    do.stride(2),
                    do.stride(3),
                    lse.stride(0),
                    lse.stride(1),
                    lse.stride(2),
                    delta.stride(0),
                    delta.stride(1),
                    delta.stride(2),
                    inv_topk.stride(0),
                    inv_topk.stride(1),
                    inv_topk.stride(2),
                    inv_count.stride(0),
                    inv_count.stride(1),
                    dkv.stride(0),
                    dkv.stride(1),
                    dkv.stride(2),
                    H=H,
                    D=D,
                    K_MAX=k_max,
                    H_PADDED=H_PADDED,
                    BLOCK_Q=block_q,
                )

                return dq, dkv, dsink


__all__ = ["sparse_attn_bwd_v2"]
