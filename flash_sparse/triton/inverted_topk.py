"""Build the inverted-topk index used by the inverted-topk backward kernel.

For each compressed-key index ``s_global``, ``inv_topk[b, s_global, j]`` is
the ``j``-th query token that selected ``s_global`` in its top-k. We also
record ``inv_slots[b, s_global, j]`` — which slot of that query's top-k
``s_global`` occupies (used in some bwd variants; not needed if we recompute
P from Q+K+LSE).

This index turns the sparse-scatter dKV write pattern into a structured
gather-then-sum-reduce, removing global atomic contention on H200's atomic
units. See ``docs/fusion_analysis.md`` § 6.

Build cost (per layer, once per fwd+bwd pair) measured in
``ir_analysis.md`` § 5: about 8 KB / token (read top-k + write inv_topk).
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _build_inverted_topk_kernel(
    TopkIdxs_ptr,  # [B, S, K_TOPK] int32
    InvTopk_ptr,  # [B, N_kv, K_MAX] int32 — query indices
    InvCount_ptr,  # [B, N_kv] int32 — number of queries that selected each kv row
    # strides
    stride_tib,
    stride_tis,
    stride_tik,
    stride_qb,
    stride_qn,
    stride_qk,
    stride_cb,
    stride_cn,
    # constants
    K_TOPK: tl.constexpr,
    K_MAX: tl.constexpr,
    N_KV: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One program per (batch, query_position, k_block): scatters its top-k
    entries into the inverted index via atomic-add to per-`s_global` counters."""
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_kb = tl.program_id(2)

    k_offsets = pid_kb * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K_TOPK

    # Load this query's top-k entries.
    idx_ptrs = TopkIdxs_ptr + pid_b * stride_tib + pid_s * stride_tis + k_offsets * stride_tik
    s_globals = tl.load(idx_ptrs, mask=k_mask, other=-1)
    valid = (s_globals >= 0) & (s_globals < N_KV) & k_mask

    # Atomic-add to InvCount[b, s_global] returns the slot to write to.
    # Threads contending on the same s_global serialize internally; each gets
    # a unique `j`.
    safe_s = tl.where(valid, s_globals, 0).to(tl.int32)
    count_ptrs = InvCount_ptr + pid_b * stride_cb + safe_s * stride_cn
    increments = tl.where(valid, 1, 0).to(tl.int32)
    js = tl.atomic_add(count_ptrs, increments, mask=valid)

    # Write into the inverted index iff the slot fits.
    write_mask = valid & (js < K_MAX)
    inv_ptrs = InvTopk_ptr + pid_b * stride_qb + safe_s * stride_qn + js * stride_qk
    tl.store(inv_ptrs, pid_s.to(tl.int32), mask=write_mask)


def build_inverted_topk(
    topk_idxs: torch.Tensor,
    n_kv: int,
    k_max: int,
    *,
    block_k: int = 64,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the inverted top-k index.

    Args:
    topk_idxs: ``[B, S, K_TOPK]`` int32. ``-1`` = masked.
    n_kv: number of compressed-key rows (size of the kv tensor's seq dim).
    k_max: maximum number of queries any single ``s_global`` can be
    selected by (sets the inv_topk last-dim size). If a ``s_global``
    is selected by more than k_max queries, the overflow is dropped
    and the count still increments — caller may want to clamp
    the count to k_max before using.
    block_k: K-block size for the build kernel. Default 64.

    Returns:
    ``(inv_topk, inv_count)`` where:
    inv_topk: ``[B, n_kv, k_max]`` int32 — query indices that selected each s
    (only the first ``inv_count[b, s]`` (clamped to k_max) entries
    are valid; the rest contain stale data).
    inv_count: ``[B, n_kv]`` int32 — number of queries that selected each s
    (NOT clamped to k_max — caller can detect overflow if needed).
    """
    assert topk_idxs.is_cuda
    B, S, K_TOPK = topk_idxs.shape

    idxs = topk_idxs.to(torch.int32).contiguous()
    inv_topk = torch.empty((B, n_kv, k_max), dtype=torch.int32, device=topk_idxs.device)
    inv_count = torch.zeros((B, n_kv), dtype=torch.int32, device=topk_idxs.device)

    grid = (B, S, triton.cdiv(K_TOPK, block_k))
    _build_inverted_topk_kernel[grid](
        idxs,
        inv_topk,
        inv_count,
        idxs.stride(0),
        idxs.stride(1),
        idxs.stride(2),
        inv_topk.stride(0),
        inv_topk.stride(1),
        inv_topk.stride(2),
        inv_count.stride(0),
        inv_count.stride(1),
        K_TOPK=K_TOPK,
        K_MAX=k_max,
        N_KV=n_kv,
        BLOCK_K=block_k,
    )

    return inv_topk, inv_count


__all__ = ["build_inverted_topk"]
