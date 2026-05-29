"""Chunked indexer + top-k — prototype of Insight 1 at long context.

The full streaming top-k (in-kernel heap) is parked due to Triton 3.7 API
limitations (see ``indexer_streaming_topk.py``). This module gives the same
asymptotic IO improvement via a coarser-grained chunking strategy that's
implementable as a pure-python wrapper around our existing Triton
``indexer_score`` kernel:

for s_chunk in S blocks:
for t_chunk in T blocks:
scores_chunk = indexer_score(q[s_chunk], k_idx[t_chunk], w[s_chunk])
chunk_topk_v, chunk_topk_idx = scores_chunk.topk(min(k, |t_chunk|))
merge with running top_k_v / top_k_idx (size k)

Peak HBM never holds more than one (chunk_S × chunk_T) score matrix.
At V4-Pro 1M context (T = 250K), the *unchunked* score matrix is
1M × 250K × 4 B = 1 TB; chunked at chunk_S = chunk_T = 4096 it's 64 MB.

The mathematical correctness of "merge per-chunk top-ks into a global top-k"
is the streaming-top-k theorem in ``docs/streaming_topk.md``: top-k of a
multiset is invariant under partition-then-merge, since top-k is the union
of the per-partition top-ks (a strictly larger set than the global top-k,
which we then trim).

Tradeoff vs the in-kernel version:
- (+) actually works in Triton today
- (+) HBM peak reduced by ~chunk_T / T
- (-) still does HBM round-trips for chunk-level scores (chunk_S × chunk_T
bytes per chunk pair) — proportional to S · T total, so total HBM is
the same as the non-chunked version. Only PEAK is reduced.
- (-) doesn't realize the full streaming-top-k IO win (which avoids ALL
score-matrix HBM writes, not just peak)

For our purposes (showing scaling at long context), the peak reduction is
what matters: it's the difference between "won't fit on H200" and "fits".
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from flash_sparse.triton.indexer_score import indexer_score


def chunked_indexer_topk(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
    *,
    causal_mask: Optional[torch.Tensor] = None,
    causal_ratio: Optional[int] = None,
    chunk_s: int = 4096,
    chunk_t: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Chunked indexer + top-k. Returns ``(top_k_indices, top_k_scores)``.

    Args:
    q: ``[B, S, H_I, D_I]``.
    k_idx: ``[B, T, D_I]``.
    weights: ``[B, S, H_I]``.
    top_k: number of compressed entries to select per query.
    causal_mask: optional ``[B, S, T]`` bool mask; True = legal. Materializes
    O(S·T) memory; use ``causal_ratio`` instead for long context.
    causal_ratio: optional compression-ratio integer. When set, applies
    the V4 indexer's causal mask ``t < (s+1) // ratio`` on
    a per-chunk basis without materializing the full
    ``[B, S, T]`` bool tensor — peak mask memory drops to
    ``O(chunk_s · chunk_t)``. Cannot combine with
    ``causal_mask``.
    chunk_s: query chunk size for the outer S loop.
    chunk_t: compressed-key chunk size for the inner T loop.

    Returns:
    ``(top_k_indices: [B, S, top_k] int64, top_k_scores: [B, S, top_k] FP32)``.
    """
    assert q.is_cuda and k_idx.is_cuda and weights.is_cuda
    B, S, H_I, D_I = q.shape
    T = k_idx.shape[1]
    assert k_idx.shape == (B, T, D_I)
    assert weights.shape == (B, S, H_I)
    if causal_mask is not None:
        assert causal_mask.shape == (B, S, T)
        assert causal_ratio is None, "specify either causal_mask or causal_ratio, not both"

        k_eff = min(top_k, T)
        out_idx = torch.empty((B, S, top_k), dtype=torch.int64, device=q.device)
        out_scores = torch.empty((B, S, top_k), dtype=torch.float32, device=q.device)

        for s_start in range(0, S, chunk_s):
            s_end = min(s_start + chunk_s, S)
            q_s = q[:, s_start:s_end].contiguous()
            w_s = weights[:, s_start:s_end].contiguous()
            chunk_S = s_end - s_start

            # Running top-k state for this S chunk.
            run_v = torch.full(
                (B, chunk_S, top_k),
                float("-inf"),
                dtype=torch.float32,
                device=q.device,
            )
            run_i = torch.full(
                (B, chunk_S, top_k),
                -1,
                dtype=torch.int64,
                device=q.device,
            )

            for t_start in range(0, T, chunk_t):
                t_end = min(t_start + chunk_t, T)
                k_t = k_idx[:, t_start:t_end].contiguous()
                chunk_T = t_end - t_start

                # Per-chunk score matrix [B, chunk_S, chunk_T] FP32 — small.
                scores_ct = indexer_score(q_s, k_t, w_s)

                if causal_mask is not None:
                    mask_ct = causal_mask[:, s_start:s_end, t_start:t_end]
                    scores_ct = scores_ct.masked_fill(~mask_ct, float("-inf"))
                elif causal_ratio is not None:
                    # Compute the V4 indexer causal mask on-the-fly per (s, t)
                    # chunk: legal iff t < (s+1) // ratio. Mask shape is
                    # [chunk_S, chunk_T], independent of total S, T.
                    s_idx = torch.arange(
                        s_start,
                        s_end,
                        device=q.device,
                    ).unsqueeze(1)  # [chunk_S, 1]
                    t_idx = torch.arange(
                        t_start,
                        t_end,
                        device=q.device,
                    ).unsqueeze(0)  # [1, chunk_T]
                    legal_chunk = t_idx < (s_idx + 1) // causal_ratio  # [chunk_S, chunk_T]
                    scores_ct = scores_ct.masked_fill(
                        ~legal_chunk.unsqueeze(0),
                        float("-inf"),
                    )

                    # Per-chunk top-k.
                    k_chunk = min(top_k, chunk_T)
                    chunk_v, chunk_i = scores_ct.topk(k_chunk, dim=-1)
                    chunk_i = chunk_i + t_start  # offset to global compressed-key index

                    # Merge with running top-k.
                    combined_v = torch.cat([run_v, chunk_v], dim=-1)
                    combined_i = torch.cat([run_i, chunk_i.to(torch.int64)], dim=-1)
                    run_v, perm = combined_v.topk(top_k, dim=-1)
                    run_i = combined_i.gather(-1, perm)

                    # Mask out -inf entries (no legal entries) by setting their idx to -1.
                    run_i = torch.where(torch.isinf(run_v), torch.full_like(run_i, -1), run_i)

                    out_idx[:, s_start:s_end] = run_i
                    out_scores[:, s_start:s_end] = run_v

                    return out_idx, out_scores


__all__ = ["chunked_indexer_topk"]
