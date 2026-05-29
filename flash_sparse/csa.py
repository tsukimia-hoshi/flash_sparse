"""Public API for Compressed Sparse Attention (CSA) — the kernel-level work integration.

End-to-end CSA forward. Composes:
1. Token-level compressor (`reference_token_compressor` for now — torch ops,
not perf-critical; Triton port is on the the kernel-level work backlog).
2. Lightning indexer + top-k (`flash_sparse.triton.indexer_score`).
3. Sliding-window indices + concatenation with compressed top-k.
4. Sparse attention core (`flash_sparse.triton.sparse_attn`).

Mirrors the prefill path of `references/DeepSeek-V4-Pro/inference/model.py:Attention.forward`
for ``compress_ratio = 4`` (CSA) and ``start_pos = 0``.
"""

from __future__ import annotations

from typing import Optional

import torch

from flash_sparse.reference import (
    _build_sliding_window_idxs,
    reference_lightning_indexer,
    reference_sparse_attn,
    reference_token_compressor,
)

__all__ = ["flash_csa_forward", "build_csa_topk_idxs"]


def build_csa_topk_idxs(
    indexer_scores: torch.Tensor,
    n_win: int,
    top_k: int,
    *,
    seq_len: int,
    n_compressed: int,
    m: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the [B, S, n_win + top_k] index tensor consumed by sparse_attn.

    Layout: first ``n_win`` columns point into the uncompressed-KV region
    (positions [0, S)); next ``top_k`` columns point into the compressed-KV
    region with offset ``S`` (positions [S, S + n_compressed)).

    Args:
    indexer_scores: ``[B, S, n_compressed]`` raw scores from the indexer.
    n_win: sliding window size.
    top_k: number of compressed entries to select.
    seq_len: the input sequence length S (used as the compressed-region offset).
    n_compressed: number of compressed entries in this layer = S / m.
    m: compression ratio (used for causal masking).
    device: tensor device.
    """
    B, S, T = indexer_scores.shape
    assert T == n_compressed
    assert S == seq_len

    # Causal mask: query at position t can attend to compressed block i iff
    # m·(i+1) - 1 ≤ t.
    block_last = (torch.arange(n_compressed, device=device) + 1) * m - 1  # [T]
    q_pos = torch.arange(S, device=device).unsqueeze(-1)  # [S, 1]
    legal = q_pos >= block_last  # [S, T]
    masked_scores = indexer_scores.masked_fill(~legal.unsqueeze(0), float("-inf"))

    k_eff = min(top_k, n_compressed)
    selected_scores, top_idx_in_comp = masked_scores.topk(k_eff, dim=-1)
    invalid = torch.isinf(selected_scores) & (selected_scores < 0)
    top_idx_in_comp = top_idx_in_comp.masked_fill(invalid, -1)

    # Pad to ``top_k`` columns if k_eff < top_k.
    if k_eff < top_k:
        pad = torch.full((B, S, top_k - k_eff), -1, dtype=top_idx_in_comp.dtype, device=device)
        top_idx_in_comp = torch.cat([top_idx_in_comp, pad], dim=-1)

        # Sliding-window indices, absolute positions in [0, S).
        win_idxs = _build_sliding_window_idxs(B, S, n_win, device)  # [B, S, n_win]

        # Translate compressed indices to global kv coordinates: + seq_len offset.
        minus_one = torch.full_like(top_idx_in_comp, -1)
        comp_idxs_global = torch.where(top_idx_in_comp >= 0, top_idx_in_comp + S, minus_one)

        return torch.cat([win_idxs.long(), comp_idxs_global.long()], dim=-1).int()


def flash_csa_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_compressed: torch.Tensor,
    q_idx: torch.Tensor,
    k_idx_compressed: torch.Tensor,
    indexer_weights: torch.Tensor,
    attn_sink: torch.Tensor,
    *,
    n_win: int,
    top_k: int,
    m: int = 4,
    softmax_scale: Optional[float] = None,
    use_triton: bool = True,
    use_chunked_indexer: Optional[bool] = None,
    chunk_s: int = 1024,
    chunk_t: int = 4096,
    auto_chunk_threshold_bytes: int = 1 << 30,  # 1 GB
) -> torch.Tensor:
    """End-to-end CSA forward (prefill path, single layer).

    Args:
    q: ``[B, S, n_h, d]`` BF16. RoPE-rotated query.
    kv: ``[B, S, d]`` BF16. Uncompressed KV (sliding-window source).
    kv_compressed: ``[B, S/m, d]`` BF16. CSA-compressed KV from this layer's
    compressor (caller is responsible for this — caller likely
    uses ``reference_token_compressor`` until the kernel-level work lands).
    q_idx: ``[B, S, n_I_h, c_I]`` BF16. Indexer queries.
    k_idx_compressed: ``[B, S/m, c_I]`` BF16. Indexer compressed keys.
    indexer_weights: ``[B, S, n_I_h]`` FP32. Per-head weights, pre-scaled.
    attn_sink: ``[n_h]`` FP32.
    n_win: sliding window size.
    top_k: number of compressed entries to select per query.
    m: compression ratio (4 for CSA).
    softmax_scale: defaults to ``1 / sqrt(d)``.
    use_triton: if True, use the Triton kernels for indexer_score and
    sparse_attn; otherwise use the pytorch reference path.

    Returns:
    ``o: [B, S, n_h, d]``.
    """
    B, S, n_h, d = q.shape
    assert kv.shape == (B, S, d)
    n_compressed = kv_compressed.shape[1]
    assert kv_compressed.shape == (B, n_compressed, d)
    assert k_idx_compressed.shape[1] == n_compressed

    # Auto-pick chunked indexer when the materialized score matrix would exceed
    # the threshold. Small-S (≤ ~32K) stays on the fast materialize path; long
    # context switches to chunked so peak HBM stays bounded.
    if use_chunked_indexer is None:
        score_matrix_bytes = B * S * n_compressed * 4
        use_chunked_indexer = score_matrix_bytes > auto_chunk_threshold_bytes

    # 1) Indexer score → top-k
    # Two implementations:
    # - Default: indexer_score (full [B, S, T] FP32 score matrix in HBM) +
    # torch.topk. Faster at small S, OOMs at long context.
    # - use_chunked_indexer=True: chunked_indexer_topk processes (chunk_S,
    # chunk_T) tiles, peak HBM bounded by chunk size. Slower at small S
    # (Python launch overhead), required for S ≳ 64K. See
    # `flash_sparse.triton.chunked_indexer` and
    # `benchmarks/bench_long_context.py`.
    if use_chunked_indexer:
        from flash_sparse.triton.chunked_indexer import chunked_indexer_topk

        # Build causal mask for the indexer.
        block_last = (torch.arange(n_compressed, device=q.device) + 1) * m - 1
        q_pos = torch.arange(S, device=q.device).unsqueeze(-1)
        causal = (q_pos >= block_last).unsqueeze(0).expand(B, -1, -1)

        top_idx_in_comp, _ = chunked_indexer_topk(
            q_idx,
            k_idx_compressed,
            indexer_weights,
            top_k=top_k,
            causal_mask=causal,
            chunk_s=chunk_s,
            chunk_t=chunk_t,
        )
        # Pad if k_eff < top_k (shouldn't happen with our chunk sizes but be safe).
        if top_idx_in_comp.shape[-1] < top_k:
            pad = torch.full(
                (B, S, top_k - top_idx_in_comp.shape[-1]),
                -1,
                dtype=top_idx_in_comp.dtype,
                device=q.device,
            )
            top_idx_in_comp = torch.cat([top_idx_in_comp, pad], dim=-1)

        win_idxs = _build_sliding_window_idxs(B, S, n_win, q.device)
        minus_one = torch.full_like(top_idx_in_comp, -1)
        comp_idxs_global = torch.where(top_idx_in_comp >= 0, top_idx_in_comp + S, minus_one)
        topk_idxs = torch.cat([win_idxs.long(), comp_idxs_global.long()], dim=-1).int()
    else:
        if use_triton:
            from flash_sparse.triton.indexer_score import indexer_score

            scores = indexer_score(q_idx, k_idx_compressed, indexer_weights)
        else:
            scores = reference_lightning_indexer(q_idx, k_idx_compressed, indexer_weights)

        topk_idxs = build_csa_topk_idxs(
            scores,
            n_win=n_win,
            top_k=top_k,
            seq_len=S,
            n_compressed=n_compressed,
            m=m,
            device=q.device,
        )

    # 3) Concatenate KV: [uncompressed (S rows); compressed (n_compressed rows)]
    kv_full = torch.cat([kv, kv_compressed], dim=1)

    # 4) Sparse attention core (Triton or reference)
    if use_triton:
        from flash_sparse.triton import sparse_attn

        return sparse_attn(q, kv_full, attn_sink, topk_idxs, softmax_scale)
    else:
        return reference_sparse_attn(q, kv_full, attn_sink, topk_idxs, softmax_scale)
