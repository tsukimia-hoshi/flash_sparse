"""2D tiled indexer + top-k for image/video-style local retrieval.

This module intentionally lives outside ``flash_sparse.triton.chunked_indexer`` so
that the 2D non-causal spatial indexer does not share control flow with the
legacy 1D causal indexer.

The public wrapper accepts spatial query/key tensors, iterates over Q in 2D
``(tile_h, tile_w)`` blocks, expands each block to a K halo box on every frame,
optionally projects that halo onto a compressed K grid, flattens the selected
local tokens to the 1D contract consumed by the existing ``indexer_score``
kernel, and maps local top-k positions back to absolute flattened K-grid
indices.
"""

from __future__ import annotations

from typing import Tuple

import torch

from flash_sparse.triton.indexer_score import indexer_score


Halo = int | Tuple[int, int]
CompressionRate = int | Tuple[int, int]


def _normalize_pair(value: int | Tuple[int, int], *, name: str, allow_zero: bool) -> Tuple[int, int]:
    if isinstance(value, int):
        first = second = value
    else:
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or a ({name}_h, {name}_w) tuple, got {value!r}")
        first, second = value

    lower_bound = 0 if allow_zero else 1
    if first < lower_bound or second < lower_bound:
        comparator = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} values must be {comparator}, got {(first, second)}")
    return first, second


def _normalize_halo(halo: Halo) -> Tuple[int, int]:
    return _normalize_pair(halo, name="halo", allow_zero=True)


def _normalize_compression_rate(compression_rate: CompressionRate) -> Tuple[int, int]:
    return _normalize_pair(compression_rate, name="compression_rate", allow_zero=False)


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _validate_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _halo_bounds(
    q_h_start: int,
    q_h_end: int,
    q_w_start: int,
    q_w_end: int,
    image_h: int,
    image_w: int,
    key_h: int,
    key_w: int,
    halo_h: int,
    halo_w: int,
    compression_h: int,
    compression_w: int,
) -> Tuple[int, int, int, int]:
    """Return clipped compressed-K ``(h_start, h_end, w_start, w_end)`` bounds.

    ``halo`` is expressed in full-resolution Q-token coordinates.  Compressed K
    cells cover ``compression_h * compression_w`` source tokens, so full-res halo
    bounds are converted to the compressed grid by selecting every compressed
    cell that overlaps the full-res box.
    """
    full_h_start = max(q_h_start - halo_h, 0)
    full_h_end = min(q_h_end + halo_h, image_h)
    full_w_start = max(q_w_start - halo_w, 0)
    full_w_end = min(q_w_end + halo_w, image_w)

    k_h_start = min(full_h_start // compression_h, key_h)
    k_h_end = min(_ceil_div(full_h_end, compression_h), key_h)
    k_w_start = min(full_w_start // compression_w, key_w)
    k_w_end = min(_ceil_div(full_w_end, compression_w), key_w)
    return k_h_start, k_h_end, k_w_start, k_w_end


def _make_halo_global_indices(
    *,
    num_frames: int,
    key_h: int,
    key_w: int,
    h_start: int,
    h_end: int,
    w_start: int,
    w_end: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the local-relative to global-absolute compressed-K index table.

    The returned vector matches the flatten order used for K halo blocks:
    ``[T, halo_h, halo_w] -> [T * halo_h * halo_w]``.  Absolute indices use
    row-major K-grid order ``frame * H_k * W_k + y_k * W_k + x_k``.  When
    ``compression_rate=1``, this is the original full-resolution ``T * H * W``
    index space.  When compression is enabled, this mirrors the 1D indexer's
    behavior by returning indices in the compressed K grid.
    """
    frame_offsets = torch.arange(num_frames, device=device, dtype=torch.int64)[:, None, None]
    compressed_y = torch.arange(h_start, h_end, device=device, dtype=torch.int64)[None, :, None]
    compressed_x = torch.arange(w_start, w_end, device=device, dtype=torch.int64)[None, None, :]
    global_indices = frame_offsets * (key_h * key_w) + compressed_y * key_w + compressed_x
    return global_indices.reshape(-1)


def _score_local_halo(
    q_tile: torch.Tensor,
    k_halo: torch.Tensor,
    weights_tile: torch.Tensor,
) -> torch.Tensor:
    """Score a flattened Q tile against a flattened, head-shared K halo.

    This matches the original 1D lightning-indexer contract: Q has an indexer
    head dimension, K has no head dimension, and per-head scores are weighted
    and summed.
    """
    if q_tile.is_cuda and k_halo.is_cuda and weights_tile.is_cuda:
        return indexer_score(q_tile, k_halo, weights_tile)

    work_dtype = torch.float32
    raw = torch.einsum("bqhd,bkd->bqhk", q_tile.to(work_dtype), k_halo.to(work_dtype))
    scores = torch.relu(raw) * weights_tile.to(work_dtype).unsqueeze(-1)
    return scores.sum(dim=2)


def chunked_indexer_2d_topk(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
    *,
    tile_h: int,
    tile_w: int,
    halo: Halo,
    compression_rate: CompressionRate = 1,
    m_2d: CompressionRate | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run 2D local-halo indexer top-k.

    Args:
        q: Query indexer tensor ``[B, H, W, H_I, D_I]``.
        k_idx: Head-shared key indexer tensor ``[B, T, H_k, W_k, D_I]``.
            This intentionally matches the original 1D lightning-indexer
            contract: Q has indexer heads, K does not. ``H_k`` and ``W_k`` equal
            ``H`` and ``W`` when uncompressed, or ``ceil(H / m_h)`` and
            ``ceil(W / m_w)`` when spatial compression is enabled.
        weights: Per-query head weights ``[B, H, W, H_I]``.
        top_k: Number of local halo keys selected for each spatial query.
        tile_h: Query tile height for the outer 2D loop.
        tile_w: Query tile width for the outer 2D loop.
        halo: Spatial halo radius in full-resolution Q-token coordinates.
            Either one int for both dimensions or ``(halo_h, halo_w)``.
        compression_rate: Spatial K compression ratio. ``1`` means K has the
            same ``H x W`` grid as Q. ``m`` means each compressed K cell covers
            an ``m x m`` source-token block, and ``(m_h, m_w)`` supports
            rectangular compression. For example, ``compression_rate=2`` expects
            K spatial shape ``ceil(H / 2) x ceil(W / 2)`` and returns selected
            positions in that compressed K grid.
        m_2d: Alias for ``compression_rate`` for callers that use the 1D
            indexer's ``m`` naming convention. Do not specify both.

    Returns:
        ``(top_k_indices, top_k_scores)`` where indices have shape
        ``[B, H, W, top_k]`` and store absolute flattened key positions in
        row-major K-grid order ``frame * H_k * W_k + y_k * W_k + x_k``. Invalid
        padded positions are ``-1`` with score ``-inf`` when ``top_k`` is larger
        than a tile's local halo size.
    """
    _validate_positive("tile_h", tile_h)
    _validate_positive("tile_w", tile_w)
    _validate_positive("top_k", top_k)
    halo_h, halo_w = _normalize_halo(halo)
    if m_2d is not None:
        if compression_rate != 1:
            raise ValueError("specify either compression_rate or m_2d, not both")
        compression_rate = m_2d
    compression_h, compression_w = _normalize_compression_rate(compression_rate)

    if q.dim() != 5:
        raise ValueError(f"q must be [B,H,W,H_I,D_I], got {tuple(q.shape)}")
    if k_idx.dim() != 5:
        raise ValueError(f"k_idx must be head-shared [B,T,H_k,W_k,D_I], got {tuple(k_idx.shape)}")
    if weights.dim() != 4:
        raise ValueError(f"weights must be [B,H,W,H_I], got {tuple(weights.shape)}")

    B, image_h, image_w, num_heads, head_dim = q.shape
    if weights.shape != (B, image_h, image_w, num_heads):
        raise ValueError(f"weights shape mismatch: expected {(B, image_h, image_w, num_heads)}, got {tuple(weights.shape)}")

    expected_key_h = _ceil_div(image_h, compression_h)
    expected_key_w = _ceil_div(image_w, compression_w)

    B_k, num_frames, key_h, key_w, key_dim = k_idx.shape
    if (B_k, key_h, key_w, key_dim) != (B, expected_key_h, expected_key_w, head_dim):
        raise ValueError(
            "head-shared k_idx shape mismatch: expected "
            f"{(B, num_frames, expected_key_h, expected_key_w, head_dim)} "
            f"for compression_rate={(compression_h, compression_w)}, got {tuple(k_idx.shape)}"
        )

    out_idx = torch.full((B, image_h, image_w, top_k), -1, dtype=torch.int64, device=q.device)
    out_scores = torch.full((B, image_h, image_w, top_k), float("-inf"), dtype=torch.float32, device=q.device)

    for h_start in range(0, image_h, tile_h):
        h_end = min(h_start + tile_h, image_h)
        for w_start in range(0, image_w, tile_w):
            w_end = min(w_start + tile_w, image_w)
            k_h_start, k_h_end, k_w_start, k_w_end = _halo_bounds(
                h_start,
                h_end,
                w_start,
                w_end,
                image_h,
                image_w,
                key_h,
                key_w,
                halo_h,
                halo_w,
                compression_h,
                compression_w,
            )

            q_tile = q[:, h_start:h_end, w_start:w_end].contiguous()
            weights_tile = weights[:, h_start:h_end, w_start:w_end].contiguous()
            q_token_count = (h_end - h_start) * (w_end - w_start)
            q_flat = q_tile.reshape(B, q_token_count, num_heads, head_dim)
            weights_flat = weights_tile.reshape(B, q_token_count, num_heads)

            k_halo = k_idx[:, :, k_h_start:k_h_end, k_w_start:k_w_end].contiguous()
            k_flat = k_halo.reshape(B, -1, head_dim)

            scores = _score_local_halo(q_flat, k_flat, weights_flat)
            k_eff = min(top_k, k_flat.shape[1])
            top_scores, top_local_idx = scores.topk(k_eff, dim=-1)

            global_index_table = _make_halo_global_indices(
                num_frames=num_frames,
                key_h=key_h,
                key_w=key_w,
                h_start=k_h_start,
                h_end=k_h_end,
                w_start=k_w_start,
                w_end=k_w_end,
                device=q.device,
            )
            top_global_idx = global_index_table.gather(0, top_local_idx.reshape(-1)).reshape(
                B, h_end - h_start, w_end - w_start, k_eff
            )
            top_scores = top_scores.reshape(B, h_end - h_start, w_end - w_start, k_eff)

            out_idx[:, h_start:h_end, w_start:w_end, :k_eff] = top_global_idx
            out_scores[:, h_start:h_end, w_start:w_end, :k_eff] = top_scores

    return out_idx, out_scores


__all__ = [
    "chunked_indexer_2d_topk",
    "_halo_bounds",
    "_make_halo_global_indices",
]
