"""2D tiled indexer + top-k for image/video-style local retrieval.

This module intentionally lives outside ``flash_sparse.triton.chunked_indexer`` so
that the 2D non-causal spatial indexer does not share control flow with the
legacy 1D causal indexer.

The public wrapper accepts spatial query/key tensors, iterates over Q in 2D
``(tile_h, tile_w)`` blocks, expands each block to a K halo box on every frame,
flattens the selected local tokens to the 1D contract consumed by the existing
``indexer_score`` kernel, and maps local top-k positions back to absolute
``T * H * W`` key indices.
"""

from __future__ import annotations

from typing import Tuple

import torch

from flash_sparse.triton.indexer_score import indexer_score


Halo = int | Tuple[int, int]


def _normalize_halo(halo: Halo) -> Tuple[int, int]:
    if isinstance(halo, int):
        halo_h = halo_w = halo
    else:
        if len(halo) != 2:
            raise ValueError(f"halo must be an int or a (halo_h, halo_w) tuple, got {halo!r}")
        halo_h, halo_w = halo
    if halo_h < 0 or halo_w < 0:
        raise ValueError(f"halo values must be non-negative, got {(halo_h, halo_w)}")
    return halo_h, halo_w


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
    halo_h: int,
    halo_w: int,
) -> Tuple[int, int, int, int]:
    """Return clipped ``(k_h_start, k_h_end, k_w_start, k_w_end)`` bounds."""
    k_h_start = max(q_h_start - halo_h, 0)
    k_h_end = min(q_h_end + halo_h, image_h)
    k_w_start = max(q_w_start - halo_w, 0)
    k_w_end = min(q_w_end + halo_w, image_w)
    return k_h_start, k_h_end, k_w_start, k_w_end


def _make_halo_global_indices(
    *,
    num_frames: int,
    image_h: int,
    image_w: int,
    h_start: int,
    h_end: int,
    w_start: int,
    w_end: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the local-relative to global-absolute ``T*H*W`` index table.

    The returned vector matches the flatten order used for K halo blocks:
    ``[T, halo_h, halo_w] -> [T * halo_h * halo_w]``.  Absolute indices use
    row-major video order ``frame * H * W + y * W + x``.
    """
    frame_offsets = torch.arange(num_frames, device=device, dtype=torch.int64)[:, None, None]
    y_coords = torch.arange(h_start, h_end, device=device, dtype=torch.int64)[None, :, None]
    x_coords = torch.arange(w_start, w_end, device=device, dtype=torch.int64)[None, None, :]
    global_indices = frame_offsets * (image_h * image_w) + y_coords * image_w + x_coords
    return global_indices.reshape(-1)


def _score_head_specific_k(
    q_tile: torch.Tensor,
    k_halo: torch.Tensor,
    weights_tile: torch.Tensor,
) -> torch.Tensor:
    """Score head-specific K tensors.

    ``indexer_score``'s native 1D contract uses head-shared keys
    ``[B, T_local, D]``.  The requested 2D contract carries head-specific keys
    ``[B, T_local, H_I, D_I]``.  On CUDA, decompose this layout into one native
    ``indexer_score`` call per key head and accumulate the exact score.  The
    PyTorch path keeps CPU tests and small-head debugging usable.
    """
    if q_tile.is_cuda and k_halo.is_cuda and weights_tile.is_cuda and q_tile.shape[2] >= 16:
        scores = torch.zeros(
            (q_tile.shape[0], q_tile.shape[1], k_halo.shape[1]),
            dtype=torch.float32,
            device=q_tile.device,
        )
        for head_idx in range(q_tile.shape[2]):
            head_weights = torch.zeros_like(weights_tile)
            head_weights[:, :, head_idx] = weights_tile[:, :, head_idx]
            scores = scores + indexer_score(q_tile, k_halo[:, :, head_idx, :].contiguous(), head_weights)
        return scores

    work_dtype = torch.float32
    raw = torch.einsum("bqhd,bkhd->bqhk", q_tile.to(work_dtype), k_halo.to(work_dtype))
    scores = torch.relu(raw) * weights_tile.to(work_dtype).unsqueeze(-1)
    return scores.sum(dim=2)


def _score_local_halo(
    q_tile: torch.Tensor,
    k_halo: torch.Tensor,
    weights_tile: torch.Tensor,
) -> torch.Tensor:
    """Score a flattened Q tile against a flattened K halo.

    The head-shared key layout delegates directly to the native Triton
    ``indexer_score`` on CUDA.  The head-specific key layout is decomposed into
    per-head native calls on CUDA and uses an equivalent PyTorch implementation
    otherwise.
    """
    if k_halo.dim() == 3:
        if q_tile.is_cuda and k_halo.is_cuda and weights_tile.is_cuda:
            return indexer_score(q_tile, k_halo, weights_tile)
        work_dtype = torch.float32
        raw = torch.einsum("bqhd,bkd->bqhk", q_tile.to(work_dtype), k_halo.to(work_dtype))
        scores = torch.relu(raw) * weights_tile.to(work_dtype).unsqueeze(-1)
        return scores.sum(dim=2)
    if k_halo.dim() == 4:
        return _score_head_specific_k(q_tile, k_halo, weights_tile)
    raise ValueError(f"flattened k_halo must be [B,N,D] or [B,N,H_I,D_I], got {tuple(k_halo.shape)}")


def chunked_indexer_2d_topk(
    q: torch.Tensor,
    k_idx: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
    *,
    tile_h: int,
    tile_w: int,
    halo: Halo,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run 2D local-halo indexer top-k.

    Args:
        q: Query indexer tensor ``[B, H, W, H_I, D_I]``.
        k_idx: Key indexer tensor. The requested head-specific layout is
            ``[B, T, H, W, H_I, D_I]``. A head-shared compatibility layout
            ``[B, T, H, W, D_I]`` is also accepted and is flattened into the
            native ``indexer_score`` ABI.
        weights: Per-query head weights ``[B, H, W, H_I]``.
        top_k: Number of local halo keys selected for each spatial query.
        tile_h: Query tile height for the outer 2D loop.
        tile_w: Query tile width for the outer 2D loop.
        halo: Spatial halo radius. Either one int for both dimensions or
            ``(halo_h, halo_w)``.

    Returns:
        ``(top_k_indices, top_k_scores)`` where indices have shape
        ``[B, H, W, top_k]`` and store absolute flattened key positions in
        row-major video order ``frame * H * W + y * W + x``. Invalid padded
        positions are ``-1`` with score ``-inf`` when ``top_k`` is larger than a
        tile's local halo size.
    """
    _validate_positive("tile_h", tile_h)
    _validate_positive("tile_w", tile_w)
    _validate_positive("top_k", top_k)
    halo_h, halo_w = _normalize_halo(halo)

    if q.dim() != 5:
        raise ValueError(f"q must be [B,H,W,H_I,D_I], got {tuple(q.shape)}")
    if k_idx.dim() not in (5, 6):
        raise ValueError(f"k_idx must be [B,T,H,W,D_I] or [B,T,H,W,H_I,D_I], got {tuple(k_idx.shape)}")
    if weights.dim() != 4:
        raise ValueError(f"weights must be [B,H,W,H_I], got {tuple(weights.shape)}")

    B, image_h, image_w, num_heads, head_dim = q.shape
    if weights.shape != (B, image_h, image_w, num_heads):
        raise ValueError(f"weights shape mismatch: expected {(B, image_h, image_w, num_heads)}, got {tuple(weights.shape)}")

    if k_idx.dim() == 6:
        B_k, num_frames, key_h, key_w, key_heads, key_dim = k_idx.shape
        if (B_k, key_h, key_w, key_heads, key_dim) != (B, image_h, image_w, num_heads, head_dim):
            raise ValueError(
                "head-specific k_idx shape mismatch: expected "
                f"{(B, num_frames, image_h, image_w, num_heads, head_dim)}, got {tuple(k_idx.shape)}"
            )
    else:
        B_k, num_frames, key_h, key_w, key_dim = k_idx.shape
        if (B_k, key_h, key_w, key_dim) != (B, image_h, image_w, head_dim):
            raise ValueError(
                "head-shared k_idx shape mismatch: expected "
                f"{(B, num_frames, image_h, image_w, head_dim)}, got {tuple(k_idx.shape)}"
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
                halo_h,
                halo_w,
            )

            q_tile = q[:, h_start:h_end, w_start:w_end].contiguous()
            weights_tile = weights[:, h_start:h_end, w_start:w_end].contiguous()
            q_token_count = (h_end - h_start) * (w_end - w_start)
            q_flat = q_tile.reshape(B, q_token_count, num_heads, head_dim)
            weights_flat = weights_tile.reshape(B, q_token_count, num_heads)

            k_halo = k_idx[:, :, k_h_start:k_h_end, k_w_start:k_w_end].contiguous()
            if k_idx.dim() == 6:
                k_flat = k_halo.reshape(B, -1, num_heads, head_dim)
            else:
                k_flat = k_halo.reshape(B, -1, head_dim)

            scores = _score_local_halo(q_flat, k_flat, weights_flat)
            k_eff = min(top_k, k_flat.shape[1])
            top_scores, top_local_idx = scores.topk(k_eff, dim=-1)

            global_index_table = _make_halo_global_indices(
                num_frames=num_frames,
                image_h=image_h,
                image_w=image_w,
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
