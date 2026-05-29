"""Pure-pytorch reference implementations of the DeepSeek-V4 hybrid attention building blocks.

These exist to be the slow-but-correct ground truth for every optimized kernel we ship.
The math follows the equations in section 2.3 of the DeepSeek-V4 technical report
(2026-04-24, hosted at huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
and exactly mirrors the implementation in
`references/DeepSeek-V4-Pro/inference/{model,kernel}.py`.

Public functions:
reference_sparse_attn — equivalent to kernel.py:sparse_attn (the core fused op)
reference_lightning_indexer — equation (16): I(t,s) = sum_h w_{t,h} ReLU(q_{t,h} . K^IComp_s)
reference_token_compressor — equations (9)-(12) for CSA (overlap=True), (20)-(23) for HCA
reference_csa_forward — full CSA forward (compressor + indexer + sliding window + sparse_attn)
reference_hca_forward — full HCA forward (compressor + sliding window + dense-over-compressed)
reference_hc_split_sinkhorn — equivalent to kernel.py:hc_split_sinkhorn (Hyper-Connection mixing)

All inputs/outputs are torch tensors. All internal accumulation is FP32.
The reference is written for clarity, not speed; do not benchmark it.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "reference_sparse_attn",
    "reference_lightning_indexer",
    "reference_token_compressor",
    "reference_csa_forward",
    "reference_hca_forward",
    "reference_hc_split_sinkhorn",
]


# ---------------------------------------------------------------------------
# 1. The core fused op: sparse attention with attention-sink-augmented softmax
# ---------------------------------------------------------------------------


def reference_sparse_attn(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Sparse attention with attention sink, MQA layout. Single-head KV.

    Mirrors `references/DeepSeek-V4-Pro/inference/kernel.py:sparse_attn`.

    Args:
    q: ``[B, S, H, D]``, bfloat16 or float — RoPE-rotated queries.
    kv: ``[B, N_kv, D]``, same dtype as q. Single shared KV head (MQA).
    The first ``n_win`` rows are sliding-window KV, the rest are compressed.
    attn_sink: ``[H]``, float32 — per-head learnable sink logit z'_h
    (eq. 27 of the V4 paper). Adds ``exp(z'_h - max)`` to the softmax denom.
    topk_idxs: ``[B, S, K]``, int — index into ``kv``'s second dim. Use ``-1`` for masked.
    For CSA: K = n_win + top_k. For HCA: K = n_win + n_compressed.
    softmax_scale: scalar, default ``1 / sqrt(D)``.

    Returns:
    ``o: [B, S, H, D]`` of the same dtype as q.

    Shape note: this is the MQA layout — ``kv`` has no head dim. Each query head reads
    the same KV row but produces an independent output. ``D`` here is ``d_qk = head_dim +
    qk_rope_head_dim`` (576 for V4-Pro, since head_dim=512, rope=64). The output ``D`` is
    the full d_qk, but downstream the caller will slice the first ``head_dim`` (=512) for
    the value path; this matches FlashMLA's "shared kv" convention.
    """
    if q.dim != 4:
        raise ValueError(f"q must be 4D [B,S,H,D], got shape {tuple(q.shape)}")
    if kv.dim != 3:
        raise ValueError(f"kv must be 3D [B,N_kv,D] (MQA), got shape {tuple(kv.shape)}")
    B, S, H, D = q.shape
    B2, N_kv, D2 = kv.shape
    if B != B2 or D != D2:
        raise ValueError(f"q/kv shape mismatch: q={tuple(q.shape)}, kv={tuple(kv.shape)}")
    if attn_sink.shape != (H,):
        raise ValueError(f"attn_sink must be [{H}], got {tuple(attn_sink.shape)}")
    if topk_idxs.shape[:2] != (B, S):
        raise ValueError(f"topk_idxs first two dims must be [{B},{S}], got {tuple(topk_idxs.shape)}")
    K = topk_idxs.shape[-1]
    if softmax_scale is None:
        softmax_scale = D**-0.5

        work_dtype = torch.float32
        out_dtype = q.dtype

        # Move idxs to long for advanced indexing; clamp -1 to 0 for safe gather.
        idxs = topk_idxs.long()
        valid = idxs >= 0  # [B, S, K]
        safe_idxs = idxs.clamp_min(0)

        # Gather kv at safe_idxs along the N_kv dim.
        # kv: [B, N_kv, D] → expand to [B, S, N_kv, D] only conceptually; use torch.gather.
        safe_idxs_exp = safe_idxs.unsqueeze(-1).expand(-1, -1, -1, D)  # [B, S, K, D]
        kv_for_gather = kv.unsqueeze(1).expand(-1, S, -1, -1)  # [B, S, N_kv, D] — view, no copy
        kv_gathered = torch.gather(kv_for_gather, dim=2, index=safe_idxs_exp).to(work_dtype)  # [B, S, K, D]
        # Zero out invalid rows so they cannot contaminate the matmul output (their score is
        # also masked to -inf below, so this is belt-and-suspenders).
        kv_gathered = kv_gathered * valid.unsqueeze(-1).to(work_dtype)

        # Scores: [B, S, H, K] = einsum_kd over D.
        scores = torch.einsum("bshd,bskd->bshk", q.to(work_dtype), kv_gathered) * softmax_scale
        scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))

        sink = attn_sink.to(work_dtype).view(1, 1, H, 1)  # broadcast-ready

        # Stabilized softmax with sink. The sink contributes only to the denominator
        # (eq. 27), not the value sum.
        # Use max(scores ∪ {sink}) as the shift for numerical stability — handles the
        # all-masked case where every score is -inf but sink is finite.
        max_with_sink = torch.maximum(scores.amax(dim=-1, keepdim=True), sink)  # [B, S, H, 1]
        # Replace any remaining -inf (shouldn't happen if sink is finite) with 0 to keep the
        # subsequent exp finite.
        max_with_sink = torch.where(
            torch.isinf(max_with_sink), torch.zeros_like(max_with_sink), max_with_sink
        )

        scores_shifted = (scores - max_with_sink).exp  # [B, S, H, K], 0 at masked positions
        sink_shifted = (sink - max_with_sink).exp.squeeze(-1)  # [B, S, H]
        denom = scores_shifted.sum(dim=-1) + sink_shifted  # [B, S, H]

        # Numerator = sum_k softmax_no_sink * V_k. Output = numerator / denom.
        out = torch.einsum("bshk,bskd->bshd", scores_shifted, kv_gathered) / denom.unsqueeze(-1)
        return out.to(out_dtype)


# ---------------------------------------------------------------------------
# 2. Lightning indexer — eq. (15)-(16) of the V4 paper
# ---------------------------------------------------------------------------


def reference_lightning_indexer(
    q_idx: torch.Tensor,
    k_idx: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Equation (16) of the V4 paper.

    ``I(t, s) = sum_{h=1}^{n_I_h} w_{t,h} · ReLU(q_{t,h} · K^IComp_s)``.

    Args:
    q_idx: ``[B, S, n_I_h, c_I]`` — indexer queries (RoPE-rotated, optionally
    Hadamard-rotated and FP4-quantized in the production path).
    k_idx: ``[B, N_compressed, c_I]`` — indexer keys (compressed via the indexer's
    own Compressor, RoPE on rope dims, FP8 quantized in production).
    weights: ``[B, S, n_I_h]`` — per-head weights (already scaled by the indexer's
    ``softmax_scale * n_I_h ** -0.5``).

    Returns:
    scores: ``[B, S, N_compressed]`` float32 — raw indexer scores. Apply causal
    mask and top-k externally.
    """
    if q_idx.dim != 4:
        raise ValueError(f"q_idx must be 4D [B,S,n_I_h,c_I], got {tuple(q_idx.shape)}")
    if k_idx.dim != 3:
        raise ValueError(f"k_idx must be 3D [B,N_comp,c_I], got {tuple(k_idx.shape)}")
    work = torch.float32
    # raw scores per head: [B, S, n_I_h, N_comp]
    qk = torch.einsum("bshd,btd->bsht", q_idx.to(work), k_idx.to(work))
    qk = qk.relu_
    # weighted sum over heads: w[b,s,h] * qk[b,s,h,t], summed over h → [B, S, N_comp]
    return (qk * weights.to(work).unsqueeze(-1)).sum(dim=2)


# ---------------------------------------------------------------------------
# 3. Token compressor — eq. (11)-(12) for CSA, (22)-(23) for HCA
# ---------------------------------------------------------------------------


def reference_token_compressor(
    c_a: torch.Tensor,
    z_a: torch.Tensor,
    b_a: torch.Tensor,
    m: int,
    *,
    c_b: Optional[torch.Tensor] = None,
    z_b: Optional[torch.Tensor] = None,
    b_b: Optional[torch.Tensor] = None,
    overlap: bool,
) -> torch.Tensor:
    """Compresses ``m`` consecutive tokens of KV into one entry via softmax-gated pooling
    with learned positional bias. Two modes:

    * ``overlap=False`` (HCA, m'=128) — single stream. Eq. (22)-(23):

    .. math::

    S_{m'i:m'(i+1)-1} = \\mathrm{Softmax}_{row}(Z_{m'i:m'(i+1)-1} + B), \\quad
    C^{Comp}_i = \\sum_{j=m'i}^{m'(i+1)-1} S_j \\odot C_j

    * ``overlap=True`` (CSA, m=4) — two overlapping streams a, b. Eq. (11)-(12):

    .. math::

    [S^a_{mi:m(i+1)-1}; S^b_{m(i-1):mi-1}]
    = \\mathrm{Softmax}_{row}([Z^a + B^a; Z^b + B^b])

    .. math::

    C^{Comp}_i = \\sum S^a_j \\odot C^a_j + \\sum S^b_j \\odot C^b_j

    For ``i = 0`` the b-stream is padded with -inf score and zero value.

    Args:
    c_a, z_a: ``[B, S, c]`` — current-block content/score streams.
    b_a: ``[m, c]`` (overlap=False) or ``[2m, c]`` (overlap=True; first m rows
    bias the b-stream, last m rows the a-stream — matches model.py).
    m: compression ratio (must divide S).
    c_b, z_b, b_b: same shapes as their _a counterparts; required iff ``overlap=True``.

    Returns:
    ``[B, n_blocks, c]`` where ``n_blocks = S // m``. Trailing tokens
    (``S % m != 0``) are dropped — the production decode path keeps a partial-block
    state separately; that is out of scope here.
    """
    if c_a.dim != 3:
        raise ValueError("c_a must be [B, S, c]")
    B, S, c = c_a.shape
    if z_a.shape != c_a.shape:
        raise ValueError("z_a must match c_a shape")
    n_blocks = S // m
    if n_blocks == 0:
        return c_a.new_zeros(B, 0, c)
    cutoff = n_blocks * m

    # Build per-block tensors with overlap convention matching model.py.
    # Non-overlap path: simple unflatten, single stream.
    if not overlap:
        if b_a.shape != (m, c):
            raise ValueError(f"b_a must be [m={m}, c={c}] when overlap=False, got {tuple(b_a.shape)}")
        ca = c_a[:, :cutoff].reshape(B, n_blocks, m, c)
        za = z_a[:, :cutoff].reshape(B, n_blocks, m, c) + b_a  # [B, n_blocks, m, c]
        weights = za.softmax(dim=2)
        return (ca * weights).sum(dim=2)

    # Overlap path (CSA, m=4):
    # b_a here has shape [2m, c]. Convention from model.py:Compressor.ape:
    # ape[:m] biases the *previous-block tail* (the "b" stream — overlapping)
    # ape[m:] biases the *current block* (the "a" stream — normal)
    # In the V4 paper notation (eq. 11), Z^a covers indices [mi, m(i+1)) and gets bias B^a;
    # Z^b covers indices [m(i-1), mi) and gets bias B^b. The 2m softmax is over the joint
    # axis. For i=0 the b stream is padded with -inf score, zero value.
    if c_b is None or z_b is None or b_b is None:
        raise ValueError("overlap=True requires c_b, z_b, b_b")
    if b_a.shape != (m, c) or b_b.shape != (m, c):
        raise ValueError(
            f"with overlap=True, b_a and b_b must each be [m={m}, c={c}], "
            f"got b_a={tuple(b_a.shape)}, b_b={tuple(b_b.shape)}"
        )

        ca = c_a[:, :cutoff].reshape(B, n_blocks, m, c)
        za = z_a[:, :cutoff].reshape(B, n_blocks, m, c) + b_a
        cb = c_b[:, :cutoff].reshape(B, n_blocks, m, c)
        zb = z_b[:, :cutoff].reshape(B, n_blocks, m, c) + b_b

        # Shift the b-stream by one block: block i's b-stream is the previous block's a-source.
        cb_shifted = torch.zeros_like(cb)  # block 0 → zeros
        cb_shifted[:, 1:] = cb[:, :-1]
        zb_shifted = torch.full_like(zb, float("-inf"))  # block 0 → -inf
        zb_shifted[:, 1:] = zb[:, :-1]

        # Concatenate along the block-internal axis (length 2m).
        c_joint = torch.cat([ca, cb_shifted], dim=2)  # [B, n_blocks, 2m, c]
        z_joint = torch.cat([za, zb_shifted], dim=2)  # [B, n_blocks, 2m, c]
        weights = z_joint.softmax(dim=2)
        return (c_joint * weights).sum(dim=2)


# ---------------------------------------------------------------------------
# 4. Hyper-Connection split-Sinkhorn (eq. used in mHC of section 2.2)
# ---------------------------------------------------------------------------


def reference_hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    """Mirrors ``kernel.py:hc_split_sinkhorn``.

    Args:
    mixes: ``[..., (2 + hc_mult) * hc_mult]`` — the raw mix vector produced upstream
    by ``F.linear(LN(x).flatten(2), hc_fn) * rsqrt(...)``.
    hc_scale: ``[3]`` — per-component scale.
    hc_base: ``[(2 + hc_mult) * hc_mult]`` — per-element bias.

    Returns:
    ``(pre, post, comb)`` with shapes ``pre[..., hc_mult]``, ``post[..., hc_mult]``,
    ``comb[..., hc_mult, hc_mult]``. ``comb`` is doubly-stochastic (after Sinkhorn).
    """
    hc = hc_mult
    mix_hc = (2 + hc) * hc
    if mixes.shape[-1] != mix_hc:
        raise ValueError(f"mixes last dim must be {mix_hc} (=(2+hc)*hc), got {mixes.shape[-1]}")
    if hc_scale.shape != (3,):
        raise ValueError(f"hc_scale must be [3], got {tuple(hc_scale.shape)}")
    if hc_base.shape != (mix_hc,):
        raise ValueError(f"hc_base must be [{mix_hc}], got {tuple(hc_base.shape)}")

    work = torch.float32
    mixes = mixes.to(work)
    hc_scale = hc_scale.to(work)
    hc_base = hc_base.to(work)

    pre_logits = mixes[..., :hc] * hc_scale[0] + hc_base[:hc]
    post_logits = mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    comb_logits = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).reshape(*mixes.shape[:-1], hc, hc)

    pre = torch.sigmoid(pre_logits) + eps
    post = 2 * torch.sigmoid(post_logits)

    # Sinkhorn-normalize comb.
    # First: row-softmax + eps (this is the "warm" first row-norm — matches kernel.py).
    comb = comb_logits.softmax(dim=-1) + eps
    # Column-norm.
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    # Iterate row-norm + col-norm.
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb


# ---------------------------------------------------------------------------
# 5. Top-level CSA / HCA composition
# ---------------------------------------------------------------------------


def reference_csa_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_compressed: torch.Tensor,
    indexer_scores: torch.Tensor,
    attn_sink: torch.Tensor,
    n_win: int,
    top_k: int,
    *,
    softmax_scale: Optional[float] = None,
    causal_mask_for_compressed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Full CSA forward for the prefill path (one layer). Builds ``topk_idxs`` and dispatches
    to ``reference_sparse_attn``. Mirrors ``Attention.forward`` in the DeepSeek reference for
    ``compress_ratio=4`` and ``start_pos=0``.

    Layout convention (matches model.py:528 ``kv = cat([kv, kv_compress], dim=1)``):
    the kv tensor passed to sparse_attn is the concatenation of the full uncompressed
    sequence (length S) and the per-layer compressed entries (length N_compressed).
    Sliding-window indices point into ``[0, S)`` and compressed indices point into
    ``[S, S + N_compressed)``.

    Args:
    q: ``[B, S, H, D]`` — RoPE-rotated queries.
    kv: ``[B, S, D]`` — full uncompressed KV for the input sequence
    (one shared KV head, MQA).
    kv_compressed: ``[B, N_compressed, D]`` — compressed KV from this layer's
    ``Compressor`` (m=4 with overlap), already RoPE-rotated.
    indexer_scores: ``[B, S, N_compressed]`` — raw output of
    ``reference_lightning_indexer``. Pass
    ``causal_mask_for_compressed`` to apply the causal mask, or
    pre-mask the scores with ``-inf`` yourself.
    attn_sink: ``[H]`` per-head sink logit.
    n_win: sliding window size.
    top_k: number of compressed entries to select per query.
    softmax_scale: defaults to ``1/sqrt(D)``.
    causal_mask_for_compressed: optional ``[B, S, N_compressed]`` bool mask;
    True = legal entry, False = masked.

    Returns:
    ``o: [B, S, H, D]``.
    """
    B, S, H, D = q.shape
    if kv.shape[:2] != (B, S):
        raise ValueError(f"kv must be [B={B}, S={S}, D], got {tuple(kv.shape)}")
    n_compressed = kv_compressed.shape[1]
    if indexer_scores.shape != (B, S, n_compressed):
        raise ValueError(
            f"indexer_scores must be [{B},{S},{n_compressed}], got {tuple(indexer_scores.shape)}"
        )

        if causal_mask_for_compressed is not None:
            indexer_scores = indexer_scores.masked_fill(~causal_mask_for_compressed, float("-inf"))

            # Top-k selection over compressed entries, per query position.
            k = min(top_k, n_compressed)
            selected_scores, top_idxs_in_comp = indexer_scores.topk(k, dim=-1)  # both [B, S, k]
            # If a selected score is -inf, treat it as masked.
            invalid_top = torch.isinf(selected_scores) & (selected_scores < 0)
            top_idxs_in_comp = top_idxs_in_comp.masked_fill(invalid_top, -1)

            # Sliding-window indices: absolute positions in [0, S), or -1 if out of range/causal.
            win_idxs = _build_sliding_window_idxs(B, S, n_win, q.device)

            # Concatenate full KV with compressed KV; shift compressed indices by S.
            kv_full = torch.cat([kv, kv_compressed], dim=1)  # [B, S + n_compressed, D]
            minus_one = torch.full_like(top_idxs_in_comp, -1)
            comp_idxs_global = torch.where(top_idxs_in_comp >= 0, top_idxs_in_comp + S, minus_one)
            topk_idxs = torch.cat([win_idxs.long(), comp_idxs_global], dim=-1).int()

            return reference_sparse_attn(q, kv_full, attn_sink, topk_idxs, softmax_scale=softmax_scale)


def reference_hca_forward(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_compressed: torch.Tensor,
    attn_sink: torch.Tensor,
    n_win: int,
    *,
    m_prime: int,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Full HCA forward for the prefill path (one layer). HCA uses *all* causally-legal
    compressed KV positions plus the sliding window — no top-k selection.

    Same kv layout convention as CSA: ``[full_uncompressed_kv (S rows); compressed_kv]``.

    Args:
    q: ``[B, S, H, D]``.
    kv: ``[B, S, D]`` — full uncompressed KV (MQA).
    kv_compressed: ``[B, N_compressed, D]``.
    attn_sink: ``[H]``.
    n_win: sliding window size.
    m_prime: HCA compression ratio (used to build the causal mask).
    """
    B, S, H, D = q.shape
    n_compressed = kv_compressed.shape[1]

    win_idxs = _build_sliding_window_idxs(B, S, n_win, q.device)

    # Compressed block i covers tokens [m_prime*i, m_prime*(i+1)). Causal: query t may
    # attend to block i iff m_prime*(i+1) - 1 <= t.
    block_last = (torch.arange(n_compressed, device=q.device) + 1) * m_prime - 1  # [N_comp]
    q_pos = torch.arange(S, device=q.device).unsqueeze(-1)  # [S, 1]
    legal = q_pos >= block_last  # [S, N_comp]
    base_idx = torch.arange(n_compressed, device=q.device).view(1, 1, -1).expand(B, S, -1)
    legal_b = legal.unsqueeze(0).expand(B, -1, -1)
    comp_idxs = torch.where(legal_b, base_idx, torch.full_like(base_idx, -1))
    comp_idxs_global = torch.where(comp_idxs >= 0, comp_idxs + S, torch.full_like(comp_idxs, -1))

    kv_full = torch.cat([kv, kv_compressed], dim=1)
    topk_idxs = torch.cat([win_idxs.long(), comp_idxs_global], dim=-1).int()

    return reference_sparse_attn(q, kv_full, attn_sink, topk_idxs, softmax_scale=softmax_scale)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_sliding_window_idxs(B: int, S: int, n_win: int, device) -> torch.Tensor:
    """Build the sliding-window topk_idxs of shape [B, S, n_win] for a prefill of length S.

    For query position t, the window is ``[max(0, t - n_win + 1), t + 1)``. Returned
    indices are ABSOLUTE positions in ``[0, S)`` — they index directly into the
    uncompressed KV tensor. Out-of-window or causally-illegal positions are ``-1``.

    Returned dtype is ``int64`` (long) to compose cleanly with ``torch.where``; the
    caller casts to int32 just before passing to ``reference_sparse_attn``.
    """
    base = torch.arange(S, device=device).unsqueeze(-1)  # [S, 1]
    offsets = torch.arange(n_win, device=device).unsqueeze(0)  # [1, n_win]
    idx = (base - n_win + 1) + offsets  # [S, n_win]
    legal = (idx >= 0) & (idx <= base)
    idx = torch.where(legal, idx, torch.full_like(idx, -1))
    return idx.unsqueeze(0).expand(B, -1, -1).contiguous()
