# 2D Lightning Indexer

`flash_sparse.triton.indexer_2d` provides a standalone 2D, non-causal lightning
indexer for image/video-style retrieval.  It is intentionally kept separate from
`flash_sparse.triton.chunked_indexer`, which is the legacy 1D causal chunked
indexer.

The 2D indexer processes full-resolution query tokens on an `H x W` grid, looks
up candidate key tokens from a local 2D halo on every frame, optionally projects
that halo onto a spatially compressed key grid, scores the local candidates, and
returns top-k key indices for every query location.

## Import

```python
from flash_sparse.triton.indexer_2d import chunked_indexer_2d_topk
```

## Public API

```python
top_idx, top_scores = chunked_indexer_2d_topk(
    q,
    k_idx,
    weights,
    top_k,
    *,
    tile_h,
    tile_w,
    halo,
    compression_rate=1,
    m_2d=None,
)
```

## Inputs

### `q`

Query indexer tensor with shape:

```text
[B, H, W, H_I, D_I]
```

Meaning:

- `B`: batch size.
- `H`, `W`: full-resolution query spatial grid.
- `H_I`: number of indexer heads.
- `D_I`: indexer head dimension.

### `k_idx`

Key indexer tensor with shape:

```text
[B, T, H_k, W_k, D_I]
```

K is intentionally head-shared, matching the original 1D `indexer_score` ABI:
Q has an indexer-head dimension, but K does not. After the local 2D halo is
flattened, the scorer sees the same contract as the 1D indexer:
`[B, Q_local, H_I, D_I]` queries against `[B, K_local, D_I]` keys.

Meaning:

- `T`: number of frames.
- `H_k`, `W_k`: spatial key grid size.
- If `compression_rate=1`, then `H_k = H` and `W_k = W`.
- If `compression_rate=(m_h, m_w)`, then:

```text
H_k = ceil(H / m_h)
W_k = ceil(W / m_w)
```

For example, if `H = W = 64` and `compression_rate=2`, the expected key grid is
`32 x 32`, and each key cell represents a `2 x 2` block of full-resolution query
positions.

### `weights`

Per-query indexer-head weights with shape:

```text
[B, H, W, H_I]
```

These weights are applied to the per-head ReLU dot-product scores before summing
across heads.

### `top_k`

Number of local key candidates selected for each query token. The output always
has `top_k` slots. If a local halo has fewer than `top_k` key cells, extra slots
are filled with index `-1` and score `-inf`.

### `tile_h`, `tile_w`

2D query tile size used by the outer loop. The indexer iterates over query tiles
of shape approximately:

```text
tile_h x tile_w
```

The final tiles on the bottom or right edge may be smaller when `H` or `W` is not
divisible by the tile size.

### `halo`

Spatial halo radius, expressed in full-resolution query-token coordinates.

Accepted forms:

```python
halo=4          # halo_h = halo_w = 4
halo=(4, 8)     # halo_h = 4, halo_w = 8
```

For a query tile covering full-resolution rows `[h0, h1)` and columns `[w0, w1)`,
the full-resolution halo box is clipped to image boundaries:

```text
[max(h0 - halo_h, 0), min(h1 + halo_h, H))
[max(w0 - halo_w, 0), min(w1 + halo_w, W))
```

When spatial compression is enabled, every compressed key cell overlapping this
full-resolution halo box is selected.

### `compression_rate` / `m_2d`

Spatial key compression ratio. `m_2d` is an alias for callers that prefer the 1D
indexer naming convention. Do not pass both `compression_rate` and `m_2d`.

Accepted forms:

```python
compression_rate=1        # no spatial compression
compression_rate=2        # each K cell covers 2 x 2 full-resolution tokens
compression_rate=(2, 4)   # each K cell covers 2 x 4 full-resolution tokens
m_2d=2                    # alias for compression_rate=2
```

The compression ratio must be positive. `halo` remains in full-resolution
coordinates even when compression is enabled.

## Outputs

The function returns:

```python
top_idx, top_scores
```

### `top_idx`

Shape:

```text
[B, H, W, top_k]
```

Each value is an absolute flattened key-grid index in row-major video order:

```text
frame * H_k * W_k + y_k * W_k + x_k
```

Where:

- `frame` is in `[0, T)`.
- `y_k` is in `[0, H_k)`.
- `x_k` is in `[0, W_k)`.

When `compression_rate=1`, this index space is the full-resolution `T * H * W`
space. When compression is enabled, this index space is the compressed
`T * H_k * W_k` key grid.

Invalid padded slots are `-1`.

### `top_scores`

Shape:

```text
[B, H, W, top_k]
```

Each value is the corresponding lightning-indexer score for `top_idx`. Invalid
padded slots are `-inf`.

## Scoring semantics

For each query tile, the implementation:

1. Flattens the selected query tile from `[B, tile_h, tile_w, H_I, D_I]` to
   `[B, tile_h * tile_w, H_I, D_I]`.
2. Extracts the per-frame local key halo and flattens it from the 2D/3D video
   grid into a 1D key list.
3. Scores the flattened local query tokens against the flattened local key list.
4. Runs `topk` over only the local halo candidates.
5. Maps local top-k positions back to absolute flattened key-grid indices.

The flattened local scorer delegates to the native Triton `indexer_score` path on
CUDA. The CPU/debug fallback uses the same head-shared-K formula in PyTorch:
every Q indexer head is dotted with the same K vector, then per-head scores are
weighted and summed.

## Non-causal behavior

This 2D indexer is non-causal. It does not accept `causal_mask` or
`causal_ratio`, and it does not mask future frames. For every query tile, the
candidate key halo is taken from all `T` frames:

```python
k_halo = k_idx[:, :, k_h_start:k_h_end, k_w_start:k_w_end]
```

Use this module when retrieval should be constrained by 2D spatial locality, not
by 1D autoregressive causality.

## Example: uncompressed K grid

```python
import torch
from flash_sparse.triton.indexer_2d import chunked_indexer_2d_topk

B, T = 2, 8
H, W = 64, 64
H_I, D_I = 16, 64
top_k = 32

q = torch.randn(B, H, W, H_I, D_I, device="cuda", dtype=torch.bfloat16)
k_idx = torch.randn(B, T, H, W, D_I, device="cuda", dtype=torch.bfloat16)
weights = torch.randn(B, H, W, H_I, device="cuda", dtype=torch.float32)

top_idx, top_scores = chunked_indexer_2d_topk(
    q,
    k_idx,
    weights,
    top_k,
    tile_h=16,
    tile_w=16,
    halo=4,
)

assert top_idx.shape == (B, H, W, top_k)
assert top_scores.shape == (B, H, W, top_k)
```

## Example: compressed `2 x 2` K grid

```python
import torch
from flash_sparse.triton.indexer_2d import chunked_indexer_2d_topk

B, T = 2, 8
H, W = 64, 64
H_I, D_I = 16, 64
m_2d = 2
H_k = (H + m_2d - 1) // m_2d
W_k = (W + m_2d - 1) // m_2d
top_k = 32

q = torch.randn(B, H, W, H_I, D_I, device="cuda", dtype=torch.bfloat16)
k_idx = torch.randn(B, T, H_k, W_k, D_I, device="cuda", dtype=torch.bfloat16)
weights = torch.randn(B, H, W, H_I, device="cuda", dtype=torch.float32)

top_idx, top_scores = chunked_indexer_2d_topk(
    q,
    k_idx,
    weights,
    top_k,
    tile_h=16,
    tile_w=16,
    halo=4,
    m_2d=m_2d,
)

assert top_idx.shape == (B, H, W, top_k)
assert top_scores.shape == (B, H, W, top_k)
```

In this example, `top_idx` values are in `[0, T * H_k * W_k)` unless they are
padded `-1` values.
