"""research artifact — fused streaming top-k indexer (Insight 1).

Status: **research artifact, not production-ready.** Kept in-tree for the
the kernel-level work CUDA port and to document what doesn't work in Triton 3.7.

Goal (per ``docs/streaming_topk.md``): keep the [B, S, T] indexer score
matrix entirely in registers / SRAM, write only the top-k indices to HBM.
Saves ~2 MB / token at V4-Pro 1M context.

Why the Triton attempt is blocked:

1. **`tl.cat` requires same-shape operands.** A natural streaming design —
``cat([top_k[k], new_block[BLOCK_T]])`` then ``tl.sort`` — is rejected.
Workaround forces ``BLOCK_T = K_TOPK = 1024`` per iteration; loads
1024 × c_I FP8 KV per inner step (256 KB), busting the SRAM budget for
V4-Pro c_I = 128.

2. **Bit-packed (score, index) into uint64 fails on negative scores.**
The lightning indexer score `I(t, s) = sum_h w_h · ReLU(q_h · k_s)` can
be negative when the trained weights `w_h` are negative (which happens
in practice; ``weights_proj`` is unconstrained). The naive
``fp32-bit-cast >> 32 | idx`` packing gives wrong sort order in that
regime. The correct fix needs the FP32→sortable-uint32 encoding
(XOR sign bit / invert if negative), but combining that with Triton's
`tl.cat`/`tl.sort`/`tl.split` constraints is fragile.

3. **No `tl.argsort`** in Triton 3.7 — must track indices alongside scores
manually, which the constraints above make awkward.

code review (2026-04-25) recommended: separate FP32 score / INT32 index
buffers + bitonic merge of only the BLOCK_T new candidates against the
current top-k tail. That's a 200+-line custom merge implementation in
Triton; deferred to CUDA where shared-memory atomics + SHFL primitives
make heap-of-k natural.

For now the production indexer path is ``flash_sparse.triton.indexer_score``
+ ``torch.topk`` — which materializes the score matrix in HBM but is
correct and fast for our test sizes (T ≤ 1024).
"""

from __future__ import annotations

# Imports kept so the module is import-safe; all kernels are commented-out
# below. To revive, see docs/streaming_topk.md § 3.
import torch  # noqa: F401


def streaming_indexer_topk(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise NotImplementedError(
        "Streaming top-k Triton kernel is parked. See module docstring. "
        "Use flash_sparse.triton.indexer_score + torch.topk instead."
    )


__all__ = ["streaming_indexer_topk"]
