"""FP8 / FP4 quantization utilities.

Phase-0 stubs that mirror the DeepSeek-V4 quantization scheme:

- ``act_quant``: per-block (block_size=128) FP8 e4m3 with FP32 or UE8M0 scale.
- ``fp4_act_quant``: per-block (block_size=32) FP4 e2m1 with UE8M0 scale, plus
an optional Hadamard rotation before quantization.

Real kernels arrive in future work.
"""

from __future__ import annotations

import torch

__all__ = ["act_quant_simulate", "fp4_act_quant_simulate"]


def act_quant_simulate(
    x: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Quant-then-dequant FP8 simulation, returning a tensor of the same dtype.

    Used by the reference implementation to model FP8 quantization noise without
    going through CUDA kernels. Per-block-of-128 dynamic scaling on the last dim.
    """
    *prefix, n = x.shape
    assert n % block_size == 0, f"last dim {n} must be divisible by block_size {block_size}"
    fp8_max = 448.0
    blocks = x.reshape(*prefix, n // block_size, block_size)
    amax = blocks.abs.amax(dim=-1, keepdim=True).clamp_min(1e-4)
    scale = amax / fp8_max
    quant = (blocks / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    dequant = quant.float() * scale
    return dequant.reshape(*prefix, n).to(x.dtype)


def fp4_act_quant_simulate(
    x: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """Quant-then-dequant FP4 simulation, returning a tensor of the same dtype.

    Per-block-of-32 dynamic scaling. The FP4 e2m1 format has max magnitude 6.0.
    """
    *prefix, n = x.shape
    assert n % block_size == 0, f"last dim {n} must be divisible by block_size {block_size}"
    fp4_max = 6.0
    blocks = x.reshape(*prefix, n // block_size, block_size)
    amax = blocks.abs.amax(dim=-1, keepdim=True).clamp_min(6 * 2.0**-126)
    scale = amax / fp4_max
    # Round-to-nearest at the FP4 grid (representable values: 0, 0.5, 1, 1.5, 2, 3, 4, 6).
    fp4_grid = torch.tensor(
        [-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6],
        dtype=x.dtype,
        device=x.device,
    )
    scaled = (blocks / scale).clamp(-fp4_max, fp4_max)
    # Snap to nearest FP4 value.
    diffs = (scaled.unsqueeze(-1) - fp4_grid).abs
    quant_idx = diffs.argmin(dim=-1)
    quant = fp4_grid[quant_idx]
    dequant = quant * scale
    return dequant.reshape(*prefix, n).to(x.dtype)
