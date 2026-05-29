"""FlashSparse — fused IO-aware kernels for DeepSeek-V4 CSA + HCA."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("flash-sparse")
except PackageNotFoundError:
    __version__ = "0.0.1.dev0"

__all__ = ["__version__"]
