"""
CuPy backend for GPU computation.
"""

from typing import Any, Optional, Tuple
from . import Backend

# Lazy import of CuPy
_cp = None


def _get_cupy():
    """Lazy import of CuPy."""
    global _cp
    if _cp is None:
        try:
            import cupy as cp
            _cp = cp
        except ImportError:
            raise RuntimeError(
                "CuPy is not installed. Install it with:\n"
                "  pip install cupy-cuda11x  # for CUDA 11.x\n"
                "  pip install cupy-cuda12x  # for CUDA 12.x"
            )
    return _cp


class CuPyBackend(Backend):
    """CuPy-based GPU backend."""

    name = "cupy"

    def __init__(self, device_index: int = 0):
        self._device_index = device_index
        self._cp = _get_cupy()

    @property
    def cp(self):
        """Get CuPy module."""
        return self._cp

    def array(self, data: Any, dtype: Any = None) -> Any:
        """Create an array on GPU."""
        if dtype is None:
            import numpy as np
            dtype = np.float32
        return self._cp.array(data, dtype=dtype)

    def asarray(self, data: Any, dtype: Any = None) -> Any:
        """Convert to array on GPU."""
        return self._cp.asarray(data, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create zeros array on GPU."""
        if dtype is None:
            import numpy as np
            dtype = np.float32
        return self._cp.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create ones array on GPU."""
        if dtype is None:
            import numpy as np
            dtype = np.float32
        return self._cp.ones(shape, dtype=dtype)

    def empty(self, shape: Tuple[int, ...], dtype: Any = None) -> Any:
        """Create empty array on GPU."""
        if dtype is None:
            import numpy as np
            dtype = np.float32
        return self._cp.empty(shape, dtype=dtype)

    def arange(self, start: int, stop: Optional[int] = None, step: int = 1, dtype: Any = None) -> Any:
        """Create range array on GPU."""
        return self._cp.arange(start, stop, step, dtype=dtype)

    def linspace(self, start: float, stop: float, num: int = 50, dtype: Any = None) -> Any:
        """Create linearly spaced array on GPU."""
        if dtype is None:
            import numpy as np
            dtype = np.float32
        return self._cp.linspace(start, stop, num, dtype=dtype)

    def eye(self, N: int, M: Optional[int] = None, dtype: Any = None) -> Any:
        """Create identity matrix on GPU."""
        if dtype is None:
            import numpy as np
            dtype = np.float32
        return self._cp.eye(N, M, dtype=dtype)

    def random(self) -> Any:
        """Return random module."""
        return self._cp.random

    def linalg(self) -> Any:
        """Return linalg module."""
        return self._cp.linalg

    def get_array_module(self, array: Any) -> Any:
        """Get the array module (cupy) for the given array."""
        return self._cp.get_array_module(array)

    def to_cpu(self, data: Any) -> Any:
        """Move data to CPU (NumPy array)."""
        if hasattr(data, 'get'):
            return data.get()
        return data

    def to_gpu(self, data: Any, device: int = 0) -> Any:
        """Data is already on GPU or move to GPU."""
        if self.is_on_gpu(data):
            return data
        return self._cp.asarray(data)

    def is_on_gpu(self, data: Any) -> bool:
        """Check if data is on GPU."""
        return isinstance(data, self._cp.ndarray)

    # Expose cupy functions directly
    @property
    def sqrt(self):
        return self._cp.sqrt

    @property
    def exp(self):
        return self._cp.exp

    @property
    def log(self):
        return self._cp.log

    @property
    def sin(self):
        return self._cp.sin

    @property
    def cos(self):
        return self._cp.cos

    @property
    def tanh(self):
        return self._cp.tanh

    @property
    def maximum(self):
        return self._cp.maximum

    @property
    def minimum(self):
        return self._cp.minimum

    @property
    def where(self):
        return self._cp.where

    @property
    def clip(self):
        return self._cp.clip

    @property
    def concatenate(self):
        return self._cp.concatenate

    @property
    def stack(self):
        return self._cp.stack

    @property
    def split(self):
        return self._cp.split

    @property
    def squeeze(self):
        return self._cp.squeeze

    @property
    def expand_dims(self):
        return self._cp.expand_dims

    @property
    def transpose(self):
        return self._cp.transpose

    @property
    def reshape(self):
        return self._cp.reshape

    @property
    def flip(self):
        return self._cp.flip

    @property
    def roll(self):
        return self._cp.roll

    @property
    def pad(self):
        return self._cp.pad

    @property
    def einsum(self):
        return self._cp.einsum

    @property
    def tensordot(self):
        return self._cp.tensordot

    @property
    def dot(self):
        return self._cp.dot

    @property
    def matmul(self):
        return self._cp.matmul

    @property
    def sliding_window_view(self):
        # CuPy doesn't have sliding_window_view, use custom implementation
        return self._sliding_window_view

    def _sliding_window_view(self, x, window_shape, axis=None):
        """CuPy implementation of sliding_window_view."""
        import numpy as np
        # Fall back to numpy for now, then move to GPU
        if hasattr(x, 'get'):
            x_np = x.get()
        else:
            x_np = np.asarray(x)
        result = np.lib.stride_tricks.sliding_window_view(x_np, window_shape, axis)
        return self._cp.asarray(result)

    @property
    def meshgrid(self):
        return self._cp.meshgrid

    @property
    def isnan(self):
        return self._cp.isnan

    @property
    def isinf(self):
        return self._cp.isinf

    @property
    def inf(self):
        return self._cp.inf

    @property
    def newaxis(self):
        return self._cp.newaxis

    @property
    def pi(self):
        return self._cp.pi

    @property
    def e(self):
        return self._cp.e
