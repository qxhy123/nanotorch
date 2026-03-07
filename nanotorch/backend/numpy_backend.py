"""
NumPy backend for CPU computation.
"""

import numpy as np
from typing import Any, Optional, Tuple
from . import Backend


class NumPyBackend(Backend):
    """NumPy-based CPU backend."""

    name = "numpy"

    def __init__(self):
        self._np = np

    def array(self, data: Any, dtype: Any = np.float32) -> np.ndarray:
        """Create an array."""
        return self._np.array(data, dtype=dtype)

    def asarray(self, data: Any, dtype: Any = None) -> np.ndarray:
        """Convert to array."""
        return self._np.asarray(data, dtype=dtype)

    def zeros(self, shape: Tuple[int, ...], dtype: Any = np.float32) -> np.ndarray:
        """Create zeros array."""
        return self._np.zeros(shape, dtype=dtype)

    def ones(self, shape: Tuple[int, ...], dtype: Any = np.float32) -> np.ndarray:
        """Create ones array."""
        return self._np.ones(shape, dtype=dtype)

    def empty(self, shape: Tuple[int, ...], dtype: Any = np.float32) -> np.ndarray:
        """Create empty array."""
        return self._np.empty(shape, dtype=dtype)

    def arange(self, start: int, stop: Optional[int] = None, step: int = 1, dtype: Any = None) -> np.ndarray:
        """Create range array."""
        return self._np.arange(start, stop, step, dtype=dtype)

    def linspace(self, start: float, stop: float, num: int = 50, dtype: Any = np.float32) -> np.ndarray:
        """Create linearly spaced array."""
        return self._np.linspace(start, stop, num, dtype=dtype)

    def eye(self, N: int, M: Optional[int] = None, dtype: Any = np.float32) -> np.ndarray:
        """Create identity matrix."""
        return self._np.eye(N, M, dtype=dtype)

    def random(self) -> np.random:
        """Return random module."""
        return self._np.random

    def linalg(self) -> np.linalg:
        """Return linalg module."""
        return self._np.linalg

    def get_array_module(self, array: np.ndarray) -> Any:
        """Get the array module (numpy) for the given array."""
        return self._np

    def to_cpu(self, data: np.ndarray) -> np.ndarray:
        """Data is already on CPU."""
        return data

    def to_gpu(self, data: np.ndarray, device: int = 0) -> Any:
        """Move data to GPU (requires CuPy)."""
        try:
            import cupy as cp
            return cp.asarray(data)
        except ImportError:
            raise RuntimeError("CuPy is not installed. Cannot move data to GPU.")

    def is_on_gpu(self, data: Any) -> bool:
        """CPU data is never on GPU."""
        return False

    # Expose numpy functions directly
    @property
    def sqrt(self):
        return self._np.sqrt

    @property
    def exp(self):
        return self._np.exp

    @property
    def log(self):
        return self._np.log

    @property
    def sin(self):
        return self._np.sin

    @property
    def cos(self):
        return self._np.cos

    @property
    def tanh(self):
        return self._np.tanh

    @property
    def maximum(self):
        return self._np.maximum

    @property
    def minimum(self):
        return self._np.minimum

    @property
    def where(self):
        return self._np.where

    @property
    def clip(self):
        return self._np.clip

    @property
    def concatenate(self):
        return self._np.concatenate

    @property
    def stack(self):
        return self._np.stack

    @property
    def split(self):
        return self._np.split

    @property
    def squeeze(self):
        return self._np.squeeze

    @property
    def expand_dims(self):
        return self._np.expand_dims

    @property
    def transpose(self):
        return self._np.transpose

    @property
    def reshape(self):
        return self._np.reshape

    @property
    def flip(self):
        return self._np.flip

    @property
    def roll(self):
        return self._np.roll

    @property
    def pad(self):
        return self._np.pad

    @property
    def einsum(self):
        return self._np.einsum

    @property
    def tensordot(self):
        return self._np.tensordot

    @property
    def dot(self):
        return self._np.dot

    @property
    def matmul(self):
        return self._np.matmul

    @property
    def sliding_window_view(self):
        return self._np.lib.stride_tricks.sliding_window_view

    @property
    def meshgrid(self):
        return self._np.meshgrid

    @property
    def isnan(self):
        return self._np.isnan

    @property
    def isinf(self):
        return self._np.isinf

    @property
    def inf(self):
        return self._np.inf

    @property
    def newaxis(self):
        return self._np.newaxis

    @property
    def pi(self):
        return self._np.pi

    @property
    def e(self):
        return self._np.e
