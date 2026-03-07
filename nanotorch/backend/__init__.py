"""
Backend module for nanotorch - provides CPU (NumPy) and GPU (CuPy) backends.
"""

from typing import Any, Optional, Tuple, Union
from abc import ABC, abstractmethod
import numpy as np

# Global backend instance
_current_backend: Optional["Backend"] = None


class Backend(ABC):
    """Abstract base class for computation backends."""

    name: str = "base"

    @abstractmethod
    def array(self, data: Any, dtype: Any = np.float32) -> Any:
        """Create an array."""
        pass

    @abstractmethod
    def asarray(self, data: Any, dtype: Any = None) -> Any:
        """Convert to array."""
        pass

    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype: Any = np.float32) -> Any:
        """Create zeros array."""
        pass

    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype: Any = np.float32) -> Any:
        """Create ones array."""
        pass

    @abstractmethod
    def empty(self, shape: Tuple[int, ...], dtype: Any = np.float32) -> Any:
        """Create empty array."""
        pass

    @abstractmethod
    def arange(self, start: int, stop: Optional[int] = None, step: int = 1, dtype: Any = None) -> Any:
        """Create range array."""
        pass

    @abstractmethod
    def linspace(self, start: float, stop: float, num: int = 50, dtype: Any = np.float32) -> Any:
        """Create linearly spaced array."""
        pass

    @abstractmethod
    def eye(self, N: int, M: Optional[int] = None, dtype: Any = np.float32) -> Any:
        """Create identity matrix."""
        pass

    @abstractmethod
    def random(self) -> Any:
        """Return random module."""
        pass

    @abstractmethod
    def linalg(self) -> Any:
        """Return linalg module."""
        pass

    @abstractmethod
    def get_array_module(self, array: Any) -> Any:
        """Get the array module (numpy or cupy) for the given array."""
        pass

    @abstractmethod
    def to_cpu(self, data: Any) -> Any:
        """Move data to CPU."""
        pass

    @abstractmethod
    def to_gpu(self, data: Any, device: int = 0) -> Any:
        """Move data to GPU."""
        pass

    @abstractmethod
    def is_on_gpu(self, data: Any) -> bool:
        """Check if data is on GPU."""
        pass


def get_backend() -> Backend:
    """Get the current backend."""
    global _current_backend
    if _current_backend is None:
        from .numpy_backend import NumPyBackend
        _current_backend = NumPyBackend()
    return _current_backend


def set_backend(backend: str, device_index: int = 0) -> Backend:
    """Set the current backend.

    Args:
        backend: 'cpu' or 'cuda'
        device_index: GPU device index (for multi-GPU)
    """
    global _current_backend
    if backend == 'cpu':
        from .numpy_backend import NumPyBackend
        _current_backend = NumPyBackend()
    elif backend == 'cuda':
        from .cupy_backend import CuPyBackend
        _current_backend = CuPyBackend(device_index)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    return _current_backend


def get_backend_for_device(device: str, device_index: int = 0) -> Backend:
    """Get backend for a specific device."""
    if device == 'cpu':
        from .numpy_backend import NumPyBackend
        return NumPyBackend()
    elif device == 'cuda':
        from .cupy_backend import CuPyBackend
        return CuPyBackend(device_index)
    else:
        raise ValueError(f"Unknown device: {device}")
