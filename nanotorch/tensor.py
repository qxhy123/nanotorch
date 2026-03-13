"""
Tensor class for nanotorch - Core tensor operations with autograd support.

This module implements the core Tensor class with automatic differentiation
capabilities, similar to PyTorch's Tensor API.
"""

import numpy as np
from numpy.typing import NDArray
from types import TracebackType
from typing import Optional, Tuple, List, Union, Any, cast, Type

# Import device module
from nanotorch.device import Device, cpu as cpu_device


class Tensor:
    """A multi-dimensional array with automatic differentiation support.

    The Tensor class stores data, gradients, and computational graph information
    to enable automatic differentiation (autograd).

    Attributes:
        data: The underlying numpy/cupy array storing tensor values.
        requires_grad: Whether to track gradients for this tensor.
        grad: Gradient tensor (same shape as data).
        _parents: Parent tensors in the computational graph.
        _device: Device where the tensor is stored (cpu or cuda).
    """

    __slots__ = ["data", "requires_grad", "grad", "_parents", "_ctx", "_fn", "_device"]
    data: NDArray[np.float32]

    def __init__(
        self,
        data: Union[NDArray[Any], List[Any], int, float],
        requires_grad: bool = False,
        _parents: Tuple["Tensor", ...] = (),
        _ctx: Optional[Any] = None,
        _fn: Optional[Any] = None,
        device: Optional[Union[Device, str]] = None,
    ) -> None:
        """Initialize a Tensor.

        Args:
            data: Input data, can be numpy array, list, or scalar.
            requires_grad: Whether to compute gradients for this tensor.
            _parents: Internal use - parent tensors in computational graph.
            device: Device to place tensor on ('cpu', 'cuda', or Device instance).
        """
        # Handle device
        if device is None:
            # Infer device from data or default to CPU
            if hasattr(data, 'device') and hasattr(data.device, 'type'):
                # CuPy array
                self._device = Device('cuda', getattr(data.device, 'id', 0))
            else:
                self._device = cpu_device
        elif isinstance(device, str):
            self._device = Device.from_string(device)
        elif isinstance(device, Device):
            self._device = device
        else:
            raise TypeError(f"device must be str or Device, got {type(device)}")

        # Convert input to array with float32 dtype
        if isinstance(data, (int, float)):
            arr = np.array(data, dtype=np.float32)
        elif isinstance(data, list):
            arr = np.array(data, dtype=np.float32)
        else:
            arr = data.astype(np.float32) if data.dtype != np.float32 else data

        # Move to appropriate device
        if self._device.is_cuda:
            try:
                import cupy as cp
                self.data = cp.asarray(arr)
            except ImportError:
                raise RuntimeError(
                    "CuPy is not installed. Cannot create CUDA tensor. "
                    "Install with: pip install cupy-cuda11x or pip install cupy-cuda12x"
                )
        else:
            self.data = arr

        # Disable gradient tracking if no_grad context is active
        if not _ENABLE_GRAD:
            requires_grad = False

        self.requires_grad = requires_grad
        self.grad: Optional["Tensor"] = None
        self._parents = _parents
        self._ctx = _ctx
        self._fn = _fn

        # Initialize gradient if requires_grad is True
        if requires_grad:
            self.zero_grad()

    @property
    def device(self) -> Device:
        """Return the device where the tensor is stored."""
        return self._device

    @property
    def is_cuda(self) -> bool:
        """Check if tensor is on CUDA device."""
        return self._device.is_cuda

    def to(self, device: Union[Device, str]) -> "Tensor":
        """Move tensor to the specified device.

        Args:
            device: Target device ('cpu', 'cuda', 'cuda:0', or Device instance).

        Returns:
            New tensor on the target device.
        """
        if isinstance(device, str):
            target_device = Device.from_string(device)
        else:
            target_device = device

        # If already on target device, return self
        if self._device == target_device:
            return self

        # Get the data on CPU first
        if hasattr(self.data, 'get'):
            # CuPy array
            cpu_data = self.data.get()
        else:
            cpu_data = self.data

        # Create new tensor on target device
        return Tensor(
            cpu_data,
            requires_grad=self.requires_grad,
            _parents=self._parents,
            _ctx=self._ctx,
            _fn=self._fn,
            device=target_device,
        )

    def cuda(self, device: Union[int, str] = 0) -> "Tensor":
        """Move tensor to CUDA device.

        Args:
            device: CUDA device index or string like 'cuda:0'.

        Returns:
            Tensor on CUDA device.
        """
        if isinstance(device, int):
            target = Device('cuda', device)
        else:
            target = Device.from_string(device)
        return self.to(target)

    def cpu(self) -> "Tensor":
        """Move tensor to CPU.

        Returns:
            Tensor on CPU.
        """
        return self.to('cpu')

    def numpy(self) -> np.ndarray:
        """Return tensor data as NumPy array (on CPU).

        Returns:
            NumPy array copy of the data.
        """
        if hasattr(self.data, 'get'):
            return self.data.get()
        return np.asarray(self.data)

    def _get_xp(self) -> Any:
        """Get the array module (numpy or cupy) for this tensor's device.

        Returns:
            numpy or cupy module.
        """
        if self._device.is_cuda:
            try:
                import cupy as cp
                return cp
            except ImportError:
                return np
        return np

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the tensor shape."""
        return self.data.shape

    @property
    def size(self) -> int:
        """Return total number of elements in the tensor."""
        return self.data.size

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype[np.float32]:
        """Return the data type."""
        return self.data.dtype

    def __repr__(self) -> str:
        """String representation of the tensor."""
        grad_str = f", grad={self.grad}" if self.grad is not None else ""
        device_str = f", device='{self._device}'" if self._device.is_cuda else ""
        return (
            f"Tensor(shape={self.shape}, requires_grad={self.requires_grad}{grad_str}{device_str})"
        )

    def __getitem__(self, key) -> "Tensor":
        """Index into the tensor.

        Args:
            key: Index or slice.

        Returns:
            Indexed tensor (view, not copy).
        """
        from nanotorch.autograd import getitem_tensor as autograd_getitem

        return autograd_getitem(self, key)

    @staticmethod
    def _reduce_gradient_numpy(
        grad_contrib_data: NDArray[np.float32], target_shape: Tuple[int, ...]
    ) -> NDArray[np.float32]:
        """Compatibility proxy to the autograd gradient reducer."""
        from nanotorch.autograd import reduce_gradient_numpy

        return cast(NDArray[np.float32], reduce_gradient_numpy(grad_contrib_data, target_shape))

    @staticmethod
    def _accumulate_grad(
        parent: "Tensor", grad_contrib: Union["Tensor", NDArray[Any]]
    ) -> None:
        """Compatibility proxy to autograd gradient accumulation."""
        from nanotorch.autograd import accumulate_grad

        accumulate_grad(parent, grad_contrib)

    @staticmethod
    def _accumulate_grad_batch(
        accumulations: List[Tuple["Tensor", Union["Tensor", NDArray[Any]]]]
    ) -> None:
        """Compatibility proxy to batched autograd gradient accumulation."""
        from nanotorch.autograd import accumulate_grad_batch

        accumulate_grad_batch(accumulations)

    def zero_grad(self) -> None:
        """Reset gradient to zero."""
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)
        else:
            self.grad.data.fill(0)

    def backward(self, gradient: Optional["Tensor"] = None) -> None:
        """Perform backward pass through the computational graph.

        Args:
            gradient: Gradient to propagate (default is ones tensor).
        """
        from nanotorch.autograd import backward as autograd_backward

        autograd_backward(self, gradient)

    def __add__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Element-wise addition."""
        from nanotorch.autograd import add as autograd_add

        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        try:
            self.data + other.data
        except ValueError as e:
            raise RuntimeError(
                f"Shape mismatch in addition: {self.shape} vs {other.shape}"
            ) from e

        return autograd_add(self, other)

    def __radd__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Reverse addition."""
        return self.__add__(other)

    def __sub__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Element-wise subtraction."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        # Negate and add
        return self + (-other)

    def __rsub__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Reverse subtraction."""
        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other - self

    def __neg__(self) -> "Tensor":
        """Element-wise negation."""
        from nanotorch.autograd import neg as autograd_neg

        return autograd_neg(self)

    def __mul__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Element-wise multiplication."""
        from nanotorch.autograd import mul as autograd_mul

        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        try:
            self.data * other.data
        except ValueError as e:
            raise RuntimeError(
                f"Shape mismatch in multiplication: {self.shape} vs {other.shape}"
            ) from e

        return autograd_mul(self, other)

    def __rmul__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Reverse multiplication."""
        return self.__mul__(other)

    def __truediv__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Element-wise division."""
        from nanotorch.autograd import div as autograd_div

        if isinstance(other, (int, float)) and other == 0:
            raise ValueError("Cannot divide by zero")

        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        if np.any(other.data == 0):
            raise ValueError("Cannot divide by tensor containing zero values")

        try:
            self.data / other.data
        except ValueError as e:
            raise RuntimeError(
                f"Shape mismatch in division: {self.shape} vs {other.shape}"
            ) from e

        return autograd_div(self, other)

    def __rtruediv__(
        self, other: Union["Tensor", NDArray[np.float32], int, float]
    ) -> "Tensor":
        """Reverse division."""
        # Check for division by zero in self
        if np.any(self.data == 0):
            raise ValueError("Cannot divide by tensor containing zero values")

        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)
        return other / self

    def __pow__(self, exponent: Union[int, float]) -> "Tensor":
        """Element-wise power."""
        from nanotorch.autograd import pow as autograd_pow

        return autograd_pow(self, exponent)

    def __matmul__(self, other: Union["Tensor", NDArray[np.float32]]) -> "Tensor":
        """Matrix multiplication."""
        from nanotorch.autograd import matmul as autograd_matmul

        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        if self.ndim != 2 or other.ndim != 2:
            raise RuntimeError("matmul requires 2D tensors")

        if self.shape[1] != other.shape[0]:
            raise RuntimeError(
                f"Shape mismatch in matmul: {self.shape} vs {other.shape}. "
                f"Expected self.shape[1] == other.shape[0]"
            )

        return autograd_matmul(self, other)

    def matmul(self, other: Union["Tensor", NDArray[np.float32]]) -> "Tensor":
        """General matrix multiplication supporting batched inputs.

        For 2D inputs: standard matrix multiplication
        For ND inputs: batched matrix multiplication over last two dimensions

        Args:
            other: Tensor to multiply with.

        Returns:
            Result tensor.
        """
        from nanotorch.autograd import batch_matmul as autograd_batch_matmul

        if not isinstance(other, Tensor):
            other = Tensor(other, requires_grad=False)

        if self.ndim < 1 or other.ndim < 1:
            raise RuntimeError("matmul requires at least 1D tensors")

        if self.shape[-1] != other.shape[-2 if other.ndim > 1 else 0]:
            raise RuntimeError(
                f"Shape mismatch in matmul: {self.shape} vs {other.shape}"
            )

        return autograd_batch_matmul(self, other)

    @property
    def T(self) -> "Tensor":
        """Transpose the tensor by reversing dimension order."""
        return self.permute(*range(self.ndim - 1, -1, -1))

    # Activation functions
    def relu(self) -> "Tensor":
        """Rectified Linear Unit activation."""
        from nanotorch.autograd import relu_tensor as autograd_relu

        return autograd_relu(self)

    def sigmoid(self) -> "Tensor":
        """Sigmoid activation."""
        from nanotorch.autograd import sigmoid_tensor as autograd_sigmoid

        return autograd_sigmoid(self)

    def tanh(self) -> "Tensor":
        """Hyperbolic tangent activation."""
        from nanotorch.autograd import tanh_tensor as autograd_tanh

        return autograd_tanh(self)
    
    def gelu(self) -> "Tensor":
        """Gaussian Error Linear Unit activation (fast approximation).
        
        Uses the fast approximation: gelu(x) ≈ x * sigmoid(1.702 * x).
        
        Returns:
            Tensor with GELU activation applied element-wise.
        """
        return self * (self * 1.702).sigmoid()
    
    def swish(self, beta: float = 1.0) -> "Tensor":
        """Swish activation (also called SiLU).
        
        Args:
            beta: Scaling factor for sigmoid. Default 1.0 (standard Swish).
        
        Returns:
            Tensor with Swish activation applied element-wise.
        """
        # Swish: x * sigmoid(beta * x)
        return self * (self * beta).sigmoid()
    
    def leaky_relu(self, negative_slope: float = 0.01) -> "Tensor":
        """Leaky ReLU activation.
        
        Args:
            negative_slope: Slope for negative values. Default 0.01.
        
        Returns:
            Tensor with LeakyReLU activation applied element-wise.
        
        Formula: leaky_relu(x) = max(0, x) + negative_slope * min(0, x)
        """
        return self.relu() + (self - self.relu()) * negative_slope
    
    def elu(self, alpha: float = 1.0) -> "Tensor":
        """Exponential Linear Unit activation.
        
        Args:
            alpha: Scaling factor for negative values. Default 1.0.
        
        Returns:
            Tensor with ELU activation applied element-wise.
        
        Formula: elu(x) = x if x > 0 else alpha * (exp(x) - 1)
        """
        # elu(x) = relu(x) + alpha * (exp(min(x, 0)) - 1)
        return self.relu() + alpha * ((self - self.relu()).exp() - 1)
    
    def softplus(self) -> "Tensor":
        """Softplus activation.
        
        Formula: softplus(x) = log(1 + exp(x))
        
        Returns:
            Tensor with Softplus activation applied element-wise.
        """
        # softplus(x) = log(1 + exp(x))
        # For numerical stability: log(1 + exp(x)) = max(x, 0) + log(1 + exp(-|x|))
        return (self.relu()) + ((-self.abs()).exp() + 1).log()
    
    def hardswish(self) -> "Tensor":
        """Hard Swish activation.
        
        Formula: hardswish(x) = x * relu6(x + 3) / 6
        
        Returns:
            Tensor with HardSwish activation applied element-wise.
        """
        # relu6(x) = min(max(0, x), 6)
        relu6 = (self + 3).relu()
        relu6 = relu6.clamp(0, 6)
        return self * relu6 * (1.0 / 6.0)
    
    def hardsigmoid(self) -> "Tensor":
        """Hard Sigmoid activation.
        
        Formula: hardsigmoid(x) = relu6(x + 3) / 6
        
        Returns:
            Tensor with HardSigmoid activation applied element-wise.
        """
        relu6 = (self + 3).relu()
        relu6 = relu6.clamp(0, 6)
        return relu6 * (1.0 / 6.0)
    
    def silu(self) -> "Tensor":
        """SiLU (Sigmoid Linear Unit) activation, equivalent to Swish with beta=1.
        
        Returns:
            Tensor with SiLU activation applied element-wise.
        """
        return self.swish(beta=1.0)
    
    def prelu(self, weight: "Tensor") -> "Tensor":
        """Parametric ReLU activation.
        
        Args:
            weight: Learnable parameter with same shape as input or broadcastable.
        
        Returns:
            Tensor with PReLU activation applied element-wise.
        
        Formula: prelu(x) = max(0, x) + weight * min(0, x)
        """
        return self.relu() + (self - self.relu()) * weight

    # Shape operations
    def reshape(self, new_shape: Tuple[int, ...]) -> "Tensor":
        """Reshape the tensor.

        Supports -1 in new_shape to infer the dimension size.
        """
        from nanotorch.autograd import reshape_tensor as autograd_reshape

        new_shape_list = list(new_shape)
        if -1 in new_shape_list:
            known_size = np.prod([s for s in new_shape_list if s != -1])
            total_size = np.prod(self.shape)
            inferred_size = total_size // known_size
            idx = new_shape_list.index(-1)
            new_shape_list[idx] = int(inferred_size)
            new_shape = tuple(new_shape_list)

        if np.prod(new_shape) != np.prod(self.shape):
            raise ValueError(
                f"Cannot reshape tensor of shape {self.shape} to {new_shape}. "
                f"Total elements must match."
            )

        return autograd_reshape(self, new_shape)

    def squeeze(self, axis: Union[int, Tuple[int, ...], None] = None) -> "Tensor":
        """Remove dimensions of size 1.

        Args:
            axis: Selects a subset of the dimensions of size 1 to remove.
                If None, all dimensions of size 1 are removed.

        Returns:
            Squeezed tensor.
        """
        from nanotorch.autograd import squeeze_tensor as autograd_squeeze

        return autograd_squeeze(self, axis)

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        """Transpose two dimensions of the tensor.

        Args:
            dim0: First dimension to transpose.
            dim1: Second dimension to transpose.

        Returns:
            Transposed tensor.
        """
        if dim0 < -self.ndim or dim0 >= self.ndim:
            raise ValueError(f"dim0 out of range: {dim0}")
        if dim1 < -self.ndim or dim1 >= self.ndim:
            raise ValueError(f"dim1 out of range: {dim1}")

        perm = list(range(self.ndim))
        dim0_pos = dim0 if dim0 >= 0 else self.ndim + dim0
        dim1_pos = dim1 if dim1 >= 0 else self.ndim + dim1
        perm[dim0_pos], perm[dim1_pos] = perm[dim1_pos], perm[dim0_pos]

        return self.permute(*perm)

    def permute(self, *dims: int) -> "Tensor":
        """Permute the dimensions of the tensor.

        Args:
            *dims: Desired ordering of dimensions.

        Returns:
            Permuted tensor.

        Raises:
            ValueError: If dimensions are invalid or contain duplicates.
        """
        from nanotorch.autograd import permute_tensor as autograd_permute

        if len(dims) != self.ndim:
            raise ValueError(
                f"Number of dimensions must match tensor ndim. "
                f"Expected {self.ndim}, got {len(dims)}"
            )

        seen = set()
        for d in dims:
            if d < -self.ndim or d >= self.ndim:
                raise ValueError(
                    f"Dimension out of range (expected to be in range "
                    f"[-{self.ndim}, {self.ndim-1}]), got {d}"
                )
            pos_d = d if d >= 0 else self.ndim + d
            if pos_d in seen:
                raise ValueError(f"Repeated dimension: {d}")
            seen.add(pos_d)

        pos_dims = tuple(d if d >= 0 else self.ndim + d for d in dims)
        return autograd_permute(self, *pos_dims)

    def view(self, *shape: int) -> "Tensor":
        """Return a tensor with same data but different shape.
        
        For now, this is an alias for reshape since NumPy doesn't distinguish
        between views and copies in the same way PyTorch does.
        
        Args:
            *shape: Desired shape.
            
        Returns:
            Tensor with new shape.
        """
        # Convert to tuple
        shape_tuple = tuple(shape)
        return self.reshape(shape_tuple)

    def gather(self, dim: int, index: "Tensor") -> "Tensor":
        """Gather values along an axis using indices.

        Args:
            dim: Dimension to gather along.
            index: Tensor containing indices to gather.

        Returns:
            Tensor with same shape as index.

        Raises:
            ValueError: If dimensions are invalid.
            RuntimeError: If index shape is incompatible.
        """
        from nanotorch.autograd import gather_tensor as autograd_gather

        if dim < 0:
            dim = self.ndim + dim

        if dim < 0 or dim >= self.ndim:
            raise ValueError(
                f"Dimension out of range. Expected 0 <= dim < {self.ndim}, got {dim}"
            )

        if index.ndim != self.ndim:
            raise RuntimeError(
                f"Index tensor must have same number of dimensions as input. "
                f"Expected {self.ndim}, got {index.ndim}"
            )

        for i in range(self.ndim):
            if i != dim and index.shape[i] != self.shape[i]:
                raise RuntimeError(
                    f"Index shape mismatch at dimension {i}. "
                    f"Expected {self.shape[i]}, got {index.shape[i]}"
                )

        return autograd_gather(self, dim, index)

    def scatter(self, dim: int, index: "Tensor", src: "Tensor") -> "Tensor":
        """Scatter values from src into self at positions specified by index along dim.
        
        Args:
            dim: Dimension along which to index.
            index: Tensor containing indices where to scatter.
            src: Tensor containing values to scatter.
            
        Returns:
            Tensor with same shape as self.
            
        Raises:
            ValueError: If dimensions are invalid.
            RuntimeError: If shape mismatches.
        """
        from nanotorch.autograd import scatter_tensor as autograd_scatter

        if dim < 0:
            dim = self.ndim + dim

        if dim < 0 or dim >= self.ndim:
            raise ValueError(
                f"Dimension out of range. Expected 0 <= dim < {self.ndim}, got {dim}"
            )

        if index.ndim != self.ndim or src.ndim != self.ndim:
            raise RuntimeError(
                f"Index and src must have same number of dimensions as self. "
                f"Expected {self.ndim}, got index.ndim={index.ndim}, src.ndim={src.ndim}"
            )

        if index.shape != src.shape:
            raise RuntimeError(
                f"Index and src must have same shape. "
                f"Got index.shape={index.shape}, src.shape={src.shape}"
            )

        for i in range(self.ndim):
            if i != dim and index.shape[i] != self.shape[i]:
                raise RuntimeError(
                    f"Index shape mismatch at dimension {i}. "
                    f"Expected {self.shape[i]}, got {index.shape[i]}"
                )

        max_index = self.shape[dim]
        if np.any(index.data < 0) or np.any(index.data >= max_index):
            raise ValueError(f"Index values must be in range [0, {max_index})")

        return autograd_scatter(self, dim, index, src)

    def sum(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> "Tensor":
        """Sum tensor elements along the specified axis(es)."""
        from nanotorch.autograd import sum_tensor as autograd_sum

        return autograd_sum(self, axis, keepdims)

    def mean(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> "Tensor":
        """Mean of tensor elements along the specified axis(es)."""
        from nanotorch.autograd import mean_tensor as autograd_mean

        return autograd_mean(self, axis, keepdims)

    def prod(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> "Tensor":
        """Product of tensor elements along the specified axis(es)."""
        from nanotorch.autograd import prod_tensor as autograd_prod

        return autograd_prod(self, axis, keepdims)

    def var(
        self,
        axis: Union[int, Tuple[int, ...], None] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> "Tensor":
        """Variance of tensor elements along the specified axis(es).

        Args:
            axis: Axis or axes along which to compute variance.
            keepdims: Whether to keep reduced dimensions.
            ddof: Delta degrees of freedom. The divisor used is N - ddof,
                  where N is the number of elements. Default is 0.

        Returns:
            Tensor with variance values.
        """
        # Compute mean along axis (keep dimensions for broadcasting)
        mean = self.mean(axis=axis, keepdims=True)
        # Compute squared differences
        squared_diff = (self - mean) ** 2
        # Compute mean of squared differences (variance with ddof=0)
        var = squared_diff.mean(axis=axis, keepdims=keepdims)

        # Adjust for ddof (degrees of freedom)
        if ddof != 0:
            # Compute number of elements along reduced axes
            if axis is None:
                n = self.data.size
            else:
                if isinstance(axis, int):
                    axis = (axis,)
                n = 1
                for ax in axis:
                    if ax < 0:
                        ax = self.ndim + ax
                    n *= self.shape[ax]
            # Scale variance by n/(n-ddof)
            scale = n / (n - ddof)
            var = var * scale

        return var

    def std(
        self,
        axis: Union[int, Tuple[int, ...], None] = None,
        keepdims: bool = False,
        ddof: int = 0,
    ) -> "Tensor":
        """Standard deviation of tensor elements along the specified axis(es).

        Args:
            axis: Axis or axes along which to compute standard deviation.
            keepdims: Whether to keep reduced dimensions.
            ddof: Delta degrees of freedom. The divisor used is N - ddof,
                  where N is the number of elements. Default is 0.

        Returns:
            Tensor with standard deviation values.
        """
        # Standard deviation is square root of variance
        var = self.var(axis=axis, keepdims=True, ddof=ddof)
        std = var.sqrt()

        # If keepdims is False and axis is not None, we need to squeeze the dimensions
        # that were kept for broadcasting but should be removed
        if not keepdims and axis is not None:
            # squeeze only the dimensions that were reduced
            if isinstance(axis, int):
                axis = (axis,)
            # Convert negative axis to positive
            axis = tuple(ax if ax >= 0 else self.ndim + ax for ax in axis)
            # Sort axis to squeeze from highest to lowest
            for ax in sorted(axis, reverse=True):
                std = std.squeeze(axis=ax)

        return std

    def max(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> "Tensor":
        """Maximum of tensor elements along the specified axis(es)."""
        from nanotorch.autograd import max_tensor as autograd_max

        return autograd_max(self, axis, keepdims)

    def min(
        self, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False
    ) -> "Tensor":
        """Minimum of tensor elements along the specified axis(es)."""
        from nanotorch.autograd import min_tensor as autograd_min

        return autograd_min(self, axis, keepdims)

    def exp(self) -> "Tensor":
        """Element-wise exponential."""
        from nanotorch.autograd import exp_tensor as autograd_exp

        return autograd_exp(self)

    def log(self) -> "Tensor":
        """Element-wise natural logarithm (matches PyTorch behavior).
        
        Returns:
            Tensor with natural logarithm of each element.
            log(0) = -inf, log(negative) = nan (matching NumPy behavior).
        """
        from nanotorch.autograd import log_tensor as autograd_log

        return autograd_log(self)

    def abs(self) -> "Tensor":
        """Element-wise absolute value."""
        from nanotorch.autograd import abs_tensor as autograd_abs

        return autograd_abs(self)

    def sqrt(self) -> "Tensor":
        """Element-wise square root."""
        from nanotorch.autograd import sqrt_tensor as autograd_sqrt

        return autograd_sqrt(self)

    def clamp(
        self, min_val: Optional[float] = None, max_val: Optional[float] = None
    ) -> "Tensor":
        """Clamp tensor values to [min_val, max_val] range.

        Args:
            min_val: Minimum value (if None, no lower bound).
            max_val: Maximum value (if None, no upper bound).

        Returns:
            Clamped tensor.
        """
        from nanotorch.autograd import clamp_tensor as autograd_clamp

        return autograd_clamp(self, min_val, max_val)

    def softmax(self, dim: int = -1) -> "Tensor":
        """Apply softmax along the specified dimension.

        Args:
            dim: Dimension along which softmax will be computed.

        Returns:
            Tensor with softmax applied along dimension.
        """
        from nanotorch.autograd import softmax_tensor as autograd_softmax

        if dim < 0:
            dim = self.ndim + dim

        return autograd_softmax(self, dim)

    def log_softmax(self, dim: int = -1) -> "Tensor":
        """Apply log-softmax along the specified dimension.

        Args:
            dim: Dimension along which log-softmax will be computed.

        Returns:
            Tensor with log-softmax applied along dimension.
        """
        from nanotorch.autograd import log_softmax_tensor as autograd_log_softmax

        if dim < 0:
            dim = self.ndim + dim

        return autograd_log_softmax(self, dim)

    # Utility methods
    def check_finite(self, check_nan: bool = True, check_inf: bool = True) -> bool:
        """Check if tensor contains NaN or Inf values.

        Args:
            check_nan: Whether to check for NaN values.
            check_inf: Whether to check for Inf values.

        Returns:
            True if tensor is finite (no NaN/Inf), False otherwise.
        """
        if check_nan and np.any(np.isnan(self.data)):
            return False
        if check_inf and np.any(np.isinf(self.data)):
            return False
        return True

    def assert_finite(self, msg: str = "Tensor contains NaN or Inf values") -> None:
        """Assert that tensor does not contain NaN or Inf values.

        Args:
            msg: Error message to raise if assertion fails.

        Raises:
            ValueError: If tensor contains NaN or Inf values.
        """
        if not self.check_finite():
            raise ValueError(msg)

    def item(self) -> float:
        """Convert scalar tensor to Python float."""
        if self.shape != ():
            raise ValueError(
                f"item() can only be called on scalar tensors, got shape {self.shape}"
            )
        return float(self.data)

    def detach(self) -> "Tensor":
        """Return a new tensor detached from the computational graph."""
        return Tensor(self.data, requires_grad=False, device=self._device)

    def clone(self) -> "Tensor":
        """Return a copy of the tensor."""
        if hasattr(self.data, 'get'):
            # CuPy array - copy and keep on GPU
            return Tensor(self.data.copy(), requires_grad=self.requires_grad, device=self._device)
        return Tensor(self.data.copy(), requires_grad=self.requires_grad, device=self._device)

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        """Flatten a contiguous range of dims into a single dimension.

        Args:
            start_dim: First dim to flatten (default: 0).
            end_dim: Last dim to flatten (default: -1, meaning up to last dim).

        Returns:
            Flattened tensor.
        """
        if end_dim < 0:
            end_dim = self.ndim + end_dim

        if start_dim == end_dim:
            return self

        new_shape = list(self.shape[:start_dim])
        flat_size = 1
        for i in range(start_dim, end_dim + 1):
            flat_size *= self.shape[i]
        new_shape.append(flat_size)
        new_shape.extend(self.shape[end_dim + 1:])

        return self.reshape(tuple(new_shape))

    def expand(self, *sizes: int) -> "Tensor":
        """Expand tensor to given sizes (broadcasting).

        Args:
            *sizes: Desired sizes for each dimension. Use -1 to keep original size.

        Returns:
            Expanded tensor (view).
        """
        sizes_list = list(sizes)
        new_shape = list(self.shape)
        
        # Pad with leading 1s if needed
        while len(new_shape) < len(sizes_list):
            new_shape.insert(0, 1)
        
        # Build expanded shape
        result_shape = []
        for i, (orig, target) in enumerate(zip(reversed(new_shape), reversed(sizes_list))):
            if target == -1:
                result_shape.insert(0, orig)
            elif orig == 1:
                result_shape.insert(0, target)
            elif orig == target:
                result_shape.insert(0, target)
            else:
                raise RuntimeError(
                    f"Cannot expand shape {self.shape} to size {sizes}"
                )
        
        # Use numpy broadcast_to
        result_data = np.broadcast_to(self.data, tuple(result_shape))
        return Tensor(result_data.copy(), requires_grad=self.requires_grad)

    def repeat(self, *repeats: int) -> "Tensor":
        """Repeat tensor along each dimension.

        Args:
            *repeats: Number of repetitions for each dimension.

        Returns:
            Repeated tensor.
        """
        if len(repeats) < self.ndim:
            repeats = (1,) * (self.ndim - len(repeats)) + repeats
        
        result_data = np.tile(self.data, repeats)
        return Tensor(result_data, requires_grad=self.requires_grad)

    def split(self, split_size: int, dim: int = 0) -> List["Tensor"]:
        """Split tensor into chunks of given size along dimension.

        Args:
            split_size: Size of each chunk.
            dim: Dimension along which to split.

        Returns:
            List of tensor chunks.
        """
        chunks = []
        size = self.shape[dim]
        for i in range(0, size, split_size):
            end = min(i + split_size, size)
            idx = [slice(None)] * self.ndim
            idx[dim] = slice(i, end)
            chunks.append(self[tuple(idx)])
        return chunks

    def chunk(self, chunks: int, dim: int = 0) -> List["Tensor"]:
        """Split tensor into given number of chunks along dimension.

        Args:
            chunks: Number of chunks to create.
            dim: Dimension along which to split.

        Returns:
            List of tensor chunks.
        """
        if chunks <= 0:
            raise ValueError(f"chunks must be greater than 0, got {chunks}")
        
        dim_size = self.shape[dim]
        chunk_size = (dim_size + chunks - 1) // chunks
        return self.split(chunk_size, dim)

    def topk(self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> Tuple["Tensor", "Tensor"]:
        """Return the k largest/smallest elements along a dimension.

        Args:
            k: Number of top elements to return.
            dim: Dimension to sort along.
            largest: If True, return largest. Otherwise, smallest.
            sorted: If True, return elements in sorted order.

        Returns:
            Tuple of (values, indices) tensors.
        """
        if dim < 0:
            dim = self.ndim + dim

        if largest:
            indices = np.argsort(-self.data, axis=dim)
        else:
            indices = np.argsort(self.data, axis=dim)

        # Take top k
        slices = [slice(None)] * self.ndim
        slices[dim] = slice(0, k)
        top_indices = indices[tuple(slices)]
        
        # Gather values
        values = np.take_along_axis(self.data, top_indices, axis=dim)

        return (
            Tensor(values, requires_grad=self.requires_grad),
            Tensor(top_indices.astype(np.int64), requires_grad=False)
        )

    def sort(self, dim: int = -1, descending: bool = False) -> Tuple["Tensor", "Tensor"]:
        """Sort tensor along a dimension.

        Args:
            dim: Dimension to sort along.
            descending: If True, sort in descending order.

        Returns:
            Tuple of (sorted_values, indices) tensors.
        """
        if dim < 0:
            dim = self.ndim + dim

        if descending:
            indices = np.argsort(-self.data, axis=dim)
        else:
            indices = np.argsort(self.data, axis=dim)

        sorted_data = np.take_along_axis(self.data, indices, axis=dim)

        return (
            Tensor(sorted_data, requires_grad=self.requires_grad),
            Tensor(indices.astype(np.int64), requires_grad=False)
        )

    def where(self, condition: "Tensor", y: "Tensor") -> "Tensor":
        """Select elements based on condition.

        Args:
            condition: Boolean tensor mask.
            y: Tensor for False values.

        Returns:
            Tensor with elements from self where condition is True, else from y.
        """
        result = np.where(condition.data.astype(bool), self.data, y.data)
        return Tensor(result, requires_grad=self.requires_grad or y.requires_grad)

    # Static factory methods
    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = False, device: Optional[Union[Device, str]] = None) -> "Tensor":
        """Create a tensor filled with zeros."""
        return Tensor(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad, device=device)

    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = False, device: Optional[Union[Device, str]] = None) -> "Tensor":
        """Create a tensor filled with ones."""
        return Tensor(np.ones(shape, dtype=np.float32), requires_grad=requires_grad, device=device)

    @staticmethod
    def ones_like(other: "Tensor", requires_grad: bool = False) -> "Tensor":
        """Create a tensor filled with ones with the same shape as another tensor."""
        xp = np if not hasattr(other.data, 'get') else other.data.__class__
        return Tensor(xp.ones_like(other.data), requires_grad=requires_grad, device=other._device)

    @staticmethod
    def zeros_like(other: "Tensor", requires_grad: bool = False) -> "Tensor":
        """Create a tensor filled with zeros with the same shape as another tensor."""
        xp = np if not hasattr(other.data, 'get') else other.data.__class__
        return Tensor(xp.zeros_like(other.data), requires_grad=requires_grad, device=other._device)

    @staticmethod
    def randn(shape: Tuple[int, ...], requires_grad: bool = False, device: Optional[Union[Device, str]] = None) -> "Tensor":
        """Create a tensor with random values from standard normal distribution."""
        return Tensor(
            np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad, device=device
        )

    @staticmethod
    def rand(shape: Tuple[int, ...], requires_grad: bool = False, device: Optional[Union[Device, str]] = None) -> "Tensor":
        """Create a tensor with random values from uniform distribution [0, 1)."""
        return Tensor(
            np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad, device=device
        )

    @staticmethod
    def eye(n: int, requires_grad: bool = False, device: Optional[Union[Device, str]] = None) -> "Tensor":
        """Create an identity matrix."""
        return Tensor(np.eye(n, dtype=np.float32), requires_grad=requires_grad, device=device)

    @staticmethod
    def arange(
        *args: Union[int, float], requires_grad: bool = False, device: Optional[Union[Device, str]] = None, **kwargs: Any
    ) -> "Tensor":
        """Create a 1D tensor with values from start to stop with given step.

        Similar to numpy.arange and torch.arange.
        Supports both numpy-style (start, stop, step) and
        PyTorch-style (start, end, step) signatures.

        Usage:
            Tensor.arange(stop) -> values from 0 to stop-1
            Tensor.arange(start, stop) -> values from start to stop-1
            Tensor.arange(start, stop, step) -> values from start to stop-1 with step

        Args:
            *args: Either (stop,) or (start, stop) or (start, stop, step)
            requires_grad: Whether to track gradients.
            device: Device to place tensor on.
            **kwargs: Ignored for compatibility.

        Returns:
            1D Tensor with values [start, start+step, ..., stop-step].
        """
        start: Union[int, float]
        stop: Union[int, float]
        step: Union[int, float]

        if len(args) == 1:
            start, stop, step = 0, args[0], 1
        elif len(args) == 2:
            start, stop, step = args[0], args[1], 1
        elif len(args) == 3:
            start, stop, step = args[0], args[1], args[2]
        else:
            raise ValueError(
                f"arange expected 1-3 positional arguments, got {len(args)}"
            )

        return Tensor(
            np.arange(start, stop, step, dtype=np.float32), requires_grad=requires_grad, device=device
        )

    @staticmethod
    def cat(tensors: List["Tensor"], dim: int = 0) -> "Tensor":
        """Concatenate tensors along a dimension.
        
        Args:
            tensors: List of tensors to concatenate.
            dim: Dimension along which to concatenate.
            
        Returns:
            Concatenated tensor.
        """
        from nanotorch.utils import cat
        return cat(tensors, dim=dim)
    
    @staticmethod
    def stack(tensors: List["Tensor"], dim: int = 0) -> "Tensor":
        """Stack tensors along a new dimension.
        
        Args:
            tensors: List of tensors to stack.
            dim: Dimension to insert for stacking.
            
        Returns:
            Stacked tensor.
        """
        from nanotorch.utils import stack
        return stack(tensors, dim=dim)


# Context manager for gradient tracking
class no_grad:
    """Context manager to disable gradient tracking.

    Example:
        with no_grad():
            y = x * 2  # No gradient tracking
    """

    def __init__(self) -> None:
        self.prev_enable_grad = _ENABLE_GRAD

    def __enter__(self) -> None:
        global _ENABLE_GRAD
        self.prev_enable_grad = _ENABLE_GRAD
        _ENABLE_GRAD = False

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        global _ENABLE_GRAD
        _ENABLE_GRAD = self.prev_enable_grad


# Global flag for gradient tracking
_ENABLE_GRAD = True
