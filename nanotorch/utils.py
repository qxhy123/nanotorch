"""
Utility functions for nanotorch.

This module provides various utility functions for tensor operations,
initialization, and other common tasks.
"""

import logging
import numpy as np
import random
from typing import Tuple, Dict, Any, List, Callable, cast, Union
from numpy.typing import NDArray
from nanotorch.tensor import Tensor

logger = logging.getLogger(__name__)


# Random seed management
def manual_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    logger.info(f"Random seed set to {seed}")


# Initialization functions
def xavier_uniform_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill tensor with values from Xavier uniform distribution.

    Also known as Glorot initialization.

    Args:
        tensor: Tensor to initialize.
        gain: Scaling factor.

    Returns:
        Initialized tensor.
    """
    if tensor.ndim < 2:
        raise ValueError("Xavier initialization requires at least 2 dimensions")

    fan_in = int(tensor.shape[1] if tensor.ndim == 2 else np.prod(tensor.shape[1:]))
    fan_out = int(
        tensor.shape[0]
        if tensor.ndim == 2
        else tensor.shape[0] * np.prod(tensor.shape[2:])
    )

    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    tensor.data = np.random.uniform(-limit, limit, tensor.shape).astype(np.float32)
    return tensor


def xavier_normal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill tensor with values from Xavier normal distribution.

    Args:
        tensor: Tensor to initialize.
        gain: Scaling factor.

    Returns:
        Initialized tensor.
    """
    if tensor.ndim < 2:
        raise ValueError("Xavier initialization requires at least 2 dimensions")

    fan_in = int(tensor.shape[1] if tensor.ndim == 2 else np.prod(tensor.shape[1:]))
    fan_out = int(
        tensor.shape[0]
        if tensor.ndim == 2
        else tensor.shape[0] * np.prod(tensor.shape[2:])
    )

    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    tensor.data = np.random.randn(*tensor.shape).astype(np.float32) * std
    return tensor


def kaiming_uniform_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> Tensor:
    """Fill tensor with values from Kaiming uniform distribution.

    Also known as He initialization.

    Args:
        tensor: Tensor to initialize.
        a: Negative slope of the rectifier (default 0 for ReLU).
        mode: Either 'fan_in' (default) or 'fan_out'.
        nonlinearity: Nonlinearity function ('relu' or 'leaky_relu').

    Returns:
        Initialized tensor.
    """
    if tensor.ndim < 2:
        raise ValueError("Kaiming initialization requires at least 2 dimensions")

    fan_in = int(tensor.shape[1] if tensor.ndim == 2 else np.prod(tensor.shape[1:]))
    fan_out = int(
        tensor.shape[0]
        if tensor.ndim == 2
        else tensor.shape[0] * np.prod(tensor.shape[2:])
    )

    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError(f"Mode must be 'fan_in' or 'fan_out', got {mode}")

    if nonlinearity == "relu":
        gain = np.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        gain = np.sqrt(2.0 / (1 + a**2))
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

    bound = gain * np.sqrt(3.0 / fan)
    tensor.data = np.random.uniform(-bound, bound, tensor.shape).astype(np.float32)
    return tensor


def kaiming_normal_(
    tensor: Tensor, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu"
) -> Tensor:
    """Fill tensor with values from Kaiming normal distribution.

    Args:
        tensor: Tensor to initialize.
        a: Negative slope of the rectifier (default 0 for ReLU).
        mode: Either 'fan_in' (default) or 'fan_out'.
        nonlinearity: Nonlinearity function ('relu' or 'leaky_relu').

    Returns:
        Initialized tensor.
    """
    if tensor.ndim < 2:
        raise ValueError("Kaiming initialization requires at least 2 dimensions")

    fan_in = int(tensor.shape[1] if tensor.ndim == 2 else np.prod(tensor.shape[1:]))
    fan_out = int(
        tensor.shape[0]
        if tensor.ndim == 2
        else tensor.shape[0] * np.prod(tensor.shape[2:])
    )

    if mode == "fan_in":
        fan = fan_in
    elif mode == "fan_out":
        fan = fan_out
    else:
        raise ValueError(f"Mode must be 'fan_in' or 'fan_out', got {mode}")

    if nonlinearity == "relu":
        gain = np.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        gain = np.sqrt(2.0 / (1 + a**2))
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

    std = gain / np.sqrt(fan)
    tensor.data = np.random.randn(*tensor.shape).astype(np.float32) * std
    return tensor


def zeros_(tensor: Tensor) -> Tensor:
    """Fill tensor with zeros.

    Args:
        tensor: Tensor to initialize.

    Returns:
        Initialized tensor.
    """
    tensor.data = np.zeros(tensor.shape, dtype=np.float32)
    return tensor


def ones_(tensor: Tensor) -> Tensor:
    """Fill tensor with ones.

    Args:
        tensor: Tensor to initialize.

    Returns:
        Initialized tensor.
    """
    tensor.data = np.ones(tensor.shape, dtype=np.float32)
    return tensor


def uniform_(tensor: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
    """Fill tensor with values from uniform distribution.

    Args:
        tensor: Tensor to initialize.
        low: Lower bound of uniform distribution.
        high: Upper bound of uniform distribution.

    Returns:
        Initialized tensor.
    """
    tensor.data = np.random.uniform(low, high, tensor.shape).astype(np.float32)
    return tensor


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> Tensor:
    """Fill tensor with values from normal distribution.

    Args:
        tensor: Tensor to initialize.
        mean: Mean of normal distribution.
        std: Standard deviation of normal distribution.

    Returns:
        Initialized tensor.
    """
    tensor.data = np.random.randn(*tensor.shape).astype(np.float32) * std + mean
    return tensor


def trunc_normal_(
    tensor: Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0
) -> Tensor:
    """Fill tensor with values from truncated normal distribution.

    Values are drawn from a normal distribution N(mean, std) and then
    clamped to the range [a, b].

    Args:
        tensor: Tensor to initialize.
        mean: Mean of normal distribution.
        std: Standard deviation of normal distribution.
        a: Lower bound for truncation (in units of std from mean).
        b: Upper bound for truncation (in units of std from mean).

    Returns:
        Initialized tensor.
    """
    lower = mean + a * std
    upper = mean + b * std

    data = np.random.randn(*tensor.shape).astype(np.float32) * std + mean
    data = np.clip(data, lower, upper)
    tensor.data = data
    return tensor


def orthogonal_(tensor: Tensor, gain: float = 1.0) -> Tensor:
    """Fill tensor with (semi) orthogonal matrix.

    Args:
        tensor: Tensor to initialize (must be at least 2D).
        gain: Scaling factor.

    Returns:
        Initialized tensor.
    """
    if tensor.ndim < 2:
        raise ValueError("Orthogonal initialization requires at least 2 dimensions")

    rows = tensor.shape[0]
    cols = int(np.prod(tensor.shape[1:]))

    flat_shape = (rows, cols)

    random_matrix = np.random.randn(*flat_shape).astype(np.float32)

    if rows > cols:
        q, _ = np.linalg.qr(random_matrix)
    else:
        q, _ = np.linalg.qr(random_matrix.T)
        q = q.T

    q = q * gain

    tensor.data = q.reshape(tensor.shape).astype(np.float32)
    return tensor


def sparse_(tensor: Tensor, sparsity: float, std: float = 0.01) -> Tensor:
    """Fill tensor with sparse random values.

    Args:
        tensor: Tensor to initialize.
        sparsity: Fraction of elements to set to zero.
        std: Standard deviation of non-zero elements.

    Returns:
        Initialized tensor.
    """
    data = np.random.randn(*tensor.shape).astype(np.float32) * std
    mask = np.random.rand(*tensor.shape) > sparsity
    tensor.data = data * mask
    return tensor


def eye_(tensor: Tensor) -> Tensor:
    """Fill 2D tensor with identity matrix.

    Args:
        tensor: Tensor to initialize (must be 2D).

    Returns:
        Initialized tensor.
    """
    if tensor.ndim != 2:
        raise ValueError("Eye initialization requires 2D tensor")

    tensor.data = np.eye(tensor.shape[0], tensor.shape[1], dtype=np.float32)
    return tensor


def dirac_(tensor: Tensor) -> Tensor:
    """Fill tensor with Dirac delta function.

    For 3D/4D/5D tensors, fills the main diagonal with 1.

    Args:
        tensor: Tensor to initialize (must be 3D, 4D, or 5D).

    Returns:
        Initialized tensor.
    """
    if tensor.ndim not in (3, 4, 5):
        raise ValueError("Dirac initialization requires 3D, 4D, or 5D tensor")

    tensor.data = np.zeros(tensor.shape, dtype=np.float32)

    min_size = min(tensor.shape[0], min(tensor.shape[2:]))

    for i in range(min_size):
        if tensor.ndim == 3:
            tensor.data[i, i, :] = 1.0
        elif tensor.ndim == 4:
            tensor.data[i, i, i, i] = 1.0
        elif tensor.ndim == 5:
            tensor.data[i, i, i, i, i] = 1.0

    return tensor


def constant_(tensor: Tensor, value: float) -> Tensor:
    """Fill tensor with constant value.

    Args:
        tensor: Tensor to initialize.
        value: Value to fill.

    Returns:
        Initialized tensor.
    """
    tensor.data = np.full(tensor.shape, value, dtype=np.float32)
    return tensor


def calculate_gain(nonlinearity: str, param: float = 0.0) -> float:
    """Calculate gain value for a given nonlinearity function.

    Args:
        nonlinearity: Name of nonlinearity function.
        param: Optional parameter for nonlinearity (e.g., negative slope for leaky_relu).

    Returns:
        Gain value.
    """
    if nonlinearity == "linear" or nonlinearity == "conv1d" or nonlinearity == "conv2d" or nonlinearity == "conv3d":
        return 1.0
    elif nonlinearity == "sigmoid":
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return np.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        if param < 0:
            raise ValueError(f"Negative slope for leaky_relu must be positive, got {param}")
        return np.sqrt(2.0 / (1 + param**2))
    elif nonlinearity == "selu":
        return 0.6
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")


# Tensor utilities
def flatten(tensor: Tensor, start_dim: int = 0, end_dim: int = -1) -> Tensor:
    """Flatten a tensor.

    Args:
        tensor: Input tensor.
        start_dim: First dimension to flatten.
        end_dim: Last dimension to flatten.

    Returns:
        Flattened tensor.
    """
    if end_dim == -1:
        end_dim = tensor.ndim - 1

    if start_dim < 0:
        start_dim = tensor.ndim + start_dim
    if end_dim < 0:
        end_dim = tensor.ndim + end_dim

    if start_dim > end_dim:
        raise ValueError(f"start_dim ({start_dim}) must be <= end_dim ({end_dim})")

    # Calculate new shape
    new_shape = list(tensor.shape[:start_dim])
    flattened_size = int(np.prod(tensor.shape[start_dim : end_dim + 1]))
    new_shape.append(flattened_size)
    new_shape.extend(tensor.shape[end_dim + 1 :])

    return tensor.reshape(tuple(new_shape))


def cat(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Concatenate tensors along a dimension.

    Args:
        tensors: List of tensors to concatenate.
        dim: Dimension along which to concatenate.

    Returns:
        Concatenated tensor.
    """
    if not tensors:
        raise ValueError("cat expects a non-empty list of tensors")

    if any(t.requires_grad for t in tensors):
        from nanotorch.autograd import cat as cat_func
        return cat_func(tensors, dim=dim)
    
    arrays = [t.data for t in tensors]
    result_data = np.concatenate(arrays, axis=dim)
    return Tensor(result_data, requires_grad=False)


def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """Stack tensors along a new dimension.

    Args:
        tensors: List of tensors to stack.
        dim: Dimension to insert for stacking.

    Returns:
        Stacked tensor.
    """
    if not tensors:
        raise ValueError("stack expects a non-empty list of tensors")

    # Check all tensors have same shape
    first_shape = tensors[0].shape
    for t in tensors[1:]:
        if t.shape != first_shape:
            raise ValueError(
                f"All tensors must have the same shape, got {first_shape} and {t.shape}"
            )

    # Use autograd function if any tensor requires gradient
    if any(t.requires_grad for t in tensors):
        from nanotorch.autograd import stack as stack_func
        return stack_func(tensors, dim=dim)

    # Convert to numpy arrays for stacking
    arrays = [t.data for t in tensors]
    result_data = np.stack(arrays, axis=dim)

    return Tensor(result_data, requires_grad=False)


def split(
    tensor: Tensor, 
    split_size_or_sections: Union[int, List[int]], 
    dim: int = 0
) -> Tuple[Tensor, ...]:
    """Split a tensor into multiple sub-tensors along a dimension.
    
    Args:
        tensor: Input tensor to split.
        split_size_or_sections: 
            If int, size of each chunk (except possibly last).
            If list, sizes of each section.
        dim: Dimension along which to split.
            
    Returns:
        Tuple of tensors.
        
    Raises:
        ValueError: If split parameters are invalid.
    """
    if dim < 0:
        dim = tensor.ndim + dim
    
    if dim < 0 or dim >= tensor.ndim:
        raise ValueError(
            f"Dimension out of range. Expected 0 <= dim < {tensor.ndim}, got {dim}"
        )
    
    dim_size = tensor.shape[dim]
    
    if isinstance(split_size_or_sections, int):
        # Split into chunks of given size
        split_size = split_size_or_sections
        if split_size <= 0:
            raise ValueError(f"split_size must be positive, got {split_size}")
        
        # Calculate split indices
        full_chunks = dim_size // split_size
        remainder = dim_size % split_size
        
        split_sizes = [split_size] * full_chunks
        if remainder > 0:
            split_sizes.append(remainder)
    else:
        # Split into sections of given sizes
        split_sizes = split_size_or_sections
        if sum(split_sizes) != dim_size:
            raise ValueError(
                f"Sum of split sizes ({sum(split_sizes)}) must equal "
                f"dimension size ({dim_size})"
            )
    
    results = []
    start = 0
    for split_size in split_sizes:
        end = start + split_size
        idx = [slice(None)] * tensor.ndim
        idx[dim] = slice(start, end)
        results.append(tensor[tuple(idx)])
        start = end

    return tuple(results)


def chunk(tensor: Tensor, chunks: int, dim: int = 0) -> Tuple[Tensor, ...]:
    """Split a tensor into a specific number of chunks.
    
    Args:
        tensor: Input tensor to chunk.
        chunks: Number of chunks to split into.
        dim: Dimension along which to chunk.
            
    Returns:
        Tuple of tensors.
        
    Raises:
        ValueError: If chunks is not positive or dimension size is not divisible.
    """
    if chunks <= 0:
        raise ValueError(f"chunks must be positive, got {chunks}")
    
    if dim < 0:
        dim = tensor.ndim + dim
    
    dim_size = tensor.shape[dim]
    
    # Calculate chunk sizes (as even as possible)
    chunk_size = dim_size // chunks
    remainder = dim_size % chunks
    
    split_sizes = [chunk_size + 1 if i < remainder else chunk_size 
                   for i in range(chunks)]
    
    # Use split function
    return split(tensor, split_sizes, dim=dim)


# Model utilities
def num_parameters(module: Any) -> int:
    """Count total number of parameters in a module.

    Args:
        module: Module to count parameters for.

    Returns:
        Total number of parameters.
    """
    return sum(p.data.size for p in module.parameters())


def count_parameters(module: Any, trainable_only: bool = True) -> Tuple[int, int]:
    """Count parameters in a module.

    Args:
        module: Module to count parameters for.
        trainable_only: If True, only count parameters with requires_grad=True.

    Returns:
        Tuple of (total parameters, trainable parameters).
    """
    total = 0
    trainable = 0

    for param in module.parameters():
        if not trainable_only or param.requires_grad:
            total += param.data.size
        if param.requires_grad:
            trainable += param.data.size

    return total, trainable


# Serialization utilities
def save_state_dict(state_dict: Dict[str, NDArray[np.float32]], filepath: str) -> None:
    """Save state dictionary to file using pickle.

    Args:
        state_dict: State dictionary to save.
        filepath: Path to save file.
    """
    import pickle

    with open(filepath, "wb") as f:
        pickle.dump(state_dict, f)


def load_state_dict(filepath: str) -> Dict[str, NDArray[np.float32]]:
    """Load state dictionary from pickle file.

    Args:
        filepath: Path to load file.

    Returns:
        Loaded state dictionary.
    """
    import pickle

    with open(filepath, "rb") as f:
        return cast(Dict[str, NDArray[np.float32]], pickle.load(f))


def save(module: Any, filepath: str) -> None:
    """Save module state to file.

    Args:
        module: Module to save.
        filepath: Path to save file.
    """
    save_state_dict(module.state_dict(), filepath)


def load(module: Any, filepath: str, strict: bool = True) -> None:
    """Load module state from file.

    Args:
        module: Module to load state into.
        filepath: Path to load file.
        strict: Whether to enforce exact key matching.
    """
    state_dict = load_state_dict(filepath)
    module.load_state_dict(state_dict, strict=strict)


# Performance utilities
def benchmark_operation(
    op_func: Callable[..., Any], *args: Any, iterations: int = 100, warmup: int = 10
) -> float:
    """Benchmark an operation.

    Args:
        op_func: Function to benchmark.
        *args: Arguments to pass to the function.
        iterations: Number of iterations to run.
        warmup: Number of warmup iterations.

    Returns:
        Average time per iteration in milliseconds.
    """
    import time

    # Warmup
    for _ in range(warmup):
        op_func(*args)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        op_func(*args)
    end = time.perf_counter()

    avg_time = (end - start) * 1000 / iterations  # milliseconds
    return avg_time


# Gradient clipping utilities
def clip_grad_norm_(
    parameters: Union[Tensor, List[Tensor]],
    max_norm: float,
    norm_type: float = 2.0,
) -> float:
    """Clip gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Args:
        parameters: An iterable of Tensors or a single Tensor.
        max_norm: Maximum norm of the gradients.
        norm_type: Norm type (1, 2, or float('inf')). Default: 2.0.

    Returns:
        Total norm of the gradients (before clipping).
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]

    # Filter parameters with gradients
    params_with_grad = [p for p in parameters if p.grad is not None]
    
    if len(params_with_grad) == 0:
        return 0.0

    # Compute total norm
    if norm_type == float('inf'):
        total_norm = max(np.abs(p.grad.data).max() for p in params_with_grad)
    else:
        total_norm = 0.0
        for p in params_with_grad:
            param_norm = np.float64(np.linalg.norm(p.grad.data.flatten(), norm_type))
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1.0 / norm_type)

    # Clip if needed
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params_with_grad:
            p.grad.data = p.grad.data * clip_coef

    return total_norm


def clip_grad_value_(
    parameters: Union[Tensor, List[Tensor]],
    clip_value: float,
) -> None:
    """Clip gradient values of an iterable of parameters.

    Gradients are modified in-place to be within [-clip_value, clip_value].

    Args:
        parameters: An iterable of Tensors or a single Tensor.
        clip_value: Maximum absolute value of the gradients.
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]

    for p in parameters:
        if p.grad is not None:
            p.grad.data = np.clip(p.grad.data, -clip_value, clip_value)


def get_grad_norm_(
    parameters: Union[Tensor, List[Tensor]],
    norm_type: float = 2.0,
) -> float:
    """Compute gradient norm of an iterable of parameters.

    Args:
        parameters: An iterable of Tensors or a single Tensor.
        norm_type: Norm type (1, 2, or float('inf')). Default: 2.0.

    Returns:
        Total norm of the gradients.
    """
    if isinstance(parameters, Tensor):
        parameters = [parameters]

    params_with_grad = [p for p in parameters if p.grad is not None]
    
    if len(params_with_grad) == 0:
        return 0.0

    if norm_type == float('inf'):
        return max(np.abs(p.grad.data).max() for p in params_with_grad)
    
    total_norm = 0.0
    for p in params_with_grad:
        param_norm = np.float64(np.linalg.norm(p.grad.data.flatten(), norm_type))
        total_norm += param_norm ** norm_type
    return total_norm ** (1.0 / norm_type)
