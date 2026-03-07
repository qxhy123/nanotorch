"""Utilities for serializing nanotorch tensors."""

import numpy as np
from typing import Any, List, Union
from nanotorch.tensor import Tensor


def tensor_to_dict(tensor: Tensor, name: str = None) -> dict:
    """Convert a nanotorch Tensor to a dictionary for JSON serialization.

    Args:
        tensor: The nanotorch Tensor to convert
        name: Optional name for the tensor

    Returns:
        Dictionary with shape, data, dtype, and name
    """
    # Get the numpy array from the tensor
    data = tensor.data if hasattr(tensor, 'data') else tensor

    # Convert to nested list
    if data.ndim == 1:
        nested_data = data.tolist()
    elif data.ndim == 2:
        nested_data = data.tolist()
    elif data.ndim == 3:
        nested_data = data.tolist()
    elif data.ndim == 4:
        nested_data = data.tolist()
    else:
        # For higher dimensions, flatten
        nested_data = data.flatten().tolist()

    return {
        "shape": list(data.shape),
        "data": nested_data,
        "dtype": str(data.dtype),
        "name": name,
    }


def dict_to_tensor(data: dict) -> Tensor:
    """Convert a dictionary back to a nanotorch Tensor.

    Args:
        data: Dictionary with shape, data, dtype

    Returns:
        nanotorch Tensor
    """
    np_array = np.array(data["data"], dtype=data["dtype"])
    np_array = np_array.reshape(data["shape"])
    return Tensor(np_array)


def ensure_list_format(data: np.ndarray) -> list:
    """Ensure numpy array is in proper list format for JSON serialization.

    Args:
        data: Numpy array

    Returns:
        Nested list
    """
    if data.ndim == 1:
        return data.tolist()
    elif data.ndim == 2:
        return data.tolist()
    elif data.ndim == 3:
        return data.tolist()
    elif data.ndim == 4:
        return data.tolist()
    else:
        return data.flatten().tolist()


def create_tensor_data(shape: List[int], data: Union[List, np.ndarray], dtype: str = "float32") -> dict:
    """Create a tensor data dictionary.

    Args:
        shape: Tensor shape
        data: Tensor data
        dtype: Data type

    Returns:
        Tensor data dictionary
    """
    if isinstance(data, np.ndarray):
        data = ensure_list_format(data)
    elif not isinstance(data, list):
        data = list(data)

    return {
        "shape": shape,
        "data": data,
        "dtype": dtype,
    }


def create_attention_data(
    weights: np.ndarray,
    queries: np.ndarray = None,
    keys: np.ndarray = None,
    values: np.ndarray = None,
    scale: float = None,
) -> dict:
    """Create attention data dictionary.

    Args:
        weights: Attention weights (batch, heads, seq_len, seq_len)
        queries: Query projections (batch, heads, seq_len, head_dim)
        keys: Key projections (batch, heads, seq_len, head_dim)
        values: Value projections (batch, heads, seq_len, head_dim)
        scale: Scale factor

    Returns:
        Attention data dictionary
    """
    head_dim = weights.shape[-1] if scale is None else int(1 / (scale ** 2))

    return {
        "weights": create_tensor_data(list(weights.shape), weights),
        "queries": create_tensor_data(list(queries.shape), queries) if queries is not None else None,
        "keys": create_tensor_data(list(keys.shape), keys) if keys is not None else None,
        "values": create_tensor_data(list(values.shape), values) if values is not None else None,
        "scale": scale or (1.0 / np.sqrt(head_dim)),
    }
