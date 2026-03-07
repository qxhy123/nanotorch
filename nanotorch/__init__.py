"""
nanotorch - A minimal PyTorch implementation from scratch.

This package provides a minimal implementation of core PyTorch functionality
including tensors, automatic differentiation, neural network modules,
and optimizers.

Main components:
- tensor: Core Tensor class with autograd support
- nn: Neural network modules (Linear, ReLU, etc.)
- optim: Optimizers (SGD, Adam)
- autograd: Automatic differentiation engine
- data: Data loading utilities (Dataset, DataLoader)
- device: Device management (CPU/GPU)
"""

from nanotorch.tensor import Tensor, no_grad
from nanotorch.device import (
    Device,
    cpu,
    is_cuda_available,
    device_count,
    current_device,
    set_device,
    get_device_name,
    get_device_capability,
    cuda,
)
from nanotorch.nn import (
    MaxPool2d,
    AvgPool2d,
    max_pool2d,
    avg_pool2d,
    AdaptiveAvgPool2d,
    AdaptiveMaxPool2d,
    adaptive_avg_pool2d,
    adaptive_max_pool2d,
    LayerNorm,
    layer_norm,
    GroupNorm,
    group_norm,
    InstanceNorm2d,
    instance_norm,
)
from nanotorch.data import (
    Dataset,
    TensorDataset,
    Subset,
    random_split,
    Sampler,
    SequentialSampler,
    RandomSampler,
    BatchSampler,
    DataLoader,
)
from nanotorch.backend import (
    Backend,
    get_backend,
    set_backend,
    get_backend_for_device,
)

__version__ = "0.1.0"
__all__ = [
    "Tensor",
    "no_grad",
    "Device",
    "cpu",
    "is_cuda_available",
    "device_count",
    "current_device",
    "set_device",
    "get_device_name",
    "get_device_capability",
    "cuda",
    "MaxPool2d",
    "AvgPool2d",
    "max_pool2d",
    "avg_pool2d",
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d",
    "adaptive_avg_pool2d",
    "adaptive_max_pool2d",
    "LayerNorm",
    "layer_norm",
    "GroupNorm",
    "group_norm",
    "InstanceNorm2d",
    "instance_norm",
    "Dataset",
    "TensorDataset",
    "Subset",
    "random_split",
    "Sampler",
    "SequentialSampler",
    "RandomSampler",
    "BatchSampler",
    "DataLoader",
    # Backend module
    "Backend",
    "get_backend",
    "set_backend",
    "get_backend_for_device",
]
