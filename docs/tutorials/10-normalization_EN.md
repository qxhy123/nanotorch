# Tutorial 10: Normalization Layers

## Calming the Storm Within...

Imagine a conversation where everyone speaks at different volumes.

One person whispers, another shouts. Some are barely audible, others deafening. The listener—trying to understand all of them—becomes confused, overwhelmed, unable to focus on what matters.

**Deep neural networks face the same problem.**

```
The Problem of Shifting Distributions:

  Layer 1 outputs values around 0.5
       ↓
  Layer 2 receives them, outputs values around 10
       ↓
  Layer 3 receives them, outputs values around 0.001
       ↓
  Each layer must constantly adapt:
  "What range am I working with today?"

Training becomes a game of chasing moving targets.
Learning slows. Gradients vanish or explode.
The network struggles to find its footing.
```

**Normalization layers bring calm to the chaos.** They take whatever values come in—large or small, clustered or scattered—and reshape them to a consistent distribution. Mean zero. Variance one. Every layer receives data in a format it can work with.

BatchNorm does this across the batch. LayerNorm does it across features. GroupNorm finds a middle ground. Each has its philosophy, its strengths, its loyal applications.

In this tutorial, we'll implement these normalization layers and understand when to use which. We'll see why BatchNorm needs running statistics, why LayerNorm powers Transformers, and why GroupNorm saves the day for small batches.

---

## Table of Contents

1. [Overview](#overview)
2. [Why Normalization is Needed](#why-normalization-is-needed)
3. [BatchNorm Implementation](#batchnorm-implementation)
4. [LayerNorm Implementation](#layernorm-implementation)
5. [GroupNorm Implementation](#groupnorm-implementation)
6. [InstanceNorm Implementation](#instancenorm-implementation)
7. [Comparison of Normalization Methods](#comparison-of-normalization-methods)
8. [Usage Examples](#usage-examples)
9. [Summary](#summary)

---

## Overview

Normalization layers are core components of modern deep neural networks. They can:
- **Accelerate Training**: Allow using larger learning rates
- **Stabilize Training**: Reduce Internal Covariate Shift
- **Regularization Effect**: BatchNorm has certain regularization properties

nanotorch implements the following normalization layers:
- **BatchNorm1d/2d/3d**: Batch normalization
- **LayerNorm**: Layer normalization
- **GroupNorm**: Group normalization
- **InstanceNorm1d/2d/3d**: Instance normalization

---

## Why Normalization is Needed

### Internal Covariate Shift

In deep networks, the input distribution of each layer changes as the parameters of previous layers are updated. This leads to:
1. Each layer must constantly adapt to new input distributions
2. Learning rates must be set very small
3. Training is unstable and convergence is slow

### The Effect of Normalization

```
Before Normalization:        After Normalization:
    ┌─────┐                     ┌─────┐
    │Large│                     │Mean 0│
    │Var  │                     │Var 1 │
    └─────┘                     └─────┘
 Unstable distribution        Stable distribution
```

Through normalization, we stabilize each layer's input around mean 0 and variance 1, making training more stable.

---

## BatchNorm Implementation

### Principle

BatchNorm calculates mean and variance for each channel across samples within a batch:

```
Input: (N, C, H, W)
For each channel c:
    mean = mean(x[:, c, :, :])  # Average over N, H, W
    var = var(x[:, c, :, :])
    x_norm[:, c, :, :] = (x[:, c, :, :] - mean) / sqrt(var + eps)
    output[:, c, :, :] = gamma * x_norm[:, c, :, :] + beta
```

### Difference Between Training and Inference

| Phase | Statistics Used |
|-------|-----------------|
| Training | Mean and variance from current batch |
| Inference | Running mean/var accumulated during training |

### Base Class Implementation

```python
# nanotorch/nn/normalization.py

class _BatchNorm(Module):
    """BatchNorm base class"""
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # Learnable parameters
        self.gamma = None  # Scale parameter
        self.beta = None   # Shift parameter
        if self.affine:
            self.gamma = Tensor.ones((num_features,), requires_grad=True)
            self.beta = Tensor.zeros((num_features,), requires_grad=True)
        
        # Running statistics
        self.running_mean = None
        self.running_var = None
        if self.track_running_stats:
            self.running_mean = Tensor.zeros((num_features,), requires_grad=False)
            self.running_var = Tensor.ones((num_features,), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        N, C = x.shape[0], x.shape[1]
        # Calculate mean over batch and spatial dimensions
        axes = (0,) + tuple(range(2, x.ndim))
        
        if self.training and self.track_running_stats:
            # Calculate current batch statistics
            mean = x.mean(axis=axes, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
            
            # Update running statistics
            mean_squeezed = mean.squeeze(axis=axes)
            var_squeezed = var.squeeze(axis=axes)
            
            if self.momentum is not None:
                self.running_mean.data = (
                    1 - self.momentum
                ) * self.running_mean.data + self.momentum * mean_squeezed.data
                self.running_var.data = (
                    1 - self.momentum
                ) * self.running_var.data + self.momentum * var_squeezed.data
            
            # Normalize using batch statistics
            x_normalized = (x - mean) / (var + self.eps) ** 0.5
        else:
            # Use running statistics
            broadcast_shape = (1, C) + (1,) * (x.ndim - 2)
            mean = self.running_mean.reshape(broadcast_shape)
            var = self.running_var.reshape(broadcast_shape)
            x_normalized = (x - mean) / (var + self.eps) ** 0.5
        
        # Affine transformation
        if self.affine:
            broadcast_shape = (1, C) + (1,) * (x.ndim - 2)
            gamma_reshaped = self.gamma.reshape(broadcast_shape)
            beta_reshaped = self.beta.reshape(broadcast_shape)
            x_normalized = gamma_reshaped * x_normalized + beta_reshaped
        
        return x_normalized
```

### BatchNorm2d Implementation

```python
class BatchNorm2d(_BatchNorm):
    """2D Batch Normalization.
    
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W)
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
    
    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"BatchNorm2d expects 4D input, got {x.ndim}D")
```

### Usage Example

```python
from nanotorch import Tensor
from nanotorch.nn import BatchNorm2d

# Create BatchNorm layer
bn = BatchNorm2d(num_features=64)

# Training mode
bn.train()
x = Tensor.randn((16, 64, 32, 32))
output = bn(x)

# Inference mode
bn.eval()
x_test = Tensor.randn((1, 64, 32, 32))
output_test = bn(x_test)  # Uses running statistics
```

---

## LayerNorm Implementation

### Principle

LayerNorm normalizes each sample across the feature dimension, without relying on batch statistics:

```
Input: (N, C, H, W)
For each sample n:
    mean = mean(x[n, :, :, :])  # Average over C, H, W
    var = var(x[n, :, :, :])
    x_norm[n, :, :, :] = (x[n, :, :, :] - mean) / sqrt(var + eps)
```

### Implementation

```python
class LayerNorm(Module):
    """Layer Normalization.
    
    Normalizes over the last (or last few) dimensions.
    Commonly used in Transformers and NLP tasks.
    
    Shape:
        - Input: (*, normalized_shape)
        - Output: (*, normalized_shape)
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        self.gamma = None
        self.beta = None
        if self.elementwise_affine:
            self.gamma = Tensor.ones(normalized_shape, requires_grad=True)
            self.beta = Tensor.zeros(normalized_shape, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # Use LayerNormFunction for forward propagation
        return LayerNormFunction.apply(
            x, self.normalized_shape, self.gamma, self.beta, self.eps
        )
```

### Usage Example

```python
from nanotorch.nn import LayerNorm

# Normalize over the last dimension
ln = LayerNorm(normalized_shape=512)

# Usage in Transformer
x = Tensor.randn((32, 100, 512))  # (batch, seq_len, hidden_dim)
output = ln(x)

# Normalize over last two dimensions
ln_2d = LayerNorm(normalized_shape=(64, 64))
x_2d = Tensor.randn((8, 3, 64, 64))
output_2d = ln_2d(x_2d)
```

---

## GroupNorm Implementation

### Principle

GroupNorm divides channels into groups and normalizes within each group's channels and spatial dimensions:

```
Input: (N, C, H, W), groups = G
Divide C channels into G groups, each with C/G channels
For each sample n and group g:
    mean = mean(x[n, g*C/G:(g+1)*C/G, :, :])
    var = var(x[n, g*C/G:(g+1)*C/G, :, :])
```

### Implementation

```python
class GroupNorm(Module):
    """Group Normalization.
    
    Normalizes within groups after dividing channels.
    Between LayerNorm and InstanceNorm.
    
    Args:
        num_groups: Number of groups
        num_channels: Number of channels (must be divisible by num_groups)
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        self.gamma = None
        self.beta = None
        if self.affine:
            self.gamma = Tensor.ones((num_channels,), requires_grad=True)
            self.beta = Tensor.zeros((num_channels,), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return GroupNormFunction.apply(
            x, self.num_groups, self.gamma, self.beta, self.eps
        )
```

### Usage Example

```python
from nanotorch.nn import GroupNorm

# Divide 32 channels into 8 groups
gn = GroupNorm(num_groups=8, num_channels=32)

x = Tensor.randn((16, 32, 64, 64))
output = gn(x)

# Special cases: GroupNorm(num_groups=1, ...) = LayerNorm (channel dimension)
# Special cases: GroupNorm(num_groups=C, ...) = InstanceNorm
```

---

## InstanceNorm Implementation

### Principle

InstanceNorm normalizes each channel of each sample independently:

```
Input: (N, C, H, W)
For each sample n and each channel c:
    mean = mean(x[n, c, :, :])
    var = var(x[n, c, :, :])
```

### Implementation

```python
class InstanceNorm2d(_InstanceNorm):
    """Instance Normalization for 2D inputs.
    
    Commonly used in style transfer tasks.
    Equivalent to GroupNorm(num_groups=num_features).
    
    Shape:
        - Input: (N, C, H, W) or (C, H, W)
        - Output: same as input
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,  # InstanceNorm doesn't learn parameters by default
        track_running_stats: bool = False,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            num_spatial_dims=2,
        )
```

---

## Comparison of Normalization Methods

### Normalization Dimension Comparison

Assuming input shape `(N, C, H, W)`:

```
BatchNorm:    Calculate statistics over (N, H, W), each channel independent
LayerNorm:    Calculate statistics over (C, H, W), each sample independent
InstanceNorm: Calculate statistics over (H, W), each channel of each sample independent
GroupNorm:    Calculate statistics over (C/G, H, W), each group independent
```

### Visual Comparison

```
Input Tensor (N=2, C=4, H=2, W=2):

BatchNorm (across N, H, W):  LayerNorm (across C, H, W):
┌───┬───┐                    ┌───┬───┐
│ N │ N │                    │C=1│C=2│
│ 0 │ 1 │                    │   │   │
├───┼───┤                    ├───┼───┤
│ N │ N │                    │C=3│C=4│
│ 0 │ 1 │                    │   │   │
└───┴───┘                    └───┴───┘

GroupNorm (G=2):             InstanceNorm:
┌─────┬─────┐                ┌───┬───┐
│G=1  │G=2  │                │N=0│N=1│
│C=1,2│C=3,4│                │each│each│
└─────┴─────┘                │channel│channel│
                             │independent│independent│
                             └───┴───┘
```

### Selection Guide

| Scenario | Recommended Normalization | Reason |
|----------|---------------------------|--------|
| CNN Image Classification | BatchNorm | Works well with large batches |
| Small batches / Memory constrained | GroupNorm | Doesn't depend on batch size |
| RNN/NLP | LayerNorm | Handles variable-length sequences |
| Style Transfer | InstanceNorm | Preserves content, ignores style |
| Object Detection/Segmentation | GroupNorm or SyncBN | Batches usually small |
| Transformer | LayerNorm | Standard Transformer architecture |

---

## Usage Examples

### BatchNorm in CNN

```python
from nanotorch import Tensor
from nanotorch.nn import Conv2D, BatchNorm2d, ReLU, Sequential

# Standard CNN block
def conv_block(in_ch, out_ch):
    return Sequential(
        Conv2D(in_ch, out_ch, kernel_size=3, padding=1),
        BatchNorm2d(out_ch),
        ReLU(),
    )

block = conv_block(64, 128)
x = Tensor.randn((4, 64, 32, 32))
output = block(x)
```

### LayerNorm in Transformer

```python
from nanotorch.nn import Linear, LayerNorm, Dropout, ReLU

class TransformerBlock:
    def __init__(self, d_model=512, d_ff=2048):
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.ff = Sequential(
            Linear(d_model, d_ff),
            ReLU(),
            Linear(d_ff, d_model),
        )
    
    def __call__(self, x):
        # Pre-norm architecture
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
```

### GroupNorm in ResNet

```python
class ResBlock:
    def __init__(self, channels, groups=32):
        self.conv1 = Conv2D(channels, channels, 3, padding=1)
        self.gn1 = GroupNorm(groups, channels)
        self.conv2 = Conv2D(channels, channels, 3, padding=1)
        self.gn2 = GroupNorm(groups, channels)
    
    def __call__(self, x):
        identity = x
        out = self.gn1(self.conv1(x)).relu()
        out = self.gn2(self.conv2(out))
        return (out + identity).relu()
```

---

## Summary

This tutorial introduced four normalization layers in nanotorch:

| Normalization | Normalization Dimensions | Characteristics | Use Case |
|---------------|-------------------------|-----------------|----------|
| **BatchNorm** | Batch + Spatial | Depends on batch size | CNN, large batch training |
| **LayerNorm** | Feature + Spatial | Batch independent | Transformer, NLP |
| **GroupNorm** | Within groups | Batch independent | Small batches, detection/segmentation |
| **InstanceNorm** | Single channel spatial | Batch independent | Style transfer |

### Key Points

1. **BatchNorm** behaves differently during training and inference (running statistics)
2. **LayerNorm** is suitable for handling variable-length sequences
3. **GroupNorm** is a good alternative to BatchNorm for small batches
4. **InstanceNorm** is commonly used in generation tasks

### Next Steps

In [Tutorial 11: Recurrent Neural Networks](11-rnn.md), we will learn how to implement RNN, LSTM, and GRU.

---

**References**:
- [Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Group Normalization](https://arxiv.org/abs/1803.08494)
- [Instance Normalization](https://arxiv.org/abs/1607.08022)
