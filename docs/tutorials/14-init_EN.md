# Tutorial 14: Parameter Initialization

## Every Journey Needs a Good Beginning...

Imagine setting off on a hike.

Start at the right trailhead, and the path unfolds naturally. Start in the wrong direction, and you might wander for hours, never finding your destination.

**Neural networks face the same challenge.**

```
The Peril of Poor Beginnings:

  All zeros:
    Every neuron computes the same output
    Every gradient is identical
    Every update is the same
    → A hundred neurons collapse into one
    → The network cannot learn

  Too large:
    Signals explode through layers
    Activations saturate (all 0s or all 1s)
    Gradients vanish to nothing
    → Training dies before it begins

  Too small:
    Signals fade with each layer
    Gradients shrink to insignificance
    Learning becomes imperceptibly slow
    → Training crawls toward failure

The right initialization is like the right trailhead:
It doesn't guarantee success, but it makes success possible.
```

**Initialization is the art of giving your network a fighting chance.** Not all random starts are equal. Xavier initialization keeps variance stable for smooth activations. Kaiming initialization compensates for ReLU's tendency to zero out half its inputs. Orthogonal initialization preserves gradient flow through deep architectures.

In this tutorial, we'll implement these initialization strategies and understand the mathematics behind them. We'll see why modern networks almost always use Kaiming, why Transformers prefer truncated normal, and why the right initialization can mean the difference between a model that learns and one that never converges.

---

## Table of Contents

1. [Overview](#overview)
2. [Why Initialization Matters](#why-initialization-matters)
3. [Constant Initialization](#constant-initialization)
4. [Xavier/Glorot Initialization](#xavierglorot-initialization)
5. [Kaiming/He Initialization](#kaiminghe-initialization)
6. [Orthogonal Initialization](#orthogonal-initialization)
7. [Truncated Normal Distribution](#truncated-normal-distribution)
8. [Usage Examples](#usage-examples)
9. [Summary](#summary)

---

## Overview

Parameter initialization is an underrated but extremely important aspect of deep learning. Good initialization can:
- **Accelerate Convergence**: Reduce training time
- **Avoid Vanishing/Exploding Gradients**: Stabilize training
- **Improve Final Performance**: Find better local optima

nanotorch provides various initialization methods:

| Method | Use Case |
|--------|----------|
| `zeros_`, `ones_`, `constant_` | Bias initialization |
| `xavier_uniform_`, `xavier_normal_` | Tanh/Sigmoid activation |
| `kaiming_uniform_`, `kaiming_normal_` | ReLU family activation |
| `orthogonal_` | RNN/Deep networks |
| `trunc_normal_` | Transformer |
| `sparse_` | Sparse connections |

---

## Why Initialization Matters

### Problem: Vanishing/Exploding Gradients

```
Gradient propagation in deep networks:
∂L/∂W_1 = ∂L/∂W_L × ∏_{i=2}^{L} ∂h_i/∂h_{i-1}

If each layer's gradient < 1: Exponential decay (vanishing)
If each layer's gradient > 1: Exponential growth (exploding)
```

### Problems with All-Zero Initialization

```python
# Wrong example
W = np.zeros((in_features, out_features))  # All neurons identical
# Forward propagation: All outputs identical
# Backward propagation: All gradients identical
# Result: Symmetry cannot be broken, network cannot learn
```

### Problems with Too Large/Small Weights

```
Weights too large:    Activations saturate → Gradients near 0
Weights too small:    Activations too small → Gradients near 0
Weights appropriate:  Activations reasonable → Gradients propagate effectively
```

---

## Constant Initialization

### Implementation

```python
# nanotorch/utils.py

def zeros_(tensor: Tensor) -> Tensor:
    """Fill tensor with zeros."""
    tensor.data.fill(0)
    return tensor

def ones_(tensor: Tensor) -> Tensor:
    """Fill tensor with ones."""
    tensor.data.fill(1)
    return tensor

def constant_(tensor: Tensor, value: float) -> Tensor:
    """Fill tensor with constant value."""
    tensor.data.fill(value)
    return tensor

def eye_(tensor: Tensor) -> Tensor:
    """Fill 2D tensor with identity matrix."""
    assert tensor.ndim == 2, "eye_ only supports 2D tensors"
    tensor.data = np.eye(tensor.shape[0], tensor.shape[1], dtype=tensor.data.dtype)
    return tensor

def dirac_(tensor: Tensor) -> Tensor:
    """Fill convolution kernel with Dirac delta function.
    
    Preserves forward propagation signal strength.
    Commonly used for initializing convolutional layers.
    """
    if tensor.ndim < 2:
        raise ValueError("dirac_ requires at least 2D tensor")
    
    tensor.data.fill(0)
    out_channels = tensor.shape[0]
    in_channels = tensor.shape[1]
    
    # Set center positions to 1
    min_channels = min(out_channels, in_channels)
    for c in range(min_channels):
        center_idx = tuple(
            s // 2 if i >= 2 else c
            for i, s in enumerate(tensor.shape)
        )
        tensor.data[center_idx] = 1.0
    
    return tensor
```

### Usage

```python
from nanotorch import Tensor
from nanotorch.utils import zeros_, ones_, constant_

# Initialize bias
bias = Tensor(np.zeros(128))
zeros_(bias)

# Initialize weights
weight = Tensor(np.empty((256, 128)))
constant_(weight, 0.5)
```

---

## Xavier/Glorot Initialization

### Principle

Xavier initialization assumes the activation function is linear (like Tanh), with the goal of keeping **forward and backward propagation variances consistent**.

```
Variance analysis:
- Input x: Var(x)
- Weight W: Var(W) 
- Output y = Wx: Var(y) = n_in * Var(W) * Var(x)

For Var(y) = Var(x):
Var(W) = 1 / n_in

For backward propagation variance consistency:
Var(W) = 1 / n_out

Compromise:
Var(W) = 2 / (n_in + n_out)
```

### Implementation

```python
def xavier_uniform_(
    tensor: Tensor,
    gain: float = 1.0
) -> Tensor:
    """Xavier uniform distribution initialization.
    
    Also known as Glorot initialization.
    
    Weights sampled from uniform distribution U(-a, a), where:
    a = gain * sqrt(6 / (fan_in + fan_out))
    
    Args:
        tensor: Tensor to initialize
        gain: Scaling factor
    
    Returns:
        Initialized tensor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # uniform(-a, a) has variance a^2/3
    
    tensor.data = np.random.uniform(-a, a, tensor.shape).astype(tensor.data.dtype)
    return tensor

def xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0
) -> Tensor:
    """Xavier normal distribution initialization.
    
    Weights sampled from normal distribution N(0, std^2), where:
    std = gain * sqrt(2 / (fan_in + fan_out))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    tensor.data = np.random.normal(0, std, tensor.shape).astype(tensor.data.dtype)
    return tensor

def _calculate_fan_in_and_fan_out(tensor: Tensor) -> Tuple[int, int]:
    """Calculate fan_in and fan_out.
    
    - Linear: fan_in = in_features, fan_out = out_features
    - Conv2D: fan_in = in_channels * kernel_h * kernel_w
              fan_out = out_channels * kernel_h * kernel_w
    """
    if tensor.ndim < 2:
        raise ValueError("Cannot calculate fan_in and fan_out for tensor with less than 2 dimensions")
    
    if tensor.ndim == 2:
        # Linear: (out_features, in_features)
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        # Conv: (out_channels, in_channels, *kernel_size)
        receptive_field_size = 1
        for s in tensor.shape[2:]:
            receptive_field_size *= s
        fan_in = tensor.shape[1] * receptive_field_size
        fan_out = tensor.shape[0] * receptive_field_size
    
    return fan_in, fan_out
```

### Usage

```python
from nanotorch.nn import Linear, Tanh
from nanotorch.utils import xavier_normal_

# Xavier initialization suitable for Tanh/Sigmoid
linear = Linear(128, 64)
xavier_normal_(linear.weight)  # Initialize weights
```

---

## Kaiming/He Initialization

### Principle

Kaiming initialization is specifically designed for **ReLU family activation functions**. Since ReLU sets half of the inputs to 0, extra compensation is needed:

```
ReLU variance analysis:
- After ReLU, only half of neurons are activated
- To maintain variance, weight variance needs to be doubled

Kaiming uniform:

$$a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in}}} \quad \text{(fan\_in mode)}$$

$$a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_out}}} \quad \text{(fan\_out mode)}$$

For ReLU, $\text{gain} = \sqrt{2}$

### Implementation

```python
def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu"
) -> Tensor:
    """Kaiming uniform distribution initialization.
    
    Also known as He initialization.
    
    Suitable for ReLU, LeakyReLU, PReLU, etc.
    
    Args:
        tensor: Tensor to initialize
        a: LeakyReLU negative slope (for calculating gain)
        mode: 'fan_in' or 'fan_out'
        nonlinearity: Activation function type
    
    Returns:
        Initialized tensor
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    
    tensor.data = np.random.uniform(-bound, bound, tensor.shape).astype(tensor.data.dtype)
    return tensor

def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu"
) -> Tensor:
    """Kaiming normal distribution initialization.
    
    Weights sampled from N(0, std^2), where:
    std = gain / sqrt(fan)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    
    tensor.data = np.random.normal(0, std, tensor.shape).astype(tensor.data.dtype)
    return tensor

def calculate_gain(nonlinearity: str, param: float = 0) -> float:
    """Calculate gain value for activation function."""
    if nonlinearity == "linear" or nonlinearity == "sigmoid":
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        return math.sqrt(2.0 / (1 + param ** 2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
```

### Usage

```python
from nanotorch.nn import Linear, ReLU, Conv2D
from nanotorch.utils import kaiming_normal_, zeros_

# Kaiming initialization suitable for ReLU
linear = Linear(128, 64)
kaiming_normal_(linear.weight, nonlinearity="relu")
zeros_(linear.bias)

# Convolutional layer
conv = Conv2D(3, 64, kernel_size=3)
kaiming_normal_(conv.weight, nonlinearity="relu")
zeros_(conv.bias)

# LeakyReLU
from nanotorch.nn import LeakyReLU
lrelu = LeakyReLU(negative_slope=0.01)
linear2 = Linear(64, 32)
kaiming_normal_(linear2.weight, a=0.01, nonlinearity="leaky_relu")
```

---

## Orthogonal Initialization

### Principle

Orthogonal initialization makes the rows (or columns) of the weight matrix mutually orthogonal:
- Forward propagation: Preserves signal strength
- Backward propagation: Gradients are uncorrelated
- Especially suitable for RNNs, preventing vanishing/exploding gradients

### Implementation

```python
def orthogonal_(
    tensor: Tensor,
    gain: float = 1.0
) -> Tensor:
    """Orthogonal matrix initialization.
    
    Uses QR decomposition to generate orthogonal matrix.
    
    Args:
        tensor: Tensor to initialize (at least 2D)
        gain: Scaling factor
    
    Returns:
        Initialized tensor
    """
    if tensor.ndim < 2:
        raise ValueError("orthogonal_ requires at least 2D tensor")
    
    # Flatten all dimensions except last two
    original_shape = tensor.shape
    if tensor.ndim > 2:
        flat_shape = (np.prod(original_shape[:-2]), *original_shape[-2:])
    else:
        flat_shape = original_shape
    
    rows, cols = flat_shape[-2], flat_shape[-1]
    
    # Generate random matrix
    flat_tensor = np.random.randn(*flat_shape).astype(np.float32)
    
    # QR decomposition
    if rows >= cols:
        q, r = np.linalg.qr(flat_tensor.reshape(-1, cols))
        q = q.reshape(flat_shape)
    else:
        # Transpose then do QR
        flat_tensor = flat_tensor.transpose(-1, -2)
        q, r = np.linalg.qr(flat_tensor.reshape(-1, rows))
        q = q.reshape(flat_shape[:-2] + (cols, rows))
        q = q.transpose(-1, -2)
    
    # Apply gain
    q = q * gain
    
    tensor.data = q.reshape(original_shape).astype(tensor.data.dtype)
    return tensor
```

### Usage

```python
from nanotorch.nn import LSTM
from nanotorch.utils import orthogonal_, zeros_

# RNN uses orthogonal initialization
lstm = LSTM(input_size=64, hidden_size=128)

for name, param in lstm.named_parameters():
    if 'weight' in name:
        orthogonal_(param)
    elif 'bias' in name:
        zeros_(param)
```

---

## Truncated Normal Distribution

### Principle

Truncated normal distribution limits sampled values to `[mean - 2*std, mean + 2*std]` range, avoiding extreme values. Commonly used in Transformer initialization.

### Implementation

```python
def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0
) -> Tensor:
    """Truncated normal distribution initialization.
    
    Samples from N(mean, std^2) but limits to [a, b] range.
    
    Args:
        tensor: Tensor to initialize
        mean: Normal distribution mean
        std: Normal distribution standard deviation
        a: Lower truncation bound (in std units)
        b: Upper truncation bound (in std units)
    
    Returns:
        Initialized tensor
    """
    # Generate truncated normal distribution
    lower = mean + a * std
    upper = mean + b * std
    
    # Use rejection sampling
    size = tensor.shape
    result = np.zeros(size)
    
    remaining = np.ones(size, dtype=bool)
    while remaining.any():
        # Generate candidate values
        candidates = np.random.normal(mean, std, size)
        
        # Accept values within range
        valid = (candidates >= lower) & (candidates <= upper)
        result = np.where(remaining & valid, candidates, result)
        remaining = remaining & ~valid
    
    tensor.data = result.astype(tensor.data.dtype)
    return tensor
```

### Usage

```python
from nanotorch.nn import Linear
from nanotorch.utils import trunc_normal_

# Transformer style initialization
linear = Linear(512, 512)
trunc_normal_(linear.weight, std=0.02)  # Transformer commonly uses std=0.02
```

---

## Sparse Initialization

### Implementation

```python
def sparse_(
    tensor: Tensor,
    sparsity: float,
    std: float = 0.01
) -> Tensor:
    """Sparse initialization.
    
    Most weights are 0, only a small portion are non-zero.
    
    Args:
        tensor: Tensor to initialize
        sparsity: Sparsity (proportion of zeros)
        std: Standard deviation for non-zero values
    
    Returns:
        Initialized tensor
    """
    tensor.data.fill(0)
    
    # Randomly select non-zero positions
    total_elements = tensor.data.size
    num_nonzero = int(total_elements * (1 - sparsity))
    
    flat_indices = np.random.choice(total_elements, num_nonzero, replace=False)
    flat_tensor = tensor.data.flatten()
    flat_tensor[flat_indices] = np.random.normal(0, std, num_nonzero)
    
    tensor.data = flat_tensor.reshape(tensor.shape)
    return tensor
```

---

## Usage Examples

### Initialization Helper Functions

```python
from nanotorch import Tensor
from nanotorch.nn import Module, Linear, Conv2D, ReLU, BatchNorm2d
from nanotorch.utils import kaiming_normal_, xavier_normal_, zeros_, ones_

def init_weights(module: Module, init_type: str = "kaiming"):
    """Unified weight initialization function."""
    
    for name, param in module.named_parameters():
        if 'weight' in name:
            if init_type == "kaiming":
                kaiming_normal_(param, nonlinearity="relu")
            elif init_type == "xavier":
                xavier_normal_(param)
            elif init_type == "normal":
                param.data = np.random.normal(0, 0.02, param.shape).astype(np.float32)
        elif 'bias' in name:
            zeros_(param)

def init_bn(module: Module):
    """Initialize BatchNorm layers."""
    for name, param in module.named_parameters():
        if 'weight' in name or 'gamma' in name:
            ones_(param)
        elif 'bias' in name or 'beta' in name:
            zeros_(param)
```

### CNN Initialization

```python
class SimpleCNN(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv2D(3, 64, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2d(64)
        self.conv2 = Conv2D(64, 128, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(128)
        self.fc = Linear(128 * 28 * 28, num_classes)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, Conv2D):
                kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    zeros_(module.bias)
            elif isinstance(module, BatchNorm2d):
                ones_(module.weight)
                zeros_(module.bias)
            elif isinstance(module, Linear):
                kaiming_normal_(module.weight, nonlinearity="relu")
                zeros_(module.bias)
```

### Transformer Initialization

```python
def init_transformer(module):
    """Transformer style initialization."""
    for name, param in module.named_parameters():
        if 'weight' in name:
            if 'layernorm' in name.lower() or 'norm' in name.lower():
                ones_(param)
            else:
                trunc_normal_(param, std=0.02)
        elif 'bias' in name:
            zeros_(param)
```

### ResNet Initialization

```python
def init_resnet(module):
    """ResNet style initialization.
    
    - Conv layers: Kaiming normal
    - BN layers: weight=1, bias=0
    - Final FC: Special initialization
    """
    for name, module in module.named_modules():
        if isinstance(module, Conv2D):
            kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, BatchNorm2d):
            ones_(module.weight)
            zeros_(module.bias)
        elif isinstance(module, Linear):
            # Final FC layer initialized to smaller values
            module.weight.data = np.random.normal(0, 0.01, module.weight.shape).astype(np.float32)
            zeros_(module.bias)
```

---

## Summary

This tutorial introduced parameter initialization methods in nanotorch:

| Method | Formula | Use Case |
|--------|---------|----------|
| **zeros/ones/constant** | Constant | Bias |
| **Xavier** | $\sqrt{\frac{2}{n_{in}+n_{out}}}$ | Tanh/Sigmoid |
| **Kaiming** | $\sqrt{\frac{2}{n_{in}}}$ | ReLU family |
| **Orthogonal** | QR decomposition | RNN |
| **Trunc Normal** | Truncated normal | Transformer |

### Key Points

1. **ReLU uses Kaiming**, **Tanh uses Xavier**
2. **Bias usually initialized to 0**
3. **BatchNorm**: weight=1, bias=0
4. **RNN**: Orthogonal initialization prevents gradient problems
5. **Transformer**: Truncated normal, std=0.02

### Next Steps

In [Tutorial 15: Advanced Topics](15-advanced.md), we will explore advanced topics including gradient clipping, mixed precision training, etc.

---

**References**:
- [Understanding the difficulty of training deep feedforward neural networks (Xavier)](http://proceedings.mlr.press/v9/glorot10a.html)
- [Delving Deep into Rectifiers (Kaiming)](https://arxiv.org/abs/1502.01852)
- [All you need is a good init](https://arxiv.org/abs/1511.06422)
