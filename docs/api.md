# nanotorch API Reference

This document provides a complete reference for the nanotorch API.

## Table of Contents
- [Core Tensor Operations](#core-tensor-operations)
- [Automatic Differentiation](#automatic-differentiation)
- [Neural Network Modules](#neural-network-modules)
  - [Layers](#layers)
  - [Convolution Layers](#convolution-layers)
  - [Pooling Layers](#pooling-layers)
  - [Normalization Layers](#normalization-layers)
  - [Activation Functions](#activation-functions)
  - [Utility Modules](#utility-modules)
- [Loss Functions](#loss-functions)
- [Optimizers](#optimizers)
- [Learning Rate Schedulers](#learning-rate-schedulers)
- [Utilities](#utilities)
- [Gradient Utilities](#gradient-utilities)
- [Common Patterns](#common-patterns)

## Core Tensor Operations

### `Tensor` Class

The `Tensor` class is the core data structure in nanotorch, similar to PyTorch's Tensor.

#### Creation Methods

```python
from nanotorch import Tensor

# Create from data
t = Tensor([1, 2, 3])
t = Tensor(np.array([1, 2, 3]))
t = Tensor(5.0)  # Scalar

# Factory methods
zeros = Tensor.zeros((3, 4))           # 3x4 tensor of zeros
ones = Tensor.ones((2, 3))             # 2x3 tensor of ones
rand = Tensor.rand((2, 2))             # Uniform random [0, 1)
randn = Tensor.randn((3, 3))           # Standard normal
eye = Tensor.eye(4)                    # 4x4 identity matrix
arange = Tensor.arange(10)             # [0, 1, ..., 9]
arange = Tensor.arange(0, 10, 2)       # [0, 2, 4, 6, 8]

# From existing tensors
zeros_like = Tensor.zeros_like(t)
ones_like = Tensor.ones_like(t)
```

#### Tensor Properties

```python
t.shape     # Tuple of dimensions
t.ndim      # Number of dimensions
t.dtype     # Data type (always np.float32)
t.data      # Underlying numpy array
t.grad      # Gradient tensor
t.requires_grad  # Whether gradient tracking is enabled
```

#### Basic Operations

```python
# Arithmetic (element-wise)
c = a + b
c = a - b
c = a * b
c = a / b
c = -a
c = a ** 2

# Matrix operations
c = a @ b          # Matrix multiplication
c = a.T            # Transpose
c = a.matmul(b)    # Matrix multiplication (method form)

# Reduction operations
c = a.sum()        # Sum of all elements
c = a.sum(axis=0)  # Sum along axis 0
c = a.sum(axis=0, keepdims=True)  # Keep dimensions
c = a.mean()       # Mean of all elements
c = a.mean(axis=1) # Mean along axis 1
c = a.max()        # Maximum value
c = a.max(axis=0)  # Maximum along axis
c = a.min()        # Minimum value
c = a.var()        # Variance
c = a.std()        # Standard deviation

# Shape operations
c = a.reshape((2, 3))  # Reshape tensor
c = a.squeeze()        # Remove dimensions of size 1
c = a.squeeze(axis=0)  # Remove specific dimension
c = a.unsqueeze(dim)   # Add dimension of size 1
c = a.transpose(dim0, dim1)  # Swap dimensions
c = a.permute((2, 0, 1))     # Rearrange dimensions

# New shape operations
c = a.flatten()              # Flatten all dims from 0
c = a.flatten(start_dim=1)   # Flatten from dim 1 to end
c = a.flatten(start_dim=1, end_dim=-1)  # Custom range
c = a.expand(4, 3)           # Broadcast to size
c = a.repeat(2, 1)           # Repeat along dims

# Splitting operations
chunks = a.split(split_size=2, dim=0)  # Split into chunks of size 2
chunks = a.chunk(chunks=3, dim=0)      # Split into 3 chunks

# Sorting & Selection
values, indices = a.topk(k=2, dim=-1)           # Top 2 values
values, indices = a.topk(k=2, largest=False)    # Bottom 2 values
values, indices = a.sort(dim=-1)                # Sort ascending
values, indices = a.sort(dim=-1, descending=True)  # Sort descending

# Conditional selection
result = a.where(condition, other)  # Select from a or other

# Indexing operations
c = a.gather(dim=0, index=indices)          # Gather values
c = a.scatter(dim=0, index=indices, src=src) # Scatter values

# Clamping
c = a.clamp(min=0.0)           # Clamp minimum
c = a.clamp(max=1.0)           # Clamp maximum
c = a.clamp(min=0.0, max=1.0)  # Clamp to range

# Element-wise functions
c = a.relu()       # Rectified Linear Unit
c = a.sigmoid()    # Sigmoid activation
c = a.tanh()       # Hyperbolic tangent
c = a.exp()        # Exponential
c = a.log()        # Natural logarithm
c = a.abs()        # Absolute value

# New activation functions (on tensor)
c = a.leaky_relu(negative_slope=0.01)  # Leaky ReLU
c = a.elu(alpha=1.0)                    # ELU
c = a.gelu()                            # GELU
c = a.silu()                            # SiLU (Swish)
c = a.softplus()                        # Softplus
c = a.hardswish()                       # Hard Swish
c = a.hardsigmoid()                     # Hard Sigmoid
c = a.prelu(weight)                     # PReLU
```

#### Gradient Tracking

```python
# Enable gradient tracking
x = Tensor([1.0, 2.0], requires_grad=True)
y = x * 2
z = y.sum()

# Compute gradients
z.backward()
print(x.grad)  # Gradient of z with respect to x

# Disable gradient tracking (for inference)
from nanotorch import no_grad
with no_grad():
    y = model(x)  # No gradient tracking
```

#### Utility Methods

```python
t.numpy()          # Convert to numpy array
t.item()           # Convert scalar tensor to Python float
t.detach()         # Return a detached tensor (no gradient tracking)
t.clone()          # Return a copy of the tensor
t.zero_grad()      # Reset gradient to zero
t.copy_(other)     # Copy data from other tensor
```

#### Static Methods

```python
# Concatenation and stacking
Tensor.cat([a, b], dim=0)   # Concatenate along dim
Tensor.stack([a, b], dim=0) # Stack along new dim
```

## Automatic Differentiation

### `no_grad` Context Manager

```python
from nanotorch import no_grad

with no_grad():
    # Operations inside this block won't track gradients
    y = x * 2 + 1
```

### Gradient Computation

The automatic differentiation system works automatically when you:
1. Create tensors with `requires_grad=True`
2. Perform operations on them
3. Call `.backward()` on the final tensor

```python
# Simple example
x = Tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # dy/dx = 2x + 3 = 7.0

# Multi-variable example
x1 = Tensor([1.0], requires_grad=True)
x2 = Tensor([2.0], requires_grad=True)
y = x1 * x2 + x1 ** 2
y.backward()
print(x1.grad)  # dy/dx1 = x2 + 2*x1 = 4.0
print(x2.grad)  # dy/dx2 = x1 = 1.0
```

## Neural Network Modules

### `Module` Base Class

All neural network modules inherit from `Module`.

```python
from nanotorch.nn import Module

class MyModule(Module):
    def __init__(self):
        super().__init__()
        self.linear = Linear(10, 5)
        self.activation = ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
```

#### Module Methods

```python
module = MyModule()

# Forward pass
output = module(input)

# Access parameters
params = list(module.parameters())
print(f"Number of parameters: {len(params)}")

# State dict (for saving/loading)
state_dict = module.state_dict()
module.load_state_dict(state_dict)

# Training/evaluation modes
module.train()    # Enable training mode (affects Dropout, BatchNorm)
module.eval()     # Enable evaluation mode

# Register buffers
module.register_buffer("running_mean", Tensor.zeros((10,)))
```

### Layers

#### `Linear`
```python
from nanotorch.nn import Linear

layer = Linear(in_features=10, out_features=5, bias=True)
output = layer(input)  # input shape: (batch_size, 10) or (10,)
```

#### `Dropout`
```python
from nanotorch.nn import Dropout

dropout = Dropout(p=0.5)  # 50% dropout probability
output = dropout(input)   # Only applied in training mode
```

#### `Sequential`
```python
from nanotorch.nn import Sequential

model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 5),
    Sigmoid()
)
output = model(input)
```

### Convolution Layers

#### `Conv1D`
```python
from nanotorch.nn import Conv1D

conv = Conv1D(in_channels=3, out_channels=16, kernel_size=3, 
              stride=1, padding=1, dilation=1, bias=True)
output = conv(input)  # input shape: (batch_size, 3, length)
```

#### `Conv2D`
```python
from nanotorch.nn import Conv2D

conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3,
              stride=1, padding=0, dilation=1, bias=True)
output = conv(input)  # input shape: (batch_size, 3, height, width)
```

#### `Conv3D`
```python
from nanotorch.nn import Conv3D

conv = Conv3D(in_channels=3, out_channels=16, kernel_size=3,
              stride=1, padding=0, dilation=1, bias=True)
# kernel_size, stride, padding, dilation can also be tuples of 3
output = conv(input)  # input shape: (batch_size, 3, depth, height, width)
```

#### `ConvTranspose2D`
```python
from nanotorch.nn import ConvTranspose2D

conv = ConvTranspose2D(in_channels=16, out_channels=3, kernel_size=3,
                       stride=2, padding=1, output_padding=1, bias=True)
output = conv(input)  # Upsampling convolution
```

#### `ConvTranspose3D`
```python
from nanotorch.nn import ConvTranspose3D

conv = ConvTranspose3D(in_channels=16, out_channels=3, kernel_size=3,
                       stride=2, padding=1, output_padding=1, bias=True)
output = conv(input)  # 3D upsampling convolution
```

### Pooling Layers

#### `MaxPool1d`
```python
from nanotorch.nn import MaxPool1d

pool = MaxPool1d(kernel_size=2, stride=2, padding=0)
output = pool(input)  # input shape: (N, C, L)
```

#### `MaxPool2d`
```python
from nanotorch.nn import MaxPool2d

pool = MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False)
output = pool(input)  # input shape: (N, C, H, W)
```

#### `MaxPool3d`
```python
from nanotorch.nn import MaxPool3d

pool = MaxPool3d(kernel_size=2, stride=2, padding=0)
output = pool(input)  # input shape: (N, C, D, H, W)
```

#### `AvgPool1d`
```python
from nanotorch.nn import AvgPool1d

pool = AvgPool1d(kernel_size=2, stride=2, padding=0,
                 ceil_mode=False, count_include_pad=True)
output = pool(input)  # input shape: (N, C, L)
```

#### `AvgPool2d`
```python
from nanotorch.nn import AvgPool2d

pool = AvgPool2d(kernel_size=2, stride=2, padding=0,
                 ceil_mode=False, count_include_pad=True)
output = pool(input)  # input shape: (N, C, H, W)
```

#### `AvgPool3d`
```python
from nanotorch.nn import AvgPool3d

pool = AvgPool3d(kernel_size=2, stride=2, padding=0)
output = pool(input)  # input shape: (N, C, D, H, W)
```

#### `AdaptiveAvgPool2d`
```python
from nanotorch.nn import AdaptiveAvgPool2d

pool = AdaptiveAvgPool2d(output_size=(1, 1))  # Global average pooling
output = pool(input)  # Any H, W -> (1, 1)
```

#### `AdaptiveMaxPool2d`
```python
from nanotorch.nn import AdaptiveMaxPool2d

pool = AdaptiveMaxPool2d(output_size=1, return_indices=False)
output = pool(input)
```

### Normalization Layers

#### `BatchNorm1d/2d/3d`
```python
from nanotorch.nn import BatchNorm1d, BatchNorm2d, BatchNorm3d

bn1d = BatchNorm1d(num_features=16)
bn2d = BatchNorm2d(num_features=16)
bn3d = BatchNorm3d(num_features=16)
```

#### `LayerNorm`
```python
from nanotorch.nn import LayerNorm

ln = LayerNorm(normalized_shape=64, eps=1e-5, elementwise_affine=True)
output = ln(input)  # input shape: (*, 64)
```

#### `GroupNorm`
```python
from nanotorch.nn import GroupNorm

gn = GroupNorm(num_groups=2, num_channels=6, eps=1e-5, affine=True)
output = gn(input)  # input shape: (N, C, *)
```

#### `InstanceNorm1d/2d/3d`
```python
from nanotorch.nn import InstanceNorm1d, InstanceNorm2d, InstanceNorm3d

in1d = InstanceNorm1d(num_features=16)
in2d = InstanceNorm2d(num_features=64)
in3d = InstanceNorm3d(num_features=32)
```

### Activation Functions

```python
from nanotorch.nn import (
    ReLU, LeakyReLU, ELU, GELU, SiLU, PReLU,
    Sigmoid, Tanh, Softmax, LogSoftmax,
    Softplus, Hardswish, Hardsigmoid
)

# Standard activations
relu = ReLU()
leaky_relu = LeakyReLU(negative_slope=0.01)
elu = ELU(alpha=1.0)
gelu = GELU()
silu = SiLU()
prelu = PReLU(num_parameters=1, init=0.25)
sigmoid = Sigmoid()
tanh = Tanh()
softmax = Softmax(dim=-1)
log_softmax = LogSoftmax(dim=-1)

# Mobile-friendly activations
softplus = Softplus()
hardswish = Hardswish()
hardsigmoid = Hardsigmoid()

output = activation(input)
```

### Utility Modules

#### `Flatten`
```python
from nanotorch.nn import Flatten

flatten = Flatten(start_dim=1, end_dim=-1)
output = flatten(input)  # Flatten dims 1 to end
```

#### `Identity`
```python
from nanotorch.nn import Identity

identity = Identity()
output = identity(input)  # Returns input unchanged
```

## Loss Functions

### `MSE` (Mean Squared Error)
```python
from nanotorch.nn import MSE

criterion = MSE(reduction='mean')  # 'mean', 'sum', or 'none'
loss = criterion(predictions, targets)
```

### `L1Loss` (Mean Absolute Error)
```python
from nanotorch.nn import L1Loss

criterion = L1Loss(reduction='mean')
loss = criterion(predictions, targets)
```

### `SmoothL1Loss` (Huber Loss)
```python
from nanotorch.nn import SmoothL1Loss

criterion = SmoothL1Loss(reduction='mean', beta=1.0)
loss = criterion(predictions, targets)
```

### `CrossEntropyLoss`
```python
from nanotorch.nn import CrossEntropyLoss

criterion = CrossEntropyLoss(reduction='mean')
loss = criterion(logits, targets)  # targets are class indices
```

### `BCELoss` (Binary Cross Entropy)
```python
from nanotorch.nn import BCELoss

criterion = BCELoss(reduction='mean')
loss = criterion(predictions, targets)  # predictions should be in [0, 1]
```

### `BCEWithLogitsLoss`
```python
from nanotorch.nn import BCEWithLogitsLoss

criterion = BCEWithLogitsLoss(reduction='mean')
loss = criterion(logits, targets)  # Numerically stable BCE with sigmoid
```

### `NLLLoss` (Negative Log Likelihood)
```python
from nanotorch.nn import NLLLoss

criterion = NLLLoss(reduction='mean')
loss = criterion(log_probs, targets)  # Use with LogSoftmax
```

### Functional Interface
```python
from nanotorch.nn import (
    mse_loss, l1_loss, smooth_l1_loss,
    cross_entropy_loss, bce_loss, bce_with_logits_loss, nll_loss
)

loss = mse_loss(predictions, targets)
loss = cross_entropy_loss(logits, targets)
```

## Optimizers

### `Optimizer` Base Class

All optimizers inherit from `Optimizer`.

```python
from nanotorch.optim import Optimizer

optimizer = Optimizer(params=model.parameters(), defaults={'lr': 0.01})
optimizer.zero_grad()  # Clear gradients
optimizer.step()       # Update parameters
```

### `SGD` (Stochastic Gradient Descent)
```python
from nanotorch.optim import SGD

optimizer = SGD(
    params=model.parameters(),
    lr=0.01,
    momentum=0.9,
    dampening=0,
    weight_decay=0.0001,
    nesterov=False
)

# Training loop
for epoch in range(num_epochs):
    optimizer.zero_grad()      # Clear gradients
    loss.backward()            # Compute gradients
    optimizer.step()           # Update parameters
```

### `Adam`
```python
from nanotorch.optim import Adam

optimizer = Adam(
    params=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0
)
```

### `AdamW`
```python
from nanotorch.optim import AdamW

optimizer = AdamW(
    params=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01  # Decoupled weight decay
)
```

### `RMSprop`
```python
from nanotorch.optim import RMSprop

optimizer = RMSprop(
    params=model.parameters(),
    lr=0.01,
    alpha=0.99,
    eps=1e-8,
    weight_decay=0.0,
    momentum=0.0
)
```

### `Adagrad`
```python
from nanotorch.optim import Adagrad

optimizer = Adagrad(
    params=model.parameters(),
    lr=0.01,
    lr_decay=0.0,
    weight_decay=0.0,
    eps=1e-10
)
```

## Learning Rate Schedulers

### `StepLR`
```python
from nanotorch.optim import StepLR

scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 60
# lr = 0.0005   if 60 <= epoch < 90

for epoch in range(100):
    train(...)
    scheduler.step()
```

### `MultiStepLR`
```python
from nanotorch.optim import MultiStepLR

scheduler = MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
# lr = 0.05     if epoch < 30
# lr = 0.005    if 30 <= epoch < 80
# lr = 0.0005   if epoch >= 80
```

### `ExponentialLR`
```python
from nanotorch.optim import ExponentialLR

scheduler = ExponentialLR(optimizer, gamma=0.9)
# lr = 0.1 * 0.9^epoch
```

### `CosineAnnealingLR`
```python
from nanotorch.optim import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0001)
# lr follows cosine curve from base_lr to eta_min
```

### `LinearLR`
```python
from nanotorch.optim import LinearLR

scheduler = LinearLR(
    optimizer, 
    start_factor=0.333,  # Start at 1/3 of base_lr
    end_factor=1.0,       # End at base_lr
    total_iters=5
)
```

### `ReduceLROnPlateau`
```python
from nanotorch.optim import ReduceLROnPlateau

scheduler = ReduceLROnPlateau(
    optimizer, 
    mode='min',        # 'min' or 'max'
    factor=0.1,        # Multiply LR by this
    patience=10,       # Epochs to wait
    threshold=1e-4,
    cooldown=0,
    min_lr=0
)

for epoch in range(100):
    train(...)
    val_loss = validate(...)
    scheduler.step(val_loss)  # Pass metric to scheduler
```

### `ConstantLR`
```python
from nanotorch.optim import ConstantLR

scheduler = ConstantLR(optimizer, factor=1.0, total_iters=5)
```

## Utilities

### Serialization

```python
from nanotorch.utils import save, load, save_state_dict, load_state_dict

# Save/load entire model
save(model, "model.pth")
loaded_model = load("model.pth")

# Save/load only parameters
save_state_dict(model.state_dict(), "model_state.pth")
state_dict = load_state_dict("model_state.pth")
model.load_state_dict(state_dict)
```

### Initialization

```python
from nanotorch.utils import (
    xavier_uniform_, xavier_normal_,
    kaiming_uniform_, kaiming_normal_,
    zeros_, ones_, uniform_, normal_
)

# Xavier/Glorot initialization
xavier_uniform_(tensor, gain=1.0)
xavier_normal_(tensor, gain=1.0)

# Kaiming/He initialization
kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')
kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu')

# Simple initialization
zeros_(tensor)
ones_(tensor)
uniform_(tensor, low=0.0, high=1.0)
normal_(tensor, mean=0.0, std=1.0)
```

### Tensor Operations

```python
from nanotorch.utils import flatten, cat, stack

# Flatten
flat = flatten(tensor, start_dim=0, end_dim=-1)

# Concatenate
result = cat([tensor1, tensor2], dim=0)

# Stack
result = stack([tensor1, tensor2], dim=0)
```

### Model Utilities

```python
from nanotorch.utils import num_parameters, count_parameters

# Get number of parameters
n = num_parameters(model)
total, trainable = count_parameters(model, trainable=True)
```

### Random Seed

```python
from nanotorch.utils import manual_seed

manual_seed(42)  # Set random seed for reproducibility
```

## Gradient Utilities

### `clip_grad_norm_`
```python
from nanotorch.utils import clip_grad_norm_

# Clip gradient norm to max_norm
total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)

# Different norm types
total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2.0)  # L2
total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=1.0)  # L1
total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=float('inf'))  # L-inf
```

### `clip_grad_value_`
```python
from nanotorch.utils import clip_grad_value_

# Clip gradient values to [-clip_value, clip_value]
clip_grad_value_(model.parameters(), clip_value=0.5)
```

### `get_grad_norm_`
```python
from nanotorch.utils import get_grad_norm_

# Get gradient norm without clipping
norm = get_grad_norm_(model.parameters(), norm_type=2.0)
```

## Common Patterns

### Training a Neural Network

```python
import numpy as np
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from nanotorch.optim import SGD, StepLR
from nanotorch.utils import clip_grad_norm_

# Create model
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)

# Loss and optimizer
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Training data
inputs = Tensor.randn((100, 784))
targets = Tensor(np.random.randint(0, 10, (100,)))

# Training loop
for epoch in range(100):
    # Forward pass
    predictions = model(inputs)
    loss = criterion(predictions, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Gradient clipping
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    scheduler.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Using Convolutional Networks

```python
from nanotorch.nn import Sequential, Conv2D, MaxPool2d, Flatten, Linear, ReLU

model = Sequential(
    Conv2D(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),       # 28x28 -> 14x14
    Conv2D(32, 64, kernel_size=3, padding=1), # 14x14 -> 14x14
    ReLU(),
    MaxPool2d(kernel_size=2, stride=2),       # 14x14 -> 7x7
    Flatten(),
    Linear(64 * 7 * 7, 128),
    ReLU(),
    Linear(128, 10)
)
```

### Learning Rate Scheduling with Validation

```python
from nanotorch.optim import Adam, ReduceLROnPlateau

optimizer = Adam(model.parameters(), lr=0.001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

for epoch in range(100):
    train_loss = train(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)
    
    scheduler.step(val_loss)  # Reduce LR if val_loss doesn't improve
```

## Performance Tips

1. **Use `no_grad()` for inference**: Disable gradient tracking when you don't need gradients.
2. **Batch operations**: Use matrix operations instead of loops.
3. **Clear gradients**: Always call `optimizer.zero_grad()` before `loss.backward()`.
4. **Gradient clipping**: Use `clip_grad_norm_()` for training stability.
5. **Learning rate scheduling**: Use schedulers for better convergence.

## Limitations

- Only CPU support (no GPU acceleration)
- Limited set of operations compared to PyTorch
- No distributed training support
- Groups > 1 not yet implemented for transposed convolutions

## See Also

- [Design Documentation](design.md) - Architecture and implementation details
- [Examples](../examples/) - Example scripts and tutorials
- [Benchmarks](../benchmarks/) - Performance benchmarks
