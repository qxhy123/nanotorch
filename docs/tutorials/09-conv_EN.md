# Tutorial 09: Convolution Layers

## The Sliding Window of Insight...

Imagine looking at a painting through a small cardboard frame.

You slide the frame across the canvas, one position at a time. At each stop, you see only a small patch—but your brain stitches these glimpses into a complete understanding of the whole.

**This is the essence of convolution.**

```
The Magic of the Sliding Window:

  Input image (a vast canvas of pixels)
       ↓
  A small kernel slides across (the frame)
       ↓
  At each position: multiply, sum, capture
       ↓
  A new image emerges (edges, textures, patterns)

What was once millions of raw pixels
becomes thousands of meaningful features.
The kernel learns to see—
edges here, corners there, textures everywhere.
```

**Convolution is how neural networks learn to see.** Unlike fully connected layers that connect everything to everything (impossibly expensive for images), convolution uses a small, shared kernel that slides across the input. Local connectivity. Weight sharing. The same edge detector that works on the top-left corner works on the bottom-right.

One kernel can detect edges. Stack dozens, and you detect shapes. Stack hundreds, and you detect objects. This hierarchical feature extraction is why convolutional networks revolutionized computer vision.

In this tutorial, we'll implement convolution from scratch—not just 2D for images, but 1D for sequences and 3D for videos. We'll see how the sliding window works, how gradients flow backward through it, and how transposed convolution lets us go from small to large again.

---

## Table of Contents

1. [Overview](#overview)
2. [Basic Concepts of Convolution](#basic-concepts-of-convolution)
3. [Conv1D Implementation](#conv1d-implementation)
4. [Conv2D Implementation](#conv2d-implementation)
5. [Conv3D Implementation](#conv3d-implementation)
6. [Transposed Convolution](#transposed-convolution)
7. [Gradient Computation](#gradient-computation)
8. [Usage Examples](#usage-examples)
9. [Summary](#summary)

---

## Overview

Convolutional Neural Networks (CNNs) are the core architecture in deep learning for processing images, audio, and sequence data. Convolution layers significantly reduce the number of parameters through two key features: **local connectivity** and **weight sharing**, while maintaining powerful feature extraction capabilities.

This tutorial will detail how to implement convolution layers in nanotorch, including:
- Conv1D: 1D convolution for sequence data
- Conv2D: 2D convolution for image processing
- Conv3D: 3D convolution for video or volumetric data
- ConvTranspose2D/3D: Transposed convolution (deconvolution)

---

## Basic Concepts of Convolution

### What is Convolution?

In deep learning, convolution is the operation of sliding a small **kernel/filter** over input data, computing the dot product and summing at each position.

```
Input Image (5x5):         Kernel (3x3):
[1  2  3  4  5]           [1  0  1]
[6  7  8  9  10]          [0  1  0]
[11 12 13 14 15]          [1  0  1]
[16 17 18 19 20]
[21 22 23 24 25]

Calculation at position (0,0):
1*1 + 2*0 + 3*1 +
6*0 + 7*1 + 8*0 +
11*1 + 12*0 + 13*1 = 1 + 3 + 7 + 11 + 13 = 35
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `kernel_size` | Size of the convolution kernel | - |
| `stride` | Step size for sliding | 1 |
| `padding` | Edge padding | 0 |
| `dilation` | Dilation rate to increase receptive field | 1 |
| `groups` | Grouped convolution | 1 |
| `bias` | Whether to add bias | True |

### Output Size Calculation

For input size $H_{in} \times W_{in}$, the output size is:

$$H_{out} = \left\lfloor \frac{H_{in} + 2 \times padding - dilation \times (kernel\_size - 1) - 1}{stride} \right\rfloor + 1$$

---

## Conv1D Implementation

### Data Layout

Conv1D processes input of shape `(N, C_in, L)`:
- N: Batch size
- C_in: Number of input channels
- L: Sequence length

### Implementation Code

```python
# nanotorch/nn/conv.py

class Conv1D(Module):
    """1D convolution layer.
    
    Applies a 1D convolution over an input signal composed of several input planes.
    
    Args:
        in_channels: Number of channels in the input signal.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        bias: If True, adds a learnable bias. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Weight shape: (out_channels, in_channels, kernel_size)
        weight_shape = (out_channels, in_channels, kernel_size)
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True
        )
        
        self.bias = None
        if bias:
            self.bias = Tensor(
                np.zeros((out_channels, 1), dtype=np.float32),
                requires_grad=True
            )
        
        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Conv1D expects 3D input (N, C, L), got {x.ndim}D")
        
        return Conv1DFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )
```

### Usage Example

```python
from nanotorch import Tensor
from nanotorch.nn import Conv1D

# Create Conv1D layer
conv = Conv1D(in_channels=16, out_channels=32, kernel_size=3, padding=1)

# Input: (batch_size, in_channels, length)
x = Tensor.randn((8, 16, 100))

# Forward pass
output = conv(x)
print(output.shape)  # (8, 32, 100)
```

---

## Conv2D Implementation

### Data Layout

Conv2D processes input of shape `(N, C_in, H, W)`:
- N: Batch size
- C_in: Number of input channels
- H: Height
- W: Width

### Implementation Code

```python
class Conv2D(Module):
    """2D convolution layer.
    
    Applies a 2D convolution over an input signal composed of several input planes.
    
    Shape:
        - Input: (N, C_in, H_in, W_in)
        - Output: (N, C_out, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        
        # Convert to tuple
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Weight shape: (out_channels, in_channels, kernel_height, kernel_width)
        weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True
        )
        
        self.bias = None
        if bias:
            self.bias = Tensor(
                np.zeros((out_channels, 1, 1), dtype=np.float32),
                requires_grad=True
            )
        
        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Conv2D expects 4D input (N, C, H, W), got {x.ndim}D")
        
        return Conv2DFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )
```

### Usage Example

```python
from nanotorch import Tensor
from nanotorch.nn import Conv2D

# Create Conv2D layer
conv = Conv2D(
    in_channels=3,      # RGB image
    out_channels=64,    # Output feature maps
    kernel_size=3,      # 3x3 kernel
    stride=1,
    padding=1           # Keep size unchanged
)

# Input: (batch_size, channels, height, width)
x = Tensor.randn((16, 3, 224, 224))

# Forward pass
output = conv(x)
print(output.shape)  # (16, 64, 224, 224)
```

### Simple Implementation (Educational)

To better understand convolution, here's a simplified implementation:

```python
def _conv2d_forward(self, x: Tensor, H_out: int, W_out: int) -> Tensor:
    """Naive convolution implementation (for educational purposes)"""
    N, C_in, H_in, W_in = x.shape
    C_out = self.out_channels
    K_H, K_W = self.kernel_size
    
    output = Tensor.zeros((N, C_out, H_out, W_out), requires_grad=x.requires_grad)
    
    # Sliding window convolution
    for n in range(N):           # Batch dimension
        for c_out in range(C_out):  # Output channels
            for h_out in range(H_out):  # Height
                for w_out in range(W_out):  # Width
                    # Calculate input window position
                    h_start = h_out * self.stride
                    w_start = w_out * self.stride
                    h_end = h_start + K_H
                    w_end = w_start + K_W
                    
                    # Extract input window
                    window = x.data[n, :, h_start:h_end, w_start:w_end]
                    
                    # Get corresponding weights
                    weight_slice = self.weight.data[c_out]
                    
                    # Compute convolution sum
                    conv_sum = np.sum(window * weight_slice)
                    
                    output.data[n, c_out, h_out, w_out] = conv_sum
    
    return output
```

---

## Conv3D Implementation

### Data Layout

Conv3D processes input of shape `(N, C_in, D, H, W)`:
- N: Batch size
- C_in: Number of input channels
- D: Depth (time frames or volumetric depth)
- H: Height
- W: Width

### Use Cases

- **Video Processing**: D is the temporal dimension
- **Medical Imaging**: CT/MRI volumetric data
- **3D Object Recognition**: Point clouds or voxel data

### Implementation Code

```python
class Conv3D(Module):
    """3D convolution layer.
    
    Shape:
        - Input: (N, C_in, D_in, H_in, W_in)
        - Output: (N, C_out, D_out, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        
        # Convert to triplets
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Weight shape: (out_channels, in_channels, kD, kH, kW)
        weight_shape = (out_channels, in_channels, *kernel_size)
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True
        )
        
        # ... bias initialization
```

---

## Transposed Convolution

### Concept

Transposed Convolution, also known as **Deconvolution** or **Fractionally-Strided Convolution**, is used for **upsampling** operations.

Unlike regular convolution, transposed convolution converts low-resolution feature maps to high-resolution outputs.

```
Regular Convolution (downsampling):    Transposed Convolution (upsampling):
[H, W] -> [H/2, W/2]                   [H, W] -> [2H, 2W]
```

### Output Size Calculation

For transposed convolution:

$$H_{out} = (H_{in} - 1) \times stride - 2 \times padding + dilation \times (kernel\_size - 1) + output\_padding + 1$$

### ConvTranspose2D Implementation

```python
class ConvTranspose2D(Module):
    """2D transposed convolution layer.
    
    Also known as deconvolution or fractionally-strided convolution.
    Used for upsampling in segmentation networks, GANs, etc.
    
    Shape:
        - Input: (N, C_in, H_in, W_in)
        - Output: (N, C_out, H_out, W_out)
    
    Note:
        Weight shape is (in_channels, out_channels, K_H, K_W) for ConvTranspose2D,
        different from Conv2D's (out_channels, in_channels, K_H, K_W).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        
        # Note: Weight shape is reversed from Conv2D!
        # ConvTranspose2D: (in_channels, out_channels, K_H, K_W)
        weight_shape = (in_channels, out_channels, kernel_size[0], kernel_size[1])
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True
        )
        # ...
```

### Usage Example

```python
from nanotorch.nn import ConvTranspose2D

# Upsampling layer
upsample = ConvTranspose2D(
    in_channels=256,
    out_channels=128,
    kernel_size=4,
    stride=2,
    padding=1
)

# Input: (N, C, H, W) -> Output: (N, C, 2H, 2W)
x = Tensor.randn((8, 256, 32, 32))
output = upsample(x)
print(output.shape)  # (8, 128, 64, 64)
```

---

## Gradient Computation

Gradient computation for convolution layers is a key part of automatic differentiation.

### Input Gradient

For output gradient $\frac{\partial L}{\partial Y}$, the input gradient is:

$$\frac{\partial L}{\partial X} = \text{conv}(\frac{\partial L}{\partial Y}, W^{flipped})$$

Where $W^{flipped}$ is the weight rotated 180 degrees.

### Weight Gradient

$$\frac{\partial L}{\partial W} = \text{conv}(X, \frac{\partial L}{\partial Y})$$

### Implementation in autograd.py

```python
# nanotorch/autograd.py

class Conv2DFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation):
        # Save for backward
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        
        # Execute forward convolution
        output = conv2d_forward(x.data, weight.data, bias, stride, padding, dilation)
        return Tensor(output, requires_grad=x.requires_grad or weight.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        
        # Compute input gradient
        grad_input = conv2d_backward_input(grad_output, weight.data, ctx.stride, ctx.padding)
        
        # Compute weight gradient
        grad_weight = conv2d_backward_weight(grad_output, x.data, ctx.stride, ctx.padding)
        
        return grad_input, grad_weight, None, None, None, None
```

---

## Usage Examples

### Building a Simple CNN

```python
from nanotorch import Tensor
from nanotorch.nn import Conv2D, BatchNorm2d, ReLU, MaxPool2d, Linear, Sequential

# Simple CNN model
class SimpleCNN:
    def __init__(self, num_classes=10):
        self.features = Sequential(
            Conv2D(3, 32, kernel_size=3, padding=1),   # 224 -> 224
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2, 2),                            # 224 -> 112
            
            Conv2D(32, 64, kernel_size=3, padding=1),  # 112 -> 112
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2, 2),                            # 112 -> 56
            
            Conv2D(64, 128, kernel_size=3, padding=1), # 56 -> 56
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(2, 2),                            # 56 -> 28
        )
        
        self.classifier = Sequential(
            Linear(128 * 28 * 28, 512),
            ReLU(),
            Linear(512, num_classes)
        )
    
    def __call__(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x
    
    def parameters(self):
        return self.features.parameters() + self.classifier.parameters()

# Use the model
model = SimpleCNN()
x = Tensor.randn((4, 3, 224, 224))
output = model(x)
print(output.shape)  # (4, 10)
```

### Using Different Parameters

```python
# Dilated Convolution - increase receptive field
conv_dilated = Conv2D(64, 64, kernel_size=3, dilation=2, padding=2)

# Large Stride Convolution - fast downsampling
conv_stride = Conv2D(64, 128, kernel_size=3, stride=2, padding=1)

# 1x1 Convolution - channel transformation
conv_1x1 = Conv2D(256, 64, kernel_size=1)
```

---

## Summary

This tutorial detailed the implementation of convolution layers in nanotorch:

1. **Conv1D**: Processing sequence data, shape `(N, C, L)`
2. **Conv2D**: Processing image data, shape `(N, C, H, W)`
3. **Conv3D**: Processing video/volumetric data, shape `(N, C, D, H, W)`
4. **ConvTranspose2D/3D**: Transposed convolution for upsampling

Key parameters:
- `kernel_size`: Convolution kernel size
- `stride`: Step size
- `padding`: Padding
- `dilation`: Dilation rate

### Next Steps

In [Tutorial 10: Normalization Layers](10-normalization.md), we will learn how to implement BatchNorm, LayerNorm, GroupNorm, and other normalization layers that are crucial for stable training.

---

**References**:
- [CS231n: Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)
- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)
