# Chapter 9: Convolution Layers

## How Do Human Eyes See the World...

You're admiring an oil painting.

You don't treat the entire painting as a pile of meaningless pixels. Your eyes focus: here's a stroke of red, there's a touch of blue; curves on the left, angles on the right.

You see **local features**—edges, colors, textures—and then your brain combines them into a complete picture.

Convolutional Neural Networks (CNNs) do the same thing.

It holds a small window—3×3 or 5×5 in size—and slides across the image. At each position, it asks: "Is there an edge here? A corner? A texture?"

```
The Wisdom of Convolution:

  One image = 224 × 224 × 3 = 150,528 numbers

  Don't process all at once
  Instead, scan block by block with a small window
  Extract local features, layer by layer abstraction

  Layer 1: Edges, corners
  Layer 2: Textures, shapes
  Layer 3: Parts, objects
  Layer 4: Complete cats, dogs, cars
```

**Convolution is the cornerstone of computer vision.** It teaches machines to understand images from local to global, just like humans.

---

## 9.1 Why Do We Need Convolution?

### Problem: Fully Connected Layers Are Clumsy with Images

```
A 224x224 color image:
  - Pixels: 224 × 224 × 3 = 150,528
  - Connected to 1000 neurons: 150,528 × 1000 = 150 million parameters!

Problems:
  1. Too many parameters, slow training
  2. Doesn't utilize spatial structure of images
  3. Won't recognize if position changes
```

### Solution: Two Magic Weapons of Convolution

```
1. Local connectivity:
   Each neuron only looks at a small region (e.g., 3×3)

2. Weight sharing:
   The same magnifying glass scans the entire image

Effect:
  - Parameters: 3 × 3 × 3 × 64 = 1,728 (tens of thousands fewer!)
  - Position invariance: Can recognize cat in top-left or bottom-right
```

---

## 9.2 How Does Convolution Work?

### Single Step Calculation

```
Input image (5×5):         Kernel (3×3):
                           ┌─────────┐
[1  2  3  4  5]            │ 1  0  1 │
[6  7  8  9  10]           │ 0  1  0 │
[11 12 13 14 15]           │ 1  0  1 │
[16 17 18 19 20]           └─────────┘
[21 22 23 24 25]

Step 1: Kernel covers top-left 3×3 region

[1  2  3]       [1  0  1]
[6  7  8]   ⊙   [0  1  0]   = 1×1 + 2×0 + 3×1 + 6×0 + 7×1 + 8×0 + 11×1 + 12×0 + 13×1
[11 12 13]      [1  0  1]

              = 1 + 0 + 3 + 0 + 7 + 0 + 11 + 0 + 13 = 35

⊙ = Element-wise multiply then sum
```

### Sliding Process

```
Kernel slides across the image:

Position 1:  Position 2:  ...  Position n:
┌─────┐      ┌─────┐           ┌─────┐
│1 2 3│      │  2 3│           │  3 4│
│6 7  │  →   │6 7  │  → ... →  │7 8  │
│11 12│      │11 12│           │12 13│
└─────┘      └─────┘           └─────┘
  ↓            ↓                 ↓
  35           40                45

Output: [35, 40, 45, ...]
```

### Convolution Diagram

```
Input (H×W)          Kernel (K×K)         Output (H'×W')

┌─────────────┐     ┌───────┐           ┌─────────┐
│             │     │ w w w │           │         │
│   ┌─────┐   │  ⊙  │ w w w │  =        │ o o o   │
│   │scan │   │     │ w w w │           │ o o o   │
│   └─────┘   │     └───────┘           │         │
│             │                         └─────────┘
└─────────────┘

After scanning → Output is a smaller image (feature map)
```

---

## 9.3 Key Parameters

### Stride

```
Stride = 1: Move 1 cell each time
Stride = 2: Move 2 cells each time

Stride=1:            Stride=2:
[1 2 3 4]            [1 2 3 4]
[● ● ○ ○]            [●   ○   ]
[○ ○ ● ●]            [    ●   ]
 Move 1               Move 2

Larger stride, smaller output
```

### Padding

```
Problem: Image shrinks after convolution
  Input 5×5 → Convolution 3×3 → Output 3×3

Solution: Pad zeros at edges

Original:       After padding:
[1 2 3]         [0 0 0 0 0]
[4 5 6]    →    [0 1 2 3 0]
[7 8 9]         [0 4 5 6 0]
                [0 7 8 9 0]
                [0 0 0 0 0]

After padding: Input 5×5 → Output still 5×5
```

### Output Size Calculation

```
Output size = (Input size + 2×padding - Kernel size) / stride + 1

Example:
  Input: 224×224
  Kernel: 3×3
  Padding: 1
  Stride: 1

  Output = (224 + 2×1 - 3) / 1 + 1 = 224

  → Size unchanged! This is a common configuration
```

---

## 9.4 Conv2D Implementation

### Data Layout

```
Conv2D input shape: (N, C, H, W)
  N = batch size (how many images at once)
  C = channels (RGB=3)
  H = height
  W = width

Example:
  16 RGB images, each 224×224
  Shape: (16, 3, 224, 224)
```

### Implementation Code

```python
class Conv2D(Module):
    """
    2D Convolution Layer

    Analogy:
      - Input: One or more images
      - Kernel: Multiple "magnifying glasses"
      - Output: Multiple feature maps (one per magnifying glass)

    Weight shape: (out_channels, in_channels, kernel_h, kernel_w)

    Example:
      in_channels=3 (RGB)
      out_channels=64 (64 different magnifying glasses)
      kernel_size=3 (3×3 magnifying glass)
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

        # Weights: 64 filters of 3×3×3 = 64×3×3×3 parameters
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

# Create convolution layer
conv = Conv2D(
    in_channels=3,      # RGB image
    out_channels=64,    # 64 feature maps
    kernel_size=3,      # 3×3 kernel
    stride=1,
    padding=1           # Keep size unchanged
)

# Input: (batch_size, channels, height, width)
x = Tensor.randn((16, 3, 224, 224))

# Forward pass
output = conv(x)
print(output.shape)  # (16, 64, 224, 224)
```

---

## 9.5 Naive Implementation (Educational)

```python
def conv2d_simple(x, weight, stride=1, padding=0):
    """
    Naive convolution implementation (for understanding)

    Four nested loops:
      - n: Which image
      - c_out: Which output channel
      - h: Sliding in height direction
      - w: Sliding in width direction
    """
    N, C_in, H_in, W_in = x.shape
    C_out, _, K_H, K_W = weight.shape

    # Calculate output size
    H_out = (H_in + 2*padding - K_H) // stride + 1
    W_out = (W_in + 2*padding - K_W) // stride + 1

    # Padding
    if padding > 0:
        x_padded = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)))
    else:
        x_padded = x

    output = np.zeros((N, C_out, H_out, W_out))

    # Sliding window
    for n in range(N):              # Each image
        for c_out in range(C_out):  # Each output channel
            for h in range(H_out):  # Slide in height
                for w in range(W_out):  # Slide in width
                    # Calculate window position
                    h_start = h * stride
                    w_start = w * stride

                    # Extract window
                    window = x_padded[n, :, h_start:h_start+K_H, w_start:w_start+K_W]

                    # Convolution calculation
                    output[n, c_out, h, w] = np.sum(window * weight[c_out])

    return output
```

---

## 9.6 Conv1D: Processing Sequences

```
Conv1D for sequence data (text, audio)

Input shape: (N, C, L)
  N = batch size
  C = channels (word vector dimension)
  L = length (sequence length)

Sliding direction: Only one dimension (left to right)

Text example:
  Input: "I love learning" (4 words)
  Kernel: 3-word window
  Sliding: [I love learning] → [love learning]
```

```python
class Conv1D(Module):
    """
    1D Convolution Layer

    Used for: Text classification, time series, audio processing

    Input: (N, C_in, L)
    Output: (N, C_out, L')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()

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
```

---

## 9.7 Conv3D: Processing Video

```
Conv3D for volumetric data (video, CT scans)

Input shape: (N, C, D, H, W)
  N = batch size
  C = channels
  D = depth (time/depth)
  H = height
  W = width

Video example:
  D = 16 frames
  Each frame 224×224
  Input: (N, 3, 16, 224, 224)

Sliding direction: Three dimensions (time + space)
```

```python
class Conv3D(Module):
    """
    3D Convolution Layer

    Used for: Video analysis, medical imaging, 3D object recognition

    Input: (N, C_in, D_in, H_in, W_in)
    Output: (N, C_out, D_out, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        # Weight shape: (out_channels, in_channels, kD, kH, kW)
        weight_shape = (out_channels, in_channels, *kernel_size)
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True
        )
```

---

## 9.8 Transposed Convolution: Upsampling

### What is Transposed Convolution?

```
Regular convolution: Downsampling (gets smaller)
  224×224 → 112×112

Transposed convolution: Upsampling (gets larger)
  112×112 → 224×224

Uses:
  - Image segmentation: Restore resolution
  - GANs: Generate large images
  - Super-resolution: Image enlargement
```

### Diagram

```
Regular convolution:              Transposed convolution:

┌─────────┐                      ┌───┐
│  Large  │  → Conv →            │Small│
└─────────┘                      └───┘

┌───┐                            ┌─────────┐
│Small│  → Transposed Conv →     │  Large  │
└───┘                            └─────────┘
```

### Implementation

```python
class ConvTranspose2D(Module):
    """
    2D Transposed Convolution (Deconvolution)

    Used for upsampling: Make low-resolution feature maps larger

    Note: Weight shape is opposite to Conv2D!
      Conv2D: (out_channels, in_channels, K_H, K_W)
      ConvTranspose2D: (in_channels, out_channels, K_H, K_W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: int = 2,      # Usually stride=2 (2x upscaling)
        padding: int = 1,
        output_padding: int = 0,
    ) -> None:
        super().__init__()

        # Note: Weight shape is opposite to Conv2D
        weight_shape = (in_channels, out_channels, kernel_size[0], kernel_size[1])
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True
        )
```

### Usage

```python
# Upsampling layer: 32×32 → 64×64
upsample = ConvTranspose2D(
    in_channels=256,
    out_channels=128,
    kernel_size=4,
    stride=2,
    padding=1
)

x = Tensor.randn((8, 256, 32, 32))
output = upsample(x)
print(output.shape)  # (8, 128, 64, 64)
```

---

## 9.9 Special Convolutions

### 1×1 Convolution

```
1×1 convolution: Only changes channel count, not spatial size

Uses:
  - Dimension reduction/increase
  - Add non-linearity
  - Channel mixing

Example:
  Input: (N, 512, 28, 28)
  1×1 convolution: (N, 64, 28, 28)
  Parameters: 512×64 = 32,768 (very few!)
```

```python
conv_1x1 = Conv2D(512, 64, kernel_size=1)  # Dimension reduction
```

### Dilated Convolution

```
Dilated convolution: Kernel "skips" some positions

Regular 3×3:           Dilated 3×3 (dilation=2):
┌─────────┐           ┌───────────────┐
│ w w w │             │ w   w   w     │
│ w w w │             │   \   /       │
│ w w w │             │ w   w   w     │
└─────────┘           │   /   \       │
                      │ w   w   w     │
Receptive field: 3×3   └───────────────┘
                      Receptive field: 5×5

Benefit: Larger receptive field without more parameters
```

```python
conv_dilated = Conv2D(64, 64, kernel_size=3, dilation=2, padding=2)
```

### Depthwise Separable Convolution

```
Regular convolution: Each filter looks at all channels
Depthwise separable: Each channel looked at separately

Regular convolution params: 3×3×3×64 = 1,728
Depthwise separable params: 3×3×3 + 1×1×3×64 = 27 + 192 = 219

8× fewer!
```

---

## 9.10 Gradient Computation

### Input Gradient

```
Backpropagation: Given ∂L/∂output, find ∂L/∂input

∂L/∂input = conv(∂L/∂output, W_flipped)

W_flipped = Weight rotated 180 degrees
```

### Weight Gradient

```
∂L/∂W = conv(input, ∂L/∂output)

Convolve input with output gradient to get weight gradient
```

---

## 9.11 Building a CNN

```python
from nanotorch import Tensor
from nanotorch.nn import Conv2D, BatchNorm2d, ReLU, MaxPool2d, Linear, Sequential

class SimpleCNN:
    """
    Simple CNN classifier

    Structure:
      Conv → BN → ReLU → Pool (repeat 3 times)
      Flatten → FC → ReLU → FC
    """

    def __init__(self, num_classes=10):
        self.features = Sequential(
            # Block 1: 224 → 112
            Conv2D(3, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2, 2),

            # Block 2: 112 → 56
            Conv2D(32, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2, 2),

            # Block 3: 56 → 28
            Conv2D(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(2, 2),
        )

        self.classifier = Sequential(
            Linear(128 * 28 * 28, 512),
            ReLU(),
            Linear(512, num_classes)
        )

    def __call__(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)  # Flatten
        x = self.classifier(x)
        return x

    def parameters(self):
        return self.features.parameters() + self.classifier.parameters()

# Usage
model = SimpleCNN()
x = Tensor.randn((4, 3, 224, 224))
output = model(x)
print(output.shape)  # (4, 10)
```

---

## 9.12 Common Pitfalls

### Pitfall 1: Wrong padding causes size mismatch

```python
# Problem: 224 / 2 = 112, but 225 / 2 = 112.5
# Solution: Ensure input size is divisible by stride

# Correct configuration
Conv2D(64, 128, kernel_size=3, stride=2, padding=1)
# 224 → 112 (exactly divisible)
```

### Pitfall 2: Wrong channel configuration

```python
# Wrong: Channel count mismatch
Conv2D(3, 64, ...)   # Outputs 64 channels
Conv2D(32, 128, ...) # Expects 32 input channels ← Wrong!

# Correct
Conv2D(3, 64, ...)    # Outputs 64 channels
Conv2D(64, 128, ...)  # Expects 64 input channels ← Correct!
```

### Pitfall 3: Forgetting bias

```python
# In some cases, bias is not needed
Conv2D(..., bias=False)  # Can skip bias when followed by BatchNorm
```

---

## 9.13 Convolution Layer Comparison

| Layer | Input Dims | Use Case | Example |
|-------|------------|----------|---------|
| Conv1D | (N,C,L) | Sequence | Text, Audio |
| Conv2D | (N,C,H,W) | Image | Image classification |
| Conv3D | (N,C,D,H,W) | Volume | Video, CT |
| ConvTranspose2D | (N,C,H,W) | Upsampling | Segmentation, GAN |

---

## 9.14 Exercises

### Basic Exercises

1. Manually calculate the output of a 3×3 convolution

2. Implement convolution with `kernel_size=5, stride=2, padding=2`, calculate output size

3. Implement `AvgPool2d` (average pooling)

### Advanced Exercises

4. Implement depthwise separable convolution

5. Implement grouped convolution (groups>1)

---

## Summary in One Sentence

| Concept | One Sentence |
|---------|--------------|
| Convolution | Scanning image with a magnifying glass |
| Kernel | That magnifying glass (template for finding features) |
| Stride | How many cells to move each time |
| Padding | Pad zeros at edges to maintain size |
| Conv2D | Convolution for processing images |
| Transposed Conv | Reverse convolution, for upsampling |

---

## Next Chapter

Now we've learned convolution!

In the next chapter, we'll learn about **normalization layers** — the secret to stable training.

→ [Chapter 10: Normalization Layers](10-normalization.md)

```python
# Preview: What you'll learn next chapter
BatchNorm2d(64)   # Batch normalization
LayerNorm(768)    # Layer normalization (for Transformers)
GroupNorm(32, 64) # Group normalization
```
