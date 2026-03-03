# Tutorial 10: Normalization Layers

## The Harmony Secret of a Choir...

Everyone in the choir is singing. But here's the problem—

The tenor is too loud, drowning out the alto.
The mezzo-soprano is pitchy, throwing off the whole section.
Some voices are unsteady, now strong, now weak.

If everyone follows their own rhythm, the result can only be noise.

So the conductor steps forward. He constantly adjusts: "Tenor, a little softer. Alto, louder. Overall, more steady."

**Normalization layers are the "conductors" of neural networks.**

In deep networks, each layer's output may deviate from the "right track"—some values explode, others vanish. Subsequent layers become confused: "What should I learn?"

Normalization layers "pull back" the data at every step, returning them to a stable distribution.

```
Without normalization:
  Layer 1 output: [0.001, 0.002, ...]
  Layer 2 output: [0.0001, 0.0002, ...]
  Layer 3 output: [0.00001, ...]  ← Vanished

With normalization:
  Every layer output: mean ≈ 0, variance ≈ 1
  Stable, controllable, learnable
```

**Normalization is the stabilizer of deep networks.** It makes training more stable and convergence faster.

---

## 10.1 Why Do We Need Normalization?

### Problem: Internal Covariate Shift

```
The Dilemma of Deep Networks:

Layer 1 output → Layer 2 → Layer 3 → ... → Layer N
      ↓             ↓           ↓
  Distribution   Distribution   Distribution
    change         change         change

Problem:
  - Each layer's "input distribution" constantly changes
  - Later layers must constantly adapt
  - Training is unstable, convergence is slow
```

### Life Analogy

```
Like taking exams:

Scenario 1: Each exam has different difficulty
  1st exam: Very easy, average 90
  2nd exam: Very hard, average 50
  3rd exam: Easy, average 85
  → Hard to adjust your study strategy

Scenario 2: Each exam is standardized
  1st exam: After standardization, average 70
  2nd exam: After standardization, average 70
  3rd exam: After standardization, average 70
  → Easy to see your progress
```

### Solution: Normalization

```
Effect of normalization:

Input [very large numbers, very small numbers, ...]
        ↓
   Subtract mean, divide by standard deviation
        ↓
Output [numbers close to 0, numbers close to 0, ...]

Each layer's input is stable → Training is more stable
```

---

## 10.2 BatchNorm: Normalize by Batch

### Principle

```
BatchNorm calculates statistics for each channel within the batch:

Input: (N, C, H, W) = (16, 64, 32, 32)
      Batch  Channels Height Width

For each channel c:
  - Calculate mean and variance of these 16 images on channel c
  - Normalize using these statistics

Intuitive understanding:
  Suppose channel 1 represents "red"
  → Look at the average brightness of red in these 16 images
  → Adjust to standard brightness
```

### Diagram

```
Input data (N=4, C=3):

Channel 0 (Red):      Channel 1 (Green):   Channel 2 (Blue):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 255 200 180 │     │ 100 120 90  │     │ 50  60  40  │
│ 220 190 210 │     │ 110 100 130 │     │ 55  45  70  │
│ ...         │     │ ...         │     │ ...         │
└─────────────┘     └─────────────┘     └─────────────┘
      ↓                   ↓                   ↓
  mean=200           mean=110            mean=50
  std=30             std=15              std=10
      ↓                   ↓                   ↓
   Normalize          Normalize           Normalize
```

### Training vs Inference

```
During training:
  - Use current batch mean/variance
  - Simultaneously update running_mean/running_var

During inference:
  - Use running_mean/running_var accumulated during training
  - Because inference may have only 1 sample, cannot calculate statistics
```

### Implementation

```python
class BatchNorm2d(Module):
    """
    2D Batch Normalization

    Analogy:
      Training = Live conducting, adjusting based on current situation
      Inference = Using past experience to adjust
    """

    def __init__(
        self,
        num_features: int,      # Number of channels
        eps: float = 1e-5,      # Prevent division by 0
        momentum: float = 0.1,  # Running statistics update speed
        affine: bool = True,    # Whether to learn gamma and beta
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters: scale and shift
        if affine:
            self.gamma = Tensor.ones((num_features,), requires_grad=True)  # Scale
            self.beta = Tensor.zeros((num_features,), requires_grad=True)  # Shift

        # Running statistics (not parameters, no gradients)
        self.running_mean = Tensor.zeros((num_features,))
        self.running_var = Tensor.ones((num_features,))

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.shape

        if self.training:
            # Training mode: use current batch statistics
            # Calculate mean and variance over N, H, W dimensions
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)

            # Update running statistics
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + \
                                     self.momentum * mean.squeeze().data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + \
                                    self.momentum * var.squeeze().data
        else:
            # Inference mode: use running statistics
            mean = self.running_mean.reshape(1, C, 1, 1)
            var = self.running_var.reshape(1, C, 1, 1)

        # Normalize
        x_norm = (x - mean) / (var + self.eps) ** 0.5

        # Scale and shift (let the network decide the best distribution)
        if hasattr(self, 'gamma'):
            x_norm = self.gamma.reshape(1, C, 1, 1) * x_norm + \
                     self.beta.reshape(1, C, 1, 1)

        return x_norm
```

### Usage

```python
from nanotorch.nn import BatchNorm2d

# Create BatchNorm
bn = BatchNorm2d(num_features=64)  # 64 channels

# Training
bn.train()
output = bn(x_train)

# Inference
bn.eval()
output = bn(x_test)
```

---

## 10.3 LayerNorm: Normalize by Layer

### Principle

```
LayerNorm normalizes all features of each sample:

Input: (N, C, H, W)
For each sample n:
  - Calculate mean and variance of all features in this sample
  - Does not depend on batch size

Intuitive understanding:
  Each person adjusts themselves, regardless of others
  I look at my own scores in various subjects, adjust to average
```

### BatchNorm vs LayerNorm

```
BatchNorm (across samples):    LayerNorm (across features):

┌─────────────────┐            ┌─────────────────┐
│ Sample1 │ Sample2  │          │ Sample1         │
│  ↓    │  ↓      │          │  All features    │
│ Stats │ Stats   │          │  together       │
└─────────────────┘          │  ↓              │
                             │  Stats          │
                             └─────────────────┘
Each channel separately       Each sample separately

Needs large batch             Doesn't depend on batch
```

### Implementation

```python
class LayerNorm(Module):
    """
    Layer Normalization

    Commonly used in: Transformers, NLP tasks
    Advantages: Doesn't depend on batch size, suitable for variable-length sequences
    """

    def __init__(
        self,
        normalized_shape: int,   # Size of last dimension
        eps: float = 1e-5,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # Learnable parameters
        self.gamma = Tensor.ones((normalized_shape,), requires_grad=True)
        self.beta = Tensor.zeros((normalized_shape,), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, hidden_dim)
        # Normalize over the last dimension

        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)

        x_norm = (x - mean) / (var + self.eps) ** 0.5

        return self.gamma * x_norm + self.beta
```

### Usage

```python
from nanotorch.nn import LayerNorm

# Commonly used in Transformers
ln = LayerNorm(normalized_shape=512)  # hidden_dim = 512

x = Tensor.randn((32, 100, 512))  # (batch, seq_len, hidden_dim)
output = ln(x)
```

---

## 10.4 GroupNorm: Normalize by Groups

### Principle

```
GroupNorm divides channels into groups, normalizes within each group:

Input: (N, C, H, W), C=32, divide into 8 groups
→ Each group has 4 channels
→ Calculate statistics over 4 channels + H + W in each group

Between LayerNorm and InstanceNorm
```

### Visual Comparison

```
Input shape: (N, C, H, W), assuming C=6

BatchNorm:  For each channel, across (N, H, W) statistics
            Channel 1: [All pixels of channel 1 from all samples]
            Channel 2: [All pixels of channel 2 from all samples]
            ...

LayerNorm:  For each sample, across (C, H, W) statistics
            Sample 1: [All pixels of all channels]
            Sample 2: [All pixels of all channels]
            ...

GroupNorm:  For each group of each sample, across (C/G, H, W) statistics
            Sample 1-Group 1: [All pixels of channels 1,2]
            Sample 1-Group 2: [All pixels of channels 3,4]
            Sample 1-Group 3: [All pixels of channels 5,6]
            ...

InstanceNorm: For each channel of each sample, across (H, W) statistics
            Sample 1-Channel 1: [All pixels of channel 1]
            Sample 1-Channel 2: [All pixels of channel 2]
            ...
```

### Implementation

```python
class GroupNorm(Module):
    """
    Group Normalization

    Advantages: Doesn't depend on batch size
    Commonly used in: Object detection, segmentation (batch usually small)
    """

    def __init__(
        self,
        num_groups: int,      # Number of groups
        num_channels: int,    # Number of channels (must be divisible by num_groups)
        eps: float = 1e-5,
    ):
        super().__init__()
        assert num_channels % num_groups == 0

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.gamma = Tensor.ones((num_channels,), requires_grad=True)
        self.beta = Tensor.zeros((num_channels,), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        G = self.num_groups

        # Reshape to (N, G, C/G, H, W)
        x = x.reshape(N, G, C // G, H, W)

        # Calculate statistics over (C/G, H, W) dimensions
        mean = x.mean(axis=(2, 3, 4), keepdims=True)
        var = ((x - mean) ** 2).mean(axis=(2, 3, 4), keepdims=True)

        x_norm = (x - mean) / (var + self.eps) ** 0.5

        # Restore shape
        x_norm = x_norm.reshape(N, C, H, W)

        return self.gamma.reshape(1, C, 1, 1) * x_norm + \
               self.beta.reshape(1, C, 1, 1)
```

### Usage

```python
from nanotorch.nn import GroupNorm

# Divide 32 channels into 8 groups
gn = GroupNorm(num_groups=8, num_channels=32)

x = Tensor.randn((16, 32, 64, 64))
output = gn(x)

# Special cases:
# GroupNorm(num_groups=1, ...) ≈ LayerNorm
# GroupNorm(num_groups=C, ...) = InstanceNorm
```

---

## 10.5 Comparison of Normalization Methods

### Selection Guide

```
Scenario                          Recommended
──────────────────────────────────────
CNN image classification, large batch    BatchNorm
CNN, small batch (<8)                    GroupNorm
Transformer / NLP                        LayerNorm
Style transfer                           InstanceNorm
Object detection/segmentation            GroupNorm
```

### Comparison Table

| Normalization | Statistics Dimensions | Depends on batch | Use case |
|--------|---------|-----------|----------|
| BatchNorm | (N, H, W) | Yes | CNN, large batches |
| LayerNorm | (C, H, W) | No | Transformer |
| GroupNorm | (C/G, H, W) | No | Small batches |
| InstanceNorm | (H, W) | No | Style transfer |

### Visualization

```
Assuming input is a tensor of (N=3, C=4, H=2, W=2)

Different colored areas represent elements normalized together:

BatchNorm (per channel):
  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
  │ C=0   │  │ C=1   │  │ C=2   │  │ C=3   │
  │ across│  │ across│  │ across│  │ across│
  │ NHW   │  │ NHW   │  │ NHW   │  │ NHW   │
  └───────┘  └───────┘  └───────┘  └───────┘

LayerNorm (per sample):
  ┌─────────────────────────────────────┐
  │ Sample 0: across C, H, W            │
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │ Sample 1: across C, H, W            │
  └─────────────────────────────────────┘

GroupNorm (per sample per group, assuming 2 groups):
  ┌─────────────┐  ┌─────────────┐
  │ Sample 0    │  │ Sample 0    │
  │ Group 0     │  │ Group 1     │
  │ C=0,1       │  │ C=2,3       │
  └─────────────┘  └─────────────┘
```

---

## 10.6 Common Pitfalls

### Pitfall 1: Training/Inference Mode Confusion

```python
# Wrong: Not switching mode during inference
model.eval()
output = bn(x)  # If bn is still in training mode, will use current batch statistics

# Correct
bn.eval()  # Set individually
# or
model.eval()  # Set entire model
```

### Pitfall 2: Using BatchNorm with Small Batches

```python
# Problem: When batch=1 or 2, statistics are unstable
bn = BatchNorm2d(64)
x = Tensor.randn((2, 64, 32, 32))  # batch=2 is too small

# Solution: Use GroupNorm
gn = GroupNorm(num_groups=32, num_channels=64)
```

### Pitfall 3: Using Dropout After BatchNorm

```python
# Usually not needed, BatchNorm itself has regularization effect
model = Sequential(
    Conv2D(64, 128),
    BatchNorm2d(128),
    # Dropout(0.5),  # Usually not needed
    ReLU(),
)
```

---

## 10.7 Summary in One Sentence

| Concept | One Sentence |
|------|--------|
| Normalization | Makes each layer's output stable, accelerates training |
| BatchNorm | Statistics by batch, suitable for large batches |
| LayerNorm | Statistics by sample, suitable for Transformers |
| GroupNorm | Statistics by group, suitable for small batches |
| Training/Inference | Training uses current statistics, inference uses historical statistics |

---

## Next Chapter

Now we've learned normalization!

Next chapter, we'll learn **Recurrent Neural Networks** — the classic architecture for processing sequential data.

→ [Chapter 11: Recurrent Neural Networks](11-rnn.md)

```python
# Preview: What you'll learn in the next chapter
lstm = LSTM(input_size=64, hidden_size=128)
# Remember historical information, process variable-length sequences
```
