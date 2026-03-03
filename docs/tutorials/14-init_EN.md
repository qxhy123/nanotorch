# Tutorial 14: Parameter Initialization

## Everything is Hard at the Beginning, Neural Networks Too...

Before training starts, neural network parameters are all random.

But "random" and "random" are not the same.

If all set to zero, all neurons are doing the same thing, learning the same things, and end up exactly the same. This is called the "symmetry problem" — like a hundred students copying the same homework, no one learns anything.

If randomized too extremely, values may be too large or too small. Too large causes gradient explosion, too small causes gradient vanishing. Training fails before it even begins.

**Good initialization finds the balance between "diversity" and "numerical stability".**

```
The Art of Initialization:

  All-zero initialization:
    W = 0
    All neurons → Same output → Same gradient → Same update
    A hundred layers equal one layer, tragedy

  Xavier initialization:
    W ~ N(0, 2/(n_in + n_out))
    Keeps forward propagation variance stable
    Suitable for Sigmoid, Tanh

  Kaiming initialization:
    W ~ N(0, 2/n_in)
    Designed specifically for ReLU
    It's the standard for modern deep networks
```

**Initialization is the starting line of training.** A good start makes the rest run fast.

---

## 14.1 Why is Initialization Important?

### Problem 1: All-Zero Initialization

```python
# Wrong: All zeros
W = np.zeros((128, 64))

Problem:
  - All neurons output the same
  - All gradients are the same
  - After update, all weights are still the same
  → Symmetry cannot be broken, network learns nothing
```

### Problem 2: Weights Too Large

```python
# Problem: Values too large
W = np.random.randn(128, 64) * 10

Forward propagation:
  Output = Input × W → Very large numbers
  Through activation function → Saturated (output close to 0 or 1)
  Gradient → Close to 0
→ Gradient vanishing, cannot learn
```

### Problem 3: Weights Too Small

```python
# Problem: Values too small
W = np.random.randn(128, 64) * 0.001

Forward propagation:
  Output = Input × W → Very small numbers
  Propagating through layers → Getting smaller and smaller
  Gradient → Also getting smaller and smaller
→ Gradient vanishing, cannot learn
```

### Solution: Appropriate Initialization

```
Good initialization:

Weights not too large, not too small, just right so that:
  - Forward propagation: Signal doesn't decay
  - Backward propagation: Gradient doesn't vanish

Analogy: Archery
  - Too close: Arrow doesn't reach the target center
  - Too far: Arrow can't reach the target
  - Just right: Arrow can cover the target face
```

---

## 14.2 Xavier Initialization

### Principle

```
Xavier initialization is suitable for Tanh/Sigmoid activations

Goal: Keep variance unchanged during forward and backward propagation

Derivation:
  Input x, weight W, output y = W @ x

  Var(y) = n_in × Var(W) × Var(x)

  To make Var(y) = Var(x):
  Var(W) = 1 / n_in

  Same for backward propagation:
  Var(W) = 1 / n_out

  Compromise:
  Var(W) = 2 / (n_in + n_out)
```

### Formula

```
Xavier uniform distribution:
  W ~ U(-a, a)
  a = sqrt(6 / (n_in + n_out))

Xavier normal distribution:
  W ~ N(0, std²)
  std = sqrt(2 / (n_in + n_out))
```

### Implementation

```python
def xavier_normal_(tensor, gain=1.0):
    """
    Xavier normal initialization

    Suitable for: Tanh, Sigmoid activation functions
    """
    # Calculate fan_in and fan_out
    if tensor.ndim == 2:
        # Linear: (out_features, in_features)
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        # Conv: (out_channels, in_channels, *kernel)
        receptive_field = np.prod(tensor.shape[2:])
        fan_in = tensor.shape[1] * receptive_field
        fan_out = tensor.shape[0] * receptive_field

    std = gain * np.sqrt(2.0 / (fan_in + fan_out))

    tensor.data = np.random.normal(0, std, tensor.shape).astype(np.float32)
    return tensor
```

### Usage

```python
from nanotorch.nn import Linear, Tanh
from nanotorch.utils import xavier_normal_

# Xavier suitable for Tanh
linear = Linear(128, 64)
xavier_normal_(linear.weight)  # Initialize weights
```

---

## 14.3 Kaiming Initialization

### Principle

```
Kaiming initialization is suitable for ReLU family activations

ReLU sets half of inputs to 0, so extra compensation is needed

Var(y) = (n_in/2) × Var(W) × Var(x)

To make Var(y) = Var(x):
Var(W) = 2 / n_in
```

### Formula

```
Kaiming uniform distribution:
  W ~ U(-a, a)
  a = sqrt(6 / n_in)  (for ReLU)

Kaiming normal distribution:
  W ~ N(0, std²)
  std = sqrt(2 / n_in)  (for ReLU)
```

### Implementation

```python
def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    """
    Kaiming normal initialization

    Suitable for: ReLU, LeakyReLU activation functions

    Args:
        a: LeakyReLU negative slope
        mode: 'fan_in' or 'fan_out'
        nonlinearity: Activation function type
    """
    # Calculate fan_in and fan_out
    if tensor.ndim == 2:
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        receptive_field = np.prod(tensor.shape[2:])
        fan_in = tensor.shape[1] * receptive_field
        fan_out = tensor.shape[0] * receptive_field

    fan = fan_in if mode == 'fan_in' else fan_out

    # Calculate gain
    if nonlinearity == 'relu':
        gain = np.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        gain = np.sqrt(2.0 / (1 + a**2))
    else:
        gain = 1.0

    std = gain / np.sqrt(fan)

    tensor.data = np.random.normal(0, std, tensor.shape).astype(np.float32)
    return tensor
```

### Usage

```python
from nanotorch.nn import Linear, ReLU, Conv2D
from nanotorch.utils import kaiming_normal_, zeros_

# Kaiming suitable for ReLU
linear = Linear(128, 64)
kaiming_normal_(linear.weight, nonlinearity='relu')
zeros_(linear.bias)  # Bias initialized to 0

# Convolutional layer also uses Kaiming
conv = Conv2D(3, 64, kernel_size=3)
kaiming_normal_(conv.weight, nonlinearity='relu')
zeros_(conv.bias)
```

---

## 14.4 Initialization Selection Guide

### By Activation Function

```
Activation Function     Recommended Initialization
─────────────────────────────────────────────────
ReLU                   Kaiming
LeakyReLU              Kaiming
PReLU                  Kaiming
Tanh                   Xavier
Sigmoid                Xavier
GELU                   Xavier or Kaiming
```

### By Network Type

```
Network Type           Recommended Initialization
─────────────────────────────────────────────────
CNN (with ReLU)        Kaiming
RNN/LSTM               Orthogonal initialization
Transformer            Truncated normal (std=0.02)
ResNet                 Kaiming + special handling
```

---

## 14.5 Other Initialization Methods

### Orthogonal Initialization

```python
def orthogonal_(tensor, gain=1.0):
    """
    Orthogonal matrix initialization

    Suitable for: RNN, deep networks
    Benefit: Prevents gradient vanishing/explosion
    """
    # Generate random matrix
    flat_shape = (tensor.shape[0], np.prod(tensor.shape[1:]))
    random_matrix = np.random.randn(*flat_shape)

    # QR decomposition to get orthogonal matrix
    q, r = np.linalg.qr(random_matrix)

    # Apply gain
    tensor.data = (q * gain).reshape(tensor.shape).astype(np.float32)
    return tensor
```

### Truncated Normal

```python
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    Truncated normal distribution initialization

    Suitable for: Transformer (std=0.02)
    Benefit: Avoids extreme values
    """
    lower = mean + a * std
    upper = mean + b * std

    # Sample until all within range
    data = np.random.normal(mean, std, tensor.shape)
    while np.any((data < lower) | (data > upper)):
        mask = (data < lower) | (data > upper)
        data[mask] = np.random.normal(mean, std, mask.sum())

    tensor.data = data.astype(np.float32)
    return tensor
```

---

## 14.6 Bias Initialization

```python
# Bias usually initialized to 0
bias = np.zeros(out_features)

# Special cases:
# BatchNorm/LayerNorm gamma = 1, beta = 0
gamma = np.ones(num_features)
beta = np.zeros(num_features)
```

---

## 14.7 Complete Example

```python
from nanotorch.nn import Linear, ReLU, Conv2D, BatchNorm2d, Sequential
from nanotorch.utils import kaiming_normal_, zeros_, ones_

def init_weights(model):
    """Unified initialization function"""
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                zeros_(module.bias)

        elif isinstance(module, Conv2D):
            kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                zeros_(module.bias)

        elif isinstance(module, BatchNorm2d):
            ones_(module.weight)
            zeros_(module.bias)

# Usage
model = Sequential(
    Conv2D(3, 64, 3),
    BatchNorm2d(64),
    ReLU(),
    Linear(64, 10)
)

init_weights(model)
```

---

## 14.8 Common Pitfalls

### Pitfall 1: Forgetting to Initialize

```python
# Problem: Using default random initialization
linear = Linear(128, 64)
# Weights are random, may not be appropriate

# Correct: Explicit initialization
linear = Linear(128, 64)
kaiming_normal_(linear.weight, nonlinearity='relu')
```

### Pitfall 2: Activation Function and Initialization Mismatch

```python
# Wrong: ReLU with Xavier
linear = Linear(128, 64)
xavier_normal_(linear.weight)  # Mismatch!
relu = ReLU()

# Correct: ReLU with Kaiming
kaiming_normal_(linear.weight, nonlinearity='relu')
```

### Pitfall 3: Complex Initialization for Bias Too

```python
# Unnecessary
kaiming_normal_(linear.bias)  # Bias doesn't need this

# Simple is fine
zeros_(linear.bias)  # Usually 0
```

---

## 14.9 One-Line Summary

| Activation Function | Initialization | Reason |
|---------------------|----------------|--------|
| ReLU | Kaiming | Compensate for half being zeroed |
| Tanh | Xavier | Maintain variance |
| RNN | Orthogonal | Prevent gradient vanishing |
| Transformer | Truncated normal | Avoid extreme values |

```
Simple memory:
  ReLU → Kaiming
  Tanh → Xavier
  Bias → Zero
```

---

## Next Chapter

Now we've learned initialization!

Next chapter, we'll learn **advanced topics** — gradient clipping, learning rate warmup, and other practical techniques.

→ [Chapter 15: Advanced Topics](15-advanced.md)

```python
# Preview: What you'll learn in the next chapter
clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
# Prevents gradient explosion
```
