# Chapter 6(c): ReLU Function Family and Activation Functions

ReLU (Rectified Linear Unit) is the most commonly used activation function in deep learning. It is simple, efficient, and effectively solves the vanishing gradient problem. This section will systematically introduce ReLU and its variants, as well as how to choose appropriate activation functions.

---

## 🎯 Life Analogy: ReLU is Like a Volume Knob

```
ReLU: "If it's negative, silence it. If positive, let it through."

Input:  -5  -2   0   2   5
         ↓   ↓   ↓   ↓   ↓
ReLU:    0   0   0   2   5

Like a volume knob that can't go below zero!
```

### The "Dead Neuron" Problem

```
Regular ReLU:
Input:  -10  -5   0   5   10
Output:   0   0   0   5   10
          ↑
    "Dead" - always outputs 0, never learns

Leaky ReLU:
Input:  -10   -5    0    5   10
Output: -0.1 -0.05   0    5   10
          ↑
    Still alive! Small gradient allows recovery
```

### 📖 Activation Functions Comparison

| Function | Formula | Pros | Cons |
|----------|---------|------|------|
| **ReLU** | $\max(0, x)$ | Fast, no vanishing gradient | Dead neurons |
| **Leaky ReLU** | $\max(0.01x, x)$ | No dead neurons | Another hyperparameter |
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | Output is probability | Vanishing gradients |
| **Tanh** | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | Zero-centered | Vanishing gradients |
| **GELU** | $x \cdot \Phi(x)$ | Smooth, works well in Transformers | Slower to compute |

### Plain English Translation

| Term | Plain English |
|------|---------------|
| Activation function | Decides if a neuron "fires" |
| Vanishing gradient | Gradient gets too small to learn |
| Dead neuron | Neuron that never activates (always outputs 0) |

---

## Table of Contents

1. [ReLU Function](#relu-function)
2. [Leaky ReLU and PReLU](#leaky-relu-and-prelu)
3. [ELU and SELU](#elu-and-selu)
4. [GELU](#gelu)
5. [Activation Function Selection Guide](#activation-function-selection-guide)
6. [Applications in Deep Learning](#applications-in-deep-learning)

---

## ReLU Function

### Definition

**ReLU** (Rectified Linear Unit):

$$
\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

### Core Properties

1. **Derivative**:

$$
\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

2. **Non-negativity**: output $\geq 0$

3. **Sparse activation**: output is 0 when input is negative, producing sparse representation

4. **Computational efficiency**: only needs comparison and max operation, no exponential computation

### Advantages of ReLU

| Advantage | Description |
|-----------|-------------|
| Simple computation | Only needs max operation |
| No vanishing gradient (positive region) | Gradient is always 1 |
| Sparse activation | Negative inputs are suppressed |
| Fast convergence | Constant gradient, no saturation |

### ReLU Problem: Dead ReLU

**Dead ReLU problem**: If a neuron's input is always negative, the neuron will always output 0, the gradient will always be 0, and parameters can never update.

**Causes**:
- Learning rate too large
- Improper weight initialization
- Data distribution issues

```python
import numpy as np

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

# Example
x = np.array([-2, -1, 0, 1, 2])
print(f"ReLU: {relu(x)}")
print(f"Derivative: {relu_derivative(x)}")
```

---

## Leaky ReLU and PReLU

### Leaky ReLU

**Definition**: Give a small slope $\alpha$ in the negative region (usually $\alpha = 0.01$):

$$
\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}
$$

**Derivative**:

$$
\text{LeakyReLU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}
$$

**Advantage**: Avoids "dead ReLU" problem, gradient still exists in negative region.

```python
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Derivative of Leaky ReLU"""
    return np.where(x > 0, 1, alpha)
```

### PReLU (Parametric ReLU)

**Definition**: $\alpha$ is a learnable parameter:

$$
\text{PReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}
$$

**Advantages**:
- Adaptively learns optimal negative region slope
- Performs well on large datasets

**Disadvantages**:
- May overfit (increases parameters)
- Less stable than Leaky ReLU on small datasets

```python
class PReLU:
    """PReLU activation function"""

    def __init__(self, shape, alpha=0.25):
        self.alpha = np.full(shape, alpha)  # Learnable parameter

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, x, grad_output):
        # Gradient with respect to x
        grad_x = np.where(x > 0, grad_output, grad_output * self.alpha)

        # Gradient with respect to alpha
        grad_alpha = np.where(x <= 0, x * grad_output, 0).sum(axis=0)

        return grad_x, grad_alpha
```

---

## ELU and SELU

### ELU (Exponential Linear Unit)

**Definition**:

$$
\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}
$$

Where $\alpha > 0$ (usually take 1.0).

**Derivative**:

$$
\text{ELU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha e^x = \text{ELU}(x) + \alpha & x \leq 0 \end{cases}
$$

**Advantages**:
- Output mean close to 0 (self-normalizing)
- Smooth in negative region (unlike ReLU's kink)
- More robust to noise

**Disadvantages**:
- Computes exponentials in negative region, slightly slower

```python
def elu(x, alpha=1.0):
    """ELU activation function"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    """Derivative of ELU"""
    return np.where(x > 0, 1, elu(x, alpha) + alpha)
```

### SELU (Scaled ELU)

**Definition**: Add scaling factor to ELU:

$$
\text{SELU}(x) = \lambda \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}
$$

Where:
- $\alpha \approx 1.6732632423543772$
- $\lambda \approx 1.0507009873554805$

**Self-normalizing property**: With specific initialization (LeCun Normal), network layer outputs can automatically normalize to mean 0 and variance 1.

```python
def selu(x, alpha=1.6732632423543772, scale=1.0507009873554805):
    """SELU activation function"""
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

---

## GELU

### Definition

**GELU** (Gaussian Error Linear Unit):

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(Z \leq x)
$$

Where $\Phi(x)$ is the CDF of standard normal distribution.

### Approximation Formulas

**Tanh approximation** (most commonly used):

$$
\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right]\right)
$$

**Sigmoid approximation**:

$$
\text{GELU}(x) \approx x \cdot \sigma(1.702x)
$$

**Derivation of GELU Tanh approximation formula**:

**Step 1**: Start from the exact form.

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

Where $\text{erf}(x) = \frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2} dt$

**Step 2**: Use $\tanh$ to approximate $\text{erf}$.

We know $\text{erf}(x) \approx \tanh(ax + bx^3)$, need to determine $a, b$.

Compare through Taylor expansion:

$$\text{erf}(x) = \frac{2}{\sqrt{\pi}}\left(x - \frac{x^3}{3} + \frac{x^5}{10} - \cdots\right)$$

$$\tanh(ax + bx^3) = ax + bx^3 - \frac{a^3x^3}{3} + O(x^5)$$

**Step 3**: Match coefficients.

By minimizing approximation error, get optimal parameters:

$$a = \sqrt{\frac{2}{\pi}} \approx 0.7979$$

$$b = 0.044715$$

**Step 4**: Substitute to get final formula.

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)$$

$$\boxed{\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right]\right)}$$

**Derivation of Sigmoid approximation**:

Since $\tanh(x) = 2\sigma(2x) - 1$, can convert to Sigmoid form:

$$\text{GELU}(x) \approx x \cdot \sigma(1.702x)$$

This approximation has slightly lower accuracy than Tanh approximation, but computation is simpler.

### Exact Computation

Use error function erf:

$$
\text{GELU}(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

### Properties of GELU

- **Non-monotonic**: There is a small negative region
- **Smooth**: Differentiable everywhere
- **Zero-centered**: Output mean close to 0
- **Asymptotic behavior**: Approaches $x$ as $x \to +\infty$, approaches 0 as $x \to -\infty$

### Why does Transformer use GELU?

1. **Smoothness**: No kink like ReLU
2. **Non-monotonicity**: Allows negative values to pass in some regions
3. **Probabilistic interpretation**: Can be seen as expectation of stochastic regularization

```python
def gelu(x):
    """GELU activation function (Tanh approximation)"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_exact(x):
    """GELU activation function (exact, using erf)"""
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_sigmoid_approx(x):
    """GELU Sigmoid approximation"""
    return x * (1 / (1 + np.exp(-1.702 * x)))

# Comparison
x = np.linspace(-3, 3, 1000)
print(f"Maximum difference between GELU approximation and exact: {np.max(np.abs(gelu(x) - gelu_exact(x))):.6f}")
```

---

## Activation Function Selection Guide

### Comparison of ReLU Function Family

| Activation Function | Formula | Advantages | Disadvantages | Use Case |
|---------------------|---------|------------|---------------|----------|
| ReLU | $\max(0, x)$ | Simple, fast | Dead ReLU | General purpose |
| LeakyReLU | $\max(\alpha x, x)$ | No dead problem | Hyperparameter $\alpha$ | General purpose |
| PReLU | $\max(\alpha x, x)$, $\alpha$ learnable | Adaptive | Overfitting risk | Large data |
| ELU | $x$ if $x>0$ else $\alpha(e^x-1)$ | Self-normalizing | Exponential computation | Deep networks |
| SELU | Scaled ELU | Self-normalizing | Needs specific initialization | Self-normalizing networks |
| GELU | $x \cdot \Phi(x)$ | Smooth, non-monotonic | Slightly slower computation | Transformer |

### Visualization Comparison

```python
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 1000)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Activation functions
axes[0].plot(x, relu(x), label='ReLU', linewidth=2)
axes[0].plot(x, leaky_relu(x, 0.1), label='LeakyReLU (α=0.1)', linewidth=2)
axes[0].plot(x, elu(x), label='ELU', linewidth=2)
axes[0].plot(x, gelu(x), label='GELU', linewidth=2)

axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('Activation Function Comparison')
axes[0].legend()
axes[0].grid(True)
axes[0].set_xlim([-3, 3])
axes[0].set_ylim([-1, 3])

# Derivatives
axes[1].plot(x, relu_derivative(x), label="ReLU'", linewidth=2)
axes[1].plot(x, leaky_relu_derivative(x, 0.1), label="LeakyReLU'", linewidth=2)
axes[1].plot(x, elu_derivative(x), label="ELU'", linewidth=2)

axes[1].set_xlabel('x')
axes[1].set_ylabel("f'(x)")
axes[1].set_title('Activation Function Derivative Comparison')
axes[1].legend()
axes[1].grid(True)
axes[1].set_xlim([-3, 3])

plt.tight_layout()
plt.show()
```

### Selection Recommendations

| Scenario | Recommended Activation Function | Reason |
|----------|-------------------------------|--------|
| Hidden layer (general) | ReLU / GELU | Simple and effective |
| Hidden layer (deep) | ELU / SELU | Self-normalizing |
| Transformer | GELU | Smooth, non-monotonic |
| CNN | ReLU / LeakyReLU | Computational efficiency |
| RNN | Tanh | Zero-centered |
| Output layer (binary classification) | Sigmoid | Output probability |
| Output layer (multi-class) | Softmax | Output probability distribution |
| Output layer (regression) | None / Linear | Unrestricted |

---

## Applications in Deep Learning

### Activation Functions in nanotorch

```python
from nanotorch.nn import ReLU, LeakyReLU, GELU, PReLU, ELU

# ReLU
relu = ReLU()
x = Tensor.randn((32, 128))
y = relu(x)

# LeakyReLU
leaky_relu = LeakyReLU(negative_slope=0.01)
y = leaky_relu(x)

# GELU
gelu = GELU()
y = gelu(x)

# PReLU
prelu = PReLU(num_parameters=128)  # One parameter per channel
y = prelu(x)
```

### Complete Neural Network Example

```python
from nanotorch import Tensor
from nanotorch.nn import Linear, ReLU, GELU, BatchNorm1d, Dropout, Sequential

# Network using ReLU
model_relu = Sequential(
    Linear(784, 256),
    BatchNorm1d(256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 128),
    BatchNorm1d(128),
    ReLU(),
    Linear(128, 10)
)

# Network using GELU (similar to Transformer)
model_gelu = Sequential(
    Linear(784, 256),
    BatchNorm1d(256),
    GELU(),
    Dropout(0.1),
    Linear(256, 128),
    BatchNorm1d(128),
    GELU(),
    Linear(128, 10)
)
```

### Handling Dead ReLU

```python
def check_dead_neurons(model, x):
    """Check the ratio of dead neurons"""
    activations = []

    def hook(module, input, output):
        activations.append(output)

    # Register hook
    for name, module in model.named_modules():
        if isinstance(module, ReLU):
            module.register_forward_hook(hook)

    # Forward pass
    model(x)

    # Check dead neurons
    for i, act in enumerate(activations):
        dead_ratio = (act == 0).mean().item()
        print(f"Layer {i}: Dead neuron ratio = {dead_ratio:.2%}")

# Solutions
# 1. Use LeakyReLU
# 2. Lower learning rate
# 3. Use better initialization (He initialization)
# 4. Use BatchNorm
```

---

## Summary

This section introduced the ReLU function family and activation function selection:

| Function | Core Features | Recommended Scenarios |
|----------|---------------|-----------------------|
| ReLU | Simple, efficient | CNN, general purpose |
| LeakyReLU | No dead problem | General purpose |
| ELU/SELU | Self-normalizing | Deep networks |
| GELU | Smooth, non-monotonic | Transformer |

**Key Takeaways**:
- ReLU is the most commonly used activation function
- LeakyReLU solves the dead ReLU problem
- GELU performs excellently in Transformers
- Activation function selection affects training efficiency and model performance

---

**Previous section**: [Sigmoid and Softmax Functions](06b-sigmoid-softmax_EN.md)

**Next section**: [Loss Functions and Normalization](06d-loss-functions-normalization_EN.md) - Learn about loss functions like MSE, cross-entropy and normalization techniques like BatchNorm, LayerNorm.

**Return**: [Chapter 6: Elementary Functions](06-elementary-functions.md) | [Math Fundamentals Tutorial Index](../math-fundamentals.md)
