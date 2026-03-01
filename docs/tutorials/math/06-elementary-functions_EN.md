# Chapter 6: Elementary Functions

Elementary functions are the **building blocks of deep learning**. From activation functions to loss functions, from probability distributions to normalization operations, elementary functions are everywhere. This chapter systematically introduces mathematical functions commonly used in deep learning and their properties.

---

## Chapter Structure

For ease of learning and in-depth understanding, this chapter is divided into four sub-chapters:

### [6.1 Exponential, Logarithmic and Trigonometric Functions](06a-exponential-logarithmic-trigonometric_EN.md)

**Content Overview**:
- Definition, properties, and Taylor expansion of exponential functions
- Definition and core properties of logarithmic functions
- Trigonometric functions: sin, cos, tan and their identities
- Hyperbolic functions: sinh, cosh, tanh
- Properties of Tanh as an activation function
- Numerically stable implementation techniques

**Core Concepts**:
| Function | Definition | Key Property | Deep Learning Application |
|----------|-----------|--------------|--------------------------|
| $e^x$ | $\sum_{n=0}^{\infty} \frac{x^n}{n!}$ | $(e^x)' = e^x$ | Softmax, probability |
| $\ln x$ | Inverse of $e^y = x$ | $(\ln x)' = 1/x$ | Cross-entropy, MLE |
| $\tanh$ | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | Zero-centered, $(-1,1)$ | RNNs, GELU |

**[Start Learning →](06a-exponential-logarithmic-trigonometric_EN.md)**

---

### [6.2 Sigmoid and Softmax Functions](06b-sigmoid-softmax_EN.md)

**Content Overview**:
- Sigmoid function and its derivative
- Sigmoid function family: Hard Sigmoid, Swish, Mish
- Definition and properties of Softmax function
- Derivative of Softmax (Jacobian matrix)
- Combination of Softmax and cross-entropy
- Role of temperature parameter

**Core Concepts**:
| Function | Formula | Use |
|----------|---------|-----|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | Binary classification, gating |
| Swish | $x \cdot \sigma(x)$ | Hidden layer activation |
| Softmax | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | Multi-class output |

**[Start Learning →](06b-sigmoid-softmax_EN.md)**

---

### [6.3 ReLU Family and Activation Functions](06c-relu-activation-functions_EN.md)

**Content Overview**:
- Definition of ReLU and "dying ReLU" problem
- Leaky ReLU and PReLU
- ELU and SELU (self-normalizing)
- GELU (Transformer standard)
- Activation function selection guide

**Core Concepts**:
| Function | Core Feature | Recommended Scenario |
|----------|--------------|---------------------|
| ReLU | Simple, efficient | CNNs, general |
| LeakyReLU | No dying problem | General |
| GELU | Smooth, non-monotonic | Transformers |

**[Start Learning →](06c-relu-activation-functions_EN.md)**

---

### [6.4 Loss Functions and Normalization](06d-loss-functions-normalization_EN.md)

**Content Overview**:
- Regression losses: MSE, MAE, Huber Loss
- Classification losses: BCE, Cross-entropy, Focal Loss
- Batch Normalization principles and implementation
- Layer Normalization (essential for Transformers)
- Other normalization: InstanceNorm, GroupNorm, RMSNorm

**Core Concepts**:
| Loss/Normalization | Formula/Method | Use |
|-------------------|----------------|-----|
| MSE | $\frac{1}{n}\sum(y-\hat{y})^2$ | Regression |
| CE | $-\sum y \ln\hat{y}$ | Classification |
| BatchNorm | Normalize by batch | CNNs |
| LayerNorm | Normalize by feature | Transformers |

**[Start Learning →](06d-loss-functions-normalization_EN.md)**

---

## Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│               Chapter 6: Elementary Functions                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  6.1 Exponential → 6.2 Sigmoid → 6.3 ReLU → 6.4 Loss       │
│  & Logarithmic    & Softmax     Family      Functions &     │
│  & Trig                                    Normalization    │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ e^x, ln  │   │ Sigmoid  │   │ ReLU     │   │ MSE, CE  │ │
│  │ tanh     │   │ Softmax  │   │ GELU     │   │ BatchNorm│ │
│  │ Numerical│   │ Temperature│  │ Selection│   │ LayerNorm│ │
│  │ stability│   │ parameter │   │ guide    │   │          │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                             │
│  Applications: Activation functions, Loss functions,        │
│  Normalization, Probabilistic modeling                       │
└─────────────────────────────────────────────────────────────┘
```

## Why are Elementary Functions Important for Deep Learning?

### 1. Activation Functions

Every layer of deep networks needs nonlinear activation functions:

```
Linear transformation: y = Wx + b
     ↓
Activation function: a = σ(y)
     ↓
Nonlinearity makes deep networks meaningful
```

### 2. Loss Functions and Probability

| Loss Function | Probabilistic Model | Activation Function |
|--------------|--------------------|--------------------|
| MSE | Gaussian distribution | None |
| BCE | Bernoulli distribution | Sigmoid |
| CE | Categorical distribution | Softmax |

### 3. Necessity of Normalization

- **Internal covariate shift**: Input distribution changes at each layer
- **Vanishing/exploding gradients**: Signal decay in deep networks
- **Normalization solves**: Stabilizes training, accelerates convergence

---

## Core Formula Quick Reference

### Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

### Softmax

$$
\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### ReLU

$$
\text{ReLU}(x) = \max(0, x)
$$

### Cross-Entropy

$$
\mathcal{L} = -\sum_i y_i \ln \hat{y}_i
$$

---

## Python Code Examples

### Activation Functions

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

### Loss Functions

```python
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def cross_entropy(logits, labels):
    log_probs = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
    return -np.mean(log_probs[np.arange(len(labels)), labels])
```

### BatchNorm

```python
def batch_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

---

## Study Recommendations

1. **Understand function properties**: Derivatives, range, special points
2. **Master numerical stability**: Prevent overflow and underflow
3. **Connect to practical applications**: Where each function is used
4. **Hands-on implementation**: Implement all functions with NumPy
5. **Visualization**: Plot graphs of functions and their derivatives

---

## Further Reading

- [Chapter 5: Optimization Methods](05-optimization_EN.md) - Gradient descent and optimization algorithms
- [Chapter 4: Mathematical Statistics](04-statistics_EN.md) - Probability distributions and statistical inference

---

**Back to**: [Mathematical Foundations Directory](../math-fundamentals_EN.md)
