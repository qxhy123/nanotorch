# Chapter 6(a): Exponential, Logarithmic, and Trigonometric Functions

Exponential and logarithmic functions are the most fundamental mathematical functions in deep learning. From Softmax to cross-entropy loss, from weight initialization to learning rate scheduling, these functions are everywhere. This section will systematically introduce these fundamental functions and their applications in deep learning.

---

## 🎯 Life Analogy: Exponential Growth Like Bacteria

Imagine **bacteria in a petri dish**:
- Hour 1: 2 bacteria
- Hour 2: 4 bacteria
- Hour 3: 8 bacteria
- Hour 4: 16 bacteria
- ...

This is **exponential growth**: $2^t$ - each step multiplies by the same amount!

```
Population
    │                           ╱
    │                         ╱
    │                       ╱
    │                     ╱
    │                   ╱
    │                 ╱
    │_______________╱________________→ Time
```

### Logarithm = "The Inverse of Exponential"

**Logarithms answer the question**: "What power do I need to get this number?"

| Question | Answer |
|----------|--------|
| $\log_{10}(100) = ?$ | 2, because $10^2 = 100$ |
| $\log_{2}(8) = ?$ | 3, because $2^3 = 8$ |
| $\ln(e^3) = ?$ | 3, because $e^3 = e^3$ |

### Real-life Logarithms

| Application | Base | Example |
|-------------|------|---------|
| Earthquake magnitude | 10 | Magnitude 7 is 10× stronger than 6 |
| pH (acidity) | 10 | pH 3 is 10× more acidic than pH 4 |
| Decibels (sound) | 10 | 60dB is 10× louder than 50dB |

### 📖 Plain English Translation

| Function | Plain English |
|----------|---------------|
| $e^x$ | "Multiply by $e$, $x$ times" - explosive growth |
| $\ln(x)$ | "How many times do I divide by $e$ to get 1?" |
| $\sin(x)$, $\cos(x)$ | Oscillating waves (sound, light, pendulums) |

---

## Table of Contents

1. [Exponential Functions](#exponential-functions)
2. [Logarithmic Functions](#logarithmic-functions)
3. [Trigonometric Functions](#trigonometric-functions)
4. [Hyperbolic Functions](#hyperbolic-functions)
5. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Exponential Functions

### Definition

The **exponential function** Taylor series expansion:

$$
e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots
$$

### Core Properties

1. **Basic Identity**:

$$
e^{x+y} = e^x \cdot e^y
$$

**Proof**:

**Step 1**: Consider $f(x) = e^{x+y}$ (fix $y$).

$$\frac{d}{dx} e^{x+y} = e^{x+y}$$

**Step 2**: Consider $g(x) = e^x \cdot e^y$.

$$\frac{d}{dx}(e^x \cdot e^y) = e^x \cdot e^y$$

**Step 3**: Both functions have the same derivative, so they differ by only a constant.

Set $x = 0$:

$$f(0) = e^y = g(0) = e^0 \cdot e^y = e^y$$

Therefore $f(x) = g(x)$ for all $x$.

$$\boxed{e^{x+y} = e^x \cdot e^y}$$

2. **Derivative**:

$$
\frac{d}{dx} e^x = e^x
$$

This is the only non-zero function whose derivative equals itself!

**Derivation of exponential function derivative**:

**Method 1: From definition**

$$\frac{d}{dx} e^x = \lim_{h \to 0} \frac{e^{x+h} - e^x}{h} = \lim_{h \to 0} \frac{e^x \cdot e^h - e^x}{h} = e^x \lim_{h \to 0} \frac{e^h - 1}{h}$$

The key is to prove $\lim_{h \to 0} \frac{e^h - 1}{h} = 1$.

**Method 2: Using inverse function derivative**

Let $y = e^x$, then $x = \ln y$.

$$\frac{dy}{dx} = \frac{1}{\frac{dx}{dy}} = \frac{1}{\frac{1}{y}} = y = e^x$$

$$\boxed{\frac{d}{dx} e^x = e^x}$$

3. **Positivity**: $e^x > 0, \forall x \in \mathbb{R}$

4. **Monotonicity**: $e^x$ is strictly monotonically increasing

5. **Limit behavior**:
   - $\lim_{x \to -\infty} e^x = 0$
   - $\lim_{x \to +\infty} e^x = +\infty$

6. **Symmetry**:

$$
e^{-x} = \frac{1}{e^x}
$$

### Generalized Exponential Function

For any base $a > 0, a \neq 1$:

$$
a^x = e^{x \ln a}
$$

**Properties**:
- $\frac{d}{dx} a^x = a^x \ln a$
- $a^{x+y} = a^x \cdot a^y$

---

## Logarithmic Functions

### Definition

The **natural logarithm** is the inverse function of the exponential function:

$$
y = \ln x \iff x = e^y, \quad x > 0
$$

### Core Properties

1. **Basic identities**:
   - $\ln(xy) = \ln x + \ln y$
   - $\ln(x/y) = \ln x - \ln y$
   - $\ln(x^n) = n \ln x$

2. **Derivative**:

$$
\frac{d}{dx} \ln x = \frac{1}{x}
$$

3. **Composite function derivative**:

$$
\frac{d}{dx} \ln f(x) = \frac{f'(x)}{f(x)}
$$

4. **Limit behavior**:
   - $\lim_{x \to 0^+} \ln x = -\infty$
   - $\lim_{x \to +\infty} \ln x = +\infty$

5. **Change of base formula**:

$$
\log_a x = \frac{\ln x}{\ln a}
$$

### Taylor Series Expansion

For $|x| < 1$:

$$
\ln(1 + x) = \sum_{n=1}^{\infty} \frac{(-1)^{n+1} x^n}{n} = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots
$$

### Role of Exponential and Logarithmic Functions in Deep Learning

**1. Probability Representation (Softmax)**

$$
P(y=i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

**2. Log Likelihood**

$$
\mathcal{L}(\theta) = \sum_{i=1}^N \ln P(y_i | x_i; \theta)
$$

**3. Exponential Moving Average**

$$
\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta) \mathbf{g}_t
$$

### Numerically Stable Implementation

```python
import numpy as np

def log_sum_exp(x, axis=None):
    """
    Numerically stable log-sum-exp

    log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    result = x_max + np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
    return np.squeeze(result, axis=axis) if axis is not None else result

def softmax(x, axis=-1):
    """Numerically stable softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def log_softmax(x, axis=-1):
    """Numerically stable log-softmax"""
    return x - log_sum_exp(x, axis=axis, keepdims=True)

# Example
x = np.array([1.0, 2.0, 3.0])
print(f"Softmax: {softmax(x)}")
print(f"Log-softmax: {log_softmax(x)}")
print(f"Log-sum-exp: {log_sum_exp(x)}")
```

---

## Trigonometric Functions

### Basic Definitions

On the unit circle, angle $\theta$ corresponds to:

$$
\begin{align}
\sin \theta &= \frac{\text{opposite}}{\text{hypotenuse}} = \frac{y}{r} \\[6pt]
\cos \theta &= \frac{\text{adjacent}}{\text{hypotenuse}} = \frac{x}{r} \\[6pt]
\tan \theta &= \frac{\sin \theta}{\cos \theta} = \frac{y}{x}
\end{align}
$$

### Core Identities

**Pythagorean theorem**:

$$
\sin^2 \theta + \cos^2 \theta = 1
$$

**Sum and difference formulas**:

$$
\begin{align}
\sin(a \pm b) &= \sin a \cos b \pm \cos a \sin b \\[6pt]
\cos(a \pm b) &= \cos a \cos b \mp \sin a \sin b
\end{align}
$$

**Double angle formulas**:

$$
\begin{align}
\sin 2\theta &= 2 \sin \theta \cos \theta \\[6pt]
\cos 2\theta &= \cos^2 \theta - \sin^2 \theta = 2\cos^2 \theta - 1 = 1 - 2\sin^2 \theta
\end{align}
$$

### Derivatives

$$
\frac{d}{dx} \sin x = \cos x
$$

$$
\frac{d}{dx} \cos x = -\sin x
$$

$$
\frac{d}{dx} \tan x = \sec^2 x = \frac{1}{\cos^2 x}
$$

### Periodicity

- $\sin(x + 2\pi) = \sin x$
- $\cos(x + 2\pi) = \cos x$
- $\tan(x + \pi) = \tan x$

---

## Hyperbolic Functions

### Definitions

**Hyperbolic sine**:

$$
\sinh x = \frac{e^x - e^{-x}}{2}
$$

**Hyperbolic cosine**:

$$
\cosh x = \frac{e^x + e^{-x}}{2}
$$

**Hyperbolic tangent**:

$$
\tanh x = \frac{\sinh x}{\cosh x} = \frac{e^x - e^{-x}}{e^x + e^{-x}}
$$

### Core Identities

**Hyperbolic Pythagorean theorem**:

$$
\cosh^2 x - \sinh^2 x = 1
$$

**Sum and difference formulas**:

$$
\begin{align}
\sinh(a \pm b) &= \sinh a \cosh b \pm \cosh a \sinh b \\[6pt]
\cosh(a \pm b) &= \cosh a \cosh b \pm \sinh a \sinh b
\end{align}
$$

### Derivatives

$$
\frac{d}{dx} \sinh x = \cosh x
$$

$$
\frac{d}{dx} \cosh x = \sinh x
$$

$$
\frac{d}{dx} \tanh x = 1 - \tanh^2 x = \text{sech}^2 x
$$

### Tanh Activation Function

**Relationship with Sigmoid**:

$$
\tanh(x) = 2\sigma(2x) - 1
$$

Where $\sigma$ is the Sigmoid function.

**Properties**:
- Output range: $(-1, 1)$
- Zero-centered (better than Sigmoid)
- Gradient: $\frac{d}{dx} \tanh(x) = 1 - \tanh^2(x)$
- Vanishing gradient problem: gradient approaches 0 when $|x|$ is large

### Comparison of Tanh and Sigmoid

| Feature | Sigmoid | Tanh |
|---------|---------|------|
| Output range | $(0, 1)$ | $(-1, 1)$ |
| Zero-centered | No | Yes |
| Gradient range | $(0, 0.25]$ | $(0, 1]$ |
| Use case | Binary classification output | RNN hidden layers |

```python
import numpy as np
import matplotlib.pyplot as plt

# Visualization
x = np.linspace(-5, 5, 1000)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Sigmoid
sigmoid = 1 / (1 + np.exp(-x))
axes[0, 0].plot(x, sigmoid, label='σ(x)', linewidth=2)
axes[0, 0].plot(x, sigmoid * (1 - sigmoid), label="σ'(x)", linewidth=2)
axes[0, 0].set_title('Sigmoid and its derivative')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Tanh
axes[0, 1].plot(x, np.tanh(x), label='tanh(x)', linewidth=2)
axes[0, 1].plot(x, 1 - np.tanh(x)**2, label="tanh'(x)", linewidth=2)
axes[0, 1].set_title('Tanh and its derivative')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Hyperbolic functions
axes[1, 0].plot(x, np.sinh(x), label='sinh(x)', linewidth=2)
axes[1, 0].plot(x, np.cosh(x), label='cosh(x)', linewidth=2)
axes[1, 0].set_title('Hyperbolic functions')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Trigonometric functions
axes[1, 1].plot(x, np.sin(x), label='sin(x)', linewidth=2)
axes[1, 1].plot(x, np.cos(x), label='cos(x)', linewidth=2)
axes[1, 1].set_title('Trigonometric functions')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()
```

---

## Applications in Deep Learning

### Numerical Stability of Softmax

When implementing Softmax, directly calculating $e^{z_i}$ may cause numerical overflow:

```python
# Dangerous implementation
def softmax_unstable(x):
    exp_x = np.exp(x)  # May overflow
    return exp_x / np.sum(exp_x)

# Numerically stable implementation
def softmax_stable(x):
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)  # Subtract maximum to prevent overflow
    return exp_x / np.sum(exp_x)
```

### Cross-Entropy Loss

Using logarithms can avoid numerical underflow:

```python
def cross_entropy_loss(logits, labels, eps=1e-15):
    """
    Numerically stable cross-entropy loss

    Args:
        logits: Model output (unnormalized)
        labels: True labels
    """
    # Use log-softmax to avoid numerical issues
    log_probs = log_softmax(logits, axis=-1)

    if labels.ndim == logits.ndim:
        # One-hot labels
        return -np.sum(labels * log_probs, axis=-1)
    else:
        # Integer labels
        return -log_probs[np.arange(len(labels)), labels]
```

### Cosine Function in Learning Rate Scheduling

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\frac{t}{T}\pi))
$$

```python
def cosine_annealing(epoch, max_epochs, eta_max, eta_min=1e-6):
    """Cosine annealing learning rate"""
    return eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / max_epochs))
```

### GELU Activation Function

Using hyperbolic tangent approximation:

$$
\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right]\right)
$$

```python
def gelu(x):
    """GELU activation function (tanh approximation)"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```

---

## Summary

This section introduced fundamental mathematical functions in deep learning:

| Function | Definition | Key Properties | Deep Learning Applications |
|----------|------------|----------------|----------------------------|
| $e^x$ | $\sum_{n=0}^{\infty} \frac{x^n}{n!}$ | $(e^x)' = e^x$ | Softmax, probabilities |
| $\ln x$ | Inverse of $e^y = x$ | $(\ln x)' = 1/x$ | Cross-entropy, MLE |
| $\sin, \cos$ | Trigonometric ratios | Periodicity | Position encoding, cosine scheduling |
| $\tanh$ | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | Zero-centered, $(-1,1)$ | RNN, GELU |

**Key Takeaways**:
- Exponential and logarithmic functions are the foundation of probability modeling
- Numerical stability is a key implementation challenge
- Tanh is a commonly used activation function in RNNs
- Understanding these functions helps in understanding more complex activation functions

---

**Next section**: [Sigmoid and Softmax Functions](06b-sigmoid-softmax_EN.md) - Learn about the Sigmoid function family and Softmax functions.

**Return**: [Chapter 6: Elementary Functions](06-elementary-functions.md) | [Math Fundamentals Tutorial Index](../math-fundamentals.md)
