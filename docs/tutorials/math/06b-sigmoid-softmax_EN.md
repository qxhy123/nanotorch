# Chapter 6(b): Sigmoid and Softmax Functions

Sigmoid and Softmax are the most important output layer activation functions in deep learning. Sigmoid is used for binary classification, while Softmax is used for multi-class classification. This section will deeply explore the mathematical properties of these functions and their applications in deep learning.

---

## 🎯 Life Analogy: Converting Scores to Probabilities

### Sigmoid = "Yes/No Probability Converter"

Imagine a **spam filter**:
- Raw score: +5 (strongly looks like spam)
- Sigmoid(5) = 0.993 → 99.3% probability it's spam

- Raw score: -5 (strongly NOT spam)
- Sigmoid(-5) = 0.007 → 0.7% probability it's spam

```
Score → Sigmoid → Probability
━━━━━━━━━━━━━━━━━━━━━━━━━━━
  +5   →  0.993  →  99.3%
   0   →  0.500  →  50.0%
  -5   →  0.007  →   0.7%
```

### Softmax = "Voting to Percentage Converter"

Imagine an **election with 3 candidates**:
- Candidate A: 100 votes
- Candidate B: 50 votes
- Candidate C: 10 votes

Softmax converts votes to **win probabilities**:
$$P(A) = \frac{e^{100}}{e^{100} + e^{50} + e^{10}} \approx 100\%$$

### 📝 Step-by-Step Softmax Calculation

**Raw scores**: [3, 1, 0.5]

**Step 1**: Compute $e^x$
- $e^3 = 20.1$
- $e^1 = 2.7$
- $e^{0.5} = 1.6$

**Step 2**: Sum = 20.1 + 2.7 + 1.6 = 24.4

**Step 3**: Normalize
- P(class 1) = 20.1/24.4 = 82.4%
- P(class 2) = 2.7/24.4 = 11.1%
- P(class 3) = 1.6/24.4 = 6.6%

### 📖 Plain English Translation

| Function | Plain English |
|----------|---------------|
| Sigmoid | Turns any number into a probability (0 to 1) |
| Softmax | Turns a list of numbers into probabilities (sum to 1) |
| Temperature | Controls how "confident" the distribution is |

---

## Table of Contents

1. [Sigmoid Function](#sigmoid-function)
2. [Sigmoid Function Family](#sigmoid-function-family)
3. [Softmax Function](#softmax-function)
4. [Softmax and Cross-Entropy](#softmax-and-cross-entropy)
5. [Temperature Parameter](#temperature-parameter)
6. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Sigmoid Function

### Definition

**Sigmoid function** (also called Logistic function):

$$
\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1}
$$

### Core Properties

1. **Range**: $(0, 1)$, output can be interpreted as probability

2. **Derivative** (most important property):

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

**Proof**:

$$
\begin{align}
\sigma'(x) &= \frac{d}{dx}\left(\frac{1}{1+e^{-x}}\right) \\
&= \frac{e^{-x}}{(1+e^{-x})^2} \\
&= \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} \\
&= \sigma(x) \cdot (1 - \sigma(x))
\end{align}
$$

3. **Symmetry**:

$$
\sigma(-x) = 1 - \sigma(x)
$$

4. **Inverse function** (Logit function):

$$
\sigma^{-1}(y) = \ln\left(\frac{y}{1-y}\right) = \text{logit}(y)
$$

5. **Asymptotes**:
   - $\lim_{x \to -\infty} \sigma(x) = 0$
   - $\lim_{x \to +\infty} \sigma(x) = 1$

6. **Special values**:
   - $\sigma(0) = 0.5$
   - $\sigma'(0) = 0.25$ (maximum gradient)

### Pros and Cons of Sigmoid

**Advantages**:
- Output range $(0, 1)$, suitable for probability interpretation
- Differentiable everywhere, smooth
- Simple derivative calculation

**Disadvantages**:
- **Vanishing gradient**: when $|x|$ is large, $\sigma'(x) \to 0$
- **Non-zero centered**: output is always positive, affecting gradient update direction
- **Computational cost**: exponential operations are relatively slow

### Numerically Stable Implementation

```python
import numpy as np

def sigmoid(x):
    """Numerically stable sigmoid"""
    # Use different calculation methods for large positive and negative numbers
    pos_mask = x >= 0
    neg_mask = ~pos_mask

    result = np.zeros_like(x, dtype=np.float64)

    # Positive case: 1 / (1 + exp(-x))
    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))

    # Negative case: exp(x) / (1 + exp(x))
    exp_x = np.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1 + exp_x)

    return result

def sigmoid_derivative(x):
    """Derivative of Sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

# Example
x = np.array([-10, -1, 0, 1, 10])
print(f"Sigmoid: {sigmoid(x)}")
print(f"Derivative: {sigmoid_derivative(x)}")
```

---

## Sigmoid Function Family

### Hard Sigmoid

Faster linear approximation:

$$
\text{HardSigmoid}(x) = \max(0, \min(1, 0.2x + 0.5))
$$

```python
def hard_sigmoid(x):
    """Hard Sigmoid - linear approximation"""
    return np.clip(0.2 * x + 0.5, 0, 1)
```

**Advantage**: Fast computation, suitable for mobile devices
**Disadvantage**: Not differentiable at turning points

### Swish / SiLU (Sigmoid Linear Unit)

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**Derivative**:

$$
\text{Swish}'(x) = \sigma(x) + x \cdot \sigma'(x) = \sigma(x) \left(1 + x - \frac{x}{1 + e^x}\right)
$$

**Properties**:
- Non-monotonic (different from ReLU)
- Lower bounded (approaches 0 as $-\infty$), unbounded above
- Smooth
- Usually performs better than ReLU in deep networks

```python
def swish(x):
    """Swish/SiLU activation function"""
    return x * sigmoid(x)

def swish_derivative(x):
    """Derivative of Swish"""
    s = sigmoid(x)
    return s + x * s * (1 - s)
```

### Mish

$$
\text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x)) = x \cdot \tanh(\text{softplus}(x))
$$

**Properties**:
- Smoother than Swish
- Usually performs better on vision tasks
- Higher computational cost

```python
def mish(x):
    """Mish activation function"""
    return x * np.tanh(np.log1p(np.exp(x)))  # log1p(exp(x)) = log(1 + exp(x))
```

### Comparison of Sigmoid Function Family

| Function | Formula | Features |
|----------|---------|----------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | Classic, probability output |
| Hard Sigmoid | $\max(0, \min(1, 0.2x+0.5))$ | Fast, not differentiable |
| Swish/SiLU | $x \cdot \sigma(x)$ | Non-monotonic, smooth |
| Mish | $x \cdot \tanh(\ln(1+e^x))$ | Smoother, slower computation |

```python
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(12, 6))
plt.plot(x, sigmoid(x), label='Sigmoid', linewidth=2)
plt.plot(x, hard_sigmoid(x), label='Hard Sigmoid', linewidth=2)
plt.plot(x, swish(x), label='Swish/SiLU', linewidth=2)
plt.plot(x, mish(x), label='Mish', linewidth=2)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Comparison of Sigmoid Function Family')
plt.legend()
plt.grid(True)
plt.xlim([-5, 5])
plt.ylim([-1, 2])

plt.tight_layout()
plt.show()
```

---

## Softmax Function

### Definition

**Softmax** maps vector $\mathbf{z} = (z_1, \ldots, z_K)$ to probability distribution:

$$
\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad i = 1, \ldots, K
$$

### Core Properties

1. **Normalization**: $\sum_{i=1}^K \text{Softmax}(\mathbf{z})_i = 1$

2. **Non-negativity**: $\text{Softmax}(\mathbf{z})_i > 0$

3. **Maximum preservation**:

$$
\arg\max_i z_i = \arg\max_i \text{Softmax}(\mathbf{z})_i
$$

4. **Translation invariance**:

$$
\text{Softmax}(\mathbf{z} + c) = \text{Softmax}(\mathbf{z}), \quad \forall c \in \mathbb{R}
$$

5. **Scaling property**:

$$
\text{Softmax}(\alpha \mathbf{z}) \xrightarrow{\alpha \to \infty} \text{OneHot}(\arg\max \mathbf{z})
$$

### Derivative of Softmax

Let $s_i = \text{Softmax}(\mathbf{z})_i$, then:

$$
\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)
$$

Where $\delta_{ij}$ is Kronecker delta (1 when $i=j$, 0 otherwise).

**Two cases**:
- $i = j$: $\frac{\partial s_i}{\partial z_i} = s_i(1 - s_i)$
- $i \neq j$: $\frac{\partial s_i}{\partial z_j} = -s_i s_j$

**Jacobian matrix**:

$$
\mathbf{J} = \text{diag}(\mathbf{s}) - \mathbf{s} \mathbf{s}^\top
$$

### Log-Softmax

To avoid numerical issues, log-softmax is commonly used:

$$
\ln s_i = z_i - \ln\left(\sum_{j=1}^K e^{z_j}\right)
$$

**Numerically stable implementation**:

$$
\ln s_i = z_i - \max_k z_k - \ln\left(\sum_{j=1}^K e^{z_j - \max_k z_k}\right)
$$

```python
def softmax(x, axis=-1):
    """Numerically stable Softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def log_softmax(x, axis=-1):
    """Numerically stable Log-Softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
```

---

## Softmax and Cross-Entropy

### Cross-Entropy Loss

For true label $y$ (one-hot encoded) and predicted probability $\mathbf{s}$:

$$
\mathcal{L} = -\sum_{i=1}^K y_i \ln s_i = -\ln s_{y}
$$

Where $y$ is the true class (only one non-zero for one-hot).

### Gradient (Cross-Entropy + Softmax)

**Most elegant property**:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = s_i - y_i
$$

This is a very concise form! The gradient is simply the predicted probability minus the true label.

**Proof outline**:

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial z_i} &= -\sum_j y_j \frac{\partial \ln s_j}{\partial z_i} \\
&= -\sum_j y_j \frac{1}{s_j} \frac{\partial s_j}{\partial z_i} \\
&= -\sum_j y_j \frac{1}{s_j} s_j(\delta_{ij} - s_i) \\
&= -\sum_j y_j (\delta_{ij} - s_i) \\
&= s_i - y_i
\end{align}
$$

```python
def softmax_cross_entropy(logits, labels):
    """
    Softmax cross-entropy loss

    Args:
        logits: Model output (unnormalized)
        labels: True labels (integer or one-hot)
    """
    # Log-softmax
    log_probs = log_softmax(logits, axis=-1)

    # Cross-entropy
    if labels.ndim == logits.ndim:
        # One-hot labels
        return -np.sum(labels * log_probs, axis=-1)
    else:
        # Integer labels
        return -log_probs[np.arange(len(labels)), labels]

# Example
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
labels = np.array([0, 1])  # True classes

probs = softmax(logits)
print("Softmax probabilities:")
print(probs)

loss = softmax_cross_entropy(logits, labels)
print(f"\nCross-entropy loss: {loss}")

# Gradient
def softmax_cross_entropy_gradient(logits, labels):
    """Gradient of cross-entropy + Softmax"""
    probs = softmax(logits)
    if labels.ndim == logits.ndim:
        return probs - labels
    else:
        # One-hot encode labels
        one_hot = np.zeros_like(logits)
        one_hot[np.arange(len(labels)), labels] = 1
        return probs - one_hot

grad = softmax_cross_entropy_gradient(logits, labels)
print(f"\nGradient:\n{grad}")
```

---

## Temperature Parameter

### Definition

Softmax with temperature parameter $T$:

$$
\text{Softmax}_T(\mathbf{z})_i = \frac{e^{z_i/T}}{\sum_{j=1}^K e^{z_j/T}}
$$

### Effect of Temperature

- **$T > 1$**: Distribution is **smoother**, probability differences decrease
- **$T = 1$**: Standard Softmax
- **$T < 1$**: Distribution is **sharper**, probability differences increase
- **$T \to 0$**: Approaches argmax (one-hot)

```python
def softmax_with_temperature(x, temperature=1.0, axis=-1):
    """Softmax with temperature parameter"""
    x = x / temperature
    return softmax(x, axis=axis)

# Example: probability distribution at different temperatures
logits = np.array([2.0, 1.0, 0.1])

print("Probability distribution at different temperatures:")
for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
    probs = softmax_with_temperature(logits, T)
    print(f"T={T:4.1f}: {probs}")
```

### Application Scenarios

1. **Knowledge distillation**: Use high temperature to let student models learn soft labels from teacher models
2. **Reinforcement learning**: Control the balance between exploration and exploitation
3. **Text generation**: Control output diversity

---

## Applications in Deep Learning

### Binary Classification Output Layer

```python
# Binary classification
logits = model(x)
probs = sigmoid(logits)  # Output probabilities
loss = binary_cross_entropy(probs, labels)
```

### Multi-class Classification Output Layer

```python
# Multi-class classification (note: usually no explicit softmax call needed)
logits = model(x)
loss = cross_entropy_loss(logits, labels)  # Contains softmax internally
```

### Multi-label Classification

```python
# Multi-label classification (each class independent)
logits = model(x)
probs = sigmoid(logits)  # Use sigmoid independently for each class
loss = binary_cross_entropy(probs, labels)
```

### Gating Mechanism (LSTM/GRU)

```python
# Gating in LSTM
forget_gate = sigmoid(W_f @ h + b_f)
input_gate = sigmoid(W_i @ h + b_i)
output_gate = sigmoid(W_o @ h + b_o)
```

### Attention Weights

```python
# Weight computation in self-attention
attention_scores = queries @ keys.T / sqrt(d_k)
attention_weights = softmax(attention_scores, axis=-1)
```

### Usage in nanotorch

```python
from nanotorch import Tensor
from nanotorch.nn import Sigmoid, Softmax, BCELoss, CrossEntropyLoss

# Binary classification
sigmoid = Sigmoid()
bce_loss = BCELoss()

logits = Tensor.randn((32, 1))
targets = Tensor(np.random.randint(0, 2, (32, 1)).astype(np.float32))

probs = sigmoid(logits)
loss = bce_loss(probs, targets)

# Multi-class classification
softmax = Softmax(dim=-1)
ce_loss = CrossEntropyLoss()

logits = Tensor.randn((32, 10))
labels = Tensor(np.random.randint(0, 10, 32))

loss = ce_loss(logits, labels)  # Contains softmax internally
```

---

## Summary

This section introduced Sigmoid and Softmax functions:

| Function | Formula | Use Case |
|----------|---------|----------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | Binary classification, gating |
| Swish | $x \cdot \sigma(x)$ | Hidden layer activation |
| Softmax | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | Multi-class output |
| Log-Softmax | $z_i - \log\sum_j e^{z_j}$ | Numerically stable |

**Key Takeaways**:
- Sigmoid is used for binary classification, Softmax for multi-class classification
- Softmax + cross-entropy gradient is very concise: $s_i - y_i$
- Temperature parameter controls distribution smoothness
- Numerical stability is a key implementation challenge

---

**Previous section**: [Exponential, Logarithmic, and Trigonometric Functions](06a-exponential-logarithmic-trigonometric_EN.md)

**Next section**: [ReLU Function Family and Activation Functions](06c-relu-activation-functions_EN.md) - Learn about ReLU, LeakyReLU, GELU, and other activation functions.

**Return**: [Chapter 6: Elementary Functions](06-elementary-functions_EN.md) | [Math Fundamentals Tutorial Index](../math-fundamentals.md)
