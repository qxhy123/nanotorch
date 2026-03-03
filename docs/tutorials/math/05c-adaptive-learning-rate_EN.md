# Chapter 5(c): Adaptive Learning Rate Methods

Fixed learning rates require selecting the same learning rate for all parameters, which may not be optimal. Adaptive learning rate methods solve this problem by maintaining independent learning rates for each parameter, greatly simplifying the hyperparameter tuning process.

---

## 🎯 Life Analogy: Different Students, Different Paces

Imagine a **classroom with students at different levels**:

```
Some students learn quickly → Give them smaller steps (smaller learning rate)
Some students need more time → Give them larger steps (larger learning rate)

Adaptive methods = Personalized learning pace for each parameter!
```

### The Auto-Adjusting Treadmill

Think of **AdaGrad like a treadmill that slows down automatically**:

```
Running fast (large gradients)?  → Belt slows down (learning rate decreases)
Running slowly (small gradients)? → Belt speeds up (learning rate increases)

Result: Everyone converges at a good pace!
```

### Adam = "Best of Both Worlds"

Adam combines **momentum** (ball rolling) + **adaptive learning rates** (treadmill):

```
┌────────────────────────────────────────────┐
│  Adam = Momentum + RMSprop                 │
├────────────────────────────────────────────┤
│  Momentum: Remembers where you were going  │
│  RMSprop:  Adjusts step size per parameter │
│                                            │
│  Result: Fast, stable, adaptive!           │
└────────────────────────────────────────────┘
```

### 📖 Plain English Translation

| Method | Key Idea | Best For |
|--------|----------|----------|
| AdaGrad | Slow down frequently updated params | Sparse data |
| RMSprop | Exponential moving average of gradients | Non-stationary |
| Adam | Momentum + adaptive rates | Most situations (default choice) |

---

## Table of Contents

1. [Motivation for Adaptive Learning Rates](#motivation-for-adaptive-learning-rates)
2. [AdaGrad](#adagrad)
3. [RMSprop](#rmsprop)
4. [Adam](#adam)
5. [AdamW](#adamw)
6. [Optimizer Comparison and Selection](#optimizer-comparison-and-selection)
7. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Motivation for Adaptive Learning Rates

### Problems with Fixed Learning Rate

In deep networks, gradient characteristics of different parameters may vary significantly:
- **Frequently updated parameters**: Large gradients, should use smaller learning rate
- **Sparsely updated parameters**: Small gradients, should use larger learning rate
- **Parameters in different layers**: Gradient magnitudes may differ by orders of magnitude

### Idea of Adaptation

Maintain independent learning rates $\eta_i$ for each parameter $\theta_i$:

$$
\theta_{i,t+1} = \theta_{i,t} - \eta_{i,t} \cdot g_{i,t}
$$

Where $\eta_{i,t}$ is dynamically adjusted based on historical gradients.

---

## AdaGrad

### Algorithm Principle

**AdaGrad** (Adaptive Gradient) adjusts learning rate based on the sum of squared historical gradients:

$$
\begin{align}
\mathbf{s}_{t+1} &= \mathbf{s}_t + \mathbf{g}_t \odot \mathbf{g}_t \\
\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{s}_{t+1}} + \epsilon} \odot \mathbf{g}_t
\end{align}
$$

Where:
- $\mathbf{s}_t$: Cumulative sum of squared gradients
- $\odot$: Element-wise multiplication
- $\epsilon$: Numerical stability constant (e.g., $10^{-8}$)

### Intuitive Understanding

- **Parameters with large gradients**: $\sqrt{s}$ is large, effective learning rate is small
- **Parameters with small gradients**: $\sqrt{s}$ is small, effective learning rate is large

### AdaGrad Implementation

```python
import numpy as np

class AdaGrad:
    """AdaGrad Optimizer"""

    def __init__(self, params, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.s = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        """Execute one update step"""
        for key in params:
            # Accumulate squared gradients
            self.s[key] += grads[key] ** 2

            # Adaptive learning rate update
            params[key] -= self.lr * grads[key] / (np.sqrt(self.s[key]) + self.eps)

    def get_effective_lr(self, key):
        """Get effective learning rate"""
        return self.lr / (np.sqrt(self.s[key]) + self.eps)

# Example
params = {'w': np.array([1.0, 2.0, 3.0])}
optimizer = AdaGrad(params, lr=0.1)

# Simulate several updates
for i in range(5):
    grads = {'w': np.array([0.1, 0.5, 0.01])}  # Different gradient magnitudes
    optimizer.step(params, grads)
    print(f"Step {i+1}: params = {params['w']}")
    print(f"         effective_lr = {optimizer.get_effective_lr('w')}")
```

### Problems with AdaGrad

**Monotonically decreasing learning rate**: $\mathbf{s}_t$ only increases and never decreases, causing learning rate to continuously decrease, potentially stopping learning prematurely.

**Applicable scenarios**: Sparse data (e.g., word embeddings in NLP), because frequently occurring words will get smaller learning rates.

---

## RMSprop

### Algorithm Principle

**RMSprop** (Root Mean Square Propagation) uses **exponential moving average** instead of cumulative sum:

$$
\begin{align}
\mathbf{s}_{t+1} &= \beta \mathbf{s}_t + (1 - \beta) \mathbf{g}_t \odot \mathbf{g}_t \\
\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\mathbf{s}_{t+1}} + \epsilon} \odot \mathbf{g}_t
\end{align}
$$

Where $\beta$ is the decay rate (typically 0.9).

### Difference from AdaGrad

| Feature | AdaGrad | RMSprop |
|---------|---------|---------|
| Gradient square accumulation | Cumulative sum | Exponential moving average |
| Learning rate change | Monotonically decreasing | Dynamically adjusted |
| Long-term training | May stop | Can continue |

### RMSprop Implementation

```python
class RMSprop:
    """RMSprop Optimizer"""

    def __init__(self, params, lr=0.01, alpha=0.9, eps=1e-8, weight_decay=0):
        self.lr = lr
        self.alpha = alpha  # Decay rate
        self.eps = eps
        self.weight_decay = weight_decay
        self.s = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        """Execute one update step"""
        for key in params:
            g = grads[key]

            # Weight decay
            if self.weight_decay != 0:
                g = g + self.weight_decay * params[key]

            # Update moving average of squared gradients
            self.s[key] = self.alpha * self.s[key] + (1 - self.alpha) * (g ** 2)

            # RMSprop update
            params[key] -= self.lr * g / (np.sqrt(self.s[key]) + self.eps)

# Example
params = {'w': np.random.randn(10, 5)}
grads = {'w': np.random.randn(10, 5) * 0.1}

optimizer = RMSprop(params, lr=0.01, alpha=0.9)
for _ in range(100):
    optimizer.step(params, grads)
```

---

## Adam

### Algorithm Principle

**Adam** (Adaptive Moment Estimation) combines momentum and adaptive learning rate:

$$
\begin{align}
\mathbf{m}_{t+1} &= \beta_1 \mathbf{m}_t + (1 - \beta_1) \mathbf{g}_t & \text{(First moment estimate)} \\
\mathbf{v}_{t+1} &= \beta_2 \mathbf{v}_t + (1 - \beta_2) \mathbf{g}_t^2 & \text{(Second moment estimate)} \\
\hat{\mathbf{m}}_{t+1} &= \frac{\mathbf{m}_{t+1}}{1 - \beta_1^{t+1}} & \text{(Bias correction)} \\
\hat{\mathbf{v}}_{t+1} &= \frac{\mathbf{v}_{t+1}}{1 - \beta_2^{t+1}} & \text{(Bias correction)} \\
\boldsymbol{\theta}_{t+1} &= \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{\mathbf{v}}_{t+1}} + \epsilon} \hat{\mathbf{m}}_{t+1}
\end{align}
$$

### Why Bias Correction?

Initialization sets $\mathbf{m}_0 = \mathbf{0}$, $\mathbf{v}_0 = \mathbf{0}$, causing initial estimates to be biased low:

$$
\mathbf{m}_1 = 0.1 \cdot \mathbf{g}_0 \quad \text{(instead of} \approx \mathbf{g}_0 \text{)}
$$

Bias correction compensates by dividing by $(1 - \beta^t)$:

$$
\hat{\mathbf{m}}_1 = \frac{0.1 \cdot \mathbf{g}_0}{1 - 0.9^1} = \mathbf{g}_0
$$

As $t \to \infty$, $1 - \beta^t \to 1$, and the correction effect disappears.

**Complete derivation of Adam bias correction**:

**Step 1**: Expand first moment estimate $\mathbf{m}_t$.

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1-\beta_1)\mathbf{g}_{t-1}$$

Recursive expansion:

$$\mathbf{m}_t = (1-\beta_1)\sum_{i=1}^{t} \beta_1^{t-i} \mathbf{g}_{i-1}$$

**Step 2**: Compute expectation.

Assuming gradients come from a stationary distribution, $\mathbb{E}[\mathbf{g}_i] = \mathbf{g}$ (true gradient), then:

$$\mathbb{E}[\mathbf{m}_t] = (1-\beta_1)\mathbf{g}\sum_{i=1}^{t} \beta_1^{t-i} = (1-\beta_1)\mathbf{g}\frac{1-\beta_1^t}{1-\beta_1} = (1-\beta_1^t)\mathbf{g}$$

**Step 3**: Identify bias.

$$\mathbb{E}[\mathbf{m}_t] = (1-\beta_1^t)\mathbf{g} \neq \mathbf{g}$$

The bias factor is $(1-\beta_1^t)$, which is large when $t$ is small.

**Step 4**: Construct unbiased estimate.

$$\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}$$

Verify unbiasedness:

$$\mathbb{E}[\hat{\mathbf{m}}_t] = \frac{\mathbb{E}[\mathbf{m}_t]}{1-\beta_1^t} = \frac{(1-\beta_1^t)\mathbf{g}}{1-\beta_1^t} = \mathbf{g}$$

**Step 5**: Similarly handle second moment.

$$\mathbb{E}[\mathbf{v}_t] = (1-\beta_2^t)\mathbb{E}[\mathbf{g}^2]$$

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}$$

**Step 6**: Convergence.

When $t \to \infty$:
- $\beta_1^t \to 0$, $\hat{\mathbf{m}}_t \to \mathbf{m}_t$
- $\beta_2^t \to 0$, $\hat{\mathbf{v}}_t \to \mathbf{v}_t$

$$\boxed{\hat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1-\beta_1^t}, \quad \hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1-\beta_2^t}}$$

**Numerical example** ($\beta_1 = 0.9$):

| $t$ | $1 - \beta_1^t$ | Correction factor |
|-----|----------------|-------------------|
| 1 | 0.1 | 10.0 |
| 5 | 0.41 | 2.44 |
| 10 | 0.65 | 1.54 |
| 50 | 0.995 | 1.005 |
| 100 | 0.99997 | 1.00003 |

### Adam Hyperparameters

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| $\eta$ | 0.001 | Learning rate |
| $\beta_1$ | 0.9 | First moment decay rate |
| $\beta_2$ | 0.999 | Second moment decay rate |
| $\epsilon$ | $10^{-8}$ | Numerical stability |

**Experience**: Default parameters usually work well, only need to adjust learning rate.

### Adam Implementation

```python
class Adam:
    """Adam Optimizer"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        # Initialize states
        self.m = {k: np.zeros_like(v) for k, v in params.items()}  # First moment
        self.v = {k: np.zeros_like(v) for k, v in params.items()}  # Second moment

    def step(self, params, grads):
        """Execute one update step"""
        self.t += 1

        for key in params:
            g = grads[key]

            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                g = g + self.weight_decay * params[key]

            # Update first moment
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g

            # Update second moment
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Adam update
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def get_lr(self):
        return self.lr

# Usage example
params = {
    'w1': np.random.randn(784, 256) * 0.01,
    'b1': np.zeros(256),
    'w2': np.random.randn(256, 10) * 0.01,
    'b2': np.zeros(10)
}

optimizer = Adam(params, lr=0.001, betas=(0.9, 0.999))

# Simulate training
for epoch in range(10):
    grads = {k: np.random.randn(*v.shape) * 0.01 for k, v in params.items()}
    optimizer.step(params, grads)
    print(f"Epoch {epoch+1}, w1 norm: {np.linalg.norm(params['w1']):.4f}")
```

### Advantages of Adam

1. **Adaptive learning rate**: Each parameter has independent learning rate
2. **Momentum effect**: Utilizes historical gradient information
3. **Bias correction**: More stable in early stages
4. **Hyperparameter robustness**: Default parameters usually work
5. **Memory efficiency**: Only need to store $\mathbf{m}$ and $\mathbf{v}$

---

## AdamW

### Problem: Weight Decay in Adam

In Adam, weight decay is typically implemented through $L_2$ regularization:

$$
\mathbf{g} = \nabla f(\boldsymbol{\theta}) + \lambda \boldsymbol{\theta}
$$

But this differs from "true" weight decay because the gradient is adaptively scaled.

### AdamW Solution

**AdamW** (Adam with Decoupled Weight Decay) decouples weight decay from gradient update:

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \left( \frac{\hat{\mathbf{m}}_{t+1}}{\sqrt{\hat{\mathbf{v}}_{t+1}} + \epsilon} + \lambda \boldsymbol{\theta}_t \right)
$$

Weight decay is applied directly to parameters, not added to gradients.

### AdamW Implementation

```python
class AdamW:
    """AdamW Optimizer (Decoupled Weight Decay)"""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        """Execute one update step"""
        self.t += 1

        for key in params:
            g = grads[key]

            # Update first and second moments (without weight decay)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)

            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # AdamW update: weight decay separated from gradient update
            params[key] -= self.lr * (
                m_hat / (np.sqrt(v_hat) + self.eps) +
                self.weight_decay * params[key]  # Decoupled weight decay
            )

# Compare Adam and AdamW
def compare_adam_adamw():
    """Compare behavior of Adam and AdamW"""
    np.random.seed(42)

    # Initial parameters
    params_adam = {'w': np.random.randn(100) * 0.1}
    params_adamw = {'w': params_adam['w'].copy()}

    optimizer_adam = Adam(params_adam, lr=0.01, weight_decay=0.01)
    optimizer_adamw = AdamW(params_adamw, lr=0.01, weight_decay=0.01)

    # Record weight norms
    norm_adam = [np.linalg.norm(params_adam['w'])]
    norm_adamw = [np.linalg.norm(params_adamw['w'])]

    for _ in range(100):
        grads = {'w': np.random.randn(100) * 0.1}

        optimizer_adam.step(params_adam, grads)
        optimizer_adamw.step(params_adamw, grads)

        norm_adam.append(np.linalg.norm(params_adam['w']))
        norm_adamw.append(np.linalg.norm(params_adamw['w']))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(norm_adam, label='Adam (L2 regularization)', linewidth=2)
    plt.plot(norm_adamw, label='AdamW (Decoupled weight decay)', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Weight Norm')
    plt.title('Adam vs AdamW: Weight Decay Effect Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()

compare_adam_adamw()
```

### Advantages of AdamW

1. **Better generalization**: True weight decay effect
2. **Works better with learning rate scheduling**: Weight decay unaffected by adaptive scaling
3. **Standard choice for Transformer training**

---

## Optimizer Comparison and Selection

### Convergence Comparison

| Optimizer | Type | Memory | Applicable Scenarios |
|-----------|------|--------|---------------------|
| SGD + Momentum | Momentum | $O(n)$ | CV, fine-tuning |
| AdaGrad | Adaptive | $O(n)$ | Sparse data |
| RMSprop | Adaptive | $O(n)$ | RNN, non-stationary objectives |
| Adam | Adaptive+Momentum | $O(2n)$ | General, fast prototyping |
| AdamW | Adaptive+Momentum | $O(2n)$ | Transformer, large models |

### Experimental Comparison

```python
def compare_optimizers():
    """Compare performance of different optimizers on the same problem"""
    import matplotlib.pyplot as plt

    # Define objective function (non-convex)
    def f(x):
        return x[0]**2 + 10*x[1]**2 + 0.1*np.sin(5*x[0])*np.cos(5*x[1])

    def grad_f(x):
        return np.array([
            2*x[0] + 0.5*np.cos(5*x[0])*np.cos(5*x[1]),
            20*x[1] - 0.5*np.sin(5*x[0])*np.sin(5*x[1])
        ])

    x0 = np.array([2.0, 2.0])
    n_iters = 200

    # SGD with Momentum
    params_sgd = {'x': x0.copy()}
    opt_sgd = SGDMomentum(params_sgd, lr=0.01, momentum=0.9)
    f_sgd = [f(params_sgd['x'])]
    for _ in range(n_iters):
        g = {'x': grad_f(params_sgd['x'])}
        opt_sgd.step(params_sgd, g)
        f_sgd.append(f(params_sgd['x']))

    # Adam
    params_adam = {'x': x0.copy()}
    opt_adam = Adam(params_adam, lr=0.1)
    f_adam = [f(params_adam['x'])]
    for _ in range(n_iters):
        g = {'x': grad_f(params_adam['x'])}
        opt_adam.step(params_adam, g)
        f_adam.append(f(params_adam['x']))

    # RMSprop
    params_rms = {'x': x0.copy()}
    opt_rms = RMSprop(params_rms, lr=0.1)
    f_rms = [f(params_rms['x'])]
    for _ in range(n_iters):
        g = {'x': grad_f(params_rms['x'])}
        opt_rms.step(params_rms, g)
        f_rms.append(f(params_rms['x']))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(f_sgd, label='SGD + Momentum', linewidth=2)
    plt.plot(f_adam, label='Adam', linewidth=2)
    plt.plot(f_rms, label='RMSprop', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Convergence Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(f_sgd, label='SGD + Momentum', linewidth=2)
    plt.semilogy(f_adam, label='Adam', linewidth=2)
    plt.semilogy(f_rms, label='RMSprop', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value (Log)')
    plt.title('Convergence Curve (Log Scale)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Helper class
class SGDMomentum:
    def __init__(self, params, lr, momentum):
        self.lr = lr
        self.momentum = momentum
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        for key in params:
            self.v[key] = self.momentum * self.v[key] + grads[key]
            params[key] -= self.lr * self.v[key]

compare_optimizers()
```

### Selection Guide

| Scenario | Recommended Optimizer | Learning Rate Suggestion |
|----------|---------------------|-------------------------|
| Fast prototyping | Adam | 0.001 (default) |
| Computer Vision | SGD + Momentum | 0.1 + scheduler |
| NLP / Transformer | AdamW | 0.0001 - 0.001 |
| RNN / LSTM | RMSprop / Adam | 0.001 |
| Fine-tuning | SGD + Momentum | 0.01 - 0.1 |
| Sparse data | AdaGrad / Adam | 0.01 |

---

## Applications in Deep Learning

### Optimizers in nanotorch

```python
from nanotorch.optim import SGD, Adam, AdamW, RMSprop
from nanotorch import Tensor
from nanotorch.nn import Linear, ReLU, Sequential, CrossEntropyLoss

# Create model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

# Choose optimizer
# Option 1: SGD + Momentum (suitable for CV)
optimizer = SGD(model.parameters(), lr=0.1, momentum=0.9)

# Option 2: Adam (fast prototyping)
optimizer = Adam(model.parameters(), lr=0.001)

# Option 3: AdamW (suitable for Transformer)
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Training loop
criterion = CrossEntropyLoss()

for epoch in range(100):
    for batch_x, batch_y in train_loader:
        # Forward propagation
        output = model(Tensor(batch_x))
        loss = criterion(output, Tensor(batch_y))

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Gradient Clipping

For RNN or certain unstable situations, gradient clipping is important:

```python
def clip_grad_norm_(grads, max_norm):
    """Gradient norm clipping"""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for key in grads:
            grads[key] *= scale
    return total_norm

# Use in training
for batch_x, batch_y in train_loader:
    output = model(Tensor(batch_x))
    loss = criterion(output, Tensor(batch_y))

    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    clip_grad_norm_(model.grads(), max_norm=1.0)

    optimizer.step()
```

---

## Summary

This section introduced adaptive learning rate methods:

| Method | Core Idea | Advantages | Disadvantages |
|--------|-----------|------------|----------------|
| AdaGrad | Accumulate squared gradients | Sparse data friendly | Learning rate decreases |
| RMSprop | EMA of squared gradients | Non-stationary objectives | Needs tuning |
| Adam | Momentum + adaptive | General, robust | May overfit |
| AdamW | Adam + decoupled regularization | Better generalization | - |

**Key Takeaways**:
- Adam is the most commonly used optimizer, default parameters usually work
- AdamW performs better in Transformer training
- SGD + Momentum may achieve better generalization during fine-tuning
- Adaptive methods simplify learning rate tuning

---

**Previous Section**: [Momentum Methods and Acceleration Techniques](05b-momentum-acceleration_EN.md)

**Next Section**: [Learning Rate Scheduling and Advanced Techniques](05d-lr-scheduling-advanced_EN.md) - Learn learning rate scheduling, second-order methods, and constrained optimization.

**Return**: [Chapter 5: Optimization Methods](05-optimization.md) | [Math Fundamentals Tutorial Table of Contents](../math-fundamentals.md)
