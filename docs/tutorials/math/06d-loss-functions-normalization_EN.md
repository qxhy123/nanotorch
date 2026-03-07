# Chapter 6(d): Loss Functions and Normalization

Loss functions are the core of machine learning—they define the optimization objective for models. Normalization techniques are key innovations in modern deep learning that make training deep networks possible. This section will systematically introduce commonly used loss functions and normalization techniques.

---

## 🎯 Life Analogy: Loss Functions are Like Scoring a Dart Game

Imagine you're playing **darts**:
- Bullseye = prediction matches target
- Far from center = bad prediction
- **Loss = Distance from bullseye**

```
              Target
               ●
           ╱   │   ╲
         ╱     │     ╲
       ●───────┼───────●  Your darts
           ╲   │   ╱
             ╲ │ ╱
               ●

Loss = How far are your darts from the center?
```

### MSE vs MAE = Different Penalty Schemes

**Imagine salary deduction schemes**:

| Scheme | Formula | Analogy |
|--------|---------|---------|
| **MSE** | $(y - \hat{y})^2$ | Big mistakes are punished SEVERELY (squared) |
| **MAE** | $|y - \hat{y}|$ | Every dollar of error costs the same |

```
Error of 2:
MSE = 2² = 4
MAE = |2| = 2

Error of 10:
MSE = 10² = 100  ← MUCH worse!
MAE = |10| = 10
```

### Cross-Entropy = "Surprise Score"

**Scenario**: You predict it will rain with 90% confidence.
- If it rains: Not surprised (you predicted it!)
- If it doesn't: Very surprised! (you were confident but wrong)

**Cross-entropy measures your average "surprise"** when reality differs from predictions.

### 📖 Plain English Translation

| Term | Plain English |
|------|---------------|
| Loss function | Score of how wrong predictions are |
| MSE | Square the errors, then average |
| MAE | Absolute errors, then average |
| Cross-entropy | "Surprise" when predictions differ from reality |
| Normalization | Making values comparable (e.g., converting to 0-1 range) |

---

## Table of Contents

1. [Regression Loss Functions](#regression-loss-functions)
2. [Classification Loss Functions](#classification-loss-functions)
3. [Batch Normalization](#batch-normalization)
4. [Layer Normalization](#layer-normalization)
5. [Other Normalization Techniques](#other-normalization-techniques)
6. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Regression Loss Functions

### Mean Squared Error (MSE / L2 Loss)

**Definition**:

$$
\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

**Gradient**:

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{2}{n}(\hat{y}_i - y_i)
$$

**Probabilistic interpretation**: Assuming errors follow a Gaussian distribution, MSE is equivalent to negative log-likelihood.

**Advantages**:
- Differentiable everywhere
- Simple computation

**Disadvantages**:
- Sensitive to outliers (squared amplification)

```python
import numpy as np

def mse_loss(y_pred, y_true):
    """Mean squared error loss"""
    return np.mean((y_pred - y_true) ** 2)

def mse_loss_gradient(y_pred, y_true):
    """Gradient of MSE loss"""
    n = y_pred.shape[0]
    return 2 * (y_pred - y_true) / n
```

### Mean Absolute Error (MAE / L1 Loss)

**Definition**:

$$
\mathcal{L}_{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

**Subgradient**:

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{1}{n} \cdot \text{sign}(\hat{y}_i - y_i)
$$

**Advantages**:
- More robust to outliers

**Disadvantages**:
- Not differentiable at zero
- Slower convergence

```python
def mae_loss(y_pred, y_true):
    """Mean absolute error loss"""
    return np.mean(np.abs(y_pred - y_true))

def mae_loss_gradient(y_pred, y_true):
    """Gradient of MAE loss"""
    return np.sign(y_pred - y_true) / y_pred.shape[0]
```

### Huber Loss (Smooth L1)

Combines advantages of MSE and MAE:

$$
\mathcal{L}_{Huber} = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & |y - \hat{y}| \leq \delta \\
\delta |y - \hat{y}| - \frac{1}{2}\delta^2 & |y - \hat{y}| > \delta
\end{cases}
$$

**Advantages**:
- Uses MSE for small errors (fast convergence)
- Uses MAE for large errors (robust to outliers)
- Differentiable everywhere

```python
def huber_loss(y_pred, y_true, delta=1.0):
    """Huber loss (Smooth L1)"""
    diff = np.abs(y_pred - y_true)
    quadratic = np.minimum(diff, delta)
    linear = diff - quadratic
    return np.mean(0.5 * quadratic ** 2 + delta * linear)

def huber_loss_gradient(y_pred, y_true, delta=1.0):
    """Gradient of Huber loss"""
    diff = y_pred - y_true
    return np.where(np.abs(diff) <= delta, diff, delta * np.sign(diff))
```

### Loss Function Comparison

```python
import matplotlib.pyplot as plt

# Visualize different loss functions
errors = np.linspace(-3, 3, 1000)

mse = errors ** 2
mae = np.abs(errors)
huber = np.array([huber_loss(np.array([e]), np.array([0]), 1.0) for e in errors])

plt.figure(figsize=(10, 6))
plt.plot(errors, mse, label='MSE (L2)', linewidth=2)
plt.plot(errors, mae, label='MAE (L1)', linewidth=2)
plt.plot(errors, huber, label='Huber (δ=1)', linewidth=2)

plt.xlabel('Error (y_pred - y_true)')
plt.ylabel('Loss')
plt.title('Regression Loss Function Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Classification Loss Functions

### Binary Cross-Entropy (BCE)

**Definition**:

$$
\mathcal{L}_{BCE} = -\frac{1}{n}\sum_{i=1}^n [y_i \ln \hat{y}_i + (1-y_i) \ln(1-\hat{y}_i)]
$$

**Gradient**:

$$
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = -\left(\frac{y_i}{\hat{y}_i} - \frac{1-y_i}{1-\hat{y}_i}\right) = \frac{\hat{y}_i - y_i}{\hat{y}_i(1-\hat{y}_i)}
$$

**Probabilistic interpretation**: Negative log-likelihood of Bernoulli distribution.

```python
def binary_cross_entropy(y_pred, y_true, eps=1e-7):
    """Binary cross-entropy loss"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def bce_with_logits(logits, y_true):
    """Numerically stable BCE with logits"""
    # Use log-sum-exp trick
    max_val = np.maximum(logits, 0)
    loss = max_val - logits * y_true + np.log(1 + np.exp(-np.abs(logits)))
    return np.mean(loss)
```

### Categorical Cross-Entropy (CCE)

**Definition**:

$$
\mathcal{L}_{CCE} = -\frac{1}{n}\sum_{i=1}^n \sum_{c=1}^C y_{i,c} \ln \hat{y}_{i,c}
$$

If labels are one-hot encoded, simplifies to:

$$
\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \ln \hat{y}_{i, y_i}
$$

**Gradient when combined with Softmax**:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i
$$

```python
def categorical_cross_entropy(y_pred, y_true, eps=1e-7):
    """Categorical cross-entropy loss"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

def cross_entropy_with_logits(logits, labels):
    """Cross-entropy loss (includes softmax)"""
    # Numerically stable implementation
    log_probs = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))

    if labels.ndim == logits.ndim:
        # One-hot labels
        return -np.mean(np.sum(labels * log_probs, axis=-1))
    else:
        # Integer labels
        return -np.mean(log_probs[np.arange(len(labels)), labels])
```

### Focal Loss

Handles class imbalance:

$$
\mathcal{L}_{Focal} = -\alpha (1 - \hat{y})^\gamma \ln \hat{y}
$$

**Parameters**:
- $\alpha$: Weight to balance positive and negative samples
- $\gamma$: Focusing parameter, reduces weight for easy-to-classify samples (usually $\gamma = 2$)

```python
def focal_loss(y_pred, y_true, alpha=0.25, gamma=2.0, eps=1e-7):
    """Focal Loss"""
    y_pred = np.clip(y_pred, eps, 1 - eps)
    ce = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    # Focal weight
    focal_weight = y_true * (1 - y_pred) ** gamma + (1 - y_true) * y_pred ** gamma

    return np.mean(alpha * focal_weight * ce)
```

---

## Batch Normalization

### Definition

**During training**:

$$
\begin{align}
\mu_B &= \frac{1}{m} \sum_{i=1}^m x_i \\
\sigma_B^2 &= \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2 \\
\hat{x}_i &= \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \\
y_i &= \gamma \hat{x}_i + \beta
\end{align}
$$

Where:
- $\mu_B, \sigma_B^2$: Mean and variance of current mini-batch
- $\gamma, \beta$: Learnable scaling and shifting parameters
- $\epsilon$: Numerical stability constant

**During inference**: Use global mean and variance accumulated during training:

$$
\begin{align}
\mu_{running} &= (1 - \alpha)\mu_{running} + \alpha \mu_B \\
\sigma^2_{running} &= (1 - \alpha)\sigma^2_{running} + \alpha \sigma_B^2
\end{align}
$$

### Effects of BatchNorm

1. **Accelerates convergence**: Allows using larger learning rates
2. **Reduces sensitivity to initialization**
3. **Regularization effect**: Normalization of each sample depends on other samples
4. **Alleviates internal covariate shift**

### Derivation of BatchNorm Gradient

**Derivation of BatchNorm backpropagation formula**:

**Forward pass**:
$$\mu = \frac{1}{m}\sum_i x_i, \quad \sigma^2 = \frac{1}{m}\sum_i (x_i - \mu)^2$$
$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

**Step 1**: Gradient with respect to $\gamma$ and $\beta$.

$$\frac{\partial L}{\partial \gamma} = \sum_i \frac{\partial L}{\partial y_i} \cdot \hat{x}_i$$

$$\frac{\partial L}{\partial \beta} = \sum_i \frac{\partial L}{\partial y_i}$$

**Step 2**: Gradient with respect to $\hat{x}_i$.

$$\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma$$

**Step 3**: Gradient with respect to $\sigma^2$.

$$\frac{\partial L}{\partial \sigma^2} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \sigma^2}$$

Since $\hat{x}_i = (x_i - \mu)(\sigma^2 + \epsilon)^{-1/2}$:

$$\frac{\partial \hat{x}_i}{\partial \sigma^2} = (x_i - \mu) \cdot \left(-\frac{1}{2}\right)(\sigma^2 + \epsilon)^{-3/2} = -\frac{1}{2}(x_i - \mu) \cdot (\sigma^2 + \epsilon)^{-1} \cdot \hat{x}_i$$

Therefore:

$$\frac{\partial L}{\partial \sigma^2} = -\frac{1}{2}(\sigma^2 + \epsilon)^{-1} \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot (x_i - \mu)$$

**Step 4**: Gradient with respect to $\mu$.

$$\frac{\partial L}{\partial \mu} = \sum_i \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \mu} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial \mu}$$

$$\frac{\partial \hat{x}_i}{\partial \mu} = -(\sigma^2 + \epsilon)^{-1/2}$$

$$\frac{\partial \sigma^2}{\partial \mu} = \frac{2}{m}\sum_i (x_i - \mu) \cdot (-1) = -\frac{2}{m}\sum_i (x_i - \mu) = 0$$

(Because $\sum_i(x_i - \mu) = 0$)

So:

$$\frac{\partial L}{\partial \mu} = -(\sigma^2 + \epsilon)^{-1/2} \sum_i \frac{\partial L}{\partial \hat{x}_i}$$

**Step 5**: Gradient with respect to $x_i$.

$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial L}{\partial \sigma^2} \cdot \frac{\partial \sigma^2}{\partial x_i} + \frac{\partial L}{\partial \mu} \cdot \frac{\partial \mu}{\partial x_i}$$

Each term:
- $\frac{\partial \hat{x}_i}{\partial x_i} = (\sigma^2 + \epsilon)^{-1/2}$
- $\frac{\partial \sigma^2}{\partial x_i} = \frac{2}{m}(x_i - \mu)$
- $\frac{\partial \mu}{\partial x_i} = \frac{1}{m}$

**Step 6**: Combine all terms.

Let $\frac{\partial L}{\partial \hat{x}_i} = d\hat{x}_i$, $\sigma^{-1} = (\sigma^2 + \epsilon)^{-1/2}$, then:

$$\boxed{\frac{\partial L}{\partial x_i} = \sigma^{-1}\left(d\hat{x}_i - \frac{1}{m}\sum_j d\hat{x}_j - \frac{\hat{x}_i}{m}\sum_j d\hat{x}_j \cdot \hat{x}_j\right)}$$

**Simplified form**:

$$\frac{\partial L}{\partial x_i} = \frac{\gamma}{m\sigma}\left[m\frac{\partial L}{\partial y_i} - \sum_j \frac{\partial L}{\partial y_j} - \hat{x}_i \sum_j \frac{\partial L}{\partial y_j} \hat{x}_j\right]$$

```python
class BatchNorm1D:
    """1D Batch Normalization"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum
        self.training = True

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x):
        """Forward pass

        Args:
            x: (batch, features)
        """
        if self.training:
            # Compute batch statistics
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta

    def backward(self, x, grad_output):
        """Backward pass"""
        m = x.shape[0]
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Gradient for gamma and beta
        grad_gamma = np.sum(grad_output * x_norm, axis=0)
        grad_beta = np.sum(grad_output, axis=0)

        # Gradient for x
        std_inv = 1 / np.sqrt(var + self.eps)
        dx_norm = grad_output * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * std_inv**3, axis=0)
        dmean = np.sum(dx_norm * -std_inv, axis=0) + dvar * np.mean(-2 * (x - mean), axis=0)
        grad_x = dx_norm * std_inv + dvar * 2 * (x - mean) / m + dmean / m

        return grad_x, grad_gamma, grad_beta
```

---

## Layer Normalization

### Definition

**LayerNorm** normalizes on the feature dimension (not batch dimension):

$$
\begin{align}
\mu &= \frac{1}{d} \sum_{j=1}^d x_j \\
\sigma^2 &= \frac{1}{d} \sum_{j=1}^d (x_j - \mu)^2 \\
\hat{x}_j &= \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
y_j &= \gamma \hat{x}_j + \beta
\end{align}
$$

### BatchNorm vs LayerNorm

| Feature | BatchNorm | LayerNorm |
|----------|-----------|------------|
| Normalization dimension | Batch dim | Feature dim |
| Depends on batch size | Yes | No |
| Use case | CNN | RNN, Transformer |
| Inference consistency | Needs running stats | Fully consistent |
| Small batch performance | Poor | Stable |

```python
class LayerNorm:
    """Layer Normalization"""

    def __init__(self, normalized_shape, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)

    def forward(self, x):
        """Forward pass

        Args:
            x: (..., normalized_shape)
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta

    def backward(self, x, grad_output):
        """Backward pass"""
        d = x.shape[-1]
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        grad_gamma = np.sum(grad_output * x_norm, axis=tuple(range(x.ndim - 1)))
        grad_beta = np.sum(grad_output, axis=tuple(range(x.ndim - 1)))

        std_inv = 1 / np.sqrt(var + self.eps)
        dx_norm = grad_output * self.gamma
        dvar = np.sum(dx_norm * (x - mean) * -0.5 * std_inv**3, axis=-1, keepdims=True)
        dmean = np.sum(dx_norm * -std_inv, axis=-1, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=-1, keepdims=True)
        grad_x = dx_norm * std_inv + dvar * 2 * (x - mean) / d + dmean / d

        return grad_x, grad_gamma, grad_beta
```

---

## Other Normalization Techniques

### Instance Normalization

Normalizes **each sample's each channel** independently:

```python
class InstanceNorm2D:
    """Instance Normalization for 2D inputs"""

    def __init__(self, num_features, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

    def forward(self, x):
        """Forward pass

        Args:
            x: (N, C, H, W)
        """
        # Normalize on H, W dimensions
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        return self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]
```

### Group Normalization

Divides channels into $G$ groups, normalizes within each group:

```python
class GroupNorm:
    """Group Normalization"""

    def __init__(self, num_groups, num_channels, eps=1e-5):
        self.num_groups = num_groups
        self.eps = eps
        self.gamma = np.ones(num_channels)
        self.beta = np.zeros(num_channels)

    def forward(self, x):
        """Forward pass

        Args:
            x: (N, C, H, W)
        """
        N, C, H, W = x.shape
        G = self.num_groups

        # Reshape: (N, G, C//G, H, W)
        x = x.reshape(N, G, C // G, H, W)

        # Normalize within each group
        mean = np.mean(x, axis=(2, 3, 4), keepdims=True)
        var = np.var(x, axis=(2, 3, 4), keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Reshape back
        x_norm = x_norm.reshape(N, C, H, W)

        return self.gamma[None, :, None, None] * x_norm + self.beta[None, :, None, None]
```

### RMSNorm

**Root Mean Square Normalization** (no mean centering):

$$
\bar{x}_i = \frac{x_i}{\sqrt{\frac{1}{d} \sum_{j=1}^d x_j^2 + \epsilon}} \cdot \gamma
$$

**Advantage**: Simpler computation, performs comparably to LayerNorm on some tasks.

```python
class RMSNorm:
    """RMS Normalization"""

    def __init__(self, normalized_shape, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)

    def forward(self, x):
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / rms) * self.gamma
```

### Normalization Technique Comparison

| Method | Normalization Dimension | Advantages | Disadvantages |
|--------|------------------------|------------|---------------|
| BatchNorm | Batch | Accelerates convergence | Depends on batch size |
| LayerNorm | Feature | Stable | May be inferior to BN |
| InstanceNorm | (H, W) | Style transfer | - |
| GroupNorm | Group | Stable | - |
| RMSNorm | RMS | Efficient | No mean normalization |

---

## Applications in Deep Learning

### Loss Functions in nanotorch

```python
from nanotorch.nn import MSELoss, L1Loss, CrossEntropyLoss, BCELoss

# Regression tasks
mse = MSELoss()
l1 = L1Loss()

# Classification tasks
ce = CrossEntropyLoss()  # Includes softmax
bce = BCELoss()

# Usage example
logits = Tensor.randn((32, 10))
labels = Tensor(np.random.randint(0, 10, 32))

loss = ce(logits, labels)
loss.backward()
```

### Normalization Layers in nanotorch

```python
from nanotorch.nn import BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm

# Use BatchNorm in CNN
bn = BatchNorm2d(num_features=64)

# Use LayerNorm in Transformer
ln = LayerNorm(normalized_shape=512)

# Use GroupNorm for small batch size
gn = GroupNorm(num_groups=32, num_channels=128)
```

### Complete Training Example

```python
from nanotorch import Tensor
from nanotorch.nn import Linear, ReLU, BatchNorm1d, Dropout, Sequential, CrossEntropyLoss
from nanotorch.optim import Adam

# Create model
model = Sequential(
    Linear(784, 256),
    BatchNorm1d(256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 128),
    BatchNorm1d(128),
    ReLU(),
    Dropout(0.2),
    Linear(128, 10)
)

# Loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    model.train()
    for batch_x, batch_y in train_loader:
        # Forward pass
        output = model(Tensor(batch_x))
        loss = criterion(output, Tensor(batch_y))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = validate(model, val_loader)
    print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
```

---

## Summary

This section introduced loss functions and normalization techniques:

### Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| MSE | $\frac{1}{n}\sum(y-\hat{y})^2$ | Regression |
| MAE | $\frac{1}{n}\sum|y-\hat{y}|$ | Regression (robust) |
| BCE | $-[y\ln\hat{y}+(1-y)\ln(1-\hat{y})]$ | Binary classification |
| CE | $-\sum y \ln\hat{y}$ | Multi-class classification |
| Focal | $-\alpha(1-\hat{y})^\gamma\ln\hat{y}$ | Class imbalance |

### Normalization

| Method | Use Case |
|--------|----------|
| BatchNorm | CNN, large batch |
| LayerNorm | RNN, Transformer |
| GroupNorm | Small batch |
| RMSNorm | Efficient computation |

**Key Takeaways**:
- Loss functions correspond to probabilistic models (MSE↔Gaussian, CE↔classification)
- BatchNorm accelerates convergence but depends on batch size
- LayerNorm is stable and doesn't depend on batch size
- Normalization is a key component of modern deep networks

---

**Previous section**: [ReLU Function Family and Activation Functions](06c-relu-activation-functions_EN.md)

**Return**: [Chapter 6: Elementary Functions](06-elementary-functions_EN.md) | [Math Fundamentals Tutorial Index](../math-fundamentals.md)
