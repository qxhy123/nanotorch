# Chapter 5(d): Learning Rate Scheduling and Advanced Techniques

Learning rate is the most important hyperparameter in deep learning training. Appropriate learning rate scheduling strategies can significantly improve training efficiency and final performance. This section will introduce common learning rate scheduling strategies, second-order optimization methods, and some advanced training techniques.

---

## 🎯 Life Analogy: Learning Rate is Like a Car's Gas Pedal

```
Start: Pedal to the metal (high learning rate)
       ┌────────────────────────────────────────────┐
       🚗💨━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ 🎯
       "General direction is right, go fast!"

Middle: Gradually ease off (learning rate decay)
       ┌────────────────────────────────────────────┐
       🚗━━━━━━━━━━━━━━━━━━━━━━━━━━━━→ 🎯
       "Getting close, slow down for precision"

End: Barely touching the pedal (learning rate → 0)
       ┌────────────────────────────────────────────┐
       🚗━━→ 🎯
       "Fine-tuning the final position"
```

### Step Decay = Seasonal Sales

Like **store discounts that drop periodically**:

| Week | Discount | Analogy |
|------|----------|---------|
| 1-10 | 50% off | High learning rate |
| 11-20 | 25% off | Learning rate halved |
| 21-30 | 12.5% off | Halved again |
| 31+ | 6.25% off | Very small adjustments |

### Cosine Annealing = Smooth Deceleration

Unlike step decay (sudden drops), **cosine decay is smooth**:

```
Step decay:    ═══╤═══╤═══╤═══  (sudden drops)
                     ↓     ↓
Cosine:        ╲            ╲    (smooth curve)
                 ╲          ╲
                   ╲        ╲
                     ╲      ╲
```

### Warmup = Engine Warm-up

You don't floor a cold engine! Warmup gradually increases learning rate:

```
Learning Rate
    │       ╱─────────────────╲
    │      ╱                    ╲
    │     ╱                      ╲
    │    ╱                        ╲
    │___╱                          ╲___→ Steps
       Warmup        Main Training      Cooldown
```

### 📖 Plain English Translation

| Technique | Plain English |
|-----------|---------------|
| Step decay | Cut learning rate in half every N epochs |
| Cosine decay | Smoothly decrease following a cosine curve |
| Warmup | Start slow, then ramp up |
| Cyclic LR | Go up and down repeatedly |

---

## Table of Contents

1. [Importance of Learning Rate](#importance-of-learning-rate)
2. [Common Learning Rate Scheduling Strategies](#common-learning-rate-scheduling-strategies)
3. [Second-Order Optimization Methods](#second-order-optimization-methods)
4. [Constrained Optimization](#constrained-optimization)
5. [Advanced Training Techniques](#advanced-training-techniques)
6. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Importance of Learning Rate

### Impact of Learning Rate on Training

```
Learning rate too large:     Learning rate too small:     Learning rate appropriate:

  ↗ ↖ ↗ ↖                    ·                            ↘
 ↗   ↖   ↗                  · ·                           ↘
↗ Diverge  ↖                · · ·  Slow                    ↘ Stable convergence
                       ·     ·                              ↘
                      ·       ·                              ↘
```

### Learning Rate and Training Phases

| Phase | Learning Rate | Goal |
|-------|--------------|------|
| Early | Larger | Fast descent |
| Middle | Moderate | Stable optimization |
| Late | Smaller | Fine-tuning |

---

## Common Learning Rate Scheduling Strategies

### 1. Step Decay

**Formula**:

$$
\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T \rfloor}
$$

Multiply learning rate by $\gamma$ every $T$ epochs.

**Example**: Initial $\eta_0 = 0.1$, multiply by 0.1 every 30 epochs

```
Epoch 0-29:   lr = 0.1
Epoch 30-59:  lr = 0.01
Epoch 60-89:  lr = 0.001
```

```python
class StepLR:
    """Step learning rate decay"""

    def __init__(self, optimizer, step_size, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        """Update learning rate"""
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma

    def get_lr(self):
        return self.optimizer.lr
```

### 2. Exponential Decay

**Formula**:

$$
\eta_t = \eta_0 \cdot \gamma^t
$$

Multiply learning rate by $\gamma$ each epoch.

```python
class ExponentialLR:
    """Exponential learning rate decay"""

    def __init__(self, optimizer, gamma=0.95):
        self.optimizer = optimizer
        self.gamma = gamma
        self.epoch = 0
        self.base_lr = optimizer.lr

    def step(self):
        self.epoch += 1
        self.optimizer.lr = self.base_lr * (self.gamma ** self.epoch)

    def get_lr(self):
        return self.optimizer.lr
```

### 3. Cosine Annealing

**Formula**:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

**Characteristics**:
- Learning rate decreases slowly in early stages
- Decreases fastest in middle stages
- Tends to stabilize in late stages

```python
import numpy as np

class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler"""

    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_max = optimizer.lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        self.optimizer.lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
                           (1 + np.cos(np.pi * self.epoch / self.T_max))

    def get_lr(self):
        return self.optimizer.lr

# Visualization
def plot_cosine_annealing():
    import matplotlib.pyplot as plt

    epochs = 100
    t = np.arange(1, epochs + 1)

    eta_max, eta_min = 0.1, 1e-6
    cosine_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / epochs))

    plt.figure(figsize=(10, 6))
    plt.plot(t, cosine_lr, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing Learning Rate')
    plt.grid(True)
    plt.show()

plot_cosine_annealing()
```

### 4. Linear Warmup

**Motivation**: In early training, parameters are random, large learning rate may cause instability.

**Strategy**: Linearly increase learning rate for the first $T_{\text{warmup}}$ steps:

$$
\eta_t = \eta_{\max} \cdot \frac{t}{T_{\text{warmup}}}, \quad t \leq T_{\text{warmup}}
$$

```python
class LinearWarmup:
    """Linear warmup"""

    def __init__(self, optimizer, warmup_steps, target_lr):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.target_lr = target_lr
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup_steps:
            self.optimizer.lr = self.target_lr * self.step_count / self.warmup_steps

    def is_warmup_done(self):
        return self.step_count >= self.warmup_steps
```

### 5. Warmup + Cosine Annealing

Standard configuration for Transformer training:

```python
class CosineWarmupScheduler:
    """Cosine annealing scheduler with warmup"""

    def __init__(self, optimizer, warmup_epochs, max_epochs, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.eta_max = optimizer.lr
        self.epoch = 0

    def step(self):
        self.epoch += 1
        if self.epoch <= self.warmup_epochs:
            # Linear warmup
            self.optimizer.lr = self.eta_max * self.epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            self.optimizer.lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
                               (1 + np.cos(np.pi * progress))

    def get_lr(self):
        return self.optimizer.lr

# Visualization
def plot_warmup_cosine():
    import matplotlib.pyplot as plt

    epochs = 100
    warmup = 10
    eta_max, eta_min = 0.1, 1e-6

    lr = []
    for t in range(1, epochs + 1):
        if t <= warmup:
            lr.append(eta_max * t / warmup)
        else:
            progress = (t - warmup) / (epochs - warmup)
            lr.append(eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * progress)))

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), lr, linewidth=2)
    plt.axvline(x=warmup, color='r', linestyle='--', label=f'Warmup end (epoch {warmup})')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Warmup + Cosine Annealing')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_warmup_cosine()
```

### 6. Reduce on Plateau

Reduce learning rate when validation loss stops decreasing:

```python
class ReduceLROnPlateau:
    """Reduce learning rate when validation loss stagnates"""

    def __init__(self, optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.bad_epochs = 0

    def step(self, metric):
        """Update learning rate based on metric"""
        if self.mode == 'min':
            improved = metric < self.best
        else:
            improved = metric > self.best

        if improved:
            self.best = metric
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            if self.bad_epochs >= self.patience:
                self.optimizer.lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.bad_epochs = 0
                print(f"Learning rate reduced to {self.optimizer.lr}")
```

### 7. Cyclical Learning Rate

Learning rate changes periodically within a range:

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{2\pi t}{T}))
$$

```python
class CyclicLR:
    """Cyclical learning rate"""

    def __init__(self, optimizer, base_lr, max_lr, step_size_up, step_size_down=None):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.step_count = 0

    def step(self):
        self.step_count += 1
        cycle = self.step_size_up + self.step_size_down

        if self.step_count % cycle < self.step_size_up:
            # Rising phase
            pct = (self.step_count % cycle) / self.step_size_up
            self.optimizer.lr = self.base_lr + pct * (self.max_lr - self.base_lr)
        else:
            # Falling phase
            pct = ((self.step_count % cycle) - self.step_size_up) / self.step_size_down
            self.optimizer.lr = self.max_lr - pct * (self.max_lr - self.base_lr)
```

### Learning Rate Scheduling Visualization

```python
def plot_all_schedulers():
    """Visualize all learning rate scheduling strategies"""
    import matplotlib.pyplot as plt

    epochs = 100
    warmup = 10
    t = np.arange(1, epochs + 1)
    eta_max, eta_min = 0.1, 1e-6

    # 1. Step decay
    step_lr = 0.1 * (0.1 ** (t // 30))

    # 2. Exponential decay
    exp_lr = 0.1 * (0.97 ** t)

    # 3. Cosine annealing
    cosine_lr = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * t / epochs))

    # 4. Warmup + cosine
    warmup_cosine = np.zeros(epochs)
    for i in range(epochs):
        if i < warmup:
            warmup_cosine[i] = eta_max * (i + 1) / warmup
        else:
            progress = (i - warmup) / (epochs - warmup)
            warmup_cosine[i] = eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(np.pi * progress))

    # 5. Cyclic learning rate
    cycle = 20
    cyclic = []
    for i in range(epochs):
        pct = (i % cycle) / cycle
        if pct < 0.5:
            cyclic.append(eta_min + 2 * pct * (eta_max - eta_min))
        else:
            cyclic.append(eta_max - 2 * (pct - 0.5) * (eta_max - eta_min))

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    axes[0, 0].plot(t, step_lr, linewidth=2)
    axes[0, 0].set_title('Step Decay')

    axes[0, 1].plot(t, exp_lr, linewidth=2)
    axes[0, 1].set_title('Exponential Decay')

    axes[0, 2].plot(t, cosine_lr, linewidth=2)
    axes[0, 2].set_title('Cosine Annealing')

    axes[1, 0].plot(t, warmup_cosine, linewidth=2)
    axes[1, 0].axvline(x=warmup, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Warmup + Cosine')

    axes[1, 1].plot(t, cyclic, linewidth=2)
    axes[1, 1].set_title('Cyclic LR')

    axes[1, 2].plot(t, cosine_lr, label='Cosine', linewidth=2)
    axes[1, 2].plot(t, warmup_cosine, label='Warmup+Cosine', linewidth=2)
    axes[1, 2].legend()
    axes[1, 2].set_title('Comparison')

    for ax in axes.flat:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.grid(True)

    plt.tight_layout()
    plt.show()

plot_all_schedulers()
```

---

## Second-Order Optimization Methods

### Newton's Method

**Principle**: Uses second-order derivative (Hessian) information:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - [\nabla^2 f(\mathbf{x}_t)]^{-1} \nabla f(\mathbf{x}_t)
$$

**Derivation**: Second-order Taylor expansion:

$$
f(\mathbf{x}) \approx f(\mathbf{x}_t) + \nabla f(\mathbf{x}_t)^\top (\mathbf{x} - \mathbf{x}_t) + \frac{1}{2}(\mathbf{x} - \mathbf{x}_t)^\top \nabla^2 f(\mathbf{x}_t) (\mathbf{x} - \mathbf{x}_t)
$$

Differentiating with respect to $\mathbf{x}$ and setting to zero gives Newton's method update formula.

**Advantages**: Quadratic convergence (very fast)

**Disadvantages**:
- Computing and storing Hessian: $O(n^2)$
- Inversion: $O(n^3)$
- Requires positive definite Hessian

```python
def newton_method(f, grad_f, hess_f, x0, max_iters=100, tol=1e-6):
    """Newton's method"""
    x = x0.copy()
    history = [x.copy()]

    for i in range(max_iters):
        g = grad_f(x)

        if np.linalg.norm(g) < tol:
            print(f"Converged after {i} iterations")
            break

        H = hess_f(x)

        # Solve H * d = -g
        try:
            d = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            # Hessian singular, use pseudo-inverse
            d = -np.linalg.pinv(H) @ g

        x = x + d
        history.append(x.copy())

    return x, history

# Example: Rosenbrock function
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])

def rosenbrock_hess(x):
    return np.array([
        [1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]],
        [-400 * x[0], 200]
    ])

x0 = np.array([-1.0, 1.0])
x_opt, history = newton_method(rosenbrock, rosenbrock_grad, rosenbrock_hess, x0)
print(f"Optimal solution: {x_opt}")
print(f"Number of iterations: {len(history)}")
```

### Quasi-Newton: BFGS

Avoids directly computing and inverting Hessian, by iteratively updating the inverse Hessian approximation:

$$
\mathbf{B}_{t+1} = \mathbf{B}_t - \frac{\mathbf{B}_t \mathbf{s}_t \mathbf{s}_t^\top \mathbf{B}_t}{\mathbf{s}_t^\top \mathbf{B}_t \mathbf{s}_t} + \frac{\mathbf{y}_t \mathbf{y}_t^\top}{\mathbf{y}_t^\top \mathbf{s}_t}
$$

Where $\mathbf{s}_t = \mathbf{x}_{t+1} - \mathbf{x}_t$, $\mathbf{y}_t = \nabla f(\mathbf{x}_{t+1}) - \nabla f(\mathbf{x}_t)$.

### L-BFGS

Limited-memory BFGS: Only stores the most recent $m$ $(\mathbf{s}_t, \mathbf{y}_t)$ pairs, reducing memory from $O(n^2)$ to $O(mn)$.

### Second-Order Methods in Deep Learning

| Method | Complexity | Applicability |
|--------|------------|---------------|
| Newton's method | $O(n^3)$ | Small-scale problems |
| BFGS | $O(n^2)$ | Medium scale |
| L-BFGS | $O(mn)$ | Large scale |

**Challenges in deep learning**:
- Huge number of parameters (millions to billions)
- Hessian computation and storage infeasible
- Typically use first-order methods + clever learning rate scheduling

---

## Constrained Optimization

### Projected Gradient Descent

For constraint set $C$, project back to feasible region after gradient descent:

$$
\mathbf{x}_{t+1} = \text{Proj}_C(\mathbf{x}_t - \eta \nabla f(\mathbf{x}_t))
$$

**Common Projections**:

1. **Box constraint** $[l, u]$:

$$
\text{Proj}_{[l,u]}(x) = \max(l, \min(u, x))
$$

2. **L2 ball** $\{\mathbf{x} : \|\mathbf{x}\|_2 \leq r\}$:

$$
\text{Proj}(\mathbf{x}) = \frac{r \mathbf{x}}{\max(r, \|\mathbf{x}\|_2)}
$$

```python
def project_to_ball(x, radius=1.0):
    """Project to L2 ball"""
    norm = np.linalg.norm(x)
    if norm <= radius:
        return x
    return radius * x / norm

def project_to_box(x, lower, upper):
    """Project to box constraint"""
    return np.clip(x, lower, upper)

def projected_gradient_descent(f, grad_f, x0, project_func,
                                learning_rate=0.01, max_iters=1000):
    """Projected gradient descent"""
    x = x0.copy()
    history = [x.copy()]

    for _ in range(max_iters):
        g = grad_f(x)
        x = project_func(x - learning_rate * g)
        history.append(x.copy())

    return x, history
```

### Weight Clipping

In deep learning, commonly used to keep weight norms bounded:

```python
def weight_clipping(model, max_norm=1.0):
    """Weight clipping"""
    for param in model.parameters():
        norm = np.linalg.norm(param)
        if norm > max_norm:
            param *= max_norm / norm
```

---

## Advanced Training Techniques

### Gradient Clipping

Prevent gradient explosion:

```python
def clip_grad_norm_(grads, max_norm):
    """Gradient norm clipping"""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        for key in grads:
            grads[key] *= scale
    return total_norm

def clip_grad_value_(grads, clip_value):
    """Gradient value clipping"""
    for key in grads:
        grads[key] = np.clip(grads[key], -clip_value, clip_value)
```

### Learning Rate Finder

Start with small learning rate, gradually increase, record loss changes:

```python
def lr_finder(model, optimizer, train_loader, min_lr=1e-6, max_lr=1, num_steps=100):
    """Learning rate finder"""
    lrs = np.logspace(np.log10(min_lr), np.log10(max_lr), num_steps)
    losses = []

    for i, lr in enumerate(lrs):
        optimizer.lr = lr

        # Single step training
        for batch_x, batch_y in train_loader:
            output = model(batch_x)
            loss = model.compute_loss(output, batch_y)
            grads = model.backward()
            optimizer.step(model.params, grads)
            losses.append(loss)
            break

        if i > 0 and losses[-1] > 2 * losses[0]:
            break

    return lrs[:len(losses)], losses
```

### Mixed Precision Training

Use FP16 to accelerate training while maintaining FP32 precision:

```python
class MixedPrecisionTrainer:
    """Mixed precision training (simplified example)"""

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        # FP32 master weight copy
        self.master_params = {k: v.astype(np.float32) for k, v in model.params.items()}

    def step(self, grads):
        # Accumulate gradients in FP32
        for key in self.master_params:
            self.master_params[key] -= self.optimizer.lr * grads[key].astype(np.float32)

        # Sync to FP16
        for key in self.model.params:
            self.model.params[key] = self.master_params[key].astype(np.float16)
```

---

## Applications in Deep Learning

### Learning Rate Schedulers in nanotorch

```python
from nanotorch.optim import SGD, AdamW
from nanotorch.optim import StepLR, CosineAnnealingLR, CosineWarmupScheduler, ReduceLROnPlateau

# Create model and optimizer
model = create_model()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

# Choose learning rate scheduler

# Option 1: Step decay
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# Option 2: Cosine annealing
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# Option 3: Warmup + cosine (recommended for Transformer)
scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=5, max_epochs=100)

# Option 4: Reduce on Plateau
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

# Training loop
for epoch in range(100):
    train_loss = train_one_epoch(model, optimizer, train_loader)
    val_loss = validate(model, val_loader)

    # Update learning rate
    if isinstance(scheduler, ReduceLROnPlateau):
        scheduler.step(val_loss)
    else:
        scheduler.step()

    print(f"Epoch {epoch}, LR: {scheduler.get_lr():.6f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
```

### Complete Training Example

```python
def train_with_scheduler():
    """Complete training example with learning rate scheduling"""
    from nanotorch.optim import AdamW, CosineWarmupScheduler

    # Model, optimizer, scheduler
    model = create_model()
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=5,
        max_epochs=100,
        eta_min=1e-6
    )

    best_val_loss = float('inf')

    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            output = model(Tensor(batch_x))
            loss = criterion(output, Tensor(batch_y))

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            clip_grad_norm_(model.grads(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = validate(model, val_loader)

        # Update learning rate
        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, 'best_model.npz')

        print(f"Epoch {epoch+1}/100, LR: {scheduler.get_lr():.6f}, "
              f"Train: {train_loss/len(train_loader):.4f}, "
              f"Val: {val_loss:.4f}")
```

---

## Summary

This section introduced learning rate scheduling and advanced techniques:

### Learning Rate Scheduling Strategies

| Strategy | Formula/Method | Applicable Scenarios |
|----------|---------------|---------------------|
| Step Decay | $\eta \cdot \gamma^{\lfloor t/T \rfloor}$ | General |
| Cosine Annealing | $\eta_{\min} + \frac{1}{2}(\eta_0-\eta_{\min})(1+\cos(\pi t/T))$ | General |
| Warmup + Cosine | Linear increase then cosine annealing | Transformer |
| Reduce on Plateau | Reduce when validation loss stagnates | Scenarios requiring monitoring |

### Second-Order Methods

| Method | Convergence Rate | Complexity |
|--------|-----------------|------------|
| Newton's method | Quadratic | $O(n^3)$ |
| BFGS | Superlinear | $O(n^2)$ |
| L-BFGS | Superlinear | $O(mn)$ |

### Practical Recommendations

1. **Transformer training**: AdamW + Warmup + Cosine
2. **CNN training**: SGD + Momentum + Step/Cosine
3. **Monitor learning rate**: Record learning rate for each update
4. **Gradient clipping**: Essential for RNN, Transformer training
5. **Learning rate finder**: Use to determine initial learning rate when uncertain

---

**Previous Section**: [Adaptive Learning Rate Methods](05c-adaptive-learning-rate_EN.md)

**Next Chapter**: [Chapter 6: Elementary Functions](06-elementary-functions_EN.md) - Learn mathematical functions such as exponentials, logarithms, and activation functions in deep learning.

**Return**: [Chapter 5: Optimization Methods](05-optimization_EN.md) | [Math Fundamentals Tutorial Table of Contents](../math-fundamentals.md)
