# Chapter 5(b): Momentum Methods and Acceleration Techniques

While gradient descent is simple and effective, it can converge slowly in certain situations. This section will introduce momentum methods and acceleration techniques, which accelerate convergence by leveraging historical gradient information, particularly excelling in narrow valley (high condition number) scenarios.

---

## Table of Contents

1. [Momentum Method](#momentum-method)
2. [Nesterov Accelerated Gradient (NAG)](#nesterov-accelerated-gradient-nag)
3. [Convergence Rate Analysis](#convergence-rate-analysis)
4. [Intuitive Understanding of Acceleration Techniques](#intuitive-understanding-of-acceleration-techniques)
5. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Momentum Method

### Motivation

Problems with standard gradient descent:
1. **Oscillation**: In high condition number problems, gradient direction may point in the wrong direction
2. **Slow convergence**: Moving slowly in flat regions
3. **Stuck in local optima**: Lack of "momentum" to escape shallow local optima

### Physical Analogy

Imagine a ball rolling down a mountain:
- **Velocity**: Accumulates kinetic energy, has inertia
- **Momentum**: Current velocity + influence of historical velocity
- **Effect**: Can rush over small depressions, accelerate along main direction

### Momentum Formula

**Standard Momentum** (PyTorch style):

$$
\begin{align}
\mathbf{v}_{t+1} &= \beta \mathbf{v}_t + \nabla f(\mathbf{x}_t) \\
\mathbf{x}_{t+1} &= \mathbf{x}_t - \eta \mathbf{v}_{t+1}
\end{align}
$$

Where:
- $\beta \in [0, 1)$ is the **momentum coefficient** (typically 0.9)
- $\mathbf{v}_t$ is the **velocity** (cumulative gradient)

**Expanded Form**:

$$
\mathbf{v}_t = \sum_{k=0}^{t-1} \beta^{t-1-k} \nabla f(\mathbf{x}_k)
$$

### Effect of Momentum Coefficient

| $\beta$ Value | Effect |
|---------------|--------|
| $\beta = 0$ | Reduces to standard GD |
| $\beta = 0.5$ | Weak momentum, small historical influence |
| $\beta = 0.9$ | Standard momentum (recommended) |
| $\beta = 0.99$ | Strong momentum, may over-smooth |

### Exponential Moving Average Interpretation

Momentum is essentially the **exponential moving average** (EMA) of gradients:

$$
\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta) \nabla f(\mathbf{x}_t)
$$

Effective window size is approximately $\frac{1}{1-\beta}$:
- $\beta = 0.9$: Average of about 10 steps of history
- $\beta = 0.99$: Average of about 100 steps of history

### Advantages of Momentum

1. **Accelerated convergence**: Accumulates velocity in consistent directions
2. **Damped oscillation**: Cancels out in oscillating directions
3. **Escape local optima**: Uses inertia to rush past shallow depressions

```python
import numpy as np

def sgd_with_momentum(grad_f, x0, learning_rate=0.01, momentum=0.9,
                       max_iters=1000, tol=1e-6):
    """
    Stochastic gradient descent with momentum

    Args:
        grad_f: Gradient function
        x0: Initial point
        learning_rate: Learning rate
        momentum: Momentum coefficient
        max_iters: Maximum number of iterations
        tol: Convergence tolerance
    """
    x = x0.copy()
    v = np.zeros_like(x)  # Velocity
    history = {'x': [x.copy()], 'f': [], 'grad_norm': []}

    for i in range(max_iters):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g)
        history['grad_norm'].append(grad_norm)

        if grad_norm < tol:
            print(f"Converged after {i} iterations")
            break

        # Momentum update
        v = momentum * v + g
        x = x - learning_rate * v

        history['x'].append(x.copy())

    return x, history

# Example: Rosenbrock function (classic test function)
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])

x0 = np.array([-1.0, 1.0])
x_opt, history = sgd_with_momentum(rosenbrock_grad, x0, learning_rate=0.001, momentum=0.9)
print(f"Optimal solution: {x_opt}")
print(f"Number of iterations: {len(history['x'])}")
```

---

## Nesterov Accelerated Gradient (NAG)

### Motivation

Standard momentum computes gradients at the current position, but velocity has already been accumulated. Nesterov proposed: **compute gradients at the "future position"**.

### NAG Formula

$$
\begin{align}
\mathbf{x}_{t+1/2} &= \mathbf{x}_t - \eta \beta \mathbf{v}_t \quad \text{(predicted position)} \\
\mathbf{v}_{t+1} &= \beta \mathbf{v}_t + \nabla f(\mathbf{x}_{t+1/2}) \\
\mathbf{x}_{t+1} &= \mathbf{x}_t - \eta \mathbf{v}_{t+1}
\end{align}
$$

**Simplified Form**:

$$
\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\mathbf{x}_t - \eta \beta \mathbf{v}_t)
$$

### NAG vs Standard Momentum

```
Standard momentum:
  Current position → Compute gradient → Apply velocity

NAG:
  Current position → Predicted position → Compute gradient → Apply velocity
       ↓           ↑
       └───────────┘
         Look ahead one step
```

### Advantages of NAG

1. **Smarter update**: Evaluates gradient at predicted position
2. **Early deceleration**: Decelerates faster when approaching optimum
3. **Theoretical guarantee**: Achieves $O(1/t^2)$ convergence rate for convex functions

### NAG Visualization

```python
def nesterov_accelerated_gradient(f, grad_f, x0, learning_rate=0.01,
                                    momentum=0.9, max_iters=1000, tol=1e-6):
    """
    Nesterov Accelerated Gradient

    Args:
        f: Objective function
        grad_f: Gradient function
        x0: Initial point
        learning_rate: Learning rate
        momentum: Momentum coefficient
        max_iters: Maximum number of iterations
        tol: Convergence tolerance
    """
    x = x0.copy()
    v = np.zeros_like(x)
    history = {'x': [x.copy()], 'f': [f(x)]}

    for i in range(max_iters):
        # Compute gradient at "future position"
        look_ahead = x - learning_rate * momentum * v
        g = grad_f(look_ahead)

        if np.linalg.norm(g) < tol:
            print(f"Converged after {i} iterations")
            break

        # NAG update
        v = momentum * v + g
        x = x - learning_rate * v

        history['x'].append(x.copy())
        history['f'].append(f(x))

    return x, history

def compare_momentum_methods():
    """Compare standard momentum and NAG"""
    import matplotlib.pyplot as plt

    # Define objective function
    def f(x):
        return x[0]**2 + 10*x[1]**2

    def grad_f(x):
        return np.array([2*x[0], 20*x[1]])

    x0 = np.array([2.0, 2.0])

    # Standard momentum
    _, hist_momentum = sgd_with_momentum(grad_f, x0, 0.05, 0.9, 100)

    # NAG
    _, hist_nag = nesterov_accelerated_gradient(f, grad_f, x0, 0.05, 0.9, 100)

    # Create contour plot
    x1_range = np.linspace(-2.5, 2.5, 100)
    x2_range = np.linspace(-1.5, 2.5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = X1**2 + 10*X2**2

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, traj, title in [
        (axes[0], hist_momentum['x'], 'Standard Momentum'),
        (axes[1], hist_nag['x'], 'Nesterov Accelerated Gradient')
    ]:
        ax.contour(X1, X2, Z, levels=20, cmap='viridis')
        trajectory = np.array(traj)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', linewidth=1.5, markersize=3)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, zorder=5, label='Start')
        ax.scatter(0, 0, color='red', s=100, zorder=5, label='Optimal Point')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

compare_momentum_methods()
```

---

## Convergence Rate Analysis

### Convergence Rate Comparison

For $\mu$-strongly convex and $L$-smooth functions, with condition number $\kappa = L/\mu$:

| Method | Convergence Rate | Iterations needed for $\epsilon$ accuracy |
|--------|------------------|--------------------------------------------|
| Gradient Descent | $O((1-1/\kappa)^t)$ | $O(\kappa \log(1/\epsilon))$ |
| Standard Momentum | $O((1-\sqrt{1/\kappa})^t)$ | $O(\sqrt{\kappa} \log(1/\epsilon))$ |
| NAG | $O((1-\sqrt{1/\kappa})^t)$ | $O(\sqrt{\kappa} \log(1/\epsilon))$ |

**Key Improvement**: Momentum methods improve dependence from $\kappa$ to $\sqrt{\kappa}$!

### Momentum Convergence Proof Sketch

**Theorem**: For $\mu$-strongly convex and $L$-smooth quadratic function $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top\mathbf{Q}\mathbf{x}$, momentum converges fastest at $\eta = \frac{4}{(\sqrt{L}+\sqrt{\mu})^2}$, $\beta = \frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}}$.

**Proof Steps**:

**Step 1**: Write momentum as a second-order recurrence.

Momentum:
$$\mathbf{v}_{t+1} = \beta\mathbf{v}_t + \nabla f(\mathbf{x}_t)$$
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\mathbf{v}_{t+1}$$

Substitute $\mathbf{v}_{t+1}$:
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta(\beta\mathbf{v}_t + \nabla f(\mathbf{x}_t))$$

Using $\eta\mathbf{v}_t = \mathbf{x}_t - \mathbf{x}_{t-1}$:
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\nabla f(\mathbf{x}_t) + \beta(\mathbf{x}_t - \mathbf{x}_{t-1})$$

**Step 2**: For quadratic function $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top\mathbf{Q}\mathbf{x}$, we have $\nabla f(\mathbf{x}) = \mathbf{Qx}$.

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\mathbf{Qx}_t + \beta(\mathbf{x}_t - \mathbf{x}_{t-1})$$

**Step 3**: Analyze eigen-directions.

Let $\mathbf{q}_i$ be eigenvector of $\mathbf{Q}$ with eigenvalue $\lambda_i \in [\mu, L]$.

Error in eigen-direction: $e_t^{(i)} = \mathbf{q}_i^\top(\mathbf{x}_t - \mathbf{x}^*)$

Error recurrence:
$$e_{t+1}^{(i)} = e_t^{(i)} - \eta\lambda_i e_t^{(i)} + \beta(e_t^{(i)} - e_{t-1}^{(i)})$$

**Step 4**: Analyze characteristic equation.

This is a second-order linear recurrence, characteristic equation is:
$$r^2 - (1 - \eta\lambda_i + \beta)r + \beta = 0$$

Roots:
$$r = \frac{(1-\eta\lambda_i+\beta) \pm \sqrt{(1-\eta\lambda_i+\beta)^2 - 4\beta}}{2}$$

**Step 5**: Optimal parameter selection.

For fastest convergence, need to minimize $\max_i |r_i|$. Through analysis, optimal parameters:
$$\eta^* = \frac{4}{(\sqrt{L}+\sqrt{\mu})^2}, \quad \beta^* = \frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}} = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$$

Convergence rate:
$$|r| \leq \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} = 1 - \frac{2}{\sqrt{\kappa}+1} \approx 1 - \frac{1}{\sqrt{\kappa}}$$

$$\boxed{\text{Momentum convergence rate} = O\left(\left(1 - \frac{1}{\sqrt{\kappa}}\right)^t\right)}$$

**Comparison**: Gradient descent convergence rate is $O\left(\left(1 - \frac{1}{\kappa}\right)^t\right)$

When $\kappa = 100$:
- GD: $(0.99)^t$
- Momentum: $(0.82)^t$

To reach $10^{-6}$ accuracy:
- GD: About 1400 iterations
- Momentum: About 70 iterations

### Convergence Rate for Convex Functions (Not Strongly Convex)

| Method | Convergence Rate |
|--------|-----------------|
| Gradient Descent | $O(1/t)$ |
| Standard Momentum | $O(1/t)$ |
| NAG | $O(1/t^2)$ |

**Unique Advantage of NAG**: Achieves $O(1/t^2)$ convergence rate in general convex cases.

### Numerical Example

```python
def compare_convergence_rates():
    """Compare convergence rates of different methods"""
    import matplotlib.pyplot as plt

    # Define objective function (high condition number)
    def f(x):
        return x[0]**2 + 100*x[1]**2

    def grad_f(x):
        return np.array([2*x[0], 200*x[1]])

    x0 = np.array([2.0, 2.0])

    # Standard gradient descent
    def gradient_descent(grad_f, x0, lr, max_iters=200):
        x = x0.copy()
        history = [f(x)]
        for _ in range(max_iters):
            x = x - lr * grad_f(x)
            history.append(f(x))
        return history

    # Standard momentum
    _, hist_momentum = sgd_with_momentum(grad_f, x0, 0.01, 0.9, 200)

    # NAG
    _, hist_nag = nesterov_accelerated_gradient(f, grad_f, x0, 0.01, 0.9, 200)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(gradient_descent(grad_f, x0, 0.01), label='Gradient Descent', linewidth=2)
    plt.plot([f(x) for x in hist_momentum['x']], label='Standard Momentum', linewidth=2)
    plt.plot(hist_nag['f'], label='NAG', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Convergence Curve (Linear Scale)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.semilogy(gradient_descent(grad_f, x0, 0.01), label='Gradient Descent', linewidth=2)
    plt.semilogy([f(x) for x in hist_momentum['x']], label='Standard Momentum', linewidth=2)
    plt.semilogy(hist_nag['f'], label='NAG', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value (Log)')
    plt.title('Convergence Curve (Log Scale)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

compare_convergence_rates()
```

---

## Intuitive Understanding of Acceleration Techniques

### Why Momentum Works?

**1. Direction Consistency**
- If consecutive gradient directions are consistent, momentum accumulates velocity
- If gradient directions oscillate, momentum cancels out

**2. Oscillation Damping**

```
Without momentum:            With momentum:
  ↑                          ↑
  │  ·  →  ·                 │    ·→→→·
  │    ↗                     │      ↓
  │  ·  ←  ·                 │    ·←←←·

  Oscillatory progress       Smooth progress
```

**3. Valley Problem**

```
Valley cross-section:

        ╱╲
       ╱  ╲
      ╱    ╲    Without momentum: oscillatory descent
     ╱      ╲   With momentum: smooth descent
    ╱        ╲
   ╱          ╲
  ╱            ╲
```

### NAG's Look-ahead Advantage

NAG computes gradients at the predicted position, equivalent to "anticipating":

```
Current position: ●
Predicted position:  ○
True gradient:    ↓
NAG gradient:  ↘ (more accurate direction)
```

When close to optimum, NAG can sense it earlier and decelerate.

### Momentum Parameter Selection Guide

| Scenario | Recommended $\beta$ | Explanation |
|----------|-------------------|-------------|
| General training | 0.9 | Balances stability and speed |
| Deep networks | 0.9-0.99 | Stronger accumulation effect |
| Noisy gradients | 0.99 | Stronger smoothing |
| Fine-tuning | 0.5-0.7 | Faster response |

---

## Applications in Deep Learning

### PyTorch-style Implementation

```python
class SGDMomentum:
    """SGD Optimizer with Momentum (PyTorch style)"""

    def __init__(self, params, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        """Execute one update step"""
        for key in params:
            # Update velocity
            self.velocities[key] = self.momentum * self.velocities[key] + grads[key]
            # Update parameters
            params[key] -= self.lr * self.velocities[key]

    def zero_grad(self):
        """Clear gradients (placeholder)"""
        pass


class NAG:
    """Nesterov Accelerated Gradient Optimizer"""

    def __init__(self, params, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads_func):
        """Execute one update step (requires gradient function to compute gradient at predicted position)"""
        for key in params:
            # Compute predicted position
            look_ahead = params[key] - self.lr * self.momentum * self.velocities[key]
            # Compute gradient at predicted position (simplified implementation)
            # In practice, need to re-forward propagate
            params[key] = look_ahead

        # Re-compute gradients (simplified here)
        grads = grads_func(params)

        for key in params:
            # Update velocity
            self.velocities[key] = self.momentum * self.velocities[key] + grads[key]
            # Update from original position
            params[key] -= self.lr * self.velocities[key]
```

### Usage in nanotorch

```python
from nanotorch.optim import SGD
from nanotorch import Tensor
from nanotorch.nn import Linear, ReLU, Sequential

# Create model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# Use SGD with momentum
optimizer = SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=False  # Set to True to use NAG
)

# Training loop
for epoch in range(100):
    for batch_x, batch_y in train_loader:
        # Forward propagation
        output = model(Tensor(batch_x))
        loss = criterion(output, Tensor(batch_y))

        # Backward propagation
        optimizer.zero_grad()
        loss.backward()

        # Parameter update
        optimizer.step()
```

### Practical Recommendations

1. **Use momentum by default**
   - Almost always better than pure SGD
   - $\beta = 0.9$ is a safe default

2. **NAG vs Standard Momentum**
   - NAG theoretically converges faster
   - In practice, difference may not be obvious
   - Try both, choose the better one

3. **Coordinate with learning rate scheduling**
   - Momentum doesn't affect learning rate scheduling strategies
   - Can work with StepLR, CosineAnnealing, etc.

4. **Monitor training curves**
   - Sudden increase in gradient norm may require reducing learning rate
   - Training oscillation may require increasing momentum or reducing learning rate

---

## Summary

This section introduced momentum methods and acceleration techniques:

| Method | Formula | Convergence Rate (Strongly Convex) | Characteristics |
|--------|---------|-----------------------------------|----------------|
| Gradient Descent | $x_{t+1} = x_t - \eta \nabla f$ | $O((1-1/\kappa)^t)$ | Simple, slow |
| Standard Momentum | $v_{t+1} = \beta v_t + \nabla f$ | $O((1-\sqrt{1/\kappa})^t)$ | Stable, fast |
| NAG | $v_{t+1} = \beta v_t + \nabla f(x_t - \eta \beta v_t)$ | $O((1-\sqrt{1/\kappa})^t)$ | Look-ahead, faster |

**Key Takeaways**:
- Momentum accelerates convergence by accumulating historical gradient information
- Momentum is significantly effective on high condition number problems
- NAG further improves through look-ahead mechanism
- $\beta = 0.9$ is a commonly used default value

---

**Previous Section**: [Optimization Basics and Gradient Descent](05a-optimization-basics-gradient-descent_EN.md)

**Next Section**: [Adaptive Learning Rate Methods](05c-adaptive-learning-rate_EN.md) - Learn AdaGrad, RMSprop, Adam, and other adaptive optimization algorithms.

**Return**: [Chapter 5: Optimization Methods](05-optimization.md) | [Math Fundamentals Tutorial Table of Contents](../math-fundamentals.md)
