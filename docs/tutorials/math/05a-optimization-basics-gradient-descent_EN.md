# Chapter 5(a): Optimization Basics and Gradient Descent

Optimization is the **core engine of machine learning**. From minimizing loss functions to finding optimal parameters, optimization theory provides the mathematical foundation for deep learning. This section will introduce the basic concepts of optimization problems, convex optimization theory, and gradient descent algorithms.

---

## Table of Contents

1. [Overview of Optimization Problems](#overview-of-optimization-problems)
2. [Convex Optimization Fundamentals](#convex-optimization-fundamentals)
3. [Gradient Descent Method](#gradient-descent-method)
4. [Step Size Selection Strategies](#step-size-selection-strategies)
5. [Convergence Analysis](#convergence-analysis)
6. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Overview of Optimization Problems

### General Form

The standard form of an optimization problem is:

$$
\min_{\mathbf{x} \in \mathbb{R}^n} f(\mathbf{x})
$$

Where:
- $f: \mathbb{R}^n \to \mathbb{R}$ is the **objective function** (loss function)
- $\mathbf{x} = (x_1, x_2, \ldots, x_n)^\top$ is the **decision variable** (model parameters)

### Constrained Optimization Problems

Optimization problems with constraints:

$$
\begin{align}
\min_{\mathbf{x}} \quad & f(\mathbf{x}) \\
\text{s.t.} \quad & g_i(\mathbf{x}) \leq 0, \quad i = 1, \ldots, m \quad \text{(inequality constraints)} \\
& h_j(\mathbf{x}) = 0, \quad j = 1, \ldots, p \quad \text{(equality constraints)}
\end{align}
$$

### Optimality Conditions

**Global Optimum**: $\mathbf{x}^*$ satisfies $f(\mathbf{x}^*) \leq f(\mathbf{x}), \forall \mathbf{x} \in \mathbb{R}^n$

**Local Optimum**: $\mathbf{x}^*$ satisfies existence of $\epsilon > 0$, such that $f(\mathbf{x}^*) \leq f(\mathbf{x}), \forall \mathbf{x} \in B_\epsilon(\mathbf{x}^*)$

**First-order Necessary Condition** (unconstrained):

If $\mathbf{x}^*$ is a local optimum and $f$ is differentiable at $\mathbf{x}^*$, then:

$$
\nabla f(\mathbf{x}^*) = \mathbf{0}
$$

**Second-order Necessary Condition**:

If $\mathbf{x}^*$ is a local optimum and $f$ is twice differentiable, then:

$$
\nabla^2 f(\mathbf{x}^*) \succeq 0 \quad \text{(Hessian matrix is positive semidefinite)}
$$

**Second-order Sufficient Condition**:

If $\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $\nabla^2 f(\mathbf{x}^*) \succ 0$ (positive definite), then $\mathbf{x}^*$ is a strict local optimum.

```python
import numpy as np

def check_optimality_conditions(f, grad_f, hess_f, x):
    """Check optimality conditions"""
    grad = grad_f(x)
    hess = hess_f(x)

    # First-order condition: is gradient close to zero
    grad_norm = np.linalg.norm(grad)
    first_order = grad_norm < 1e-6

    # Second-order condition: is Hessian positive semidefinite
    eigenvalues = np.linalg.eigvalsh(hess)
    second_order = np.all(eigenvalues >= -1e-10)

    return {
        'gradient_norm': grad_norm,
        'first_order_satisfied': first_order,
        'hessian_eigenvalues': eigenvalues,
        'second_order_satisfied': second_order
    }

# Example: f(x) = x^2
f = lambda x: x**2
grad_f = lambda x: 2*x
hess_f = lambda x: np.array([[2.0]])

result = check_optimality_conditions(f, grad_f, hess_f, np.array([0.0]))
print(f"Optimal point check: {result}")
```

---

## Convex Optimization Fundamentals

### Convex Sets

**Definition**: A set $C \subseteq \mathbb{R}^n$ is a convex set if and only if:

$$
\forall \mathbf{x}, \mathbf{y} \in C, \forall \theta \in [0, 1]: \quad \theta \mathbf{x} + (1 - \theta) \mathbf{y} \in C
$$

**Intuition**: The line segment connecting any two points in the set remains within the set.

**Common Convex Sets**:
- **Hyperplane**: $\{\mathbf{x} : \mathbf{a}^\top \mathbf{x} = b\}$
- **Half-space**: $\{\mathbf{x} : \mathbf{a}^\top \mathbf{x} \leq b\}$
- **Ball**: $\{\mathbf{x} : \|\mathbf{x} - \mathbf{x}_0\|_2 \leq r\}$
- **Norm ball**: $\{\mathbf{x} : \|\mathbf{x}\|_p \leq 1\}$
- **Polytope**: Intersection of finitely many half-spaces

### Convex Functions

**Definition**: A function $f: \mathbb{R}^n \to \mathbb{R}$ is convex if and only if:

$$
f(\theta \mathbf{x} + (1 - \theta) \mathbf{y}) \leq \theta f(\mathbf{x}) + (1 - \theta) f(\mathbf{y}), \quad \forall \theta \in [0, 1]
$$

**First-order Condition** (Jensen's inequality): If $f$ is differentiable, $f$ is convex if and only if:

$$
f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x}), \quad \forall \mathbf{x}, \mathbf{y}
$$

**Second-order Condition**: If $f$ is twice differentiable, $f$ is convex if and only if:

$$
\nabla^2 f(\mathbf{x}) \succeq 0, \quad \forall \mathbf{x}
$$

### Strong Convexity

A function $f$ is $\mu$-strongly convex if:

$$
f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top (\mathbf{y} - \mathbf{x}) + \frac{\mu}{2}\|\mathbf{y} - \mathbf{x}\|^2
$$

**Equivalent condition**: $\nabla^2 f(\mathbf{x}) \succeq \mu \mathbf{I}$

**Significance**: Strong convexity guarantees the objective function has a "bowl-shaped" form with a unique global optimum.

### Smoothness

A function $f$ is $L$-smooth (Lipschitz continuous gradient) if:

$$
\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L \|\mathbf{x} - \mathbf{y}\|
$$

**Equivalent condition**: $\nabla^2 f(\mathbf{x}) \preceq L \mathbf{I}$

### Importance of Convex Optimization

**Key Property**: For convex optimization problems, **local optimum = global optimum**

This means:
1. Stationary points ($\nabla f(\mathbf{x}) = 0$) must be global optima
2. Any local search algorithm can find the global optimum
3. Theoretical analysis is more straightforward

```python
import numpy as np
import matplotlib.pyplot as plt

# Convex function example: f(x) = x^2
def convex_func(x):
    return x ** 2

# Non-convex function example: f(x) = x^4 - 2x^2 + x
def non_convex_func(x):
    return x**4 - 2*x**2 + x

# Visualization
x = np.linspace(-2, 2, 100)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, convex_func(x), linewidth=2)
axes[0].set_title('Convex Function: $f(x) = x^2$')
axes[0].scatter([0], [0], color='red', s=100, zorder=5, label='Global Optimum')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(x, non_convex_func(x), linewidth=2)
axes[1].set_title('Non-convex Function: $f(x) = x^4 - 2x^2 + x$')
axes[1].scatter([-1.1, 0.9], [non_convex_func(-1.1), non_convex_func(0.9)],
                color=['blue', 'red'], s=100, zorder=5)
axes[1].legend(['Function', 'Local Optimum', 'Global Optimum'])
axes[1].grid(True)

plt.tight_layout()
plt.show()
```

---

## Gradient Descent Method

### Basic Algorithm

**Gradient descent** is the most fundamental first-order optimization method:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)
$$

Where $\eta > 0$ is the **learning rate** (step size).

### Intuitive Understanding

The gradient $\nabla f(\mathbf{x})$ points in the direction of the **fastest increase** of the function value, so moving along the negative gradient direction reduces the function value.

**Taylor Expansion Perspective**:

$$
f(\mathbf{x}_{t+1}) \approx f(\mathbf{x}_t) + \nabla f(\mathbf{x}_t)^\top (\mathbf{x}_{t+1} - \mathbf{x}_t) = f(\mathbf{x}_t) - \eta \|\nabla f(\mathbf{x}_t)\|^2
$$

As long as $\eta > 0$ and the gradient is non-zero, the function value will decrease.

### Geometric Interpretation of Gradient Descent

```
Trajectory on contour map:

    x₂
     ↑
     │    ∘ Starting point
     │   ∘
     │  ∘
     │ ∘
     │∘ → Along negative gradient direction
     └──────────→ x₁

Goal: Reach the lowest point (center of contours)
```

### Algorithm Implementation

```python
import numpy as np

def gradient_descent(f, grad_f, x0, learning_rate=0.01, max_iters=1000, tol=1e-6):
    """
    Gradient descent algorithm

    Args:
        f: Objective function
        grad_f: Gradient function
        x0: Initial point
        learning_rate: Learning rate
        max_iters: Maximum number of iterations
        tol: Convergence tolerance

    Returns:
        x: Optimal solution
        history: Iteration history
    """
    x = x0.copy()
    history = {'x': [x.copy()], 'f': [f(x)], 'grad_norm': []}

    for i in range(max_iters):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g)
        history['grad_norm'].append(grad_norm)

        # Check convergence
        if grad_norm < tol:
            print(f"Converged after {i} iterations")
            break

        # Gradient descent update
        x = x - learning_rate * g

        history['x'].append(x.copy())
        history['f'].append(f(x))

    return x, history

# Example: Minimize f(x) = x₁² + 2x₂²
def f(x):
    return x[0]**2 + 2 * x[1]**2

def grad_f(x):
    return np.array([2 * x[0], 4 * x[1]])

x0 = np.array([2.0, 2.0])
x_opt, history = gradient_descent(f, grad_f, x0, learning_rate=0.1)
print(f"Optimal solution: {x_opt}")
print(f"Number of iterations: {len(history['f'])}")
print(f"Optimal value: {history['f'][-1]:.6f}")
```

### Gradient Descent Visualization

```python
import matplotlib.pyplot as plt

def visualize_gradient_descent(f, grad_f, x0, lr, max_iters=50):
    """Visualize gradient descent process"""
    x, history = gradient_descent(f, grad_f, x0, lr, max_iters)

    # Create contour plot
    x1_range = np.linspace(-2.5, 2.5, 100)
    x2_range = np.linspace(-2.5, 2.5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = X1**2 + 2*X2**2

    fig, ax = plt.subplots(figsize=(10, 8))

    # Contours
    contour = ax.contour(X1, X2, Z, levels=20, cmap='viridis')
    ax.clabel(contour, inline=True, fontsize=8)

    # Gradient descent trajectory
    trajectory = np.array(history['x'])
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', linewidth=2, markersize=8)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=150, zorder=5, label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=150, zorder=5, label='End')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_title(f'Gradient Descent (lr={lr})')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    return x, history

# Comparison with different learning rates
x0 = np.array([2.0, 2.0])
for lr in [0.05, 0.1, 0.2]:
    x_opt, history = visualize_gradient_descent(f, grad_f, x0, lr)
    print(f"lr={lr}: final value={history['f'][-1]:.6f}")
```

---

## Step Size Selection Strategies

### Importance of Step Size

The learning rate $\eta$ is the most critical hyperparameter in gradient descent:
- **Too large**: May diverge or oscillate
- **Too small**: Converges too slowly
- **Just right**: Fast and stable convergence

### Fixed Step Size

$$
\eta = \text{const}
$$

**Theoretical optimal value**: For $L$-smooth functions, the optimal fixed step size is $\eta = \frac{1}{L}$.

### Exact Line Search

$$
\eta_t = \arg\min_\eta f(\mathbf{x}_t - \eta \nabla f(\mathbf{x}_t))
$$

**Advantages**: Optimal at each step
**Disadvantages**: High computational cost, usually infeasible

### Backtracking Line Search

**Armijo Condition** (sufficient decrease):

$$
f(\mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)) \leq f(\mathbf{x}_t) - c \eta \|\nabla f(\mathbf{x}_t)\|^2, \quad c \in (0, 1)
$$

**Algorithm**:
1. Initialize $\eta = \eta_0$ (e.g., 1.0)
2. If Armijo condition is not satisfied, $\eta := \beta \eta$ (e.g., $\beta = 0.5$)
3. Repeat until condition is satisfied

```python
def backtracking_line_search(f, grad_f, x, d, c=0.5, beta=0.8, max_iters=20):
    """
    Backtracking line search

    Args:
        f: Objective function
        grad_f: Gradient function
        x: Current point
        d: Search direction (usually negative gradient)
        c: Armijo parameter
        beta: Step size decay rate
        max_iters: Maximum backtracking iterations
    """
    eta = 1.0
    g = grad_f(x)
    f_x = f(x)

    for _ in range(max_iters):
        x_new = x + eta * d
        if f(x_new) <= f_x + c * eta * np.dot(g, d):
            return eta
        eta *= beta

    return eta

def gradient_descent_backtracking(f, grad_f, x0, max_iters=1000, tol=1e-6):
    """Gradient descent with backtracking line search"""
    x = x0.copy()
    history = {'x': [x.copy()], 'f': [f(x)], 'lr': []}

    for i in range(max_iters):
        g = grad_f(x)

        if np.linalg.norm(g) < tol:
            break

        # Backtracking line search to determine step size
        eta = backtracking_line_search(f, grad_f, x, -g)
        history['lr'].append(eta)

        x = x - eta * g
        history['x'].append(x.copy())
        history['f'].append(f(x))

    return x, history
```

### Decaying Step Size

For non-convex problems or stochastic optimization:

$$
\eta_t = \frac{\eta_0}{1 + \alpha t}
$$

Or

$$
\eta_t = \frac{\eta_0}{\sqrt{t}}
$$

### Step Size Selection Summary

| Strategy | Formula | Characteristics |
|----------|---------|-----------------|
| Fixed step size | $\eta = \frac{1}{L}$ | Need to know smoothness constant $L$ |
| Exact line search | $\eta_t = \arg\min_\eta f(x_t - \eta g_t)$ | High computational cost |
| Backtracking line search | Gradually decrease until Armijo satisfied | High practicality |
| Decaying step size | $\eta_t = \eta_0/(1+\alpha t)$ | Suitable for non-convex problems |

---

## Convergence Analysis

### Convergence Rate Definitions

Let $\{\mathbf{x}_t\}$ be the sequence generated by an optimization algorithm, and $\mathbf{x}^*$ be the optimal solution.

| Type | Definition | Meaning |
|------|-----------|---------|
| Sublinear | $\|\mathbf{x}_t - \mathbf{x}^*\| = O(1/t)$ | Fixed number of iterations needed to double precision |
| Linear | $\|\mathbf{x}_t - \mathbf{x}^*\| = O(r^t), r < 1$ | Precision multiplied by constant each iteration |
| Superlinear | $\|\mathbf{x}_{t+1} - \mathbf{x}^*\| / \|\mathbf{x}_t - \mathbf{x}^*\| \to 0$ | Converges faster and faster |
| Quadratic | $\|\mathbf{x}_{t+1} - \mathbf{x}^*\| = O(\|\mathbf{x}_t - \mathbf{x}^*\|^2)$ | Precision digits double |

### Convergence for Convex Functions

**Theorem** (Convex + $L$-smooth): Let $f$ be convex and $L$-smooth, using fixed step size $\eta \leq \frac{1}{L}$, then:

$$
f(\mathbf{x}_T) - f(\mathbf{x}^*) \leq \frac{\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{2 \eta T}
$$

**Proof of gradient descent convergence for convex functions**:

**Step 1**: Use $L$-smoothness (Lipschitz continuous gradient).

If $f$ is $L$-smooth, then for any $\mathbf{x}, \mathbf{y}$:

$$f(\mathbf{y}) \leq f(\mathbf{x}) + \nabla f(\mathbf{x})^\top(\mathbf{y} - \mathbf{x}) + \frac{L}{2}\|\mathbf{y} - \mathbf{x}\|^2$$

**Step 2**: Let $\mathbf{y} = \mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$.

$$f(\mathbf{x}_{t+1}) \leq f(\mathbf{x}_t) + \nabla f(\mathbf{x}_t)^\top(-\eta \nabla f(\mathbf{x}_t)) + \frac{L}{2}\|\eta \nabla f(\mathbf{x}_t)\|^2$$

$$= f(\mathbf{x}_t) - \eta \|\nabla f(\mathbf{x}_t)\|^2 + \frac{L\eta^2}{2}\|\nabla f(\mathbf{x}_t)\|^2$$

$$= f(\mathbf{x}_t) - \eta\left(1 - \frac{L\eta}{2}\right)\|\nabla f(\mathbf{x}_t)\|^2$$

**Step 3**: When $\eta \leq \frac{1}{L}$, $1 - \frac{L\eta}{2} \geq \frac{1}{2}$, therefore:

$$f(\mathbf{x}_{t+1}) \leq f(\mathbf{x}_t) - \frac{\eta}{2}\|\nabla f(\mathbf{x}_t)\|^2$$

**Step 4**: Use convexity, $f(\mathbf{x}^*) \geq f(\mathbf{x}_t) + \nabla f(\mathbf{x}_t)^\top(\mathbf{x}^* - \mathbf{x}_t)$.

Rearranging: $\|\nabla f(\mathbf{x}_t)\| \cdot \|\mathbf{x}_t - \mathbf{x}^*\| \geq f(\mathbf{x}_t) - f(\mathbf{x}^*)$

**Step 5**: Combining the above results, we can prove:

$$f(\mathbf{x}_T) - f(\mathbf{x}^*) \leq \frac{\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{2\eta T}$$

**Conclusion**: $O(1/T)$ convergence rate (sublinear).

### Convergence for Strongly Convex Functions

**Theorem** ($\mu$-strongly convex + $L$-smooth): Let $f$ be $\mu$-strongly convex and $L$-smooth, with condition number $\kappa = L/\mu$, using optimal step size $\eta = 1/L$, then:

$$
\|\mathbf{x}_T - \mathbf{x}^*\|^2 \leq \left(1 - \frac{1}{\kappa}\right)^T \|\mathbf{x}_0 - \mathbf{x}^*\|^2
$$

**Conclusion**: Linear convergence rate, convergence speed depends on condition number $\kappa$.

### Impact of Condition Number

The **condition number** $\kappa = L/\mu$ reflects the "difficulty" of the problem:
- $\kappa$ small (close to 1): Objective function has nearly circular contours, easy to optimize
- $\kappa$ large: Objective function is elongated (narrow valley), difficult to optimize

```
Small condition number (κ ≈ 1)          Large condition number (κ >> 1)
    ○                       ═══
  ○   ○                   ═══════
 ○     ○                ═══════════
○       ○              ═══════════════
 ○     ○                ═══════════
  ○   ○                   ═══════
    ○                       ═══

  Fast convergence              Oscillatory convergence
```

```python
def compare_condition_numbers():
    """Compare the effect of different condition numbers on convergence"""
    import matplotlib.pyplot as plt

    # Function with condition number 2: f(x) = x₁² + 2x₂²
    def f_easy(x):
        return x[0]**2 + 2*x[1]**2

    def grad_easy(x):
        return np.array([2*x[0], 4*x[1]])

    # Function with condition number 10: f(x) = x₁² + 10x₂²
    def f_hard(x):
        return x[0]**2 + 10*x[1]**2

    def grad_hard(x):
        return np.array([2*x[0], 20*x[1]])

    x0 = np.array([2.0, 2.0])

    _, hist_easy = gradient_descent(f_easy, grad_easy, x0, 0.1, 100)
    _, hist_hard = gradient_descent(f_hard, grad_hard, x0, 0.05, 100)

    plt.figure(figsize=(10, 6))
    plt.plot(hist_easy['f'], label='κ=2 (Easy)', linewidth=2)
    plt.plot(hist_hard['f'], label='κ=10 (Hard)', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.title('Effect of Condition Number on Convergence Speed')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.show()

compare_condition_numbers()
```

---

## Applications in Deep Learning

### Stochastic Gradient Descent (SGD)

In deep learning, the objective function is typically the average of training sample losses:

$$
f(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^N \ell(\mathbf{w}; \mathbf{x}_i, y_i)
$$

**Stochastic Gradient**: Use a single sample or mini-batch to estimate the gradient:

$$
\mathbf{g}_t = \frac{1}{B} \sum_{i \in \mathcal{B}_t} \nabla \ell(\mathbf{w}_t; \mathbf{x}_i, y_i) \approx \nabla f(\mathbf{w}_t)
$$

```python
def sgd_step(params, grads, lr):
    """SGD parameter update"""
    for key in params:
        params[key] -= lr * grads[key]
    return params

def mini_batch_sgd(model, data_loader, lr, epochs):
    """Mini-batch SGD training loop"""
    for epoch in range(epochs):
        for batch_x, batch_y in data_loader:
            # Forward propagation
            output = model.forward(batch_x)

            # Compute loss and gradients
            loss = model.compute_loss(output, batch_y)
            grads = model.backward()

            # SGD update
            model.params = sgd_step(model.params, grads, lr)
```

### Non-convexity in Deep Learning

Neural network loss functions are **non-convex**:
- Many local optima exist
- Saddle points exist
- Local optimum ≠ global optimum

**Empirical Findings**:
- Most local optima have similar quality
- Saddle points are more common than local optima
- SGD noise helps escape saddle points

### Practical Recommendations

1. **Learning Rate Selection**
   - Start with a small value (e.g., 0.001)
   - Use a learning rate finder
   - Monitor training curves

2. **Batch Size**
   - Small batch: Strong regularization effect, high noise
   - Large batch: Fast convergence, possibly poor generalization

3. **Initialization**
   - Use appropriate weight initialization (Xavier, He)
   - Avoid all-zero initialization

---

## Summary

This section introduced optimization basics and gradient descent core concepts:

| Concept | Definition/Formula | Importance |
|---------|------------------|------------|
| Convex Function | $f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$ | Local optimum = global optimum |
| Strong Convexity | $\nabla^2 f \succeq \mu I$ | Guarantees linear convergence |
| Smoothness | $\nabla^2 f \preceq L I$ | Determines maximum step size |
| Gradient Descent | $x_{t+1} = x_t - \eta \nabla f(x_t)$ | Most fundamental first-order method |
| Condition Number | $\kappa = L/\mu$ | Determines convergence difficulty |

**Key Takeaways**:
- Convex optimization has theoretical guarantees, but deep learning is non-convex
- Learning rate is the most important hyperparameter
- Condition number affects convergence speed
- Stochastic gradient descent is the standard method for deep learning

---

**Next Section**: [Momentum Methods and Acceleration Techniques](05b-momentum-acceleration_EN.md) - Learn momentum methods, Nesterov accelerated gradient, and other advanced optimization techniques.

**Return**: [Chapter 5: Optimization Methods](05-optimization.md) | [Math Fundamentals Tutorial Table of Contents](../math-fundamentals.md)
