# Chapter 5: Optimization Methods

Optimization is the **core engine of machine learning**. From minimizing loss functions to finding optimal parameters, optimization theory provides the mathematical foundation and practical algorithms for deep learning. This chapter systematically introduces core concepts of optimization methods and their applications in deep learning.

---

## Chapter Structure

For ease of learning and in-depth understanding, this chapter is divided into four sub-chapters:

### [5.1 Optimization Basics and Gradient Descent](05a-optimization-basics-gradient-descent_EN.md)

**Content Overview**:
- General form of optimization problems and optimality conditions
- Convex sets, convex functions, and importance of convex optimization
- Principles and implementation of gradient descent algorithm
- Step size selection strategies (fixed step size, line search, backtracking)
- Convergence analysis and impact of condition number
- Stochastic gradient descent and Mini-batch

**Core Concepts**:
| Concept | Formula | Application in Deep Learning |
|---------|---------|----------------------------|
| Gradient descent | $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$ | Parameter updates |
| Convex function | $f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$ | Global optimum guarantee |
| Strong convexity | $\nabla^2 f \succeq \mu I$ | Linear convergence |
| Condition number | $\kappa = L/\mu$ | Convergence speed |

**[Start Learning →](05a-optimization-basics-gradient-descent_EN.md)**

---

### [5.2 Momentum Methods and Acceleration Techniques](05b-momentum-acceleration_EN.md)

**Content Overview**:
- Physical analogy and formulas of Momentum method
- Role and selection of momentum coefficient
- Nesterov Accelerated Gradient (NAG) lookahead mechanism
- Convergence rate comparison: $O(1/t)$ vs $O(1/t^2)$
- Advantages of momentum on high condition number problems

**Core Concepts**:
| Method | Update Formula | Convergence Rate (Strongly Convex) |
|--------|---------------|-----------------------------------|
| Gradient descent | $x_{t+1} = x_t - \eta \nabla f$ | $O((1-1/\kappa)^t)$ |
| Standard momentum | $v_{t+1} = \beta v_t + \nabla f$ | $O((1-\sqrt{1/\kappa})^t)$ |
| NAG | $v_{t+1} = \beta v_t + \nabla f(x_t - \eta\beta v_t)$ | $O((1-\sqrt{1/\kappa})^t)$ |

**[Start Learning →](05b-momentum-acceleration_EN.md)**

---

### [5.3 Adaptive Learning Rate Methods](05c-adaptive-learning-rate_EN.md)

**Content Overview**:
- Motivation for adaptive learning rates
- AdaGrad: Accumulating gradient squares to adjust learning rate
- RMSprop: Exponential moving average to solve learning rate decay
- Adam: Momentum + adaptive learning rate
- AdamW: Decoupled weight decay
- Optimizer selection guide

**Core Concepts**:
| Method | Core Idea | Suitable Scenarios |
|--------|-----------|-------------------|
| AdaGrad | Accumulated gradient squares | Sparse data |
| RMSprop | EMA of gradient squares | RNNs, non-stationary objectives |
| Adam | Momentum + adaptive | General, rapid prototyping |
| AdamW | Adam + decoupled regularization | Transformers |

**[Start Learning →](05c-adaptive-learning-rate_EN.md)**

---

### [5.4 Learning Rate Scheduling and Advanced Techniques](05d-lr-scheduling-advanced_EN.md)

**Content Overview**:
- Importance of learning rate scheduling
- Common strategies: Step Decay, Cosine Annealing, Warmup
- Reduce on Plateau and Cyclic LR
- Second-order optimization methods: Newton's method, BFGS, L-BFGS
- Constrained optimization and projected gradient descent
- Gradient clipping, learning rate finder

**Core Concepts**:
| Strategy | Formula/Method | Suitable Scenarios |
|----------|---------------|-------------------|
| Step Decay | $\eta \cdot \gamma^{\lfloor t/T \rfloor}$ | General |
| Cosine Annealing | $\eta_{\min} + \frac{1}{2}(\eta_0-\eta_{\min})(1+\cos(\pi t/T))$ | General |
| Warmup + Cosine | Linear increase then cosine annealing | Transformers |

**[Start Learning →](05d-lr-scheduling-advanced_EN.md)**

---

## Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│               Chapter 5: Optimization Methods                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  5.1 Optimization → 5.2 Momentum → 5.3 Adaptive → 5.4 LR    │
│  Basics &          & Acceleration  Learning Rate  Scheduling │
│  Gradient Descent    Techniques    Methods       & Advanced │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Convex   │   │ Momentum │   │ AdaGrad  │   │ Cosine   │ │
│  │ Gradient │   │ NAG      │   │ RMSprop  │   │ Warmup   │ │
│  │ Descent  │   │ Convergence│   │ Adam     │   │ 2nd Order│ │
│  │ Convergence│  │ Rate     │   │          │   │ Methods  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                             │
│  Applications: SGD, Adam, AdamW, LR scheduling, Grad clipping│
└─────────────────────────────────────────────────────────────┘
```

## Why is Optimization Important for Deep Learning?

### 1. Training is Optimization

```
Training neural networks = Finding optimal parameters
     ↓
Minimize loss function: min_θ L(θ)
     ↓
Use optimization algorithms to iteratively update parameters
```

### 2. Impact of Optimizer Choice

| Aspect | Impact of Optimizer |
|--------|---------------------|
| Convergence speed | Adam usually faster than SGD |
| Generalization performance | SGD + Momentum may be better |
| Stability | Adaptive methods more stable |
| Memory usage | SGD minimal, Adam needs 2x |

### 3. Necessity of Learning Rate Scheduling

- **Early training**: Need larger learning rate for fast descent
- **Mid training**: Need stable learning
- **Late training**: Need small learning rate for fine-tuning

---

## Core Formula Quick Reference

### Gradient Descent

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)
$$

### Momentum Method

$$
\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\mathbf{x}_t)
$$

### Adam

$$
\mathbf{m}_{t+1} = \beta_1 \mathbf{m}_t + (1-\beta_1) \mathbf{g}_t
$$

$$
\mathbf{v}_{t+1} = \beta_2 \mathbf{v}_t + (1-\beta_2) \mathbf{g}_t^2
$$

### Cosine Annealing

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi t/T))
$$

---

## Python Code Examples

### Gradient Descent

```python
import numpy as np

def gradient_descent(f, grad_f, x0, lr=0.01, max_iters=1000):
    x = x0.copy()
    for _ in range(max_iters):
        x = x - lr * grad_f(x)
    return x

# Example
f = lambda x: x[0]**2 + 2*x[1]**2
grad_f = lambda x: np.array([2*x[0], 4*x[1]])
x0 = np.array([2.0, 2.0])
x_opt = gradient_descent(f, grad_f, x0, lr=0.1)
print(f"Optimal solution: {x_opt}")
```

### Adam Optimizer

```python
class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999)):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
    
    def step(self, params, grads):
        self.t += 1
        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
```

### Cosine Annealing Scheduler

```python
class CosineAnnealingLR:
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
```

---

## Study Recommendations

1. **Start from basics**: Understand gradient descent first, then learn advanced methods
2. **Hands-on implementation**: Implement SGD, Momentum, Adam yourself
3. **Experimental comparison**: Compare different optimizers on the same problem
4. **Understand tradeoffs**: Convergence speed vs generalization performance
5. **Focus on learning rate**: This is the most important hyperparameter

---

## Further Reading

- [Chapter 4: Mathematical Statistics](04-statistics_EN.md) - Parameter estimation and hypothesis testing
- [Chapter 6: Elementary Functions](06-elementary-functions_EN.md) - Activation functions and loss functions

---

**Back to**: [Mathematical Foundations Directory](../math-fundamentals_EN.md)
