# Chapter 7: Sequences and Series

Sequences and series are **the mathematical foundation of deep learning**. From learning rate decay strategies to recursive relationships in sequence modeling, from positional encoding in attention mechanisms to sequence processing in Transformers, the concepts of sequences and series are everywhere. This chapter will systematically introduce the core concepts of sequences and series and their applications in deep learning.

---

## Chapter Structure

To facilitate learning and in-depth understanding, this chapter is divided into four subsections:

### [7.1 Sequence Basics](07a-sequence-basics_EN.md)

**Content Overview**:
- Definition and representation of sequences
- Arithmetic sequences: general term formula, summation formula
- Geometric sequences: general term formula, summation formula
- Recursive sequences and recurrence relations
- Monotonicity and boundedness of sequences

**Core Concepts**:
| Concept | Formula | Applications in Deep Learning |
|---------|---------|------------------------------|
| Arithmetic sequence term | $a_n = a_1 + (n-1)d$ | Linear learning rate decay |
| Geometric sequence term | $a_n = a_1 \cdot r^{n-1}$ | Exponential learning rate decay, positional encoding |
| Geometric sequence sum | $S_n = a_1 \frac{1-r^n}{1-r}$ | RNN gradient propagation analysis |
| Recurrence relation | $a_n = f(a_{n-1}, a_{n-2}, \ldots)$ | RNN hidden state update |

**[Start Learning →](07a-sequence-basics_EN.md)**

---

### [7.2 Sequence Limits](07b-sequence-limits_EN.md)

**Content Overview**:
- $\epsilon-N$ definition of sequence limits
- Properties and operation rules of limits
- Convergence criteria for sequences
- Cauchy convergence criterion
- Important limits: definition of $e$, monotone bounded principle

**Core Concepts**:
| Concept | Definition/Formula | Importance |
|---------|-------------------|-------------|
| Limit definition | $\forall \epsilon > 0, \exists N, \forall n > N: \|a_n - L\| < \epsilon$ | Rigorous mathematical foundation |
| Uniqueness | Convergent sequences have unique limits | Theoretical guarantee |
| Boundedness | Convergent sequences are bounded | Decision tool |
| Cauchy criterion | $\forall \epsilon > 0, \exists N, \forall m,n > N: \|a_m - a_n\| < \epsilon$ | Foundation of completeness |

**[Start Learning →](07b-sequence-limits_EN.md)**

---

### [7.3 Series and Summation](07c-series-summation_EN.md)

**Content Overview**:
- Convergence and divergence of numerical series
- Convergence tests for positive series (comparison, ratio, root)
- Alternating series and Leibniz test
- Power series and radius of convergence
- Summation of common series

**Core Concepts**:
| Series Type | Convergence Condition | Application Scenarios |
|-------------|---------------------|---------------------|
| Geometric series $\sum r^n$ | $\|r\| < 1$ | RNN long-term dependency analysis |
| p-series $\sum \frac{1}{n^p}$ | $p > 1$ | Regularization term analysis |
| Harmonic series $\sum \frac{1}{n}$ | Divergent | Learning rate scheduling theory |
| Taylor series | Within radius of convergence | Function approximation, optimization theory |

**[Start Learning →](07c-series-summation_EN.md)**

---

### [7.4 Applications of Sequences in Deep Learning](07d-sequences-dl-applications_EN.md)

**Content Overview**:
- Learning rate decay strategies (exponential decay, cosine annealing)
- Sequence modeling and gradient propagation in RNNs
- Transformer positional encoding (sinusoidal positional encoding)
- Softmax sequences in self-attention
- Sequence generation and sampling strategies

**Core Concepts**:
| Application | Sequence/Series Concept | Specific Form |
|------------|------------------------|---------------|
| Exponential decay | Geometric sequence | $\eta_t = \eta_0 \cdot \gamma^t$ |
| Cosine annealing | Trigonometric sequence | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\frac{\pi t}{T}))$ |
| Positional encoding | Sine/cosine functions | $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$ |
| Gradient propagation | Geometric series of matrices | Eigenvalue analysis of $\prod_{t} W_h$ |

**[Start Learning →](07d-sequences-dl-applications_EN.md)**

---

## Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│               Chapter 7: Sequences and Series                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  7.1 Sequence Basics → 7.2 Sequence Limits → 7.3 Series → 7.4 DL Applications │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │Arithmetic│   │ ε-N Def  │   │Convergence│   │LR Decay  │ │
│  │Geometric │   │ Cauchy   │   │Power Series│   │Pos Encoding│ │
│  │Recurrence│   │Important │   │Taylor    │   │Gradient  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                             │
│  Applications: LR scheduling, RNN, Transformer pos encoding, sequence modeling   │
└─────────────────────────────────────────────────────────────┘
```

## Why are Sequences and Series Important for Deep Learning?

### 1. Sequences During Training

```
Training process = Parameter sequence {θ₁, θ₂, θ₃, ...}
                    ↓
         Learning rate sequence {η₁, η₂, η₃, ...}
                    ↓
          Loss sequence {L₁, L₂, L₃, ...} → Expected convergence to optimum
```

### 2. Core of Sequence Modeling

| Model | Sequence Processing Method | Sequence Concept |
|-------|--------------------------|-----------------|
| RNN | Recursive hidden state update | Recursive sequence |
| LSTM | Gated recurrence relation | Complex recursive system |
| Transformer | Positional encoding | Sine/cosine sequences |
| Attention | Softmax weights | Normalized sequence |

### 3. Theoretical Analysis Tools

- **Gradient vanishing/exploding**: Analyze limit behavior of $\prod_{t} W_h$
- **Convergence proofs**: Convergence conditions for loss sequences
- **Regularization**: Regularization terms in series summation form

---

## Core Formulas Quick Reference

### Arithmetic Sequence

$$
a_n = a_1 + (n-1)d
$$

$$
S_n = \frac{n(a_1 + a_n)}{2} = \frac{n[2a_1 + (n-1)d]}{2}
$$

### Geometric Sequence

$$
a_n = a_1 \cdot r^{n-1}
$$

$$
S_n = \begin{cases} na_1 & r = 1 \\ a_1 \frac{1-r^n}{1-r} & r \neq 1 \end{cases}
$$

### Infinite Geometric Series

When $|r| < 1$:

$$
\sum_{n=0}^{\infty} a_1 r^n = \frac{a_1}{1-r}
$$

### Important Limit

$$
\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e \approx 2.71828
$$

---

## Python Code Examples

### Sequence Generation

```python
import numpy as np

def arithmetic_sequence(a1, d, n):
    """Arithmetic sequence"""
    return np.array([a1 + i * d for i in range(n)])

def geometric_sequence(a1, r, n):
    """Geometric sequence"""
    return np.array([a1 * (r ** i) for i in range(n)])

# Examples
print("Arithmetic sequence:", arithmetic_sequence(1, 2, 10))
print("Geometric sequence:", geometric_sequence(1, 0.5, 10))
```

### Series Summation

```python
def geometric_sum(a1, r, n):
    """Sum of first n terms of geometric series"""
    if abs(r - 1) < 1e-10:
        return n * a1
    return a1 * (1 - r**n) / (1 - r)

def infinite_geometric_sum(a1, r):
    """Sum of infinite geometric series (converges when |r| < 1)"""
    if abs(r) >= 1:
        return float('inf')  # Divergent
    return a1 / (1 - r)

# Examples
print("Sum of first 10 terms:", geometric_sum(1, 0.5, 10))
print("Infinite series sum:", infinite_geometric_sum(1, 0.5))
```

### Learning Rate Decay

```python
class ExponentialDecay:
    """Exponential learning rate decay"""
    def __init__(self, initial_lr, decay_rate, decay_steps):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

    def get_lr(self, step):
        return self.initial_lr * (self.decay_rate ** (step / self.decay_steps))

class CosineAnnealing:
    """Cosine annealing"""
    def __init__(self, initial_lr, min_lr, total_steps):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps

    def get_lr(self, step):
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
               (1 + np.cos(np.pi * step / self.total_steps))

# Visualization comparison
import matplotlib.pyplot as plt

steps = np.arange(0, 1000)
exp_decay = ExponentialDecay(0.1, 0.96, 100)
cosine = CosineAnnealing(0.1, 0.001, 1000)

plt.figure(figsize=(10, 5))
plt.plot(steps, [exp_decay.get_lr(s) for s in steps], label='Exponential Decay')
plt.plot(steps, [cosine.get_lr(s) for s in steps], label='Cosine Annealing')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Decay Strategy Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Learning Recommendations

1. **Start from basics**: First master arithmetic and geometric sequences, then learn limits and series
2. **Understand convergence**: Convergence is the core goal of deep learning training
3. **Focus on applications**: Connect sequence concepts with learning rate scheduling and RNNs
4. **Hands-on practice**: Implement various sequences and learning rate strategies in Python
5. **Theory to practice**: Understand the sequential nature of gradient vanishing/exploding

---

## Further Reading

- [Chapter 2: Calculus](02-calculus.md) - Limits and derivatives
- [Chapter 5: Optimization Methods](05-optimization.md) - Learning rate scheduling
- [Chapter 6: Elementary Functions](06-elementary-functions.md) - Exponential and trigonometric functions

---

**Back to**: [Math Fundamentals Tutorial Directory](../math-fundamentals_EN.md)
