# Chapter 7(a): Sequence Basics

A sequence is a list of numbers arranged in a certain order. From simple arithmetic sequences to complex recurrence relations, sequences are the foundation for understanding core deep learning concepts such as sequence modeling and learning rate scheduling. This section will systematically introduce the basic concepts and common types of sequences.

---

## 🎯 Life Analogy: A Piggy Bank

Imagine a **piggy bank where you save money regularly**:

| Day | Amount Added | Total |
|-----|-------------|-------|
| 1 | $1 | $1 |
| 2 | $2 | $3 |
| 3 | $3 | $6 |
| 4 | $4 | $10 |
| 5 | $5 | $15 |

This is an **arithmetic sequence**: $1, 2, 3, 4, 5, ...$

### Arithmetic Sequence = "Climbing Stairs"

```
Each step is the SAME height:
Step 1: ___
Step 2: ___
Step 3: ___
Step 4: ___

Height: 1, 2, 3, 4, 5, ...

Formula: a_n = a_1 + (n-1)d
Where d = step height (common difference)
```

### Geometric Sequence = "Doubling"

```
Each step DOUBLES:
Day 1: ●                     (1 cent)
Day 2: ●●                    (2 cents)
Day 3: ●●●●                  (4 cents)
Day 4: ●●●●●●●●              (8 cents)
Day 5: ●●●●●●●●●●●●●●●●      (16 cents)

Formula: a_n = a_1 × r^(n-1)
Where r = multiplier (common ratio)
```

### 📖 Plain English Translation

| Term | Plain English |
|------|---------------|
| Sequence | A list of numbers in order |
| Arithmetic sequence | Add the same amount each time |
| Geometric sequence | Multiply by the same amount each time |
| General term $a_n$ | The formula for the n-th number |
| Common difference $d$ | What you add each step |
| Common ratio $r$ | What you multiply each step |

---

## Table of Contents

1. [Definition and Representation of Sequences](#definition-and-representation-of-sequences)
2. [Arithmetic Sequences](#arithmetic-sequences)
3. [Geometric Sequences](#geometric-sequences)
4. [Recursive Sequences](#recursive-sequences)
5. [Properties of Sequences](#properties-of-sequences)
6. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Definition and Representation of Sequences

### Basic Definition

A **sequence** is a list of numbers arranged in a certain order:

$$
\{a_n\} = a_1, a_2, a_3, \ldots, a_n, \ldots
$$

where $a_n$ is called the **general term** (the $n$-th term) of the sequence.

### Representation Methods

| Representation Method | Example | Characteristics |
|---------------------|---------|------------------|
| Listing method | $1, 3, 5, 7, 9, \ldots$ | Intuitive but verbose |
| General term formula | $a_n = 2n - 1$ | Concise and clear |
| Recurrence formula | $a_1 = 1, a_{n+1} = a_n + 2$ | Suitable for programming implementation |
| Graphical method | Sequence of points on number line | Visualizable |

### Classification of Sequences

- **Finite sequence**: Has a finite number of terms, e.g., $1$, $2$, $3$, $4$, $5$
- **Infinite sequence**: Has an infinite number of terms, e.g., $1$, $\frac{1}{2}$, $\frac{1}{3}$, $\ldots$
- **Bounded sequence**: There exists $M > 0$ such that $|a_n| \leq M, \forall n$
- **Unbounded sequence**: No such $M$ exists

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_sequence(sequence, title="Sequence Visualization"):
    """Visualize a sequence"""
    n = len(sequence)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.stem(range(1, n+1), sequence, basefmt=' ')
    plt.xlabel('n')
    plt.ylabel('$a_n$')
    plt.title(f'{title} - Scatter Plot')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, n+1), sequence, 'o-', markersize=4)
    plt.xlabel('n')
    plt.ylabel('$a_n$')
    plt.title(f'{title} - Line Plot')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Example: decreasing sequence
seq = [1/n for n in range(1, 21)]
visualize_sequence(seq, "Decreasing Sequence $a_n = 1/n$")
```

---

## Arithmetic Sequences

### Definition

An **arithmetic sequence** is a sequence where the difference between consecutive terms is constant:

$$
a_{n+1} - a_n = d \quad (\text{constant})
$$

where $d$ is called the **common difference**.

### General Term Formula

$$
a_n = a_1 + (n-1)d
$$

**Derivation**:
$$
\begin{align}
a_2 &= a_1 + d \\
a_3 &= a_2 + d = a_1 + 2d \\
&\vdots \\
a_n &= a_1 + (n-1)d
\end{align}
$$

### Sum of First $n$ Terms Formula

$$
S_n = \frac{n(a_1 + a_n)}{2} = \frac{n[2a_1 + (n-1)d]}{2}
$$

**Derivation** (Gauss summation method):

Let $S_n = a_1 + a_2 + \cdots + a_n$

Then $S_n = a_n + a_{n-1} + \cdots + a_1$ (reverse order)

Adding the two equations: $2S_n = n(a_1 + a_n)$, thus $S_n = \frac{n(a_1 + a_n)}{2}$

### Properties

1. **Equal spacing property**: If $m + n = p + q$, then $a_m + a_n = a_p + a_q$
2. **Middle term property**: $2a_n = a_{n-1} + a_{n+1}$
3. **Segmented sums**: $S_n, S_{2n} - S_n, S_{3n} - S_{2n}$ still form an arithmetic sequence

```python
class ArithmeticSequence:
    """Arithmetic sequence class"""
    def __init__(self, a1, d):
        self.a1 = a1  # First term
        self.d = d    # Common difference

    def term(self, n):
        """nth term"""
        return self.a1 + (n - 1) * self.d

    def sum(self, n):
        """Sum of first n terms"""
        return n * (self.a1 + self.term(n)) / 2

    def generate(self, n):
        """Generate first n terms"""
        return [self.term(i) for i in range(1, n+1)]

# Example
seq = ArithmeticSequence(a1=1, d=2)
print(f"First 10 terms: {seq.generate(10)}")
print(f"10th term: {seq.term(10)}")
print(f"Sum of first 10 terms: {seq.sum(10)}")

# Verify equal spacing property
print(f"a_2 + a_8 = {seq.term(2) + seq.term(8)}")
print(f"a_4 + a_6 = {seq.term(4) + seq.term(6)}")
print(f"a_3 + a_7 = {seq.term(3) + seq.term(7)}")  # All equal to 20
```

### Application in Deep Learning: Linear Learning Rate Decay

```python
def linear_lr_decay(initial_lr, final_lr, total_steps, current_step):
    """
    Linear learning rate decay (arithmetic sequence form)

    Args:
        initial_lr: Initial learning rate
        final_lr: Final learning rate
        total_steps: Total number of steps
        current_step: Current step number
    """
    d = (final_lr - initial_lr) / total_steps
    return initial_lr + current_step * d

# Visualization
import matplotlib.pyplot as plt

steps = 1000
lrs = [linear_lr_decay(0.1, 0.001, steps, s) for s in range(steps)]

plt.figure(figsize=(8, 4))
plt.plot(lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Linear Learning Rate Decay (Arithmetic Sequence)')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Geometric Sequences

### Definition

A **geometric sequence** is a sequence where the ratio of consecutive terms is constant:

$$
\frac{a_{n+1}}{a_n} = r \quad (\text{constant})
$$

where $r$ is called the **common ratio**.

### General Term Formula

$$
a_n = a_1 \cdot r^{n-1}
$$

### Sum of First $n$ Terms Formula

$$
S_n = \begin{cases}
na_1 & r = 1 \\
a_1 \frac{1 - r^n}{1 - r} & r \neq 1
\end{cases}
$$

**Derivation** (displaced subtraction method):

Let $S_n = a_1 + a_1 r + a_1 r^2 + \cdots + a_1 r^{n-1}$

Then $rS_n = a_1 r + a_1 r^2 + \cdots + a_1 r^{n-1} + a_1 r^n$

Subtracting: $(1-r)S_n = a_1 - a_1 r^n = a_1(1 - r^n)$

Thus $S_n = a_1 \frac{1 - r^n}{1 - r}$

### Infinite Geometric Series

When $|r| < 1$, the infinite geometric series converges:

$$
\sum_{n=0}^{\infty} a_1 r^n = \frac{a_1}{1 - r}
$$

### Properties

1. **Equal spacing property**: If $m + n = p + q$, then $a_m \cdot a_n = a_p \cdot a_q$
2. **Middle term property**: $a_n^2 = a_{n-1} \cdot a_{n+1}$
3. **Segmented product**: The product of consecutive $k$ terms forms a new geometric sequence

```python
class GeometricSequence:
    """Geometric sequence class"""
    def __init__(self, a1, r):
        self.a1 = a1  # First term
        self.r = r    # Common ratio

    def term(self, n):
        """nth term"""
        return self.a1 * (self.r ** (n - 1))

    def sum(self, n):
        """Sum of first n terms"""
        if abs(self.r - 1) < 1e-10:
            return n * self.a1
        return self.a1 * (1 - self.r**n) / (1 - self.r)

    def infinite_sum(self):
        """Sum of infinite series (converges only when |r| < 1)"""
        if abs(self.r) >= 1:
            return float('inf')  # Divergent
        return self.a1 / (1 - self.r)

    def generate(self, n):
        """Generate first n terms"""
        return [self.term(i) for i in range(1, n+1)]

# Example
seq = GeometricSequence(a1=1, r=0.5)
print(f"First 10 terms: {seq.generate(10)}")
print(f"10th term: {seq.term(10)}")
print(f"Sum of first 10 terms: {seq.sum(10):.4f}")
print(f"Infinite series sum: {seq.infinite_sum():.4f}")  # Theoretical value = 2

# Verify middle term property
print(f"a_5² = {seq.term(5)**2:.6f}")
print(f"a_4 × a_6 = {seq.term(4) * seq.term(6):.6f}")
```

### Application in Deep Learning: Exponential Learning Rate Decay

```python
def exponential_lr_decay(initial_lr, decay_rate, decay_steps, current_step):
    """
    Exponential learning rate decay (geometric sequence form)

    Args:
        initial_lr: Initial learning rate
        decay_rate: Decay rate (common ratio)
        decay_steps: Decay period
        current_step: Current step number
    """
    return initial_lr * (decay_rate ** (current_step / decay_steps))

# Comparison of different decay rates
import matplotlib.pyplot as plt

steps = 1000
decay_rates = [0.9, 0.95, 0.99]

plt.figure(figsize=(10, 5))
for rate in decay_rates:
    lrs = [exponential_lr_decay(0.1, rate, 100, s) for s in range(steps)]
    plt.plot(lrs, label=f'decay_rate={rate}')

plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('Exponential Learning Rate Decay (Geometric Sequence)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## Recursive Sequences

### Definition

A **recursive sequence** is a sequence defined by a recurrence relation, where each term is obtained from previous terms through some rule.

### First-Order Recurrence

$$
a_{n+1} = f(a_n)
$$

**Example**: $a_{n+1} = 2a_n + 1$, $a_1 = 1$

### Second-Order Recurrence

$$
a_{n+2} = f(a_{n+1}, a_n)
$$

**Fibonacci sequence**: $F_{n+2} = F_{n+1} + F_n$, $F_1 = F_2 = 1$

### General Solution for Linear Recurrence Relations

For $a_{n+2} = pa_{n+1} + qa_n$:

1. Write the **characteristic equation**: $x^2 - px - q = 0$
2. Find the characteristic roots $\alpha, \beta$
3. Write the general solution based on the roots:
   - If $\alpha \neq \beta$: $a_n = A \cdot \alpha^n + B \cdot \beta^n$
   - If $\alpha = \beta$: $a_n = (A + Bn) \cdot \alpha^n$
4. Determine $A, B$ using initial conditions

```python
def fibonacci(n):
    """Fibonacci sequence"""
    if n <= 2:
        return 1
    return fibonacci(n-1) + fibonacci(n-2)

def fibonacci_iterative(n):
    """Fibonacci sequence (iterative version, more efficient)"""
    if n <= 2:
        return 1
    a, b = 1, 1
    for _ in range(n - 2):
        a, b = b, a + b
    return b

# Using closed-form formula
def fibonacci_closed_form(n):
    """Fibonacci sequence closed-form formula"""
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    psi = (1 - np.sqrt(5)) / 2
    return int((phi**n - psi**n) / np.sqrt(5))

# Verification
print("First 20 Fibonacci numbers:")
fib_seq = [fibonacci_iterative(i) for i in range(1, 21)]
print(fib_seq)

# Verify closed-form formula
print("\nVerify closed-form formula:")
for i in range(1, 11):
    print(f"F_{i}: iterative={fibonacci_iterative(i)}, formula={fibonacci_closed_form(i)}")
```

### Application in Deep Learning: RNN Hidden States

The hidden state update in an RNN is a recurrence relation:

$$
h_t = \tanh(W_h h_{t-1} + W_x x_t + b)
$$

```python
class SimpleRNN:
    """Simple RNN demonstration (recursive sequence form)"""
    def __init__(self, hidden_size, input_size):
        self.W_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_x = np.random.randn(hidden_size, input_size) * 0.01
        self.b = np.zeros(hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x_sequence):
        """
        Forward propagation - recursively compute hidden state sequence

        Args:
            x_sequence: Input sequence (seq_len, input_size)
        """
        seq_len = x_sequence.shape[0]
        h_sequence = np.zeros((seq_len + 1, self.hidden_size))

        for t in range(seq_len):
            # Recurrence relation: h_t = f(h_{t-1}, x_t)
            h_sequence[t+1] = np.tanh(
                self.W_h @ h_sequence[t] +
                self.W_x @ x_sequence[t] +
                self.b
            )

        return h_sequence[1:]  # Return hidden states at all time steps

# Example
rnn = SimpleRNN(hidden_size=4, input_size=2)
x = np.random.randn(5, 2)  # Sequence length of 5
h = rnn.forward(x)
print("RNN hidden state sequence (recursive sequence):")
print(h)
```

---

## Properties of Sequences

### Monotonicity

- **Monotonically increasing**: $a_{n+1} > a_n$, $\forall n$
- **Monotonically decreasing**: $a_{n+1} < a_n$, $\forall n$
- **Monotonically non-decreasing**: $a_{n+1} \geq a_n$, $\forall n$
- **Monotonically non-increasing**: $a_{n+1} \leq a_n$, $\forall n$

### Boundedness

- **Upper bound**: $\exists M$, $a_n \leq M$, $\forall n$
- **Lower bound**: $\exists m$, $a_n \geq m$, $\forall n$
- **Bounded**: Has both upper and lower bounds

### Monotone Bounded Theorem

**A monotone bounded sequence must converge.**

This is an important tool for determining the existence of sequence limits.

```python
def analyze_sequence(sequence, name):
    """Analyze properties of a sequence"""
    arr = np.array(sequence)

    # Monotonicity
    diffs = np.diff(arr)
    if np.all(diffs > 0):
        monotonicity = "Monotonically increasing"
    elif np.all(diffs < 0):
        monotonicity = "Monotonically decreasing"
    elif np.all(diffs >= 0):
        monotonicity = "Monotonically non-decreasing"
    elif np.all(diffs <= 0):
        monotonicity = "Monotonically non-increasing"
    else:
        monotonicity = "Non-monotonic"

    # Boundedness
    if np.isfinite(arr).all():
        bounded = f"Bounded [{arr.min():.4f}, {arr.max():.4f}]"
    else:
        bounded = "Unbounded"

    print(f"{name}:")
    print(f"  Monotonicity: {monotonicity}")
    print(f"  Boundedness: {bounded}")
    print(f"  First 5 terms: {arr[:5]}")
    print(f"  Last 5 terms: {arr[-5:]}")
    print()

# Analyze various sequences
analyze_sequence([1/n for n in range(1, 100)], "$a_n = 1/n$")
analyze_sequence([np.log(n) for n in range(1, 100)], "$a_n = \\ln n$")
analyze_sequence([1 - 1/n for n in range(1, 100)], "$a_n = 1 - 1/n$")
analyze_sequence([(-1)**n / n for n in range(1, 100)], "$a_n = (-1)^n / n$")
```

---

## Applications in Deep Learning

### 1. Learning Rate Scheduling

| Strategy | Sequence Type | Formula |
|----------|--------------|---------|
| Linear decay | Arithmetic sequence | $\eta_t = \eta_0 - kt$ |
| Exponential decay | Geometric sequence | $\eta_t = \eta_0 \cdot \gamma^t$ |
| Step decay | Piecewise constant | $\eta_t = \eta_0 \cdot \gamma^{\lfloor t/T \rfloor}$ |

### 2. RNN Gradient Analysis

The core of RNN gradients is the geometric series of matrix products:

$$
\frac{\partial h_T}{\partial h_1} = \prod_{t=1}^{T-1} W_h
$$

When the maximum eigenvalue of $W_h$ satisfies $|\lambda_{\max}| < 1$, the gradient decays exponentially (gradient vanishing); when $|\lambda_{\max}| > 1$, the gradient grows exponentially (gradient explosion).

### 3. Positional Encoding

Transformer's sinusoidal positional encoding uses geometric sequences of different frequencies:

$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
$$

where $10000^{2i/d}$ forms a geometric sequence.

```python
def positional_encoding(max_len, d_model):
    """Transformer positional encoding"""
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))

    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)  # Even dimensions
    pe[:, 1::2] = np.cos(position * div_term)  # Odd dimensions

    return pe

# Visualize positional encoding
pe = positional_encoding(100, 64)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.imshow(pe, aspect='auto', cmap='coolwarm')
plt.colorbar(label='Encoding Value')
plt.xlabel('Dimension')
plt.ylabel('Position')
plt.title('Positional Encoding Matrix')

plt.subplot(1, 2, 2)
# Show div_term (geometric sequence)
div_term = np.exp(np.arange(0, 64, 2) * (-np.log(10000.0) / 64))
plt.semilogy(div_term, 'o-', markersize=4)
plt.xlabel('Dimension Index')
plt.ylabel('Frequency Divisor (log scale)')
plt.title('Frequency Factor (Geometric Sequence)')
plt.grid(True)

plt.tight_layout()
plt.show()
```

---

## Summary

This section introduced the basic concepts of sequences:

| Concept | Definition/Formula | Characteristics |
|---------|-------------------|----------------|
| Arithmetic sequence | $a_n = a_1 + (n-1)d$ | Difference of consecutive terms is constant |
| Geometric sequence | $a_n = a_1 \cdot r^{n-1}$ | Ratio of consecutive terms is constant |
| Recursive sequence | $a_n = f(a_{n-1}, \ldots)$ | Derived from previous terms |
| Monotonicity | $a_{n+1} \geq a_n$ or $a_{n+1} \leq a_n$ | Trend of change |
| Boundedness | $\|a_n\| \leq M$ | Range of values |

**Key Takeaways**:
- Arithmetic sequences are used for linear learning rate decay
- Geometric sequences are used for exponential decay and RNN gradient analysis
- Recursive sequences are the mathematical essence of RNNs
- Monotone bounded sequences must converge

---

**Next Section**: [Sequence Limits](07b-sequence-limits_EN.md) - Learn the rigorous definition of sequence limits and convergence criteria.

**Back to**: [Chapter 7: Sequences and Series](07-sequences-series_EN.md) | [Math Fundamentals Tutorial Directory](../math-fundamentals_EN.md)
