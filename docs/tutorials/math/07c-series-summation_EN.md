# Chapter 7(c): Series and Summation

Series are the natural extension of sequence summation and have widespread applications in deep learning. From the analysis of regularization terms to RNN gradient propagation, from the design of positional encoding to the theoretical foundations of attention mechanisms, the concept of series is ubiquitous. This section systematically introduces the basic concepts of series, convergence tests, and summation techniques.

---

## 🎯 Life Analogy: Adding Up Money

Imagine someone offers you two payment options:
- Option A: $1 million today
- Option B: 1 cent today, 2 cents tomorrow, 4 cents the next day, ... for 30 days

**Which is better?**

```
Day 1:   $0.01
Day 2:   $0.02
Day 3:   $0.04
...
Day 30:  $5,368,709.12

Total: Over $10 million!
```

This is a **geometric series**: $1 + 2 + 4 + 8 + ...$

### Convergence vs Divergence = "Bucket Filling"

```
Convergent series: Like filling a bucket that eventually stops
    ╭───╮
    │   │ ← Water level settles
    │   │
    └───┘
Water added: 1 + 1/2 + 1/4 + 1/8 + ... = 2 (finite!)

Divergent series: Like filling an infinite tank
    │
    │ ← Water keeps rising forever
    │
    │
Water added: 1 + 2 + 3 + 4 + ... = ∞
```

### 📖 Plain English Translation

| Term | Plain English |
|------|---------------|
| Series | Adding up the terms of a sequence |
| Convergent | The sum approaches a finite number |
| Divergent | The sum grows to infinity |
| Partial sum | Sum of the first n terms |

---

## Table of Contents

1. [Basic Concepts of Series](#basic-concepts-of-series)
2. [Convergence Tests for Positive Term Series](#convergence-tests-for-positive-term-series)
3. [Alternating Series](#alternating-series)
4. [Power Series](#power-series)
5. [Summation of Common Series](#summation-of-common-series)
6. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Basic Concepts of Series

### Definitions

**Series** is the sequential summation of terms in a sequence:

$$
\sum_{n=1}^{\infty} a_n = a_1 + a_2 + a_3 + \cdots
$$

**Partial sum**:
$$
S_n = \sum_{k=1}^{n} a_k = a_1 + a_2 + \cdots + a_n
$$

### Convergence and Divergence

- **Convergence**: If the sequence of partial sums $\{S_n\}$ converges to $S$, the series is said to converge, and $S$ is called the sum of the series.
  $$
  \sum_{n=1}^{\infty} a_n = S = \lim_{n \to \infty} S_n
  $$

- **Divergence**: If $\{S_n\}$ diverges, the series is said to diverge.

### Necessary Condition for Convergence

**Theorem**: If $\sum a_n$ converges, then $\lim_{n \to \infty} a_n = 0$.

**Note**: This is a necessary condition, not a sufficient condition!

Example: The harmonic series $\sum \frac{1}{n}$ diverges, even though $\frac{1}{n} \to 0$.

```python
import numpy as np
import matplotlib.pyplot as plt

def partial_sum(sequence, n_terms):
    """Calculate partial sum"""
    return np.sum(sequence[:n_terms])

def visualize_series(sequence_func, name, n_max=50):
    """Visualize series partial sums"""
    terms = np.array([sequence_func(n) for n in range(1, n_max + 1)])
    partial_sums = np.cumsum(terms)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Series terms
    axes[0].stem(range(1, n_max + 1), terms, basefmt=' ')
    axes[0].set_xlabel('n')
    axes[0].set_ylabel('$a_n$')
    axes[0].set_title(f'Series terms: $a_n = {name}$')
    axes[0].grid(True, alpha=0.3)

    # Partial sums
    axes[1].plot(range(1, n_max + 1), partial_sums, 'b-', linewidth=2)
    axes[1].set_xlabel('n')
    axes[1].set_ylabel('$S_n$')
    axes[1].set_title(f'Partial sum: $S_n = \\sum_{{k=1}}^n a_k$')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"First 10 partial sums: {partial_sums[:10]}")
    print(f"Last 5 partial sums: {partial_sums[-5:]}")

# Example: Geometric series (convergent)
visualize_series(lambda n: 0.5**n, "(0.5)^n")

# Example: Harmonic series (divergent)
visualize_series(lambda n: 1/n, "1/n")
```

---

## Convergence Tests for Positive Term Series

For positive term series $\sum a_n$ (where $a_n \geq 0$), there are specialized convergence tests.

### Comparison Test

**Theorem**: Let $0 \leq a_n \leq b_n$, then:
- If $\sum b_n$ converges, then $\sum a_n$ converges
- If $\sum a_n$ diverges, then $\sum b_n$ diverges

**Limit form**: If $\lim_{n \to \infty} \frac{a_n}{b_n} = c$ (where $0 < c < \infty$), then both series either converge or diverge together.

### Ratio Test (D'Alembert)

**Theorem**: Let $\lim_{n \to \infty} \frac{a_{n+1}}{a_n} = r$, then:
- $r < 1$: Series converges
- $r > 1$: Series diverges
- $r = 1$: Cannot determine

### Root Test (Cauchy)

**Theorem**: Let $\lim_{n \to \infty} \sqrt[n]{a_n} = r$, then:
- $r < 1$: Series converges
- $r > 1$: Series diverges
- $r = 1$: Cannot determine

### Integral Test

**Theorem**: Let $f(x)$ be non-negative, continuous, and monotonically decreasing on $[1, +\infty)$, and $a_n = f(n)$, then:
$$
\sum_{n=1}^{\infty} a_n \text{ converges} \Leftrightarrow \int_1^{\infty} f(x)\,dx \text{ converges}
$$

```python
def test_convergence_ratio(a_func, n_max=100):
    """Ratio test"""
    ratios = []
    for n in range(1, n_max):
        a_n = a_func(n)
        a_next = a_func(n+1)
        if a_n > 1e-10:
            ratios.append(a_next / a_n)

    if ratios:
        limit = np.mean(ratios[-10:])  # Estimate limit using average of last few terms
        print(f"Ratio test: lim(a_{{n+1}}/a_n) ≈ {limit:.4f}")
        if limit < 1:
            print(f"  → Convergent (because {limit:.4f} < 1)")
        elif limit > 1:
            print(f"  → Divergent (because {limit:.4f} > 1)")
        else:
            print(f"  → Cannot determine (because {limit:.4f} = 1)")

    return ratios

def test_convergence_root(a_func, n_max=100):
    """Root test"""
    roots = []
    for n in range(1, n_max):
        a_n = a_func(n)
        if a_n > 0:
            roots.append(a_n ** (1/n))

    if roots:
        limit = np.mean(roots[-10:])
        print(f"Root test: lim(a_n^(1/n)) ≈ {limit:.4f}")
        if limit < 1:
            print(f"  → Convergent (because {limit:.4f} < 1)")
        elif limit > 1:
            print(f"  → Divergent (because {limit:.4f} > 1)")
        else:
            print(f"  → Cannot determine (because {limit:.4f} = 1)")

    return roots

# Test geometric series
print("=== Geometric Series Σ(0.5)^n ===")
test_convergence_ratio(lambda n: 0.5**n)

# Test harmonic series
print("\n=== Harmonic Series Σ1/n ===")
test_convergence_ratio(lambda n: 1/n)

# Test p-series (p=2)
print("\n=== p-Series Σ1/n² (p=2) ===")
test_convergence_ratio(lambda n: 1/n**2)
```

### Common Positive Term Series

| Series | Convergence Condition | Sum (if convergent) |
|--------|---------------------|---------------------|
| Geometric series $\sum r^n$ | $\|r\| < 1$ | $\frac{1}{1-r}$ |
| p-series $\sum \frac{1}{n^p}$ | $p > 1$ | No closed form |
| Harmonic series $\sum \frac{1}{n}$ | Divergent | - |

---

## Alternating Series

### Definition

**Alternating series** is a series where positive and negative terms alternate:

$$
\sum_{n=1}^{\infty} (-1)^{n-1} a_n = a_1 - a_2 + a_3 - a_4 + \cdots
$$

### Leibniz Test

**Theorem**: If $\{a_n\}$ is monotonically decreasing and $\lim_{n \to \infty} a_n = 0$, then the alternating series $\sum (-1)^{n-1} a_n$ converges.

**Error estimate**: If $S_n$ is used to approximate the series sum $S$, the error is:
$$
|S - S_n| \leq a_{n+1}
$$

```python
def alternating_series_sum(a_func, n_terms, true_sum=None):
    """Alternating series summation"""
    terms = [((-1)**(n-1)) * a_func(n) for n in range(1, n_terms + 1)]
    partial_sums = np.cumsum(terms)

    print(f"First {n_terms} terms partial sum of alternating series: {partial_sums[-1]:.6f}")

    if true_sum is not None:
        error = abs(partial_sums[-1] - true_sum)
        next_term = a_func(n_terms + 1)
        print(f"Actual error: {error:.6f}")
        print(f"Leibniz error upper bound: {next_term:.6f}")
        print(f"Error <= next term: {error <= next_term}")

    return partial_sums

# Alternating harmonic series: 1 - 1/2 + 1/3 - 1/4 + ... = ln(2)
print("=== Alternating Harmonic Series ===")
print("Theoretical value: ln(2) ≈", np.log(2))
alternating_series_sum(lambda n: 1/n, 100, np.log(2))

# Visualization
n_max = 50
terms = [((-1)**(n-1)) / n for n in range(1, n_max + 1)]
partial_sums = np.cumsum(terms)

plt.figure(figsize=(10, 5))
plt.plot(range(1, n_max + 1), partial_sums, 'b-', label='Partial sum $S_n$')
plt.axhline(y=np.log(2), color='r', linestyle='--', label=f'$\\ln(2) = {np.log(2):.4f}$')
plt.xlabel('n')
plt.ylabel('$S_n$')
plt.title('Alternating Harmonic Series: $\\sum (-1)^{{n-1}}/n = \\ln(2)$')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### Absolute Convergence and Conditional Convergence

- **Absolute convergence**: $\sum |a_n|$ converges $\Rightarrow$ $\sum a_n$ converges
- **Conditional convergence**: $\sum a_n$ converges but $\sum |a_n|$ diverges

**Properties**:
- Absolutely convergent series can be rearranged arbitrarily without changing the sum
- Conditionally convergent series may change their sum after rearrangement (Riemann rearrangement theorem)

---

## Power Series

### Definition

**Power series** is a series of the form:
$$
\sum_{n=0}^{\infty} c_n (x-a)^n = c_0 + c_1(x-a) + c_2(x-a)^2 + \cdots
$$

When $a=0$:
$$
\sum_{n=0}^{\infty} c_n x^n = c_0 + c_1x + c_2x^2 + \cdots
$$

### Radius of Convergence

**Theorem**: The convergence domain of a power series is an interval $(-R, R)$, where $R$ is called the radius of convergence:

$$
R = \frac{1}{\limsup_{n \to \infty} \sqrt[n]{|c_n|}}
$$

Or using the ratio method:
$$
R = \lim_{n \to \infty} \left| \frac{c_n}{c_{n+1}} \right|
$$

- When $|x| < R$: Series converges absolutely
- When $|x| > R$: Series diverges
- When $|x| = R$: Must be determined separately

### Common Power Series Expansions

| Function | Power Series Expansion | Convergence Domain |
|----------|---------------------|-------------------|
| $e^x$ | $\sum_{n=0}^{\infty} \frac{x^n}{n!}$ | $(-\infty, +\infty)$ |
| $\sin x$ | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!}$ | $(-\infty, +\infty)$ |
| $\cos x$ | $\sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!}$ | $(-\infty, +\infty)$ |
| $\frac{1}{1-x}$ | $\sum_{n=0}^{\infty} x^n$ | $(-1, 1)$ |
| $\ln(1+x)$ | $\sum_{n=1}^{\infty} \frac{(-1)^{n-1} x^n}{n}$ | $(-1, 1]$ |
| $(1+x)^\alpha$ | $\sum_{n=0}^{\infty} \binom{\alpha}{n} x^n$ | $(-1, 1)$ |

```python
def power_series_approximation(x, series_func, name, true_func, max_terms=20):
    """Power series approximation visualization"""
    terms = [series_func(n) * (x**n) for n in range(max_terms)]
    partial_sums = np.cumsum(terms)

    true_value = true_func(x)

    plt.figure(figsize=(10, 5))
    plt.plot(range(max_terms), partial_sums, 'b-o', label='Partial sum')
    plt.axhline(y=true_value, color='r', linestyle='--',
               label=f'True value = {true_value:.6f}')
    plt.xlabel('Number of terms')
    plt.ylabel('Approximation')
    plt.title(f'Power Series Approximation: {name} at x={x}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"True value: {true_value:.10f}")
    print(f"20-term approximation: {partial_sums[-1]:.10f}")
    print(f"Error: {abs(partial_sums[-1] - true_value):.2e}")

# Power series for e^x
power_series_approximation(
    x=1,
    series_func=lambda n: 1/np.math.factorial(n),
    name='$e^x$',
    true_func=np.exp
)

# Power series for sin(x)
power_series_approximation(
    x=np.pi/4,
    series_func=lambda n: ((-1)**n) / np.math.factorial(2*n + 1) if n >= 0 else 0,
    name='$\\sin(x)$',
    true_func=np.sin
)
```

---

## Summation of Common Series

### 1. Geometric Series

$$
\sum_{n=0}^{\infty} r^n = \frac{1}{1-r}, \quad |r| < 1
$$

Finite sum:
$$
\sum_{n=0}^{N-1} r^n = \frac{1-r^N}{1-r}
$$

### 2. Arithmetic Series

$$
\sum_{n=1}^{N} n = \frac{N(N+1)}{2}
$$

$$
\sum_{n=1}^{N} n^2 = \frac{N(N+1)(2N+1)}{6}
$$

$$
\sum_{n=1}^{N} n^3 = \left[\frac{N(N+1)}{2}\right]^2
$$

### 3. Approximation of Harmonic Series

Partial sum of harmonic series:
$$
H_N = \sum_{n=1}^{N} \frac{1}{n} \approx \ln N + \gamma + \frac{1}{2N}
$$

where $\gamma \approx 0.5772$ is the Euler-Mascheroni constant.

### 4. Common Infinite Series

| Series | Sum |
|--------|-----|
| $\sum_{n=1}^{\infty} \frac{1}{n^2}$ | $\frac{\pi^2}{6}$ |
| $\sum_{n=1}^{\infty} \frac{1}{n^4}$ | $\frac{\pi^4}{90}$ |
| $\sum_{n=0}^{\infty} \frac{1}{n!}$ | $e$ |
| $\sum_{n=0}^{\infty} \frac{(-1)^n}{2n+1}$ | $\frac{\pi}{4}$ |

```python
def verify_famous_sums():
    """Verify famous series sums"""
    n_terms = 10000

    # Σ1/n² = π²/6
    sum_1_n2 = np.sum(1 / np.arange(1, n_terms+1)**2)
    true_val = np.pi**2 / 6
    print(f"Σ1/n² ≈ {sum_1_n2:.10f}")
    print(f"π²/6 = {true_val:.10f}")
    print(f"Error: {abs(sum_1_n2 - true_val):.2e}\n")

    # Σ1/n! = e
    sum_1_fact = np.sum(1 / np.array([np.math.factorial(n) for n in range(20)]))
    true_val = np.e
    print(f"Σ1/n! ≈ {sum_1_fact:.10f}")
    print(f"e = {true_val:.10f}")
    print(f"Error: {abs(sum_1_fact - true_val):.2e}\n")

    # Σ(-1)^n/(2n+1) = π/4 (Leibniz formula)
    n_leibniz = 100000
    terms = ((-1)**np.arange(n_leibniz)) / (2*np.arange(n_leibniz) + 1)
    sum_leibniz = np.sum(terms)
    true_val = np.pi / 4
    print(f"Σ(-1)^n/(2n+1) ≈ {sum_leibniz:.10f} (n={n_leibniz})")
    print(f"π/4 = {true_val:.10f}")
    print(f"Error: {abs(sum_leibniz - true_val):.2e}")

verify_famous_sums()
```

---

## Applications in Deep Learning

### 1. Series Analysis of RNN Gradient Propagation

RNN gradients involve infinite series of matrix products:

$$
\frac{\partial h_T}{\partial h_0} = \sum_{k=0}^{T-1} \prod_{t=T-k}^{T-1} W_h
$$

When the spectral radius $\rho(W_h) < 1$, the gradient converges (will not explode).

```python
def rnn_gradient_analysis():
    """Series analysis of RNN gradients"""
    eigenvalues = [0.5, 0.9, 1.0, 1.1]
    T_max = 100

    plt.figure(figsize=(12, 5))

    for i, eig in enumerate(eigenvalues):
        gradients = [eig ** t for t in range(T_max)]
        plt.subplot(1, 2, 1)
        plt.semilogy(gradients, label=f'$\\lambda = {eig}$')

        plt.subplot(1, 2, 2)
        cumsum = np.cumsum(gradients)
        plt.plot(cumsum, label=f'$\\lambda = {eig}$')

    plt.subplot(1, 2, 1)
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Gradient (log scale)')
    plt.title('RNN Gradient Propagation: $\\lambda^t$')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.xlabel('Time Step')
    plt.ylabel('Cumulative Gradient')
    plt.title('Gradient Accumulation: $\\sum \\lambda^t$')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

rnn_gradient_analysis()
```

### 2. Series Form of Regularization

L2 regularization (weight decay) gradients involve series:

$$
W_t = W_0(1-\lambda)^t + \text{(gradient contributions)}
$$

Without gradients, weights decay according to a geometric series:
$$
W_t = W_0(1-\lambda)^t \to 0
$$

### 3. Softmax in Attention Mechanisms

Softmax weights can be viewed as a probability distribution, satisfying:
$$
\sum_{i=1}^{n} \text{softmax}(z)_i = 1
$$

This is a finite series sum form.

### 4. Taylor Series in Optimization

Newton's method uses the second-order approximation of Taylor series:

$$
f(x + \Delta x) \approx f(x) + \nabla f(x)^\top \Delta x + \frac{1}{2} \Delta x^\top H \Delta x
$$

```python
def taylor_optimization_demo():
    """Taylor series optimization demonstration"""
    # Objective function: f(x) = x^4 - 2x^2 + 1
    f = lambda x: x**4 - 2*x**2 + 1
    df = lambda x: 4*x**3 - 4*x
    d2f = lambda x: 12*x**2 - 4

    x = np.linspace(-2, 2, 200)

    # Taylor expansion at x = 1.5
    x0 = 1.5
    taylor_order1 = f(x0) + df(x0) * (x - x0)
    taylor_order2 = f(x0) + df(x0) * (x - x0) + 0.5 * d2f(x0) * (x - x0)**2

    plt.figure(figsize=(10, 6))
    plt.plot(x, f(x), 'b-', linewidth=2, label='Original function $f(x)$')
    plt.plot(x, taylor_order1, 'g--', linewidth=2, label='First-order Taylor')
    plt.plot(x, taylor_order2, 'r--', linewidth=2, label='Second-order Taylor')
    plt.axvline(x=x0, color='k', linestyle=':', alpha=0.5)
    plt.scatter([x0], [f(x0)], color='k', s=100, zorder=5)
    plt.xlim(-2, 2)
    plt.ylim(-2, 6)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(f'Taylor Series Approximation (at $x_0 = {x0}$)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

taylor_optimization_demo()
```

---

## Summary

This section introduced the core concepts of series:

| Concept | Definition/Method | Application |
|---------|------------------|-------------|
| Series convergence | $\lim S_n$ exists | Determine infinite summation |
| Ratio test | $\lim a_{n+1}/a_n < 1$ | Determine convergence |
| Root test | $\lim \sqrt[n]{a_n} < 1$ | Determine convergence |
| Alternating series | Leibniz test | Conditional convergence |
| Power series | Convergence radius $R$ | Taylor expansion |
| Geometric series | $\sum r^n = \frac{1}{1-r}$ | Learning rate analysis |

**Key Points**:
- The necessary condition for convergence is that the general term tends to zero (but not sufficient)
- Positive term series have multiple convergence tests
- Power series converge absolutely within the radius of convergence
- Taylor series are important tools for function approximation

---

**Next**: [Applications of Sequences in Deep Learning](07d-sequences-dl-applications_EN.md) - Learn about specific applications of sequences and series in deep learning.

**Back**: [Chapter 7: Sequences and Series](07-sequences-series_EN.md) | [Math Fundamentals Tutorial Contents](../math-fundamentals_EN.md)
