# Chapter 7(b): Sequence Limits

Sequence limits are the foundation of calculus and key to understanding the convergence of deep learning training. This section introduces the rigorous definition, properties, and criteria for determining sequence limits.

---

## 🎯 Life Analogy: Approaching a Door But Never Arriving

Imagine you're standing in a room, with a door 1 meter away. You follow this rule:
- Step 1: Walk 1/2 meter (1/2 meter remaining)
- Step 2: Walk half the remaining distance (1/4 meter)
- Step 3: Walk half the remaining distance (1/8 meter)
- ...

Total distance walked: $\frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \frac{1}{16} + ...$

**Something magical happens**: No matter how many steps you take, you **never reach the door**! But you get **infinitely close** to it.

This is the essence of a **limit**: A sequence gets infinitely close to some value, but may never reach it.

```
Your position:
Start ●────────────────────────────────────────→ Door
     0        1/2      3/4      7/8      15/16...  1m
     │─────────│────────│────────│────────│─────→
        Step 1   Step 2   Step 3   Step 4   Limit=1

Sequence: 1/2, 3/4, 7/8, 15/16, ... → Limit = 1
```

### 📖 Plain English Translation

| Math Language | Plain English |
|---------------|---------------|
| $\lim_{n \to \infty} a_n = L$ | As n gets huge, $a_n$ eventually "sticks" to L |
| $\forall \epsilon > 0$ | No matter how precise you want to be (how small the error) |
| $\exists N$ | I can always find a step number |
| $n > N \Rightarrow \|a_n - L\| < \epsilon$ | After that step, distance is less than your requirement |

---

## Table of Contents

1. [Intuitive Understanding of Limits](#intuitive-understanding-of-limits)
2. [Rigorous Definition of Limits](#rigorous-definition-of-limits)
3. [Properties of Limits](#properties-of-limits)
4. [Criteria for Convergent Sequences](#criteria-for-convergent-sequences)
5. [Important Limits](#important-limits)
6. [Applications in Deep Learning](#applications-in-deep-learning)

---

## Intuitive Understanding of Limits

### Basic Concepts

**Intuitive Definition**: When $n$ increases without bound, if $a_n$ approaches some constant $L$ infinitely closely, then $L$ is called the limit of the sequence $\{a_n\}$, denoted as:

$$
\lim_{n \to \infty} a_n = L \quad \text{or} \quad a_n \to L \ (n \to \infty)
$$

### Examples

| Sequence | Limit | Reason |
|----------|-------|--------|
| $a_n = \frac{1}{n}$ | $0$ | As $n$ increases, $\frac{1}{n}$ approaches $0$ |
| $a_n = \frac{n+1}{n}$ | $1$ | $\frac{n+1}{n} = 1 + \frac{1}{n} \to 1 + 0 = 1$ |
| $a_n = 0.99^n$ | $0$ | Exponential decay approaches $0$ |
| $a_n = (-1)^n$ | None | Oscillates between $-1$ and $1$ |
| $a_n = n$ | $\infty$ | Grows without bound, diverges to infinity |

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_limit(sequence_func, name, true_limit=None, n_max=100):
    """Visualize sequence limit"""
    n = np.arange(1, n_max + 1)
    a_n = np.array([sequence_func(i) for i in n])

    plt.figure(figsize=(10, 5))
    plt.plot(n, a_n, 'o-', markersize=3, label=f'$a_n = {name}$')

    if true_limit is not None:
        plt.axhline(y=true_limit, color='r', linestyle='--',
                   label=f'Limit L = {true_limit}', linewidth=2)

    plt.xlabel('n', fontsize=12)
    plt.ylabel('$a_n$', fontsize=12)
    plt.title(f'Sequence Limit Visualization: $\\lim a_n = {true_limit if true_limit is not None else "?"}$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Display last few terms
    print(f"Last 5 terms: {a_n[-5:]}")
    if true_limit is not None:
        print(f"Error from limit: {np.abs(a_n[-5:] - true_limit)}")

# Examples
visualize_limit(lambda n: 1/n, "1/n", 0)
visualize_limit(lambda n: (n+1)/n, "(n+1)/n", 1)
visualize_limit(lambda n: 0.99**n, "0.99^n", 0)
```

---

## Rigorous Definition of Limits

### ε-N Definition

**Definition** ($\epsilon-N$ definition): The limit of sequence $\{a_n\}$ is $L$, denoted as $\lim_{n \to \infty} a_n = L$, if:

$$
\forall \epsilon > 0, \exists N \in \mathbb{N}, \forall n > N: |a_n - L| < \epsilon
$$

**Interpretation**:
- "For any given error $\epsilon$"
- "There exists a sufficiently large $N$"
- "When $n$ exceeds $N$"
- "The distance between $a_n$ and $L$ is less than $\epsilon$"

### Geometric Meaning

```
    L + ε  ─────────────────────────────────
           │    •  •  • • •••••••••••••••••
    L      ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
           │  •
    L - ε  ─────────────────────────────────
           │ •
           │•
           └──────────────────────────────→ n
                  N

When n > N, all a_n fall within the interval (L-ε, L+ε)
```

### Proof Techniques

**Steps to prove limits using definition**:
1. Start from $|a_n - L| < \epsilon$
2. Solve for the condition that $n$ must satisfy
3. Take $N$ as some integer satisfying the condition

**Example**: Prove $\lim_{n \to \infty} \frac{1}{n} = 0$

**Proof**:
$$
\left|\frac{1}{n} - 0\right| = \frac{1}{n} < \epsilon \Leftrightarrow n > \frac{1}{\epsilon}
$$

Take $N = \lceil \frac{1}{\epsilon} \rceil$, then when $n > N$, we have $|a_n - 0| < \epsilon$.

```python
def prove_limit_1_over_n(epsilon):
    """
    Verify the ε-N definition for lim(1/n) = 0

    For a given ε, find the minimal N such that |1/n| < ε
    """
    # From |1/n - 0| < ε, we get n > 1/ε
    N = int(np.ceil(1 / epsilon))

    print(f"For ε = {epsilon}:")
    print(f"  Need n > {1/epsilon:.4f}")
    print(f"  Take N = {N}")
    print(f"  Verify: at n = {N+1}, |1/{N+1}| = {1/(N+1):.6f} < {epsilon}")
    print()

# Different precision requirements
for eps in [0.1, 0.01, 0.001, 0.0001]:
    prove_limit_1_over_n(eps)
```

### Another Example

**Example**: Prove $\lim_{n \to \infty} \frac{n+1}{n} = 1$

**Proof**:
$$
\left|\frac{n+1}{n} - 1\right| = \left|\frac{1}{n}\right| = \frac{1}{n} < \epsilon \Leftrightarrow n > \frac{1}{\epsilon}
$$

Take $N = \lceil \frac{1}{\epsilon} \rceil$, then when $n > N$, we have $|a_n - 1| < \epsilon$.

---

## Properties of Limits

### Uniqueness

**Theorem**: The limit of a convergent sequence is unique.

**Proof** (Proof by contradiction):
Suppose $\lim a_n = L_1$ and $\lim a_n = L_2$, and $L_1 \neq L_2$.

Take $\epsilon = \frac{|L_1 - L_2|}{2} > 0$.

By the definition of limits, there exists $N_1$ such that for $n > N_1$, $|a_n - L_1| < \epsilon$.

There exists $N_2$ such that for $n > N_2$, $|a_n - L_2| < \epsilon$.

Take $n > \max(N_1, N_2)$, then:
$$
|L_1 - L_2| \leq |L_1 - a_n| + |a_n - L_2| < 2\epsilon = |L_1 - L_2|
$$

Contradiction! Therefore $L_1 = L_2$.

### Boundedness

**Theorem**: Every convergent sequence is bounded.

**Proof**:
Let $\lim a_n = L$, take $\epsilon = 1$, then there exists $N$ such that for $n > N$, $|a_n - L| < 1$.

That is, $|a_n| < |L| + 1$ (for $n > N$).

Take $M = \max(|a_1|, |a_2|, \ldots, |a_N|, |L| + 1)$, then for all $n$, $|a_n| \leq M$.

**Note**: Boundedness is a necessary condition for convergence, but not a sufficient condition.

Example: $a_n = (-1)^n$ is bounded but does not converge.

### Sign Preservation

**Theorem**: If $\lim a_n = L > 0$, then there exists $N$ such that for $n > N$, $a_n > 0$.

### Inequality Preservation

**Theorem**: If $a_n \leq b_n$ (for sufficiently large $n$), and $\lim a_n = L_1$, $\lim b_n = L_2$, then $L_1 \leq L_2$.

### Squeeze Theorem (Sandwich Theorem)

**Theorem**: If $a_n \leq c_n \leq b_n$ (for sufficiently large $n$), and $\lim a_n = \lim b_n = L$, then $\lim c_n = L$.

```python
def squeeze_theorem_demo():
    """Squeeze Theorem demonstration"""
    n = np.arange(1, 101)

    # Example: Find lim(1/n * sin(n))
    a_n = -1/n  # Lower bound
    b_n = 1/n   # Upper bound
    c_n = np.sin(n) / n  # Target sequence

    plt.figure(figsize=(10, 5))
    plt.plot(n, a_n, 'g--', label='$a_n = -1/n$', alpha=0.7)
    plt.plot(n, b_n, 'g--', label='$b_n = 1/n$', alpha=0.7)
    plt.plot(n, c_n, 'b-', label='$c_n = \\sin(n)/n$', alpha=0.7, linewidth=1)
    plt.axhline(y=0, color='r', linestyle='-', label='Limit $L = 0$', linewidth=2)

    plt.xlabel('n')
    plt.ylabel('Value')
    plt.title('Squeeze Theorem: $\\lim \\sin(n)/n = 0$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.2, 0.2)
    plt.show()

squeeze_theorem_demo()
```

### Arithmetic Operations

Let $\lim a_n = L_1$, $\lim b_n = L_2$, then:

| Operation | Result |
|-----------|--------|
| $\lim(a_n + b_n)$ | $L_1 + L_2$ |
| $\lim(a_n - b_n)$ | $L_1 - L_2$ |
| $\lim(a_n \cdot b_n)$ | $L_1 \cdot L_2$ |
| $\lim(a_n / b_n)$ | $L_1 / L_2$ (if $L_2 \neq 0$) |
| $\lim(k \cdot a_n)$ | $k \cdot L_1$ |

---

## Criteria for Convergent Sequences

### Monotone Convergence Theorem

**Theorem**: A monotone bounded sequence must converge.

- Monotonically increasing + bounded above $\Rightarrow$ converges
- Monotonically decreasing + bounded below $\Rightarrow$ converges

**Application**: Prove that the sequence $a_n = (1 + \frac{1}{n})^n$ converges (limit is $e$)

```python
def monotone_convergence_demo():
    """Monotone Convergence Theorem demonstration"""
    n = np.arange(1, 101)
    a_n = (1 + 1/n) ** n

    print("Verifying properties of the sequence (1 + 1/n)^n:")
    print(f"  First 5 terms: {a_n[:5]}")
    print(f"  Last 5 terms: {a_n[-5:]}")
    print(f"  e ≈ {np.e:.10f}")
    print(f"  a_100 = {a_n[-1]:.10f}")

    # Verify monotonicity
    diffs = np.diff(a_n)
    print(f"  Is monotonically increasing: {np.all(diffs > 0)}")

    # Verify boundedness
    print(f"  Upper bound: {a_n[-1]:.6f} < 3")

    plt.figure(figsize=(10, 5))
    plt.plot(n, a_n, 'b-', label='$(1 + 1/n)^n$', linewidth=2)
    plt.axhline(y=np.e, color='r', linestyle='--', label=f'$e = {np.e:.6f}$', linewidth=2)
    plt.xlabel('n')
    plt.ylabel('$a_n$')
    plt.title('Monotone Bounded Sequence Convergence: $\\lim(1+1/n)^n = e$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

monotone_convergence_demo()
```

### Cauchy Convergence Criterion

**Theorem**: A sequence $\{a_n\}$ converges $\Leftrightarrow$ $\forall \epsilon > 0$, $\exists N$, $\forall m, n > N$: $|a_m - a_n| < \epsilon$

**Significance**:
- No need to know the limit value; only need to check the closeness of terms within the sequence
- This is a manifestation of the completeness of real numbers

```python
def cauchy_criterion_demo():
    """Cauchy Criterion demonstration"""
    # Convergent sequence
    a_n_conv = np.array([1/n for n in range(1, 101)])

    # Divergent sequence (oscillating)
    a_n_div = np.array([(-1)**n for n in range(1, 101)])

    def check_cauchy(sequence, epsilon=0.01, tail_size=20):
        """Check if the tail of the sequence satisfies Cauchy condition"""
        tail = sequence[-tail_size:]
        max_diff = np.max(np.abs(tail[:, None] - tail[None, :]))
        return max_diff < epsilon, max_diff

    print("Cauchy Criterion Check:")
    is_cauchy, diff = check_cauchy(a_n_conv)
    print(f"  1/n: Satisfies Cauchy condition = {is_cauchy}, Max difference = {diff:.6f}")

    is_cauchy, diff = check_cauchy(a_n_div)
    print(f"  (-1)^n: Satisfies Cauchy condition = {is_cauchy}, Max difference = {diff:.6f}")

cauchy_criterion_demo()
```

### Subsequence Criterion

**Theorem**: A sequence $\{a_n\}$ converges $\Leftrightarrow$ all its subsequences converge to the same limit.

**Corollary**: If a sequence has two subsequences converging to different limits, then the original sequence diverges.

Example: For the sequence $a_n = (-1)^n$, the subsequence $a_{2k} = 1 \to 1$, $a_{2k-1} = -1 \to -1$, therefore the original sequence diverges.

---

## Important Limits

### 1. Natural Constant $e$

$$
\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e \approx 2.71828
$$

**More general form**:
$$
\lim_{n \to \infty} \left(1 + \frac{x}{n}\right)^n = e^x
$$

### 2. Exponential Decay

For $|r| < 1$:
$$
\lim_{n \to \infty} r^n = 0
$$

### 3. Polynomial Growth

For any $k > 0$:
$$
\lim_{n \to \infty} \frac{n^k}{a^n} = 0 \quad (a > 1)
$$

Exponential growth is much faster than polynomial growth.

### 4. Logarithmic Growth

$$
\lim_{n \to \infty} \frac{\ln n}{n} = 0
$$

### 5. Stolz Theorem

For sequences $\{a_n\}$ and $\{b_n\}$, if $b_n$ is strictly monotonically increasing and $\lim b_n = +\infty$:

$$
\lim_{n \to \infty} \frac{a_n}{b_n} = \lim_{n \to \infty} \frac{a_{n+1} - a_n}{b_{n+1} - b_n}
$$

(Assuming the limit on the right exists)

```python
def important_limits_demo():
    """Important Limits demonstration"""
    n = np.arange(1, 201)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. (1 + 1/n)^n → e
    axes[0, 0].plot(n, (1 + 1/n)**n, label='$(1+1/n)^n$')
    axes[0, 0].axhline(y=np.e, color='r', linestyle='--', label=f'$e = {np.e:.4f}$')
    axes[0, 0].set_title('$\\lim(1+1/n)^n = e$')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. r^n → 0 (|r| < 1)
    for r in [0.9, 0.5, 0.1]:
        axes[0, 1].plot(n, r**n, label=f'${r}^n$')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_title('$\\lim r^n = 0$ for $|r| < 1$')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(-0.1, 1)

    # 3. n^k / 2^n → 0
    for k in [1, 2, 5]:
        axes[1, 0].plot(n, n**k / 2**n, label=f'$n^{k}/2^n$')
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_title('$\\lim n^k/a^n = 0$ (exponential dominates)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. ln(n)/n → 0
    axes[1, 1].plot(n, np.log(n)/n, label='$\\ln(n)/n$')
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_title('$\\lim \\ln(n)/n = 0$')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

important_limits_demo()
```

---

## Applications in Deep Learning

### 1. Training Convergence

During the training process, the loss sequence $\{L_t\}$ should converge:

$$
\lim_{t \to \infty} L_t = L^* \quad \text{(optimal loss)}
$$

**Verification Methods**:
- Check if the loss curve tends to flatten
- Use Cauchy criterion: loss changes over consecutive epochs are below a threshold

```python
def check_training_convergence(losses, epsilon=1e-4, window=10):
    """Check if training has converged"""
    if len(losses) < window:
        return False, 0

    recent = losses[-window:]
    max_diff = np.max(recent) - np.min(recent)

    converged = max_diff < epsilon
    return converged, max_diff

# Simulate training loss
np.random.seed(42)
epochs = 200
losses = 1.0 / np.arange(1, epochs+1) + np.random.randn(epochs) * 0.01

plt.figure(figsize=(10, 5))
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Convergence Check')
plt.grid(True, alpha=0.3)

# Mark convergence point
for i in range(10, len(losses)):
    converged, diff = check_training_convergence(losses[:i+1])
    if converged:
        plt.axvline(x=i, color='r', linestyle='--', label=f'Converged at epoch {i}')
        break

plt.legend()
plt.show()
```

### 2. Learning Rate Decay

The learning rate sequence $\{\eta_t\}$ is typically designed to converge to $0$:

$$
\lim_{t \to \infty} \eta_t = 0
$$

This ensures that the parameter update step size becomes progressively smaller, eventually stabilizing.

### 3. Gradient Vanishing/Explosion

In RNNs, gradient propagation involves the product of terms:

$$
\frac{\partial L}{\partial h_1} = \prod_{t=1}^{T-1} \frac{\partial h_{t+1}}{\partial h_t}
$$

If each gradient term $\approx \lambda$, then the total gradient $\approx \lambda^T$.

- When $|\lambda| < 1$: $\lim_{T \to \infty} \lambda^T = 0$ (gradient vanishing)
- When $|\lambda| > 1$: $\lim_{T \to \infty} \lambda^T = \infty$ (gradient explosion)

```python
def gradient_analysis():
    """RNN gradient vanishing/explosion analysis"""
    T = 50  # Sequence length

    # Different eigenvalue cases
    lambdas = [0.9, 1.0, 1.1]

    plt.figure(figsize=(10, 5))
    for lam in lambdas:
        gradients = [lam ** t for t in range(T)]
        plt.semilogy(gradients, label=f'$\\lambda = {lam}$')

    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Time Step')
    plt.ylabel('Gradient Magnitude (log)')
    plt.title('RNN Gradient Propagation: Limit Behavior of $\\lambda^T$')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

gradient_analysis()
```

---

## Summary

This section introduced core concepts of sequence limits:

| Concept | Definition/Property | Importance |
|---------|---------------------|------------|
| $\epsilon-N$ definition | $\forall \epsilon > 0, \exists N, n > N: \|a_n - L\| < \epsilon$ | Rigorous mathematical definition |
| Uniqueness | Limit value is unique | Theoretical guarantee |
| Boundedness | Convergent sequences are bounded | Determination tool |
| Squeeze Theorem | $a_n \leq c_n \leq b_n$, equal limits on both sides $\Rightarrow$ equal limit in middle | Limit evaluation technique |
| Monotone Convergence Theorem | Monotone + Bounded $\Rightarrow$ Convergence | Convergence determination |
| Cauchy Criterion | Convergence determination independent of limit value | Foundation of completeness |

**Key Takeaways**:
- The $\epsilon-N$ definition is the rigorous mathematical description of limits
- Convergent sequences have a unique limit and are bounded
- The Squeeze Theorem and Monotone Convergence Theorem are important tools for determining convergence
- The Cauchy Criterion does not require knowing the limit value

---

**Next Section**: [Series and Summation](07c-series-summation_EN.md) - Learn about the convergence and summation techniques of infinite series.

**Back to**: [Chapter 7: Sequences and Series](07-sequences-series_EN.md) | [Math Fundamentals Tutorial Index](../math-fundamentals_EN.md)
