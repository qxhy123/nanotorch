# Chapter 3 (d): Limit Theorems and Information Theory

Limit theorems (law of large numbers and central limit theorem) are the cornerstones of probability theory, explaining why many techniques in deep learning (such as batch normalization, stochastic gradient descent) work effectively. Information theory provides a mathematical framework for quantifying uncertainty and information quantity. Cross-entropy loss functions and KL divergence are core concepts in information theory.

---

## 🎯 Life Analogy: Limit Theorems are "What Happens on Average"

### Law of Large Numbers = "Strength in Numbers, Averages Stabilize"

Imagine you're estimating the average height of all students:
- Ask 1 person: Might be very tall or very short, not accurate
- Ask 10 people: Better, but still might be biased
- Ask 1000 people: Very close to the true average height!
- Ask everyone: Almost exactly the true average

```
Sample size:   1 person   10 people   100 people   1000 people
                 ↓           ↓           ↓            ↓
Estimate:     185cm      172cm       168cm        170.1cm
                 ↑           ↑           ↑            ↑
            Very unstable  Stabilizing  More stable  Very close to true (170cm)
```

**Law of Large Numbers tells us**: More samples = more stable average = converges to true expectation.

### Central Limit Theorem = "Why Bell Curves Are Everywhere"

No matter what the original distribution looks like, **the sum of many independent random variables** tends toward a **normal distribution** (bell curve).

```
Rolling 1 die: Each face equally likely ━━━━━━━━ (uniform distribution)

Sum of 100 dice: ╭───╮
                  ╱     ╲
                 ╱       ╲
                ╱         ╲
               ───────────── (normal distribution!)
```

**That's why**: Height, test scores, measurement errors are all approximately normal—they're sums of many small factors.

### Entropy = "Measuring Uncertainty"

**Entropy = Average surprise level**

| Event | Probability | Surprise Level |
|-------|-------------|----------------|
| Sun rises tomorrow | 99.99% | Very small (expected) |
| Win lottery | 0.01% | Very large (shocking!) |
| Coin flip heads | 50% | Medium |

```
High entropy ─────────────────────────────────→ Low entropy
    │                                                │
Very uncertain (coin flip)                    Very certain (sun rises)
    │                                                │
Lots of information (result tells you a lot)  Little information (not surprising)
```

### 📖 Cross Entropy = "Encoding with Wrong Distribution"

**Scenario**: You're guessing if a coin is heads or tails
- Reality: The coin has 90% chance of heads
- Your guess: You think it's 50-50

Using your guess to encode, you'll waste many bits!

**Cross-entropy loss**: Measures the gap between your predicted distribution and the true distribution.

---

## Table of Contents

1. [Law of Large Numbers](#law-of-large-numbers)
2. [Central Limit Theorem](#central-limit-theorem)
3. [Monte Carlo Methods](#monte-carlo-methods)
4. [Information Theory Basics](#information-theory-basics)
    - [Entropy](#entropy)
    - [Joint Entropy and Conditional Entropy](#joint-entropy-and-conditional-entropy)
    - [Cross Entropy](#cross-entropy)
5. [KL Divergence](#kl-divergence-kullback-leibler-divergence)
6. [Mutual Information](#mutual-information)
7. [Relationship Between Information Theory and Machine Learning](#relationship-between-information-theory-and-machine-learning)
8. [Applications in Deep Learning](#applications-in-deep-learning)
9. [Summary](#summary)

---

## Law of Large Numbers

### Weak Law of Large Numbers (WLLN)

Let $X_1, X_2, \ldots, X_n$ be **independent and identically distributed (i.i.d.)** random variables, $\mathbb{E}[X_i] = \mu$. Then the **sample mean** converges in probability to the expectation:

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \xrightarrow{P} \mu
$$

That is, for any $\epsilon > 0$:

$$
\lim_{n \to \infty} P(|\bar{X}_n - \mu| < \epsilon) = 1
$$

### Strong Law of Large Numbers (SLLN)

The sample mean converges **almost surely** to the expectation:

$$
P\left(\lim_{n \to \infty} \bar{X}_n = \mu\right) = 1
$$

### Intuitive Understanding

When the sample size is sufficiently large, the sample mean will be "close" to the true expectation. This explains why:
- **Training with large amounts of data** learns the true distribution
- **Large batch training** gives more stable gradient estimates

### Conditions for Law of Large Numbers

1. Samples are independent and identically distributed
2. Expectation exists and is finite

```python
import numpy as np
import matplotlib.pyplot as plt

# Law of large numbers demonstration
np.random.seed(42)

# Settings
true_mean = 3.5  # Expectation of a die
max_samples = 10000

# Die roll experiment
dice_rolls = np.random.randint(1, 7, max_samples)

# Calculate cumulative mean
cumulative_mean = np.cumsum(dice_rolls) / np.arange(1, max_samples + 1)

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(np.arange(1, max_samples + 1), cumulative_mean, 'b-', alpha=0.7, label='Sample mean')
plt.axhline(y=true_mean, color='r', linestyle='--', linewidth=2, label=f'True expectation μ={true_mean}')
plt.xlabel('Sample size n')
plt.ylabel('Sample mean')
plt.title('Law of Large Numbers Demonstration: Rolling a Die')
plt.legend()
plt.grid(True, alpha=0.3)

# Zoom in on first 500
plt.subplot(1, 2, 2)
plt.plot(np.arange(1, 501), cumulative_mean[:500], 'b-', alpha=0.7)
plt.axhline(y=true_mean, color='r', linestyle='--', linewidth=2)
plt.xlabel('Sample size n')
plt.ylabel('Sample mean')
plt.title('Law of Large Numbers Demonstration (First 500)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('law_of_large_numbers.png', dpi=100)
print("Law of large numbers demonstration image saved")

# Convergence speed analysis
print("\nConvergence speed analysis:")
print("="*50)
checkpoints = [10, 50, 100, 500, 1000, 5000, 10000]
for n in checkpoints:
    error = abs(cumulative_mean[n-1] - true_mean)
    print(f"n = {n:5d}: sample mean = {cumulative_mean[n-1]:.4f}, error = {error:.4f}")
```

---

## Central Limit Theorem

### Standard Form

Let $X_1, X_2, \ldots, X_n$ be i.i.d. random variables, $\mathbb{E}[X_i] = \mu$, $\text{Var}(X_i) = \sigma^2$. Then:

$$
\frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

Equivalently:

$$
\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)
$$

### Practical Form

For large $n$:

$$
\bar{X}_n \approx \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)
$$

$$
\sum_{i=1}^n X_i \approx \mathcal{N}(n\mu, n\sigma^2)
$$

### Significance of Central Limit Theorem

1. **Universality of normal distribution**: Regardless of the original distribution, sample means tend to normal
2. **Foundation of statistical inference**: Confidence intervals, hypothesis testing
3. **Neural network weights**: Weighted sums of many inputs approximate normal distribution

### Application Conditions

- Sample size $n$ sufficiently large (usually $n \geq 30$)
- Finite variance of original distribution

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Central limit theorem demonstration
np.random.seed(42)

# Settings
n_samples = 10000  # Number of repetitions
sample_sizes = [1, 5, 10, 30, 100, 500]  # Different sample sizes

# Original distribution: uniform distribution (0, 1)
# E[X] = 0.5, Var(X) = 1/12

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, n in enumerate(sample_sizes):
    row, col = idx // 3, idx % 3

    # Generate n_samples sample means
    # Each sample mean is the average of n uniform distributions
    samples = np.random.uniform(0, 1, (n_samples, n))
    sample_means = samples.mean(axis=1)

    # Standardize
    true_mean = 0.5
    true_std = np.sqrt(1/12 / n)
    standardized = (sample_means - true_mean) / true_std

    # Plot histogram
    axes[row, col].hist(standardized, bins=50, density=True, alpha=0.7, label='Standardized sample mean')

    # Overlay standard normal distribution
    x = np.linspace(-4, 4, 100)
    axes[row, col].plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')

    axes[row, col].set_title(f'n = {n}')
    axes[row, col].set_xlabel('Standardized value')
    axes[row, col].set_ylabel('Density')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)
    axes[row, col].set_xlim(-4, 4)

plt.suptitle('Central Limit Theorem: Sample Means of Uniform Distribution Tend to Normal', fontsize=14)
plt.tight_layout()
plt.savefig('central_limit_theorem.png', dpi=100)
print("Central limit theorem demonstration image saved")

# Quantitative analysis
print("\nCentral limit theorem quantitative analysis:")
print("="*60)
print(f"{'Sample size n':>10} | {'Sample mean':>10} | {'Theoretical mean':>10} | {'Sample variance':>10} | {'Theoretical variance':>10}")
print("-"*60)

for n in [10, 30, 100, 500]:
    samples = np.random.uniform(0, 1, (10000, n))
    sample_means = samples.mean(axis=1)

    theoretical_mean = 0.5
    theoretical_var = 1/12 / n

    print(f"{n:>10} | {sample_means.mean():>10.4f} | {theoretical_mean:>10.4f} | {sample_means.var():>10.6f} | {theoretical_var:>10.6f}")
```

---

## Monte Carlo Methods

### Basic Idea

Use **random sampling** to estimate numerical results:

$$
\mathbb{E}[g(X)] \approx \frac{1}{n} \sum_{i=1}^n g(x_i)
$$

Where $x_1, x_2, \ldots, x_n$ are independent samples from the distribution.

### Monte Carlo Integration

Calculate integral $\int_a^b f(x) dx$:

$$
\int_a^b f(x) dx = (b-a) \mathbb{E}[f(X)] \approx \frac{b-a}{n} \sum_{i=1}^n f(x_i)
$$

Where $X \sim \text{Uniform}(a, b)$.

### Error Analysis

Standard error of Monte Carlo estimate:

$$
\text{SE} = \frac{\sigma}{\sqrt{n}}
$$

Convergence speed: $O(n^{-1/2})$ (independent of dimension)

```python
import numpy as np

# Monte Carlo integration example: Calculate π
np.random.seed(42)

def monte_carlo_pi(n_samples):
    """Estimate π using Monte Carlo method"""
    # Random sampling within unit square
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)

    # Calculate proportion falling within unit circle
    inside = (x**2 + y**2) <= 1
    pi_estimate = 4 * inside.mean()

    return pi_estimate, inside

# Estimates for different sample sizes
sample_sizes = [100, 1000, 10000, 100000, 1000000]

print("Monte Carlo estimation of π:")
print("="*50)
print(f"{'Sample size':>10} | {'Estimate':>10} | {'Error':>10}")
print("-"*50)

for n in sample_sizes:
    pi_est, _ = monte_carlo_pi(n)
    error = abs(pi_est - np.pi)
    print(f"{n:>10} | {pi_est:>10.6f} | {error:>10.6f}")

# Visualization
n_vis = 5000
pi_est, inside = monte_carlo_pi(n_vis)
x = np.random.uniform(-1, 1, n_vis)
y = np.random.uniform(-1, 1, n_vis)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 8))
plt.scatter(x[inside], y[inside], c='blue', alpha=0.5, s=1, label='Inside circle')
plt.scatter(x[~inside], y[~inside], c='red', alpha=0.5, s=1, label='Outside circle')
circle = plt.Circle((0, 0), 1, fill=False, color='green', linewidth=2)
plt.gca().add_patch(circle)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect('equal')
plt.title(f'Monte Carlo Estimate π ≈ {pi_est:.4f} (n={n_vis})')
plt.legend()
plt.savefig('monte_carlo_pi.png', dpi=100)
print("\nMonte Carlo π image saved")

# Monte Carlo integration example: Calculate E[sin(X)] where X ~ N(0,1)
print("\nMonte Carlo integration:")
print("="*50)

# True value (numerical integration)
from scipy import integrate
true_value, _ = integrate.quad(lambda x: np.sin(x) * stats.norm.pdf(x), -10, 10)
print(f"True value E[sin(X)] = {true_value:.6f}")

# Monte Carlo estimate
for n in [100, 1000, 10000, 100000]:
    samples = np.random.normal(0, 1, n)
    estimate = np.sin(samples).mean()
    error = abs(estimate - true_value)
    print(f"n = {n:>6}: estimate = {estimate:.6f}, error = {error:.6f}")
```

---

## Information Theory Basics

### Entropy

#### Definition

**Entropy** measures the **uncertainty** or **information quantity** of a random variable.

**Discrete entropy**:

$$
H(X) = -\sum_{x} p(x) \log p(x) = \mathbb{E}[-\log p(X)]
$$

**Differential entropy** (continuous):

$$
H(X) = -\int f(x) \log f(x) \, dx
$$

#### Entropy Units

| Logarithm base | Unit |
|--------|------|
| 2 | bit |
| $e$ | nat |
| 10 | hart |

In machine learning, nat (natural logarithm) is typically used.

#### Properties of Entropy

1. **Non-negativity**: $H(X) \geq 0$ (discrete case)
2. **Maximum at equal probability**: For discrete variables with $n$ possible values, $H(X) \leq \log n$, with equality when $p(x) = 1/n$
3. **Zero entropy for deterministic variables**: If $X$ always equals some value, then $H(X) = 0$
4. **Information quantity**: $-\log p(x)$ is the "surprise" of observing event $x$

#### Intuitive Understanding of Entropy

- **High entropy**: Uniform distribution, high uncertainty, many "surprises"
- **Low entropy**: Concentrated distribution, low uncertainty, predictable results

```python
import numpy as np
import matplotlib.pyplot as plt

def entropy(p):
    """Calculate entropy of discrete distribution"""
    # Filter zero probabilities
    p = p[p > 0]
    return -np.sum(p * np.log(p))

# Example: Entropy of different distributions
print("Entropy calculation examples:")
print("="*50)

# 1. Uniform distribution (maximum entropy)
p_uniform = np.array([0.25, 0.25, 0.25, 0.25])
print(f"Uniform distribution {p_uniform}: H = {entropy(p_uniform):.4f} (maximum entropy = {np.log(4):.4f})")

# 2. Skewed distribution
p_skewed = np.array([0.7, 0.1, 0.1, 0.1])
print(f"Skewed distribution {p_skewed}: H = {entropy(p_skewed):.4f}")

# 3. Deterministic distribution (minimum entropy)
p_deterministic = np.array([1.0, 0.0, 0.0, 0.0])
print(f"Deterministic distribution {p_deterministic}: H = {entropy(p_deterministic):.4f}")

# 4. Binary distribution: H(p) = -p*log(p) - (1-p)*log(1-p)
print("\nBinary entropy function H(p) = -p*log(p) - (1-p)*log(1-p):")
p_values = np.linspace(0.01, 0.99, 100)
binary_entropy = lambda p: -p * np.log(p) - (1-p) * np.log(1-p)
H_values = [binary_entropy(p) for p in p_values]

plt.figure(figsize=(10, 5))
plt.plot(p_values, H_values, 'b-', linewidth=2)
plt.axvline(x=0.5, color='r', linestyle='--', label='p=0.5 (maximum entropy)')
plt.xlabel('p')
plt.ylabel('H(p)')
plt.title('Binary Entropy Function H(p) = -p log p - (1-p) log(1-p)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('binary_entropy.png', dpi=100)
print("Binary entropy function image saved")

print(f"Maximum entropy at p=0.5: H(0.5) = {binary_entropy(0.5):.4f} = log(2)")
```

---

### Joint Entropy and Conditional Entropy

#### Joint Entropy

$$
H(X, Y) = -\sum_{x,y} p(x, y) \log p(x, y)
$$

#### Conditional Entropy

$$
H(Y|X) = -\sum_{x,y} p(x, y) \log p(y|x) = \mathbb{E}_{X}[-\log p(Y|X)]
$$

#### Chain Rule

$$
H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
$$

#### Intuitive Understanding

- $H(Y|X)$: Remaining uncertainty in $Y$ given $X$
- If $X$ and $Y$ are independent: $H(Y|X) = H(Y)$

```python
import numpy as np

# Joint entropy and conditional entropy examples
# Define joint distribution P(X, Y)
joint = np.array([
    [0.1, 0.1, 0.05],  # X=0
    [0.15, 0.2, 0.1],  # X=1
    [0.1, 0.15, 0.05]  # X=2
])

# Marginal distributions
p_X = joint.sum(axis=1)
p_Y = joint.sum(axis=0)

# Calculate various entropies
def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

H_X = entropy(p_X)
H_Y = entropy(p_Y)
H_XY = entropy(joint.flatten())

# Conditional entropies
H_Y_given_X = H_XY - H_X
H_X_given_Y = H_XY - H_Y

print("Joint entropy and conditional entropy:")
print("="*50)
print(f"H(X) = {H_X:.4f}")
print(f"H(Y) = {H_Y:.4f}")
print(f"H(X,Y) = {H_XY:.4f}")
print(f"H(Y|X) = H(X,Y) - H(X) = {H_Y_given_X:.4f}")
print(f"H(X|Y) = H(X,Y) - H(Y) = {H_X_given_Y:.4f}")

# Verify chain rule
print(f"\nVerify chain rule:")
print(f"H(X,Y) = H(X) + H(Y|X) = {H_X + H_Y_given_X:.4f}")
print(f"H(X,Y) = H(Y) + H(X|Y) = {H_Y + H_X_given_Y:.4f}")
```

---

### Cross Entropy

#### Definition

**Cross entropy** measures the average number of bits required to encode distribution $P$ using distribution $Q$:

$$
H(P, Q) = -\sum_x p(x) \log q(x) = \mathbb{E}_{x \sim P}[-\log q(x)]
$$

#### Relationship with Entropy

$$
H(P, Q) = H(P) + D_{KL}(P \| Q)
$$

#### Cross-Entropy Loss

In classification problems, true distribution $P$ is one-hot encoded, predicted distribution is $Q$:

$$
L = H(P, Q) = -\sum_{i=1}^K y_i \log \hat{y}_i
$$

For one-hot labels, this simplifies to:

$$
L = -\log \hat{y}_{\text{true}}
$$

```python
import numpy as np

def cross_entropy(p, q):
    """Calculate cross entropy H(P, Q)"""
    # Filter zero probabilities (avoid log(0))
    mask = p > 0
    return -np.sum(p[mask] * np.log(q[mask]))

# Example
print("Cross entropy example:")
print("="*50)

# True distribution
P = np.array([1, 0, 0, 0])  # one-hot: class 0 is true label

# Different prediction distributions
Q1 = np.array([0.9, 0.05, 0.03, 0.02])  # Good prediction
Q2 = np.array([0.5, 0.3, 0.15, 0.05])   # Medium prediction
Q3 = np.array([0.1, 0.3, 0.4, 0.2])     # Bad prediction

print(f"True distribution P: {P}")
print(f"\nPrediction Q1: {Q1}")
print(f"  Cross entropy H(P, Q1) = {cross_entropy(P, Q1):.4f}")
print(f"  -log(0.9) = {-np.log(0.9):.4f}")

print(f"\nPrediction Q2: {Q2}")
print(f"  Cross entropy H(P, Q2) = {cross_entropy(P, Q2):.4f}")
print(f"  -log(0.5) = {-np.log(0.5):.4f}")

print(f"\nPrediction Q3: {Q3}")
print(f"  Cross entropy H(P, Q3) = {cross_entropy(P, Q3):.4f}")
print(f"  -log(0.1) = {-np.log(0.1):.4f}")

print("\nConclusion: Better prediction gives smaller cross entropy")
```

---

## KL Divergence (Kullback-Leibler Divergence)

### Definition

**KL divergence** (relative entropy) measures the "distance" between two distributions:

$$
D_{KL}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(P, Q) - H(P)
$$

Continuous case:

$$
D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} \, dx
$$

### Properties

1. **Non-negativity**: $D_{KL}(P \| Q) \geq 0$
2. **Zero value**: $D_{KL}(P \| Q) = 0$ if and only if $P = Q$
3. **Asymmetry**: $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$
4. **Not a true distance**: Doesn't satisfy triangle inequality

**Proof of KL divergence non-negativity** (using Jensen's inequality):

**Step 1**: Write the definition of KL divergence.

$$D_{KL}(P \| Q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

**Step 2**: Convert to negative form.

$$= -\sum_x p(x) \log \frac{q(x)}{p(x)}$$

**Step 3**: Identify this as an expectation.

$$= -\mathbb{E}_{x \sim P}\left[\log \frac{q(x)}{p(x)}\right]$$

**Step 4**: Apply Jensen's inequality.

Since $\log$ is a **concave function**, Jensen's inequality gives:

$$\mathbb{E}[\log f(x)] \leq \log \mathbb{E}[f(x)]$$

Therefore:

$$-\mathbb{E}\left[\log \frac{q(x)}{p(x)}\right] \geq -\log \mathbb{E}\left[\frac{q(x)}{p(x)}\right]$$

**Step 5**: Calculate the expectation.

$$\mathbb{E}\left[\frac{q(x)}{p(x)}\right] = \sum_x p(x) \cdot \frac{q(x)}{p(x)} = \sum_x q(x) = 1$$

**Step 6**: Draw the conclusion.

$$D_{KL}(P \| Q) \geq -\log(1) = 0$$

$$\boxed{D_{KL}(P \| Q) \geq 0}$$

**Equality condition**: When $\frac{q(x)}{p(x)}$ is constant for all $x$, i.e., $p(x) = q(x)$.

**Why KL divergence is not symmetric**:

Consider a simple binary distribution:
- $P = (1, 0)$ (deterministic on first state)
- $Q = (0.5, 0.5)$ (uniform distribution)

$$D_{KL}(P \| Q) = 1 \cdot \log\frac{1}{0.5} + 0 \cdot \log\frac{0}{0.5} = \log 2$$

But $\log\frac{0}{0.5}$ is undefined ($-\infty$), so $D_{KL}(Q \| P) = +\infty$

### Forward KL vs Reverse KL

| Type | Formula | Behavior |
|------|------|------|
| Forward KL | $D_{KL}(P \| Q)$ | Q covers all support of P |
| Reverse KL | $D_{KL}(Q \| P)$ | Q focuses on high-probability regions of P |

### Deep Learning Applications

- **VAE**: KL divergence as regularization term
- **Knowledge distillation**: Student network learning teacher network's distribution
- **Generative models**: Minimize KL divergence between generated and true distributions

```python
import numpy as np
import matplotlib.pyplot as plt

def kl_divergence(p, q):
    """Calculate KL divergence D_KL(P || Q)"""
    # Filter zero probabilities
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

# Example: KL divergence between two Gaussian distributions
print("KL divergence example:")
print("="*50)

# Discrete distribution example
P = np.array([0.3, 0.4, 0.2, 0.1])
Q = np.array([0.25, 0.25, 0.25, 0.25])

kl_pq = kl_divergence(P, Q)
kl_qp = kl_divergence(Q, P)

print(f"P = {P}")
print(f"Q = {Q}")
print(f"D_KL(P || Q) = {kl_pq:.4f}")
print(f"D_KL(Q || P) = {kl_qp:.4f}")
print(f"Asymmetric: D_KL(P||Q) ≠ D_KL(Q||P)")

# KL divergence of Gaussian distributions (analytical solution available)
print("\nKL divergence of Gaussian distributions:")
print("-"*50)

def kl_gaussian(mu1, sigma1, mu2, sigma2):
    """
    Calculate KL divergence between N(mu1, sigma1²) and N(mu2, sigma2²)
    D_KL(N1 || N2) = log(sigma2/sigma1) + (sigma1² + (mu1-mu2)²) / (2*sigma2²) - 1/2
    """
    return np.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5

# Example
mu1, sigma1 = 0, 1
mu2, sigma2 = 1, 2

kl_n1_n2 = kl_gaussian(mu1, sigma1, mu2, sigma2)
kl_n2_n1 = kl_gaussian(mu2, sigma2, mu1, sigma1)

print(f"N1 = N({mu1}, {sigma1}²)")
print(f"N2 = N({mu2}, {sigma2}²)")
print(f"D_KL(N1 || N2) = {kl_n1_n2:.4f}")
print(f"D_KL(N2 || N1) = {kl_n2_n1:.4f}")

# Visualization
x = np.linspace(-5, 7, 200)
from scipy.stats import norm

plt.figure(figsize=(10, 5))
plt.plot(x, norm.pdf(x, mu1, sigma1), 'b-', linewidth=2, label=f'N1: N({mu1}, {sigma1}²)')
plt.plot(x, norm.pdf(x, mu2, sigma2), 'r-', linewidth=2, label=f'N2: N({mu2}, {sigma2}²)')
plt.xlabel('x')
plt.ylabel('Density')
plt.title(f'KL Divergence: D_KL(N1||N2) = {kl_n1_n2:.4f}, D_KL(N2||N1) = {kl_n2_n1:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('kl_divergence.png', dpi=100)
print("\nKL divergence visualization image saved")
```

---

## Mutual Information

### Definition

**Mutual information** measures the degree of **mutual dependence** between two random variables:

$$
I(X; Y) = \sum_{x,y} p(x, y) \log \frac{p(x, y)}{p(x)p(y)}
$$

### Equivalent Forms

$$
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
$$

$$
I(X; Y) = H(X) + H(Y) - H(X, Y)
$$

$$
I(X; Y) = D_{KL}(P(X,Y) \| P(X)P(Y))
$$

### Properties

1. **Non-negativity**: $I(X; Y) \geq 0$
2. **Symmetry**: $I(X; Y) = I(Y; X)$
3. **Independence**: $X$ and $Y$ independent $\Leftrightarrow$ $I(X; Y) = 0$
4. **Self-information**: $I(X; X) = H(X)$

### Intuitive Understanding of Mutual Information

- $I(X; Y)$: How much uncertainty about $X$ is reduced given $Y$
- Also: How much uncertainty about $Y$ is reduced given $X$

### Information Diagram (Information Diagram)

```
         H(X)              H(Y)
     ┌───────────┐     ┌───────────┐
     │    ╭──────┴─────╶──────╮    │
     │    │   I(X;Y)    │    │
     │    ╰──────┬─────╶──────╯    │
     │ H(X|Y)    │    H(Y|X) │
     └───────────┴───────────┘
               H(X,Y)
```

```python
import numpy as np

def mutual_information(joint):
    """Calculate mutual information I(X; Y)"""
    # Marginal distributions
    p_X = joint.sum(axis=1, keepdims=True)
    p_Y = joint.sum(axis=0, keepdims=True)

    # Independent joint distribution
    p_independent = p_X * p_Y

    # I(X;Y) = D_KL(P(X,Y) || P(X)P(Y))
    # Filter zero probabilities
    mask = (joint > 0) & (p_independent > 0)
    mi = np.sum(joint[mask] * np.log(joint[mask] / p_independent[mask]))

    return mi

# Example
print("Mutual information example:")
print("="*50)

# Complete dependence (Y completely determined by X)
joint_dependent = np.array([
    [0.25, 0, 0, 0],
    [0, 0.25, 0, 0],
    [0, 0, 0.25, 0],
    [0, 0, 0, 0.25]
])

# Independent
joint_independent = np.outer([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25])

# Partial dependence
joint_partial = np.array([
    [0.2, 0.05, 0, 0],
    [0.05, 0.2, 0, 0],
    [0, 0, 0.2, 0.05],
    [0, 0, 0.05, 0.2]
])

mi_dependent = mutual_information(joint_dependent)
mi_independent = mutual_information(joint_independent)
mi_partial = mutual_information(joint_partial)

print(f"Complete dependence: I(X;Y) = {mi_dependent:.4f} (maximum dependence)")
print(f"Independent: I(X;Y) = {mi_independent:.4f} (independent)")
print(f"Partial dependence: I(X;Y) = {mi_partial:.4f}")

# Verify I(X;Y) = H(X) - H(X|Y)
def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

p_X = joint_partial.sum(axis=1)
p_Y = joint_partial.sum(axis=0)
H_X = entropy(p_X)
H_XY = entropy(joint_partial.flatten())
H_X_given_Y = H_XY - H_Y

print(f"\nVerify I(X;Y) = H(X) - H(X|Y):")
print(f"H(X) = {H_X:.4f}")
print(f"H(X|Y) = {H_X_given_Y:.4f}")
print(f"H(X) - H(X|Y) = {H_X - H_X_given_Y:.4f}")
print(f"I(X;Y) = {mi_partial:.4f}")
```

---

## Relationship Between Information Theory and Machine Learning

### Key Connections

| Information Theory Concept | Machine Learning Application |
|------------|--------------|
| Entropy | Decision tree splitting criteria |
| Cross entropy | Classification loss function |
| KL divergence | VAE regularization, knowledge distillation |
| Mutual information | Feature selection, representation learning |
| Information gain | Decision trees, active learning |

### Cross-Entropy Loss = Maximum Likelihood Estimation

Minimizing cross-entropy loss is equivalent to maximizing likelihood:

$$
\min_\theta H(P_{\text{data}}, P_\theta) \Leftrightarrow \max_\theta \mathbb{E}_{x \sim P_{\text{data}}}[\log P_\theta(x)]
$$

### Information Gain in Decision Trees

Choose features that maximize information gain for splitting:

$$
IG(Y|X) = H(Y) - H(Y|X)
$$

Information gain = Mutual information $I(X; Y)$

```python
import numpy as np

# Decision tree information gain example
def entropy(p):
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def information_gain(y, x):
    """
    Calculate information gain of feature x on target y
    IG(Y|X) = H(Y) - H(Y|X)
    """
    # H(Y)
    p_y = np.bincount(y) / len(y)
    H_y = entropy(p_y)

    # H(Y|X) = sum_x P(X=x) * H(Y|X=x)
    unique_x = np.unique(x)
    H_y_given_x = 0
    for val in unique_x:
        mask = x == val
        p_x = mask.mean()  # P(X=x)
        y_subset = y[mask]
        p_y_given_x = np.bincount(y_subset) / len(y_subset)
        H_y_given_x += p_x * entropy(p_y_given_x)

    return H_y - H_y_given_x

# Example: Play tennis
# Features: weather (0=sunny, 1=cloudy, 2=rainy)
# Target: play or not (0=no, 1=yes)

weather = np.array([0, 0, 1, 2, 2, 2, 1, 0, 0, 2, 0, 1, 1, 2])
play = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

print("Decision tree information gain example:")
print("="*50)
print(f"Information gain of weather on playing: {information_gain(play, weather):.4f}")

# More features
temp = np.array([0, 0, 0, 1, 2, 2, 2, 1, 2, 1, 1, 1, 0, 1])  # 0=hot, 1=mild, 2=cool
humidity = np.array([0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0])  # 0=high, 1=normal
windy = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1])  # 0=no wind, 1=windy

print(f"Information gain of temperature on playing: {information_gain(play, temp):.4f}")
print(f"Information gain of humidity on playing: {information_gain(play, humidity):.4f}")
print(f"Information gain of windiness on playing: {information_gain(play, windy):.4f}")
```

---

## Applications in Deep Learning

### 1. Cross-Entropy Loss Function

```python
import numpy as np

def softmax(logits):
    """Numerically stable Softmax"""
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

def cross_entropy_loss(logits, targets):
    """
    Cross-entropy loss

    Parameters:
    -----------
    logits : array, shape (batch_size, num_classes)
        Unnormalized prediction scores
    targets : array, shape (batch_size,)
        Class indices
    """
    batch_size = logits.shape[0]

    # Softmax
    probs = softmax(logits)

    # Numerical stability
    probs = np.clip(probs, 1e-15, 1 - 1e-15)

    # Cross-entropy
    log_probs = np.log(probs[np.arange(batch_size), targets])
    loss = -np.mean(log_probs)

    return loss

# Example
logits = np.array([
    [2.0, 1.0, 0.1],
    [0.1, 3.0, 0.5],
    [0.5, 0.5, 2.0]
])
targets = np.array([0, 1, 2])

loss = cross_entropy_loss(logits, targets)
print(f"Cross-entropy loss: {loss:.4f}")

# Relationship with MLE
print("\nRelationship between cross-entropy loss and MLE:")
print("Minimizing cross-entropy = Maximizing log-likelihood")
print("L = -1/N * sum(log p(y_i|x_i))")
print("= -1/N * sum(log softmax(z_i)[y_i])")
```

### 2. KL Divergence Regularization in VAE

```python
import numpy as np

def vae_kl_loss(mu, log_var):
    """
    KL divergence loss in VAE

    KL(N(μ, σ²) || N(0, 1)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
    """
    return -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))

def vae_loss(x_reconstructed, x, mu, log_var, beta=1.0):
    """
    β-VAE loss function

    L = Reconstruction Loss + β * KL Divergence
    """
    # Reconstruction loss (assuming Gaussian distribution)
    recon_loss = np.mean((x_reconstructed - x) ** 2)

    # KL divergence
    kl_loss = vae_kl_loss(mu, log_var)

    return recon_loss + beta * kl_loss

# Example
np.random.seed(42)
batch_size = 32
latent_dim = 10

mu = np.random.randn(batch_size, latent_dim)
log_var = np.random.randn(batch_size, latent_dim) * 0.1
x = np.random.randn(batch_size, 784)
x_recon = x + np.random.randn(batch_size, 784) * 0.1

loss = vae_loss(x_recon, x, mu, log_var)
print(f"VAE loss: {loss:.4f}")
print(f"  KL divergence: {vae_kl_loss(mu, log_var):.4f}")
```

### 3. Knowledge Distillation

```python
import numpy as np

def distillation_loss(teacher_logits, student_logits, temperature=2.0):
    """
    Knowledge distillation loss

    Use soft labels (temperature T) to transfer knowledge
    """
    # Soften probability distributions
    teacher_probs = softmax(teacher_logits / temperature)
    student_probs = softmax(student_logits / temperature)

    # KL divergence
    # D_KL(teacher || student)
    mask = teacher_probs > 0
    kl = np.sum(teacher_probs[mask] * np.log(teacher_probs[mask] / student_probs[mask]))

    # Scale (because temperature affects entropy)
    return kl * (temperature ** 2)

# Example
np.random.seed(42)
teacher_logits = np.array([[2.0, 1.0, 0.1, 0.5, 0.3]])
student_logits = np.array([[1.5, 0.8, 0.2, 0.4, 0.2]])

print("Knowledge distillation example:")
print("="*50)
print(f"Temperature T=1.0: loss = {distillation_loss(teacher_logits, student_logits, 1.0):.4f}")
print(f"Temperature T=2.0: loss = {distillation_loss(teacher_logits, student_logits, 2.0):.4f}")
print(f"Temperature T=4.0: loss = {distillation_loss(teacher_logits, student_logits, 4.0):.4f}")
print("\nHigher temperature makes distribution smoother, transferring more 'dark knowledge'")
```

### 4. Statistical Foundation of Batch Normalization

```python
import numpy as np

# Application of law of large numbers in Batch Normalization
np.random.seed(42)

# True distribution parameters
true_mean = 5.0
true_var = 4.0

# Different batch sizes
batch_sizes = [8, 32, 128, 512, 2048]
n_trials = 1000

print("Batch Normalization statistics estimation:")
print("="*60)
print(f"{'Batch size':>10} | {'Mean error':>15} | {'Variance error':>15}")
print("-"*60)

for batch_size in batch_sizes:
    # Mean and variance estimates from multiple samples
    mean_estimates = []
    var_estimates = []

    for _ in range(n_trials):
        batch = np.random.normal(true_mean, np.sqrt(true_var), batch_size)
        mean_estimates.append(batch.mean())
        var_estimates.append(batch.var(ddof=1))  # Unbiased estimate

    mean_error = np.abs(np.mean(mean_estimates) - true_mean)
    var_error = np.abs(np.mean(var_estimates) - true_var)

    print(f"{batch_size:>10} | {mean_error:>15.6f} | {var_error:>15.6f}")

print("\nConclusion: Larger batches give more accurate statistics (Law of Large Numbers)")
```

### 5. Mathematical Foundation of Diffusion Models

Diffusion models are currently one of the most advanced generative models. Their mathematical foundation involves stochastic processes, stochastic differential equations, and score matching.

#### Forward Diffusion Process

**Definition**: Step-by-step addition of Gaussian noise to data until it becomes pure noise.

$$
\mathbf{x}_t = \sqrt{1-\beta_t}\mathbf{x}_{t-1} + \sqrt{\beta_t}\boldsymbol{\epsilon}_{t-1}, \quad \boldsymbol{\epsilon}_{t-1} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

Where $\beta_t$ is the noise schedule, typically $\beta_1 < \beta_2 < \cdots < \beta_T$.

**Reparameterization**: Let $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s$, then we can directly sample $\mathbf{x}_t$ from $\mathbf{x}_0$ at any timestep:

$$
\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

**Derivation**:

**Step 1**: Expand the first two steps.

$$\mathbf{x}_1 = \sqrt{\alpha_1}\mathbf{x}_0 + \sqrt{1-\alpha_1}\boldsymbol{\epsilon}_0$$

$$\mathbf{x}_2 = \sqrt{\alpha_2}\mathbf{x}_1 + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_1$$

$$= \sqrt{\alpha_2}\left(\sqrt{\alpha_1}\mathbf{x}_0 + \sqrt{1-\alpha_1}\boldsymbol{\epsilon}_0\right) + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_1$$

$$= \sqrt{\alpha_1\alpha_2}\mathbf{x}_0 + \sqrt{\alpha_2(1-\alpha_1)}\boldsymbol{\epsilon}_0 + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_1$$

**Step 2**: Combine independent Gaussian noises.

Weighted sum of two independent Gaussians $\mathcal{N}(0, \sigma_1^2)$ and $\mathcal{N}(0, \sigma_2^2)$:

$$\sigma_1\epsilon_1 + \sigma_2\epsilon_2 \sim \mathcal{N}(0, \sigma_1^2 + \sigma_2^2)$$

Therefore:
$$\sqrt{\alpha_2(1-\alpha_1)}\boldsymbol{\epsilon}_0 + \sqrt{1-\alpha_2}\boldsymbol{\epsilon}_1 \sim \mathcal{N}\left(\mathbf{0}, (1-\alpha_1)\alpha_2 + (1-\alpha_2)\mathbf{I}\right)$$

$$= \mathcal{N}\left(\mathbf{0}, (1 - \alpha_1\alpha_2)\mathbf{I}\right) = \mathcal{N}\left(\mathbf{0}, (1 - \bar{\alpha}_2)\mathbf{I}\right)$$

**Step 3**: Induce the general form.

$$\boxed{\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}$$

**Intuitive understanding**:
- When $t=0$, $\bar{\alpha}_0=1$, $\mathbf{x}_0$ is original data
- When $t=T$, $\bar{\alpha}_T\approx 0$, $\mathbf{x}_T\approx\boldsymbol{\epsilon}$ is pure noise

#### Reverse Diffusion Process

**Objective**: Learn reverse process $p_\theta(\mathbf{x}_{t-1}|\mathbf{x}_t)$ to reconstruct data from noise.

**Posterior distribution** (given $\mathbf{x}_0$):

$$
q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\tilde{\boldsymbol{\mu}}_t, \tilde{\beta}_t\mathbf{I})
$$

Where:

$$
\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t
$$

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t
$$

**Posterior mean formula derivation**:

Use Bayes' formula and Gaussian product property.

**Step 1**: Write the conditional distribution.

$$q(\mathbf{x}_{t-1}|\mathbf{x}_t, \mathbf{x}_0) \propto q(\mathbf{x}_t|\mathbf{x}_{t-1}, \mathbf{x}_0) \cdot q(\mathbf{x}_{t-1}|\mathbf{x}_0)$$

**Step 2**: Expand the two terms.

$$q(\mathbf{x}_t|\mathbf{x}_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}\mathbf{x}_{t-1}, \beta_t\mathbf{I})$$

$$q(\mathbf{x}_{t-1}|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0, (1-\bar{\alpha}_{t-1})\mathbf{I})$$

**Step 3**: Use Gaussian product formula.

Product of two Gaussians is still Gaussian, with mean and variance obtained by completing the square:

$$\boxed{\tilde{\boldsymbol{\mu}}_t = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t}\mathbf{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\mathbf{x}_t}$$

#### Training Objective: Score Matching

**Key insight**: Neural network predicts noise $\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)$.

**Loss function** (simplified version):

$$
\mathcal{L}_{simple} = \mathbb{E}_{t, \mathbf{x}_0, \boldsymbol{\epsilon}}\left[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2\right]
$$

Where $\mathbf{x}_t = \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1-\bar{\alpha}_t}\boldsymbol{\epsilon}$.

**Why predicting noise is equivalent to learning score function**:

Score function is defined as $\nabla_{\mathbf{x}}\log p(\mathbf{x})$.

**Step 1**: Write the distribution of $\mathbf{x}_t$.

$$p(\mathbf{x}_t|\mathbf{x}_0) = \mathcal{N}(\sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I})$$

$$\log p(\mathbf{x}_t|\mathbf{x}_0) = -\frac{1}{2(1-\bar{\alpha}_t)}\|\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0\|^2 + C$$

**Step 2**: Calculate the score.

$$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t|\mathbf{x}_0) = -\frac{\mathbf{x}_t - \sqrt{\bar{\alpha}_t}\mathbf{x}_0}{1-\bar{\alpha}_t} = -\frac{\boldsymbol{\epsilon}}{\sqrt{1-\bar{\alpha}_t}}$$

**Step 3**: Establish the equivalence.

$$\boxed{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) = -\sqrt{1-\bar{\alpha}_t} \cdot \nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t)}$$

Predicting noise = Predicting score function (with scaling)

```python
import numpy as np

class SimpleDiffusion:
    """Simplified Diffusion model implementation"""

    def __init__(self, timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.timesteps = timesteps

        # Linear noise schedule
        self.betas = np.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)

        # Precompute
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: sample x_t from x_0"""
        if noise is None:
            noise = np.random.randn(*x_0.shape)

        return (
            self.sqrt_alphas_cumprod[t] * x_0 +
            self.sqrt_one_minus_alphas_cumprod[t] * noise
        )

    def predict_start_from_noise(self, x_t, t, noise_pred):
        """Reconstruct x_0 from predicted noise"""
        return (
            x_t - self.sqrt_one_minus_alphas_cumprod[t] * noise_pred
        ) / self.sqrt_alphas_cumprod[t]

    def p_mean_variance(self, model_pred, x_t, t):
        """Calculate mean and variance of reverse process"""
        # Predict x_0
        x_0_pred = self.predict_start_from_noise(x_t, t, model_pred)

        # Calculate mean (using posterior formula)
        alpha_t = self.alphas[t]
        alpha_cumprod_t = self.alphas_cumprod[t]
        alpha_cumprod_prev = self.alphas_cumprod[t-1] if t > 0 else 1.0

        mean = (
            np.sqrt(alpha_cumprod_prev) * (1 - alpha_t) / (1 - alpha_cumprod_t) * x_0_pred +
            np.sqrt(alpha_t) * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * x_t
        )

        # Variance
        var = (1 - alpha_cumprod_prev) / (1 - alpha_cumprod_t) * self.betas[t]

        return mean, var

    def p_sample(self, model_pred, x_t, t):
        """Single-step reverse sampling"""
        mean, var = self.p_mean_variance(model_pred, x_t, t)

        if t == 0:
            return mean

        noise = np.random.randn(*x_t.shape)
        return mean + np.sqrt(var) * noise

    def training_loss(self, x_0, noise_pred, noise_true):
        """Calculate training loss"""
        return np.mean((noise_pred - noise_true) ** 2)

# Example
diffusion = SimpleDiffusion(timesteps=1000)

# Simulated data
x_0 = np.random.randn(32, 3, 32, 32)  # Batch of 32x32 RGB images
t = 500  # Intermediate timestep

# Forward diffusion
noise = np.random.randn(*x_0.shape)
x_t = diffusion.q_sample(x_0, t, noise)

# Assume model predicts (neural network in practice)
noise_pred = noise + np.random.randn(*x_0.shape) * 0.1  # Add noise to simulate imperfect prediction

# Calculate loss
loss = diffusion.training_loss(x_0, noise_pred, noise)
print(f"Training loss: {loss:.4f}")

# Reconstruct x_0
x_0_reconstructed = diffusion.predict_start_from_noise(x_t, t, noise_pred)
reconstruction_error = np.mean((x_0 - x_0_reconstructed) ** 2)
print(f"Reconstruction error: {reconstruction_error:.4f}")
```

#### Connections to Other Generative Models

| Model | Core Idea | Loss Function |
|------|---------|---------|
| VAE | Variational inference + reparameterization | $\mathcal{L} = \text{Recon} + \text{KL}$ |
| GAN | Adversarial training | Min-Max game |
| Diffusion | Stepwise denoising | $\mathbb{E}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta\|^2]$ |
| Flow | Invertible transformations | $\log p(\mathbf{x}) = \log p(\mathbf{z}) + \log\|\det \mathbf{J}\|$ |

### 6. Game Theory Foundation of GANs

Generative Adversarial Networks (GANs) have their mathematical foundation in the Min-Max Game from game theory.

#### Basic Framework

**Generator** $G$: Generates fake samples from noise $\mathbf{z}$

**Discriminator** $D$: Distinguishes real samples $\mathbf{x}$ from fake samples $G(\mathbf{z})$

**Objective function**:

$$
\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z}[\log(1 - D(G(\mathbf{z})))]
$$

#### Theoretical Analysis: Optimal Discriminator

**Theorem**: For a fixed generator $G$, the optimal discriminator is:

$$
D^*(\mathbf{x}) = \frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}
$$

**Proof**:

**Step 1**: Expand the objective function.

$$V(D, G) = \int_{\mathbf{x}} p_{data}(\mathbf{x})\log D(\mathbf{x})\,d\mathbf{x} + \int_{\mathbf{z}} p_z(\mathbf{z})\log(1-D(G(\mathbf{z})))\,d\mathbf{z}$$

**Step 2**: Variable substitution.

Let $\mathbf{x} = G(\mathbf{z})$, then:

$$\int_{\mathbf{z}} p_z(\mathbf{z})\log(1-D(G(\mathbf{z})))\,d\mathbf{z} = \int_{\mathbf{x}} p_g(\mathbf{x})\log(1-D(\mathbf{x}))\,d\mathbf{x}$$

**Step 3**: Combine the integrals.

$$V(D, G) = \int_{\mathbf{x}} \left[p_{data}(\mathbf{x})\log D(\mathbf{x}) + p_g(\mathbf{x})\log(1-D(\mathbf{x}))\right]\,d\mathbf{x}$$

**Step 4**: Maximize the integrand at each point $\mathbf{x}$.

$$f(D) = a\log D + b\log(1-D), \quad a = p_{data}(\mathbf{x}), \quad b = p_g(\mathbf{x})$$

Differentiate and set to zero:

$$\frac{df}{dD} = \frac{a}{D} - \frac{b}{1-D} = 0$$

$$a(1-D) = bD$$

$$D^* = \frac{a}{a+b} = \frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}$$

$$\boxed{D^*(\mathbf{x}) = \frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}}$$

#### Theoretical Analysis: Optimal Generator

**Theorem**: When $p_g = p_{data}$, the generator reaches optimality, where $V(D^*, G) = -\log 4$.

**Proof**:

**Step 1**: Substitute the optimal discriminator into the objective function.

$$V(D^*, G) = \mathbb{E}_{\mathbf{x} \sim p_{data}}\left[\log \frac{p_{data}(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}\right] + \mathbb{E}_{\mathbf{x} \sim p_g}\left[\log \frac{p_g(\mathbf{x})}{p_{data}(\mathbf{x}) + p_g(\mathbf{x})}\right]$$

**Step 2**: Relate to KL divergence.

Note that:

$$V(D^*, G) = -\log 4 + D_{KL}(p_{data}\|p_{data}+p_g) + D_{KL}(p_g\|p_{data}+p_g)$$

**Step 3**: Introduce Jensen-Shannon divergence (JSD).

$$JSD(P\|Q) = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M), \quad M = \frac{P+Q}{2}$$

Therefore:

$$V(D^*, G) = -\log 4 + 2 \cdot JSD(p_{data}\|p_g)$$

**Step 4**: Optimal case.

JSD is non-negative, and equals 0 when $p_g = p_{data}$:

$$\boxed{\min_G V(D^*, G) = -\log 4 \approx -1.386}$$

At this point, $D^*(\mathbf{x}) = 0.5$ for all $\mathbf{x}$.

```python
import numpy as np

class SimpleGAN:
    """Simplified GAN implementation (linear generator and discriminator)"""

    def __init__(self, data_dim, noise_dim):
        self.data_dim = data_dim
        self.noise_dim = noise_dim

        # Initialize parameters
        self.G_W = np.random.randn(noise_dim, data_dim) * 0.01
        self.G_b = np.zeros(data_dim)
        self.D_W = np.random.randn(data_dim, 1) * 0.01
        self.D_b = np.zeros(1)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def generate(self, z):
        """Generator forward propagation"""
        return z @ self.G_W + self.G_b

    def discriminate(self, x):
        """Discriminator forward propagation"""
        return self.sigmoid(x @ self.D_W + self.D_b)

    def train_discriminator(self, real_data, fake_data, lr=0.01):
        """Train discriminator"""
        batch_size = real_data.shape[0]

        # Forward propagation
        d_real = self.discriminate(real_data)
        d_fake = self.discriminate(fake_data)

        # Discriminator loss: -[log(D(x)) + log(1-D(G(z)))]
        d_loss = -(np.log(d_real + 1e-8).mean() + np.log(1 - d_fake + 1e-8).mean())

        # Backward propagation (simplified)
        d_real_grad = (d_real - 1) / batch_size
        d_fake_grad = d_fake / batch_size

        # Update discriminator
        dW = real_data.T @ d_real_grad + fake_data.T @ d_fake_grad
        db = d_real_grad.sum() + d_fake_grad.sum()

        self.D_W -= lr * dW
        self.D_b -= lr * db

        return d_loss

    def train_generator(self, fake_data, lr=0.01):
        """Train generator"""
        batch_size = fake_data.shape[0]

        # Forward propagation
        d_fake = self.discriminate(fake_data)

        # Generator loss: -log(D(G(z))) (non-saturated version)
        g_loss = -np.log(d_fake + 1e-8).mean()

        # Backward propagation
        d_fake_grad = (d_fake - 1) / batch_size

        # Backpropagate through discriminator to generator
        dW = d_fake_grad @ self.D_W.T
        db = d_fake_grad.sum(axis=0)

        # Update generator
        gW_grad = np.random.randn(self.noise_dim, batch_size) @ dW / batch_size
        gb_grad = db

        self.G_W -= lr * gW_grad
        self.G_b -= lr * gb_grad

        return g_loss

# Demonstration
np.random.seed(42)
data_dim = 2
noise_dim = 10

gan = SimpleGAN(data_dim, noise_dim)

# Real data distribution: 2D Gaussian
real_mean = np.array([1, 1])
real_cov = np.array([[0.5, 0.2], [0.2, 0.5]])

print("GAN training demonstration:")
print("="*50)

for epoch in range(1000):
    # Sample real data
    real_data = np.random.multivariate_normal(real_mean, real_cov, 32)

    # Sample noise and generate fake data
    z = np.random.randn(32, noise_dim)
    fake_data = gan.generate(z)

    # Train discriminator
    d_loss = gan.train_discriminator(real_data, fake_data)

    # Regenerate fake data (because discriminator updated)
    z = np.random.randn(32, noise_dim)
    fake_data = gan.generate(z)

    # Train generator
    g_loss = gan.train_generator(fake_data)

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: D_loss={d_loss:.4f}, G_loss={g_loss:.4f}")

print("\nTheoretical optimal loss: -log(4) ≈ -1.386 (corresponding to D_loss ≈ 0.693)")
```

#### Challenges in GAN Training

| Problem | Cause | Solution |
|------|------|---------|
| Mode collapse | Generator only produces few modes | Minibatch discrimination, WGAN |
| Training instability | Discriminator too strong | Progressive training, label smoothing |
| Vanishing gradients | Discriminator too accurate | Non-saturated loss, WGAN-GP |

#### Wasserstein GAN (WGAN)

Use Wasserstein distance to replace JS divergence:

$$
W(P_r, P_g) = \inf_{\gamma \in \Pi(P_r, P_g)} \mathbb{E}_{(x,y)\sim\gamma}[\|x-y\|]
$$

**Kantorovich-Rubinstein duality**:

$$
W(P_r, P_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x\sim P_r}[f(x)] - \mathbb{E}_{x\sim P_g}[f(x)]
$$

Where $\|f\|_L \leq 1$ means $f$ is a 1-Lipschitz function.

**WGAN loss**:

$$
\min_G \max_{D \in \mathcal{D}} \mathbb{E}_{x\sim P_r}[D(x)] - \mathbb{E}_{z\sim p_z}[D(G(z))]
$$

WGAN provides more meaningful gradients and more stable training.

---

## Summary

This chapter introduced the two pillars of probability theory—limit theorems and information theory, which have wide applications in deep learning.

### Limit Theorems Summary

| Theorem | Content | Applications |
|------|------|------|
| Law of large numbers | $\bar{X}_n \xrightarrow{P} \mu$ | Batch statistics estimation, SGD convergence |
| Central limit theorem | $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} N(0, \sigma^2)$ | Confidence intervals, weight initialization |

### Information Theory Summary

| Concept | Formula | Meaning | Applications |
|------|------|------|------|
| Entropy | $H(X) = -\sum p \log p$ | Uncertainty | Decision trees |
| Cross entropy | $H(P,Q) = -\sum p \log q$ | Coding efficiency | Classification loss |
| KL divergence | $D_{KL} = \sum p \log(p/q)$ | Distribution distance | VAE regularization |
| Mutual information | $I(X;Y) = H(X) - H(X\|Y)$ | Mutual dependence | Feature selection |

### Key Points

1. **Law of large numbers**: When sample size is sufficiently large, sample statistics converge to population parameters
2. **Central limit theorem**: Sample mean approximates normal distribution
3. **Entropy**: Measures uncertainty, maximum when equally probable
4. **Cross entropy**: Most commonly used loss function in machine learning
5. **KL divergence**: Measures distribution difference, asymmetric
6. **Mutual information**: Measures dependence between variables

### Core Formulas in Deep Learning

Cross-entropy loss: $L = -log p(y_true)$

VAE loss:   $L = Reconstruction + β·KL(q(z|x) \|\| p(z))$

Knowledge distillation:   $L = KL(teacher || student) · T²$

Information gain:   $IG = H(Y) - H(Y\|X) = I(X; Y)$

---

**Previous section**: [Chapter 3 (c): Multivariate Random Variables and Numerical Characteristics](03c-multivariate-random-variables_EN.md)

**Next chapter**: [Chapter 4: Mathematical Statistics](04-statistics_EN.md) - Learn about statistical inference, parameter estimation, hypothesis testing, and other concepts.

**Return**: [Mathematics Fundamentals Tutorial Table of Contents](../math-fundamentals.md)
