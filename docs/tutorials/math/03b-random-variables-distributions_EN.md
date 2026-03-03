# Chapter 3 (b): Random Variables and Common Distributions

Random variables are a core concept in probability theory. They quantify the outcomes of random experiments, allowing us to use mathematical tools to analyze random phenomena. This chapter will provide an in-depth explanation of discrete and continuous random variables, as well as various probability distributions widely used in deep learning.

---

## 🎯 Life Analogy: Converting Uncertain Events to Numbers

Random variables are about **converting random events into numbers**:

| Random Event | Original Outcome | Random Variable (Numerical) |
|--------------|------------------|----------------------------|
| Coin flip | Heads/Tails | 1/0 |
| Weather | Sunny/Cloudy/Rainy | 0/1/2 |
| Exam | Pass/Fail | 1/0 |
| Tomorrow's temp | 20-30°C | 25.3 (specific value) |

**The essence of random variables**: Use numbers to describe uncertain outcomes, so we can analyze them mathematically!

### 📖 Why Do We Need Random Variables?

- **Original problem**: "Will it rain tomorrow?" → Hard to analyze mathematically
- **As a random variable**: "Let X=1 mean rain, X=0 mean no rain" → Can compute probability, expected value, variance

**Converting "word descriptions" into "number games"**.

### Common Distributions = Common Patterns

| Distribution | Real-life Example | Shape |
|--------------|-------------------|-------|
| **Bernoulli** | Coin flip (yes/no) | Two outcomes |
| **Binomial** | Number of heads in 10 flips | Bell-ish |
| **Normal** | Heights, test scores | Bell curve |
| **Uniform** | Rolling a fair die | Flat |

### 📖 Plain English Translation

| Math Term | Plain English |
|-----------|---------------|
| Random variable | A number that depends on chance |
| PMF (Probability Mass Function) | Probability of each discrete value |
| PDF (Probability Density Function) | Curve showing where values cluster |
| Expected value $\mathbb{E}[X]$ | Average if you repeated forever |
| Variance $\text{Var}(X)$ | How spread out the values are |

---

## Table of Contents

1. [Definition of Random Variables](#definition-of-random-variables)
2. [Discrete Random Variables](#discrete-random-variables)
3. [Continuous Random Variables](#continuous-random-variables)
4. [Cumulative Distribution Function](#cumulative-distribution-function)
5. [Discrete Probability Distributions](#discrete-probability-distributions)
    - [Bernoulli Distribution](#bernoulli-distribution)
    - [Binomial Distribution](#binomial-distribution)
    - [Poisson Distribution](#poisson-distribution)
    - [Categorical Distribution](#categorical-distribution)
6. [Continuous Probability Distributions](#continuous-probability-distributions)
    - [Uniform Distribution](#uniform-distribution)
    - [Normal Distribution](#normal-distributiongaussian)
    - [Exponential Distribution](#exponential-distribution)
    - [Laplace Distribution](#laplace-distribution)
    - [Beta Distribution](#beta-distribution)
    - [Gamma Distribution](#gamma-distribution)
7. [Relationships Between Distributions](#relationships-between-distributions)
8. [Applications in Deep Learning](#applications-in-deep-learning)
9. [Summary](#summary)

---

## Definition of Random Variables

### Basic Definition

A **random variable** is a real-valued function defined on the sample space $\Omega$:

$$
X: \Omega \to \mathbb{R}
$$

That is, for each sample point $\omega \in \Omega$, there is a unique real number $X(\omega)$ associated with it.

### Intuitive Understanding

Random variables are rules that map the **outcomes of random phenomena** to **numerical values**:

| Random experiment | Sample space $\Omega$ | Random variable $X$ |
|----------|------------------|--------------|
| Rolling a die | $\{1,2,3,4,5,6\}$ | The point value itself |
| Flipping a coin | $\{Heads, Tails\}$ | Heads=1, Tails=0 |
| Measuring height | $(0, \infty)$ | Height value (cm) |
| Image classification | All images | Class label (0-9) |

### Types of Random Variables

| Type | Characteristic of values | Example |
|------|----------|------|
| **Discrete** | Finite or countably infinite values | Die roll, word count |
| **Continuous** | Values within continuous intervals | Height, temperature, weight |

```python
import numpy as np
import matplotlib.pyplot as plt

# Discrete random variable example: rolling a die
def dice_random_variable(outcome):
    """Map die roll outcome to a numerical value"""
    return outcome

# Continuous random variable example: measurement error
def measurement_error():
    """Measurement error follows normal distribution"""
    return np.random.normal(0, 1)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Discrete
outcomes = [1, 2, 3, 4, 5, 6]
axes[0].bar(outcomes, [1/6]*6, alpha=0.7)
axes[0].set_title('Discrete Random Variable: Rolling a Die')
axes[0].set_xlabel('X (points)')
axes[0].set_ylabel('P(X)')

# Continuous
x = np.linspace(-4, 4, 100)
pdf = (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)
axes[1].plot(x, pdf, 'b-', linewidth=2)
axes[1].fill_between(x, pdf, alpha=0.3)
axes[1].set_title('Continuous Random Variable: Measurement Error')
axes[1].set_xlabel('X (error)')
axes[1].set_ylabel('f(x)')

plt.tight_layout()
plt.savefig('random_variables.png', dpi=100)
print("Image saved")
```

---

## Discrete Random Variables

### Definition

The values of a discrete random variable are **finite** or **countably infinite** values $x_1, x_2, x_3, \ldots$

### Probability Mass Function (PMF)

The **probability mass function** describes the probability that a discrete random variable takes each value:

$$
p(x) = P(X = x)
$$

**Properties**:
1. **Non-negativity**: $p(x) \geq 0$ for all $x$
2. **Normalization**: $\displaystyle\sum_x p(x) = 1$
3. **Probability calculation**: $P(X \in A) = \displaystyle\sum_{x \in A} p(x)$

### Expectation (Mean)

Expectation of a discrete random variable:

$$
\mathbb{E}[X] = \sum_x x \cdot p(x)
$$

**Properties of expectation**:
- **Linearity**: $\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]$
- **Constant**: $\mathbb{E}[c] = c$
- **Function expectation**: $\mathbb{E}[g(X)] = \sum_x g(x) \cdot p(x)$

### Variance

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

**Properties of variance**:
- $\text{Var}(X) \geq 0$
- $\text{Var}(c) = 0$
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$

```python
import numpy as np

# Discrete random variable example: unfair die
# PMF: P(1)=0.1, P(2)=0.1, P(3)=0.2, P(4)=0.2, P(5)=0.2, P(6)=0.2

x_values = np.array([1, 2, 3, 4, 5, 6])
pmf = np.array([0.1, 0.1, 0.2, 0.2, 0.2, 0.2])

# Verify normalization
print(f"Sum of probabilities: {pmf.sum():.2f}")

# Calculate expectation
expected_value = np.sum(x_values * pmf)
print(f"Expectation E[X] = {expected_value:.2f}")

# Calculate variance
expected_x_squared = np.sum(x_values**2 * pmf)
variance = expected_x_squared - expected_value**2
print(f"Variance Var(X) = {variance:.2f}")

# Standard deviation
std_dev = np.sqrt(variance)
print(f"Standard deviation σ = {std_dev:.2f}")

# Simulation verification
n_samples = 100000
samples = np.random.choice(x_values, size=n_samples, p=pmf)
print(f"\nSimulation results:")
print(f"Sample mean: {samples.mean():.4f}")
print(f"Sample variance: {samples.var():.4f}")
```

---

## Continuous Random Variables

### Definition

The values of a continuous random variable fill one or more **continuous intervals**.

### Probability Density Function (PDF)

The **probability density function** $f(x)$ describes the distribution of a continuous random variable:

$$
P(a \leq X \leq b) = \int_a^b f(x) \, dx
$$

**Properties**:
1. **Non-negativity**: $f(x) \geq 0$ for all $x$
2. **Normalization**: $\displaystyle\int_{-\infty}^{+\infty} f(x) \, dx = 1$
3. **Zero probability for single points**: $P(X = x) = 0$ (for any single point $x$)

### ⚠️ Important Note

$f(x)$ **is not a probability itself**! It's "probability density" and can be greater than 1.

- For continuous variables, probability is the **area under the PDF curve**
- $f(x) = 2$ on $[0, 0.5]$ is valid (integral equals 1)

### Expectation and Variance

**Expectation**:

$$
\mathbb{E}[X] = \int_{-\infty}^{+\infty} x \cdot f(x) \, dx
$$

**Variance**:

$$
\text{Var}(X) = \int_{-\infty}^{+\infty} (x - \mu)^2 f(x) \, dx = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

```python
import numpy as np
from scipy import integrate

# Continuous random variable example: custom PDF
def custom_pdf(x):
    """Triangular distribution: f(x) = 2x for x in [0, 1]"""
    if 0 <= x <= 1:
        return 2 * x
    return 0

# Vectorize
custom_pdf_vec = np.vectorize(custom_pdf)

# Verify normalization
integral, _ = integrate.quad(custom_pdf, -np.inf, np.inf)
print(f"PDF integral = {integral:.4f}")

# Calculate expectation
def x_times_pdf(x):
    return x * custom_pdf(x)

expected, _ = integrate.quad(x_times_pdf, 0, 1)
print(f"Expectation E[X] = {expected:.4f}")

# Calculate variance
def x_squared_times_pdf(x):
    return x**2 * custom_pdf(x)

expected_x2, _ = integrate.quad(x_squared_times_pdf, 0, 1)
variance = expected_x2 - expected**2
print(f"Variance Var(X) = {variance:.4f}")

# Visualization
x = np.linspace(-0.2, 1.2, 200)
pdf_values = custom_pdf_vec(x)

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 5))
plt.plot(x, pdf_values, 'b-', linewidth=2, label='PDF: f(x) = 2x')
plt.fill_between(x[x >= 0], pdf_values[x >= 0], alpha=0.3)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('PDF of a Continuous Random Variable')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('continuous_pdf.png', dpi=100)
print("\nPDF image saved")
```

---

## Cumulative Distribution Function

### Definition

The **cumulative distribution function** (CDF) describes the probability that a random variable is less than or equal to a given value:

$$
F(x) = P(X \leq x)
$$

### Discrete Case

$$
F(x) = \sum_{x_i \leq x} p(x_i)
$$

### Continuous Case

$$
F(x) = \int_{-\infty}^x f(t) \, dt
$$

### Properties of CDF

1. **Range**: $0 \leq F(x) \leq 1$
2. **Boundaries**: $\lim_{x \to -\infty} F(x) = 0$, $\lim_{x \to +\infty} F(x) = 1$
3. **Monotonicity**: Monotonically non-decreasing ($x_1 < x_2 \Rightarrow F(x_1) \leq F(x_2)$)
4. **Right-continuous**: $\lim_{t \to x^+} F(t) = F(x)$

### Relationship Between PDF and CDF

$$
F(x) = \int_{-\infty}^x f(t) \, dt \quad \Longleftrightarrow \quad f(x) = F'(x) = \frac{dF}{dx}
$$

### Probability Calculation

For continuous random variables:

$$
P(a < X \leq b) = F(b) - F(a)
$$

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# CDF of standard normal distribution
x = np.linspace(-4, 4, 200)
pdf = stats.norm.pdf(x)  # Probability density function
cdf = stats.norm.cdf(x)  # Cumulative distribution function

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# PDF
axes[0].plot(x, pdf, 'b-', linewidth=2)
axes[0].fill_between(x, pdf, alpha=0.3)
axes[0].set_title('PDF: f(x)')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].grid(True, alpha=0.3)

# CDF
axes[1].plot(x, cdf, 'r-', linewidth=2)
axes[1].set_title('CDF: F(x) = P(X ≤ x)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('F(x)')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-0.05, 1.05)

# Relationship between CDF and PDF
axes[2].plot(x, pdf, 'b-', linewidth=2, label='PDF')
x_shade = x[(x >= -1) & (x <= 1)]
axes[2].fill_between(x_shade, stats.norm.pdf(x_shade), alpha=0.3, color='blue')
axes[2].axvline(x=-1, color='r', linestyle='--', label=f'F(-1)={stats.norm.cdf(-1):.3f}')
axes[2].axvline(x=1, color='g', linestyle='--', label=f'F(1)={stats.norm.cdf(1):.3f}')
axes[2].set_title('P(-1 ≤ X ≤ 1) = F(1) - F(-1)')
axes[2].legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pdf_cdf.png', dpi=100)
print("PDF and CDF images saved")

# Calculate probabilities using CDF
print(f"\nCalculate probabilities using CDF:")
print(f"P(X ≤ 0) = F(0) = {stats.norm.cdf(0):.4f}")
print(f"P(X > 1) = 1 - F(1) = {1 - stats.norm.cdf(1):.4f}")
print(f"P(-1 < X ≤ 1) = F(1) - F(-1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f}")
```

---

## Discrete Probability Distributions

### Bernoulli Distribution

#### Definition

The **Bernoulli distribution** describes the result of a single binary trial (success/failure).

$$
X \sim \text{Bernoulli}(p)
$$

#### PMF

$$
P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}
$$

Or equivalently:
- $P(X = 1) = p$ (success)
- $P(X = 0) = 1 - p$ (failure)

#### Expectation and Variance

$$
\mathbb{E}[X] = p
$$

$$
\text{Var}(X) = p(1 - p)
$$

#### Deep Learning Applications

- **Binary classification**: Sigmoid + threshold in output layer
- **Dropout**: Bernoulli sampling of neuron retention

```python
import numpy as np
from scipy import stats

# Bernoulli distribution
p = 0.7  # Success probability

# PMF
print("Bernoulli distribution PMF:")
print(f"P(X=0) = {1 - p:.2f}")
print(f"P(X=1) = {p:.2f}")

# Using scipy
bernoulli = stats.bernoulli(p)
print(f"\nTheoretical mean: {bernoulli.mean():.2f}")
print(f"Theoretical variance: {bernoulli.var():.4f}")

# Simulation
samples = bernoulli.rvs(size=10000)
print(f"\nSimulated mean: {samples.mean():.4f}")
print(f"Simulated variance: {samples.var():.4f}")

# Deep learning example: Dropout mask
def bernoulli_dropout(shape, p=0.5):
    """Generate Dropout mask"""
    return (np.random.random(shape) < p).astype(float) / p

mask = bernoulli_dropout((5, 5), p=0.5)
print(f"\nDropout mask (p=0.5):")
print(mask)
```

---

### Binomial Distribution

#### Definition

The **binomial distribution** describes the number of successes in $n$ independent Bernoulli trials.

$$
X \sim \text{Binomial}(n, p)
$$

#### PMF

$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, 2, \ldots, n
$$

Where $\displaystyle\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient.

#### Expectation and Variance

$$
\mathbb{E}[X] = np
$$

$$
\text{Var}(X) = np(1-p)
$$

#### Relationship with Bernoulli Distribution

If $X_1, X_2, \ldots, X_n \stackrel{\text{iid}}{\sim} \text{Bernoulli}(p)$, then:

$$
\sum_{i=1}^n X_i \sim \text{Binomial}(n, p)
$$

#### Deep Learning Applications

- **Ensemble learning**: Number of successes from multiple independent predictions
- **Data augmentation**: Success/failure count of random transformations

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Binomial distribution parameters
n, p = 20, 0.3

# PMF
k_values = np.arange(0, n + 1)
pmf = stats.binom.pmf(k_values, n, p)

print(f"Binomial distribution B({n}, {p}):")
print(f"Theoretical mean: {n * p:.2f}")
print(f"Theoretical variance: {n * p * (1-p):.2f}")

# P(X = 6)
print(f"\nP(X = 6) = {stats.binom.pmf(6, n, p):.4f}")
# P(X <= 10)
print(f"P(X ≤ 10) = {stats.binom.cdf(10, n, p):.4f}")

# Visualize PMF
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(k_values, pmf, alpha=0.7, color='steelblue')
axes[0].axvline(x=n*p, color='r', linestyle='--', label=f'Mean = {n*p:.1f}')
axes[0].set_xlabel('k (number of successes)')
axes[0].set_ylabel('P(X = k)')
axes[0].set_title(f'Binomial PMF: B({n}, {p})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Comparison of different parameters
for p_val in [0.2, 0.5, 0.8]:
    axes[1].plot(k_values, stats.binom.pmf(k_values, n, p_val),
                 'o-', label=f'p = {p_val}', alpha=0.7)
axes[1].set_xlabel('k')
axes[1].set_ylabel('P(X = k)')
axes[1].set_title(f'Binomial distributions with different p values (n={n})')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('binomial.png', dpi=100)
print("\nBinomial distribution image saved")

# Simulation
samples = np.random.binomial(n, p, size=10000)
print(f"\nSimulation results:")
print(f"Sample mean: {samples.mean():.4f}")
print(f"Sample variance: {samples.var():.4f}")
```

---

### Poisson Distribution

#### Definition

The **Poisson distribution** describes the number of **rare events** occurring per unit time/space.

$$
X \sim \text{Poisson}(\lambda)
$$

#### PMF

$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots
$$

#### Expectation and Variance

$$
\mathbb{E}[X] = \text{Var}(X) = \lambda
$$

**Characteristic**: Expectation equals variance!

#### Poisson's Theorem

When $n \to \infty$, $p \to 0$, $np = \lambda$ (constant):

$$
\text{Binomial}(n, p) \approx \text{Poisson}(\lambda)
$$

#### Deep Learning Applications

- **Sparse coding**: Modeling sparsity of neuron activations
- **Count data**: Word frequencies in text, click counts in recommendation systems

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Poisson distribution
lambdas = [1, 4, 10]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# PMF comparison
k = np.arange(0, 25)
for lam in lambdas:
    pmf = stats.poisson.pmf(k, lam)
    axes[0].plot(k, pmf, 'o-', label=f'λ = {lam}', alpha=0.7)

axes[0].set_xlabel('k (number of events)')
axes[0].set_ylabel('P(X = k)')
axes[0].set_title('Poisson Distribution PMF')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Binomial approximating Poisson
n, p = 1000, 0.01
lambda_approx = n * p  # = 10

k = np.arange(0, 25)
binom_pmf = stats.binom.pmf(k, n, p)
poisson_pmf = stats.poisson.pmf(k, lambda_approx)

axes[1].plot(k, binom_pmf, 'o-', label=f'Binom({n}, {p})', alpha=0.7)
axes[1].plot(k, poisson_pmf, 's--', label=f'Poisson({lambda_approx})', alpha=0.7)
axes[1].set_xlabel('k')
axes[1].set_ylabel('P(X = k)')
axes[1].set_title('Binomial → Poisson Approximation')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('poisson.png', dpi=100)
print("Poisson distribution image saved")

# Application example: Sparse neuron activation
print("\nSparse neuron activation simulation:")
lam = 3  # Average activation of 3 neurons
n_neurons = 100
n_samples = 10

activations = np.random.poisson(lam, (n_samples, n_neurons))
print(f"Activation matrix shape: {activations.shape}")
print(f"Average number of activations: {activations.sum(axis=1).mean():.2f}")
print(f"Activation sparsity: {(activations == 0).mean():.2%}")
```

---

### Categorical Distribution

#### Definition

The **categorical distribution** (also called Multinoulli) describes the probability of choosing one from $K$ categories.

$$
X \sim \text{Categorical}(p_1, p_2, \ldots, p_K)
$$

#### PMF

$$
P(X = i) = p_i, \quad \sum_{i=1}^K p_i = 1
$$

#### Representation

Usually represented using **one-hot encoding**:

$$
\mathbf{x} \in \{0, 1\}^K, \quad \|\mathbf{x}\|_1 = 1
$$

#### Relationship with Softmax

Softmax output is the probability vector of categorical distribution:

$$
P(y = i | \mathbf{z}) = \text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}
$$

#### Deep Learning Applications

- **Multi-class classification**: Image classification, text classification
- **Language models**: Next word prediction (vocabulary size = K)

```python
import numpy as np

# Categorical distribution
probabilities = np.array([0.1, 0.2, 0.3, 0.15, 0.25])  # 5 categories
K = len(probabilities)

print(f"Number of categories: {K}")
print(f"Probability distribution: {probabilities}")
print(f"Sum of probabilities: {probabilities.sum():.2f}")

# Sampling
n_samples = 1000
samples = np.random.choice(K, size=n_samples, p=probabilities)

# Count frequencies
counts = np.bincount(samples, minlength=K)
print(f"\nSampling frequencies: {counts / n_samples}")

# One-hot encoding
def one_hot(indices, K):
    """Convert indices to one-hot encoding"""
    n = len(indices)
    one_hot_matrix = np.zeros((n, K))
    one_hot_matrix[np.arange(n), indices] = 1
    return one_hot_matrix

one_hot_samples = one_hot(samples[:5], K)
print(f"\nOne-hot encoding of first 5 samples:")
print(one_hot_samples)

# Softmax generates categorical distribution
def softmax(logits):
    """Convert logits to probability distribution"""
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / exp_logits.sum()

logits = np.array([1.0, 2.0, 3.0, 0.5, 1.5])
probs = softmax(logits)
print(f"\nLogits: {logits}")
print(f"Softmax probabilities: {probs}")
print(f"Sum of probabilities: {probs.sum():.6f}")
```

---

## Continuous Probability Distributions

### Uniform Distribution

#### PDF

$$
f(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b] \\ 0 & \text{otherwise} \end{cases}
$$

Denoted as $X \sim \text{Uniform}(a, b)$ or $X \sim U(a, b)$.

#### Expectation and Variance

$$
\mathbb{E}[X] = \frac{a + b}{2}
$$

$$
\text{Var}(X) = \frac{(b - a)^2}{12}
$$

#### Deep Learning Applications

- **Parameter initialization**: Xavier initialization uses uniform distribution
- **Data augmentation**: Random crop positions
- **Hyperparameter search**: Uniform sampling of search space

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Uniform distribution
a, b = 0, 10

# PDF and CDF
x = np.linspace(-2, 12, 200)
pdf = stats.uniform.pdf(x, loc=a, scale=b-a)
cdf = stats.uniform.cdf(x, loc=a, scale=b-a)

print(f"Uniform distribution U({a}, {b}):")
print(f"Mean: {(a + b) / 2:.2f}")
print(f"Variance: {(b - a)**2 / 12:.2f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, pdf, 'b-', linewidth=2)
axes[0].fill_between(x[(x >= a) & (x <= b)], pdf[(x >= a) & (x <= b)], alpha=0.3)
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title(f'Uniform Distribution PDF: U({a}, {b})')
axes[0].grid(True, alpha=0.3)

axes[1].plot(x, cdf, 'r-', linewidth=2)
axes[1].set_xlabel('x')
axes[1].set_ylabel('F(x)')
axes[1].set_title(f'Uniform Distribution CDF: U({a}, {b})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('uniform.png', dpi=100)
print("\nUniform distribution image saved")

# Deep learning application: Xavier initialization
def xavier_uniform_init(fan_in, fan_out):
    """Xavier uniform initialization"""
    bound = np.sqrt(6 / (fan_in + fan_out))
    return np.random.uniform(-bound, bound, (fan_out, fan_in))

# Example: Initialize Linear layer weights
fan_in, fan_out = 784, 256
weights = xavier_uniform_init(fan_in, fan_out)
print(f"\nXavier initialization weights:")
print(f"Shape: {weights.shape}")
print(f"Theoretical bound: ±{np.sqrt(6 / (fan_in + fan_out)):.4f}")
print(f"Actual range: [{weights.min():.4f}, {weights.max():.4f}]")
print(f"Actual mean: {weights.mean():.4f}")
print(f"Actual variance: {weights.var():.6f}")
```

---

### Normal Distribution

#### PDF

$$
f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

Denoted as $X \sim \mathcal{N}(\mu, \sigma^2)$.

#### Expectation and Variance

$$
\mathbb{E}[X] = \mu, \quad \text{Var}(X) = \sigma^2
$$

#### Standard Normal Distribution

When $\mu = 0$, $\sigma = 1$, it's called the **standard normal distribution**, denoted as $Z \sim \mathcal{N}(0, 1)$:

$$
\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}
$$

#### Standardization

Any normal distribution can be standardized:

$$
Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)
$$

#### Empirical Rule (68-95-99.7 Rule)

- $P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 68.27\%$
- $P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 95.45\%$
- $P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 99.73\%$

#### Deep Learning Applications

- **Weight initialization**: He initialization, truncated normal
- **Noise injection**: Data augmentation, variational inference
- **VAE**: Latent space distribution

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Normal distribution
mu, sigma = 0, 1
x = np.linspace(-4, 4, 200)
pdf = stats.norm.pdf(x, mu, sigma)
cdf = stats.norm.cdf(x, mu, sigma)

print(f"Normal distribution N({mu}, {sigma**2}):")
print(f"68.27% interval: [{mu-sigma:.2f}, {mu+sigma:.2f}]")
print(f"95.45% interval: [{mu-2*sigma:.2f}, {mu+2*sigma:.2f}]")
print(f"99.73% interval: [{mu-3*sigma:.2f}, {mu+3*sigma:.2f}]")

# Verification
print(f"\nVerification:")
print(f"P(-1 ≤ X ≤ 1) = {stats.norm.cdf(1) - stats.norm.cdf(-1):.4f}")
print(f"P(-2 ≤ X ≤ 2) = {stats.norm.cdf(2) - stats.norm.cdf(-2):.4f}")
print(f"P(-3 ≤ X ≤ 3) = {stats.norm.cdf(3) - stats.norm.cdf(-3):.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Normal distributions with different parameters
for mu_val, sigma_val in [(0, 1), (0, 2), (2, 1)]:
    pdf = stats.norm.pdf(x, mu_val, sigma_val)
    label = f'μ={mu_val}, σ={sigma_val}'
    axes[0].plot(x, pdf, label=label, linewidth=2)

axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('Normal Distribution PDF')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 68-95-99.7 rule visualization
x_fill = np.linspace(-3, 3, 100)
pdf_standard = stats.norm.pdf(x_fill, 0, 1)

axes[1].plot(x_fill, pdf_standard, 'b-', linewidth=2)
axes[1].fill_between(x_fill[(x_fill >= -1) & (x_fill <= 1)],
                     pdf_standard[(x_fill >= -1) & (x_fill <= 1)],
                     alpha=0.3, label='68.27%')
axes[1].fill_between(x_fill[(x_fill >= -2) & (x_fill <= 2)],
                     pdf_standard[(x_fill >= -2) & (x_fill <= 2)],
                     alpha=0.2, label='95.45%')
axes[1].fill_between(x_fill, pdf_standard, alpha=0.1, label='99.73%')
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].set_title('68-95-99.7 Rule')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('normal.png', dpi=100)
print("\nNormal distribution image saved")

# He initialization
def he_normal_init(fan_in, fan_out):
    """He normal initialization (for ReLU activation)"""
    std = np.sqrt(2 / fan_in)
    return np.random.normal(0, std, (fan_out, fan_in))

weights = he_normal_init(784, 256)
print(f"\nHe initialization weights:")
print(f"Shape: {weights.shape}")
print(f"Theoretical standard deviation: {np.sqrt(2/784):.4f}")
print(f"Actual standard deviation: {weights.std():.4f}")
```

---

### Exponential Distribution

#### PDF

$$
f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

Denoted as $X \sim \text{Exp}(\lambda)$.

#### Expectation and Variance

$$
\mathbb{E}[X] = \frac{1}{\lambda}
$$

$$
\text{Var}(X) = \frac{1}{\lambda^2}
$$

#### Memoryless Property (Key Property)

$$
P(X > s + t | X > s) = P(X > t)
$$

"The probability of waiting another t time units, given that we've already waited s time units, is the same as waiting t time units from the start."

#### Relationship with Poisson Distribution

If events arrive at rate $\lambda$ according to Poisson, then the **waiting time** follows $\text{Exp}(\lambda)$.

#### Deep Learning Applications

- **Dropout timing**: Intervals of random deactivation
- **Regularization**: L2 regularization can be viewed as exponential prior on parameters

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Exponential distribution
lambdas = [0.5, 1, 2]
x = np.linspace(0, 5, 200)

plt.figure(figsize=(10, 4))

for lam in lambdas:
    pdf = stats.expon.pdf(x, scale=1/lam)  # scipy uses β=1/λ as scale
    plt.plot(x, pdf, label=f'λ = {lam} (E[X] = {1/lam:.2f})', linewidth=2)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Exponential Distribution PDF')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('exponential.png', dpi=100)
print("Exponential distribution image saved")

# Verify memoryless property
lam = 1
print(f"\nVerify memoryless property (λ = {lam}):")
s, t = 2, 1

# P(X > s + t | X > s) = P(X > s + t) / P(X > s)
p_s_plus_t = stats.expon.sf(s + t, scale=1/lam)
p_s = stats.expon.sf(s, scale=1/lam)
conditional = p_s_plus_t / p_s

# P(X > t)
p_t = stats.expon.sf(t, scale=1/lam)

print(f"P(X > {s+t} | X > {s}) = {conditional:.4f}")
print(f"P(X > {t}) = {p_t:.4f}")
print(f"Equal? {np.isclose(conditional, p_t)}")

# Simulation verification of memoryless property
samples = np.random.exponential(1/lam, 100000)
# Condition: X > s
filtered = samples[samples > s]
conditional_sim = (filtered > s + t).mean()
print(f"\nSimulated P(X > {s+t} | X > {s}) = {conditional_sim:.4f}")
```

---

### Laplace Distribution

#### PDF

$$
f(x) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right)
$$

Denoted as $X \sim \text{Laplace}(\mu, b)$.

#### Expectation and Variance

$$
\mathbb{E}[X] = \mu
$$

$$
\text{Var}(X) = 2b^2
$$

#### Difference from Normal Distribution

- Normal distribution: $(x - \mu)^2$ → smooth peak
- Laplace distribution: $|x - \mu|$ → sharp peak, heavier tails

#### Deep Learning Applications

- **L1 regularization**: Laplace prior leads to L1 penalty
- **Sparse modeling**: Tends to produce sparser solutions than Gaussian
- **Anomaly detection**: Heavy-tailed property is more robust to outliers

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Compare Laplace and normal distributions
x = np.linspace(-5, 5, 200)

# Laplace distribution (μ=0, b=1)
laplace_pdf = stats.laplace.pdf(x, loc=0, scale=1)

# Normal distribution (μ=0, σ=1)
# Make variance same: Var(Laplace) = 2b² = 2, Var(Normal) = σ² = 2 → σ = √2
normal_pdf = stats.norm.pdf(x, loc=0, scale=np.sqrt(1))

plt.figure(figsize=(10, 5))
plt.plot(x, laplace_pdf, 'b-', linewidth=2, label='Laplace(0, 1)')
plt.plot(x, normal_pdf, 'r-', linewidth=2, label='Normal(0, 1)')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Laplace Distribution vs Normal Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('laplace_vs_normal.png', dpi=100)
print("Laplace vs Normal distribution comparison image saved")

# L1 regularization = Laplace prior MAP
print("\nBayesian interpretation of L1 regularization:")
print("Prior: θ ~ Laplace(0, b)")
print("Log prior: log p(θ) = -|θ|/b + const")
print("MAP: min -log p(y|x,θ) - log p(θ)")
print("    = min Loss + λ|θ|  (L1 regularization)")
```

---

### Beta Distribution

#### PDF

$$
f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad x \in [0, 1]
$$

Where $B(\alpha, \beta) = \dfrac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$ is the Beta function.

#### Expectation and Variance

$$
\mathbb{E}[X] = \frac{\alpha}{\alpha + \beta}
$$

$$
\text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
$$

#### Shape Characteristics

| Parameters | Shape |
|------|------|
| $\alpha = \beta = 1$ | Uniform distribution |
| $\alpha = \beta > 1$ | Symmetric, central peak |
| $\alpha = \beta < 1$ | Symmetric, U-shape |
| $\alpha > \beta$ | Right-skewed |
| $\alpha < \beta$ | Left-skewed |

#### Deep Learning Applications

- **Bayesian inference**: Conjugate prior for Bernoulli/Binomial distributions
- **Hyperparameter tuning**: Uncertainty modeling of learning rates, Dropout rates
- **Reinforcement learning**: Thompson Sampling in Bandit problems

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Different shapes of Beta distribution
params = [
    (1, 1, 'Uniform'),
    (2, 2, 'Symmetric peak'),
    (0.5, 0.5, 'U-shape'),
    (2, 5, 'Left-skewed'),
    (5, 2, 'Right-skewed'),
]

x = np.linspace(0, 1, 200)

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for i, (alpha, beta, name) in enumerate(params):
    pdf = stats.beta.pdf(x, alpha, beta)
    axes[i].plot(x, pdf, 'b-', linewidth=2)
    axes[i].fill_between(x, pdf, alpha=0.3)
    axes[i].set_title(f'Beta({alpha}, {beta}) - {name}\nE[X] = {alpha/(alpha+beta):.2f}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('f(x)')
    axes[i].grid(True, alpha=0.3)
    axes[i].set_ylim(0, None)

# Remove extra subplot
axes[5].axis('off')

plt.tight_layout()
plt.savefig('beta.png', dpi=100)
print("Beta distribution image saved")

# Bayesian update example: Coin flip experiment
print("\nBayesian update: Coin flip experiment")
print("="*50)

# Prior: Beta(1, 1) = uniform distribution
alpha_prior, beta_prior = 1, 1

# Observed data: 7 heads, 3 tails
heads, tails = 7, 3

# Posterior: Beta(α + heads, β + tails)
alpha_post = alpha_prior + heads
beta_post = beta_prior + tails

print(f"Prior: Beta({alpha_prior}, {beta_prior})")
print(f"Data: {heads} heads, {tails} tails")
print(f"Posterior: Beta({alpha_post}, {beta_post})")
print(f"Posterior mean (coin fairness estimate): {alpha_post/(alpha_post+beta_post):.4f}")

# Visualize update process
x = np.linspace(0, 1, 200)
prior_pdf = stats.beta.pdf(x, alpha_prior, beta_prior)
posterior_pdf = stats.beta.pdf(x, alpha_post, beta_post)

plt.figure(figsize=(10, 5))
plt.plot(x, prior_pdf, 'b--', linewidth=2, label='Prior: Beta(1,1)')
plt.plot(x, posterior_pdf, 'r-', linewidth=2, label=f'Posterior: Beta({alpha_post},{beta_post})')
plt.axvline(x=0.5, color='gray', linestyle=':', label='Fair coin (p=0.5)')
plt.axvline(x=heads/(heads+tails), color='green', linestyle=':', label=f'MLE (p={heads/(heads+tails):.2f})')
plt.xlabel('Probability of Heads (p)')
plt.ylabel('Density')
plt.title('Bayesian Updating: Coin Flip')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('bayesian_update.png', dpi=100)
print("\nBayesian update image saved")
```

---

### Gamma Distribution

#### PDF

$$
f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0
$$

Denoted as $X \sim \text{Gamma}(\alpha, \beta)$.

- $\alpha$ (shape parameter): controls distribution shape
- $\beta$ (rate parameter): controls decay speed

#### Expectation and Variance

$$
\mathbb{E}[X] = \frac{\alpha}{\beta}
$$

$$
\text{Var}(X) = \frac{\alpha}{\beta^2}
$$

#### Special Cases

- $\alpha = 1$: Exponential distribution $\text{Exp}(\beta)$
- $\alpha = n/2, \beta = 1/2$: Chi-squared distribution $\chi^2_n$
- $\alpha$ integer: Sum of $n$ independent exponential distributions

#### Deep Learning Applications

- **Waiting time modeling**: Total waiting time for multiple Poisson events
- **Conjugate prior**: Conjugate prior for Poisson distribution
- **Variational inference**: Prior distribution for parameters

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Gamma distribution
x = np.linspace(0, 20, 200)

# Different shape parameters
params = [(1, 0.5), (2, 0.5), (3, 0.5), (5, 1), (9, 2)]

plt.figure(figsize=(10, 5))

for alpha, beta in params:
    pdf = stats.gamma.pdf(x, a=alpha, scale=1/beta)
    plt.plot(x, pdf, linewidth=2, label=f'α={alpha}, β={beta} (E[X]={alpha/beta:.1f})')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Gamma Distribution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gamma.png', dpi=100)
print("Gamma distribution image saved")
```

---

## Relationships Between Distributions

### Important Relationship Diagram

```
                    Bernoulli Distribution
                          │
         n independent sums  │
                          ↓
                    Binomial Distribution
                          │
    n→∞, p→0, np=λ       │
                          ↓
                    Poisson Distribution

───────────────────────────────────────────────

          Exponential Distribution
               │
   n independent sums │
               ↓
          Gamma Distribution
               │
    α=n/2, β=1/2
               ↓
          Chi-squared Distribution

───────────────────────────────────────────────

          Uniform Distribution Uniform(0,1)
               │
      Inverse transform sampling
               ↓
       Any distribution

───────────────────────────────────────────────

     Beta(α,β) is conjugate prior for Binomial
     Gamma(α,β) is conjugate prior for Poisson
```

### Central Position of Normal Distribution

According to the **Central Limit Theorem**, sums of many independent random variables tend to normal distribution:

$$
\frac{1}{\sqrt{n}} \sum_{i=1}^n X_i \xrightarrow{d} \mathcal{N}(0, \sigma^2)
$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Demonstrate Central Limit Theorem
n_samples = 10000

fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Original distribution: uniform distribution
original_dist = np.random.uniform(0, 1, n_samples)

# Distribution of sums for different sample sizes
for i, n in enumerate([1, 2, 5, 10, 30, 100]):
    row, col = i // 3, i % 3

    # Sample n uniform distributions and sum
    sums = np.sum(np.random.uniform(0, 1, (n_samples, n)), axis=1)
    # Standardize
    standardized = (sums - n * 0.5) / np.sqrt(n / 12)

    # Histogram
    axes[row, col].hist(standardized, bins=50, density=True, alpha=0.7)

    # Overlay normal distribution
    x = np.linspace(-4, 4, 100)
    axes[row, col].plot(x, stats.norm.pdf(x), 'r-', linewidth=2)

    axes[row, col].set_title(f'n = {n}')
    axes[row, col].set_xlim(-4, 4)
    axes[row, col].grid(True, alpha=0.3)

plt.suptitle('Central Limit Theorem Demonstration: Sum of Uniform Distributions Tends to Normal', fontsize=14)
plt.tight_layout()
plt.savefig('clt.png', dpi=100)
print("Central Limit Theorem demonstration image saved")
```

---

## Applications in Deep Learning

### 1. Weight Initialization

```python
import numpy as np

def initialize_weights(shape, method='xavier', activation='relu'):
    """
    Weight initialization methods

    Parameters:
    -----------
    shape : tuple
        Weight matrix shape
    method : str
        'xavier' or 'he'
    activation : str
        Activation function type ('relu', 'tanh', 'sigmoid')
    """
    fan_in = shape[1] if len(shape) == 2 else shape[0]
    fan_out = shape[0] if len(shape) == 2 else shape[1]

    if method == 'xavier':
        # Xavier/Glorot initialization
        # Suitable for tanh, sigmoid
        std = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0, std, shape)

    elif method == 'he':
        # He/Kaiming initialization
        # Suitable for ReLU and its variants
        std = np.sqrt(2 / fan_in)
        return np.random.normal(0, std, shape)

    elif method == 'lecun':
        # LeCun initialization
        # Suitable for SELU
        std = np.sqrt(1 / fan_in)
        return np.random.normal(0, std, shape)

# Example
print("Weight initialization examples:")
print("="*50)

# Xavier initialization (for tanh)
w_xavier = initialize_weights((256, 784), method='xavier')
print(f"Xavier initialization: {w_xavier.shape}")
print(f"  Theoretical variance: {2 / (784 + 256):.6f}")
print(f"  Actual variance: {w_xavier.var():.6f}")

# He initialization (for ReLU)
w_he = initialize_weights((256, 784), method='he')
print(f"\nHe initialization: {w_he.shape}")
print(f"  Theoretical variance: {2 / 784:.6f}")
print(f"  Actual variance: {w_he.var():.6f}")
```

### 2. Dropout Implementation

```python
import numpy as np

class Dropout:
    """
    Dropout regularization

    During training, randomly sets neurons to zero with probability p,
    during testing, keeps all neurons but scales output.
    """

    def __init__(self, p=0.5):
        """
        Parameters:
        -----------
        p : float
            Retention probability (not dropout probability!)
        """
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, x):
        if not self.training:
            return x

        # Bernoulli sampling
        self.mask = (np.random.random(x.shape) < self.p).astype(x.dtype)
        # Scale to preserve expectation
        return x * self.mask / self.p

    def backward(self, grad_output):
        if not self.training:
            return grad_output
        return grad_output * self.mask / self.p

# Example
dropout = Dropout(p=0.5)
x = np.random.randn(1000, 100)

# Training mode
dropout.training = True
x_train = dropout.forward(x)
print(f"Training mode:")
print(f"  Original mean: {x.mean():.4f}")
print(f"  Mean after Dropout: {x_train.mean():.4f}")
print(f"  Zero element ratio: {(x_train == 0).mean():.2%}")

# Test mode
dropout.training = False
x_test = dropout.forward(x)
print(f"\nTest mode:")
print(f"  Output mean: {x_test.mean():.4f}")
print(f"  Zero element ratio: {(x_test == 0).mean():.2%}")
```

### 3. Cross-Entropy Loss and Categorical Distribution

```python
import numpy as np

def cross_entropy_loss(predictions, targets):
    """
    Cross-entropy loss

    Parameters:
    -----------
    predictions : array, shape (N, K)
        Softmax output (probability distribution)
    targets : array, shape (N,) or (N, K)
        Class indices or one-hot encoding
    """
    N = predictions.shape[0]

    # Numerical stability
    eps = 1e-15
    predictions = np.clip(predictions, eps, 1 - eps)

    if targets.ndim == 1:
        # Class indices
        return -np.mean(np.log(predictions[np.arange(N), targets]))
    else:
        # One-hot encoding
        return -np.mean(np.sum(targets * np.log(predictions), axis=1))

# Softmax function
def softmax(logits):
    """Numerically stable Softmax"""
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Example: 3-class classification problem
logits = np.array([
    [2.0, 1.0, 0.1],  # Predicting class 0
    [0.1, 2.0, 1.0],  # Predicting class 1
    [0.5, 0.5, 2.0],  # Predicting class 2
])

targets = np.array([0, 1, 2])  # True classes

probs = softmax(logits)
loss = cross_entropy_loss(probs, targets)

print("Cross-entropy loss example:")
print(f"Logits:\n{logits}")
print(f"\nSoftmax probabilities:\n{probs}")
print(f"\nTrue classes: {targets}")
print(f"Cross-entropy loss: {loss:.4f}")

# Relationship with negative log likelihood
print("\nNegative log likelihood interpretation:")
for i, (p, t) in enumerate(zip(probs, targets)):
    print(f"  Sample {i}: -log(p[{t}]) = {-np.log(p[t]):.4f}")
```

### 4. Reparameterization in Variational Autoencoder (VAE)

```python
import numpy as np

def reparameterize(mu, log_var):
    """
    VAE reparameterization trick

    Sample from N(μ, σ²):
    z = μ + σ * ε, where ε ~ N(0, 1)

    This makes the sampling operation differentiable,
    allowing training through backpropagation.
    """
    # Sample from standard normal distribution
    epsilon = np.random.standard_normal(mu.shape)

    # Reparameterization
    std = np.exp(0.5 * log_var)  # σ = exp(0.5 * log(σ²))
    z = mu + std * epsilon

    return z, epsilon

def vae_loss(x_reconstructed, x, mu, log_var, beta=1.0):
    """
    VAE loss function

    Loss = Reconstruction Loss + β * KL Divergence

    KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
    """
    # Reconstruction loss (assuming Gaussian distribution)
    recon_loss = np.mean((x_reconstructed - x) ** 2)

    # KL divergence
    kl_loss = -0.5 * np.mean(1 + log_var - mu**2 - np.exp(log_var))

    return recon_loss + beta * kl_loss

# Example
np.random.seed(42)

# Encoder output
mu = np.array([0.0, 1.0, -0.5])
log_var = np.array([0.0, 0.5, 1.0])  # σ² = [1, 1.65, 2.72]

# Reparameterization sampling
z, epsilon = reparameterize(mu, log_var)

print("VAE reparameterization example:")
print(f"μ = {mu}")
print(f"log(σ²) = {log_var}")
print(f"σ = {np.exp(0.5 * log_var)}")
print(f"ε = {epsilon}")
print(f"z = μ + σ*ε = {z}")

# Multiple samples
n_samples = 10000
z_samples = np.zeros((n_samples, 3))
for i in range(n_samples):
    z_samples[i], _ = reparameterize(mu, log_var)

print(f"\nSampling statistics ({n_samples} times):")
print(f"Sample mean: {z_samples.mean(axis=0)}")
print(f"Theoretical mean: {mu}")
print(f"Sample variance: {z_samples.var(axis=0)}")
print(f"Theoretical variance: {np.exp(log_var)}")
```

---

## Summary

This chapter introduced random variables and common probability distributions, which are the foundations for understanding uncertainty and randomness in deep learning.

### Core Concept Comparison Table

| Concept | Discrete | Continuous |
|------|--------|--------|
| Description function | PMF: $p(x)$ | PDF: $f(x)$ |
| Normalization | $\sum_x p(x) = 1$ | $\int f(x)dx = 1$ |
| Probability calculation | $P(X = x) = p(x)$ | $P(a \le X \le b) = \int_a^b f(x)dx$ |
| Expectation | $\sum x \cdot p(x)$ | $\int x \cdot f(x)dx$ |
| Variance | $\mathbb{E}[X^2] - (\mathbb{E}[X])^2$ | $\mathbb{E}[X^2] - (\mathbb{E}[X])^2$ |

### Common Distributions Summary

| Distribution | Type | Parameters | Expectation | Variance | Deep Learning Applications |
|------|------|------|------|------|--------------|
| Bernoulli | Discrete | $p$ | $p$ | $p(1-p)$ | Dropout, binary classification |
| Binomial | Discrete | $n, p$ | $np$ | $np(1-p)$ | Ensemble learning |
| Poisson | Discrete | $\lambda$ | $\lambda$ | $\lambda$ | Sparse coding |
| Categorical | Discrete | $\mathbf{p}$ | - | - | Multi-classification |
| Uniform | Continuous | $a, b$ | $(a+b)/2$ | $(b-a)^2/12$ | Initialization |
| Normal | Continuous | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ | Initialization, VAE |
| Exponential | Continuous | $\lambda$ | $1/\lambda$ | $1/\lambda^2$ | Waiting time |
| Laplace | Continuous | $\mu, b$ | $\mu$ | $2b^2$ | L1 regularization |
| Beta | Continuous | $\alpha, \beta$ | $\alpha/(\alpha+\beta)$ | - | Bayesian inference |

### Key Points

1. **Random variables**: Functions that quantify random phenomena
2. **PMF vs PDF**: Use mass functions for discrete, density functions for continuous
3. **Normal distribution**: Most important, Central Limit Theorem guarantees its universality
4. **Conjugate priors**: Beta-Binomial, Gamma-Poisson, simplify Bayesian inference
5. **Reparameterization**: Makes random sampling differentiable, core technology of VAE

---

**Previous section**: [Chapter 3 (a): Probability Basics and Conditional Probability](03a-probability-basics-conditional_EN.md)

**Next section**: [Chapter 3 (c): Multivariate Random Variables and Numerical Characteristics](03c-multivariate-random-variables_EN.md) - Learn about joint distributions, marginal distributions, covariance, and other concepts.

**Return**: [Mathematics Fundamentals Tutorial Table of Contents](../math-fundamentals.md)
