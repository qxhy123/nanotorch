# Chapter 4(a): Statistics and Sampling Distributions

Mathematical statistics is the theoretical foundation of **learning from data**. Statistics are tools for extracting information from samples, while sampling distributions are the mathematical framework for understanding the behavior of these statistics. This chapter introduces basic statistical concepts, common statistics, and their distributions, which serve as the theoretical foundation for understanding Batch Normalization and model evaluation.

---

## Table of Contents

1. [Population and Sample](#population-and-sample)
2. [Definition of Statistics](#definition-of-statistics)
3. [Common Statistics](#common-statistics)
4. [Sampling Distribution](#sampling-distribution)
5. [Three Major Sampling Distributions](#three-major-sampling-distributions)
6. [Applications of the Central Limit Theorem](#applications-of-the-central-limit-theorem)
7. [Applications in Deep Learning](#applications-in-deep-learning)
8. [Summary](#summary)

---

## Population and Sample

### Basic Concepts

**Population**: The entire set of objects under study, usually described by a random variable $X$ or its distribution function $F(x)$.

**Individual**: Each basic unit in the population.

**Sample**: A collection of $n$ individuals randomly drawn from the population, denoted as $X_1, X_2, \ldots, X_n$.

**Sample Size**: The number of individuals in the sample, denoted as $n$.

### Simple Random Sample

**Definition**: A sample $X_1, X_2, \ldots, X_n$ is called a **Simple Random Sample (SRS)** if it satisfies the following two conditions:

1. **Independence**: $X_1, X_2, \ldots, X_n$ are mutually independent
2. **Identical Distribution**: Each $X_i$ has the same distribution as the population $X$

That is, $X_1, X_2, \ldots, X_n \stackrel{i.i.d.}{\sim} F(x)$

### Duality of Samples

Samples possess **duality**:

| Perspective | Meaning | Representation |
|-------------|---------|----------------|
| Random Variable | Before sampling, the result is random | $X_1, X_2, \ldots, X_n$ |
| Specific Values | After sampling, the result is deterministic | $x_1, x_2, \ldots, x_n$ |

### Sample Space

**Sample Space**: The set of all possible sample values, denoted as $\mathcal{X}^n$.

For continuous populations, the sample space is typically a subset of $\mathbb{R}^n$.

```python
import numpy as np
import matplotlib.pyplot as plt

# Example: Drawing samples from a normal population
np.random.seed(42)

# Population parameters
true_mu = 100
true_sigma = 15

# Draw sample
n = 50  # Sample size
sample = np.random.normal(true_mu, true_sigma, n)

print(f"Population parameters: μ = {true_mu}, σ = {true_sigma}")
print(f"Sample size: n = {n}")
print(f"First 5 sample values: {sample[:5]}")
print(f"Sample mean: {sample.mean():.2f}")
print(f"Sample standard deviation: {sample.std():.2f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Sample histogram
axes[0].hist(sample, bins=15, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(true_mu - 4*true_sigma, true_mu + 4*true_sigma, 100)
axes[0].plot(x, 1/(true_sigma*np.sqrt(2*np.pi)) * np.exp(-(x-true_mu)**2/(2*true_sigma**2)),
             'r-', linewidth=2, label='Population distribution')
axes[0].axvline(sample.mean(), color='green', linestyle='--', label=f'Sample mean={sample.mean():.1f}')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Density')
axes[0].set_title('Sample Histogram and Population Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Distribution of sample means from multiple samples
n_samples = 1000
sample_means = [np.random.normal(true_mu, true_sigma, n).mean() for _ in range(n_samples)]
axes[1].hist(sample_means, bins=30, density=True, alpha=0.7, edgecolor='black')
axes[1].axvline(true_mu, color='red', linestyle='--', label=f'Population mean={true_mu}')
axes[1].set_xlabel('Sample mean')
axes[1].set_ylabel('Density')
axes[1].set_title(f'Distribution of Sample Means (Repeated {n_samples} samples)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sample_distribution.png', dpi=100)
print("\nImage saved: sample_distribution.png")
```

---

## Definition of Statistics

### Concept of Statistics

**Definition**: Let $X_1, X_2, \ldots, X_n$ be a sample from a population. If $T(X_1, \ldots, X_n)$ is a function of the sample and does not depend on any unknown parameters, then $T$ is called a **statistic**.

### Characteristics of Statistics

1. **Function of the Sample**: Computed from sample data
2. **No Unknown Parameters**: Once the sample is determined, the value of the statistic is uniquely determined
3. **Random Variable**: Because the sample is random

### Statistics vs Parameters

| Concept | Symbol | Meaning | Nature |
|---------|--------|---------|--------|
| **Parameter** | $\theta$ | Quantity describing population characteristics | Unknown, fixed |
| **Statistic** | $\hat{\theta}$ | Quantity computed from sample | Known, random |

### Common Parameters and Corresponding Statistics

| Population Parameter | Symbol | Sample Statistic | Symbol |
|---------------------|--------|------------------|--------|
| Population mean | $\mu$ | Sample mean | $\bar{X}$ |
| Population variance | $\sigma^2$ | Sample variance | $S^2$ |
| Population proportion | $p$ | Sample proportion | $\hat{p}$ |
| Population correlation coefficient | $\rho$ | Sample correlation coefficient | $R$ |

```python
import numpy as np

# Demonstration of statistics
np.random.seed(42)
sample = np.random.normal(50, 10, 100)

# Define various statistic functions
def sample_mean(data):
    """Sample mean"""
    return np.mean(data)

def sample_variance_unbiased(data):
    """Unbiased sample variance (n-1)"""
    return np.var(data, ddof=1)

def sample_variance_biased(data):
    """Biased sample variance (n)"""
    return np.var(data, ddof=0)

def sample_median(data):
    """Sample median"""
    return np.median(data)

def sample_range(data):
    """Sample range"""
    return np.max(data) - np.min(data)

def sample_iqr(data):
    """Interquartile range"""
    q75, q25 = np.percentile(data, [75, 25])
    return q75 - q25

print("Statistic Calculation Examples:")
print("="*50)
print(f"Sample size: n = {len(sample)}")
print(f"Sample mean: {sample_mean(sample):.4f}")
print(f"Unbiased sample variance: {sample_variance_unbiased(sample):.4f}")
print(f"Biased sample variance: {sample_variance_biased(sample):.4f}")
print(f"Sample standard deviation: {np.sqrt(sample_variance_unbiased(sample)):.4f}")
print(f"Sample median: {sample_median(sample):.4f}")
print(f"Sample range: {sample_range(sample):.4f}")
print(f"Interquartile range: {sample_iqr(sample):.4f}")
```

---

## Common Statistics

### Sample Mean

$$
\bar{X} = \frac{1}{n} \sum_{i=1}^n X_i
$$

**Properties**:
- **Unbiasedness**: $\mathbb{E}[\bar{X}] = \mu$
- **Variance**: $\text{Var}(\bar{X}) = \frac{\sigma^2}{n}$
- **Consistency**: $\bar{X} \xrightarrow{P} \mu$ (Law of Large Numbers)

### Sample Variance

$$
S^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})^2
$$

**Why use $n-1$?**

Because $\bar{X}$ is estimated from the sample, one degree of freedom is lost. Using $n-1$ ensures that $S^2$ is an **unbiased estimator** of $\sigma^2$.

**Detailed proof of sample variance unbiasedness**:

**Step 1**: Expand the sum of squares.

$$\sum_{i=1}^n (X_i - \bar{X})^2 = \sum_{i=1}^n (X_i - \mu + \mu - \bar{X})^2$$

$$= \sum_{i=1}^n [(X_i - \mu) - (\bar{X} - \mu)]^2$$

$$= \sum_{i=1}^n (X_i - \mu)^2 - 2(\bar{X} - \mu)\sum_{i=1}^n (X_i - \mu) + n(\bar{X} - \mu)^2$$

**Step 2**: Simplify.

Since $\sum_{i=1}^n (X_i - \mu) = n(\bar{X} - \mu)$, we have:

$$= \sum_{i=1}^n (X_i - \mu)^2 - n(\bar{X} - \mu)^2$$

**Step 3**: Calculate the expectation.

$$\mathbb{E}\left[\sum_{i=1}^n (X_i - \mu)^2\right] = n\sigma^2$$

$$\mathbb{E}[(\bar{X} - \mu)^2] = \text{Var}(\bar{X}) = \frac{\sigma^2}{n}$$

Therefore:

$$\mathbb{E}\left[\sum_{i=1}^n (X_i - \bar{X})^2\right] = n\sigma^2 - n \cdot \frac{\sigma^2}{n} = (n-1)\sigma^2$$

**Step 4**: Draw the conclusion.

$$\mathbb{E}[S^2] = \mathbb{E}\left[\frac{1}{n-1}\sum_{i=1}^n (X_i - \bar{X})^2\right] = \frac{1}{n-1} \cdot (n-1)\sigma^2 = \sigma^2$$

$$\boxed{\mathbb{E}[S^2] = \sigma^2} \quad \text{(Unbiased estimator)}$$

$$
\mathbb{E}[S^2] = \sigma^2
$$

**Computational formula (simplified form)**:

$$
S^2 = \frac{1}{n-1}\left(\sum_{i=1}^n X_i^2 - n\bar{X}^2\right)
$$

### Sample Standard Deviation

$$
S = \sqrt{S^2}
$$

**Note**: $S$ is a **biased estimator** of $\sigma$ (by Jensen's inequality).

### Sample k-th Moment

**k-th Raw Moment**:

$$
A_k = \frac{1}{n} \sum_{i=1}^n X_i^k
$$

**k-th Central Moment**:

$$
B_k = \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^k
$$

### Sample Covariance (Two-dimensional Sample)

For two-dimensional sample $(X_1, Y_1), \ldots, (X_n, Y_n)$:

$$
S_{XY} = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})
$$

### Sample Correlation Coefficient

$$
R = \frac{S_{XY}}{S_X S_Y} = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2 \cdot \sum_{i=1}^n (Y_i - \bar{Y})^2}}
$$

```python
import numpy as np

# Deep understanding of sample variance unbiasedness
np.random.seed(42)

true_mu = 10
true_sigma2 = 4
n_simulations = 10000
sample_sizes = [5, 10, 30, 100]

print("Verification of sample variance unbiasedness:")
print("="*60)
print(f"True variance: σ² = {true_sigma2}")
print()

for n in sample_sizes:
    biased_vars = []
    unbiased_vars = []

    for _ in range(n_simulations):
        sample = np.random.normal(true_mu, np.sqrt(true_sigma2), n)
        biased_vars.append(np.var(sample, ddof=0))  # Denominator n
        unbiased_vars.append(np.var(sample, ddof=1))  # Denominator n-1

    print(f"Sample size n = {n}:")
    print(f"  Biased variance E[S²_b]: {np.mean(biased_vars):.4f} (bias: {np.mean(biased_vars) - true_sigma2:.4f})")
    print(f"  Unbiased variance E[S²_u]: {np.mean(unbiased_vars):.4f} (bias: {np.mean(unbiased_vars) - true_sigma2:.4f})")
    print()

# Sample covariance and correlation coefficient example
n = 100
x = np.random.randn(n)
y = 0.8 * x + 0.2 * np.random.randn(n)  # Correlated data

# Calculate sample covariance and correlation coefficient
sample_cov = np.cov(x, y)[0, 1]
sample_corr = np.corrcoef(x, y)[0, 1]

print("Sample covariance and correlation coefficient:")
print(f"  Sample covariance: {sample_cov:.4f}")
print(f"  Sample correlation coefficient: {sample_corr:.4f}")
```

---

## Sampling Distribution

### Definition

**Sampling Distribution**: The probability distribution of a statistic.

Since a statistic is a random variable, it has its own distribution. Understanding sampling distributions is the foundation for statistical inference.

### Distribution of Sample Mean

**Theorem**: Let $X_1, \ldots, X_n \stackrel{i.i.d.}{\sim} \mathcal{N}(\mu, \sigma^2)$, then:

$$
\bar{X} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)
$$

**Standardization**:

$$
Z = \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim \mathcal{N}(0, 1)
$$

**When σ is unknown**:

$$
T = \frac{\bar{X} - \mu}{S / \sqrt{n}} \sim t(n-1)
$$

### Distribution of Sample Variance

**Theorem**: Let $X_1, \ldots, X_n \stackrel{i.i.d.}{\sim} \mathcal{N}(\mu, \sigma^2)$, then:

$$
\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)
$$

**Corollary**: $\bar{X}$ and $S^2$ are mutually independent.

### Independence of Sample Mean and Sample Variance

For a normal population, $\bar{X}$ and $S^2$ are **mutually independent**. This is an important property of the normal distribution and the foundation for deriving the t-distribution.

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Verification of sample mean distribution
np.random.seed(42)
true_mu = 50
true_sigma = 10
n = 30
n_simulations = 10000

# Repeated sampling to calculate sample means
sample_means = []
sample_vars = []
for _ in range(n_simulations):
    sample = np.random.normal(true_mu, true_sigma, n)
    sample_means.append(sample.mean())
    sample_vars.append(sample.var(ddof=1))

sample_means = np.array(sample_means)
sample_vars = np.array(sample_vars)

# Theoretical distribution parameters
theoretical_mean = true_mu
theoretical_std = true_sigma / np.sqrt(n)

print("Sample Mean Distribution Verification:")
print("="*50)
print(f"Theoretical mean: {theoretical_mean:.4f}")
print(f"Simulated mean: {sample_means.mean():.4f}")
print(f"Theoretical standard deviation: {theoretical_std:.4f}")
print(f"Simulated standard deviation: {sample_means.std():.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Sample mean distribution
axes[0].hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(theoretical_mean - 4*theoretical_std, theoretical_mean + 4*theoretical_std, 100)
axes[0].plot(x, stats.norm.pdf(x, theoretical_mean, theoretical_std), 'r-', linewidth=2, label='Theoretical distribution')
axes[0].set_xlabel('Sample mean')
axes[0].set_ylabel('Density')
axes[0].set_title(f'Distribution of Sample Mean (n={n})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Standardized distribution
standardized = (sample_means - true_mu) / (true_sigma / np.sqrt(n))
axes[1].hist(standardized, bins=50, density=True, alpha=0.7, edgecolor='black')
x = np.linspace(-4, 4, 100)
axes[1].plot(x, stats.norm.pdf(x, 0, 1), 'r-', linewidth=2, label='N(0,1)')
axes[1].set_xlabel('Standardized sample mean')
axes[1].set_ylabel('Density')
axes[1].set_title('Standardized Distribution')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sampling_distribution_mean.png', dpi=100)
print("\nImage saved: sampling_distribution_mean.png")
```

---

## Three Major Sampling Distributions

### Chi-Squared Distribution (χ² Distribution)

#### Definition

If $Z_1, Z_2, \ldots, Z_n \stackrel{i.i.d.}{\sim} \mathcal{N}(0, 1)$, then:

$$
\chi^2 = \sum_{i=1}^n Z_i^2 \sim \chi^2(n)
$$

where $n$ is called the **degrees of freedom**.

#### Properties

1. **Expectation**: $\mathbb{E}[\chi^2(n)] = n$
2. **Variance**: $\text{Var}(\chi^2(n)) = 2n$
3. **Additivity**: If $\chi^2_1 \sim \chi^2(n_1)$, $\chi^2_2 \sim \chi^2(n_2)$, and they are independent, then $\chi^2_1 + \chi^2_2 \sim \chi^2(n_1 + n_2)$
4. **Asymptotic Normality**: When $n$ is large, $\chi^2(n) \approx \mathcal{N}(n, 2n)$

#### PDF

$$
f(x) = \frac{1}{2^{n/2}\Gamma(n/2)} x^{n/2-1} e^{-x/2}, \quad x > 0
$$

### t Distribution

#### Definition

If $Z \sim \mathcal{N}(0, 1)$, $V \sim \chi^2(n)$, and $Z$ and $V$ are independent, then:

$$
T = \frac{Z}{\sqrt{V/n}} \sim t(n)
$$

#### Properties

1. **Symmetry**: Symmetric about 0
2. **Expectation**: $\mathbb{E}[T] = 0$ (when $n > 1$)
3. **Variance**: $\text{Var}(T) = \frac{n}{n-2}$ (when $n > 2$)
4. **Heavier Tails**: Has heavier tails than the normal distribution
5. **Asymptotic Normality**: When $n \to \infty$, $t(n) \to \mathcal{N}(0, 1)$

#### PDF

$$
f(x) = \frac{\Gamma\left(\frac{n+1}{2}\right)}{\sqrt{n\pi}\Gamma\left(\frac{n}{2}\right)} \left(1 + \frac{x^2}{n}\right)^{-\frac{n+1}{2}}
$$

### F Distribution

#### Definition

If $U \sim \chi^2(m)$, $V \sim \chi^2(n)$, and $U$ and $V$ are independent, then:

$$
F = \frac{U/m}{V/n} \sim F(m, n)
$$

#### Properties

1. **Expectation**: $\mathbb{E}[F] = \frac{n}{n-2}$ (when $n > 2$)
2. **Reciprocal Relationship**: If $F \sim F(m, n)$, then $1/F \sim F(n, m)$
3. **Relationship with t Distribution**: $t^2(n) \sim F(1, n)$

### Relationship Diagram of Three Major Distributions

```
Standard Normal Distribution Z ~ N(0,1)
         │
         ├── Z² → χ²(1)
         │
         ├── Z / √(χ²(n)/n) → t(n)
         │
         └── χ²(m)/m / (χ²(n)/n) → F(m,n)
```

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Visualize three major sampling distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. χ² distribution
x = np.linspace(0, 30, 200)
for df in [1, 3, 5, 10, 20]:
    axes[0].plot(x, stats.chi2.pdf(x, df), label=f'ν={df}')
axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('χ² Distribution')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0, 0.5)

# 2. t distribution
x = np.linspace(-5, 5, 200)
for df in [1, 3, 10, 30]:
    axes[1].plot(x, stats.t.pdf(x, df), label=f'ν={df}')
axes[1].plot(x, stats.norm.pdf(x), 'k--', label='N(0,1)')
axes[1].set_xlabel('x')
axes[1].set_ylabel('f(x)')
axes[1].set_title('t Distribution (Compared with Standard Normal)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. F distribution
x = np.linspace(0, 5, 200)
for m, n in [(1, 10), (5, 10), (10, 10), (10, 5)]:
    axes[2].plot(x, stats.f.pdf(x, m, n), label=f'({m},{n})')
axes[2].set_xlabel('x')
axes[2].set_ylabel('f(x)')
axes[2].set_title('F Distribution')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('three_distributions.png', dpi=100)
print("Image saved: three_distributions.png")

# Verification of t distribution properties
print("\nt Distribution Property Verification:")
print("="*50)
n_simulations = 100000
df = 10

# Generate t distribution random numbers
z = np.random.randn(n_simulations)
v = np.random.chisquare(df, n_simulations)
t_samples = z / np.sqrt(v / df)

print(f"Degrees of freedom ν = {df}")
print(f"Theoretical mean: 0, Simulated mean: {t_samples.mean():.4f}")
print(f"Theoretical variance: {df/(df-2):.4f}, Simulated variance: {t_samples.var():.4f}")

# Compare tails with normal distribution
print(f"\nTail Probability Comparison P(|T| > 2):")
print(f"t({df}) distribution: {2 * (1 - stats.t.cdf(2, df)):.4f}")
print(f"Standard normal distribution: {2 * (1 - stats.norm.cdf(2)):.4f}")
```

---

## Applications of the Central Limit Theorem

### Review of the Central Limit Theorem

Let $X_1, \ldots, X_n \stackrel{i.i.d.}{\sim} F$, $\mathbb{E}[X_i] = \mu$, $\text{Var}(X_i) = \sigma^2$, then:

$$
\frac{\bar{X} - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)
$$

### Practical Applications

**Approximation**: When $n$ is sufficiently large (typically $n \geq 30$):

$$
\bar{X} \approx \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)
$$

### Case of Arbitrary Population

Regardless of the population distribution (as long as variance is finite), the sample mean approximately follows a normal distribution.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Central Limit Theorem demonstration: Different population distributions
np.random.seed(42)
n_simulations = 10000

# Define different population distributions
distributions = {
    'Uniform Distribution': lambda n: np.random.uniform(0, 1, n),
    'Exponential Distribution': lambda n: np.random.exponential(1, n),
    'Bernoulli Distribution': lambda n: np.random.binomial(1, 0.3, n),
}

sample_sizes = [5, 10, 30, 100]

fig, axes = plt.subplots(len(distributions), len(sample_sizes), figsize=(16, 10))

for i, (dist_name, dist_func) in enumerate(distributions.items()):
    for j, n in enumerate(sample_sizes):
        # Repeated sampling to calculate sample means
        sample_means = [dist_func(n).mean() for _ in range(n_simulations)]

        # Draw histogram
        axes[i, j].hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black')

        # Overlay normal distribution (CLT approximation)
        mu = np.mean(sample_means)
        sigma = np.std(sample_means)
        x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
        axes[i, j].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2)

        axes[i, j].set_title(f'{dist_name}, n={n}')
        axes[i, j].grid(True, alpha=0.3)

plt.suptitle('Central Limit Theorem: Sample Means Converge to Normal Distribution', fontsize=14)
plt.tight_layout()
plt.savefig('clt_demonstration.png', dpi=100)
print("Image saved: clt_demonstration.png")
```

---

## Applications in Deep Learning

### 1. Theoretical Foundation of Batch Normalization

Batch Normalization uses **sample statistics** to estimate **population statistics**:

$$
\hat{\mu}_B = \frac{1}{m}\sum_{i=1}^m x_i
$$

$$
\hat{\sigma}_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \hat{\mu}_B)^2
$$

**Law of Large Numbers guarantees**: When batch size $m$ is sufficiently large, $\hat{\mu}_B \approx \mu$, $\hat{\sigma}_B^2 \approx \sigma^2$.

```python
import numpy as np

# Accuracy of Batch Normalization statistics estimation
np.random.seed(42)

true_mu = 0
true_sigma2 = 1
batch_sizes = [2, 8, 32, 128, 512]
n_simulations = 10000

print("Batch Normalization: Batch Statistics Estimate Population Statistics")
print("="*60)
print(f"True parameters: μ = {true_mu}, σ² = {true_sigma2}")
print()

for m in batch_sizes:
    mu_estimates = []
    var_estimates = []

    for _ in range(n_simulations):
        batch = np.random.normal(true_mu, np.sqrt(true_sigma2), m)
        mu_estimates.append(batch.mean())
        var_estimates.append(batch.var(ddof=0))

    mu_estimates = np.array(mu_estimates)
    var_estimates = np.array(var_estimates)

    print(f"Batch size m = {m}:")
    print(f"  Mean estimate: E[μ̂] = {mu_estimates.mean():.4f}, Var(μ̂) = {mu_estimates.var():.4f}")
    print(f"  Variance estimate: E[σ̂²] = {var_estimates.mean():.4f}, Var(σ̂²) = {var_estimates.var():.4f}")
    print(f"  Theoretical Var(μ̂) = σ²/m = {true_sigma2/m:.4f}")
    print()
```

### 2. Statistical Perspective of Model Evaluation

**Generalization Error**:

$$
R(\theta) = \mathbb{E}_{(x,y) \sim P_{data}}[\ell(f(x; \theta), y)]
$$

**Empirical Risk** (Training Error):

$$
\hat{R}(\theta) = \frac{1}{n}\sum_{i=1}^n \ell(f(x_i; \theta), y_i)
$$

**Law of Large Numbers**: $\hat{R}(\theta) \xrightarrow{P} R(\theta)$

### 3. Stochastic Estimation of Gradients

**True Gradient**:

$$
\nabla f(\theta) = \mathbb{E}_{(x,y)}[\nabla \ell(\theta; x, y)]
$$

**Stochastic Gradient Estimate**:

$$
\hat{g} = \frac{1}{B}\sum_{i \in \mathcal{B}} \nabla \ell(\theta; x_i, y_i)
$$

**Properties**:
- $\mathbb{E}[\hat{g}] = \nabla f(\theta)$ (Unbiased)
- $\text{Var}(\hat{g}) = O(1/B)$

### 4. Application of Confidence Intervals in Model Comparison

When comparing the performance differences of two models, you can use **paired t-test** or **confidence intervals**:

```python
import numpy as np
from scipy import stats

# Model comparison example
np.random.seed(42)

# Assume accuracy of two models on K-fold cross-validation
K = 10
model_a_acc = np.array([0.85, 0.87, 0.84, 0.86, 0.88, 0.85, 0.87, 0.86, 0.85, 0.87])
model_b_acc = np.array([0.82, 0.84, 0.81, 0.83, 0.85, 0.82, 0.84, 0.83, 0.82, 0.84])

# Paired differences
diff = model_a_acc - model_b_acc

# Calculate confidence interval
mean_diff = diff.mean()
se_diff = diff.std(ddof=1) / np.sqrt(K)
ci_95 = stats.t.interval(0.95, K-1, loc=mean_diff, scale=se_diff)

print("Model Comparison (Paired Differences):")
print("="*50)
print(f"Model A average accuracy: {model_a_acc.mean():.4f}")
print(f"Model B average accuracy: {model_b_acc.mean():.4f}")
print(f"Mean difference: {mean_diff:.4f}")
print(f"95% confidence interval: ({ci_95[0]:.4f}, {ci_95[1]:.4f})")

# t-test
t_stat, p_value = stats.ttest_rel(model_a_acc, model_b_acc)
print(f"\nt statistic: {t_stat:.4f}")
print(f"p value: {p_value:.6f}")
print(f"Conclusion: {'Significant difference' if p_value < 0.05 else 'No significant difference'} (α=0.05)")
```

---

## Summary

This chapter introduced the basics of statistics and sampling distributions, which are the theoretical foundations for understanding statistical estimation in deep learning.

### Core Concept Comparison Table

| Concept | Formula/Definition | Deep Learning Application |
|---------|-------------------|--------------------------|
| Sample mean | $\bar{X} = \frac{1}{n}\sum X_i$ | Batch Normalization |
| Sample variance | $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ | Batch Normalization |
| χ² distribution | $\sum Z_i^2 \sim \chi^2(n)$ | Variance testing |
| t distribution | $Z/\sqrt{V/n} \sim t(n)$ | Confidence intervals, hypothesis testing |
| F distribution | $\frac{U/m}{V/n} \sim F(m,n)$ | Variance comparison |

### Key Points

1. **Statistics**: Quantities calculated from samples, are random variables
2. **Unbiasedness**: $\mathbb{E}[\hat{\theta}] = \theta$
3. **Degrees of Freedom**: Sample variance uses $n-1$ to ensure unbiasedness
4. **Sampling Distribution**: The probability distribution of a statistic
5. **Central Limit Theorem**: Sample means converge to normal distribution

### Core Applications in Deep Learning

| Technique | Statistical Concepts Used |
|-----------|---------------------------|
| Batch Normalization | Sample mean, sample variance |
| SGD | Stochastic gradient estimation, unbiasedness |
| Model Evaluation | Confidence intervals, hypothesis testing |
| Cross Validation | Statistic stability |

---

**Next Section**: [Chapter 4(b): Parameter Estimation](04b-parameter-estimation_EN.md) - Learn about maximum likelihood estimation, method of moments, Bayesian estimation, and other methods.

**Return**: [Mathematics Fundamentals Tutorial Directory](../math-fundamentals.md)
