# Chapter 4(b): Parameter Estimation

Parameter estimation is one of the core tasks of statistical inference, which involves inferring unknown parameters in a population distribution based on sample data. In deep learning, maximum likelihood estimation is the theoretical foundation for designing loss functions. Understanding parameter estimation is crucial for understanding why cross-entropy loss is effective.

---

## Table of Contents

1. [Overview of Parameter Estimation](#overview-of-parameter-estimation)
2. [Point Estimation and Evaluation Criteria](#point-estimation-and-evaluation-criteria)
3. [Method of Moments](#method-of-moments)
4. [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
5. [Properties and Computation of MLE](#properties-and-computation-of-mle)
6. [Bayesian Estimation](#bayesian-estimation)
7. [Relationship Between MLE and Loss Functions](#relationship-between-mle-and-loss-functions)
8. [Applications in Deep Learning](#applications-in-deep-learning)
9. [Summary](#summary)

---

## Overview of Parameter Estimation

### Parameter Estimation Problem

**Problem Description**: The form of the population distribution is known, but it contains unknown parameters $\theta$. Based on samples $X_1, \ldots, X_n$, we need to estimate $\theta$.

**Examples**:
- Estimate the mean $\mu$ and variance $\sigma^2$ of a normal distribution $\mathcal{N}(\mu, \sigma^2)$
- Estimate the success probability $p$ of a Bernoulli distribution
- Estimate the parameter $\lambda$ of a Poisson distribution

### Types of Estimation

| Type | Goal | Result |
|------|------|--------|
| **Point Estimation** | Estimate a parameter with a single value | $\hat{\theta}$ |
| **Interval Estimation** | Give a possible range for the parameter | $(L, U)$ |

### Symbol Convention

| Symbol | Meaning |
|--------|---------|
| $\theta$ | True parameter (unknown, fixed) |
| $\hat{\theta}$ | Estimated value of the parameter (known, random) |
| $\hat{\theta}_{MLE}$ | Maximum Likelihood Estimation |
| $\hat{\theta}_{MM}$ | Method of Moments |

---

## Point Estimation and Evaluation Criteria

### Definition of Point Estimation

**Point Estimation**: Construct a statistic $\hat{\theta} = g(X_1, \ldots, X_n)$ as an estimate of the parameter $\theta$.

- **Estimator**: $\hat{\theta}$ as a random variable
- **Estimate**: The specific value of $\hat{\theta}$

### Evaluation Criteria

#### 1. Unbiasedness

$$
\mathbb{E}[\hat{\theta}] = \theta
$$

**Bias**:

$$
\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta
$$

Unbiased estimator: $\text{Bias}(\hat{\theta}) = 0$

#### 2. Efficiency

Among unbiased estimators, the estimator with the **minimum variance** is the most efficient.

$$
\text{Var}(\hat{\theta}_1) < \text{Var}(\hat{\theta}_2) \Rightarrow \hat{\theta}_1 \text{ is more efficient}
$$

**Cramér-Rao Lower Bound**: The theoretical lower bound for the variance of unbiased estimators:

$$
\text{Var}(\hat{\theta}) \geq \frac{1}{n \cdot I(\theta)}
$$

where $I(\theta)$ is the **Fisher Information**:

$$
I(\theta) = \mathbb{E}\left[\left(\frac{\partial \ln f(X; \theta)}{\partial \theta}\right)^2\right]
$$

#### 3. Consistency

When the sample size $n \to \infty$, the estimator converges to the true parameter:

$$
\hat{\theta}_n \xrightarrow{P} \theta
$$

#### 4. Mean Squared Error

$$
\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2] = \text{Var}(\hat{\theta}) + \text{Bias}(\hat{\theta})^2
$$

**Bias-Variance Tradeoff**: Sometimes a slightly biased estimator may have a smaller MSE.

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstration of bias-variance tradeoff
np.random.seed(42)
n_simulations = 10000
n = 10
true_sigma2 = 4

# Two variance estimators
unbiased_vars = []
biased_vars = []

for _ in range(n_simulations):
    sample = np.random.normal(0, np.sqrt(true_sigma2), n)
    unbiased_vars.append(np.var(sample, ddof=1))  # n-1
    biased_vars.append(np.var(sample, ddof=0))    # n

unbiased_vars = np.array(unbiased_vars)
biased_vars = np.array(biased_vars)

print("Comparison of Variance Estimators:")
print("="*50)
print(f"True variance: σ² = {true_sigma2}")
print()
print("Unbiased estimator (denominator n-1):")
print(f"  Expectation: {unbiased_vars.mean():.4f} (bias: {unbiased_vars.mean() - true_sigma2:.4f})")
print(f"  Variance: {unbiased_vars.var():.4f}")
print(f"  MSE: {np.mean((unbiased_vars - true_sigma2)**2):.4f}")
print()
print("Biased estimator (denominator n):")
print(f"  Expectation: {biased_vars.mean():.4f} (bias: {biased_vars.mean() - true_sigma2:.4f})")
print(f"  Variance: {biased_vars.var():.4f}")
print(f"  MSE: {np.mean((biased_vars - true_sigma2)**2):.4f}")

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(unbiased_vars, bins=50, alpha=0.7, label=f'Unbiased estimator (MSE={np.mean((unbiased_vars - true_sigma2)**2):.3f})')
ax.hist(biased_vars, bins=50, alpha=0.7, label=f'Biased estimator (MSE={np.mean((biased_vars - true_sigma2)**2):.3f})')
ax.axvline(true_sigma2, color='red', linestyle='--', linewidth=2, label=f'True value σ²={true_sigma2}')
ax.set_xlabel('Estimate')
ax.set_ylabel('Frequency')
ax.set_title('Unbiased vs Biased Estimator (n=10)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bias_variance_tradeoff.png', dpi=100)
print("\nImage saved: bias_variance_tradeoff.png")
```

---

## Method of Moments

### Basic Idea

Use **sample moments** to estimate **population moments**.

### Theoretical Foundation

**Law of Large Numbers**: Sample moments converge in probability to population moments.

$$
A_k = \frac{1}{n}\sum_{i=1}^n X_i^k \xrightarrow{P} \mathbb{E}[X^k] = \mu_k
$$

### Steps of Method of Moments

1. Calculate population moments $\mu_k(\theta_1, \ldots, \theta_m)$, expressed as functions of parameters
2. Replace population moments with sample moments $A_k$
3. Solve the system of equations to obtain parameter estimates

### Example: Normal Distribution

Let $X \sim \mathcal{N}(\mu, \sigma^2)$, use method of moments to estimate $\mu$ and $\sigma^2$.

**Step 1**: Calculate population moments

$$
\mu_1 = \mathbb{E}[X] = \mu
$$

$$
\mu_2 = \mathbb{E}[X^2] = \text{Var}(X) + (\mathbb{E}[X])^2 = \sigma^2 + \mu^2
$$

**Step 2**: Establish system of equations

$$
\begin{cases}
A_1 = \mu \\
A_2 = \sigma^2 + \mu^2
\end{cases}
$$

**Step 3**: Solve

$$
\hat{\mu}_{MM} = \bar{X}
$$

$$
\hat{\sigma}^2_{MM} = A_2 - \bar{X}^2 = \frac{1}{n}\sum_{i=1}^n (X_i - \bar{X})^2
$$

**Note**: The method of moments estimate of $\hat{\sigma}^2$ is **biased** (denominator is $n$ instead of $n-1$).

```python
import numpy as np

# Method of moments example
np.random.seed(42)
true_mu = 5
true_sigma2 = 4
n = 100

sample = np.random.normal(true_mu, np.sqrt(true_sigma2), n)

# Method of moments
mu_mm = sample.mean()
sigma2_mm = np.mean(sample**2) - sample.mean()**2

# Unbiased estimate (MLE + correction)
mu_mle = sample.mean()
sigma2_mle = np.mean((sample - sample.mean())**2)
sigma2_unbiased = sample.var(ddof=1)

print("Normal Distribution Parameter Estimation:")
print("="*50)
print(f"True parameters: μ = {true_mu}, σ² = {true_sigma2}")
print()
print(f"Method of moments: μ̂ = {mu_mm:.4f}, σ̂² = {sigma2_mm:.4f}")
print(f"MLE:    μ̂ = {mu_mle:.4f}, σ̂² = {sigma2_mle:.4f}")
print(f"Unbiased:   μ̂ = {mu_mle:.4f}, σ̂² = {sigma2_unbiased:.4f}")
print()
print(f"Bias of σ² estimates:")
print(f"  Method of moments/MLE bias: {sigma2_mm - true_sigma2:.4f}")
print(f"  Unbiased estimate bias: {sigma2_unbiased - true_sigma2:.4f}")

# Method of moments for Poisson distribution
print("\n" + "="*50)
print("Poisson Distribution Parameter Estimation:")
true_lambda = 3
n = 100
sample_poisson = np.random.poisson(true_lambda, n)

# Method of moments: μ₁ = λ
lambda_mm = sample_poisson.mean()
print(f"True parameter: λ = {true_lambda}")
print(f"Method of moments: λ̂ = {lambda_mm:.4f}")
```

---

## Maximum Likelihood Estimation

### Likelihood Function

#### Definition

Given observed data $x_1, \ldots, x_n$, the **likelihood function** is a function of the parameter $\theta$:

$$
L(\theta) = L(\theta; x_1, \ldots, x_n) = \prod_{i=1}^n f(x_i; \theta)
$$

where $f(x; \theta)$ is the probability density function (continuous) or probability mass function (discrete).

#### Likelihood vs Probability

| Probability | Likelihood |
|-------------|------------|
| Parameters fixed, data varies | Data fixed, parameters vary |
| $f(x | \theta)$ | $L(\theta | x)$ |

### Log-Likelihood Function

$$
\ell(\theta) = \ln L(\theta) = \sum_{i=1}^n \ln f(x_i; \theta)
$$

**Advantages**:
1. Converts multiplication to addition, making computation simpler
2. Avoids numerical underflow
3. Logarithm is monotonic, does not affect maximization

### Maximum Likelihood Estimation (MLE)

**Definition**: Choose the parameter value that maximizes the likelihood function:

$$
\hat{\theta}_{MLE} = \arg\max_\theta L(\theta) = \arg\max_\theta \ell(\theta)
$$

**Solution Method**: Usually by solving the likelihood equation:

$$
\frac{\partial \ell(\theta)}{\partial \theta} = 0
$$

### Example: Normal Distribution Parameter Estimation

Let $X_1, \ldots, X_n \sim \mathcal{N}(\mu, \sigma^2)$, find the MLE of $\mu$ and $\sigma^2$.

**Likelihood function**:

$$
L(\mu, \sigma^2) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
$$

**Log-likelihood**:

$$
\ell(\mu, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
$$

**Partial derivative with respect to $\mu$**:

$$
\frac{\partial \ell}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0
$$

$$
\Rightarrow \hat{\mu}_{MLE} = \bar{x} = \frac{1}{n}\sum_{i=1}^n x_i
$$

**Partial derivative with respect to $\sigma^2$**:

$$
\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i - \mu)^2 = 0
$$

$$
\Rightarrow \hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2
$$

**Conclusion**:
- $\hat{\mu}_{MLE}$ is unbiased
- $\hat{\sigma}^2_{MLE}$ is **biased** (bias is $-\sigma^2/n$)

```python
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# MLE example: Normal distribution
np.random.seed(42)
true_mu = 3
true_sigma = 2
n = 100

sample = np.random.normal(true_mu, true_sigma, n)

# Analytical solution
mu_mle = sample.mean()
sigma2_mle = np.mean((sample - mu_mle)**2)

print("Normal Distribution MLE:")
print("="*50)
print(f"True parameters: μ = {true_mu}, σ = {true_sigma}")
print(f"MLE: μ̂ = {mu_mle:.4f}, σ̂ = {np.sqrt(sigma2_mle):.4f}")

# Numerical optimization verification
def neg_log_likelihood(params, data):
    """Negative log-likelihood (for minimization)"""
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    n = len(data)
    return n/2 * np.log(2*np.pi) + n * np.log(sigma) + np.sum((data - mu)**2) / (2*sigma**2)

# Numerical optimization
result = minimize(neg_log_likelihood, [0, 0], args=(sample,), method='BFGS')
mu_num, sigma_num = result.x[0], np.exp(result.x[1])

print(f"\nNumerical optimization: μ̂ = {mu_num:.4f}, σ̂ = {sigma_num:.4f}")

# Visualize likelihood function
mu_range = np.linspace(mu_mle - 2, mu_mle + 2, 100)
sigma_range = np.linspace(0.5, 4, 100)
MU, SIGMA = np.meshgrid(mu_range, sigma_range)

# Calculate log-likelihood
def log_likelihood(mu, sigma, data):
    n = len(data)
    return -n/2 * np.log(2*np.pi) - n * np.log(sigma) - np.sum((data - mu)**2) / (2*sigma**2)

LL = np.zeros_like(MU)
for i in range(MU.shape[0]):
    for j in range(MU.shape[1]):
        LL[i, j] = log_likelihood(MU[i, j], SIGMA[i, j], sample)

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
contour = ax.contour(MU, SIGMA, LL, levels=20, cmap='viridis')
ax.scatter([mu_mle], [np.sqrt(sigma2_mle)], color='red', s=100, marker='*', label=f'MLE: ({mu_mle:.2f}, {np.sqrt(sigma2_mle):.2f})')
ax.scatter([true_mu], [true_sigma], color='blue', s=100, marker='o', label=f'True: ({true_mu}, {true_sigma})')
ax.set_xlabel('μ')
ax.set_ylabel('σ')
ax.set_title('Log-Likelihood Function Contour Plot')
ax.legend()
ax.grid(True, alpha=0.3)
plt.colorbar(contour, ax=ax, label='Log-Likelihood')

plt.tight_layout()
plt.savefig('mle_likelihood.png', dpi=100)
print("\nImage saved: mle_likelihood.png")
```

---

## Properties and Computation of MLE

### Asymptotic Properties

In large sample situations, MLE has the following **desirable properties**:

#### 1. Consistency

$$
\hat{\theta}_{MLE} \xrightarrow{P} \theta \quad \text{as } n \to \infty
$$

#### 2. Asymptotic Normality

$$
\sqrt{n}(\hat{\theta}_{MLE} - \theta) \xrightarrow{d} \mathcal{N}(0, I(\theta)^{-1})
$$

where $I(\theta)$ is the Fisher information.

#### 3. Asymptotic Efficiency

MLE achieves the **Cramér-Rao Lower Bound**, i.e., it is the unbiased estimator with minimum variance.

$$
\text{Var}(\hat{\theta}_{MLE}) \approx \frac{1}{n \cdot I(\theta)}
$$

### Fisher Information

**Definition** (single observation):

$$
I(\theta) = \mathbb{E}\left[\left(\frac{\partial \ln f(X; \theta)}{\partial \theta}\right)^2\right] = -\mathbb{E}\left[\frac{\partial^2 \ln f(X; \theta)}{\partial \theta^2}\right]
$$

**Fisher information for sample**: $n \cdot I(\theta)$

### Computational Techniques

#### 1. Properties of Log-Likelihood

$$
\frac{\partial \ell}{\partial \theta} = \sum_{i=1}^n \frac{\partial \ln f(x_i; \theta)}{\partial \theta}
$$

#### 2. Likelihood Ratio Test

Likelihood ratio statistic:

$$
\Lambda = 2[\ell(\hat{\theta}) - \ell(\theta_0)] \xrightarrow{d} \chi^2(k)
$$

where $k$ is the number of parameters.

```python
import numpy as np
from scipy import stats

# Demonstration of MLE asymptotic normality
np.random.seed(42)
true_p = 0.3  # Bernoulli parameter

sample_sizes = [30, 100, 500, 1000]
n_simulations = 10000

print("MLE Asymptotic Normality Verification (Bernoulli Distribution):")
print("="*60)
print(f"True parameter: p = {true_p}")
print()

for n in sample_sizes:
    mle_estimates = []

    for _ in range(n_simulations):
        sample = np.random.binomial(1, true_p, n)
        p_mle = sample.mean()
        mle_estimates.append(p_mle)

    mle_estimates = np.array(mle_estimates)

    # Standardization
    # Fisher information: I(p) = 1/(p(1-p))
    fisher_info = 1 / (true_p * (1 - true_p))
    standardized = np.sqrt(n * fisher_info) * (mle_estimates - true_p)

    print(f"Sample size n = {n}:")
    print(f"  MLE mean: {mle_estimates.mean():.4f} (true value: {true_p})")
    print(f"  MLE standard deviation: {mle_estimates.std():.4f} (theoretical: {np.sqrt(true_p*(1-true_p)/n):.4f})")
    print(f"  Standardized mean: {standardized.mean():.4f}")
    print(f"  Standardized standard deviation: {standardized.std():.4f}")
    print()
```

---

## Bayesian Estimation

### Bayesian Framework

In the Bayesian framework, parameters $\theta$ are treated as **random variables** with their own distributions.

**Bayes' Formula**:

$$
P(\theta | \text{data}) = \frac{P(\text{data} | \theta) \cdot P(\theta)}{P(\text{data})} \propto P(\text{data} | \theta) \cdot P(\theta)
$$

| Term | Meaning |
|------|---------|
| **Prior Distribution** $P(\theta)$ | Beliefs about parameters before observing data |
| **Likelihood** $P(\text{data} | \theta)$ | Probability of data given parameters |
| **Posterior Distribution** $P(\theta | \text{data})$ | Updated beliefs about parameters after observing data |
| **Evidence** $P(\text{data})$ | Normalization constant |

### Bayesian Estimators

Common Bayesian point estimators:

#### 1. Posterior Mean

$$
\hat{\theta}_{Bayes} = \mathbb{E}[\theta | \text{data}] = \int \theta \cdot P(\theta | \text{data}) d\theta
$$

#### 2. Posterior Median

$$
\hat{\theta}_{Bayes}: P(\theta \leq \hat{\theta} | \text{data}) = 0.5
$$

#### 3. Maximum A Posteriori (MAP) Estimation

$$
\hat{\theta}_{MAP} = \arg\max_\theta P(\theta | \text{data}) = \arg\max_\theta [P(\text{data} | \theta) \cdot P(\theta)]
$$

### Conjugate Priors

If the prior distribution and posterior distribution belong to the same distribution family, the prior is called a **conjugate prior**.

| Likelihood | Conjugate Prior | Posterior |
|------------|-----------------|-----------|
| Bernoulli | Beta | Beta |
| Poisson | Gamma | Gamma |
| Normal (mean) | Normal | Normal |

### Example: Bayesian Estimation for Bernoulli Distribution

Let $X_1, \ldots, X_n \sim \text{Bernoulli}(p)$, take the prior of $p$ as $\text{Beta}(\alpha, \beta)$.

**Posterior Distribution**:

$$
p | \text{data} \sim \text{Beta}\left(\alpha + \sum_{i=1}^n x_i, \beta + n - \sum_{i=1}^n x_i\right)
$$

**Posterior Mean**:

$$
\hat{p}_{Bayes} = \frac{\alpha + \sum x_i}{\alpha + \beta + n}
$$

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Bayesian estimation example: Bernoulli distribution
np.random.seed(42)
true_p = 0.3
n = 50
data = np.random.binomial(1, true_p, n)

# Prior parameters (Beta distribution)
alpha_prior = 1  # Equivalent to uniform distribution
beta_prior = 1

# Posterior parameters
sum_x = data.sum()
alpha_post = alpha_prior + sum_x
beta_post = beta_prior + n - sum_x

# Various estimates
p_mle = data.mean()
p_bayes = alpha_post / (alpha_post + beta_post)  # Posterior mean

print("Bernoulli Parameter Estimation:")
print("="*50)
print(f"True parameter: p = {true_p}")
print(f"Data: n = {n}, successes = {sum_x}")
print()
print(f"MLE: p̂ = {p_mle:.4f}")
print(f"Bayes (posterior mean): p̂ = {p_bayes:.4f}")
print(f"MAP (posterior mode): p̂ = {(alpha_post - 1)/(alpha_post + beta_post - 2):.4f}")
print()

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

x = np.linspace(0, 1, 1000)
prior_pdf = stats.beta.pdf(x, alpha_prior, beta_prior)
posterior_pdf = stats.beta.pdf(x, alpha_post, beta_post)

ax.plot(x, prior_pdf, 'b-', linewidth=2, label=f'Prior: Beta({alpha_prior}, {beta_prior})')
ax.plot(x, posterior_pdf, 'r-', linewidth=2, label=f'Posterior: Beta({alpha_post}, {beta_post})')
ax.axvline(true_p, color='green', linestyle='--', linewidth=2, label=f'True value p={true_p}')
ax.axvline(p_mle, color='purple', linestyle=':', linewidth=2, label=f'MLE p̂={p_mle:.3f}')
ax.axvline(p_bayes, color='orange', linestyle=':', linewidth=2, label=f'Bayes p̂={p_bayes:.3f}')

ax.set_xlabel('p')
ax.set_ylabel('Density')
ax.set_title('Bayesian Update: Prior → Posterior')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bayesian_estimation.png', dpi=100)
print("Image saved: bayesian_estimation.png")
```

---

## Relationship Between MLE and Loss Functions

### Key Theorem

**Minimizing Loss Function = Maximizing Likelihood Estimation**

| Probability Model | Negative Log-Likelihood = Loss Function |
|------------------|-----------------------------------------|
| $Y \sim \mathcal{N}(f_\theta(X), \sigma^2)$ | MSE Loss |
| $Y \sim \text{Bernoulli}(\sigma(f_\theta(X)))$ | Binary Cross-Entropy |
| $Y \sim \text{Categorical}(\text{Softmax}(f_\theta(X)))$ | Categorical Cross-Entropy |

### Derivation: MSE Loss

Assume $Y | X \sim \mathcal{N}(f_\theta(X), \sigma^2)$:

$$
\ell(\theta) = \sum_{i=1}^n \ln f(y_i | x_i; \theta) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - f_\theta(x_i))^2
$$

Maximizing $\ell(\theta)$ is equivalent to minimizing:

$$
\sum_{i=1}^n (y_i - f_\theta(x_i))^2 = \text{MSE Loss}
$$

### Derivation: Cross-Entropy Loss

Assume $Y | X \sim \text{Categorical}(\text{Softmax}(f_\theta(X)))$:

$$
\ell(\theta) = \sum_{i=1}^n \ln P(y_i | x_i; \theta) = \sum_{i=1}^n \ln \text{Softmax}(f_\theta(x_i))_{y_i}
$$

Negative log-likelihood = cross-entropy loss.

```python
import numpy as np

# Verification of the relationship between MLE and loss functions

# 1. MSE Loss corresponds to Gaussian noise assumption
print("Relationship between MLE and MSE Loss:")
print("="*50)

np.random.seed(42)
n = 100
true_slope = 2
true_intercept = 1

X = np.random.randn(n)
Y = true_slope * X + true_intercept + np.random.randn(n) * 0.5  # Gaussian noise

# Least squares solution = MLE
X_design = np.column_stack([np.ones(n), X])
theta_mle = np.linalg.lstsq(X_design, Y, rcond=None)[0]

print(f"True parameters: slope = {true_slope}, intercept = {true_intercept}")
print(f"MLE/LSE:  slope = {theta_mle[1]:.4f}, intercept = {theta_mle[0]:.4f}")

# 2. Cross-entropy loss corresponds to categorical distribution
print("\n" + "="*50)
print("Relationship between MLE and Cross-Entropy Loss:")
print("="*50)

# Assume true class distribution
true_logits = np.array([1.0, 2.0, 0.5])
true_probs = np.exp(true_logits) / np.exp(true_logits).sum()
print(f"True class probabilities: {true_probs}")

# Sampling
n_samples = 1000
labels = np.random.choice(3, n_samples, p=true_probs)

# MLE estimate of logits
# For sufficient statistics (class counts), MLE is directly determined by frequencies
counts = np.bincount(labels, minlength=3)
mle_probs = counts / n_samples

# Avoid log(0), add small amount
eps = 1e-10
mle_logits = np.log(mle_probs + eps)
mle_logits = mle_logits - mle_logits.mean()  # Normalize

print(f"MLE estimated probabilities: {mle_probs}")
print(f"Cross-entropy loss (using true probabilities): {-np.sum(true_probs * np.log(true_probs + eps)):.4f}")
```

---

## Applications in Deep Learning

### 1. Loss Function Design

| Task | Probability Model | Loss Function |
|------|------------------|---------------|
| Regression | $Y \sim \mathcal{N}(f(X), \sigma^2)$ | MSE |
| Binary Classification | $Y \sim \text{Bernoulli}(\sigma(f(X)))$ | BCE |
| Multi-class Classification | $Y \sim \text{Categorical}(\text{Softmax}(f(X)))$ | CE |

### 2. Bayesian Interpretation of Regularization

**L2 Regularization** = Gaussian Prior

$$
\text{Loss}_{reg} = \text{Loss} + \lambda \|\theta\|^2 = -\ell(\theta) - \ln P(\theta)
$$

where $P(\theta) \propto \exp(-\lambda \|\theta\|^2)$ is a Gaussian prior.

**L1 Regularization** = Laplace Prior

$$
P(\theta) \propto \exp(-\lambda \|\theta\|_1)
$$

### 3. Parameter Initialization

Understanding the prior distribution of parameters helps in choosing appropriate initialization strategies.

```python
import numpy as np

# Bayesian interpretation of regularization
print("Bayesian Interpretation of Regularization:")
print("="*50)

# L2 regularization = Gaussian prior N(0, 1/(2λ))
lambda_l2 = 0.01
sigma_prior = 1 / np.sqrt(2 * lambda_l2)
print(f"L2 regularization (λ={lambda_l2})")
print(f"  Equivalent to Gaussian prior: θ ~ N(0, {sigma_prior**2:.2f})")

# L1 regularization = Laplace prior
lambda_l1 = 0.01
b_prior = 1 / lambda_l1
print(f"\nL1 regularization (λ={lambda_l1})")
print(f"  Equivalent to Laplace prior: θ ~ Laplace(0, {b_prior:.2f})")

# Visualization
import matplotlib.pyplot as plt

theta = np.linspace(-3, 3, 1000)

gaussian_prior = np.exp(-lambda_l2 * theta**2) / np.sqrt(np.pi / lambda_l2)
laplace_prior = np.exp(-lambda_l1 * np.abs(theta)) / (2 / lambda_l1)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(theta, gaussian_prior, label=f'Gaussian prior (L2, λ={lambda_l2})')
ax.plot(theta, laplace_prior, label=f'Laplace prior (L1, λ={lambda_l1})')
ax.set_xlabel('θ')
ax.set_ylabel('Probability Density')
ax.set_title('Prior Distributions Corresponding to Regularization')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regularization_prior.png', dpi=100)
print("\nImage saved: regularization_prior.png")
```

---

## Summary

This chapter introduced the core methods of parameter estimation, particularly maximum likelihood estimation and its relationship with deep learning loss functions.

### Method Comparison

| Method | Idea | Advantages | Disadvantages |
|--------|------|------------|---------------|
| Method of Moments | Sample moments ≈ Population moments | Simple, intuitive | May not be unique, inefficient |
| MLE | Maximize likelihood | Asymptotically efficient, consistent | May have no analytical solution |
| Bayesian | Combine prior information | Quantifies uncertainty | Requires choosing prior |

### Core Formulas

| Formula | Application |
|---------|-------------|
| $L(\theta) = \prod f(x_i; \theta)$ | Likelihood function |
| $\ell(\theta) = \sum \ln f(x_i; \theta)$ | Log-likelihood |
| $\text{MSE} = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Regression loss (Gaussian MLE) |
| $\text{CE} = -\sum y_i \ln \hat{y}_i$ | Classification loss (Categorical MLE) |

### Key Points

1. **MLE is the most commonly used point estimation method**
2. **Minimizing loss = Maximizing likelihood**
3. **Regularization = Log prior**
4. **Bayesian methods provide complete uncertainty quantification**

---

**Previous Section**: [Chapter 4(a): Statistics and Sampling Distributions](04a-statistics-sampling-distributions_EN.md)

**Next Section**: [Chapter 4(c): Hypothesis Testing](04c-hypothesis-testing_EN.md) - Learn about hypothesis testing, p-values, confidence intervals, and other concepts.

**Return**: [Mathematics Fundamentals Tutorial Directory](../math-fundamentals.md)
