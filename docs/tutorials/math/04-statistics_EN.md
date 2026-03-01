# Chapter 4: Mathematical Statistics

Mathematical statistics is the theoretical foundation for **learning from data**. It provides methods for inferring population characteristics from samples and is the core methodology of machine learning and deep learning. This chapter systematically introduces core concepts of mathematical statistics and their applications in deep learning.

---

## Chapter Structure

For ease of learning and in-depth understanding, this chapter is divided into four sub-chapters:

### [4.1 Statistics and Sampling Distributions](04a-statistics-sampling-distributions_EN.md)

**Content Overview**:
- Basic concepts of population and sample
- Common statistics: sample mean, sample variance, sample moments
- Properties of statistics (unbiasedness, efficiency, consistency)
- Three major sampling distributions: $\chi^2$ distribution, $t$ distribution, $F$ distribution
- Distribution of sample mean and central limit theorem applications

**Core Concepts**:

| Concept | Formula | Application in Deep Learning |
|---------|---------|----------------------------|
| Sample mean | $\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i$ | Mean estimation in batch normalization |
| Sample variance | $S^2 = \frac{1}{n-1}\sum_{i=1}^n(X_i-\bar{X})^2$ | Variance estimation in batch normalization |
| $\chi^2$ distribution | $\chi^2_n = \sum_{i=1}^n Z_i^2$ | Confidence intervals for variance |
| $t$ distribution | $T = \frac{Z}{\sqrt{V/n}}$ | Small sample mean testing |

**[Start Learning →](04a-statistics-sampling-distributions_EN.md)**

---

### [4.2 Parameter Estimation](04b-parameter-estimation_EN.md)

**Content Overview**:
- Point estimation and evaluation criteria for estimators
- Unbiasedness, efficiency, consistency, mean squared error
- Method of moments
- Principles and applications of Maximum Likelihood Estimation (MLE)
- Introduction to Bayesian estimation
- Equivalence between MLE and deep learning loss functions

**Core Concepts**:

| Concept | Formula | Deep Learning Application |
|---------|---------|--------------------------|
| MLE | $\hat{\theta} = \arg\max_\theta L(\theta)$ | Loss function design |
| Unbiased estimator | $\mathbb{E}[\hat{\theta}] = \theta$ | BatchNorm statistics |
| Mean squared error | $\text{MSE} = \text{Var}(\hat{\theta}) + \text{Bias}^2$ | Bias-variance tradeoff |

**[Start Learning →](04b-parameter-estimation_EN.md)**

---

### [4.3 Hypothesis Testing](04c-hypothesis-testing_EN.md)

**Content Overview**:
- Basic concepts of hypothesis testing: null and alternative hypotheses
- Type I and Type II errors and significance level
- Meaning and use of $p$-values
- Construction and interpretation of confidence intervals
- Common testing methods: $t$-test, $\chi^2$ test
- Multiple testing and correction

**Core Concepts**:

| Concept | Definition | Practical Application |
|---------|------------|----------------------|
| Type I error | $P(\text{reject }H_0 \| H_0\text{ is true})$ | Model selection |
| $p$-value | Probability of observing more extreme results | Significance judgment |
| Confidence interval | $P(L \leq \theta \leq U) = 1-\alpha$ | Uncertainty quantification |

**[Start Learning →](04c-hypothesis-testing_EN.md)**

---

### [4.4 Regression Analysis and Bayesian Statistics](04d-regression-bayesian-statistics_EN.md)

**Content Overview**:
- Linear regression model and least squares method
- Matrix form of multiple linear regression
- Coefficient of determination $R^2$ and model evaluation
- Bias-variance tradeoff
- Bayesian statistics basics: prior, likelihood, posterior
- Bayesian linear regression
- Bayesian deep learning: MC Dropout, variational inference

**Core Concepts**:

| Concept | Formula | Deep Learning Application |
|---------|---------|--------------------------|
| Least squares solution | $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ | Linear layer solving |
| Bias-variance decomposition | $\mathbb{E}[(\hat{f}-y)^2] = \text{Bias}^2 + \text{Var} + \sigma^2$ | Overfitting/underfitting diagnosis |
| Bayes' theorem | $P(\theta\|D) \propto P(D\|\theta)P(\theta)$ | Bayesian neural networks |

**[Start Learning →](04d-regression-bayesian-statistics_EN.md)**

---

## Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│               Chapter 4: Mathematical Statistics             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  4.1 Statistics → 4.2 Parameter → 4.3 Hypothesis → 4.4      │
│  & Sampling       Estimation      Testing         Regression │
│  Distributions                              & Bayesian Stats │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Sample   │   │ Method   │   │ Type I/II│   │ Least    │ │
│  │ Mean     │   │ of       │   │ Errors   │   │ Squares  │ │
│  │ Sample   │   │ Moments  │   │ p-value  │   │ R²      │ │
│  │ Variance │   │ MLE      │   │ Confidence│   │ Bayesian │ │
│  │ χ²/t/F   │   │ Unbiased │   │ Intervals│   │          │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                             │
│  Applications: BatchNorm, Loss function design, Model       │
│  evaluation, Uncertainty quantification                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Why is Mathematical Statistics Important for Deep Learning?

### 1. Learning from Data

```
Training data → Sample statistics → Model parameters
     ↓
Statistical inference guarantees: as n → ∞, estimates → true values
```

### 2. Statistical Perspective of Model Training

| Training Component | Statistical Interpretation |
|-------------------|---------------------------|
| Loss function | Negative log-likelihood of maximum likelihood estimation |
| BatchNorm | Using sample statistics to estimate population moments |
| Regularization | Bayesian prior |
| Dropout | Sampling from Bayesian approximation |

### 3. Equivalence between MLE and Loss Functions

| Probabilistic Model | MLE-equivalent Loss Function |
|--------------------|------------------------------|
| $Y \sim \mathcal{N}(f(X), \sigma^2)$ | MSE loss |
| $Y \sim \text{Bernoulli}(\sigma(f(X)))$ | Binary cross-entropy |
| $Y \sim \text{Categorical}(\text{softmax}(f(X)))$ | Multi-class cross-entropy |

### 4. Uncertainty Quantification

```python
# Point estimate vs probabilistic estimate
prediction = model(x)  # Single point prediction
prediction_dist = bayesian_model(x)  # Prediction distribution
# Can answer: "How confident is the model?"
```

---

## Core Formula Quick Reference

### Sample Statistics

$$
\bar{X} = \frac{1}{n}\sum_{i=1}^n X_i
$$

$$
S^2 = \frac{1}{n-1}\sum_{i=1}^n(X_i - \bar{X})^2
$$

### Maximum Likelihood Estimation

$$
\hat{\theta}_{MLE} = \arg\max_\theta L(\theta) = \arg\max_\theta \prod_{i=1}^n f(x_i; \theta)
$$

### Bias-Variance Decomposition

$$
\mathbb{E}[(\hat{f}(x) - y)^2] = \text{Bias}^2[\hat{f}(x)] + \text{Var}[\hat{f}(x)] + \sigma^2
$$

### Bayes' Theorem

$$
P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)} \propto P(D|\theta)P(\theta)
$$

---

## Python Code Examples

### Statistics Calculation

```python
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Sample mean
mean = np.mean(x)

# Sample variance (unbiased)
var_unbiased = np.var(x, ddof=1)  # ddof=1 means n-1

# Sample standard deviation
std = np.std(x, ddof=1)

print(f"Sample mean: {mean}")
print(f"Sample variance (unbiased): {var_unbiased:.4f}")
```

### Maximum Likelihood Estimation

```python
import numpy as np

# Generate data from normal distribution
true_mu, true_sigma = 5.0, 2.0
np.random.seed(42)
data = np.random.normal(true_mu, true_sigma, 1000)

# MLE estimates
mu_mle = np.mean(data)
sigma2_mle = np.mean((data - mu_mle)**2)

print(f"True values: μ={true_mu}, σ²={true_sigma**2}")
print(f"MLE: μ̂={mu_mle:.3f}, σ̂²={sigma2_mle:.3f}")
```

### Confidence Intervals

```python
import numpy as np
from scipy import stats

# Generate data
data = np.random.normal(100, 15, 50)

# Calculate confidence interval
mean = np.mean(data)
se = stats.sem(data)
ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=se)

print(f"Sample mean: {mean:.2f}")
print(f"95% confidence interval: ({ci[0]:.2f}, {ci[1]:.2f})")
```

---

## Study Recommendations

1. **Understand the essence of statistical inference**: The idea of inferring populations from samples runs throughout
2. **Master MLE principles**: Understand why cross-entropy loss is equivalent to MLE
3. **Emphasize unbiasedness**: Understand why sample variance denominator is $n-1$
4. **Understand bias-variance tradeoff**: This is the core principle of model selection
5. **Connect to practice**: Think about specific manifestations of each statistical concept in deep learning

---

## Further Reading

- [Chapter 3: Probability Theory](03-probability_EN.md) - Probability distributions and information theory basics
- [Chapter 5: Optimization](05-optimization_EN.md) - Gradient descent and optimization algorithms
- [Chapter 6: Elementary Functions](06-elementary-functions_EN.md) - Activation functions and loss functions

---

**Back to**: [Mathematical Foundations Directory](../math-fundamentals_EN.md)
