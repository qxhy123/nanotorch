# Chapter 4(c): Hypothesis Testing

Hypothesis testing is another core task of statistical inference, used to determine whether a hypothesis about a population is valid based on sample data. In deep learning, hypothesis testing is used for model comparison, A/B testing, and feature selection. This chapter introduces basic concepts of hypothesis testing, common testing methods, and their applications in deep learning.

---

## 🎯 Life Analogy: A Court Trial

Hypothesis testing is like a **court trial**:

```
┌─────────────────────────────────────────────────────────────┐
│                        COURT TRIAL                          │
├─────────────────────────────────────────────────────────────┤
│  Defendant: New drug/new model/new method                   │
│  Null hypothesis H₀: Defendant innocent (no effect)         │
│  Alternative hypothesis H₁: Defendant guilty (has effect)   │
│                                                             │
│  Evidence: Sample data                                      │
│  Threshold: Significance level α = 0.05 (5% wrongful rate)  │
│                                                             │
│  Verdict:                                                   │
│  • p-value < 0.05 → Strong enough evidence → Reject H₀      │
│  • p-value ≥ 0.05 → Not enough evidence → Don't reject H₀   │
└─────────────────────────────────────────────────────────────┘
```

### Two Types of Errors = Two Kinds of Judicial Mistakes

| Error Type | Court Analogy | Statistical Meaning | Consequence |
|------------|---------------|---------------------|-------------|
| **Type I (α)** | Convicting an innocent person | Actually no effect, but we think there is | Approving a useless drug |
| **Type II (β)** | Letting a guilty person go | Actually has effect, but we miss it | Missing a good drug |

**Key principle**: In court, we'd rather let a guilty person go than convict an innocent one. So α (wrongful conviction rate) is usually controlled at 5% or 1%.

### 📖 Plain English Translation

| Statistics Term | Plain English |
|-----------------|---------------|
| Null hypothesis H₀ | "Default position": Assume nothing happened |
| Alternative hypothesis H₁ | "The accusation": Claims something happened |
| p-value | "Evidence strength": Probability of data if H₀ is true |
| Significance level α | "Verdict threshold": How strong evidence must be |
| Reject H₀ | "Guilty": Enough evidence for H₁ |
| Don't reject H₀ | "Not enough evidence": Doesn't mean H₀ is true |

---

## Table of Contents

1. [Basic Concepts of Hypothesis Testing](#basic-concepts-of-hypothesis-testing)
2. [Two Types of Errors and Significance Level](#two-types-of-errors-and-significance-level)
3. [p-value and Decision Rules](#p-value-and-decision-rules)
4. [Common Hypothesis Tests](#common-hypothesis-tests)
5. [Confidence Intervals](#confidence-intervals)
6. [Relationship Between Hypothesis Testing and Confidence Intervals](#relationship-between-hypothesis-testing-and-confidence-intervals)
7. [Applications in Deep Learning](#applications-in-deep-learning)
8. [Summary](#summary)

---

## Basic Concepts of Hypothesis Testing

### Hypothesis Testing Problem

**Problem**: Based on sample data, determine whether a hypothesis about the population holds true.

**Basic Idea**: Under the assumption that the hypothesis is true, calculate the probability of observing the current sample (or more extreme situations). If this probability is very small, there is reason to reject the hypothesis.

### Null Hypothesis and Alternative Hypothesis

**Null Hypothesis** $H_0$: The hypothesis to be tested, usually representing "no difference" or "no effect".

**Alternative Hypothesis** $H_1$ (or $H_A$): The hypothesis opposite to the null hypothesis.

### Types of Hypotheses

| Type | Null Hypothesis | Alternative Hypothesis | Meaning |
|------|-----------------|------------------------|---------|
| Two-tailed test | $H_0: \theta = \theta_0$ | $H_1: \theta \neq \theta_0$ | Whether parameter equals a value |
| Left-tailed test | $H_0: \theta \geq \theta_0$ | $H_1: \theta < \theta_0$ | Whether parameter is less than a value |
| Right-tailed test | $H_0: \theta \leq \theta_0$ | $H_1: \theta > \theta_0$ | Whether parameter is greater than a value |

### Test Statistic

**Test Statistic**: A statistic used for decision-making, with known distribution when $H_0$ is true.

$$
T = T(X_1, \ldots, X_n)
$$

### Rejection Region

**Critical Region (Rejection Region)**: The range of values of the test statistic; when the statistic falls within this range, reject $H_0$.

**Critical Value**: The boundary value of the rejection region.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Intuitive understanding of hypothesis testing
# Example: Test whether a coin is fair

# Hypothesis: H0: p = 0.5 (coin is fair)
#           H1: p ≠ 0.5 (coin is not fair)

# Observed data: flip coin 100 times, get 65 heads
n = 100
heads = 65
p_hat = heads / n

print("Coin Fairness Test:")
print("="*50)
print(f"Observed data: n={n}, heads={heads}, proportion={p_hat:.2f}")
print()

# Theoretically, if coin is fair, number of heads ~ Binomial(100, 0.5)
# Mean = 50, Standard deviation = sqrt(100*0.5*0.5) = 5

# Standardization
z_stat = (heads - 50) / 5

# Two-tailed p-value
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print(f"Test statistic z = {z_stat:.2f}")
print(f"p-value = {p_value:.4f}")
print(f"Conclusion (α=0.05): {'Reject H0' if p_value < 0.05 else 'Do not reject H0'}")
print()

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

x = np.linspace(30, 70, 200)
# Normal approximation
pdf = stats.norm.pdf(x, 50, 5)

ax.plot(x, pdf, 'b-', linewidth=2, label='Distribution under H0: N(50, 25)')
ax.fill_between(x[x >= 60], pdf[x >= 60], alpha=0.3, color='red', label='Right-tailed rejection region')
ax.fill_between(x[x <= 40], pdf[x <= 40], alpha=0.3, color='red', label='Left-tailed rejection region')
ax.axvline(heads, color='green', linestyle='--', linewidth=2, label=f'Observed value: {heads}')
ax.axvline(60, color='red', linestyle=':', linewidth=1.5)
ax.axvline(40, color='red', linestyle=':', linewidth=1.5)

ax.set_xlabel('Number of heads')
ax.set_ylabel('Probability Density')
ax.set_title('Two-Tailed Hypothesis Test: Is the Coin Fair?')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hypothesis_test_intuition.png', dpi=100)
print("Image saved: hypothesis_test_intuition.png")
```

---

## Two Types of Errors and Significance Level

### Two Types of Errors

|  | $H_0$ is True | $H_0$ is False |
|--|---------------|----------------|
| **Reject $H_0$** | **Type I Error** | Correct |
| **Do Not Reject $H_0$** | Correct | **Type II Error** |

**Type I Error** (False Positive): $H_0$ is actually true, but is rejected.

$$
\alpha = P(\text{Reject } H_0 | H_0 \text{ is true})
$$

**Type II Error** (False Negative): $H_0$ is actually false, but is not rejected.

$$
\beta = P(\text{Do Not Reject } H_0 | H_0 \text{ is false})
$$

### Significance Level

**Significance Level** $\alpha$: The maximum allowed probability of making a Type I error.

Common values: $\alpha = 0.05$, $\alpha = 0.01$, $\alpha = 0.1$

### Power

**Test Power**: Probability of correctly rejecting a false null hypothesis.

$$
\text{Power} = 1 - \beta = P(\text{Reject } H_0 | H_1 \text{ is true})
$$

### Trade-off Between Two Types of Errors

- Reducing $\alpha$ increases $\beta$ (with other conditions unchanged)
- The only way to reduce both types of errors simultaneously is to increase sample size

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Visualization of two types of errors
fig, ax = plt.subplots(1, 1, figsize=(12, 6))

# H0: μ = 0
# H1: μ = 2
mu0, mu1 = 0, 2
sigma = 1
n = 25  # Sample size
se = sigma / np.sqrt(n)  # Standard error

x = np.linspace(-2, 5, 200)

# Distribution under H0
y0 = stats.norm.pdf(x, mu0, se)
# Distribution under H1
y1 = stats.norm.pdf(x, mu1, se)

ax.plot(x, y0, 'b-', linewidth=2, label=f'H0: N({mu0}, {se**2:.2f})')
ax.plot(x, y1, 'r-', linewidth=2, label=f'H1: N({mu1}, {se**2:.2f})')

# Critical value (right-tailed test, α = 0.05)
alpha = 0.05
critical_value = stats.norm.ppf(1 - alpha, mu0, se)

# Fill Type I error region
x_alpha = x[x >= critical_value]
ax.fill_between(x_alpha, stats.norm.pdf(x_alpha, mu0, se), alpha=0.3, color='blue', label=f'Type I error α = {alpha}')

# Fill Type II error region
x_beta = x[x < critical_value]
ax.fill_between(x_beta, stats.norm.pdf(x_beta, mu1, se), alpha=0.3, color='red', label=f'Type II error β')

# Critical value line
ax.axvline(critical_value, color='green', linestyle='--', linewidth=2, label=f'Critical value = {critical_value:.2f}')

# Calculate β
beta = stats.norm.cdf(critical_value, mu1, se)
power = 1 - beta

ax.set_xlabel('Sample mean')
ax.set_ylabel('Probability Density')
ax.set_title(f'Two Types of Errors Visualization (α={alpha}, β={beta:.4f}, Power={power:.4f})')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('type_errors.png', dpi=100)
print("Image saved: type_errors.png")

print(f"\nTwo Types of Errors:")
print(f"Significance level α = {alpha}")
print(f"Type II error β = {beta:.4f}")
print(f"Test power = {power:.4f}")
```

---

## p-value and Decision Rules

### Definition of p-value

**p-value**: Under the condition that $H_0$ is true, the probability of observing a result **more extreme** than the current result.

$$
p\text{-value} = P(T \geq t_{obs} | H_0)
$$

### Interpretation of p-value

| p-value Range | Interpretation |
|---------------|----------------|
| $p < 0.01$ | Very strong evidence to reject $H_0$ |
| $0.01 \leq p < 0.05$ | Strong evidence to reject $H_0$ |
| $0.05 \leq p < 0.1$ | Weak evidence to reject $H_0$ |
| $p \geq 0.1$ | Insufficient evidence to reject $H_0$ |

### Decision Rules

**Using p-value**:
- If $p < \alpha$: Reject $H_0$
- If $p \geq \alpha$: Do not reject $H_0$

**Using critical value**:
- If the test statistic falls within the rejection region: Reject $H_0$
- Otherwise: Do not reject $H_0$

### p-value Calculation

**Two-tailed test**:

$$
p = 2 \cdot P(Z \geq |z_{obs}|)
$$

**One-tailed test (right)**:

$$
p = P(Z \geq z_{obs})
$$

**One-tailed test (left)**:

$$
p = P(Z \leq z_{obs})
$$

```python
import numpy as np
from scipy import stats

# p-value calculation example
np.random.seed(42)

# One-sample t-test
# H0: μ = 100
# H1: μ ≠ 100

mu0 = 100
sample = np.random.normal(102, 15, 50)  # True μ = 102
n = len(sample)

# Calculate test statistic
sample_mean = sample.mean()
sample_std = sample.std(ddof=1)
se = sample_std / np.sqrt(n)

t_stat = (sample_mean - mu0) / se
df = n - 1

# p-value (two-tailed)
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

print("One-Sample t-Test:")
print("="*50)
print(f"H0: μ = {mu0}")
print(f"Sample mean: {sample_mean:.2f}")
print(f"Sample standard deviation: {sample_std:.2f}")
print(f"Standard error: {se:.2f}")
print(f"t statistic: {t_stat:.4f}")
print(f"Degrees of freedom: {df}")
print(f"p-value: {p_value:.4f}")
print()
print(f"Conclusion (α=0.05): {'Reject H0' if p_value < 0.05 else 'Do not reject H0'}")

# Verify with scipy
t_stat_scipy, p_value_scipy = stats.ttest_1samp(sample, mu0)
print(f"\nscipy verification:")
print(f"t statistic: {t_stat_scipy:.4f}")
print(f"p-value: {p_value_scipy:.4f}")
```

---

## Common Hypothesis Tests

### One-Sample t-Test

**Problem**: Test whether the sample mean equals a specific value.

**Hypotheses**:
- $H_0: \mu = \mu_0$
- $H_1: \mu \neq \mu_0$

**Test Statistic**:

$$
t = \frac{\bar{X} - \mu_0}{S / \sqrt{n}} \sim t(n-1)
$$

### Two-Sample t-Test

**Problem**: Test whether the means of two independent samples are equal.

**Hypotheses**:
- $H_0: \mu_1 = \mu_2$
- $H_1: \mu_1 \neq \mu_2$

**Test Statistic** (when variances are equal):

$$
t = \frac{\bar{X}_1 - \bar{X}_2}{S_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}
$$

where $S_p^2 = \frac{(n_1-1)S_1^2 + (n_2-1)S_2^2}{n_1+n_2-2}$ is the **pooled variance**.

### Paired t-Test

**Problem**: Test whether the mean of paired sample differences is zero.

**Test Statistic**:

$$
t = \frac{\bar{D}}{S_D / \sqrt{n}}
$$

where $D_i = X_i - Y_i$.

### χ² Test

**Problem**: Test the independence of categorical variables.

**Chi-squared statistic**:

$$
\chi^2 = \sum \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
$$

where $O_{ij}$ are observed frequencies and $E_{ij}$ are expected frequencies.

### F Test (Variance Comparison)

**Problem**: Test whether the variances of two populations are equal.

**Test Statistic**:

$$
F = \frac{S_1^2}{S_2^2} \sim F(n_1-1, n_2-1)
$$

```python
import numpy as np
from scipy import stats

# Various hypothesis test examples

print("="*60)
print("1. Two-Sample t-Test (Independent Samples)")
print("="*60)

np.random.seed(42)
group1 = np.random.normal(100, 15, 50)
group2 = np.random.normal(105, 15, 50)

t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"Group 1 mean: {group1.mean():.2f}")
print(f"Group 2 mean: {group2.mean():.2f}")
print(f"t statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Conclusion: {'Significant difference' if p_value < 0.05 else 'No significant difference'}")

print("\n" + "="*60)
print("2. Paired t-Test")
print("="*60)

# Before and after measurements from the same subjects
before = np.random.normal(100, 15, 30)
after = before + np.random.normal(5, 5, 30)  # Average improvement of 5

t_stat, p_value = stats.ttest_rel(before, after)
print(f"Before mean: {before.mean():.2f}")
print(f"After mean: {after.mean():.2f}")
print(f"Average change: {(after - before).mean():.2f}")
print(f"t statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Conclusion: {'Significant change' if p_value < 0.05 else 'No significant change'}")

print("\n" + "="*60)
print("3. χ² Test (Independence Test)")
print("="*60)

# Contingency table
observed = np.array([[50, 30], [20, 40]])
chi2, p_value, dof, expected = stats.chi2_contingency(observed)

print(f"Observed frequencies:\n{observed}")
print(f"Expected frequencies:\n{expected}")
print(f"χ² statistic: {chi2:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"p-value: {p_value:.4f}")
print(f"Conclusion: {'Dependent' if p_value < 0.05 else 'Independent'}")
```

---

## Confidence Intervals

### Definition

A **confidence interval** with confidence level $1-\alpha$ for parameter $\theta$ is a random interval $(L, U)$ such that:

$$
P(L \leq \theta \leq U) = 1 - \alpha
$$

### Interpretation

A confidence interval **does not mean** "the parameter has a $1-\alpha$ probability of falling within the interval".

Correct interpretation: If we repeat sampling many times, approximately $(1-\alpha) \times 100\%$ of the confidence intervals will contain the true parameter.

### Confidence Interval for Mean of Normal Population

#### σ Known

$$
\left(\bar{X} - z_{\alpha/2} \frac{\sigma}{\sqrt{n}}, \bar{X} + z_{\alpha/2} \frac{\sigma}{\sqrt{n}}\right)
$$

#### σ Unknown

$$
\left(\bar{X} - t_{\alpha/2}(n-1) \frac{S}{\sqrt{n}}, \bar{X} + t_{\alpha/2}(n-1) \frac{S}{\sqrt{n}}\right)
$$

### Confidence Interval for Proportion

$$
\left(\hat{p} - z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}, \hat{p} + z_{\alpha/2}\sqrt{\frac{\hat{p}(1-\hat{p})}{n}}\right)
$$

### Confidence Interval for Variance

$$
\left(\frac{(n-1)S^2}{\chi^2_{\alpha/2}(n-1)}, \frac{(n-1)S^2}{\chi^2_{1-\alpha/2}(n-1)}\right)
$$

```python
import numpy as np
from scipy import stats

# Confidence interval calculation example
np.random.seed(42)

# Generate data
sample = np.random.normal(100, 15, 50)
n = len(sample)
sample_mean = sample.mean()
sample_std = sample.std(ddof=1)

print("Confidence Interval Calculation:")
print("="*50)
print(f"Sample size: n = {n}")
print(f"Sample mean: {sample_mean:.2f}")
print(f"Sample standard deviation: {sample_std:.2f}")
print()

# 1. Confidence interval for mean (σ unknown)
confidence = 0.95
se = sample_std / np.sqrt(n)
ci_mean = stats.t.interval(confidence, n-1, loc=sample_mean, scale=se)
print(f"Mean {confidence*100:.0f}% confidence interval: ({ci_mean[0]:.2f}, {ci_mean[1]:.2f})")

# 2. Confidence interval for variance
alpha = 1 - confidence
chi2_upper = stats.chi2.ppf(1 - alpha/2, n-1)
chi2_lower = stats.chi2.ppf(alpha/2, n-1)
ci_var = ((n-1) * sample_std**2 / chi2_upper, (n-1) * sample_std**2 / chi2_lower)
print(f"Variance {confidence*100:.0f}% confidence interval: ({ci_var[0]:.2f}, {ci_var[1]:.2f})")
print(f"Standard deviation {confidence*100:.0f}% confidence interval: ({np.sqrt(ci_var[0]):.2f}, {np.sqrt(ci_var[1]):.2f})")

# 3. Confidence interval for proportion
# Example: 35 successes out of 100 trials
n_trials = 100
n_success = 35
p_hat = n_success / n_trials
se_prop = np.sqrt(p_hat * (1 - p_hat) / n_trials)
ci_prop = stats.norm.interval(confidence, loc=p_hat, scale=se_prop)
print(f"\nProportion {confidence*100:.0f}% confidence interval: ({ci_prop[0]:.4f}, {ci_prop[1]:.4f})")

# Visualization of confidence intervals from multiple samples
print("\n" + "="*50)
print("Frequency Interpretation of Confidence Intervals (Repeated Sampling)")

n_simulations = 100
true_mu = 100
true_sigma = 15
sample_size = 30

contains_true = 0
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

for i in range(n_simulations):
    sample_i = np.random.normal(true_mu, true_sigma, sample_size)
    mean_i = sample_i.mean()
    std_i = sample_i.std(ddof=1)
    se_i = std_i / np.sqrt(sample_size)
    ci_i = stats.t.interval(0.95, sample_size-1, loc=mean_i, scale=se_i)

    contains = (ci_i[0] <= true_mu <= ci_i[1])
    if contains:
        contains_true += 1
        color = 'blue'
    else:
        color = 'red'

    ax.plot([ci_i[0], ci_i[1]], [i, i], color=color, linewidth=0.5)
    ax.plot(mean_i, i, 'o', color=color, markersize=2)

ax.axvline(true_mu, color='green', linestyle='--', linewidth=2, label=f'True mean = {true_mu}')
ax.set_xlabel('Value')
ax.set_ylabel('Sample number')
ax.set_title(f'Frequency Interpretation of Confidence Intervals\n{contains_true} out of {n_simulations} intervals contain true value ({contains_true/n_simulations*100:.1f}%)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('confidence_intervals.png', dpi=100)
print(f"\nOut of {n_simulations} samples, {contains_true} confidence intervals contain the true value")
print(f"Proportion: {contains_true/n_simulations*100:.1f}% (theoretical value: 95%)")
print("Image saved: confidence_intervals.png")
```

---

## Relationship Between Hypothesis Testing and Confidence Intervals

### Duality Relationship

**Two-tailed hypothesis testing** and **confidence intervals** have a **duality relationship**:

$$
\text{Reject } H_0: \theta = \theta_0 \iff \theta_0 \notin \text{confidence interval}
$$

### Example

For mean testing $H_0: \mu = \mu_0$:
- If $\mu_0$ falls outside the $(1-\alpha)$ confidence interval, reject $H_0$ at significance level $\alpha$
- If $\mu_0$ falls within the $(1-\alpha)$ confidence interval, do not reject $H_0$ at significance level $\alpha$

```python
import numpy as np
from scipy import stats

# Duality relationship between hypothesis testing and confidence intervals
np.random.seed(42)
sample = np.random.normal(102, 15, 50)
n = len(sample)

# Method 1: Hypothesis testing
mu0 = 100
t_stat, p_value = stats.ttest_1samp(sample, mu0)

# Method 2: Confidence interval
sample_mean = sample.mean()
sample_std = sample.std(ddof=1)
se = sample_std / np.sqrt(n)
ci = stats.t.interval(0.95, n-1, loc=sample_mean, scale=se)

print("Duality Relationship Between Hypothesis Testing and Confidence Intervals:")
print("="*50)
print(f"H0: μ = {mu0}")
print(f"Sample mean: {sample_mean:.2f}")
print(f"95% confidence interval: ({ci[0]:.2f}, {ci[1]:.2f})")
print()
print("Method 1 - Hypothesis Testing:")
print(f"  t statistic: {t_stat:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Conclusion (α=0.05): {'Reject H0' if p_value < 0.05 else 'Do not reject H0'}")
print()
print("Method 2 - Confidence Interval:")
print(f"  Is {mu0} within the confidence interval? {ci[0] <= mu0 <= ci[1]}")
print(f"  Conclusion: {'Do not reject H0' if ci[0] <= mu0 <= ci[1] else 'Reject H0'}")
print()
print("Both methods give the same conclusion!")
```

---

## Applications in Deep Learning

### 1. Model Comparison

Use **paired t-test** to compare performance differences of two models on multiple datasets or multiple runs.

```python
import numpy as np
from scipy import stats

# Model comparison example
np.random.seed(42)
n_folds = 10

# Accuracies of models A and B on K-fold cross-validation
model_a_acc = np.array([0.85, 0.87, 0.84, 0.86, 0.88, 0.85, 0.87, 0.86, 0.85, 0.87])
model_b_acc = np.array([0.82, 0.84, 0.81, 0.83, 0.85, 0.82, 0.84, 0.83, 0.82, 0.84])

# Paired t-test
t_stat, p_value = stats.ttest_rel(model_a_acc, model_b_acc)

print("Model Comparison (Paired t-Test):")
print("="*50)
print(f"Model A average accuracy: {model_a_acc.mean():.4f} ± {model_a_acc.std():.4f}")
print(f"Model B average accuracy: {model_b_acc.mean():.4f} ± {model_b_acc.std():.4f}")
print(f"Average difference: {(model_a_acc - model_b_acc).mean():.4f}")
print()
print(f"t statistic: {t_stat:.4f}")
print(f"p-value: {p_value:.6f}")
print(f"Conclusion: {'Model A significantly better than B' if p_value < 0.05 else 'No significant difference'} (α=0.05)")

# Effect size (Cohen's d)
diff = model_a_acc - model_b_acc
cohens_d = diff.mean() / diff.std()
print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
print(f"Effect size magnitude: {'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'}")
```

### 2. A/B Testing

```python
import numpy as np
from scipy import stats

# A/B testing example
np.random.seed(42)

# Control group A and experimental group B
n_A = 1000
n_B = 1000

# Click-through rates
clicks_A = 100  # 10% click-through rate
clicks_B = 130  # 13% click-through rate

p_A = clicks_A / n_A
p_B = clicks_B / n_B

# Two-sample proportion test
# Pooled proportion
p_pooled = (clicks_A + clicks_B) / (n_A + n_B)
se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_A + 1/n_B))

z_stat = (p_B - p_A) / se_pooled
p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

print("A/B Testing:")
print("="*50)
print(f"Control group A: {clicks_A}/{n_A} = {p_A:.2%}")
print(f"Experimental group B: {clicks_B}/{n_B} = {p_B:.2%}")
print(f"Difference: {(p_B - p_A):.2%}")
print()
print(f"z statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4f}")
print(f"Conclusion: {'Experimental group significantly better than control' if p_value < 0.05 else 'No significant difference'} (α=0.05)")

# Confidence interval
diff = p_B - p_A
se_diff = np.sqrt(p_A*(1-p_A)/n_A + p_B*(1-p_B)/n_B)
ci_diff = stats.norm.interval(0.95, loc=diff, scale=se_diff)
print(f"\n95% confidence interval for difference: ({ci_diff[0]:.4f}, {ci_diff[1]:.4f})")
```

### 3. Feature Selection

Use statistical tests to select important features.

```python
import numpy as np
from scipy import stats

# Feature selection example
np.random.seed(42)
n_samples = 100
n_features = 10

# Generate feature data
X = np.random.randn(n_samples, n_features)
# Target variable (correlated with some features)
y = 0.5 * X[:, 0] + 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.5

# Perform t-test on each feature (correlation with target)
print("Feature Selection (Correlation Test):")
print("="*50)
print(f"{'Feature':<10} {'Correlation':<12} {'p-value':<12} {'Keep?'}")
print("-"*50)

selected_features = []
for i in range(n_features):
    # Pearson correlation test
    corr, p_value = stats.pearsonr(X[:, i], y)
    keep = p_value < 0.05
    if keep:
        selected_features.append(i)
    print(f"Feature {i:<5} {corr:>10.4f} {p_value:>10.4f} {'Yes' if keep else 'No'}")

print()
print(f"Selected features: {selected_features}")
print(f"(True correlated features: 0, 2)")
```

---

## Summary

This chapter introduced the basic concepts and methods of hypothesis testing, which are the theoretical foundation for conducting scientific experiments and model comparison.

### Core Concept Comparison Table

| Concept | Definition | Application |
|---------|------------|-------------|
| Type I Error | $P(\text{Reject } H_0 | H_0 \text{ is true})$ | Significance level |
| Type II Error | $P(\text{Do Not Reject } H_0 | H_1 \text{ is true})$ | Test power |
| p-value | Probability of observing a more extreme result | Decision basis |
| Confidence Interval | Reliable range of parameters | Parameter estimation |

### Common Test Methods

| Test | Applicable Scenario | Test Statistic Distribution |
|------|---------------------|----------------------------|
| One-sample t-test | Mean testing | $t(n-1)$ |
| Two-sample t-test | Comparison of two group means | $t(n_1+n_2-2)$ |
| Paired t-test | Paired sample comparison | $t(n-1)$ |
| χ² test | Independence of categorical variables | $\chi^2$ |
| F test | Variance comparison | $F$ |

### Key Points

1. **Hypothesis Testing**: Statistical method for inferring populations from samples
2. **Two Types of Errors**: Need to balance them
3. **p-value**: Not the probability that the hypothesis is true
4. **Confidence Intervals**: Dual to hypothesis testing
5. **Deep Learning Applications**: Model comparison, A/B testing, feature selection

---

**Previous Section**: [Chapter 4(b): Parameter Estimation](04b-parameter-estimation_EN.md)

**Next Section**: [Chapter 4(d): Regression Analysis and Bayesian Statistics](04d-regression-bayesian-statistics_EN.md) - Learn about linear regression, logistic regression, and Bayesian methods in deep learning.

**Return**: [Mathematics Fundamentals Tutorial Directory](../math-fundamentals.md)
