# Chapter 3 (c): Multivariate Random Variables and Numerical Characteristics

In practical applications, we often need to handle multiple random variables simultaneously. This chapter will introduce the joint distribution, marginal distribution, conditional distribution of multivariate random variables, as well as numerical characteristics describing relationships between random variables—covariance and correlation coefficients. These concepts have important applications in deep learning's feature engineering, Batch Normalization, and covariance matrix calculations.

---

## 🎯 Life Analogy: Height and Weight Together

Imagine collecting data on people's height AND weight:

| Person | Height (cm) | Weight (kg) |
|--------|-------------|-------------|
| Alice | 165 | 55 |
| Bob | 180 | 75 |
| Carol | 170 | 65 |
| Dave | 175 | 70 |
| Eve | 160 | 50 |

**Joint distribution**: How height and weight vary TOGETHER
**Marginal distribution**: Looking at just height (or just weight) alone
**Covariance**: Do taller people tend to be heavier? (Yes → positive covariance)

### 📝 Step-by-Step Covariance Calculation

**Data**: 5 people's heights and weights

| $X$ (Height) | $Y$ (Weight) | $X - \bar{X}$ | $Y - \bar{Y}$ | $(X-\bar{X})(Y-\bar{Y})$ |
|-------------|-------------|---------------|---------------|-------------------------|
| 165 | 55 | -5 | -10 | 50 |
| 180 | 75 | +10 | +10 | 100 |
| 170 | 65 | 0 | 0 | 0 |
| 175 | 70 | +5 | +5 | 25 |
| 160 | 50 | -10 | -15 | 150 |

**Means**: $\bar{X} = 170$, $\bar{Y} = 65$

**Covariance**: $\text{Cov}(X,Y) = \frac{50+100+0+25+150}{5} = \frac{325}{5} = 65$

**Interpretation**: Positive! Taller people tend to be heavier (as expected).

### Correlation = "Standardized Covariance"

$$\text{Correlation} = \frac{\text{Covariance}}{\text{StdDev}(X) \times \text{StdDev}(Y)}$$

| Correlation | Meaning |
|-------------|---------|
| +1 | Perfect positive relationship |
| 0 | No linear relationship |
| -1 | Perfect negative relationship |

### 📖 Plain English Translation

| Math Term | Plain English |
|-----------|---------------|
| Joint distribution | How multiple variables vary together |
| Marginal distribution | Looking at one variable, ignoring others |
| Covariance | Do they increase together? |
| Correlation | How strongly are they linearly related? (-1 to +1) |

---

## Table of Contents

1. [Overview of Multivariate Random Variables](#overview-of-multivariate-random-variables)
2. [Joint Distribution](#joint-distribution)
3. [Marginal Distribution](#marginal-distribution)
4. [Conditional Distribution](#conditional-distribution)
5. [Independence](#independence)
6. [Properties of Expectation](#properties-of-expectation)
7. [Covariance](#covariance)
8. [Correlation Coefficient](#correlation-coefficient)
9. [Covariance Matrix](#covariance-matrix)
10. [Higher Moments: Skewness and Kurtosis](#higher-moments-skewness-and-kurtosis)
11. [Applications in Deep Learning](#applications-in-deep-learning)
12. [Summary](#summary)

---

## Overview of Multivariate Random Variables

### Definition

**Multivariate random variables** (random vectors) are vectors composed of multiple random variables:

$$
\mathbf{X} = (X_1, X_2, \ldots, X_n)^\top
$$

### Examples

| Scenario | Random Variables |
|------|----------|
| Image classification | Pixel values $(X_1, X_2, \ldots, X_{784})$ |
| Natural language | Word vectors $(X_1, X_2, \ldots, X_d)$ |
| Time series | State at time t $(X_t, Y_t, Z_t)$ |
| Multi-label classification | Prediction probabilities for each label $(p_1, p_2, \ldots, p_K)$ |

### Why Are They Important?

Deep learning essentially deals with **high-dimensional random variables**:
- **Input layer**: Each feature is a random variable
- **Hidden layers**: Activation values are functions of random variables
- **Output layer**: Prediction probabilities are parameters of joint distribution

```python
import numpy as np

# Multivariate random variable example: RGB channels of an image
# Assume each 32x32 image has 3 channels
image = np.random.randint(0, 256, (32, 32, 3))  # RGB image

# Flatten to random vector
random_vector = image.flatten()
print(f"Image shape: {image.shape}")
print(f"Random vector dimension: {random_vector.shape}")
print(f"First 10 values: {random_vector[:10]}")

# Batch images → random matrix
batch_size = 64
batch_images = np.random.randint(0, 256, (batch_size, 32, 32, 3))
batch_vectors = batch_images.reshape(batch_size, -1)
print(f"\nBatch images: {batch_images.shape}")
print(f"Batch random vectors: {batch_vectors.shape}")
```

---

## Joint Distribution

### Definition

**Joint distribution** describes the probability distribution of multiple random variables simultaneously taking values.

### Discrete Case

For discrete random variables $X$ and $Y$, the **joint probability mass function (Joint PMF)**:

$$
p(x, y) = P(X = x, Y = y)
$$

**Properties**:
- $p(x$, $y) \geq 0$
- $\displaystyle\sum_x \sum_y p(x$, $y) = 1$

### Continuous Case

For continuous random variables $X$ and $Y$, the **joint probability density function (Joint PDF)**:

$$
P((X, Y) \in A) = \iint_A f(x, y) \, dx \, dy
$$

**Properties**:
- $f(x$, $y) \geq 0$
- $\displaystyle\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f(x$, $y) \$, $dx \$, $dy = 1$

### Bivariate Normal Distribution

The most important bivariate continuous distribution:

$$
f(x, y) = \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1-\rho^2}} \exp\left(-\frac{1}{2(1-\rho^2)}\left[\frac{(x-\mu_X)^2}{\sigma_X^2} - \frac{2\rho(x-\mu_X)(y-\mu_Y)}{\sigma_X\sigma_Y} + \frac{(y-\mu_Y)^2}{\sigma_Y^2}\right]\right)
$$

Where $\rho$ is the correlation coefficient.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Bivariate normal distribution
mu = [0, 0]
cov = [[1, 0.5], [0.5, 1]]  # Covariance matrix

# Create grid
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))

# Calculate joint PDF
rv = multivariate_normal(mu, cov)
pdf = rv.pdf(pos)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Contour plot
contour = axes[0].contourf(x, y, pdf, levels=20, cmap='viridis')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_title('Bivariate Normal Distribution Joint PDF (Contour)')
plt.colorbar(contour, ax=axes[0])

# 3D surface plot
ax3d = fig.add_subplot(122, projection='3d')
ax3d.plot_surface(x, y, pdf, cmap='viridis', alpha=0.8)
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('f(x,y)')
ax3d.set_title('Bivariate Normal Distribution 3D View')

plt.tight_layout()
plt.savefig('joint_distribution.png', dpi=100)
print("Joint distribution image saved")

# Discrete joint distribution example
print("\nDiscrete joint distribution example:")
print("="*40)

# Roll two dice
die1 = [1, 2, 3, 4, 5, 6]
die2 = [1, 2, 3, 4, 5, 6]

# Joint PMF table (each combination has probability 1/36)
print("P(X=x, Y=y) table (partial):")
print(f"P(1,1) = 1/36 ≈ {1/36:.4f}")
print(f"P(1,2) = 1/36 ≈ {1/36:.4f}")
print(f"P(6,6) = 1/36 ≈ {1/36:.4f}")
print(f"Sum of all probabilities: {6*6 * 1/36:.1f}")
```

---

## Marginal Distribution

### Definition

**Marginal distribution** is the distribution of a single random variable "extracted" from the joint distribution, obtained by "marginalizing" (summing or integrating) over other variables.

### Discrete Case

$$
p_X(x) = \sum_y p(x, y)
$$

$$
p_Y(y) = \sum_x p(x, y)
$$

### Continuous Case

$$
f_X(x) = \int_{-\infty}^{\infty} f(x, y) \, dy
$$

$$
f_Y(y) = \int_{-\infty}^{\infty} f(x, y) \, dx
$$

### Intuitive Understanding

Imagine the joint distribution as a table, marginal distributions are the "edge" totals obtained by summing rows/columns.

```python
import numpy as np
import pandas as pd

# Discrete joint distribution example
# Joint distribution of two dice
joint_pmf = np.ones((6, 6)) / 36  # Each cell is 1/36

# Calculate marginal distributions
marginal_X = joint_pmf.sum(axis=1)  # Sum by columns
marginal_Y = joint_pmf.sum(axis=0)  # Sum by rows

print("Joint distribution table (die1 × die2):")
df = pd.DataFrame(joint_pmf,
                  index=[f'X={i}' for i in range(1, 7)],
                  columns=[f'Y={i}' for i in range(1, 7)])
print(df.round(4))
print(f"\nX marginal distribution: {marginal_X}")
print(f"Y marginal distribution: {marginal_Y}")

# Verify normalization
print(f"\nX marginal distribution sum: {marginal_X.sum():.4f}")
print(f"Y marginal distribution sum: {marginal_Y.sum():.4f}")

# Another example: non-independent joint distribution
print("\n" + "="*50)
print("Non-independent joint distribution example:")

# Assume joint distribution
joint_custom = np.array([
    [0.1, 0.1, 0.05],
    [0.15, 0.2, 0.1],
    [0.1, 0.15, 0.05]
])

marginal_X_custom = joint_custom.sum(axis=1)
marginal_Y_custom = joint_custom.sum(axis=0)

print("Joint PMF:")
print(joint_custom)
print(f"\nX marginal distribution: {marginal_X_custom}")
print(f"Y marginal distribution: {marginal_Y_custom}")
```

---

## Conditional Distribution

### Definition

**Conditional distribution** is the probability distribution of one variable given the value of another variable.

### Conditional PMF (Discrete)

$$
p_{Y|X}(y|x) = \frac{p(x, y)}{p_X(x)}, \quad p_X(x) > 0
$$

### Conditional PDF (Continuous)

$$
f_{Y|X}(y|x) = \frac{f(x, y)}{f_X(x)}, \quad f_X(x) > 0
$$

### Distribution Form of Bayes' Formula

$$
f_{X|Y}(x|y) = \frac{f_{Y|X}(y|x) f_X(x)}{f_Y(y)}
$$

### Conditional Expectation

$$
\mathbb{E}[Y|X = x] = \sum_y y \cdot p_{Y|X}(y|x) \quad \text{(discrete)}
$$

$$
\mathbb{E}[Y|X = x] = \int_{-\infty}^{\infty} y \cdot f_{Y|X}(y|x) \, dy \quad \text{(continuous)}
$$

```python
import numpy as np

# Conditional distribution example
# Joint distribution
joint = np.array([
    [0.1, 0.1, 0.05],   # X=0
    [0.15, 0.2, 0.1],   # X=1
    [0.1, 0.15, 0.05]   # X=2
])

# Marginal distributions
marginal_X = joint.sum(axis=1)
marginal_Y = joint.sum(axis=0)

print("Conditional distribution calculation:")
print("="*40)

# Conditional distribution P(Y|X=1)
x_val = 1
conditional_Y_given_X1 = joint[x_val, :] / marginal_X[x_val]
print(f"Marginal distribution P(X={x_val}) = {marginal_X[x_val]:.2f}")
print(f"Conditional distribution P(Y|X={x_val}): {conditional_Y_given_X1}")
print(f"Conditional distribution sum: {conditional_Y_given_X1.sum():.4f}")

# Calculate conditional expectation E[Y|X=1]
y_values = np.array([0, 1, 2])
conditional_expectation = np.sum(y_values * conditional_Y_given_X1)
print(f"Conditional expectation E[Y|X={x_val}] = {conditional_expectation:.4f}")

# All conditional expectations
print("\nAll conditional expectations:")
for x in range(3):
    cond_dist = joint[x, :] / marginal_X[x]
    cond_exp = np.sum(y_values * cond_dist)
    print(f"E[Y|X={x}] = {cond_exp:.4f}")
```

---

## Independence

### Definition

Random variables $X$ and $Y$ are **independent** if and only if the joint distribution equals the product of marginal distributions:

$$
f(x, y) = f_X(x) \cdot f_Y(y)
$$

Or equivalently:

$$
P(X \in A, Y \in B) = P(X \in A) \cdot P(Y \in B)
$$

### Equivalent Conditions for Independence

The following conditions are equivalent:
1. $f(x, y) = f_X(x) f_Y(y)$
2. $F(x, y) = F_X(x) F_Y(y)$ (joint CDF = product of marginal CDFs)
3. $f_{Y|X}(y|x) = f_Y(y)$ (conditional distribution = marginal distribution)
4. $\mathbb{E}[g(X)h(Y)] = \mathbb{E}[g(X)] \mathbb{E}[h(Y)]$ (for all functions g, h)

### Independent vs Uncorrelated

| Concept | Definition | Strength |
|------|------|------|
| Independent | $f(x,y) = f_X(x)f_Y(y)$ | Strong |
| Uncorrelated | $\text{Cov}(X,Y) = 0$ | Weak |

**Important conclusions**:
- Independent ⇒ uncorrelated
- Uncorrelated ⇏ independent

```python
import numpy as np

# Independence test example
print("Independence test:")
print("="*50)

# Example 1: Independent joint distribution
joint_independent = np.outer([0.3, 0.5, 0.2], [0.4, 0.6])
marginal_X = joint_independent.sum(axis=1)
marginal_Y = joint_independent.sum(axis=0)

product = np.outer(marginal_X, marginal_Y)

print("Example 1: Test independence")
print(f"Joint distribution:\n{joint_independent}")
print(f"\nProduct of marginal distributions:\n{product}")
print(f"Difference: {np.abs(joint_independent - product).sum():.10f}")
print(f"Independent? {np.allclose(joint_independent, product)}")

# Example 2: Dependent joint distribution
joint_dependent = np.array([
    [0.1, 0.1, 0.05],
    [0.15, 0.2, 0.1],
    [0.1, 0.15, 0.05]
])
marginal_X2 = joint_dependent.sum(axis=1)
marginal_Y2 = joint_dependent.sum(axis=0)
product2 = np.outer(marginal_X2, marginal_Y2)

print("\nExample 2: Test independence")
print(f"Joint distribution:\n{joint_dependent}")
print(f"\nProduct of marginal distributions:\n{product2}")
print(f"Difference: {np.abs(joint_dependent - product2).sum():.4f}")
print(f"Independent? {np.allclose(joint_dependent, product2)}")

# Classic example of uncorrelated but not independent
print("\n" + "="*50)
print("Uncorrelated but not independent example: X ~ Uniform(-1,1), Y = X²")

n = 100000
X = np.random.uniform(-1, 1, n)
Y = X**2

# Calculate covariance
cov_XY = np.cov(X, Y)[0, 1]
corr_XY = np.corrcoef(X, Y)[0, 1]

print(f"Cov(X, Y) = {cov_XY:.6f}")
print(f"Corr(X, Y) = {corr_XY:.6f}")
print(f"Uncorrelated? {abs(corr_XY) < 0.01}")
print(f"Independent? False (Y completely determined by X: Y = X²)")
```

---

## Properties of Expectation

### Review of Basic Properties

**Linearity of expectation** (most important property):

$$
\mathbb{E}[aX + bY] = a\mathbb{E}[X] + b\mathbb{E}[Y]
$$

**Expectation of constant**:

$$
\mathbb{E}[c] = c
$$

### Expectation of Functions

$$
\mathbb{E}[g(X)] = \sum_x g(x) \cdot p(x) \quad \text{(discrete)}
$$

$$
\mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot f(x) \, dx \quad \text{(continuous)}
$$

### Moments

**k-th raw moment**:

$$
\mathbb{E}[X^k]
$$

**k-th central moment**:

$$
\mathbb{E}[(X - \mathbb{E}[X])^k]
$$

| Moment | Name | Meaning |
|----|------|------|
| $\mathbb{E}[X]$ | First raw moment | Expectation (location) |
| $\mathbb{E}[X^2]$ | Second raw moment | Energy |
| $\mathbb{E}[(X-\mu)^2]$ | Second central moment | Variance (dispersion) |
| $\mathbb{E}[(X-\mu)^3]$ | Third central moment | Skewness (symmetry) |
| $\mathbb{E}[(X-\mu)^4]$ | Fourth central moment | Kurtosis (tail thickness) |

### Expectation of Product

Generally:

$$
\mathbb{E}[XY] \neq \mathbb{E}[X]\mathbb{E}[Y]
$$

When $X$ and $Y$ are **independent**:

$$
\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]
$$

```python
import numpy as np

# Expectation properties demonstration
print("Expectation properties:")
print("="*50)

# Create random variables
np.random.seed(42)
X = np.random.normal(5, 2, 100000)
Y = np.random.normal(3, 1, 100000)

# Linearity
a, b = 2, 3
lhs = (a * X + b * Y).mean()
rhs = a * X.mean() + b * Y.mean()
print(f"Linearity: E[{a}X + {b}Y]")
print(f"  Direct calculation: {lhs:.4f}")
print(f"  Decomposed calculation: {rhs:.4f}")
print(f"  Equal? {np.isclose(lhs, rhs)}")

# Moment calculation
print(f"\nMoment calculation:")
print(f"  E[X] = {X.mean():.4f}")
print(f"  E[X²] = {(X**2).mean():.4f}")
print(f"  E[(X-μ)²] = {((X - X.mean())**2).mean():.4f}")
print(f"  Var(X) = {X.var():.4f}")

# Verify Var(X) = E[X²] - E[X]²
calculated_var = (X**2).mean() - X.mean()**2
print(f"  Verification: E[X²] - E[X]² = {calculated_var:.4f}")

# Expectation of product of independent variables
Z = np.random.normal(0, 1, 100000)  # Independent of X
print(f"\nExpectation of product of independent variables:")
print(f"  E[X] = {X.mean():.4f}")
print(f"  E[Z] = {Z.mean():.4f}")
print(f"  E[X·Z] = {(X * Z).mean():.4f}")
print(f"  E[X]·E[Z] = {X.mean() * Z.mean():.4f}")
```

---

## Covariance

### Definition

**Covariance** measures the **linear relationship** between two random variables:

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

### Calculation Formula

$$
\text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]
$$

### Properties of Covariance

1. **Symmetry**: $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
2. **Self-covariance**: $\text{Cov}(X, X) = \text{Var}(X)$
3. **Constant**: $\text{Cov}(X, c) = 0$
4. **Linearity**: $\text{Cov}(aX, bY) = ab \cdot \text{Cov}(X, Y)$
5. **Additivity**: $\text{Cov}(X + Y, Z) = \text{Cov}(X, Z) + \text{Cov}(Y, Z)$
6. **Independence**: If $X, Y$ are independent, then $\text{Cov}(X, Y) = 0$

### Variance of Sum

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)
$$

**Derivation of variance of sum formula**:

**Step 1**: Use the definition of variance.

$$\text{Var}(X + Y) = \mathbb{E}[(X + Y - \mathbb{E}[X + Y])^2]$$

**Step 2**: Use the linearity of expectation to expand.

$$\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y] = \mu_X + \mu_Y$$

Therefore:

$$\text{Var}(X + Y) = \mathbb{E}[(X - \mu_X + Y - \mu_Y)^2]$$

**Step 3**: Expand the square.

$$= \mathbb{E}[(X - \mu_X)^2 + (Y - \mu_Y)^2 + 2(X - \mu_X)(Y - \mu_Y)]$$

**Step 4**: Use linearity of expectation to separate terms.

$$= \mathbb{E}[(X - \mu_X)^2] + \mathbb{E}[(Y - \mu_Y)^2] + 2\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]$$

**Step 5**: Identify each term.

- $\mathbb{E}[(X - \mu_X)^2] = \text{Var}(X)$
- $\mathbb{E}[(Y - \mu_Y)^2] = \text{Var}(Y)$
- $\mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \text{Cov}(X$, $Y)$

**Step 6**: Combine results.

$$\boxed{\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)}$$

**Corollary**: When $X, Y$ are independent, $\text{Cov}(X, Y) = 0$, therefore:

$$\boxed{\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)}$$

**General form**: For $n$ random variables:

$$\text{Var}\left(\sum_{i=1}^n X_i\right) = \sum_{i=1}^n \text{Var}(X_i) + 2\sum_{i < j} \text{Cov}(X_i, X_j)$$

```python
import numpy as np

# Covariance calculation example
np.random.seed(42)
n = 100000

# Positive correlation
X1 = np.random.normal(0, 1, n)
Y1 = X1 + np.random.normal(0, 0.5, n)  # Y positively correlated with X

# Negative correlation
X2 = np.random.normal(0, 1, n)
Y2 = -X2 + np.random.normal(0, 0.5, n)  # Y negatively correlated with X

# Uncorrelated
X3 = np.random.normal(0, 1, n)
Y3 = np.random.normal(0, 1, n)  # Independent

# Calculate covariances
cov1 = np.cov(X1, Y1)[0, 1]
cov2 = np.cov(X2, Y2)[0, 1]
cov3 = np.cov(X3, Y3)[0, 1]

print("Covariance examples:")
print("="*50)
print(f"Positive correlation: Cov(X1, Y1) = {cov1:.4f}")
print(f"Negative correlation: Cov(X2, Y2) = {cov2:.4f}")
print(f"Uncorrelated: Cov(X3, Y3) = {cov3:.4f}")

# Manual calculation verification
def manual_cov(X, Y):
    return ((X - X.mean()) * (Y - Y.mean())).mean()

print(f"\nManual calculation verification:")
print(f"Positive correlation: {manual_cov(X1, Y1):.4f}")
print(f"Negative correlation: {manual_cov(X2, Y2):.4f}")
print(f"Uncorrelated: {manual_cov(X3, Y3):.4f}")

# Variance of sum formula verification
print(f"\nVariance of sum formula verification:")
print(f"Var(X1) = {X1.var():.4f}")
print(f"Var(Y1) = {Y1.var():.4f}")
print(f"Var(X1+Y1) = {(X1+Y1).var():.4f}")
print(f"Var(X1) + Var(Y1) + 2*Cov(X1,Y1) = {X1.var() + Y1.var() + 2*cov1:.4f}")
```

---

## Correlation Coefficient

### Definition

**Correlation coefficient** is the standardized covariance, ranging in $[-1, 1]$:

$$
\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X) \cdot \text{Var}(Y)}} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

### Properties of Correlation Coefficient

1. **Range**: $-1 \leq \rho \leq 1$
2. **Dimensionless**: Eliminates unit effects
3. **Symmetry**: $\rho_{XY} = \rho_{YX}$
4. **Linear transformation**: $\rho_{aX+b, cY+d} = \text{sign}(ac) \cdot \rho_{XY}$

### Interpretation of Correlation Coefficient

| $\rho$ value | Relationship |
|-----------|------|
| $\rho = 1$ | Perfect positive linear correlation |
| $0 < \rho < 1$ | Positive correlation |
| $\rho = 0$ | Uncorrelated (no linear relationship) |
| $-1 < \rho < 0$ | Negative correlation |
| $\rho = -1$ | Perfect negative linear correlation |

### ⚠️ Important Note

$\rho = 0$ only means **no linear relationship**, not necessarily no nonlinear relationship!

```python
import numpy as np
import matplotlib.pyplot as plt

# Correlation coefficient visualization
np.random.seed(42)
n = 500

# Generate data with different correlations
scenarios = [
    ('Perfect Positive', lambda: (np.linspace(-3, 3, n), np.linspace(-3, 3, n))),
    ('Strong Positive', lambda: (np.random.normal(0, 1, n), np.random.normal(0, 1, n) + np.random.normal(0, 0.3, n))),
    ('No Correlation', lambda: (np.random.normal(0, 1, n), np.random.normal(0, 1, n))),
    ('Strong Negative', lambda: (np.random.normal(0, 1, n), -np.random.normal(0, 1, n) + np.random.normal(0, 0.3, n))),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for ax, (title, data_func) in zip(axes.flatten(), scenarios):
    X, Y = data_func()
    corr = np.corrcoef(X, Y)[0, 1]

    ax.scatter(X, Y, alpha=0.5, s=10)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title}\nρ = {corr:.3f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('correlation.png', dpi=100)
print("Correlation coefficient visualization image saved")

# Nonlinear but zero correlation example
print("\nNonlinear but zero correlation example:")
print("="*50)

X = np.random.uniform(-1, 1, 100000)
Y = X**2  # Completely deterministic nonlinear relationship
corr = np.corrcoef(X, Y)[0, 1]
print(f"Y = X²")
print(f"Corr(X, Y) = {corr:.6f}")
print(f"Conclusion: ρ ≈ 0, but X and Y have strong nonlinear relationship!")
```

---

## Covariance Matrix

### Definition

For random vector $\mathbf{X} = (X_1, X_2, \ldots, X_n)^\top$, the **covariance matrix** is defined as:

$$
\mathbf{\Sigma} = \text{Cov}(\mathbf{X}) = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^\top]
$$

Where $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$.

### Matrix Form

$$
\mathbf{\Sigma} = \begin{bmatrix}
\text{Var}(X_1) & \text{Cov}(X_1, X_2) & \cdots & \text{Cov}(X_1, X_n) \\
\text{Cov}(X_2, X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2, X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_n, X_1) & \text{Cov}(X_n, X_2) & \cdots & \text{Var}(X_n)
\end{bmatrix}
$$

### Properties of Covariance Matrix

1. **Symmetric**: $\mathbf{\Sigma} = \mathbf{\Sigma}^\top$
2. **Positive semidefinite**: $\mathbf{a}^\top \mathbf{\Sigma} \mathbf{a} \geq 0$ for any $\mathbf{a}$
3. **Non-negative diagonal elements**: $\Sigma_{ii} = \text{Var}(X_i) \geq 0$
4. **Non-negative eigenvalues**

### Sample Covariance Matrix

Given $m$ samples $\mathbf{x}_1, \ldots, \mathbf{x}_m$, the sample covariance matrix:

$$
\hat{\mathbf{\Sigma}} = \frac{1}{m-1} \sum_{i=1}^m (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top
$$

```python
import numpy as np

# Covariance matrix calculation
np.random.seed(42)

# Generate correlated data
n_samples = 1000
n_features = 3

# True covariance matrix
true_cov = np.array([
    [1.0, 0.8, 0.3],
    [0.8, 2.0, 0.5],
    [0.3, 0.5, 1.5]
])

print("True covariance matrix:")
print(true_cov)

# Sample from multivariate normal distribution
mean = np.zeros(n_features)
samples = np.random.multivariate_normal(mean, true_cov, n_samples)

print(f"\nSample shape: {samples.shape}")

# Calculate sample covariance matrix
sample_cov = np.cov(samples.T)
print("\nSample covariance matrix:")
print(sample_cov)

# Verify covariance matrix properties
print("\nCovariance matrix property verification:")
print(f"Symmetric? {np.allclose(sample_cov, sample_cov.T)}")
print(f"Eigenvalues: {np.linalg.eigvalsh(sample_cov)}")
print(f"Positive semidefinite? {np.all(np.linalg.eigvalsh(sample_cov) >= 0)}")

# Manual covariance matrix calculation
def manual_cov_matrix(X):
    """Manual covariance matrix calculation"""
    X_centered = X - X.mean(axis=0)
    return X_centered.T @ X_centered / (X.shape[0] - 1)

manual_cov = manual_cov_matrix(samples)
print(f"\nManual calculation matches np.cov? {np.allclose(sample_cov, manual_cov)}")
```

---

## Higher Moments: Skewness and Kurtosis

### Skewness

**Skewness** measures the **asymmetry** of a distribution:

$$
\gamma_1 = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^3\right] = \frac{\mathbb{E}[(X - \mu)^3]}{\sigma^3}
$$

| Skewness | Distribution Shape |
|------|----------|
| $\gamma_1 = 0$ | Symmetric |
| $\gamma_1 > 0$ | Right-skewed (positive skew), long tail on right |
| $\gamma_1 < 0$ | Left-skewed (negative skew), long tail on left |

### Kurtosis

**Kurtosis** measures the **tail thickness** of a distribution (relative to normal distribution):

$$
\gamma_2 = \mathbb{E}\left[\left(\frac{X - \mu}{\sigma}\right)^4\right] - 3
$$

**Excess kurtosis** (subtract 3 to make normal distribution 0):

| Kurtosis | Distribution Characteristics |
|------|----------|
| $\gamma_2 = 0$ | Normal distribution |
| $\gamma_2 > 0$ | Sharp peak (heavy tail), heavy-tailed distribution |
| $\gamma_2 < 0$ | Flat top (light tail), light-tailed distribution |

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Skewness and kurtosis example
np.random.seed(42)
n = 100000

# Symmetric distribution
normal = np.random.normal(0, 1, n)

# Right-skewed distribution
right_skewed = np.random.exponential(1, n)

# Left-skewed distribution
left_skewed = -np.random.exponential(1, n) + 3

# Heavy-tailed distribution (t distribution)
heavy_tailed = np.random.standard_t(3, n)

# Calculate skewness and kurtosis
print("Skewness and kurtosis analysis:")
print("="*50)

distributions = [
    ('Normal', normal),
    ('Right-skewed (Exp)', right_skewed),
    ('Left-skewed', left_skewed),
    ('Heavy-tailed (t)', heavy_tailed)
]

for name, data in distributions:
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    print(f"{name}:")
    print(f"  Skewness: {skewness:.4f}")
    print(f"  Kurtosis: {kurtosis:.4f}")
    print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, data) in zip(axes.flatten(), distributions):
    ax.hist(data, bins=100, density=True, alpha=0.7)
    ax.axvline(data.mean(), color='r', linestyle='--', label=f'Mean={data.mean():.2f}')
    ax.axvline(np.median(data), color='g', linestyle=':', label=f'Median={np.median(data):.2f}')
    ax.set_title(f'{name}\nSkew={stats.skew(data):.2f}, Kurt={stats.kurtosis(data):.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('skewness_kurtosis.png', dpi=100)
print("Skewness and kurtosis visualization image saved")
```

---

## Applications in Deep Learning

### 1. Batch Normalization

Batch Normalization uses batch statistics to estimate population statistics:

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$

Where $\mu_B$ and $\sigma_B^2$ are the mean and variance of the mini-batch.

```python
import numpy as np

class BatchNorm:
    """Simplified Batch Normalization implementation"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)

        # Running statistics
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

        self.training = True

    def forward(self, x):
        """
        Parameters:
        -----------
        x : array, shape (batch_size, num_features)
        """
        if self.training:
            # Calculate batch statistics
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)

            # Update running statistics (moving average)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta

# Example
bn = BatchNorm(num_features=64)
batch = np.random.randn(32, 64) * 2 + 1  # Non-standardized input

output = bn.forward(batch)
print("Batch Normalization example:")
print(f"Input: mean={batch.mean():.4f}, variance={batch.var():.4f}")
print(f"Output: mean={output.mean():.4f}, variance={output.var():.4f}")
```

### 2. Covariance Matrix in PCA

```python
import numpy as np

def pca(X, n_components):
    """
    PCA dimensionality reduction

    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
    n_components : int
        Number of principal components to keep
    """
    # Center
    X_centered = X - X.mean(axis=0)

    # Calculate covariance matrix
    cov_matrix = np.cov(X_centered.T)

    # Eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort by eigenvalue in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select first n_components principal components
    principal_components = eigenvectors[:, :n_components]

    # Project
    X_pca = X_centered @ principal_components

    return X_pca, eigenvalues[:n_components]

# Example
np.random.seed(42)
X = np.random.randn(1000, 10)

# Add some correlations
X[:, 1] = X[:, 0] + np.random.randn(1000) * 0.1
X[:, 2] = X[:, 0] + X[:, 1] + np.random.randn(1000) * 0.1

X_pca, explained_var = pca(X, n_components=3)

print("PCA example:")
print(f"Original data shape: {X.shape}")
print(f"Dimensionality reduced shape: {X_pca.shape}")
print(f"Variance explained by first 3 principal components: {explained_var}")
print(f"Variance explained ratio: {explained_var / explained_var.sum()}")
```

### 3. Layer Normalization

```python
import numpy as np

class LayerNorm:
    """Layer Normalization implementation"""

    def __init__(self, normalized_shape, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)

    def forward(self, x):
        """
        Parameters:
        -----------
        x : array, shape (batch_size, seq_len, hidden_size) or (batch_size, hidden_size)
        """
        # Calculate mean and variance on last dimension
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Scale and shift
        return self.gamma * x_norm + self.beta

# Example
ln = LayerNorm(normalized_shape=512)
x = np.random.randn(32, 10, 512)  # (batch, seq_len, hidden)

output = ln.forward(x)
print("Layer Normalization example:")
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Each sample mean: {output[0, 0, :].mean():.6f}")  # Close to 0
print(f"Each sample variance: {output[0, 0, :].var():.6f}")  # Close to 1
```

---

## Summary

This chapter introduced the core concepts of multivariate random variables, which are the foundations for understanding data representation and feature engineering in deep learning.

### Core Concept Comparison Table

| Concept | Formula | Deep Learning Applications |
|------|------|--------------|
| Joint distribution | $p(x,y) = P(X=x, Y=y)$ | Multi-feature modeling |
| Marginal distribution | $p_X(x) = \sum_y p(x,y)$ | Feature extraction |
| Conditional distribution | $p(y|x) = p(x,y)/p_X(x)$ | Generative models |
| Covariance | $\text{Cov}(X,Y) = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]$ | Feature correlation |
| Correlation coefficient | $\rho = \text{Cov}(X,Y)/(\sigma_X \sigma_Y)$ | Feature selection |
| Covariance matrix | $\mathbf{\Sigma} = \mathbb{E}[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^\top]$ | PCA, BatchNorm |

### Key Points

1. **Joint distribution**: Describes the overall distribution of multiple variables
2. **Marginal distribution**: Extract single variable distribution from joint distribution
3. **Conditional distribution**: Given one variable, the distribution of another variable
4. **Covariance**: Measures linear correlation, unit-sensitive
5. **Correlation coefficient**: Standardized covariance, $[-1, 1]$ range
6. **Independent vs uncorrelated**: Independent ⇒ uncorrelated, but not vice versa
7. **Covariance matrix**: Core tool for high-dimensional data, symmetric and positive semidefinite

### Core Applications in Deep Learning

| Technique | Probability Concepts Used |
|------|---------------|
| Batch Normalization | Mean, variance estimation |
| Layer Normalization | Layer-level statistics |
| PCA | Covariance matrix eigenvalue decomposition |
| Dropout | Bernoulli sampling independence |
| Attention | Conditional probability distribution |

---

**Previous section**: [Chapter 3 (b): Random Variables and Common Distributions](03b-random-variables-distributions_EN.md)

**Next section**: [Chapter 3 (d): Limit Theorems and Information Theory](03d-limit-theorems-information-theory_EN.md) - Learn about the law of large numbers, central limit theorem, entropy, cross-entropy, and KL divergence.

**Return**: [Mathematics Fundamentals Tutorial Table of Contents](../math-fundamentals.md)
