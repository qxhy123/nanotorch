# Chapter 4(d): Regression Analysis and Bayesian Statistics

Regression analysis is the most fundamental predictive modeling method in machine learning, while Bayesian statistics provides a powerful framework for handling uncertainty. This chapter will introduce the statistical foundation of linear regression, the bias-variance tradeoff, and the applications of Bayesian methods in deep learning.

---

## 🎯 Life Analogy: Predicting House Prices

Imagine you're a real estate agent trying to predict house prices based on size:

| Size (sq ft) | Price ($1000s) |
|--------------|----------------|
| 500 | 150 |
| 800 | 240 |
| 1000 | 300 |
| 1200 | 360 |
| 1500 | 450 |

**Observation**: For every 1 sq ft increase, price increases by about $300.

This is the core idea of **linear regression**: Find a line that all data points are as close to as possible.

```
Price ($1000s)
  450 │                                    ●
      │                               ·
  360 │                          ●
      │                     ·
  300 │                ●
      │           ·
  240 │      ●
      │ ·
  150 │●
      │
      └──────────────────────────────────→ Size (sq ft)
         500   800   1000  1200  1500

Fitted line: y = 0.3x (every sq ft adds $300)
```

### Least Squares = Minimize Total Error

How to find the best line? **Make the squared error (distance to line) sum as small as possible**.

Like choosing a seat in a classroom to minimize total distance to all classmates.

### Bias-Variance Tradeoff = Memorizing vs Understanding

| Learning Style | Analogy | Machine Learning |
|----------------|---------|------------------|
| **Memorizing** | Ace old exam questions, fail new ones | **High variance (overfitting)**: Perfect on training, terrible on test |
| **Only outlines** | Inaccurate on both old and new | **High bias (underfitting)**: Bad on both training and test |
| **Understanding** | Good on both old and new | **Balanced**: Good generalization |

### 📖 Bayesian Thinking = Updating Beliefs with New Information

Bayesian statistics is like **being a detective**:

```
┌────────────────────────────────────────────────────────────┐
│  Bayesian Reasoning Process                                 │
├────────────────────────────────────────────────────────────┤
│  1. Prior: Your initial belief                              │
│     "I think this coin is fair, P(heads)=50%"              │
│                                                            │
│  2. Data: Evidence you observe                             │
│     "Flipped 10 times, all heads"                          │
│                                                            │
│  3. Posterior: Updated belief after evidence               │
│     "This coin is probably rigged, P(heads)≈90%"           │
└────────────────────────────────────────────────────────────┘

Formula: Posterior ∝ Prior × Likelihood
P(θ|data) ∝ P(θ) × P(data|θ)
```

### 📖 Plain English Translation

| Math Term | Plain English |
|-----------|---------------|
| Regression | Finding a formula to predict one thing from others |
| Least squares | Making total squared error as small as possible |
| Bias | Systematic error (consistently wrong in one direction) |
| Variance | Random error (inconsistent predictions) |
| Prior | What you believed before seeing data |
| Posterior | What you believe after seeing data |

---

## Table of Contents

1. [Statistical Foundation of Linear Regression](#statistical-foundation-of-linear-regression)
2. [Statistical Properties of Least Squares Estimation](#statistical-properties-of-least-squares-estimation)
3. [Regression Diagnostics](#regression-diagnostics)
4. [Bias-Variance Tradeoff](#bias-variance-tradeoff)
5. [Bayesian Linear Regression](#bayesian-linear-regression)
6. [Bayesian Deep Learning](#bayesian-deep-learning)
7. [Applications in Deep Learning](#applications-in-deep-learning)
8. [Summary](#summary)

---

## Statistical Foundation of Linear Regression

### Linear Regression Model

**Simple Linear Regression**:

$$
Y = \beta_0 + \beta_1 X + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2)
$$

**Multiple Linear Regression**:

$$
Y = \beta_0 + \beta_1 X_1 + \cdots + \beta_p X_p + \epsilon
$$

**Matrix Form**:

$$
\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}
$$

where:
- $\mathbf{y} \in \mathbb{R}^n$: Response variable
- $\mathbf{X} \in \mathbb{R}^{n \times (p+1)}$: Design matrix (including intercept column)
- $\boldsymbol{\beta} \in \mathbb{R}^{p+1}$: Parameter vector
- $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$: Error term

### Classical Assumptions

1. **Linearity**: The relationship between $Y$ and $X$ is linear
2. **Independence**: Error terms are mutually independent
3. **Normality**: $\epsilon \sim \mathcal{N}(0, \sigma^2)$
4. **Homoscedasticity**: Error variance is constant

### Least Squares Estimation

**Objective**: Minimize the residual sum of squares (RSS)

$$
\text{RSS}(\boldsymbol{\beta}) = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|^2
$$

**Normal Equation**:

$$
\mathbf{X}^\top \mathbf{X} \boldsymbol{\hat{\beta}} = \mathbf{X}^\top \mathbf{y}
$$

**OLS Solution**:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Linear regression OLS example
np.random.seed(42)
n = 100

# Generate data
X = np.random.randn(n, 1)
true_beta = np.array([2.0, 3.0])  # [intercept, slope]
y = true_beta[0] + true_beta[1] * X.flatten() + np.random.randn(n) * 0.5

# Design matrix (add intercept column)
X_design = np.column_stack([np.ones(n), X])

# OLS estimation
beta_hat = np.linalg.lstsq(X_design, y, rcond=None)[0]

print("Linear Regression OLS Estimation:")
print("="*50)
print(f"True parameters: β₀ = {true_beta[0]:.2f}, β₁ = {true_beta[1]:.2f}")
print(f"OLS estimate: β̂₀ = {beta_hat[0]:.4f}, β̂₁ = {beta_hat[1]:.4f}")

# Prediction
y_pred = X_design @ beta_hat

# Residuals
residuals = y - y_pred
RSS = np.sum(residuals**2)
print(f"\nResidual Sum of Squares RSS: {RSS:.4f}")

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.scatter(X, y, alpha=0.7, label='Data points')
ax.plot(X, y_pred, 'r-', linewidth=2, label=f'Fitted line: y = {beta_hat[0]:.2f} + {beta_hat[1]:.2f}x')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Linear Regression')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_regression.png', dpi=100)
print("Image saved: linear_regression.png")
```

---

## Statistical Properties of Least Squares Estimation

### Gauss-Markov Theorem

Under classical assumptions, the OLS estimator is the **Best Linear Unbiased Estimator (BLUE)**:
- **Unbiased**: $\mathbb{E}[\hat{\boldsymbol{\beta}}] = \boldsymbol{\beta}$
- **Minimum Variance**: Has the minimum variance among all linear unbiased estimators

### Distribution of Parameters

$$
\hat{\boldsymbol{\beta}} \sim \mathcal{N}\left(\boldsymbol{\beta}, \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1}\right)
$$

### Unbiased Estimation of Variance

$$
\hat{\sigma}^2 = \frac{\text{RSS}}{n - p - 1} = \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{n - p - 1}
$$

### Standard Error of Parameters

$$
\text{SE}(\hat{\beta}_j) = \hat{\sigma} \sqrt{(\mathbf{X}^\top \mathbf{X})^{-1}_{jj}}
$$

### t Test

Test $H_0: \beta_j = 0$:

$$
t = \frac{\hat{\beta}_j}{\text{SE}(\hat{\beta}_j)} \sim t(n-p-1)
$$

### Coefficient of Determination $R^2$

$$
R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}
$$

- $R^2 = 0$: Model explains none of the variation
- $R^2 = 1$: Model fits perfectly

### Adjusted $R^2$

$$
R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n-p-1}
$$

```python
import numpy as np
from scipy import stats

# Statistical inference for linear regression
np.random.seed(42)
n = 100
p = 2

X = np.random.randn(n, p)
true_beta = np.array([1.0, 2.0, -1.0])  # [intercept, β₁, β₂]
y = true_beta[0] + X @ true_beta[1:] + np.random.randn(n) * 0.5

# Design matrix
X_design = np.column_stack([np.ones(n), X])

# OLS estimation
beta_hat = np.linalg.lstsq(X_design, y, rcond=None)[0]

# Predicted values and residuals
y_pred = X_design @ beta_hat
residuals = y - y_pred
RSS = np.sum(residuals**2)

# Variance estimation
sigma2_hat = RSS / (n - p - 1)
sigma_hat = np.sqrt(sigma2_hat)

# Parameter covariance matrix
XtX_inv = np.linalg.inv(X_design.T @ X_design)
cov_beta = sigma2_hat * XtX_inv

# Standard errors
se_beta = np.sqrt(np.diag(cov_beta))

# t statistics and p-values
t_stats = beta_hat / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - p - 1))

# R²
TSS = np.sum((y - y.mean())**2)
R2 = 1 - RSS / TSS
R2_adj = 1 - (1 - R2) * (n - 1) / (n - p - 1)

print("Linear Regression Statistical Inference:")
print("="*60)
print(f"{'Parameter':<10} {'Estimate':<12} {'Std Error':<12} {'t-value':<10} {'p-value':<10}")
print("-"*60)
for i, name in enumerate(['Intercept', 'β₁', 'β₂']):
    print(f"{name:<10} {beta_hat[i]:>10.4f} {se_beta[i]:>10.4f} {t_stats[i]:>8.4f} {p_values[i]:>8.4f}")

print()
print(f"Residual standard deviation: {sigma_hat:.4f}")
print(f"R²: {R2:.4f}")
print(f"Adjusted R²: {R2_adj:.4f}")
```

---

## Regression Diagnostics

### Residual Analysis

**Residual Plot**: Residuals vs fitted values

- Ideal case: Residuals randomly distributed, no pattern
- If there's a pattern: Possible nonlinearity or heteroscedasticity

### Q-Q Plot

Test the normality assumption of residuals.

### Leverage and Influential Points

**Leverage**:

$$
h_i = \mathbf{x}_i^\top (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{x}_i
$$

**Cook's Distance**:

$$
D_i = \frac{(y_i - \hat{y}_i)^2}{(p+1)\hat{\sigma}^2} \cdot \frac{h_i}{(1-h_i)^2}
$$

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Regression diagnostics
np.random.seed(42)
n = 100
X = np.random.randn(n)
y = 2 + 3*X + np.random.randn(n) * 0.5

# OLS
X_design = np.column_stack([np.ones(n), X])
beta_hat = np.linalg.lstsq(X_design, y, rcond=None)[0]
y_pred = X_design @ beta_hat
residuals = y - y_pred

# Leverage
H = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
leverage = np.diag(H)

# Standardized residuals
sigma_hat = np.sqrt(np.sum(residuals**2) / (n - 2))
std_residuals = residuals / (sigma_hat * np.sqrt(1 - leverage))

# Cook's distance
cook_d = (std_residuals**2 / 2) * (leverage / (1 - leverage))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Residual plot
axes[0, 0].scatter(y_pred, residuals, alpha=0.7)
axes[0, 0].axhline(0, color='red', linestyle='--')
axes[0, 0].set_xlabel('Fitted values')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residual Plot')
axes[0, 0].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[0, 1])
axes[0, 1].set_title('Residual Q-Q Plot')
axes[0, 1].grid(True, alpha=0.3)

# Leverage vs standardized residuals
axes[1, 0].scatter(leverage, std_residuals, alpha=0.7)
axes[1, 0].axhline(0, color='red', linestyle='--')
axes[1, 0].axhline(2, color='orange', linestyle=':', label='±2 SD')
axes[1, 0].axhline(-2, color='orange', linestyle=':')
axes[1, 0].set_xlabel('Leverage')
axes[1, 0].set_ylabel('Standardized Residuals')
axes[1, 0].set_title('Leverage vs Standardized Residuals')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Cook's distance
axes[1, 1].stem(range(n), cook_d)
axes[1, 1].axhline(4/n, color='red', linestyle='--', label=f'Threshold = 4/n = {4/n:.4f}')
axes[1, 1].set_xlabel('Observation Number')
axes[1, 1].set_ylabel("Cook's D")
axes[1, 1].set_title("Cook's Distance")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('regression_diagnostics.png', dpi=100)
print("Regression diagnostics image saved")
```

---

## Bias-Variance Tradeoff

### Decomposition

For regression problems, prediction error can be decomposed as:

$$
\mathbb{E}[(\hat{f}(x) - y)^2] = \underbrace{\text{Bias}^2[\hat{f}(x)]}_{\text{Bias}} + \underbrace{\text{Var}[\hat{f}(x)]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}
$$

### Bias

$$
\text{Bias}[\hat{f}(x)] = \mathbb{E}[\hat{f}(x)] - f(x)
$$

- High bias → **Underfitting**
- Model is too simple

### Variance

$$
\text{Var}[\hat{f}(x)] = \mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]
$$

- High variance → **Overfitting**
- Model is too complex

### Tradeoff

| Model Complexity | Bias | Variance | Total Error |
|-----------------|------|----------|-------------|
| Too simple | High | Low | High |
| Moderate | Medium | Medium | **Low** |
| Too complex | Low | High | High |

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Bias-variance tradeoff demonstration
np.random.seed(42)

# Generate data
n = 200
X = np.random.uniform(-3, 3, n)
true_f = lambda x: 0.5 * x**2 - 0.5 * x + 1
y = true_f(X) + np.random.randn(n) * 1.0

X = X.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Different complexity models
degrees = [1, 2, 5, 15]
colors = ['blue', 'green', 'orange', 'red']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, degree, color in zip(axes.flatten(), degrees, colors):
    # Fit polynomial regression
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)

    # Prediction
    X_plot = np.linspace(-3, 3, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    # Calculate error
    train_error = np.mean((model.predict(X_train) - y_train)**2)
    test_error = np.mean((model.predict(X_test) - y_test)**2)

    ax.scatter(X_train, y_train, alpha=0.5, s=20, label='Training data')
    ax.plot(X_plot, true_f(X_plot.flatten()), 'g--', linewidth=2, label='True function')
    ax.plot(X_plot, y_plot, color=color, linewidth=2, label=f'Polynomial (d={degree})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Polynomial Degree: {degree}\nTraining Error: {train_error:.2f}, Test Error: {test_error:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 10)

plt.tight_layout()
plt.savefig('bias_variance_tradeoff.png', dpi=100)
print("Bias-variance tradeoff image saved")

# Plot bias-variance curve
degrees = range(1, 16)
train_errors = []
test_errors = []

for degree in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    model.fit(X_train, y_train)

    train_errors.append(np.mean((model.predict(X_train) - y_train)**2))
    test_errors.append(np.mean((model.predict(X_test) - y_test)**2))

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(degrees, train_errors, 'o-', label='Training error (Variance)')
ax.plot(degrees, test_errors, 's-', label='Test error (Bias+Variance+Noise)')
ax.axvline(2, color='green', linestyle='--', label='Optimal complexity')
ax.set_xlabel('Model Complexity (Polynomial Degree)')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Bias-Variance Tradeoff')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bias_variance_curve.png', dpi=100)
print("Bias-variance curve image saved")
```

---

## Bayesian Linear Regression

### Bayesian Framework

In the Bayesian framework, parameters $\boldsymbol{\beta}$ are treated as random variables with prior distributions.

### Prior Distribution

Assume $\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0}, \tau^2 \mathbf{I})$

### Posterior Distribution

$$
P(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X}) \propto P(\mathbf{y} | \mathbf{X}, \boldsymbol{\beta}) \cdot P(\boldsymbol{\beta})
$$

**Posterior Mean**:

$$
\hat{\boldsymbol{\beta}}_{Bayes} = (\mathbf{X}^\top \mathbf{X} + \frac{\sigma^2}{\tau^2}\mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}
$$

**Ridge Regression**: The posterior mean is equivalent to the ridge regression solution, with regularization parameter $\lambda = \sigma^2/\tau^2$.

### Predictive Distribution

$$
P(y_* | \mathbf{x}_*, \mathbf{y}, \mathbf{X}) = \int P(y_* | \mathbf{x}_*, \boldsymbol{\beta}) P(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X}) d\boldsymbol{\beta}
$$

```python
import numpy as np
import matplotlib.pyplot as plt

# Bayesian linear regression example
np.random.seed(42)
n = 50
X = np.linspace(-5, 5, n).reshape(-1, 1)
true_beta = np.array([0, 1])
y = true_beta[0] + true_beta[1] * X.flatten() + np.random.randn(n) * 1.0

# Design matrix
X_design = np.column_stack([np.ones(n), X])

# Bayesian parameters
sigma2 = 1.0  # Noise variance
tau2 = 10.0   # Prior variance

# Posterior parameters
XtX = X_design.T @ X_design
Xty = X_design.T @ y
posterior_cov = np.linalg.inv(XtX / sigma2 + np.eye(2) / tau2)
posterior_mean = posterior_cov @ (Xty / sigma2)

print("Bayesian Linear Regression:")
print("="*50)
print(f"True parameters: β₀ = {true_beta[0]}, β₁ = {true_beta[1]}")
print(f"Posterior mean: β̂₀ = {posterior_mean[0]:.4f}, β̂₁ = {posterior_mean[1]:.4f}")
print(f"\nPosterior covariance matrix:")
print(posterior_cov)

# Sample from posterior
n_samples = 100
beta_samples = np.random.multivariate_normal(posterior_mean, posterior_cov, n_samples)

# Predictive distribution
X_plot = np.linspace(-6, 6, 100).reshape(-1, 1)
X_plot_design = np.column_stack([np.ones(100), X_plot])

y_samples = X_plot_design @ beta_samples.T
y_mean = y_samples.mean(axis=1)
y_std = y_samples.std(axis=1)

# Visualization
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.scatter(X, y, alpha=0.7, label='Observed data')

# Draw sampled regression lines
for i in range(20):
    ax.plot(X_plot, y_samples[:, i], 'r-', alpha=0.1)

ax.plot(X_plot, y_mean, 'b-', linewidth=2, label='Posterior mean')
ax.fill_between(X_plot.flatten(), y_mean - 2*y_std, y_mean + 2*y_std,
                alpha=0.2, color='blue', label='95% prediction interval')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Bayesian Linear Regression')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('bayesian_linear_regression.png', dpi=100)
print("\nBayesian linear regression image saved")
```

---

## Bayesian Deep Learning

### Parameter Uncertainty

In Bayesian deep learning, weights $\mathbf{W}$ have prior distributions:

$$
P(\mathbf{W}) = \prod_{i,j} \mathcal{N}(w_{ij}; 0, \alpha^{-1})
$$

### Posterior Distribution

$$
P(\mathbf{W} | \mathcal{D}) = \frac{P(\mathcal{D} | \mathbf{W}) P(\mathbf{W})}{P(\mathcal{D})}
$$

### Predictive Distribution

$$
P(y | \mathbf{x}, \mathcal{D}) = \int P(y | \mathbf{x}, \mathbf{W}) P(\mathbf{W} | \mathcal{D}) d\mathbf{W}
$$

### Variational Inference

Since the true posterior is difficult to compute, use variational distribution $q(\mathbf{W})$ to approximate:

$$
\text{ELBO} = \mathbb{E}_{q(\mathbf{W})}[\log P(\mathcal{D} | \mathbf{W})] - D_{KL}(q(\mathbf{W}) \| P(\mathbf{W}))
$$

### Dropout as Bayesian Approximation

**MC Dropout**: Keep Dropout enabled during testing, sample multiple times to obtain the predictive distribution.

```python
import numpy as np

# MC Dropout example
class DropoutLayer:
    def __init__(self, p=0.5):
        self.p = p
        self.training = True

    def forward(self, x):
        if self.training:
            mask = (np.random.random(x.shape) > self.p).astype(float)
            return x * mask / (1 - self.p)
        return x

class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_p=0.5):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)
        self.dropout = DropoutLayer(dropout_p)

    def relu(self, x):
        return np.maximum(0, x)

    def forward(self, x, training=True):
        self.dropout.training = training

        h1 = x @ self.W1 + self.b1
        h1 = self.relu(h1)
        h1 = self.dropout.forward(h1)

        out = h1 @ self.W2 + self.b2
        return out

# MC Dropout prediction
np.random.seed(42)
model = SimpleNN(10, 50, 1, dropout_p=0.3)
x = np.random.randn(100, 10)

# Training mode (enable dropout)
n_samples = 100
predictions = []
for _ in range(n_samples):
    pred = model.forward(x, training=True)
    predictions.append(pred)

predictions = np.array(predictions).squeeze()

# Calculate prediction mean and uncertainty
pred_mean = predictions.mean(axis=0)
pred_std = predictions.std(axis=0)

print("MC Dropout Prediction:")
print("="*50)
print(f"Number of samples: {n_samples}")
print(f"Prediction mean range: [{pred_mean.min():.2f}, {pred_mean.max():.2f}]")
print(f"Prediction std range: [{pred_std.min():.2f}, {pred_std.max():.2f}]")
print(f"\nUncertainty quantification: First 5 samples")
for i in range(5):
    print(f"  Sample {i}: mean={pred_mean[i]:.3f}, std={pred_std[i]:.3f}")
```

---

## Applications in Deep Learning

### 1. Regularization and Priors

| Regularization | Prior Distribution |
|----------------|-------------------|
| L2 Regularization | Gaussian Prior |
| L1 Regularization | Laplace Prior |
| Elastic Net | Gaussian+Laplace Mixed Prior |

### 2. Cross-Validation

**K-fold cross-validation** is used to evaluate the generalization ability of a model and is an application of the statistical estimation stability concept.

```python
import numpy as np

def k_fold_cv(X, y, model_fn, k=5):
    """K-fold cross-validation"""
    n = len(y)
    indices = np.arange(n)
    np.random.shuffle(indices)

    fold_size = n // k
    scores = []

    for i in range(k):
        # Split training and validation sets
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train and evaluate
        model = model_fn()
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)

    return np.array(scores)

# Example
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

np.random.seed(42)
X, y = make_regression(n_samples=200, n_features=10, noise=10)

scores = k_fold_cv(X, y, LinearRegression, k=5)

print("K-fold Cross-Validation:")
print("="*50)
print(f"Scores for each fold: {scores}")
print(f"Average score: {scores.mean():.4f}")
print(f"Standard deviation: {scores.std():.4f}")
print(f"95% confidence interval: ({scores.mean() - 1.96*scores.std()/np.sqrt(5):.4f}, "
      f"{scores.mean() + 1.96*scores.std()/np.sqrt(5):.4f})")
```

### 3. Early Stopping

Based on statistical monitoring of validation set error, prevents overfitting.

---

## Summary

This chapter introduced the statistical foundation of regression analysis and Bayesian methods, which are the theoretical foundation for understanding regularization and uncertainty estimation in deep learning.

### Core Concept Comparison Table

| Concept | Formula | Application |
|---------|---------|-------------|
| OLS | $\hat{\beta} = (X^\top X)^{-1}X^\top y$ | Linear layer |
| $R^2$ | $1 - RSS/TSS$ | Model evaluation |
| Bias-Variance | $Bias^2 + Var + \sigma^2$ | Over/underfitting diagnosis |
| Bayesian Regression | Posterior mean = Ridge regression | Regularization |
| MC Dropout | Multiple samples during testing | Uncertainty estimation |

### Key Points

1. **OLS**: Best Linear Unbiased Estimator (BLUE)
2. **Bias-Variance Tradeoff**: Core of model selection
3. **Regularization = Prior**: L2 corresponds to Gaussian prior
4. **Bayesian Deep Learning**: Quantifies uncertainty
5. **Cross-Validation**: Evaluates generalization ability

---

**Previous Section**: [Chapter 4(c): Hypothesis Testing](04c-hypothesis-testing_EN.md)

**Next Chapter**: [Chapter 5: Optimization Methods](05-optimization.md) - Learn about gradient descent, momentum methods, Adam, and other optimization algorithms.

**Return**: [Mathematics Fundamentals Tutorial Directory](../math-fundamentals.md)
