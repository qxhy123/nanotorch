# Chapter 2(c): Higher-order Derivatives and Taylor Expansion

Higher-order derivatives and Taylor expansion are important tools for deeply understanding function properties. In deep learning, second-order derivatives (Hessian matrices) are used to analyze the curvature of optimization problems, while Taylor expansion is used to understand the convergence properties of optimization algorithms. This chapter also briefly introduces integral fundamentals to prepare for the probability theory chapter.

---

## Table of Contents

1. [Second-order Derivatives](#second-order-derivatives)
2. [Hessian Matrix](#hessian-matrix)
3. [Taylor Expansion](#taylor-expansion)
4. [Newton's Method](#newtons-method)
5. [Integral Fundamentals](#integral-fundamentals)
6. [Applications in Deep Learning](#applications-in-deep-learning)
7. [Summary](#summary)

---

## Second-order Derivatives

### Definition

Second-order derivative is the derivative of the derivative:

$$
f''(x) = \frac{d^2 f}{dx^2} = \frac{d}{dx}\left(\frac{df}{dx}\right)
$$

### Higher-order Derivatives

$n$-th order derivative:

$$
f^{(n)}(x) = \frac{d^n f}{dx^n} = \frac{d}{dx}\left(f^{(n-1)}(x)\right)
$$

### Meaning of Second-order Derivatives

| Condition | Function Property | Geometric Meaning |
|------|----------|----------|
| $f''(x) > 0$ | Convex function | Opens upward |
| $f''(x) < 0$ | Concave function | Opens downward |
| $f''(x) = 0$ | Possibly an inflection point | Change in concavity |

### Common Second-order Derivatives

| Function $f(x)$ | First derivative $f'(x)$ | Second derivative $f''(x)$ |
|-------------|-----------------|-------------------|
| $x^n$ | $nx^{n-1}$ | $n(n-1)x^{n-2}$ |
| $e^x$ | $e^x$ | $e^x$ |
| $\ln x$ | $\frac{1}{x}$ | $-\frac{1}{x^2}$ |
| $\sin x$ | $\cos x$ | $-\sin x$ |
| $\cos x$ | $-\sin x$ | $-\cos x$ |
| $a^x$ | $a^x \ln a$ | $a^x (\ln a)^2$ |

### Second-order Derivatives of Activation Functions

**Sigmoid**: $\sigma(x) = \frac{1}{1+e^{-x}}$

$$
\sigma'(x) = \sigma(x)(1-\sigma(x))
$$

$$
\sigma''(x) = \sigma(x)(1-\sigma(x))(1-2\sigma(x))
$$

**Tanh**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

$$
\tanh'(x) = 1 - \tanh^2(x)
$$

$$
\tanh''(x) = -2\tanh(x)(1-\tanh^2(x))
$$

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    s = sigmoid(x)
    return s * (1 - s)

def sigmoid_double_prime(x):
    s = sigmoid(x)
    return s * (1 - s) * (1 - 2*s)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

def tanh_double_prime(x):
    t = np.tanh(x)
    return -2 * t * (1 - t**2)

# Verify second-order derivatives
x = 1.0

# Sigmoid
s1_num = (sigmoid_prime(x + 1e-5) - sigmoid_prime(x - 1e-5)) / 2e-5
s1_ana = sigmoid_double_prime(x)
print(f"Sigmoid''({x}): numerical={s1_num:.6f}, analytic={s1_ana:.6f}")

# Tanh
t1_num = (tanh_prime(x + 1e-5) - tanh_prime(x - 1e-5)) / 2e-5
t1_ana = tanh_double_prime(x)
print(f"tanh''({x}): numerical={t1_num:.6f}, analytic={t1_ana:.6f}")

# Plot second-order derivatives
import matplotlib.pyplot as plt
x_vals = np.linspace(-5, 5, 100)

# Find points where Sigmoid second-order derivative is zero (inflection points)
sigmoid_d2 = sigmoid_double_prime(x_vals)
inflection_points = x_vals[np.where(np.diff(np.sign(sigmoid_d2)))[0]]
print(f"\nSigmoid inflection points: x = {inflection_points}")  # Should be near x=0
```

### Second-order Derivative Test for Extrema

**Theorem**: Let $f'(x_0) = 0$

| Condition | Conclusion |
|------|------|
| $f''(x_0) > 0$ | $x_0$ is a local **minimum** point |
| $f''(x_0) < 0$ | $x_0$ is a local **maximum** point |
| $f''(x_0) = 0$ | Requires further analysis |

```python
import numpy as np

def find_extrema(f, f_prime, f_double_prime, x_range, n_points=1000):
    """Find extremum points and classify them"""
    x = np.linspace(x_range[0], x_range[1], n_points)

    # Find points where f' = 0
    deriv = f_prime(x)
    critical_indices = np.where(np.diff(np.sign(deriv)))[0]

    print("Critical point analysis:")
    for idx in critical_indices:
        x_crit = x[idx]
        f_val = f(x_crit)
        f_d2 = f_double_prime(x_crit)

        if f_d2 > 0:
            extrema_type = "minimum"
        elif f_d2 < 0:
            extrema_type = "maximum"
        else:
            extrema_type = "inconclusive"

        print(f"  x = {x_crit:.4f}: f(x) = {f_val:.4f}, f''(x) = {f_d2:.4f} -> {extrema_type}")

# Example: f(x) = x^3 - 3x
f = lambda x: x**3 - 3*x
f_prime = lambda x: 3*x**2 - 3
f_double_prime = lambda x: 6*x

find_extrema(f, f_prime, f_double_prime, [-3, 3])
```

---

## Hessian Matrix

### Definition

The **Hessian matrix** of multivariate function $f: \mathbb{R}^n \to \mathbb{R}$ is the matrix of second-order partial derivatives:

$$
\mathbf{H}_f(\mathbf{x}) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\[6pt]
\frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\[6pt]
\vdots & \vdots & \ddots & \vdots \\[6pt]
\frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2}
\end{bmatrix}
$$

Or written as:

$$
H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}
$$

### Properties

1. **Symmetry**: If the second-order partial derivatives of $f$ are continuous, then $\mathbf{H}$ is a symmetric matrix ($H_{ij} = H_{ji}$)

2. **Critical point classification**: Let $\nabla f(\mathbf{x}^*) = \mathbf{0}$ (critical point)

| Property of $\mathbf{H}$ | Type of $\mathbf{x}^*$ |
|---------------------|----------------------|
| Positive definite (all eigenvalues > 0) | Local minimum |
| Negative definite (all eigenvalues < 0) | Local maximum |
| Indefinite (eigenvalues have both positive and negative) | Saddle point |
| Positive semidefinite / negative semidefinite | Requires further analysis |

### Positive Definiteness Criteria

**Sylvester's criterion**: A symmetric matrix $\mathbf{H}$ is positive definite if and only if all leading principal minors are positive.

For 2×2 matrix $\mathbf{H} = \begin{bmatrix} a & b \\ b & c \end{bmatrix}$:

- Positive definite: $a > 0$ and $ac - b^2 > 0$
- Negative definite: $a < 0$ and $ac - b^2 > 0$
- Indefinite: $ac - b^2 < 0$

```python
import numpy as np

def hessian_matrix(f, x, h=1e-5):
    """Numerically compute Hessian matrix"""
    n = len(x)
    H = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            # Mixed partial derivatives
            x_pp = x.copy()
            x_pm = x.copy()
            x_mp = x.copy()
            x_mm = x.copy()

            x_pp[i] += h; x_pp[j] += h
            x_pm[i] += h; x_pm[j] -= h
            x_mp[i] -= h; x_mp[j] += h
            x_mm[i] -= h; x_mm[j] -= h

            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * h * h)
            H[j, i] = H[i, j]  # Symmetry

    return H

def classify_critical_point(H):
    """Classify critical point"""
    eigenvalues = np.linalg.eigvals(H)

    if np.all(eigenvalues > 0):
        return "local minimum", eigenvalues
    elif np.all(eigenvalues < 0):
        return "local maximum", eigenvalues
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        return "saddle point", eigenvalues
    else:
        return "requires further analysis", eigenvalues

# Example 1: f(x,y) = x^2 + y^2 (convex function)
def f1(x):
    return x[0]**2 + x[1]**2

x = np.array([0.0, 0.0])  # Critical point
H1 = hessian_matrix(f1, x)
type1, eigs1 = classify_critical_point(H1)
print(f"f(x,y) = x² + y² at (0,0):")
print(f"  Hessian matrix:\n{H1}")
print(f"  Eigenvalues: {eigs1}")
print(f"  Classification: {type1}")

# Example 2: f(x,y) = x^2 - y^2 (saddle point)
def f2(x):
    return x[0]**2 - x[1]**2

H2 = hessian_matrix(f2, x)
type2, eigs2 = classify_critical_point(H2)
print(f"\nf(x,y) = x² - y² at (0,0):")
print(f"  Hessian matrix:\n{H2}")
print(f"  Eigenvalues: {eigs2}")
print(f"  Classification: {type2}")
```

### Hessian Matrices of Common Functions

**Quadratic form**: $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A} \mathbf{x}$

$$
\mathbf{H} = \mathbf{A}
$$

**Quadratic function with linear term**: $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top \mathbf{A} \mathbf{x} + \mathbf{b}^\top \mathbf{x} + c$

$$
\mathbf{H} = \mathbf{A}
$$

---

## Taylor Expansion

### Univariate Taylor Expansion

The Taylor expansion of function $f(x)$ at point $a$ is:

$$
f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots + \frac{f^{(n)}(a)}{n!}(x-a)^n + R_n
$$

**Derivation of Taylor expansion**:

**Goal**: Use polynomial $P_n(x) = c_0 + c_1(x-a) + c_2(x-a)^2 + \cdots + c_n(x-a)^n$ to approximate $f(x)$, such that at $x = a$, the first $n$ derivatives of $P_n$ and $f$ are equal.

**Step 1**: Determine the constant term $c_0$.

At $x = a$: $P_n(a) = c_0$, $f(a) = f(a)$

Requiring $P_n(a) = f(a)$, we get: $c_0 = f(a)$

**Step 2**: Determine the linear coefficient $c_1$.

$P_n'(x) = c_1 + 2c_2(x-a) + 3c_3(x-a)^2 + \cdots$

$P_n'(a) = c_1$, $f'(a) = f'(a)$

Requiring $P_n'(a) = f'(a)$, we get: $c_1 = f'(a)$

**Step 3**: Determine the quadratic coefficient $c_2$.

$P_n''(x) = 2c_2 + 6c_3(x-a) + 12c_4(x-a)^2 + \cdots$

$P_n''(a) = 2c_2$, $f''(a) = f''(a)$

Requiring $P_n''(a) = f''(a)$, we get: $c_2 = \frac{f''(a)}{2}$

**Step 4**: General case.

$P_n^{(k)}(a) = k! \cdot c_k$ (only the $(x-a)^k$ term remains a non-zero constant after $k$-th differentiation)

Requiring $P_n^{(k)}(a) = f^{(k)}(a)$, we get:

$$c_k = \frac{f^{(k)}(a)}{k!}$$

**Step 5**: Combine results.

$$P_n(x) = \sum_{k=0}^n \frac{f^{(k)}(a)}{k!}(x-a)^k$$

$$\boxed{f(x) = \sum_{k=0}^n \frac{f^{(k)}(a)}{k!}(x-a)^k + R_n}$$

Where $R_n$ is the remainder (error), which has the Lagrange form:

$$R_n = \frac{f^{(n+1)}(\xi)}{(n+1)!}(x-a)^{n+1}, \quad \xi \text{ is between } x \text{ and } a$$

**Maclaurin expansion** (at $a=0$):

$$
f(x) = f(0) + f'(0)x + \frac{f''(0)}{2!}x^2 + \cdots
$$

### Maclaurin Expansions of Common Functions

$$
e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots
$$

$$
\sin x = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots
$$

$$
\cos x = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots
$$

$$
\ln(1+x) = x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots \quad (|x| < 1)
$$

$$
(1+x)^\alpha = 1 + \alpha x + \frac{\alpha(\alpha-1)}{2!}x^2 + \cdots \quad (|x| < 1)
$$

### Multivariate Taylor Expansion

The **second-order Taylor expansion** of function $f(\mathbf{x})$ at point $\mathbf{a}$ is:

$$
f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^\top(\mathbf{x}-\mathbf{a}) + \frac{1}{2}(\mathbf{x}-\mathbf{a})^\top \mathbf{H}_f(\mathbf{a}) (\mathbf{x}-\mathbf{a})
$$

### Significance of Taylor Expansion

1. **Local linearization**: Approximate complex functions with simple polynomials
2. **Error analysis**: Provide error bounds for approximations
3. **Optimization**: Understand convergence properties of optimization algorithms

```python
import numpy as np

def taylor_approximation(f, derivatives, a, x, order):
    """
    Taylor expansion approximation

    Args:
        f: Function
        derivatives: List of derivatives [f', f'', ...]
        a: Expansion point
        x: Evaluation point
        order: Expansion order
    """
    result = f(a)
    h = x - a

    for n in range(1, order + 1):
        if n <= len(derivatives):
            result += derivatives[n-1](a) * h**n / np.math.factorial(n)

    return result

# Example: Taylor expansion of e^x at x=0
f = np.exp
derivatives = [np.exp, np.exp, np.exp, np.exp, np.exp]  # All derivatives of e^x are e^x

x = 1.0
a = 0.0
true_value = np.exp(x)

print(f"Taylor expansion approximation of e^{x} at x=0:")
print(f"True value: {true_value:.10f}")

for order in range(1, 6):
    approx = taylor_approximation(f, derivatives, a, x, order)
    error = abs(approx - true_value)
    print(f"  {order}-order approximation: {approx:.10f}, error: {error:.2e}")

# Multivariate Taylor expansion example
def multivariate_taylor(f, grad_f, hess_f, a, x):
    """Second-order Taylor expansion of multivariate function"""
    diff = x - a
    return f(a) + grad_f(a) @ diff + 0.5 * diff @ hess_f(a) @ diff

# f(x,y) = x^2 + y^2 + x*y
def f_mv(x):
    return x[0]**2 + x[1]**2 + x[0]*x[1]

def grad_f_mv(x):
    return np.array([2*x[0] + x[1], 2*x[1] + x[0]])

def hess_f_mv(x):
    return np.array([[2, 1], [1, 2]])

a = np.array([1.0, 1.0])
x = np.array([1.1, 0.9])

true_val = f_mv(x)
taylor_val = multivariate_taylor(f_mv, grad_f_mv, hess_f_mv, a, x)
print(f"\nf(x,y) = x² + y² + xy:")
print(f"Expanded at {a}, evaluated at {x}:")
print(f"  True value: {true_val:.6f}")
print(f"  Taylor approximation: {taylor_val:.6f}")
print(f"  Error: {abs(true_val - taylor_val):.2e}")
```

---

## Newton's Method

### Univariate Newton's Method

Finding roots of $f(x) = 0$:

$$
x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}
$$

### Newton's Method in Optimization

Finding minima of $f(x)$ (where $\nabla f = 0$):

$$
x_{n+1} = x_n - \frac{f'(x_n)}{f''(x_n)}
$$

### Multivariate Newton's Method

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{H}^{-1} \nabla f(\mathbf{x}_t)
$$

### Advantages and Disadvantages of Newton's Method

| Advantages | Disadvantages |
|------|------|
| Quadratic convergence (faster than gradient descent) | Requires computing Hessian matrix |
| No learning rate needed | Inverting Hessian matrix is expensive |
| Works well for quadratic functions | May converge to saddle points or maxima |

```python
import numpy as np

def newton_method_1d(f, f_prime, f_double_prime, x0, tol=1e-8, max_iter=100):
    """Univariate Newton's method optimization"""
    x = x0
    history = [x]

    for i in range(max_iter):
        grad = f_prime(x)
        hess = f_double_prime(x)

        if abs(hess) < 1e-10:
            print("Warning: Second derivative close to zero")
            break

        x_new = x - grad / hess

        if abs(x_new - x) < tol:
            print(f"Converged at iteration {i+1}")
            break

        x = x_new
        history.append(x)

    return x, history

# Example: Minimize f(x) = x^2 + 2x + 1 = (x+1)^2
f = lambda x: x**2 + 2*x + 1
f_prime = lambda x: 2*x + 2
f_double_prime = lambda x: 2

x0 = 5.0
x_min, history = newton_method_1d(f, f_prime, f_double_prime, x0)
print(f"Minimum point: x = {x_min:.6f}")
print(f"Minimum value: f(x) = {f(x_min):.6f}")
print(f"Iteration path: {history}")

def newton_method_multivariate(grad_f, hess_f, x0, tol=1e-8, max_iter=100):
    """Multivariate Newton's method optimization"""
    x = x0.copy()
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_f(x)
        H = hess_f(x)

        # Check if Hessian is invertible
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("Hessian is not invertible, using pseudo-inverse")
            H_inv = np.linalg.pinv(H)

        delta = H_inv @ grad
        x_new = x - delta

        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged at iteration {i+1}")
            break

        x = x_new
        history.append(x.copy())

    return x, history

# Example: Minimize f(x,y) = x^2 + 2y^2
def grad_f2(x):
    return np.array([2*x[0], 4*x[1]])

def hess_f2(x):
    return np.array([[2, 0], [0, 4]])

x0 = np.array([3.0, 2.0])
x_min, history = newton_method_multivariate(grad_f2, hess_f2, x0)
print(f"\nMinimum point: {x_min}")
print(f"Iterations: {len(history)}")
```

### Quasi-Newton Methods

Since computing the Hessian matrix is expensive, **quasi-Newton methods** (such as BFGS, L-BFGS) are commonly used in practice to approximate the Hessian matrix.

```python
def bfgs_update(H, s, y):
    """BFGS update of Hessian inverse"""
    rho = 1.0 / (y @ s)
    I = np.eye(len(s))
    V = I - rho * np.outer(s, y)
    H_new = V @ H @ V.T + rho * np.outer(s, s)
    return H_new

def bfgs_optimize(grad_f, x0, tol=1e-6, max_iter=100):
    """BFGS optimization"""
    n = len(x0)
    x = x0.copy()
    H = np.eye(n)  # Initial approximation of Hessian inverse
    grad = grad_f(x)
    history = [x.copy()]

    for i in range(max_iter):
        if np.linalg.norm(grad) < tol:
            print(f"Converged at iteration {i+1}")
            break

        # Search direction
        p = -H @ grad

        # Line search (simplified, using fixed step size)
        alpha = 1.0
        x_new = x + alpha * p
        grad_new = grad_f(x_new)

        # BFGS update
        s = x_new - x
        y = grad_new - grad

        if y @ s > 0:  # Curvature condition
            H = bfgs_update(H, s, y)

        x = x_new
        grad = grad_new
        history.append(x.copy())

    return x, history

# Test BFGS
x0 = np.array([5.0, 3.0])
x_min, history = bfgs_optimize(grad_f2, x0)
print(f"BFGS minimum point: {x_min}")
print(f"Iterations: {len(history)}")
```

---

## Integral Fundamentals

### Indefinite Integrals

**Definition**: If $F'(x) = f(x)$, then $F(x)$ is an **antiderivative** of $f(x)$:

$$
\int f(x) \, dx = F(x) + C
$$

Where $C$ is the constant of integration.

### Basic Integral Formulas

| Integrand $f(x)$ | Antiderivative $\int f(x) dx$ |
|-----------------|------------------------|
| $x^n$ $(n \neq -1)$ | $\frac{x^{n+1}}{n+1} + C$ |
| $\frac{1}{x}$ | $\ln|x| + C$ |
| $e^x$ | $e^x + C$ |
| $a^x$ | $\frac{a^x}{\ln a} + C$ |
| $\sin x$ | $-\cos x + C$ |
| $\cos x$ | $\sin x + C$ |
| $\sec^2 x$ | $\tan x + C$ |
| $\frac{1}{1+x^2}$ | $\arctan x + C$ |
| $\frac{1}{\sqrt{1-x^2}}$ | $\arcsin x + C$ |

### Definite Integrals

$$
\int_a^b f(x) \, dx = F(b) - F(a)
$$

**Newton-Leibniz formula**: Establishes the connection between differentiation and integration.

### Integral Properties

$$
\int_a^b [f(x) + g(x)] \, dx = \int_a^b f(x) \, dx + \int_a^b g(x) \, dx
$$

$$
\int_a^b cf(x) \, dx = c \int_a^b f(x) \, dx
$$

$$
\int_a^b f(x) \, dx = -\int_b^a f(x) \, dx
$$

$$
\int_a^b f(x) \, dx + \int_b^c f(x) \, dx = \int_a^c f(x) \, dx
$$

### Numerical Integration

```python
import numpy as np

def trapezoidal_rule(f, a, b, n=1000):
    """Trapezoidal rule for numerical integration"""
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return h * (y[0]/2 + np.sum(y[1:-1]) + y[-1]/2)

def simpson_rule(f, a, b, n=1000):
    """Simpson's rule for numerical integration (more accurate)"""
    if n % 2 == 1:
        n += 1
    x = np.linspace(a, b, n+1)
    y = f(x)
    h = (b - a) / n
    return h/3 * (y[0] + 4*np.sum(y[1:n:2]) + 2*np.sum(y[2:n-1:2]) + y[n])

# Example
f = lambda x: x**2
a, b = 0, 1

analytic = (b**3 - a**3) / 3  # ∫x²dx = x³/3
trapezoidal = trapezoidal_rule(f, a, b)
simpson = simpson_rule(f, a, b)

print(f"∫₀¹ x² dx:")
print(f"  Analytical solution: {analytic:.10f}")
print(f"  Trapezoidal rule: {trapezoidal:.10f}, error: {abs(trapezoidal - analytic):.2e}")
print(f"  Simpson's rule: {simpson:.10f}, error: {abs(simpson - analytic):.2e}")
```

### Applications of Integrals in Probability

**Normalization of probability density function**:

$$
\int_{-\infty}^{\infty} p(x) \, dx = 1
$$

**Expectation**:

$$
\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x) \, dx
$$

**Variance**:

$$
\text{Var}(X) = \int_{-\infty}^{\infty} (x - \mu)^2 \cdot p(x) \, dx
$$

---

## Applications in Deep Learning

### Analyzing Optimization Curvature

The eigenvalues of the Hessian matrix reflect the curvature of the loss function in the parameter space:

```python
import numpy as np

def analyze_optimization_landscape(H):
    """Analyze optimization curvature"""
    eigenvalues = np.linalg.eigvals(H)

    print("Optimization curvature analysis:")
    print(f"  Eigenvalues: {eigenvalues}")
    print(f"  Condition number: {max(eigenvalues)/min(eigenvalues):.2f}")

    if np.all(eigenvalues > 0):
        print("  Convex function: gradient descent guarantees convergence to global optimum")
    elif np.any(eigenvalues > 0) and np.any(eigenvalues < 0):
        print("  Non-convex function: saddle points exist, optimization may be difficult")

    # Condition number effect
    cond = max(eigenvalues) / min(eigenvalues)
    if cond > 100:
        print(f"  Large condition number ({cond:.0f}): large difference in convergence speed across directions")
        print("  Recommendation: use preconditioning or adaptive learning rates")
```

### Vanishing/Exploding Gradient Analysis

Taylor expansion is used to analyze gradient propagation in deep networks:

$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial h_n} \cdot \prod_{i=1}^{n} \frac{\partial h_i}{\partial h_{i-1}}
$$

If the derivative of each layer is less than 1, gradients vanish; if greater than 1, gradients explode.

### Second-order Effects of Batch Normalization

The gradient of BN involves second-order derivatives because it depends on batch statistics.

---

## Summary

This chapter introduced higher-order derivatives, Hessian matrices, and Taylor expansion:

| Concept | Definition/Formula | Application |
|------|----------|------|
| Second-order derivative | $f''(x) = \frac{d^2f}{dx^2}$ | Determine convexity/concavity |
| Hessian matrix | $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ | Analyze optimization curvature |
| Taylor expansion | $f(x) \approx f(a) + f'(a)(x-a) + \frac{f''(a)}{2}(x-a)^2$ | Local approximation |
| Newton's method | $\mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{H}^{-1}\nabla f$ | Second-order optimization |

### Key Formulas

**Taylor expansion (multivariate)**:

$$
f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^\top(\mathbf{x}-\mathbf{a}) + \frac{1}{2}(\mathbf{x}-\mathbf{a})^\top \mathbf{H} (\mathbf{x}-\mathbf{a})
$$

**Critical point classification**:
- Positive definite Hessian → local minimum
- Negative definite Hessian → local maximum
- Indefinite Hessian → saddle point

---

**Previous section**: [Chapter 2(b): Partial Derivatives, Gradients, and Multivariate Differentiation](02b-partial-derivatives-gradients_EN.md)

**Next section**: [Chapter 2(d): Vector Matrix Calculus and Backpropagation](02d-vector-matrix-calculus-backprop_EN.md) - Learn about matrix derivatives and the backpropagation algorithm.

**Return to**: [Mathematics Fundamentals Tutorial Index](../math-fundamentals.md)
