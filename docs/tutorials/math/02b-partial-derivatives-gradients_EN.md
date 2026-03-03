# Chapter 2(b): Partial Derivatives, Gradients, and Multivariate Differentiation

In deep learning, we often deal with multivariate functions—loss functions depend on thousands or millions of parameters. Partial derivatives and gradients are key tools for understanding and optimizing these functions. This chapter systematically introduces multivariate differentiation and its applications in gradient descent.

---

## 🎯 Life Analogy: Adjusting Speaker Settings

Imagine you're adjusting a speaker with Bass and Treble knobs:

- **Partial derivative (for Bass)**: Keep treble fixed, only adjust bass—how much does the sound change?
- **Partial derivative (for Treble)**: Keep bass fixed, only adjust treble—how much does the sound change?

**Partial derivative = Change only one factor, see how the result changes**

| Scenario | Partial Derivative Meaning |
|----------|---------------------------|
| House price (size, location, floor) | How much does price increase per extra sq meter? |
| Grades (study time, sleep, diet) | How much do grades improve with 1 more hour of study? |
| Temperature (heating power, fan speed) | How much does temperature rise per extra watt? |

### 📝 Step-by-Step Calculation

Let $f(x, y) = x^2 + 2xy + y^2$, find partial derivatives at point $(1, 2)$.

**Step 1: Find partial derivative with respect to $x$** (treat $y$ as constant)

$$\frac{\partial f}{\partial x} = 2x + 2y + 0 = 2x + 2y$$

**Step 2: Find partial derivative with respect to $y$** (treat $x$ as constant)

$$\frac{\partial f}{\partial y} = 0 + 2x + 2y = 2x + 2y$$

**Step 3: Evaluate at point $(1, 2)$**

$$\frac{\partial f}{\partial x}\bigg|_{(1,2)} = 2(1) + 2(2) = 6$$
$$\frac{\partial f}{\partial y}\bigg|_{(1,2)} = 2(1) + 2(2) = 6$$

**Interpretation**:
- At $(1,2)$, if $x$ increases by 1, the function increases by about 6
- At $(1,2)$, if $y$ increases by 1, the function increases by about 6

### 📖 Plain English Translation

| Math Term | Plain English |
|-----------|---------------|
| Partial derivative | Rate of change with respect to one variable (others fixed) |
| Gradient | Vector of all partial derivatives (points uphill) |
| Directional derivative | Rate of change in any direction |

---

## Table of Contents

1. [Partial Derivatives](#partial-derivatives)
2. [Directional Derivatives](#directional-derivatives)
3. [Gradients](#gradients)
4. [Gradient Descent Algorithm](#gradient-descent-algorithm)
5. [Multivariate Chain Rule](#multivariate-chain-rule)
6. [Total Differential](#total-differential)
7. [Applications in Deep Learning](#applications-in-deep-learning)
8. [Summary](#summary)

---

## Partial Derivatives

### Definition

The **partial derivative** of multivariate function $f(x_1, x_2, \ldots, x_n)$ with respect to $x_i$ is the derivative with respect to $x_i$ while keeping other variables fixed:

$$
\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}
$$

### Notations

| Notation | Meaning |
|--------|------|
| $\frac{\partial f}{\partial x_i}$ | Leibniz notation |
| $f_{x_i}$ | Subscript notation |
| $\partial_{x_i} f$ | Operator notation |
| $D_i f$ | Another notation |

Partial derivative at point $\mathbf{a}$: $\frac{\partial f}{\partial x_i}\bigg|_{\mathbf{x}=\mathbf{a}}$

### Computation Methods

When computing $\frac{\partial f}{\partial x_i}$, treat other variables $x_j$ ($j \neq i$) as constants.

### Examples

**Example 1**: Let $f(x, y) = x^2 y + xy^3$

Find $\frac{\partial f}{\partial x}$ (treat $y$ as constant):

$$
\frac{\partial f}{\partial x} = 2xy + y^3
$$

Find $\frac{\partial f}{\partial y}$ (treat $x$ as constant):

$$
\frac{\partial f}{\partial y} = x^2 + 3xy^2
$$

**Example 2**: Let $f(x, y, z) = x^2 + 2y^2 + 3z^2 + xy - yz$

$$
\frac{\partial f}{\partial x} = 2x + y
$$

$$
\frac{\partial f}{\partial y} = 4y + x - z
$$

$$
\frac{\partial f}{\partial z} = 6z - y
$$

### Higher-Order Partial Derivatives

Second-order partial derivatives:

$$
\frac{\partial^2 f}{\partial x_i^2} = \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_i}\right)
$$

Mixed partial derivatives:

$$
\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial}{\partial x_i}\left(\frac{\partial f}{\partial x_j}\right)
$$

**Schwarz's Theorem** (commutativity of mixed partial derivatives):

If the second-order partial derivatives of $f$ are continuous in a region, then:

$$
\frac{\partial^2 f}{\partial x_i \partial x_j} = \frac{\partial^2 f}{\partial x_j \partial x_i}
$$

```python
import numpy as np

def partial_derivative(f, var_idx, x, h=1e-5):
    """Compute partial derivative of f with respect to var_idx-th variable"""
    x_plus = x.copy()
    x_minus = x.copy()
    x_plus[var_idx] += h
    x_minus[var_idx] -= h
    return (f(x_plus) - f(x_minus)) / (2 * h)

# Example 1: f(x, y) = x^2*y + x*y^3
def f1(x):
    return x[0]**2 * x[1] + x[0] * x[1]**3

# Analytical partial derivatives
def df1_dx(x):
    return 2*x[0]*x[1] + x[1]**3

def df1_dy(x):
    return x[0]**2 + 3*x[0]*x[1]**2

x = np.array([2.0, 3.0])

# Numerical computation
partial_x = partial_derivative(f1, 0, x)
partial_y = partial_derivative(f1, 1, x)

print(f"f(x,y) = x²y + xy³ at ({x[0]}, {x[1]}):")
print(f"  ∂f/∂x: numerical={partial_x:.4f}, analytic={df1_dx(x):.4f}")
print(f"  ∂f/∂y: numerical={partial_y:.4f}, analytic={df1_dy(x):.4f}")

# Example 2: Second-order partial derivative verification
def f2(x):
    return x[0]**2 * x[1]**2

# ∂²f/∂x∂y = ∂²f/∂y∂x = 4xy
def d2f_dxdy(x):
    return 4 * x[0] * x[1]

# Numerical computation of ∂²f/∂x∂y
def mixed_second_partial(f, x, h=1e-5):
    """Compute mixed second-order partial derivative"""
    def first_partial_x(x):
        return partial_derivative(f, 0, x, h)
    return partial_derivative(first_partial_x, 1, x, h)

x = np.array([2.0, 3.0])
mixed = mixed_second_partial(f2, x)
print(f"\n∂²f/∂x∂y = ∂²f/∂y∂x verification:")
print(f"  numerical: {mixed:.4f}")
print(f"  analytic: {d2f_dxdy(x):.4f}")
```

---

## Directional Derivatives

### Definition

The **directional derivative** of function $f(\mathbf{x})$ in the direction of unit vector $\mathbf{u}$ is:

$$
D_{\mathbf{u}} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{u}) - f(\mathbf{x})}{h}
$$

### Relationship Between Directional Derivative and Gradient

$$
D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u}
$$

That is, the directional derivative is the projection of the gradient onto direction $\mathbf{u}$.

**Derivation of the relationship between directional derivative and gradient**:

**Step 1**: Define the directional derivative.

$$D_{\mathbf{u}} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{u}) - f(\mathbf{x})}{h}$$

**Step 2**: Use multivariate Taylor expansion (first order).

$$f(\mathbf{x} + h\mathbf{u}) = f(\mathbf{x}) + \nabla f(\mathbf{x}) \cdot (h\mathbf{u}) + o(h)$$

$$= f(\mathbf{x}) + h \nabla f(\mathbf{x}) \cdot \mathbf{u} + o(h)$$

**Step 3**: Substitute into the directional derivative definition.

$$D_{\mathbf{u}} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x}) + h \nabla f(\mathbf{x}) \cdot \mathbf{u} + o(h) - f(\mathbf{x})}{h}$$

$$= \lim_{h \to 0} \left( \nabla f(\mathbf{x}) \cdot \mathbf{u} + \frac{o(h)}{h} \right)$$

**Step 4**: Take the limit.

$$\boxed{D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u}}$$

**Corollary**: Maximum directional derivative.

Since $\mathbf{u}$ is a unit vector, by the Cauchy-Schwarz inequality:

$$|D_{\mathbf{u}} f| = |\nabla f \cdot \mathbf{u}| \leq \|\nabla f\| \|\mathbf{u}\| = \|\nabla f\|$$

Equality holds if and only if $\mathbf{u}$ is in the same direction as $\nabla f$. Therefore:

$$\max_{\|\mathbf{u}\|=1} D_{\mathbf{u}} f = \|\nabla f\|$$

### Properties

| Direction | Directional derivative value | Meaning |
|------|-----------|------|
| Gradient direction $\mathbf{u} = \frac{\nabla f}{\|\nabla f\|}$ | $\|\nabla f\|$ | Maximum rate of change |
| Negative gradient direction $\mathbf{u} = -\frac{\nabla f}{\|\nabla f\|}$ | $-\|\nabla f\|$ | Minimum rate of change (steepest descent) |
| Contour direction $\mathbf{u} \perp \nabla f$ | $0$ | No change |

### Direction of Maximum Rate of Change

The directional derivative attains its maximum value if and only if $\mathbf{u}$ is in the same direction as $\nabla f$:

$$
\max_{\|\mathbf{u}\|=1} D_{\mathbf{u}} f = \|\nabla f\|
$$

```python
import numpy as np

def directional_derivative(f, x, u, h=1e-5):
    """Compute directional derivative"""
    u = u / np.linalg.norm(u)  # Normalize
    return (f(x + h*u) - f(x - h*u)) / (2*h)

def gradient(f, x, h=1e-5):
    """Compute gradient"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2*h)
    return grad

# Example: f(x,y) = x^2 + y^2
def f(x):
    return x[0]**2 + x[1]**2

x = np.array([3.0, 4.0])
grad = gradient(f, x)

# Directional derivatives in different directions
u1 = np.array([1.0, 0.0])  # x direction
u2 = np.array([0.0, 1.0])  # y direction
u3 = grad / np.linalg.norm(grad)  # Gradient direction (normalized)
u4 = -grad / np.linalg.norm(grad)  # Negative gradient direction

print(f"f(x,y) = x² + y² at ({x[0]}, {x[1]}):")
print(f"  gradient: {grad}")
print(f"  gradient direction: {u3}")
print(f"\nDirectional derivatives:")
print(f"  x direction: {directional_derivative(f, x, u1):.4f}")
print(f"  y direction: {directional_derivative(f, x, u2):.4f}")
print(f"  gradient direction: {directional_derivative(f, x, u3):.4f} (maximum)")
print(f"  negative gradient direction: {directional_derivative(f, x, u4):.4f} (minimum)")

# Verify: directional derivative = gradient · direction
print(f"\nVerify D_u f = ∇f · u:")
print(f"  x direction: {directional_derivative(f, x, u1):.4f} ≈ {np.dot(grad, u1/np.linalg.norm(u1)):.4f}")
print(f"  gradient direction: {directional_derivative(f, x, u3):.4f} = ||∇f|| = {np.linalg.norm(grad):.4f}")
```

---

## Gradients

### Definition

The **gradient** of multivariate function $f: \mathbb{R}^n \to \mathbb{R}$ is a vector consisting of all partial derivatives:

$$
\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}
$$

Or written as: $\nabla f = \left(\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n}\right)^\top$

### Gradient Operator

$\nabla$ (nabla) is a vector differential operator:

$$
\nabla = \begin{bmatrix} \frac{\partial}{\partial x_1} \\ \frac{\partial}{\partial x_2} \\ \vdots \\ \frac{\partial}{\partial x_n} \end{bmatrix}
$$

### Geometric Meaning of Gradient

1. **Direction**: Gradient points in the direction of **fastest increase** of the function
2. **Magnitude**: $\|\nabla f\|$ is the directional derivative value in that direction
3. **Negative gradient**: $-\nabla f$ points in the direction of **fastest decrease** of the function
4. **Contours**: Gradient is perpendicular to contours (level sets)

### Gradient Operation Rules

Let $f, g$ be differentiable functions, and $c$ be a constant:

$$
\nabla(cf) = c\nabla f
$$

$$
\nabla(f + g) = \nabla f + \nabla g
$$

$$
\nabla(fg) = g\nabla f + f\nabla g
$$

$$
\nabla(f/g) = \frac{g\nabla f - f\nabla g}{g^2}
$$

### Gradients of Common Functions

**Linear function**: $f(\mathbf{x}) = \mathbf{a}^\top \mathbf{x}$

$$
\nabla f = \mathbf{a}
$$

**Quadratic form**: $f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A} \mathbf{x}$ (where $\mathbf{A}$ is symmetric)

$$
\nabla f = 2\mathbf{A}\mathbf{x}
$$

**Derivation of quadratic form gradient**:

**Step 1**: Expand the quadratic form.

Let $\mathbf{x} = (x_1, \ldots, x_n)^\top$, $\mathbf{A} = (a_{ij})$, then:

$$f(\mathbf{x}) = \mathbf{x}^\top \mathbf{A} \mathbf{x} = \sum_{i=1}^n \sum_{j=1}^n a_{ij} x_i x_j$$

**Step 2**: Compute $\frac{\partial f}{\partial x_k}$.

$$\frac{\partial f}{\partial x_k} = \frac{\partial}{\partial x_k} \sum_{i,j} a_{ij} x_i x_j$$

For terms containing $x_k$:
- When $i = k, j \neq k$: $a_{kj} x_k x_j$, partial derivative is $a_{kj} x_j$
- When $j = k, i \neq k$: $a_{ik} x_i x_k$, partial derivative is $a_{ik} x_i$
- When $i = j = k$: $a_{kk} x_k^2$, partial derivative is $2a_{kk} x_k$

Therefore:

$$\frac{\partial f}{\partial x_k} = \sum_{j \neq k} a_{kj} x_j + \sum_{i \neq k} a_{ik} x_i + 2a_{kk} x_k$$

**Step 3**: Use symmetry of $\mathbf{A}$ ($a_{ik} = a_{ki}$).

$$\frac{\partial f}{\partial x_k} = 2 \sum_{j=1}^n a_{kj} x_j$$

**Step 4**: Write in vector form.

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} = \begin{bmatrix} 2\sum_j a_{1j} x_j \\ \vdots \\ 2\sum_j a_{nj} x_j \end{bmatrix} = 2\mathbf{A}\mathbf{x}$$

$$\boxed{\nabla (\mathbf{x}^\top \mathbf{A} \mathbf{x}) = 2\mathbf{A}\mathbf{x} \quad (\mathbf{A} \text{ symmetric})}$$

**General case** (when $\mathbf{A}$ is not symmetric):

$$\nabla (\mathbf{x}^\top \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$$

**Quadratic function**: $f(\mathbf{x}) = \frac{1}{2}\|\mathbf{x}\|^2$

$$
\nabla f = \mathbf{x}
$$

**Squared norm**: $f(\mathbf{x}) = \|\mathbf{x} - \mathbf{a}\|^2$

$$
\nabla f = 2(\mathbf{x} - \mathbf{a})
$$

```python
import numpy as np

def compute_gradient(f, x, h=1e-5):
    """Numerically compute gradient"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2*h)
    return grad

# Example 1: f(x) = ||x||^2 / 2
def f1(x):
    return np.sum(x**2) / 2

# Analytical gradient: ∇f = x
x = np.array([1.0, 2.0, 3.0])
grad_numeric = compute_gradient(f1, x)
grad_analytic = x
print(f"Gradient of f(x) = ||x||²/2:")
print(f"  numerical: {grad_numeric}")
print(f"  analytic: {grad_analytic}")

# Example 2: f(x) = x^T A x (A symmetric)
A = np.array([[2, 1], [1, 3]], dtype=float)
def f2(x):
    return x @ A @ x

x = np.array([1.0, 2.0])
grad_numeric = compute_gradient(f2, x)
grad_analytic = 2 * A @ x
print(f"\nGradient of f(x) = x^T A x:")
print(f"  numerical: {grad_numeric}")
print(f"  analytic: {grad_analytic}")
```

---

## Gradient Descent Algorithm

### Basic Principle

Use the negative gradient direction for optimization, because the negative gradient direction is the direction of **fastest decrease** of the function value.

**Update rule**:

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)
$$

Where:
- $\mathbf{x}_t$: Current parameters
- $\eta > 0$: Learning rate (step size)
- $\nabla f(\mathbf{x}_t)$: Current gradient

### Convergence Conditions

Conditions for gradient descent to guarantee convergence:
1. $f$ is a convex function
2. Gradient is Lipschitz continuous
3. Learning rate is sufficiently small: $\eta < \frac{1}{L}$ (where $L$ is the Lipschitz constant)

### Effect of Learning Rate

| Learning rate | Behavior |
|--------|------|
| Too small | Convergence too slow |
| Appropriate | Stable convergence |
| Too large | Oscillation or divergence |

### Python Implementation

```python
import numpy as np

def gradient_descent(grad_fn, x_init, lr=0.01, max_iter=1000, tol=1e-6, verbose=False):
    """
    Gradient descent optimization

    Args:
        grad_fn: Gradient function
        x_init: Initial point
        lr: Learning rate
        max_iter: Maximum number of iterations
        tol: Convergence threshold
        verbose: Whether to print progress

    Returns:
        x: Optimal solution
        history: Iteration history
    """
    x = x_init.copy()
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_fn(x)
        x_new = x - lr * grad

        # Check convergence
        if np.linalg.norm(x_new - x) < tol:
            if verbose:
                print(f"Converged at iteration {i+1}")
            break

        x = x_new
        history.append(x.copy())

        if verbose and (i+1) % 100 == 0:
            print(f"Iteration {i+1}: x = {x}, ||∇f|| = {np.linalg.norm(grad):.6f}")

    return x, history

# Example 1: Minimize f(x,y) = x^2 + y^2
def grad_f1(x):
    return np.array([2*x[0], 2*x[1]])

x_init = np.array([5.0, 3.0])
x_min, history = gradient_descent(grad_f1, x_init, lr=0.1, verbose=True)
print(f"\nMinimum point: {x_min}")
print(f"Minimum value: {x_min[0]**2 + x_min[1]**2:.6f}")

# Example 2: Minimize Rosenbrock function
def rosenbrock(x):
    """f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    df_dx = -2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2)
    df_dy = 200*(x[1] - x[0]**2)
    return np.array([df_dx, df_dy])

x_init = np.array([-1.0, 1.0])
x_min, history = gradient_descent(grad_rosenbrock, x_init, lr=0.001, max_iter=10000)
print(f"\nRosenbrock minimum point: {x_min}")
print(f"Minimum value: {rosenbrock(x_min):.6f}")
```

### Momentum Method

Accelerate convergence and reduce oscillations:

$$
\mathbf{v}_{t+1} = \gamma \mathbf{v}_t + \eta \nabla f(\mathbf{x}_t)
$$

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{v}_{t+1}
$$

```python
def gradient_descent_with_momentum(grad_fn, x_init, lr=0.01, momentum=0.9,
                                    max_iter=1000, tol=1e-6):
    """Gradient descent with momentum"""
    x = x_init.copy()
    v = np.zeros_like(x)
    history = [x.copy()]

    for i in range(max_iter):
        grad = grad_fn(x)
        v = momentum * v + lr * grad
        x_new = x - v

        if np.linalg.norm(x_new - x) < tol:
            break

        x = x_new
        history.append(x.copy())

    return x, history
```

---

## Multivariate Chain Rule

### Scalar Case

For composite function $z = f(x, y)$, where $x = g(t)$, $y = h(t)$:

$$
\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}
$$

### Vector Case

For $z = f(\mathbf{x})$, where $\mathbf{x} = \mathbf{g}(t)$:

$$
\frac{dz}{dt} = \nabla f(\mathbf{x}) \cdot \frac{d\mathbf{x}}{dt} = \sum_i \frac{\partial f}{\partial x_i} \frac{dx_i}{dt}
$$

### General Form

For $\mathbf{y} = \mathbf{f}(\mathbf{x})$, $\mathbf{x} = \mathbf{g}(\mathbf{t})$:

$$
\frac{\partial \mathbf{y}}{\partial \mathbf{t}} = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} \cdot \frac{\partial \mathbf{x}}{\partial \mathbf{t}} = \mathbf{J}_f \cdot \mathbf{J}_g
$$

Where $\mathbf{J}$ is the Jacobian matrix.

### Example

Let $z = x^2 y$, $x = \sin t$, $y = e^t$, find $\frac{dz}{dt}$.

$$
\frac{\partial z}{\partial x} = 2xy, \quad \frac{\partial z}{\partial y} = x^2
$$

$$
\frac{dx}{dt} = \cos t, \quad \frac{dy}{dt} = e^t
$$

$$
\frac{dz}{dt} = 2xy \cdot \cos t + x^2 \cdot e^t = 2\sin t \cdot e^t \cdot \cos t + \sin^2 t \cdot e^t
$$

```python
import numpy as np

# Multivariate chain rule example
def chain_rule_example(t):
    """z = x^2 * y, x = sin(t), y = exp(t)"""
    x = np.sin(t)
    y = np.exp(t)
    z = x**2 * y

    # Direct computation of dz/dt
    # z = sin^2(t) * exp(t)
    # dz/dt = 2*sin(t)*cos(t)*exp(t) + sin^2(t)*exp(t)
    dz_dt_direct = 2*np.sin(t)*np.cos(t)*np.exp(t) + np.sin(t)**2*np.exp(t)

    # Using chain rule
    dz_dx = 2*x*y
    dz_dy = x**2
    dx_dt = np.cos(t)
    dy_dt = np.exp(t)
    dz_dt_chain = dz_dx * dx_dt + dz_dy * dy_dt

    return z, dz_dt_direct, dz_dt_chain

t = 1.0
z, direct, chain = chain_rule_example(t)
print(f"t = {t}")
print(f"z = {z:.6f}")
print(f"dz/dt direct: {direct:.6f}")
print(f"dz/dt chain rule: {chain:.6f}")
```

---

## Total Differential

### Definition

The **total differential** of multivariate function $f(x_1, \ldots, x_n)$ is:

$$
df = \frac{\partial f}{\partial x_1}dx_1 + \frac{\partial f}{\partial x_2}dx_2 + \cdots + \frac{\partial f}{\partial x_n}dx_n
$$

Or written in vector form:

$$
df = \nabla f \cdot d\mathbf{x}
$$

### Geometric Meaning

The total differential describes the **linear approximate change** of the function in all directions.

### Approximate Computation

$$
f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x}) \cdot \Delta\mathbf{x}
$$

### Example

Let $f(x, y) = x^2 y$, the total differential at point $(2, 3)$ is:

$$
df = 2xy \cdot dx + x^2 \cdot dy = 12 \cdot dx + 4 \cdot dy
$$

```python
import numpy as np

def total_differential_example():
    """Total differential example: f(x,y) = x^2 * y"""

    def f(x, y):
        return x**2 * y

    # At (2, 3)
    x0, y0 = 2.0, 3.0

    # Partial derivatives
    df_dx = 2 * x0 * y0  # 12
    df_dy = x0**2        # 4

    # Use total differential approximation
    dx, dy = 0.1, -0.05
    df_approx = df_dx * dx + df_dy * dy

    # Actual change
    f_old = f(x0, y0)
    f_new = f(x0 + dx, y0 + dy)
    df_actual = f_new - f_old

    print(f"f({x0}, {y0}) = {f_old}")
    print(f"f({x0+dx}, {y0+dy}) = {f_new}")
    print(f"Actual change: {df_actual:.6f}")
    print(f"Total differential approximation: {df_approx:.6f}")
    print(f"Relative error: {abs(df_actual - df_approx) / abs(df_actual):.4%}")

total_differential_example()
```

---

## Applications in Deep Learning

### Gradient Propagation in Multi-layer Networks

Suppose the network structure is: $\mathbf{x} \to \mathbf{h}_1 \to \mathbf{h}_2 \to \mathbf{y} \to L$

Where:
- $\mathbf{h}_1 = f_1(\mathbf{x}; \mathbf{W}_1)$
- $\mathbf{h}_2 = f_2(\mathbf{h}_1; \mathbf{W}_2)$
- $\mathbf{y} = f_3(\mathbf{h}_2; \mathbf{W}_3)$
- $L = \ell(\mathbf{y}, \mathbf{y}_{true})$

**Backpropagation** (chain rule):

$$
\frac{\partial L}{\partial \mathbf{y}} = \frac{\partial \ell}{\partial \mathbf{y}}
$$

$$
\frac{\partial L}{\partial \mathbf{h}_2} = \frac{\partial L}{\partial \mathbf{y}} \cdot \frac{\partial \mathbf{y}}{\partial \mathbf{h}_2}
$$

$$
\frac{\partial L}{\partial \mathbf{h}_1} = \frac{\partial L}{\partial \mathbf{h}_2} \cdot \frac{\partial \mathbf{h}_2}{\partial \mathbf{h}_1}
$$

$$
\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{h}_1} \cdot \frac{\partial \mathbf{h}_1}{\partial \mathbf{x}}
$$

### Neural Network Gradient Computation Example

```python
import numpy as np

class SimpleNN:
    """Simple two-layer neural network"""

    def __init__(self, input_size, hidden_size, output_size):
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        """Forward propagation"""
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.h1 = self.relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        return self.z2

    def backward(self, grad_output):
        """Backpropagation"""
        # Second layer
        self.grad_W2 = self.h1.T @ grad_output
        self.grad_b2 = np.sum(grad_output, axis=0)
        grad_h1 = grad_output @ self.W2.T

        # ReLU backward
        grad_z1 = grad_h1 * self.relu_grad(self.z1)

        # First layer
        self.grad_W1 = self.x.T @ grad_z1
        self.grad_b1 = np.sum(grad_z1, axis=0)

        return grad_z1 @ self.W1.T

    def get_gradients(self):
        """Return all gradients"""
        return {
            'W1': self.grad_W1, 'b1': self.grad_b1,
            'W2': self.grad_W2, 'b2': self.grad_b2
        }

# Example
np.random.seed(42)
nn = SimpleNN(10, 20, 5)
x = np.random.randn(32, 10)  # batch of 32

# Forward propagation
output = nn.forward(x)

# Assume gradient of loss function
grad_output = np.random.randn(32, 5)

# Backpropagation
grad_input = nn.backward(grad_output)

gradients = nn.get_gradients()
print("Gradient shapes:")
for name, grad in gradients.items():
    print(f"  {name}: {grad.shape}")
```

### Gradient Checking

```python
def gradient_check_neural_network(nn, x, epsilon=1e-5):
    """Neural network gradient checking"""
    # Compute analytical gradients
    output = nn.forward(x)
    grad_output = np.ones_like(output)
    nn.backward(grad_output)
    analytic_grads = nn.get_gradients()

    print("Gradient checking results:")
    for param_name in ['W1', 'b1', 'W2', 'b2']:
        param = getattr(nn, param_name)
        analytic = analytic_grads[param_name]
        numeric = np.zeros_like(param)

        # Numerical gradient
        it = np.nditer(param, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            old_value = param[idx]

            param[idx] = old_value + epsilon
            output_plus = nn.forward(x)
            loss_plus = np.sum(output_plus)

            param[idx] = old_value - epsilon
            output_minus = nn.forward(x)
            loss_minus = np.sum(output_minus)

            numeric[idx] = (loss_plus - loss_minus) / (2 * epsilon)
            param[idx] = old_value
            it.iternext()

        # Compute relative error
        diff = np.abs(numeric - analytic)
        rel_error = diff / (np.abs(numeric) + np.abs(analytic) + 1e-8)
        max_error = np.max(rel_error)

        status = "✓ passed" if max_error < 1e-5 else "✗ failed"
        print(f"  {param_name}: max relative error = {max_error:.2e} {status}")

# Run gradient checking
np.random.seed(42)
nn = SimpleNN(5, 10, 3)
x = np.random.randn(4, 5)
gradient_check_neural_network(nn, x)
```

---

## Summary

This chapter introduced partial derivatives, gradients, and multivariate differentiation:

| Concept | Definition/Formula | Application |
|------|----------|------|
| Partial derivative | $\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(\ldots, x_i+h, \ldots) - f(\ldots)}{h}$ | Derivative of multivariate function with respect to single variable |
| Gradient | $\nabla f = [\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}]^\top$ | Optimization direction |
| Directional derivative | $D_{\mathbf{u}} f = \nabla f \cdot \mathbf{u}$ | Rate of change in arbitrary direction |
| Gradient descent | $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f$ | Parameter optimization |
| Chain rule | $\frac{dz}{dt} = \sum_i \frac{\partial z}{\partial x_i} \frac{dx_i}{dt}$ | Backpropagation |

### Key Concepts

1. **Gradient direction**: Direction of fastest increase of function value
2. **Negative gradient direction**: Direction of fastest decrease of function value, core of gradient descent
3. **Directional derivative**: Rate of change in arbitrary direction, projection of gradient
4. **Chain rule**: Foundation for connecting gradient propagation across layers

---

**Previous section**: [Chapter 2(a): Derivatives and Differentiation Basics](02a-derivatives-differentiation-basics_EN.md)

**Next section**: [Chapter 2(c): Higher-order Derivatives and Taylor Expansion](02c-higher-derivatives-taylor_EN.md) - Learn about second-order derivatives, Hessian matrices, and Taylor expansions.

**Return to**: [Mathematics Fundamentals Tutorial Index](../math-fundamentals.md)
