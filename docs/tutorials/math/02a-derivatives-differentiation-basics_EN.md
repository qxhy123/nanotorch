# Chapter 2(a): Derivatives and Differentiation Basics

Calculus is the mathematical foundation for understanding **gradient descent** and **backpropagation**. Derivatives describe the rate of change of a function at a point, which is a core concept in optimization algorithms. This chapter systematically introduces the basic concepts, calculation methods, and differentiation rules of derivatives.

---

## Table of Contents

1. [Functions and Limits](#functions-and-limits)
2. [Definition of Derivatives](#definition-of-derivatives)
3. [Geometric and Physical Meaning of Derivatives](#geometric-and-physical-meaning-of-derivatives)
4. [Basic Derivative Formulas](#basic-derivative-formulas)
5. [Differentiation Rules](#differentiation-rules)
6. [Chain Rule](#chain-rule)
7. [Derivatives and Function Properties](#derivatives-and-function-properties)
8. [Applications in Deep Learning](#applications-in-deep-learning)
9. [Summary](#summary)

---

## Functions and Limits

### Definition of Functions

**Function** $f: \mathbb{R} \to \mathbb{R}$ is a mapping rule that maps input $x$ to a unique output $y = f(x)$.

**Multivariate function**: $f: \mathbb{R}^n \to \mathbb{R}$, mapping an $n$-dimensional vector to a scalar.

$$
f(\mathbf{x}) = f(x_1, x_2, \ldots, x_n)
$$

**Vector-valued function**: $f: \mathbb{R}^n \to \mathbb{R}^m$, mapping an $n$-dimensional vector to an $m$-dimensional vector.

$$
\mathbf{f}(\mathbf{x}) = \begin{bmatrix} f_1(\mathbf{x}) \\ f_2(\mathbf{x}) \\ \vdots \\ f_m(\mathbf{x}) \end{bmatrix}
$$

### Limits

**Definition**: The limit of function $f(x)$ as $x$ approaches $a$ is $L$, denoted as:

$$
\lim_{x \to a} f(x) = L
$$

**$\epsilon$-$\delta$ definition**: For any $\epsilon > 0$, there exists $\delta > 0$ such that when $0 < |x - a| < \delta$, we have $|f(x) - L| < \epsilon$.

### Properties of Limits

**Arithmetic operations**: 

Let $\lim_{x \to a} f(x) = A$, $\lim_{x \to a} g(x) = B$

$$
\lim_{x \to a} [f(x) \pm g(x)] = A \pm B
$$

$$
\lim_{x \to a} [f(x) \cdot g(x)] = A \cdot B
$$

$$
\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{A}{B} \quad (B \neq 0)
$$

### Important Limits

$$
\lim_{x \to 0} \frac{\sin x}{x} = 1
$$

$$
\lim_{x \to 0} \frac{e^x - 1}{x} = 1
$$

$$
\lim_{x \to 0} \frac{\ln(1+x)}{x} = 1
$$

$$
\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e
$$

$$
\lim_{x \to 0} (1+x)^{\frac{1}{x}} = e
$$

### Continuity

Function $f$ is **continuous** at point $a$ if and only if:

$$
\lim_{x \to a} f(x) = f(a)
$$

**Three conditions for continuity**:
1. $f(a)$ exists
2. $\lim_{x \to a} f(x)$ exists
3. $\lim_{x \to a} f(x) = f(a)$

**Properties of continuous functions**:
- Sum, difference, product, and quotient of continuous functions (with non-zero denominator) remain continuous
- Composition of continuous functions remains continuous
- Continuous functions on closed intervals have maximum and minimum values

```python
import numpy as np
import matplotlib.pyplot as plt

# Continuous function example
x = np.linspace(-2, 2, 1000)
f_continuous = x**2
f_discontinuous = np.where(x < 0, -1, 1)  # Step function is discontinuous at 0

print("x^2 is continuous at x=0")
print("Step function is discontinuous at x=0")

# Verify limit
def limit_example(x):
    """Verify lim_{x->0} sin(x)/x = 1"""
    if x == 0:
        return None
    return np.sin(x) / x

for h in [0.1, 0.01, 0.001, 0.0001]:
    print(f"sin({h})/{h} = {limit_example(h):.6f}")
# Output approaches 1
```

---

## Definition of Derivatives

### Definition

The **derivative** of function $f(x)$ at point $x$ is defined as:

$$
f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

**Other notations**:
- Leibniz notation: $\frac{df}{dx}$, $\frac{d}{dx}f(x)$
- Lagrange notation: $f'(x)$
- Newton notation: $\dot{f}$ (commonly used for time derivatives)

### Difference Quotient

The derivative is the **limit of the difference quotient**:

$$
\text{Difference quotient} = \frac{f(x + h) - f(x)}{h}
$$

As $h \to 0$, the difference quotient approaches the derivative.

### Left and Right Derivatives

**Left derivative**:

$$
f'_-(x) = \lim_{h \to 0^-} \frac{f(x+h) - f(x)}{h}
$$

**Right derivative**:

$$
f'_+(x) = \lim_{h \to 0^+} \frac{f(x+h) - f(x)}{h}
$$

### Differentiability

Conditions for function $f$ to be **differentiable** at point $x$:
1. $f$ is continuous at $x$
2. Left derivative equals right derivative: $f'_-(x) = f'_+(x)$

**Cases where a function is not differentiable**:
- Cusp: $f(x) = |x|$ at $x = 0$
- Vertical tangent: $f(x) = \sqrt[3]{x}$ at $x = 0$
- Discontinuity points
- ReLU at $x = 0$

```python
import numpy as np

def derivative_definition(f, x, h=1e-8):
    """Compute derivative using definition"""
    return (f(x + h) - f(x)) / h

def symmetric_derivative(f, x, h=1e-8):
    """Central difference (more accurate)"""
    return (f(x + h) - f(x - h)) / (2 * h)

# Example: f(x) = x^2
def f(x):
    return x ** 2

# Analytical derivative: f'(x) = 2x
x = 3.0
print(f"f'({x}) analytic: {2*x}")
print(f"f'({x}) definition: {derivative_definition(f, x):.6f}")
print(f"f'({x}) central diff: {symmetric_derivative(f, x):.6f}")

# ReLU non-differentiability
def relu(x):
    return np.maximum(0, x)

x = 0.0
h = 1e-6
left_deriv = (relu(x) - relu(x - h)) / h
right_deriv = (relu(x + h) - relu(x)) / h
print(f"\nReLU at x=0:")
print(f"  Left derivative: {left_deriv}")
print(f"  Right derivative: {right_deriv}")
print(f"  Differentiable? {np.isclose(left_deriv, right_deriv)}")
```

---

## Geometric and Physical Meaning of Derivatives

### Geometric Meaning

The derivative is the **slope of the tangent line** at point $(x, f(x))$ on the function's graph.

**Tangent line equation**:

$$
y - f(a) = f'(a)(x - a)
$$

**Normal line equation** (perpendicular to tangent):

$$
y - f(a) = -\frac{1}{f'(a)}(x - a) \quad (f'(a) \neq 0)
$$

### Physical Meaning

| If $f(t)$ represents | Then $f'(t)$ represents |
|-----------------|-----------------|
| Position | Velocity |
| Velocity | Acceleration |
| Charge | Current |
| Energy | Power |

### Instantaneous Rate of Change

The derivative represents the **instantaneous rate of change** of the function at a point:

$$
f'(x) = \text{Instantaneous rate of change at } x
$$

```python
import numpy as np

def tangent_line(f, f_prime, a, x):
    """Compute value of tangent at x"""
    return f(a) + f_prime(a) * (x - a)

# Example: f(x) = x^2
def f(x):
    return x ** 2

def f_prime(x):
    return 2 * x

a = 2.0  # Tangent point
x_vals = np.linspace(0, 4, 100)

# Original function
y_func = f(x_vals)
# Tangent line
y_tangent = tangent_line(f, f_prime, a, x_vals)

print(f"At x = {a}:")
print(f"  Function value: f({a}) = {f(a)}")
print(f"  Derivative value: f'({a}) = {f_prime(a)}")
print(f"  Tangent line equation: y = {f_prime(a)}(x - {a}) + {f(a)} = {f_prime(a)}x - {f_prime(a)*a - f(a)}")
```

---

## Basic Derivative Formulas

### Power Functions and Polynomials

| Function $f(x)$ | Derivative $f'(x)$ |
|-------------|--------------|
| $c$ (constant) | $0$ |
| $x$ | $1$ |
| $x^n$ | $nx^{n-1}$ |
| $\sqrt{x} = x^{1/2}$ | $\frac{1}{2\sqrt{x}}$ |
| $\frac{1}{x} = x^{-1}$ | $-\frac{1}{x^2}$ |

### Exponential and Logarithmic Functions

| Function $f(x)$ | Derivative $f'(x)$ |
|-------------|--------------|
| $e^x$ | $e^x$ |
| $a^x$ | $a^x \ln a$ |
| $\ln x$ | $\frac{1}{x}$ |
| $\log_a x$ | $\frac{1}{x \ln a}$ |

### Trigonometric Functions

| Function $f(x)$ | Derivative $f'(x)$ |
|-------------|--------------|
| $\sin x$ | $\cos x$ |
| $\cos x$ | $-\sin x$ |
| $\tan x$ | $\sec^2 x = \frac{1}{\cos^2 x}$ |
| $\cot x$ | $-\csc^2 x = -\frac{1}{\sin^2 x}$ |
| $\sec x$ | $\sec x \tan x$ |
| $\csc x$ | $-\csc x \cot x$ |

### Inverse Trigonometric Functions

| Function $f(x)$ | Derivative $f'(x)$ |
|-------------|--------------|
| $\arcsin x$ | $\frac{1}{\sqrt{1-x^2}}$ |
| $\arccos x$ | $-\frac{1}{\sqrt{1-x^2}}$ |
| $\arctan x$ | $\frac{1}{1+x^2}$ |
| $\text{arccot } x$ | $-\frac{1}{1+x^2}$ |

### Hyperbolic Functions

| Function $f(x)$ | Derivative $f'(x)$ |
|-------------|--------------|
| $\sinh x = \frac{e^x - e^{-x}}{2}$ | $\cosh x$ |
| $\cosh x = \frac{e^x + e^{-x}}{2}$ | $\sinh x$ |
| $\tanh x = \frac{\sinh x}{\cosh x}$ | $\text{sech}^2 x = 1 - \tanh^2 x$ |

```python
import numpy as np

# Verify derivative formulas
def verify_derivative(f_name, f, f_prime, x):
    """Verify derivative formula"""
    numeric = (f(x + 1e-8) - f(x - 1e-8)) / (2e-8)
    analytic = f_prime(x)
    error = abs(numeric - analytic)
    print(f"{f_name}: analytic={analytic:.6f}, numeric={numeric:.6f}, error={error:.2e}")

# Test points
x = 0.5

verify_derivative("sin(x)", np.sin, np.cos, x)
verify_derivative("exp(x)", np.exp, np.exp, x)
verify_derivative("ln(x)", np.log, lambda x: 1/x, x)
verify_derivative("x^3", lambda x: x**3, lambda x: 3*x**2, x)
verify_derivative("tanh(x)", np.tanh, lambda x: 1 - np.tanh(x)**2, x)
```

---

## Differentiation Rules

### Linear Rules

**Constant multiple rule**:

$$
(cf)' = cf'
$$

**Sum and difference rule**:

$$
(f \pm g)' = f' \pm g'
$$

### Product Rule

$$
(fg)' = f'g + fg'
$$

**Memory aid**: "Derivative of first times second plus first times derivative of second"

**Derivation of the product rule**:

**Step 1**: Use the definition of derivative.

$$(fg)'(x) = \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x)g(x)}{h}$$

**Step 2**: Add and subtract $f(x+h)g(x)$.

$$= \lim_{h \to 0} \frac{f(x+h)g(x+h) - f(x+h)g(x) + f(x+h)g(x) - f(x)g(x)}{h}$$

**Step 3**: Group and factor.

$$= \lim_{h \to 0} \left[ f(x+h) \cdot \frac{g(x+h) - g(x)}{h} + \frac{f(x+h) - f(x)}{h} \cdot g(x) \right]$$

**Step 4**: Take the limit, using continuity of $f$ ($f(x+h) \to f(x)$).

$$= f(x) \cdot g'(x) + f'(x) \cdot g(x)$$

$$\boxed{(fg)' = f'g + fg'}$$

**Extension**: Product of three functions

$$(fgh)' = f'gh + fg'h + fgh'$$

### Quotient Rule

$$
\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}
$$

**Memory aid**: "Derivative of top times bottom minus top times derivative of bottom, all divided by bottom squared"

**Derivation of the quotient rule**:

**Step 1**: Use the definition of derivative.

$$\left(\frac{f}{g}\right)'(x) = \lim_{h \to 0} \frac{\frac{f(x+h)}{g(x+h)} - \frac{f(x)}{g(x)}}{h}$$

**Step 2**: Find common denominator.

$$= \lim_{h \to 0} \frac{f(x+h)g(x) - f(x)g(x+h)}{h \cdot g(x+h)g(x)}$$

**Step 3**: Add and subtract $f(x)g(x)$.

$$= \lim_{h \to 0} \frac{f(x+h)g(x) - f(x)g(x) + f(x)g(x) - f(x)g(x+h)}{h \cdot g(x+h)g(x)}$$

**Step 4**: Group.

$$= \lim_{h \to 0} \frac{g(x) \cdot \frac{f(x+h) - f(x)}{h} - f(x) \cdot \frac{g(x+h) - g(x)}{h}}{g(x+h)g(x)}$$

**Step 5**: Take the limit, using continuity of $g$ ($g(x+h) \to g(x)$).

$$= \frac{g(x)f'(x) - f(x)g'(x)}{g(x)^2}$$

$$\boxed{\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}}$$

```python
import numpy as np

# Verify product rule: (f*g)' = f'*g + f*g'
def product_rule_check(f, g, f_prime, g_prime, x):
    """Verify product rule"""
    fg = lambda x: f(x) * g(x)

    # Left side: numerical derivative of (f*g)
    lhs_numeric = (fg(x + 1e-8) - fg(x - 1e-8)) / (2e-8)

    # Right side: f'*g + f*g'
    rhs = f_prime(x) * g(x) + f(x) * g_prime(x)

    return lhs_numeric, rhs

# Example: f(x) = x^2, g(x) = sin(x)
f = lambda x: x**2
g = lambda x: np.sin(x)
f_prime = lambda x: 2*x
g_prime = lambda x: np.cos(x)

x = 1.0
lhs, rhs = product_rule_check(f, g, f_prime, g_prime, x)
print(f"Product rule verification:")
print(f"  (f*g)' numerical: {lhs:.6f}")
print(f"  f'*g + f*g': {rhs:.6f}")
print(f"  Match? {np.isclose(lhs, rhs)}")

# Verify quotient rule: (f/g)' = (f'g - fg') / g^2
def quotient_rule_check(f, g, f_prime, g_prime, x):
    """Verify quotient rule"""
    fg = lambda x: f(x) / g(x)

    lhs_numeric = (fg(x + 1e-8) - fg(x - 1e-8)) / (2e-8)
    rhs = (f_prime(x) * g(x) - f(x) * g_prime(x)) / (g(x) ** 2)

    return lhs_numeric, rhs

lhs, rhs = quotient_rule_check(f, g, f_prime, g_prime, x)
print(f"\nQuotient rule verification:")
print(f"  (f/g)' numerical: {lhs:.6f}")
print(f"  (f'g - fg')/g^2: {rhs:.6f}")
print(f"  Match? {np.isclose(lhs, rhs)}")
```

---

## Chain Rule

### Single-variable Chain Rule

For composite function $y = f(g(x))$:

$$
\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx} = f'(g(x)) \cdot g'(x)
$$

**Memory aid**: "Derivative of outer function times derivative of inner function"

### Intuitive Understanding of Chain Rule

If $y$ depends on $u$, and $u$ depends on $x$, then:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

### Derivatives of Common Composite Functions

| Function $f(x)$ | Derivative $f'(x)$ |
|-------------|--------------|
| $\ln(g(x))$ | $\frac{g'(x)}{g(x)}$ |
| $e^{g(x)}$ | $g'(x) e^{g(x)}$ |
| $(g(x))^n$ | $n(g(x))^{n-1} g'(x)$ |
| $\sin(g(x))$ | $g'(x) \cos(g(x))$ |
| $\cos(g(x))$ | $-g'(x) \sin(g(x))$ |
| $\sqrt{g(x)}$ | $\frac{g'(x)}{2\sqrt{g(x)}}$ |

### Chain Rule Examples

**Example 1**: Find $(e^{x^2})'$

Let $u = x^2$, then $y = e^u$:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx} = e^u \cdot 2x = 2xe^{x^2}
$$

**Example 2**: Find $(\sin(3x))'$

$$
\frac{d}{dx}\sin(3x) = \cos(3x) \cdot 3 = 3\cos(3x)
$$

**Example 3**: Find $(\ln(x^2 + 1))'$

$$
\frac{d}{dx}\ln(x^2 + 1) = \frac{1}{x^2 + 1} \cdot 2x = \frac{2x}{x^2 + 1}
$$

```python
import numpy as np

def chain_rule_example():
    """Chain rule examples"""

    # Example 1: y = e^{x^2}
    def y1(x):
        return np.exp(x**2)

    def dy1_dx(x):
        # dy/du * du/dx = e^{x^2} * 2x
        return np.exp(x**2) * 2 * x

    # Example 2: y = sin(3x)
    def y2(x):
        return np.sin(3*x)

    def dy2_dx(x):
        return np.cos(3*x) * 3

    # Example 3: y = ln(x^2 + 1)
    def y3(x):
        return np.log(x**2 + 1)

    def dy3_dx(x):
        return 2*x / (x**2 + 1)

    # Verify
    x = 1.5

    for name, y, dy_dx in [("e^{x^2}", y1, dy1_dx),
                            ("sin(3x)", y2, dy2_dx),
                            ("ln(x^2+1)", y3, dy3_dx)]:
        numeric = (y(x + 1e-8) - y(x - 1e-8)) / (2e-8)
        analytic = dy_dx(x)
        print(f"{name}: analytic={analytic:.6f}, numeric={numeric:.6f}")

chain_rule_example()
```

---

## Derivatives and Function Properties

### Monotonicity

| Condition | Conclusion |
|------|------|
| $f'(x) > 0$ | $f$ is monotonically increasing |
| $f'(x) < 0$ | $f$ is monotonically decreasing |
| $f'(x) = 0$ | Possibly an extremum point |

### Concavity/Convexity

| Condition | Conclusion |
|------|------|
| $f''(x) > 0$ | $f$ is convex (opens upward) |
| $f''(x) < 0$ | $f$ is concave (opens downward) |
| $f''(x) = 0$ | Possibly an inflection point |

### Extremum Determination

**First derivative test**:
- If $f'$ changes from positive to negative at $x_0$, then $x_0$ is a local maximum
- If $f'$ changes from negative to positive at $x_0$, then $x_0$ is a local minimum

**Second derivative test**:
- If $f'(x_0) = 0$ and $f''(x_0) > 0$, then $x_0$ is a local minimum
- If $f'(x_0) = 0$ and $f''(x_0) < 0$, then $x_0$ is a local maximum
- If $f'(x_0) = 0$ and $f''(x_0) = 0$, further judgment is needed

### Inflection Points

Points where the concavity of the function changes. **Necessary condition** for $x_0$ to be an inflection point:

$$
f''(x_0) = 0 \text{ or } f''(x_0) \text{ does not exist}
$$

```python
import numpy as np

def analyze_function(f, f_prime, f_double_prime, x_range):
    """Analyze monotonicity and concavity of a function"""
    x = np.linspace(x_range[0], x_range[1], 1000)

    # Find extremum points (f' = 0)
    deriv = f_prime(x)
    sign_changes = np.where(np.diff(np.sign(deriv)))[0]

    print("Extremum points:")
    for idx in sign_changes:
        x_extreme = x[idx]
        if f_double_prime(x_extreme) > 0:
            print(f"  x = {x_extreme:.4f}: local minimum = {f(x_extreme):.4f}")
        else:
            print(f"  x = {x_extreme:.4f}: local maximum = {f(x_extreme):.4f}")

    # Find inflection points (f'' = 0)
    second_deriv = f_double_prime(x)
    inflection_indices = np.where(np.diff(np.sign(second_deriv)))[0]

    print("\nInflection points:")
    for idx in inflection_indices:
        x_inf = x[idx]
        print(f"  x = {x_inf:.4f}: f(x) = {f(x_inf):.4f}")

# Example: f(x) = x^3 - 3x
f = lambda x: x**3 - 3*x
f_prime = lambda x: 3*x**2 - 3
f_double_prime = lambda x: 6*x

analyze_function(f, f_prime, f_double_prime, [-3, 3])
```

---

## Applications in Deep Learning

### Derivatives of Activation Functions

**Sigmoid**: $\sigma(x) = \frac{1}{1+e^{-x}}$

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

**Derivation of Sigmoid derivative**:

**Step 1**: Let $u = 1 + e^{-x}$, then $\sigma(x) = \frac{1}{u} = u^{-1}$.

**Step 2**: Use the quotient rule (or chain rule).

$$\sigma'(x) = \frac{d}{dx}\left(\frac{1}{1+e^{-x}}\right)$$

Using the quotient rule $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$, where $f = 1$, $g = 1+e^{-x}$:

$$\sigma'(x) = \frac{0 \cdot (1+e^{-x}) - 1 \cdot \frac{d}{dx}(1+e^{-x})}{(1+e^{-x})^2}$$

$$= \frac{-\frac{d}{dx}(e^{-x})}{(1+e^{-x})^2}$$

**Step 3**: Compute $\frac{d}{dx}(e^{-x})$.

Using the chain rule: $\frac{d}{dx}(e^{-x}) = e^{-x} \cdot \frac{d}{dx}(-x) = e^{-x} \cdot (-1) = -e^{-x}$

**Step 4**: Substitute.

$$\sigma'(x) = \frac{-(-e^{-x})}{(1+e^{-x})^2} = \frac{e^{-x}}{(1+e^{-x})^2}$$

**Step 5**: Rewrite in terms of $\sigma(x)$.

Note that $\sigma(x) = \frac{1}{1+e^{-x}}$, so $1 - \sigma(x) = 1 - \frac{1}{1+e^{-x}} = \frac{e^{-x}}{1+e^{-x}}$.

Therefore:

$$\sigma'(x) = \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} = \sigma(x) \cdot (1 - \sigma(x))$$

$$\boxed{\sigma'(x) = \sigma(x)(1 - \sigma(x))}$$

**Intuitive understanding**:
- When $\sigma(x) \approx 0.5$ (i.e., $x \approx 0$), the derivative is at maximum, about $0.5 \times 0.5 = 0.25$
- When $\sigma(x) \approx 0$ or $\sigma(x) \approx 1$ (i.e., $|x|$ is large), the derivative approaches 0

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Verify
x = 1.0
numeric = (sigmoid(x + 1e-8) - sigmoid(x - 1e-8)) / (2e-8)
analytic = sigmoid_derivative(x)
print(f"Sigmoid'({x}): analytic={analytic:.6f}, numeric={numeric:.6f}")
```

**ReLU**: $\text{ReLU}(x) = \max(0, x)$

$$
\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \\ \text{undefined} & x = 0 \end{cases}
$$

```python
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

# ReLU is not differentiable at 0, but in practice we usually take 0 or 1
```

**Tanh**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

$$
\tanh'(x) = 1 - \tanh^2(x) = \text{sech}^2(x)
$$

**Derivation of Tanh derivative**:

**Method 1: Using definition**

**Step 1**: Let $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{u}{v}$, where $u = e^x - e^{-x}$, $v = e^x + e^{-x}$.

**Step 2**: Use the quotient rule $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$.

Compute $u'$ and $v'$:
- $u' = e^x - (-e^{-x}) = e^x + e^{-x} = v$
- $v' = e^x - e^{-x} = u$

**Step 3**: Substitute into the quotient rule.

$$\tanh'(x) = \frac{v \cdot v - u \cdot u}{v^2} = \frac{v^2 - u^2}{v^2} = 1 - \frac{u^2}{v^2} = 1 - \tanh^2(x)$$

$$\boxed{\tanh'(x) = 1 - \tanh^2(x)}$$

**Method 2: Using Sigmoid**

Since $\tanh(x) = 2\sigma(2x) - 1$:

$$\tanh'(x) = 2 \cdot \sigma'(2x) \cdot 2 = 4\sigma(2x)(1-\sigma(2x))$$

Let $s = \sigma(2x)$, then $\tanh(x) = 2s - 1$, $s = \frac{\tanh(x) + 1}{2}$.

$$\tanh'(x) = 4s(1-s) = 4 \cdot \frac{\tanh+1}{2} \cdot \frac{1-\tanh}{2} = (1+\tanh)(1-\tanh) = 1 - \tanh^2$$

**Geometric meaning**:
- When $x = 0$, $\tanh(0) = 0$, $\tanh'(0) = 1$ (maximum derivative)
- When $|x| \to \infty$, $|\tanh| \to 1$, $\tanh' \to 0$ (saturation region)

### Numerical Derivative Computation

```python
def numerical_derivative(f, x, h=1e-5):
    """Central difference numerical derivative"""
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_gradient(f, x, h=1e-5):
    """Compute numerical gradient at vector x"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

# Example
def f(x):
    return x[0]**2 + x[1]**2

x = np.array([1.0, 2.0])
grad = numerical_gradient(f, x)
print(f"∇f({x}) = {grad}")  # Should be close to [2, 4]
```

### Gradient Checking

```python
def gradient_check(f, analytic_grad, x, h=1e-5, threshold=1e-7):
    """Check if analytical gradient is correct"""
    numeric_grad = numerical_gradient(f, x, h)

    diff = np.abs(numeric_grad - analytic_grad(x))
    rel_error = diff / (np.abs(numeric_grad) + np.abs(analytic_grad(x)) + 1e-8)

    max_error = np.max(rel_error)
    passed = max_error < threshold

    return passed, max_error, numeric_grad

# Example
def f(x):
    return np.sum(x ** 2)

def df(x):
    return 2 * x

x = np.random.randn(5)
passed, error, numeric = gradient_check(f, df, x)
print(f"Gradient check: {'passed' if passed else 'failed'}, max relative error: {error:.2e}")
```

---

## Summary

This chapter introduced the basics of derivatives and differentiation:

| Concept | Definition/Formula | Application |
|------|----------|------|
| Derivative | $f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}$ | Calculate rate of change |
| Geometric meaning | Slope of tangent line | Understand function behavior |
| Product rule | $(fg)' = f'g + fg'$ | Differentiate composite functions |
| Quotient rule | $(f/g)' = \frac{f'g - fg'}{g^2}$ | Differentiate rational functions |
| Chain rule | $\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$ | Foundation for backpropagation |

### Key Formulas

**Basic derivatives**:
- $(x^n)' = nx^{n-1}$
- $(e^x)' = e^x$
- $(\ln x)' = \frac{1}{x}$
- $(\sin x)' = \cos x$

**Differentiation rules**:
- $(f \pm g)' = f' \pm g'$
- $(fg)' = f'g + fg'$
- $(f/g)' = \frac{f'g - fg'}{g^2}$
- $[f(g(x))]' = f'(g(x)) \cdot g'(x)$

---

**Next section**: [Chapter 2(b): Partial Derivatives, Gradients, and Multivariate Differentiation](02b-partial-derivatives-gradients_EN.md) - Learn about differentiation of multivariate functions, gradient descent, and directional derivatives.

**Return to**: [Mathematics Fundamentals Tutorial Index](../math-fundamentals.md)
