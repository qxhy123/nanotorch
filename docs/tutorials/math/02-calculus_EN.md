# Chapter 2: Calculus

Calculus is the mathematical foundation for understanding **gradient descent** and **backpropagation**. The core of deep learning is computing derivatives (gradients) of the loss function with respect to parameters, then updating parameters in the opposite direction of the gradient. This chapter systematically introduces core calculus concepts and their applications in deep learning.

---

## Chapter Structure

For ease of learning and in-depth understanding, this chapter is divided into four sub-chapters:

### [2.1 Derivatives and Differentiation Basics](02a-derivatives-differentiation-basics_EN.md)

**Content Overview**:
- Functions and limits
- Definition, geometric meaning, and physical meaning of derivatives
- Basic derivative formulas table
- Differentiation rules: linearity rule, product rule, quotient rule
- Chain rule and its applications

**Core Concepts**:
| Concept | Formula | Application |
|---------|---------|-------------|
| Derivative | $f'(x) = \lim_{h \to 0} \frac{f(x+h)-f(x)}{h}$ | Computing rate of change |
| Product rule | $(fg)' = f'g + fg'$ | Derivative of composite functions |
| Chain rule | $[f(g(x))]' = f'(g(x)) \cdot g'(x)$ | Foundation of backpropagation |

**[Start Learning →](02a-derivatives-differentiation-basics_EN.md)**

---

### [2.2 Partial Derivatives, Gradients and Multivariate Calculus](02b-partial-derivatives-gradients_EN.md)

**Content Overview**:
- Definition and computation of partial derivatives
- Relationship between directional derivatives and gradients
- Properties and geometric meaning of gradients
- Gradient descent algorithm
- Multivariate chain rule and total differential

**Core Concepts**:
| Concept | Formula | Application |
|---------|---------|-------------|
| Partial derivative | $\frac{\partial f}{\partial x_i}$ | Derivative of multivariate function with respect to single variable |
| Gradient | $\nabla f = [\frac{\partial f}{\partial x_1}, \ldots]^\top$ | Optimization direction |
| Gradient descent | $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f$ | Parameter optimization |

**[Start Learning →](02b-partial-derivatives-gradients_EN.md)**

---

### [2.3 Higher-Order Derivatives and Taylor Expansion](02c-higher-derivatives-taylor_EN.md)

**Content Overview**:
- Second derivatives and convexity/concavity
- Definition and properties of Hessian matrix
- Taylor expansion (univariate and multivariate)
- Newton's method and quasi-Newton methods
- Integration basics (connection to probability theory)

**Core Concepts**:
| Concept | Formula | Application |
|---------|---------|-------------|
| Hessian matrix | $H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$ | Analyzing optimization curvature |
| Taylor expansion | $f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f^\top \Delta\mathbf{x} + \frac{1}{2}\Delta\mathbf{x}^\top \mathbf{H} \Delta\mathbf{x}$ | Local approximation |
| Newton's method | $\mathbf{x}_{t+1} = \mathbf{x}_t - \mathbf{H}^{-1}\nabla f$ | Second-order optimization |

**[Start Learning →](02c-higher-derivatives-taylor_EN.md)**

---

### [2.4 Vector/Matrix Calculus and Backpropagation](02d-vector-matrix-calculus-backprop_EN.md)

**Content Overview**:
- Vector derivatives and Jacobian matrix
- Matrix derivative formulas
- Gradients of linear layers, activation functions, loss functions
- Complete backpropagation algorithm implementation
- Numerical gradient verification

**Core Concepts**:
| Concept | Formula | Application |
|---------|---------|-------------|
| Linear layer gradient | $\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top \frac{\partial L}{\partial \mathbf{Y}}$ | Backpropagation |
| Softmax+Cross-entropy | $\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$ | Classification output |
| Jacobian matrix | $\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}$ | Vector-valued function derivatives |

**[Start Learning →](02d-vector-matrix-calculus-backprop_EN.md)**

---

## Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│                     Chapter 2: Calculus                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐                                    │
│  │ 2.1 Derivatives &    │ ← Start here                      │
│  │ Differentiation      │                                    │
│  │ Basics               │                                    │
│  │ Functions/Limits/    │                                    │
│  │ Derivatives/Chain    │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │ 2.2 Partial Derivs  │                                    │
│  │ & Gradients          │                                    │
│  │ Multivariate/        │                                    │
│  │ Directional derivs   │                                    │
│  │ Gradient descent     │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │ 2.3 Higher Derivs/  │                                    │
│  │ Taylor Expansion     │                                    │
│  │ Hessian/Newton's    │                                    │
│  │ Integration basics   │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │ 2.4 Vector/Matrix   │                                    │
│  │ Calculus             │                                    │
│  │ Jacobian/Matrix      │                                    │
│  │ derivatives/Backprop │                                    │
│  └─────────────────────┘                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

### Common Derivative Formulas

**Basic Derivatives**:
- $(x^n)' = nx^{n-1}$
- $(e^x)' = e^x$
- $(\ln x)' = \frac{1}{x}$
- $(\sin x)' = \cos x$
- $(\cos x)' = -\sin x$

**Differentiation Rules**:
- $(f \pm g)' = f' \pm g'$
- $(fg)' = f'g + fg'$
- $(f/g)' = \frac{f'g - fg'}{g^2}$
- $[f(g(x))]' = f'(g(x)) \cdot g'(x)$

### Gradients and Optimization

**Gradient**:
$$
\nabla f = \left[\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}\right]^\top
$$

**Gradient Descent**:
$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)
$$

### Backpropagation Core Formulas

**Linear Layer**:
$$
\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top \frac{\partial L}{\partial \mathbf{Y}}, \quad
\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \mathbf{W}^\top
$$

**Activation Functions**:
- Sigmoid: $\sigma'(x) = \sigma(x)(1-\sigma(x))$
- ReLU: $\text{ReLU}'(x) = \begin{cases}1 & x>0 \\ 0 & x \leq 0\end{cases}$
- Tanh: $\tanh'(x) = 1 - \tanh^2(x)$

**Loss Functions**:
- MSE: $\frac{\partial L}{\partial \hat{y}} = \frac{2}{n}(\hat{y} - y)$
- Softmax+Cross-entropy: $\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i$

---

## Relationship with Deep Learning

| Calculus Concept | Deep Learning Application |
|-----------------|--------------------------|
| Derivative | Understanding rate of change, activation function properties |
| Partial derivative | Multi-parameter optimization, sensitivity of loss function to each parameter |
| Gradient | Gradient descent optimization, parameter update direction |
| Chain rule | Core of backpropagation algorithm |
| Hessian matrix | Second-order optimization, analyzing optimization curvature |
| Taylor expansion | Understanding convergence of optimization algorithms |
| Jacobian matrix | Gradient propagation of vector-valued functions |

---

## Prerequisites

- Basic algebraic operations
- Functions and graphs
- Python/NumPy basics

## Recommended Learning Order

1. **2.1 Derivatives and Differentiation Basics** → Understand basic derivative concepts
2. **2.2 Partial Derivatives and Gradients** → Understand multivariate function optimization
3. **2.3 Higher-Order Derivatives and Taylor Expansion** → Deep understanding of optimization properties
4. **2.4 Vector/Matrix Calculus** → Master backpropagation implementation

---

**Previous Chapter**: [Chapter 1: Linear Algebra](01-linear-algebra_EN.md) - Learn vector, matrix, tensor operations.

**Next Chapter**: [Chapter 3: Probability Theory](03-probability_EN.md) - Learn probability distributions, expectation, Bayes' theorem and other concepts.

**Back to**: [Mathematical Foundations Directory](../math-fundamentals_EN.md)
