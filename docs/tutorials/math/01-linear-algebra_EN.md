# Chapter 1: Linear Algebra

Linear algebra is the **core mathematical foundation** of deep learning. All computations in neural networks—from the simplest fully connected layers to complex Transformers—are essentially linear algebra operations. This chapter systematically introduces the linear algebra knowledge needed for deep learning.

---

## Chapter Structure

For ease of learning and in-depth understanding, this chapter is divided into four sub-chapters:

### [1.1 Vectors and Matrices Basics](01a-vectors-matrices-basics_EN.md)

**Content Overview**:
- Definition and representation of scalars, vectors, matrices, tensors
- Vector operations: addition, scalar multiplication, dot product, outer product, Hadamard product
- Matrix operations: addition, multiplication, transpose, inverse
- Python/NumPy implementation

**Core Concepts**:

| Concept | Application in Deep Learning |
|---------|----------------------------|
| Scalar | Learning rate, loss value, regularization coefficient |
| Vector | Word embeddings, bias, hidden states |
| Matrix | Weights, images, attention scores |
| Tensor | Image batches, video data, sequence data |

**[Start Learning →](01a-vectors-matrices-basics_EN.md)**

---

### [1.2 Linear Systems and Matrix Properties](01b-linear-systems-matrix-properties_EN.md)

**Content Overview**:
- Methods for solving linear systems
- Definition, properties, and geometric meaning of determinants
- Matrix rank and full rank
- Linear dependence and independence
- Four fundamental subspaces of matrices

**Core Concepts**:

| Concept | Application in Deep Learning |
|---------|----------------------------|
| Linear systems | Solving, least squares, backpropagation |
| Determinant | Determining invertibility, initialization |
| Matrix rank | Analyzing over-parameterization, low-rank approximation |
| Null space | Understanding constraints, gradients |

**[Start Learning →](01b-linear-systems-matrix-properties_EN.md)**

---

### [1.3 Eigenvalues and Matrix Decomposition](01c-eigenvalues-matrix-decomposition_EN.md)

**Content Overview**:
- Definition and computation of eigenvalues and eigenvectors
- Eigenvalue decomposition (EVD)
- Singular Value Decomposition (SVD) and its applications
- QR decomposition, Cholesky decomposition, LU decomposition
- Matrix decomposition method selection guide

**Core Concepts**:

| Decomposition Method | Formula | Applicable Conditions |
|---------------------|---------|----------------------|
| Eigenvalue decomposition | $\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{-1}$ | Diagonalizable square matrix |
| Singular Value decomposition | $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ | Any matrix |
| QR decomposition | $\mathbf{A} = \mathbf{Q}\mathbf{R}$ | Full column rank matrix |
| Cholesky decomposition | $\mathbf{A} = \mathbf{L}\mathbf{L}^\top$ | Symmetric positive definite matrix |

**[Start Learning →](01c-eigenvalues-matrix-decomposition_EN.md)**

---

### [1.4 Norms, Distances and Applications](01d-norms-distances-applications_EN.md)

**Content Overview**:
- Vector norms: $L_1$, $L_2$, $L_\infty$, $L_p$ norms
- Matrix norms: Frobenius, spectral norm, nuclear norm
- Distance metrics: Euclidean, Manhattan, cosine, Mahalanobis distance
- Deep learning applications: regularization, loss functions, attention mechanisms

**Core Applications**:

| Application Scenario | Norm/Distance Used |
|---------------------|-------------------|
| L2 regularization | Frobenius norm |
| L1 regularization | $L_1$ norm |
| MSE loss | $L_2^2$ norm |
| Gradient clipping | $L_2$ norm |
| Attention scaling | $L_2$ norm |
| Contrastive learning | Cosine distance |

**[Start Learning →](01d-norms-distances-applications_EN.md)**

---

## Learning Path

```
┌─────────────────────────────────────────────────────────────┐
│                   Chapter 1: Linear Algebra                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐                                    │
│  │ 1.1 Vectors &        │ ← Start here                      │
│  │ Matrices Basics     │                                    │
│  │ Scalars/Vectors/    │                                    │
│  │ Matrices/Tensors    │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │ 1.2 Linear Systems  │                                    │
│  │ & Properties        │                                    │
│  │ Det/Rank/Subspaces  │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │ 1.3 Eigenvalues &   │                                    │
│  │ Matrix Decomposition│                                    │
│  │ EVD/SVD/QR/Cholesky │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │ 1.4 Norms, Distances│                                    │
│  │ & Applications      │                                    │
│  │ Regularization/Loss │                                    │
│  └─────────────────────┘                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

### Common Formulas

**Vector Dot Product**:

$$
\mathbf{a} \cdot \mathbf{b} = \sum_{i=1}^n a_i b_i = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta
$$

**Matrix Multiplication**:

$$
(\mathbf{AB})_{ij} = \sum_{k} A_{ik} B_{kj}
$$

**SVD Decomposition**:

$$
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top
$$

**L2 Norm**:

$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}
$$

**Fully Connected Layer**:

$$
\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}
$$

**Attention Mechanism**:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{QK}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

---

## Relationship with Deep Learning

| Linear Algebra Concept | Deep Learning Application |
|----------------------|--------------------------|
| Matrix multiplication | Fully connected layers, convolutional layers (matrix form), attention mechanisms |
| Transpose | Gradient computation, dimension matching |
| Inverse matrix | Solving linear systems, certain optimization methods |
| Eigenvalues | PCA dimensionality reduction, weight initialization, stability analysis |
| SVD | Low-rank approximation, pseudo-inverse, recommendation systems |
| Norms | Regularization, gradient clipping, loss functions |
| Distance metrics | Similarity computation, contrastive learning, KNN |

---

## Prerequisites

- Basic algebraic operations
- Functions and variables
- Basic Python programming

## Recommended Learning Order

1. **1.1 Vectors and Matrices Basics** → Understand basic data structures and operations
2. **1.2 Linear Systems and Matrix Properties** → Deep understanding of matrix characteristics
3. **1.3 Eigenvalues and Matrix Decomposition** → Master advanced analysis tools
4. **1.4 Norms, Distances and Applications** → Understand practical application scenarios

---

**Next Chapter**: [Chapter 2: Calculus](02-calculus_EN.md) - Learn derivatives, gradients, chain rule, and understand the mathematical principles of backpropagation.

**Back to**: [Mathematical Foundations Directory](../math-fundamentals_EN.md)
