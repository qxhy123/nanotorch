# Chapter 1(d): Norms, Distances, and Applications

Norms are tools for measuring the "size" of vectors or matrices, while distance metrics are methods for measuring differences between two objects. These concepts have extensive applications in deep learning, from regularization to loss functions, and even attention mechanisms. This chapter will systematically introduce these concepts and their practical applications.

---

## Table of Contents

1. [Vector Norms](#vector-norms)
2. [Matrix Norms](#matrix-norms)
3. [Distance Metrics](#distance-metrics)
4. [Similarity Metrics](#similarity-metrics)
5. [Applications in Deep Learning](#applications-in-deep-learning)
6. [Summary](#summary)

---

## Vector Norms

### Definition

**Norms** are functions that map vectors to non-negative real numbers $\|\cdot\|: \mathbb{R}^n \to \mathbb{R}_{\geq 0}$, satisfying three properties:

1. **Non-negativity**: $\|\mathbf{x}\| \geq 0$, equality holds if and only if $\mathbf{x} = \mathbf{0}$
2. **Homogeneity**: $\|\alpha\mathbf{x}\| = |\alpha|\|\mathbf{x}\|$, for any scalar $\alpha$
3. **Triangle inequality**: $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$

### $L_p$ Norms

**Definition**:

$$
\|\mathbf{x}\|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}, \quad p \geq 1
$$

### Common Vector Norms

#### L₁ Norm (Manhattan Norm)

$$
\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|
$$

**Characteristics**:
- Not sensitive to outliers
- Produces sparse solutions
- Suitable for Lasso regression

#### L₂ Norm (Euclidean Norm)

$$
\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2} = \sqrt{\mathbf{x}^\top\mathbf{x}}
$$

**Characteristics**:
- Most commonly used norm
- Geometric meaning: Vector length
- Suitable for Ridge regression

#### $L_\infty$ Norm (Maximum Norm)

$$
\|\mathbf{x}\|_\infty = \max_{i} |x_i|
$$

**Characteristics**:
- Only concerned with the maximum component
- Used for constraining maximum values

#### $L_0$ "Norm" (Pseudo-norm)

$$
\|\mathbf{x}\|_0 = \#\{i : x_i \neq 0\}
$$

**Note**: Strictly speaking, $L_0$ is not a norm (doesn't satisfy homogeneity), but is commonly used to measure sparsity.

### Norm Comparison

For any $\mathbf{x} \in \mathbb{R}^n$:

$$
\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1 \leq n\|\mathbf{x}\|_\infty
$$

**Derivation of norm inequalities**:

Let $\mathbf{x} = (x_1, \ldots, x_n)$, $M = \max_i |x_i| = \|\mathbf{x}\|_\infty$.

**First inequality: $\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2$**

$$\|\mathbf{x}\|_2^2 = \sum_{i=1}^n x_i^2 \geq M^2$$

Therefore:

$$\|\mathbf{x}\|_2 \geq M = \|\mathbf{x}\|_\infty$$

$$\boxed{\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2}$$

**Second inequality: $\|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1$**

Using the Cauchy-Schwarz inequality, let $\mathbf{y} = (1, 1, \ldots, 1)$:

$$|\mathbf{x} \cdot \mathbf{y}| \leq \|\mathbf{x}\|_2 \|\mathbf{y}\|_2$$

$$\left|\sum_{i=1}^n x_i\right| \leq \|\mathbf{x}\|_2 \sqrt{n}$$

But this is not what we want. Let's use another method:

$$\|\mathbf{x}\|_1^2 = \left(\sum_{i=1}^n |x_i|\right)^2 = \sum_{i=1}^n x_i^2 + 2\sum_{i<j} |x_i||x_j| \geq \sum_{i=1}^n x_i^2 = \|\mathbf{x}\|_2^2$$

Therefore:

$$\boxed{\|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1}$$

**Third inequality: $\|\mathbf{x}\|_1 \leq n\|\mathbf{x}\|_\infty$**

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i| \leq \sum_{i=1}^n M = nM = n\|\mathbf{x}\|_\infty$$

$$\boxed{\|\mathbf{x}\|_1 \leq n\|\mathbf{x}\|_\infty}$$

**Combined conclusion**:

$$\boxed{\|\mathbf{x}\|_\infty \leq \|\mathbf{x}\|_2 \leq \|\mathbf{x}\|_1 \leq n\|\mathbf{x}\|_\infty}$$

**Equivalence**: In finite-dimensional spaces, all norms are equivalent (differ only by constant multiples), which means:
- Convergence holds simultaneously under all norms
- Continuity is the same under all norms

### Geometric Meaning of Norms

Different norms have different "unit ball" shapes ($\|\mathbf{x}\|_p = 1$):

| Norm | Unit ball shape (2D) |
|------|-------------------|
| $L_1$ | Diamond (rhombus) |
| $L_2$ | Circle |
| $L_\infty$ | Square |

### Python Implementation

```python
import numpy as np

x = np.array([3, -4])

# L1 norm
l1_norm = np.linalg.norm(x, ord=1)
print(f"L1 norm: {l1_norm}")  # |3| + |-4| = 7

# L2 norm (default)
l2_norm = np.linalg.norm(x)
print(f"L2 norm: {l2_norm}")  # sqrt(9 + 16) = 5

# L∞ norm
linf_norm = np.linalg.norm(x, ord=np.inf)
print(f"L∞ norm: {linf_norm}")  # max(|3|, |-4|) = 4

# Lp norm (general)
def lp_norm(x, p):
    """Compute Lp norm"""
    return np.sum(np.abs(x) ** p) ** (1/p)

print(f"L3 norm: {lp_norm(x, 3):.4f}")  # (27 + 64)^(1/3) ≈ 4.498

# Verify norm inequalities
print(f"\nNorm comparison:")
print(f"||x||_∞ = {np.linalg.norm(x, ord=np.inf):.2f}")
print(f"||x||_2 = {np.linalg.norm(x, ord=2):.2f}")
print(f"||x||_1 = {np.linalg.norm(x, ord=1):.2f}")
print(f"||x||_∞ ≤ ||x||_2 ≤ ||x||_1: {linf_norm <= l2_norm <= l1_norm}")

# Normalization
x_normalized = x / np.linalg.norm(x)
print(f"\nNormalized vector: {x_normalized}")
print(f"Norm after normalization: {np.linalg.norm(x_normalized)}")  # 1.0
```

### Relationship Between Norm and Dot Product

$$
\|\mathbf{x}\|_2^2 = \mathbf{x} \cdot \mathbf{x}
$$

**Cauchy-Schwarz Inequality**:

$$
|\mathbf{x} \cdot \mathbf{y}| \leq \|\mathbf{x}\|_2 \|\mathbf{y}\|_2
$$

**Proof of Cauchy-Schwarz Inequality**:

**Method 1: Using quadratic form**

**Step 1**: Construct a quadratic function. For any real number $t$, define:

$$f(t) = \|\mathbf{x} + t\mathbf{y}\|_2^2 = (\mathbf{x} + t\mathbf{y}) \cdot (\mathbf{x} + t\mathbf{y})$$

Expanding:

$$f(t) = \|\mathbf{x}\|_2^2 + 2t(\mathbf{x} \cdot \mathbf{y}) + t^2\|\mathbf{y}\|_2^2 \geq 0$$

**Step 2**: Since $f(t) \geq 0$ for all $t$, the discriminant must be non-positive:

$$\Delta = 4(\mathbf{x} \cdot \mathbf{y})^2 - 4\|\mathbf{x}\|_2^2 \|\mathbf{y}\|_2^2 \leq 0$$

**Step 3**: Rearrange the inequality:

$$(\mathbf{x} \cdot \mathbf{y})^2 \leq \|\mathbf{x}\|_2^2 \|\mathbf{y}\|_2^2$$

Taking the square root of both sides:

$$|\mathbf{x} \cdot \mathbf{y}| \leq \|\mathbf{x}\|_2 \|\mathbf{y}\|_2$$

$$\boxed{|\mathbf{x} \cdot \mathbf{y}| \leq \|\mathbf{x}\|_2 \|\mathbf{y}\|_2}$$

**Equality condition**: $\mathbf{x}$ and $\mathbf{y}$ are linearly dependent ($\mathbf{x} = c\mathbf{y}$ or $\mathbf{y} = c\mathbf{x}$).

**Method 2: Using projection**

The projection of $\mathbf{x}$ onto $\mathbf{y}$ is:

$$\text{proj}_{\mathbf{y}}(\mathbf{x}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{y}\|_2^2} \mathbf{y}$$

Projection length:

$$\|\text{proj}_{\mathbf{y}}(\mathbf{x})\|_2 = \frac{|\mathbf{x} \cdot \mathbf{y}|}{\|\mathbf{y}\|_2}$$

Since the projection length does not exceed the original vector length:

$$\frac{|\mathbf{x} \cdot \mathbf{y}|}{\|\mathbf{y}\|_2} \leq \|\mathbf{x}\|_2$$

```python
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Verify Cauchy-Schwarz inequality
dot_product = np.dot(x, y)
norm_product = np.linalg.norm(x) * np.linalg.norm(y)

print(f"|x·y| = {abs(dot_product):.2f}")
print(f"||x||_2 ||y||_2 = {norm_product:.2f}")
print(f"|x·y| ≤ ||x||_2 ||y||_2: {abs(dot_product) <= norm_product}")
```

---

## Matrix Norms

### Definition

Matrix norms satisfy the same three properties as vector norms.

### Frobenius Norm

$$
\|\mathbf{A}\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{ij}^2} = \sqrt{\text{tr}(\mathbf{A}^\top\mathbf{A})} = \sqrt{\sum_{i=1}^{\min(m,n)} \sigma_i^2}
$$

**Characteristics**:
- Square root of the sum of squares of all matrix elements
- Similar to vector L₂ norm
- Commonly used in regularization

### Spectral Norm (Operator Norm / L₂ Norm)

$$
\|\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A}) = \sqrt{\lambda_{\max}(\mathbf{A}^\top\mathbf{A})}
$$

**Characteristics**:
- Largest singular value
- Represents the maximum "stretching" factor of the matrix
- Used in spectral normalization

### Nuclear Norm

$$
\|\mathbf{A}\|_* = \sum_{i=1}^r \sigma_i
$$

**Characteristics**:
- Sum of all singular values
- Convex relaxation of rank
- Used in low-rank matrix recovery

### 1-Norm and ∞-Norm

**Matrix 1-norm** (column sum norm):

$$
\|\mathbf{A}\|_1 = \max_{j} \sum_{i=1}^m |A_{ij}|
$$

**Matrix ∞-norm** (row sum norm):

$$
\|\mathbf{A}\|_\infty = \max_{i} \sum_{j=1}^n |A_{ij}|
$$

### Induced Norms

For a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, the induced norm is defined as:

$$
\|\mathbf{A}\|_p = \sup_{\mathbf{x} \neq \mathbf{0}} \frac{\|\mathbf{Ax}\|_p}{\|\mathbf{x}\|_p} = \max_{\|\mathbf{x}\|_p = 1} \|\mathbf{Ax}\|_p
$$

### Norm Inequalities

For any matrix $\mathbf{A}$:

$$
\|\mathbf{A}\|_2 \leq \|\mathbf{A}\|_F \leq \sqrt{r}\|\mathbf{A}\|_2
$$

Where $r = \text{rank}(\mathbf{A})$.

**Derivation of relationship between Frobenius norm and spectral norm**:

Let $\mathbf{A}$ have singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$, $r = \text{rank}(\mathbf{A})$.

**Step 1**: Express both norms using singular values.

$$\|\mathbf{A}\|_2 = \sigma_1 \quad \text{(largest singular value)}$$

$$\|\mathbf{A}\|_F = \sqrt{\sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2}$$

**Step 2**: Prove $\|\mathbf{A}\|_2 \leq \|\mathbf{A}\|_F$.

$$\|\mathbf{A}\|_F^2 = \sigma_1^2 + \sigma_2^2 + \cdots + \sigma_r^2 \geq \sigma_1^2 = \|\mathbf{A}\|_2^2$$

Therefore:

$$\boxed{\|\mathbf{A}\|_2 \leq \|\mathbf{A}\|_F}$$

**Step 3**: Prove $\|\mathbf{A}\|_F \leq \sqrt{r}\|\mathbf{A}\|_2$.

Since $\sigma_1 \geq \sigma_i$ for all $i$:

$$\|\mathbf{A}\|_F^2 = \sum_{i=1}^r \sigma_i^2 \leq \sum_{i=1}^r \sigma_1^2 = r\sigma_1^2 = r\|\mathbf{A}\|_2^2$$

Therefore:

$$\boxed{\|\mathbf{A}\|_F \leq \sqrt{r}\|\mathbf{A}\|_2}$$

**Step 4**: Combined conclusion.

$$\boxed{\|\mathbf{A}\|_2 \leq \|\mathbf{A}\|_F \leq \sqrt{r}\|\mathbf{A}\|_2}$$

**Special cases**:
- When $r = 1$ (rank-1 matrix), $\|\mathbf{A}\|_F = \|\mathbf{A}\|_2$
- When $\mathbf{A}$ is full rank, $\|\mathbf{A}\|_F \leq \sqrt{\min(m,n)}\|\mathbf{A}\|_2$

**Practical significance**: This inequality is often used to analyze regularization terms and gradient norms.

### Python Implementation

```python
import numpy as np

A = np.array([[1, 2, 3],
              [4, 5, 6]], dtype=float)

# Frobenius norm
fro_norm = np.linalg.norm(A, 'fro')
print(f"Frobenius norm: {fro_norm:.4f}")  # sqrt(1+4+9+16+25+36) = sqrt(91)

# Spectral norm (2-norm)
spectral_norm = np.linalg.norm(A, 2)
print(f"Spectral norm: {spectral_norm:.4f}")

# Verify via SVD
U, S, Vt = np.linalg.svd(A)
print(f"Largest singular value: {S[0]:.4f}")  # Should equal spectral norm

# 1-norm (column sum)
norm_1 = np.linalg.norm(A, 1)
print(f"1-norm (max column sum): {norm_1:.4f}")  # max(5, 7, 9) = 9

# ∞-norm (row sum)
norm_inf = np.linalg.norm(A, np.inf)
print(f"∞-norm (max row sum): {norm_inf:.4f}")  # max(6, 15) = 15

# Nuclear norm
nuclear_norm = np.sum(S)
print(f"Nuclear norm: {nuclear_norm:.4f}")

# Manual computation of Frobenius norm
fro_manual = np.sqrt(np.trace(A.T @ A))
print(f"Manual Frobenius: {fro_manual:.4f}")
```

### Condition Number

**Definition**:

$$
\kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\|
$$

For spectral norm:

$$
\kappa_2(\mathbf{A}) = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{\sigma_1}{\sigma_r}
$$

**Meaning**:
- $\kappa \approx 1$: Well-conditioned matrix
- $\kappa \gg 1$: Ill-conditioned matrix, numerically unstable

```python
import numpy as np

# Condition number
A = np.array([[1, 2], [3, 4]], dtype=float)
cond_A = np.linalg.cond(A)
print(f"Condition number: {cond_A:.4f}")

# Verify via SVD
U, S, Vt = np.linalg.svd(A)
cond_from_svd = S[0] / S[-1]
print(f"Condition number from SVD: {cond_from_svd:.4f}")

# Ill-conditioned matrix example
ill_conditioned = np.array([[1, 1], [1, 1.00001]], dtype=float)
cond_ill = np.linalg.cond(ill_conditioned)
print(f"\nIll-conditioned matrix condition number: {cond_ill:.2e}")  # Very large

# Well-conditioned matrix
well_conditioned = np.eye(3)
cond_well = np.linalg.cond(well_conditioned)
print(f"Identity matrix condition number: {cond_well:.2f}")  # 1
```

---

## Distance Metrics

### Definition

**Distance metrics** are functions that map two objects to non-negative real numbers $d: X \times X \to \mathbb{R}_{\geq 0}$, satisfying:

1. **Non-negativity**: $d(x, y) \geq 0$, equality holds if and only if $x = y$
2. **Symmetry**: $d(x, y) = d(y, x)$
3. **Triangle inequality**: $d(x, z) \leq d(x, y) + d(y, z)$

### Euclidean Distance (L₂ Distance)

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^n (x_i - y_i)^2}
$$

**Characteristics**:
- Most intuitive distance
- Corresponds to L₂ norm
- Suitable for continuous variables

### Manhattan Distance (L₁ Distance)

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_1 = \sum_{i=1}^n |x_i - y_i|
$$

**Characteristics**:
- City block distance
- Not sensitive to outliers
- Suitable for high-dimensional data

### Chebyshev Distance (L∞ Distance)

$$
d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_\infty = \max_{i} |x_i - y_i|
$$

**Characteristics**:
- Only concerned with maximum difference
- Suitable for chessboard distance

### Minkowski Distance (General Form)

$$
d_p(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^n |x_i - y_i|^p \right)^{1/p}
$$

- $p = 1$: Manhattan distance
- $p = 2$: Euclidean distance
- $p \to \infty$: Chebyshev distance

### Hamming Distance

$$
d_H(\mathbf{x}, \mathbf{y}) = \#\{i : x_i \neq y_i\}
$$

**Characteristics**:
- Suitable for discrete/binary data
- Counts the number of differing positions

### Python Implementation

```python
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

# Euclidean distance
euclidean = np.linalg.norm(x - y)
print(f"Euclidean distance: {euclidean:.4f}")  # sqrt(9+9+9) = sqrt(27)

# Manhattan distance
manhattan = np.linalg.norm(x - y, ord=1)
print(f"Manhattan distance: {manhattan:.4f}")  # 3+3+3 = 9

# Chebyshev distance
chebyshev = np.linalg.norm(x - y, ord=np.inf)
print(f"Chebyshev distance: {chebyshev:.4f}")  # max(3,3,3) = 3

# Using scipy
points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Compute distance matrix between all point pairs
dist_matrix = cdist(points, points, metric='euclidean')
print(f"\nDistance matrix (Euclidean):\n{dist_matrix}")

# Different distance metrics
print(f"\nDifferent distance metrics:")
for metric in ['euclidean', 'cityblock', 'chebyshev']:
    d = cdist([x], [y], metric=metric)[0, 0]
    print(f"  {metric}: {d:.4f}")

# Hamming distance (binary vectors)
a = np.array([1, 0, 1, 1, 0])
b = np.array([1, 1, 1, 0, 0])
hamming = np.sum(a != b)
print(f"\nHamming distance: {hamming}")  # 2
```

### Mahalanobis Distance

Distance metric considering feature correlation:

$$
d_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x} - \mathbf{y})^\top \mathbf{\Sigma}^{-1} (\mathbf{x} - \mathbf{y})}
$$

Where $\mathbf{\Sigma}$ is the covariance matrix.

```python
def mahalanobis_distance(x, y, cov):
    """Mahalanobis distance"""
    diff = x - y
    cov_inv = np.linalg.inv(cov)
    return np.sqrt(diff @ cov_inv @ diff)

# Example
np.random.seed(42)
X = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0.8], [0.8, 1]], size=100)
cov = np.cov(X.T)

x = X[0]
y = X[1]

mahal_dist = mahalanobis_distance(x, y, cov)
euclid_dist = np.linalg.norm(x - y)

print(f"Mahalanobis distance: {mahal_dist:.4f}")
print(f"Euclidean distance: {euclid_dist:.4f}")
```

---

## Similarity Metrics

### Cosine Similarity

$$
\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}
$$

**Range**: $[-1, 1]$

| Value | Meaning |
|-------|---------|
| 1 | Completely aligned |
| 0 | Orthogonal |
| -1 | Completely opposite |

**Cosine distance**:

$$
d_{\cos}(\mathbf{x}, \mathbf{y}) = 1 - \cos(\mathbf{x}, \mathbf{y})
$$

### Pearson Correlation Coefficient

$$
\rho(\mathbf{x}, \mathbf{y}) = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2} \sqrt{\sum_i (y_i - \bar{y})^2}}
$$

**Characteristics**:
- Centered cosine similarity
- Range $[-1, 1]$

### Jaccard Similarity

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

**Characteristics**:
- Suitable for sets
- Commonly used for text similarity

### Python Implementation

```python
import numpy as np
from scipy.spatial.distance import cosine, correlation
from scipy.stats import pearsonr

x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 6, 8, 10])  # y = 2x, completely positively correlated

# Cosine similarity
cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
print(f"Cosine similarity: {cos_sim:.4f}")  # 1.0 (completely aligned)

# Cosine distance
cos_dist = cosine(x, y)
print(f"Cosine distance: {cos_dist:.4f}")  # 0

# Pearson correlation coefficient
pearson_corr, _ = pearsonr(x, y)
print(f"Pearson correlation coefficient: {pearson_corr:.4f}")  # 1.0

# Jaccard similarity (sets)
set_a = {1, 2, 3, 4}
set_b = {3, 4, 5, 6}
jaccard = len(set_a & set_b) / len(set_a | set_b)
print(f"Jaccard similarity: {jaccard:.4f}")  # 2/6 ≈ 0.333

# Jaccard similarity (binary vectors)
def jaccard_binary(a, b):
    """Jaccard similarity for binary vectors"""
    intersection = np.sum(np.logical_and(a, b))
    union = np.sum(np.logical_or(a, b))
    return intersection / union if union > 0 else 0

a = np.array([1, 1, 0, 1, 0])
b = np.array([1, 0, 1, 1, 1])
print(f"Binary Jaccard: {jaccard_binary(a, b):.4f}")
```

### Similarity in Attention

```python
def attention_scores(Q, K, scale=True):
    """
    Compute attention scores

    Q: (seq_len, d_k)
    K: (seq_len, d_k)
    """
    d_k = Q.shape[-1]

    # Dot product attention
    scores = Q @ K.T

    if scale:
        scores = scores / np.sqrt(d_k)

    return scores

# Example: Compute similarity matrix
embeddings = np.random.randn(10, 64)  # 10 word embeddings

# Dot product similarity
dot_sim = embeddings @ embeddings.T

# Cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized = embeddings / norms
cos_sim_matrix = normalized @ normalized.T

print(f"Dot product similarity matrix shape: {dot_sim.shape}")
print(f"Cosine similarity matrix shape: {cos_sim_matrix.shape}")

# Verify diagonal is 1 (self-similarity)
print(f"Cosine similarity diagonal: {np.diag(cos_sim_matrix)}")  # All 1
```

---

## Applications in Deep Learning

### Regularization

#### L2 Regularization (Weight Decay)

$$
L_{\text{reg}} = L + \lambda \|\mathbf{W}\|_F^2
$$

```python
def l2_regularization(W, lambda_reg):
    """L2 regularization"""
    return lambda_reg * np.sum(W ** 2)

def l2_regularization_grad(W, lambda_reg):
    """Gradient of L2 regularization"""
    return 2 * lambda_reg * W

# Example
W = np.random.randn(256, 128)
lambda_reg = 0.01

reg_loss = l2_regularization(W, lambda_reg)
reg_grad = l2_regularization_grad(W, lambda_reg)

print(f"L2 regularization loss: {reg_loss:.4f}")
print(f"L2 regularization gradient shape: {reg_grad.shape}")
```

#### L1 Regularization

$$
L_{\text{reg}} = L + \lambda \|\mathbf{W}\|_1
$$

```python
def l1_regularization(W, lambda_reg):
    """L1 regularization"""
    return lambda_reg * np.sum(np.abs(W))

def l1_regularization_grad(W, lambda_reg, eps=1e-8):
    """Subgradient of L1 regularization"""
    return lambda_reg * np.sign(W)

# Compare L1 and L2
W = np.random.randn(100)

l1_loss = l1_regularization(W, 0.01)
l2_loss = l2_regularization(W, 0.01)

print(f"L1 regularization loss: {l1_loss:.4f}")
print(f"L2 regularization loss: {l2_loss:.4f}")
```

#### Elastic Net

$$
L_{\text{reg}} = L + \lambda_1 \|\mathbf{W}\|_1 + \lambda_2 \|\mathbf{W}\|_F^2
$$

### Loss Functions

#### Mean Squared Error (MSE)

$$
L_{\text{MSE}} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{n}\|\mathbf{y} - \hat{\mathbf{y}}\|_2^2
$$

#### Mean Absolute Error (MAE)

$$
L_{\text{MAE}} = \frac{1}{n}\sum_{i=1}^n |y_i - \hat{y}_i| = \frac{1}{n}\|\mathbf{y} - \hat{\mathbf{y}}\|_1
$$

#### Huber Loss

Combines MSE and MAE:

$$
L_\delta(y, \hat{y}) = \begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta|y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise}
\end{cases}
$$

```python
def mse_loss(y_pred, y_true):
    """Mean squared error"""
    return np.mean((y_pred - y_true) ** 2)

def mae_loss(y_pred, y_true):
    """Mean absolute error"""
    return np.mean(np.abs(y_pred - y_true))

def huber_loss(y_pred, y_true, delta=1.0):
    """Huber loss"""
    diff = np.abs(y_pred - y_true)
    quadratic = np.minimum(diff, delta)
    linear = diff - quadratic
    return np.mean(0.5 * quadratic ** 2 + delta * linear)

# Example
y_true = np.array([1, 2, 3, 4, 5])
y_pred = np.array([1.1, 2.2, 2.8, 4.5, 5.5])

print(f"MSE: {mse_loss(y_pred, y_true):.4f}")
print(f"MAE: {mae_loss(y_pred, y_true):.4f}")
print(f"Huber: {huber_loss(y_pred, y_true):.4f}")

# With outliers
y_pred_outlier = np.array([1.1, 2.2, 10.0, 4.5, 5.5])  # Third one has outlier
print(f"\nWith outliers:")
print(f"MSE: {mse_loss(y_pred_outlier, y_true):.4f}")  # Greatly affected
print(f"MAE: {mae_loss(y_pred_outlier, y_true):.4f}")  # Less affected
print(f"Huber: {huber_loss(y_pred_outlier, y_true):.4f}")  # Between the two
```

### Batch Normalization

```python
def batch_norm(X, gamma, beta, eps=1e-5, momentum=0.1, running_mean=None, running_var=None, training=True):
    """
    Batch normalization

    X: (batch, features)
    """
    if training:
        # Compute batch statistics
        mean = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        # Update running statistics
        if running_mean is not None:
            running_mean = momentum * mean + (1 - momentum) * running_mean
        if running_var is not None:
            running_var = momentum * var + (1 - momentum) * running_var
    else:
        mean = running_mean
        var = running_var

    # Normalize
    X_norm = (X - mean) / np.sqrt(var + eps)

    # Scale and shift
    return gamma * X_norm + beta

# Example
X = np.random.randn(32, 128)
gamma = np.ones(128)
beta = np.zeros(128)

X_bn = batch_norm(X, gamma, beta, training=True)
print(f"Mean before batch norm: {np.mean(X, axis=0)[:5]}")
print(f"Mean after batch norm: {np.mean(X_bn, axis=0)[:5]}")  # Close to 0
print(f"Variance after batch norm: {np.var(X_bn, axis=0)[:5]}")   # Close to 1
```

### Layer Normalization

```python
def layer_norm(X, gamma, beta, eps=1e-5):
    """
    Layer normalization

    X: (batch, seq_len, features) or (batch, features)
    """
    mean = np.mean(X, axis=-1, keepdims=True)
    var = np.var(X, axis=-1, keepdims=True)

    X_norm = (X - mean) / np.sqrt(var + eps)

    return gamma * X_norm + beta

# Example
X = np.random.randn(32, 10, 64)  # (batch, seq_len, features)
gamma = np.ones(64)
beta = np.zeros(64)

X_ln = layer_norm(X, gamma, beta)
print(f"Mean after layer norm: {np.mean(X_ln, axis=-1)[0, :5]}")  # Close to 0
print(f"Variance after layer norm: {np.var(X_ln, axis=-1)[0, :5]}")   # Close to 1
```

### Gradient Clipping

#### Clip by Norm

$$
\mathbf{g} \leftarrow \mathbf{g} \cdot \min\left(1, \frac{c}{\|\mathbf{g}\|}\right)
$$

```python
def clip_grad_norm_(gradients, max_norm):
    """Clip gradients by norm"""
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients))

    if total_norm > max_norm:
        scale = max_norm / total_norm
        for i in range(len(gradients)):
            gradients[i] = gradients[i] * scale

    return total_norm

# Example
grads = [np.random.randn(100) * 10 for _ in range(5)]
original_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
print(f"Gradient norm before clipping: {original_norm:.4f}")

clipped_norm = clip_grad_norm_(grads, max_norm=1.0)
new_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
print(f"Gradient norm after clipping: {new_norm:.4f}")
```

#### Clip by Value

```python
def clip_grad_value_(gradients, clip_value):
    """Clip gradients by value"""
    for i in range(len(gradients)):
        gradients[i] = np.clip(gradients[i], -clip_value, clip_value)

# Example
grads = [np.random.randn(100) * 10 for _ in range(5)]
clip_grad_value_(grads, clip_value=1.0)
print(f"Maximum value after clipping: {max(np.max(np.abs(g)) for g in grads):.4f}")  # ≤ 1
```

### Weight Initialization

#### Xavier/Glorot Initialization

$$
\text{Var}(W) = \frac{2}{n_{in} + n_{out}}
$$

```python
def xavier_uniform(shape, gain=1.0):
    """Xavier uniform initialization"""
    fan_in, fan_out = shape
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    a = np.sqrt(3.0) * std
    return np.random.uniform(-a, a, shape)

def xavier_normal(shape, gain=1.0):
    """Xavier normal initialization"""
    fan_in, fan_out = shape
    std = gain * np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.randn(*shape) * std
```

#### Kaiming/He Initialization

$$
\text{Var}(W) = \frac{2}{n_{in}}
$$

```python
def kaiming_uniform(shape, mode='fan_in', nonlinearity='relu'):
    """Kaiming uniform initialization"""
    fan_in, fan_out = shape
    if mode == 'fan_in':
        fan = fan_in
    else:
        fan = fan_out

    gain = np.sqrt(2.0) if nonlinearity == 'relu' else 1.0
    std = gain / np.sqrt(fan)
    a = np.sqrt(3.0) * std
    return np.random.uniform(-a, a, shape)

def kaiming_normal(shape, mode='fan_in', nonlinearity='relu'):
    """Kaiming normal initialization"""
    fan_in, fan_out = shape
    if mode == 'fan_in':
        fan = fan_in
    else:
        fan = fan_out

    gain = np.sqrt(2.0) if nonlinearity == 'relu' else 1.0
    std = gain / np.sqrt(fan)
    return np.random.randn(*shape) * std

# Example: Compare initialization methods
shape = (784, 256)

W_xavier = xavier_normal(shape)
W_kaiming = kaiming_normal(shape)

print(f"Xavier initialization variance: {np.var(W_xavier):.6f}")  # ≈ 2/(784+256)
print(f"Kaiming initialization variance: {np.var(W_kaiming):.6f}")  # ≈ 2/784
```

### Scaling in Attention Mechanism

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Scaled dot product attention

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """
    d_k = Q.shape[-1]

    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)

    # Optional: Apply mask
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)

    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Weighted sum
    output = attention_weights @ V

    return output, attention_weights

# Example
d_k = 64
Q = np.random.randn(10, d_k)
K = np.random.randn(10, d_k)
V = np.random.randn(10, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Attention output shape: {output.shape}")       # (10, 64)
print(f"Attention weights shape: {weights.shape}")     # (10, 10)
print(f"Row-wise weight sums: {weights.sum(axis=1)}")  # All 1
```

### Distances in Contrastive Learning

```python
def contrastive_loss(z_i, z_j, temperature=0.5):
    """
    Contrastive loss (simplified version)

    z_i, z_j: Normalized embedding vectors
    """
    batch_size = z_i.shape[0]

    # Compute similarity matrix
    sim_matrix = z_i @ z_j.T / temperature

    # Positive samples on the diagonal
    labels = np.arange(batch_size)

    # Cross-entropy loss
    exp_sim = np.exp(sim_matrix - np.max(sim_matrix, axis=1, keepdims=True))
    probs = exp_sim / np.sum(exp_sim, axis=1, keepdims=True)

    loss = -np.mean(np.log(probs[np.arange(batch_size), labels] + 1e-8))

    return loss

# Example
batch_size = 32
embed_dim = 128

z_i = np.random.randn(batch_size, embed_dim)
z_j = np.random.randn(batch_size, embed_dim)

# Normalize
z_i = z_i / np.linalg.norm(z_i, axis=1, keepdims=True)
z_j = z_j / np.linalg.norm(z_j, axis=1, keepdims=True)

loss = contrastive_loss(z_i, z_j)
print(f"Contrastive loss: {loss:.4f}")
```

### Search and Ranking

```python
def knn_search(query, database, k=5, metric='euclidean'):
    """
    K-nearest neighbors search

    query: (d,) query vector
    database: (n, d) database
    """
    if metric == 'euclidean':
        distances = np.linalg.norm(database - query, axis=1)
    elif metric == 'cosine':
        # Cosine distance
        query_norm = query / np.linalg.norm(query)
        db_norm = database / np.linalg.norm(database, axis=1, keepdims=True)
        similarities = db_norm @ query_norm
        distances = 1 - similarities

    # Get k nearest
    indices = np.argsort(distances)[:k]

    return indices, distances[indices]

# Example
database = np.random.randn(1000, 128)
query = np.random.randn(128)

indices_euclid, dists_euclid = knn_search(query, database, k=5, metric='euclidean')
indices_cosine, dists_cosine = knn_search(query, database, k=5, metric='cosine')

print(f"Euclidean nearest neighbors: {indices_euclid}, distances: {dists_euclid}")
print(f"Cosine nearest neighbors: {indices_cosine}, distances: {dists_cosine}")
```

---

## Summary

This chapter introduced norms, distance metrics, and their applications in deep learning:

### Vector Norms

| Norm | Formula | Characteristics |
|------|---------|----------------|
| $L_1$ | $\sum_i |x_i|$ | Sparsity, Lasso |
| $L_2$ | $\sqrt{\sum_i x_i^2}$ | Most common, Ridge |
| $L_\infty$ | $\max_i |x_i|$ | Maximum value constraint |

### Matrix Norms

| Norm | Formula | Characteristics |
|------|---------|----------------|
| Frobenius | $\sqrt{\sum_{ij} A_{ij}^2}$ | Similar to L₂ |
| Spectral | $\sigma_{\max}$ | Largest singular value |
| Nuclear | $\sum_i \sigma_i$ | Low-rank regularization |

### Distance Metrics

| Distance | Formula | Characteristics |
|-----------|---------|----------------|
| Euclidean | $\sqrt{\sum_i (x_i-y_i)^2}$ | Most intuitive |
| Manhattan | $\sum_i |x_i-y_i|$ | Not sensitive to outliers |
| Cosine | $1 - \frac{x \cdot y}{\|x\|\|y\|}$ | Directional similarity |
| Mahalanobis | $\sqrt{(x-y)^\top\Sigma^{-1}(x-y)}$ | Considers correlation |

### Deep Learning Applications

| Application | Norm/Distance | Purpose |
|-------------|---------------|---------|
| L2 regularization | Frobenius | Prevent overfitting |
| L1 regularization | L1 | Sparsity |
| MSE loss | L₂² | Regression |
| Gradient clipping | L₂ | Prevent gradient explosion |
| Attention scaling | L₂ | Numerical stability |
| Contrastive learning | Cosine | Similarity learning |

---

**Previous section**: [Chapter 1(c): Eigenvalues and Matrix Decomposition](01c-eigenvalues-matrix-decomposition_EN.md)

**Next chapter**: [Chapter 2: Calculus](02-calculus.md) - Learn about derivatives, gradients, chain rule, and understand the mathematical principles of backpropagation.

**Back**: [Math Fundamentals Tutorial Table of Contents](../math-fundamentals.md)
