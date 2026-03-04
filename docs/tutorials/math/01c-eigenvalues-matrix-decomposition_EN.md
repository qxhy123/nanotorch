# Chapter 1(c): Eigenvalues and Matrix Decomposition

Eigenvalues and eigenvectors are key concepts for understanding the essence of matrices, while matrix decompositions are powerful tools for simplifying complex matrices into simpler forms. This chapter will introduce these concepts in depth and their applications in deep learning.

---

## 🎯 Life Analogy: Stretching Playdough

Imagine you're playing with a piece of playdough:
- Most directions: When you stretch it, it **deforms AND rotates**
- Special directions (eigenvectors): When you stretch it, it **only gets longer/shorter, direction stays the same**

**Eigenvectors** = Those directions that only stretch, don't rotate
**Eigenvalues** = The stretching factor (>1 grows, <1 shrinks, <0 reverses)

```
Original Shape          After Transformation
   ↗                         ↗↗↗  (stretched 3x in this direction, eigenvalue=3)
  /                         /
 ○-------→                ○=======
   \                         \\
    ↘                         ↘  (compressed to half, eigenvalue=0.5)
```

### 📝 Step-by-Step Calculation

Find the eigenvalues and eigenvectors of matrix $\mathbf{A} = \begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix}$

**Step 1: Write the characteristic equation**
$$\det(\mathbf{A} - \lambda\mathbf{I}) = \det\begin{bmatrix} 4-\lambda & 1 \\ 2 & 3-\lambda \end{bmatrix} = 0$$

**Step 2: Expand the determinant**
$$(4-\lambda)(3-\lambda) - 2 \times 1 = 0$$
$$12 - 4\lambda - 3\lambda + \lambda^2 - 2 = 0$$
$$\lambda^2 - 7\lambda + 10 = 0$$

**Step 3: Solve the quadratic equation**
$$(\lambda - 5)(\lambda - 2) = 0$$

**Eigenvalues: $\lambda_1 = 5$, $\lambda_2 = 2$**

**Step 4: Find eigenvectors**

For $\lambda_1 = 5$:
$$\begin{bmatrix} 4-5 & 1 \\ 2 & 3-5 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} -1 & 1 \\ 2 & -2 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \mathbf{0}$$

This gives $-v_1 + v_2 = 0$, so $v_1 = v_2$.

**Eigenvector**: $\mathbf{v}_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ (any non-zero multiple works)

**Verify**: $\begin{bmatrix} 4 & 1 \\ 2 & 3 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 5 \\ 5 \end{bmatrix} = 5 \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ ✓

### 📖 Plain English Translation

| Math Term | Plain English |
|-----------|---------------|
| Eigenvalue | How much something stretches or shrinks |
| Eigenvector | The direction that stays the same after transformation |
| SVD | Breaking a matrix into rotation → stretch → rotation |
| Decomposition | Taking apart a complex thing into simpler pieces |

---

## Table of Contents

1. [Eigenvalues and Eigenvectors](#eigenvalues-and-eigenvectors)
2. [Eigenvalue Decomposition (EVD)](#eigenvalue-decomposition-evd)
3. [Singular Value Decomposition (SVD)](#singular-value-decomposition-svd)
4. [QR Decomposition](#qr-decomposition)
5. [Cholesky Decomposition](#cholesky-decomposition)
6. [LU Decomposition](#lu-decomposition)
7. [Other Decomposition Methods](#other-decomposition-methods)
8. [Applications in Deep Learning](#applications-in-deep-learning)
9. [Summary](#summary)

---

## Eigenvalues and Eigenvectors

### Definition

For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, if there exists a non-zero vector $\mathbf{v}$ and a scalar $\lambda$ such that:

$$
\mathbf{A}\mathbf{v} = \lambda \mathbf{v}
$$

Then:
- $\lambda$ is called an **eigenvalue** of $\mathbf{A}$
- $\mathbf{v}$ is called an **eigenvector** corresponding to $\lambda$

**Intuitive understanding**: Eigenvectors are vectors that only scale (do not change direction) under the matrix transformation, and the scaling factor is the eigenvalue.

### Characteristic Equation

From $\mathbf{Av} = \lambda\mathbf{v}$, we can obtain:

$$
(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}
$$

For $\mathbf{v} \neq \mathbf{0}$ to hold, we must have:

$$
\det(\mathbf{A} - \lambda\mathbf{I}) = 0
$$

This is an $n$-th degree polynomial in $\lambda$, called the **characteristic polynomial**. Its roots are the eigenvalues.

### Eigenvalues of 2×2 Matrices

For $\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$:

$$
\det(\mathbf{A} - \lambda\mathbf{I}) = \lambda^2 - (a+d)\lambda + (ad-bc) = 0
$$

Solving:

$$
\lambda = \frac{(a+d) \pm \sqrt{(a+d)^2 - 4(ad-bc)}}{2}
$$

**Derivation of the 2×2 matrix characteristic equation**:

**Step 1**: Write the characteristic equation $\det(\mathbf{A} - \lambda\mathbf{I}) = 0$:

$$\det\begin{bmatrix} a - \lambda & b \\ c & d - \lambda \end{bmatrix} = 0$$

**Step 2**: Compute the determinant:

$$(a - \lambda)(d - \lambda) - bc = 0$$

$$ad - a\lambda - d\lambda + \lambda^2 - bc = 0$$

**Step 3**: Arrange into the characteristic polynomial:

$$\lambda^2 - (a+d)\lambda + (ad - bc) = 0$$

Let $\tau = a + d$ (trace), $\delta = ad - bc$ (determinant), then:

$$\lambda^2 - \tau\lambda + \delta = 0$$

**Step 4**: Use the quadratic formula:

$$\lambda = \frac{\tau \pm \sqrt{\tau^2 - 4\delta}}{2} = \frac{(a+d) \pm \sqrt{(a+d)^2 - 4(ad-bc)}}{2}$$

**Geometric meaning**:
- Discriminant $\Delta = \tau^2 - 4\delta = (a+d)^2 - 4(ad-bc) = (a-d)^2 + 4bc$
  - $\Delta > 0$: Two distinct real eigenvalues
  - $\Delta = 0$: One real eigenvalue (repeated root)
  - $\Delta < 0$: Two complex conjugate eigenvalues

### Properties of Eigenvalues

Let $\lambda_1, \lambda_2, \ldots, \lambda_n$ be the eigenvalues of $\mathbf{A}$ (possibly with repetitions):

| Property | Formula |
|----------|---------|
| **Trace** | $\text{tr}(\mathbf{A}) = \sum_{i=1}^n \lambda_i$ |
| **Determinant** | $\det(\mathbf{A}) = \prod_{i=1}^n \lambda_i$ |
| **Power** | Eigenvalues of $\mathbf{A}^k$ are $\lambda_1^k, \ldots, \lambda_n^k$ |
| **Inverse** | If $\mathbf{A}$ is invertible, eigenvalues of $\mathbf{A}^{-1}$ are $1/\lambda_1, \ldots, 1/\lambda_n$ |
| **Shift** | Eigenvalues of $\mathbf{A} + c\mathbf{I}$ are $\lambda_1 + c, \ldots, \lambda_n + c$ |
| **Scalar multiplication** | Eigenvalues of $c\mathbf{A}$ are $c\lambda_1, \ldots, c\lambda_n$ |

### Properties of Eigenvectors

- Eigenvectors corresponding to different eigenvalues are **linearly independent**
- All eigenvectors corresponding to the same eigenvalue form a subspace (eigenspace)
- Eigenvectors can be arbitrarily scaled: if $\mathbf{v}$ is an eigenvector, then $c\mathbf{v}$ is also (for $c \neq 0$)

### Eigenvalues of Symmetric Matrices

For a real symmetric matrix $\mathbf{A} = \mathbf{A}^\top$:

| Property | Description |
|----------|-------------|
| Real eigenvalues | All eigenvalues are real numbers |
| Orthogonal eigenvectors | Eigenvectors corresponding to different eigenvalues are orthogonal |
| Diagonalizable | There exists an orthogonal matrix $\mathbf{Q}$ such that $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$ |

### Positive Definite Matrices

A symmetric matrix $\mathbf{A}$ is called:

| Type | Condition | Eigenvalue condition |
|------|-----------|---------------------|
| **Positive definite** | $\mathbf{x}^\top\mathbf{A}\mathbf{x} > 0, \forall \mathbf{x} \neq \mathbf{0}$ | All $\lambda_i > 0$ |
| **Positive semidefinite** | $\mathbf{x}^\top\mathbf{A}\mathbf{x} \geq 0, \forall \mathbf{x}$ | All $\lambda_i \geq 0$ |
| **Negative definite** | $\mathbf{x}^\top\mathbf{A}\mathbf{x} < 0, \forall \mathbf{x} \neq \mathbf{0}$ | All $\lambda_i < 0$ |
| **Indefinite** | Both positive and negative exist | $\lambda_i$ have both positive and negative values |

### Python Implementation

```python
import numpy as np

# Compute eigenvalues and eigenvectors
A = np.array([[4, 2],
              [1, 3]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(A)

print("Eigenvalues:", eigenvalues)  # [5., 2.]
print("Eigenvectors (column vectors):\n", eigenvectors)

# Verify A @ v = λ @ v
print("\nVerification:")
for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    print(f"\nλ_{i} = {val:.4f}")
    print(f"A @ v = {A @ vec}")
    print(f"λ @ v = {val * vec}")
    print(f"Equal? {np.allclose(A @ vec, val * vec)}")

# Verify eigenvalue properties
print("\nTrace and determinant:")
print(f"tr(A) = {np.trace(A)}")
print(f"Sum of eigenvalues = {sum(eigenvalues)}")
print(f"det(A) = {np.linalg.det(A)}")
print(f"Product of eigenvalues = {np.prod(eigenvalues)}")

# Symmetric matrix
S = np.array([[4, 2],
              [2, 3]], dtype=float)
eigvals_S, eigvecs_S = np.linalg.eig(S)

print("\nSymmetric matrix eigenvalues:", eigvals_S)
print("Eigenvectors orthogonal? ", np.allclose(eigvecs_S[:, 0] @ eigvecs_S[:, 1], 0))

# Positive definiteness test
print(f"Matrix positive definite? {all(eigvals_S > 0)}")
```

### Computing Eigenvectors

Given an eigenvalue $\lambda$, compute the eigenvector by solving the linear system:

$$
(\mathbf{A} - \lambda\mathbf{I})\mathbf{v} = \mathbf{0}
$$

```python
def compute_eigenvector(A, eigenvalue, tol=1e-10):
    """Compute eigenvector given eigenvalue"""
    n = A.shape[0]
    # Construct (A - λI)
    M = A - eigenvalue * np.eye(n)

    # Solve M @ v = 0, use SVD to find null space
    U, S, Vt = np.linalg.svd(M)

    # Find vector corresponding to smallest singular value
    idx = np.argmin(S)
    if S[idx] > tol:
        print(f"Warning: Possible error, smallest singular value {S[idx]} not small enough")

    eigenvector = Vt[idx, :]
    return eigenvector / np.linalg.norm(eigenvector)

A = np.array([[4, 2], [1, 3]], dtype=float)
eigenvalues = np.linalg.eigvals(A)

for lam in eigenvalues:
    v = compute_eigenvector(A, lam)
    print(f"λ = {lam:.4f}, v = {v}")
    print(f"Verify A @ v - λ @ v = {A @ v - lam * v}")
```

---

## Eigenvalue Decomposition (EVD)

### Definition

For a diagonalizable square matrix $\mathbf{A}$, it can be decomposed as:

$$
\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{-1}
$$

Where:
- $\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n]$: Matrix composed of eigenvectors
- $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \ldots, \lambda_n)$: Diagonal matrix of eigenvalues

### Diagonalizability Conditions

Conditions for matrix $\mathbf{A}$ to be diagonalizable:
1. $\mathbf{A}$ has $n$ linearly independent eigenvectors
2. Equivalently, the algebraic multiplicity of each eigenvalue equals its geometric multiplicity

**Symmetric matrices are always diagonalizable**, and orthogonal eigenvectors can be chosen.

### Orthogonal Diagonalization of Symmetric Matrices

For a real symmetric matrix $\mathbf{A} = \mathbf{A}^\top$:

$$
\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top
$$

Where $\mathbf{Q}$ is an orthogonal matrix ($\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$).

### Computing Matrix Powers

Using eigenvalue decomposition:

$$
\mathbf{A}^k = \mathbf{V}\mathbf{\Lambda}^k\mathbf{V}^{-1}
$$

Where $\mathbf{\Lambda}^k = \text{diag}(\lambda_1^k, \ldots, \lambda_n^k)$.

### Matrix Functions

For a function $f$, we can define a matrix function:

$$
f(\mathbf{A}) = \mathbf{V}f(\mathbf{\Lambda})\mathbf{V}^{-1}
$$

Where $f(\mathbf{\Lambda}) = \text{diag}(f(\lambda_1), \ldots, f(\lambda_n))$.

Common examples:
- $e^{\mathbf{A}} = \mathbf{V}\text{diag}(e^{\lambda_1}$, $\ldots$, $e^{\lambda_n})\mathbf{V}^{-1}$
- $\ln(\mathbf{A})$ (requires $\lambda_i > 0$)
- $\sqrt{\mathbf{A}}$ (requires $\lambda_i \geq 0$)

```python
import numpy as np

# Eigenvalue decomposition
A = np.array([[4, 1, 1],
              [1, 3, 1],
              [1, 1, 2]], dtype=float)

eigenvalues, eigenvectors = np.linalg.eig(A)

# Construct diagonal matrix
Lambda = np.diag(eigenvalues)
V = eigenvectors
V_inv = np.linalg.inv(V)

# Verify decomposition
A_reconstructed = V @ Lambda @ V_inv
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")

# Orthogonal diagonalization of symmetric matrix
S = np.array([[4, 2],
              [2, 3]], dtype=float)
eigvals, eigvecs = np.linalg.eig(S)

Q = eigvecs  # Orthogonal matrix
Lambda = np.diag(eigvals)

# Verify orthogonality
print(f"\nQ^T @ Q = \n{Q.T @ Q}")  # Should be close to identity matrix
print(f"Q @ Lambda @ Q^T = \n{Q @ Lambda @ Q.T}")
print(f"Original matrix S = \n{S}")

# Matrix power
def matrix_power_eig(A, k):
    """Compute matrix power using eigenvalue decomposition"""
    eigvals, eigvecs = np.linalg.eig(A)
    return eigvecs @ np.diag(eigvals ** k) @ np.linalg.inv(eigvecs)

A_test = np.array([[2, 1], [0, 3]], dtype=float)
print(f"\nA^3 (decomposition method) = \n{matrix_power_eig(A_test, 3)}")
print(f"A^3 (direct method) = \n{np.linalg.matrix_power(A_test, 3)}")

# Matrix exponential
def matrix_exponential(A):
    """Matrix exponential e^A"""
    eigvals, eigvecs = np.linalg.eig(A)
    return eigvecs @ np.diag(np.exp(eigvals)) @ np.linalg.inv(eigvecs)

print(f"\ne^A = \n{matrix_exponential(A_test)}")
print(f"scipy result = \n{np.exp(A_test)}")  # Note: This is incorrect, just element-wise exponential
```

---

## Singular Value Decomposition (SVD)

### Definition

**Any matrix** $\mathbf{A} \in \mathbb{R}^{m \times n}$ can be decomposed via singular value decomposition:

$$
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top
$$

Where:
- $\mathbf{U} \in \mathbb{R}^{m \times m}$: **Left singular vectors** (orthogonal matrix)
- $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$: **Singular value diagonal matrix**
- $\mathbf{V} \in \mathbb{R}^{n \times n}$: **Right singular vectors** (orthogonal matrix)

### Singular Values

The singular values are the diagonal elements of $\mathbf{\Sigma}$, typically arranged in descending order:

$$
\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0
$$

Where $r = \text{rank}(\mathbf{A})$ is the rank of the matrix.

### Relationship with Eigenvalues

SVD is closely related to eigenvalue decomposition:

| Matrix | Eigenvalues | Eigenvectors |
|--------|-----------|-------------|
| $\mathbf{A}\mathbf{A}^\top$ | $\sigma_i^2$ | Columns of $\mathbf{U}$ |
| $\mathbf{A}^\top\mathbf{A}$ | $\sigma_i^2$ | Columns of $\mathbf{V}$ |

**Derivation of relationship between SVD and eigenvalues**:

Let $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ be the SVD of $\mathbf{A}$.

**Step 1**: Compute $\mathbf{A}^\top\mathbf{A}$:

$$\mathbf{A}^\top\mathbf{A} = (\mathbf{V}\mathbf{\Sigma}^\top\mathbf{U}^\top)(\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top) = \mathbf{V}\mathbf{\Sigma}^\top\mathbf{U}^\top\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$$

Since $\mathbf{U}$ is an orthogonal matrix, $\mathbf{U}^\top\mathbf{U} = \mathbf{I}$, so:

$$\mathbf{A}^\top\mathbf{A} = \mathbf{V}\mathbf{\Sigma}^\top\mathbf{\Sigma}\mathbf{V}^\top = \mathbf{V}\mathbf{\Sigma}^2\mathbf{V}^\top$$

**Step 2**: Recognize this is the form of eigenvalue decomposition.

$\mathbf{V}\mathbf{\Sigma}^2\mathbf{V}^\top$ is exactly the eigenvalue decomposition of $\mathbf{A}^\top\mathbf{A}$:
- Columns of $\mathbf{V}$ are eigenvectors of $\mathbf{A}^\top\mathbf{A}$ (right singular vectors)
- $\mathbf{\Sigma}^2 = \text{diag}(\sigma_1^2, \ldots, \sigma_r^2)$ contains eigenvalues

**Step 3**: Similarly, compute $\mathbf{A}\mathbf{A}^\top$:

$$\mathbf{A}\mathbf{A}^\top = (\mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top)(\mathbf{V}\mathbf{\Sigma}^\top\mathbf{U}^\top) = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top\mathbf{V}\mathbf{\Sigma}^\top\mathbf{U}^\top$$

Since $\mathbf{V}^\top\mathbf{V} = \mathbf{I}$:

$$\mathbf{A}\mathbf{A}^\top = \mathbf{U}\mathbf{\Sigma}\mathbf{\Sigma}^\top\mathbf{U}^\top = \mathbf{U}\mathbf{\Sigma}^2\mathbf{U}^\top$$

**Step 4**: Summary:

$$\boxed{\mathbf{A}^\top\mathbf{A} = \mathbf{V}\mathbf{\Sigma}^2\mathbf{V}^\top, \quad \mathbf{A}\mathbf{A}^\top = \mathbf{U}\mathbf{\Sigma}^2\mathbf{U}^\top}$$

This means:
- Singular values $\sigma_i$ of SVD are the square roots of eigenvalues of $\mathbf{A}^\top\mathbf{A}$ and $\mathbf{A}\mathbf{A}^\top$
- Right singular vectors $\mathbf{V}$ are eigenvectors of $\mathbf{A}^\top\mathbf{A}$
- Left singular vectors $\mathbf{U}$ are eigenvectors of $\mathbf{A}\mathbf{A}^\top$

### Compact SVD

For a matrix of rank $r$:

$$
\mathbf{A} = \mathbf{U}_r\mathbf{\Sigma}_r\mathbf{V}_r^\top
$$

Where:
- $\mathbf{U}_r \in \mathbb{R}^{m \times r}$
- $\mathbf{\Sigma}_r \in \mathbb{R}^{r \times r}$
- $\mathbf{V}_r \in \mathbb{R}^{n \times r}$

### Truncated SVD (Low-rank Approximation)

Keep the first $k$ singular values ($k < r$):

$$
\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top = \mathbf{U}_k\mathbf{\Sigma}_k\mathbf{V}_k^\top
$$

**Eckart-Young Theorem**: $\mathbf{A}_k$ is the optimal rank $k$ approximation of $\mathbf{A}$ (in both Frobenius norm and spectral norm).

**Proof sketch of Eckart-Young Theorem**:

**Theorem statement**: For any matrix $\mathbf{B}$ of rank $k$:

$$\|\mathbf{A} - \mathbf{A}_k\|_F \leq \|\mathbf{A} - \mathbf{B}\|_F$$

Where $\mathbf{A}_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top$ is the truncated SVD.

**Proof**:

**Step 1**: Error of truncated SVD.

$$\|\mathbf{A} - \mathbf{A}_k\|_F^2 = \left\|\sum_{i=k+1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^\top\right\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$$

(Using orthogonality of singular vectors)

**Step 2**: Constraint of arbitrary rank $k$ matrix.

Let $\mathbf{B}$ be any rank $k$ matrix, then $\text{Null}(\mathbf{B})$ has dimension at least $n - k$.

**Step 3**: Use Courant-Fisher minimax theorem.

For any rank $k$ matrix $\mathbf{B}$:

$$\|\mathbf{A} - \mathbf{B}\|_F^2 \geq \sum_{i=k+1}^{r} \sigma_i^2 = \|\mathbf{A} - \mathbf{A}_k\|_F^2$$

**Step 4**: Conclusion.

Truncated SVD achieves the lower bound, so it is the optimal rank $k$ approximation:

$$\boxed{\mathbf{A}_k = \arg\min_{\text{rank}(\mathbf{B}) = k} \|\mathbf{A} - \mathbf{B}\|_F}$$

**Practical significance**: This theorem guarantees the optimality of SVD in tasks like data compression, denoising, and dimensionality reduction.

### Properties of SVD

| Property | Formula |
|----------|---------|
| **Rank** | $\text{rank}(\mathbf{A}) = r$ (number of non-zero singular values) |
| **Frobenius norm** | $\|\mathbf{A}\|_F = \sqrt{\sum_i \sigma_i^2}$ |
| **Spectral norm** | $\|\mathbf{A}\|_2 = \sigma_1$ (largest singular value) |
| **Nuclear norm** | $\|\mathbf{A}\|_* = \sum_i \sigma_i$ |
| **Condition number** | $\kappa(\mathbf{A}) = \sigma_1 / \sigma_r$ |

### Python Implementation

```python
import numpy as np

# SVD decomposition
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]], dtype=float)

# Full SVD
U, S, Vt = np.linalg.svd(A, full_matrices=True)

print(f"U shape: {U.shape}")   # (4, 4)
print(f"S shape: {S.shape}")   # (3,) - Only singular values
print(f"Vt shape: {Vt.shape}") # (3, 3)

print(f"Singular values: {S}")

# Reconstruct matrix
Sigma_full = np.zeros((4, 3))
Sigma_full[:3, :3] = np.diag(S)
A_reconstructed = U @ Sigma_full @ Vt
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.2e}")

# Compact SVD
U_compact, S_compact, Vt_compact = np.linalg.svd(A, full_matrices=False)
print(f"\nCompact SVD:")
print(f"U shape: {U_compact.shape}")   # (4, 3)
print(f"S shape: {S_compact.shape}")   # (3,)
print(f"Vt shape: {Vt_compact.shape}") # (3, 3)

# Truncated SVD (low-rank approximation)
def truncated_svd(A, k):
    """Truncated SVD, keeping first k singular values"""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# Approximations with different ranks
print("\nLow-rank approximation errors:")
for k in range(1, 4):
    A_k = truncated_svd(A, k)
    error = np.linalg.norm(A - A_k, 'fro')
    print(f"  k={k}: Frobenius error = {error:.4f}")

# Compute Frobenius norm and spectral norm
fro_norm = np.sqrt(np.sum(S**2))
spectral_norm = S[0]
print(f"\nFrobenius norm: {fro_norm:.4f}")
print(f"Spectral norm (largest singular value): {spectral_norm:.4f}")

# Verify relationship with eigenvalues
AA_T = A @ A.T
eigvals_AA_T = np.linalg.eigvals(AA_T)
eigvals_AA_T = np.sort(np.abs(eigvals_AA_T))[::-1]  # Sort in descending order
print(f"\nEigenvalues of A @ A^T: {eigvals_AA_T}")
print(f"Squares of singular values: {S**2}")
```

### Applications of SVD

```python
# 1. Matrix pseudo-inverse
def pseudo_inverse(A):
    """Compute pseudo-inverse using SVD"""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # Take reciprocal of singular values
    S_inv = np.array([1/s if s > 1e-10 else 0 for s in S])
    return Vt.T @ np.diag(S_inv) @ U.T

A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
A_pinv = pseudo_inverse(A)
A_pinv_np = np.linalg.pinv(A)

print(f"Custom pseudo-inverse:\n{A_pinv}")
print(f"NumPy pseudo-inverse:\n{A_pinv_np}")
print(f"Verify A @ A_pinv @ A ≈ A: {np.allclose(A @ A_pinv @ A, A)}")

# 2. Image compression
def compress_image(image, k):
    """Compress image using truncated SVD"""
    if len(image.shape) == 3:  # RGB image
        compressed = np.zeros_like(image)
        for c in range(3):
            U, S, Vt = np.linalg.svd(image[:, :, c], full_matrices=False)
            compressed[:, :, c] = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        return np.clip(compressed, 0, 255).astype(np.uint8)
    else:  # Grayscale image
        U, S, Vt = np.linalg.svd(image, full_matrices=False)
        return np.clip(U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :], 0, 255).astype(np.uint8)

# 3. Denoising
def denoise_svd(A, threshold=0.1):
    """Denoise using SVD"""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # Filter small singular values
    S_filtered = np.where(S > threshold * S[0], S, 0)
    return U @ np.diag(S_filtered) @ Vt

# 4. Principal Component Analysis (PCA)
def pca_svd(X, n_components):
    """Implement PCA using SVD"""
    # Center
    X_centered = X - np.mean(X, axis=0)
    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Principal components
    components = Vt[:n_components]
    # Transformed data (dimensionality reduced)
    transformed = X_centered @ components.T
    return transformed, components

# PCA example
X = np.random.randn(100, 10)  # 100 samples, 10 dimensions
X_pca, components = pca_svd(X, n_components=3)
print(f"\nPCA dimensionality reduction: {X.shape} -> {X_pca.shape}")
```

---

## QR Decomposition

### Definition

Any $m \times n$ matrix $\mathbf{A}$ ($m \geq n$) can be decomposed as:

$$
\mathbf{A} = \mathbf{Q}\mathbf{R}
$$

Where:
- $\mathbf{Q} \in \mathbb{R}^{m \times m}$ or $\mathbb{R}^{m \times n}$: Orthogonal matrix
- $\mathbf{R} \in \mathbb{R}^{m \times n}$ or $\mathbb{R}^{n \times n}$: Upper triangular matrix

### Complete QR Decomposition

$$
\mathbf{A}_{m \times n} = \mathbf{Q}_{m \times m}\mathbf{R}_{m \times n}
$$

### Reduced QR Decomposition

$$
\mathbf{A}_{m \times n} = \mathbf{Q}_{m \times n}\mathbf{R}_{n \times n}
$$

### Applications

1. **Solving linear systems**
2. **Least squares problems**
3. **Computing eigenvalues (QR algorithm)**
4. **Orthogonalization**

```python
import numpy as np

# QR decomposition
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]], dtype=float)

# Complete QR decomposition
Q_full, R_full = np.linalg.qr(A, mode='complete')
print(f"Complete QR - Q shape: {Q_full.shape}, R shape: {R_full.shape}")

# Reduced QR decomposition
Q_reduced, R_reduced = np.linalg.qr(A, mode='reduced')
print(f"Reduced QR - Q shape: {Q_reduced.shape}, R shape: {R_reduced.shape}")

# Verify
print(f"\nReconstruction error: {np.linalg.norm(A - Q_reduced @ R_reduced):.2e}")
print(f"Q orthogonality (Q^T @ Q = I): {np.allclose(Q_reduced.T @ Q_reduced, np.eye(3))}")

# Use QR decomposition to solve least squares problem
def lstsq_qr(A, b):
    """Solve least squares problem min ||Ax - b|| using QR decomposition"""
    Q, R = np.linalg.qr(A, mode='reduced')
    # Solve R @ x = Q^T @ b
    return np.linalg.solve(R, Q.T @ b)

A_over = np.array([[1, 1],
                   [1, 2],
                   [1, 3]], dtype=float)
b = np.array([1, 2, 2], dtype=float)

x_qr = lstsq_qr(A_over, b)
x_lstsq = np.linalg.lstsq(A_over, b, rcond=None)[0]
print(f"\nQR least squares solution: {x_qr}")
print(f"NumPy least squares solution: {x_lstsq}")

# Gram-Schmidt orthogonalization
def gram_schmidt(A):
    """Gram-Schmidt orthogonalization"""
    n = A.shape[1]
    Q = np.zeros_like(A)
    for j in range(n):
        v = A[:, j].copy()
        for i in range(j):
            v -= np.dot(Q[:, i], A[:, j]) * Q[:, i]
        Q[:, j] = v / np.linalg.norm(v)
    R = Q.T @ A
    return Q, R

A_test = np.array([[1, 1], [1, 2], [1, 3]], dtype=float)
Q_gs, R_gs = gram_schmidt(A_test)
print(f"\nGram-Schmidt Q:\n{Q_gs}")
print(f"Gram-Schmidt R:\n{R_gs}")
print(f"Reconstruction error: {np.linalg.norm(A_test - Q_gs @ R_gs):.2e}")
```

---

## Cholesky Decomposition

### Definition

For a **symmetric positive definite matrix** $\mathbf{A}$, it can be decomposed as:

$$
\mathbf{A} = \mathbf{L}\mathbf{L}^\top
$$

Where $\mathbf{L}$ is a lower triangular matrix.

Alternatively written as:

$$
\mathbf{A} = \mathbf{L}\mathbf{D}\mathbf{L}^\top
$$

Where $\mathbf{L}$ is a unit lower triangular matrix and $\mathbf{D}$ is a diagonal matrix.

### Uniqueness

For positive definite matrices, the Cholesky decomposition exists and is unique.

### Applications

1. **Solving linear systems** (twice as fast as LU decomposition)
2. **Generating correlated random variables**
3. **Computing log-determinants**

```python
import numpy as np

# Cholesky decomposition
A = np.array([[4, 2, 1],
              [2, 5, 3],
              [1, 3, 6]], dtype=float)

# Verify symmetric positive definite
print(f"Symmetric? {np.allclose(A, A.T)}")
eigvals = np.linalg.eigvals(A)
print(f"Eigenvalues: {eigvals}")
print(f"Positive definite? {all(eigvals > 0)}")

# Cholesky decomposition
L = np.linalg.cholesky(A)
print(f"\nL =\n{L}")
print(f"L @ L^T =\n{L @ L.T}")
print(f"Reconstruction error: {np.linalg.norm(A - L @ L.T):.2e}")

# Use Cholesky decomposition to solve Ax = b
def solve_cholesky(A, b):
    """Solve Ax = b using Cholesky decomposition"""
    L = np.linalg.cholesky(A)
    # Forward substitution to solve L @ y = b
    y = np.linalg.solve(L, b)
    # Back substitution to solve L^T @ x = y
    x = np.linalg.solve(L.T, y)
    return x

b = np.array([7, 10, 10], dtype=float)
x = solve_cholesky(A, b)
print(f"\nSolution: {x}")
print(f"Verify: A @ x = {A @ x}")

# Generate correlated random variables
def generate_correlated(n_samples, cov_matrix):
    """Generate correlated random variables using Cholesky decomposition"""
    L = np.linalg.cholesky(cov_matrix)
    z = np.random.randn(n_samples, cov_matrix.shape[0])
    return z @ L.T

cov = np.array([[1, 0.8], [0.8, 1]])
samples = generate_correlated(1000, cov)
print(f"\nGenerated sample correlation coefficient: {np.corrcoef(samples.T)[0, 1]:.4f}")  # Should be close to 0.8

# Compute log-determinant (avoid numerical issues of direct determinant computation)
def log_det_cholesky(A):
    """Compute log-determinant using Cholesky decomposition"""
    L = np.linalg.cholesky(A)
    return 2 * np.sum(np.log(np.diag(L)))

print(f"\nlog|A| (Cholesky): {log_det_cholesky(A):.4f}")
print(f"log|A| (direct): {np.log(np.linalg.det(A)):.4f}")
```

---

## LU Decomposition

### Definition

For most square matrices $\mathbf{A}$, it can be decomposed as:

$$
\mathbf{A} = \mathbf{L}\mathbf{U}
$$

Where:
- $\mathbf{L}$: Lower triangular matrix (diagonal elements are 1)
- $\mathbf{U}$: Upper triangular matrix

### LU Decomposition with Row Pivoting

In practice, row exchange is usually needed:

$$
\mathbf{PA} = \mathbf{LU}
$$

Where $\mathbf{P}$ is a permutation matrix.

### Applications

1. **Solving linear systems**
2. **Computing determinants**
3. **Matrix inversion**

```python
import numpy as np
from scipy.linalg import lu_factor, lu_solve, lu

# LU decomposition
A = np.array([[2, 1, 1],
              [4, 3, 3],
              [8, 7, 9]], dtype=float)

# Use scipy
P, L, U = lu(A)
print(f"P =\n{P}")
print(f"L =\n{L}")
print(f"U =\n{U}")
print(f"P @ A = L @ U? {np.allclose(P @ A, L @ U)}")

# Use lu_factor and lu_solve (more efficient)
lu_piv = lu_factor(A)
b = np.array([4, 10, 24], dtype=float)
x = lu_solve(lu_piv, b)
print(f"\nSolution: {x}")
print(f"Verify: A @ x = {A @ x}")

# Manual implementation of LU decomposition
def lu_decomposition(A):
    """LU decomposition without row exchange"""
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return L, U

A_small = np.array([[2, 1], [4, 3]], dtype=float)
L, U = lu_decomposition(A_small)
print(f"\nManual LU decomposition:")
print(f"L =\n{L}")
print(f"U =\n{U}")
print(f"L @ U =\n{L @ U}")

# Compute determinant
det_from_lu = np.prod(np.diag(U)) * np.linalg.det(P)
print(f"\n|A| (LU): {det_from_lu:.4f}")
print(f"|A| (direct): {np.linalg.det(A_small):.4f}")
```

---

## Other Decomposition Methods

### Polar Decomposition

Any square matrix $\mathbf{A}$ can be decomposed as:

$$
\mathbf{A} = \mathbf{UP}
$$

Where:
- $\mathbf{U}$: Orthogonal matrix
- $\mathbf{P}$: Positive semidefinite symmetric matrix

Computed via SVD: $\mathbf{U} = \mathbf{U}\mathbf{V}^\top$, $\mathbf{P} = \mathbf{V}\mathbf{\Sigma}\mathbf{V}^\top$

### Schur Decomposition

Any square matrix $\mathbf{A}$ can be decomposed as:

$$
\mathbf{A} = \mathbf{QUQ}^\top
$$

Where $\mathbf{Q}$ is an orthogonal matrix and $\mathbf{U}$ is an upper triangular matrix.

### Jordan Canonical Form

Every square matrix is similar to its Jordan canonical form, but this is rarely used in numerical computing.

```python
import numpy as np
from scipy.linalg import schur, polar

# Polar decomposition
A = np.array([[1, 2], [3, 4]], dtype=float)
U_polar, P = polar(A)
print(f"Polar decomposition:")
print(f"U (orthogonal):\n{U_polar}")
print(f"P (positive semidefinite):\n{P}")
print(f"U @ P =\n{U_polar @ P}")
print(f"Reconstruction error: {np.linalg.norm(A - U_polar @ P):.2e}")

# Schur decomposition
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
T, Z = schur(A)
print(f"\nSchur decomposition:")
print(f"Z (orthogonal):\n{Z}")
print(f"T (upper triangular):\n{T}")
print(f"Z @ T @ Z^T =\n{Z @ T @ Z.T}")
print(f"Reconstruction error: {np.linalg.norm(A - Z @ T @ Z.T):.2e}")
```

---

## Applications in Deep Learning

### Principal Component Analysis (PCA)

```python
import numpy as np

def pca(X, n_components):
    """
    Implement PCA using SVD

    Parameters:
        X: (n_samples, n_features) data matrix
        n_components: Number of principal components to retain

    Returns:
        X_transformed: Dimensionality-reduced data
        components: Principal components
        explained_variance: Explained variance
    """
    # Center
    X_centered = X - np.mean(X, axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Principal components
    components = Vt[:n_components]

    # Transformed data (dimensionality reduced)
    X_transformed = X_centered @ components.T

    # Explained variance
    explained_variance = (S ** 2) / (X.shape[0] - 1)
    explained_variance_ratio = explained_variance / np.sum(explained_variance)

    return X_transformed, components, explained_variance[:n_components], explained_variance_ratio[:n_components]

# Example: Dimensionality reduction
np.random.seed(42)
X = np.random.randn(100, 50)  # 100 samples, 50 dimensions

X_pca, components, var, var_ratio = pca(X, n_components=10)
print(f"PCA dimensionality reduction: {X.shape} -> {X_pca.shape}")
print(f"First 10 principal components explain variance ratio: {np.sum(var_ratio):.4f}")
```

### Orthogonal Initialization

```python
def orthogonal_init(shape, gain=1.0):
    """
    Orthogonal initialization - preserve gradient norms

    Parameters:
        shape: (fan_in, fan_out)
        gain: Scaling factor
    """
    if len(shape) < 2:
        raise ValueError("shape must be at least 2D")

    # Flatten all dimensions except the last one
    flat_shape = (np.prod(shape[:-1]), shape[-1])

    # Generate random matrix and perform SVD
    A = np.random.randn(*flat_shape)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Choose U or V (take the smaller dimension)
    if flat_shape[0] >= flat_shape[1]:
        Q = U
    else:
        Q = Vt

    # Reshape and scale
    return (gain * Q).reshape(shape)

# Example
W = orthogonal_init((256, 128))
print(f"Orthogonal initialization weight shape: {W.shape}")
print(f"W^T @ W ≈ I? {np.allclose(W.T @ W, np.eye(128), atol=1e-5)}")

# Verify: Preserves variance in forward propagation
x = np.random.randn(32, 256)
y = x @ W
print(f"Input variance: {np.var(x):.4f}")
print(f"Output variance: {np.var(y):.4f}")  # Should be close
```

### Spectral Normalization

```python
def spectral_norm(W, u=None, n_iters=1):
    """
    Power iteration method to compute spectral norm (largest singular value)

    Parameters:
        W: Weight matrix
        u: Initial vector
        n_iters: Number of iterations

    Returns:
        sigma: Spectral norm
        u: Updated vector
    """
    if u is None:
        u = np.random.randn(W.shape[0])
        u = u / np.linalg.norm(u)

    for _ in range(n_iters):
        v = W.T @ u
        v = v / np.linalg.norm(v)
        u = W @ v
        u = u / np.linalg.norm(u)

    sigma = u @ W @ v
    return sigma, u

def spectral_normalize(W, n_iters=1):
    """Spectral normalization: W / ||W||_2"""
    sigma, _ = spectral_norm(W, n_iters=n_iters)
    return W / sigma

# Example
W = np.random.randn(256, 128)
sigma, _ = spectral_norm(W, n_iters=10)
sigma_exact = np.linalg.svd(W, compute_uv=False)[0]

print(f"Power iteration spectral norm: {sigma:.6f}")
print(f"SVD exact spectral norm: {sigma_exact:.6f}")

W_normalized = spectral_normalize(W)
new_sigma, _ = spectral_norm(W_normalized, n_iters=10)
print(f"Spectral norm after normalization: {new_sigma:.6f}")  # Should be close to 1
```

### Low-rank Approximation Acceleration

```python
def low_rank_linear_layer(X, W, b, rank):
    """
    Accelerate linear layer using low-rank approximation

    Decompose W ≈ U @ V, where U is (d_in, rank), V is (rank, d_out)
    """
    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    # Truncate
    U_k = U[:, :rank]
    S_k = S[:rank]
    Vt_k = Vt[:rank, :]

    # Decompose into two steps: (X @ U_k) @ diag(S_k) @ Vt_k
    # Complexity reduced from O(d_in * d_out) to O(rank * (d_in + d_out))

    temp = X @ U_k  # (batch, rank)
    temp = temp * S_k  # Element-wise multiplication
    Y = temp @ Vt_k + b

    return Y

# Example
batch, d_in, d_out = 32, 512, 512
X = np.random.randn(batch, d_in)
W = np.random.randn(d_in, d_out)
b = np.random.randn(d_out)

rank = 64

# Exact computation
Y_exact = X @ W + b

# Low-rank approximation
Y_lowrank = low_rank_linear_layer(X, W, b, rank)

print(f"Exact output shape: {Y_exact.shape}")
print(f"Low-rank output shape: {Y_lowrank.shape}")
print(f"Relative error: {np.linalg.norm(Y_exact - Y_lowrank) / np.linalg.norm(Y_exact):.4f}")
```

### Attention Mechanism Optimization

```python
def efficient_attention(Q, K, V, rank=None):
    """
    Efficient attention computation (optional low-rank approximation)

    Normal: O(n^2 * d)
    Low-rank: O(n * rank * d)
    """
    d_k = Q.shape[-1]

    if rank is not None and rank < d_k:
        # Low-rank approximation of K and V
        U_k, S_k, Vt_k = np.linalg.svd(K, full_matrices=False)
        K_approx = (U_k[:, :rank] * S_k[:rank]) @ Vt_k[:rank, :]

        U_v, S_v, Vt_v = np.linalg.svd(V, full_matrices=False)
        V_approx = (U_v[:, :rank] * S_v[:rank]) @ Vt_v[:rank, :]

        K, V = K_approx, V_approx

    # Standard attention
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = weights / np.sum(weights, axis=-1, keepdims=True)
    output = weights @ V

    return output

# Example
n, d = 100, 64
Q = np.random.randn(n, d)
K = np.random.randn(n, d)
V = np.random.randn(n, d)

output_full = efficient_attention(Q, K, V)
output_lowrank = efficient_attention(Q, K, V, rank=32)

print(f"Full attention output shape: {output_full.shape}")
print(f"Low-rank attention output shape: {output_lowrank.shape}")
```

### Gradient Analysis

```python
def analyze_gradient_flow(model_gradients):
    """Analyze gradient flow (using SVD)"""
    for name, grad in model_gradients.items():
        if grad.ndim >= 2:
            U, S, Vt = np.linalg.svd(grad.reshape(grad.shape[0], -1), full_matrices=False)

            print(f"\n{name}:")
            print(f"  Shape: {grad.shape}")
            print(f"  Frobenius norm: {np.linalg.norm(grad):.4f}")
            print(f"  Spectral norm: {S[0]:.4f}")
            print(f"  Condition number: {S[0] / S[-1]:.2f}" if S[-1] > 1e-10 else "  Condition number: inf")
            print(f"  Effective rank (>1% max): {np.sum(S > 0.01 * S[0])}")
```

---

## Summary

This chapter introduced eigenvalues, eigenvectors, and various matrix decomposition methods:

| Concept | Definition | Applications |
|---------|-----------|-------------|
| Eigenvalues/Eigenvectors | $\mathbf{Av} = \lambda\mathbf{v}$ | PCA, stability analysis |
| Eigenvalue decomposition | $\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{-1}$ | Matrix functions, power operations |
| Singular value decomposition | $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ | Dimensionality reduction, pseudo-inverse, low-rank approximation |
| QR decomposition | $\mathbf{A} = \mathbf{Q}\mathbf{R}$ | Least squares, orthogonalization |
| Cholesky decomposition | $\mathbf{A} = \mathbf{L}\mathbf{L}^\top$ | Efficiently solve positive definite systems |
| LU decomposition | $\mathbf{A} = \mathbf{L}\mathbf{U}$ | General linear solver |

### Key Formula Summary

| Decomposition | Formula | Applicable conditions |
|--------------|---------|---------------------|
| EVD | $\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{-1}$ | Diagonalizable square matrices |
| Orthogonal EVD | $\mathbf{A} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^\top$ | Symmetric matrices |
| SVD | $\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^\top$ | Any matrix |
| QR | $\mathbf{A} = \mathbf{Q}\mathbf{R}$ | Full column rank matrices |
| Cholesky | $\mathbf{A} = \mathbf{L}\mathbf{L}^\top$ | Symmetric positive definite matrices |
| LU | $\mathbf{PA} = \mathbf{LU}$ | Most square matrices |

### Decomposition Method Selection Guide

| Task | Recommended method | Reason |
|------|-------------------|--------|
| Solve positive definite system | Cholesky | Fastest, numerically stable |
| Solve general square system | LU | General purpose |
| Solve overdetermined system | QR or SVD | Numerically stable |
| Low-rank approximation | SVD | Optimal approximation |
| PCA | SVD | Efficient and stable |
| Orthogonal initialization | SVD or QR | Preserves orthogonality |

---

**Previous section**: [Chapter 1(b): Linear Systems and Matrix Properties](01b-linear-systems-matrix-properties_EN.md)

**Next section**: [Chapter 1(d): Norms, Distances, and Applications](01d-norms-distances-applications_EN.md) - Learn about various norms, distance metrics, and their applications in deep learning.

**Back**: [Math Fundamentals Tutorial Table of Contents](../math-fundamentals.md)
