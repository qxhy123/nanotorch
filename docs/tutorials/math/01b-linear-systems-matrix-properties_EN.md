# Chapter 1(b): Linear Systems and Matrix Properties

Linear systems are one of the core problems in linear algebra, while determinants and matrix rank are key tools for understanding matrix properties. This chapter systematically introduces these concepts and their applications in deep learning.

---

## Table of Contents

1. [Linear Systems](#linear-systems)
2. [Determinants](#determinants)
3. [Matrix Rank](#matrix-rank)
4. [Linear Dependence and Independence](#linear-dependence-and-independence)
5. [Four Fundamental Subspaces](#four-fundamental-subspaces)
6. [Applications in Deep Learning](#applications-in-deep-learning)
7. [Summary](#summary)

---

## Linear Systems

### Basic Form

**Linear systems** are systems composed of several linear equations, expressed in matrix form as:

$$
\mathbf{Ax} = \mathbf{b}
$$

Where:
- $\mathbf{A} \in \mathbb{R}^{m \times n}$: Coefficient matrix
- $\mathbf{x} \in \mathbb{R}^n$: Unknown vector
- $\mathbf{b} \in \mathbb{R}^m$: Constant vector

**Expanded form**:

$$
\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}
$$

### Augmented Matrix

Combine the coefficient matrix and constant vector:

$$
[\mathbf{A}|\mathbf{b}] = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} & | & b_1 \\
a_{21} & a_{22} & \cdots & a_{2n} & | & b_2 \\
\vdots & \vdots & \ddots & \vdots & | & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn} & | & b_m
\end{bmatrix}
$$

### Solution Determination (Rouché-Capelli Theorem)

The solution of the linear system $\mathbf{Ax} = \mathbf{b}$ depends on the rank of the coefficient matrix $\mathbf{A}$ and the augmented matrix $[\mathbf{A}|\mathbf{b}]$:

| Rank Relationship | Solution Status |
|------------------|-----------------|
| $\text{rank}(\mathbf{A}) = \text{rank}([\mathbf{A}\|\mathbf{b}]) = n$ | Unique solution |
| $\text{rank}(\mathbf{A}) = \text{rank}([\mathbf{A}\|\mathbf{b}]) < n$ | Infinitely many solutions |
| $\text{rank}(\mathbf{A}) < \text{rank}([\mathbf{A}\|\mathbf{b}])$ | No solution |

Where $n$ is the number of unknowns.

### Special Case: Square System

For an $n \times n$ square system, the determination simplifies to:

| Condition | Solution Status |
|-----------|-----------------|
| $\det(\mathbf{A}) \neq 0$ | Unique solution $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$ |
| $\det(\mathbf{A}) = 0$ and $\mathbf{b} \in \text{range}(\mathbf{A})$ | Infinitely many solutions |
| $\det(\mathbf{A}) = 0$ and $\mathbf{b} \notin \text{range}(\mathbf{A})$ | No solution |

### Homogeneous Linear Systems

When $\mathbf{b} = \mathbf{0}$, $\mathbf{Ax} = \mathbf{0}$ is called a homogeneous linear system.

**Properties**:
- Homogeneous systems **always have a solution** (at least $\mathbf{x} = \mathbf{0}$ is the zero solution)
- Non-zero solutions exist if and only if $\det(\mathbf{A}) = 0$ (or $\text{rank}(\mathbf{A}) < n$)
- The set of solutions forms a **linear subspace**

### Solution Methods

#### 1. Direct Solution (Square Matrix, Invertible)

```python
import numpy as np

A = np.array([[3, 1],
              [1, 2]], dtype=float)
b = np.array([9, 8], dtype=float)

# Method 1: numpy.linalg.solve (recommended)
x = np.linalg.solve(A, b)
print(f"Solution: {x}")  # [2., 3.]

# Verify
print(f"A @ x = {A @ x}")  # [9., 8.]
```

#### 2. Matrix Inversion (Not Recommended)

```python
# Method 2: Inverse matrix (numerically unstable, low efficiency)
x_inv = np.linalg.inv(A) @ b
print(f"Inverse matrix solution: {x_inv}")  # [2., 3.]
```

**Why not recommended**:
- Numerically unstable (sensitive to condition number)
- High computational complexity ($O(n^3)$)
- Matrices are often non-invertible

#### 3. Least Squares (Overdetermined System)

When the number of equations exceeds the number of unknowns ($m > n$), there is usually no exact solution, so use least squares:

$$
\min_{\mathbf{x}} \|\mathbf{Ax} - \mathbf{b}\|_2^2
$$

Normal equations:

$$
\mathbf{A}^\top\mathbf{Ax} = \mathbf{A}^\top\mathbf{b}
$$

**Derivation of normal equations**:

The goal is to minimize the sum of squared residuals:

$$f(\mathbf{x}) = \|\mathbf{Ax} - \mathbf{b}\|_2^2 = (\mathbf{Ax} - \mathbf{b})^\top(\mathbf{Ax} - \mathbf{b})$$

**Step 1**: Expand the objective function:

$$f(\mathbf{x}) = \mathbf{x}^\top\mathbf{A}^\top\mathbf{Ax} - \mathbf{x}^\top\mathbf{A}^\top\mathbf{b} - \mathbf{b}^\top\mathbf{Ax} + \mathbf{b}^\top\mathbf{b}$$

Since $\mathbf{b}^\top\mathbf{Ax}$ is a scalar, it equals its transpose $\mathbf{x}^\top\mathbf{A}^\top\mathbf{b}$, so:

$$f(\mathbf{x}) = \mathbf{x}^\top\mathbf{A}^\top\mathbf{Ax} - 2\mathbf{x}^\top\mathbf{A}^\top\mathbf{b} + \mathbf{b}^\top\mathbf{b}$$

**Step 2**: Take the gradient with respect to $\mathbf{x}$ and set it to zero:

Using matrix derivative formulas $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^\top\mathbf{M}\mathbf{x}) = 2\mathbf{Mx}$ ($\mathbf{M}$ symmetric) and $\frac{\partial}{\partial \mathbf{x}}(\mathbf{c}^\top\mathbf{x}) = \mathbf{c}$:

$$\nabla f(\mathbf{x}) = 2\mathbf{A}^\top\mathbf{Ax} - 2\mathbf{A}^\top\mathbf{b} = \mathbf{0}$$

**Step 3**: Rearrange to obtain the normal equations:

$$\mathbf{A}^\top\mathbf{Ax} = \mathbf{A}^\top\mathbf{b}$$

**Step 4**: Solve the normal equations:

If $\mathbf{A}^\top\mathbf{A}$ is invertible (i.e., $\mathbf{A}$ is full column rank), then:

$$\boxed{\mathbf{x}^* = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}}$$

This is the least squares solution.

**Note**: $(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$ is called the **left pseudo-inverse** of $\mathbf{A}$, denoted $\mathbf{A}^+$.

```python
# Overdetermined system (more equations than unknowns)
A_over = np.array([[1, 1],
                   [1, 2],
                   [1, 3]], dtype=float)
b_over = np.array([1, 2, 2], dtype=float)

# Least squares solution
x_lstsq, residuals, rank, s = np.linalg.lstsq(A_over, b_over, rcond=None)
print(f"Least squares solution: {x_lstsq}")
print(f"Residuals: {residuals}")
print(f"Rank: {rank}")

# Verify using normal equations
x_normal = np.linalg.solve(A_over.T @ A_over, A_over.T @ b_over)
print(f"Normal equations solution: {x_normal}")
```

#### 4. Underdetermined System ($m < n$)

When there are more unknowns than equations, there are infinitely many solutions, and a minimum norm solution can be used:

$$
\mathbf{x}^* = \mathbf{A}^\top(\mathbf{AA}^\top)^{-1}\mathbf{b}
$$

```python
# Underdetermined system (more unknowns than equations)
A_under = np.array([[1, 2, 3],
                    [4, 5, 6]], dtype=float)
b_under = np.array([6, 15], dtype=float)

# Use pseudo-inverse to find minimum norm solution
A_pinv = np.linalg.pinv(A_under)
x_min_norm = A_pinv @ b_under
print(f"Minimum norm solution: {x_min_norm}")
print(f"Norm of solution: {np.linalg.norm(x_min_norm)}")
```

### Gaussian Elimination

Gaussian elimination is a classic method for solving linear systems:

1. **Forward elimination**: Transform the matrix to upper triangular form
2. **Back substitution**: Solve starting from the last row

```python
def gaussian_elimination(A, b):
    """Solve Ax = b using Gaussian elimination"""
    n = len(b)
    # Construct augmented matrix
    Ab = np.hstack([A, b.reshape(-1, 1)])

    # Forward elimination
    for i in range(n):
        # Find pivot
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        Ab[[i, max_row]] = Ab[[max_row, i]]

        # Elimination
        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:] -= factor * Ab[i, i:]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:])) / Ab[i, i]

    return x

A = np.array([[3, 1, 2],
              [1, 2, 1],
              [2, 1, 3]], dtype=float)
b = np.array([9, 8, 11], dtype=float)

x = gaussian_elimination(A, b)
print(f"Gaussian elimination solution: {x}")
print(f"Verification: A @ x = {A @ x}")
```

### LU Decomposition Solution

LU decomposition decomposes a matrix into the product of lower and upper triangular matrices:

$$
\mathbf{A} = \mathbf{LU}
$$

Solving $\mathbf{Ax} = \mathbf{b}$ becomes:
1. Solve $\mathbf{Ly} = \mathbf{b}$ (forward substitution)
2. Solve $\mathbf{Ux} = \mathbf{y}$ (back substitution)

```python
from scipy.linalg import lu_factor, lu_solve

A = np.array([[3, 1, 2],
              [1, 2, 1],
              [2, 1, 3]], dtype=float)
b = np.array([9, 8, 11], dtype=float)

# LU decomposition
lu, piv = lu_factor(A)
x = lu_solve((lu, piv), b)
print(f"LU decomposition solution: {x}")
```

---

## Determinants

### Definition

The **determinant** $\det(\mathbf{A})$ or $|\mathbf{A}|$ of an $n \times n$ square matrix $\mathbf{A}$ is a scalar that reflects the "volume scaling factor" and invertibility of the matrix.

### Low-order Determinants

**2×2 matrix**:

$$
\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc
$$

**3×3 matrix** (Sarrus rule):

$$
\det\begin{bmatrix} a & b & c \\ d & e & f \\ g & h & i \end{bmatrix} = aei + bfg + cdh - ceg - bdi - afh
$$

### General Definition (Row/Column Expansion)

Expand along the $i$-th row (Laplace expansion):

$$
\det(\mathbf{A}) = \sum_{j=1}^n (-1)^{i+j} A_{ij} M_{ij}
$$

Where:
- $M_{ij}$ is the **minor**: the determinant of the submatrix obtained by removing the $i$-th row and $j$-th column
- $C_{ij} = (-1)^{i+j} M_{ij}$ is the **cofactor**

### Properties of Determinants

| Property | Formula |
|----------|---------|
| Identity matrix | $\det(\mathbf{I}) = 1$ |
| Transpose invariant | $\det(\mathbf{A}^\top) = \det(\mathbf{A})$ |
| Product | $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$ |
| Inverse matrix | $\det(\mathbf{A}^{-1}) = \frac{1}{\det(\mathbf{A})}$ |
| Scalar multiplication | $\det(\alpha\mathbf{A}) = \alpha^n \det(\mathbf{A})$ ($n \times n$ matrix) |
| Swapping two rows | Determinant changes sign |
| Two identical rows | Determinant is zero |
| A row is zero | Determinant is zero |
| Linear combination of rows | Does not change determinant |

**Proof of determinant multiplication property $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$**:

**Method 1: Using eigenvalues**

**Step 1**: Let $\lambda_1, \ldots, \lambda_n$ be the eigenvalues of $\mathbf{A}$, and $\mu_1, \ldots, \mu_n$ be the eigenvalues of $\mathbf{B}$.

We know $\det(\mathbf{A}) = \prod_{i=1}^n \lambda_i$ and $\det(\mathbf{B}) = \prod_{i=1}^n \mu_i$.

**Step 2**: Consider the eigenvalues of $\mathbf{AB}$.

If $\mathbf{v}$ is an eigenvector of $\mathbf{B}$, $\mathbf{Bv} = \mu \mathbf{v}$, then:

$$\mathbf{ABv} = \mathbf{A}(\mu\mathbf{v}) = \mu(\mathbf{Av})$$

But $\mathbf{Av}$ is not necessarily an eigenvector of $\mathbf{A}$. Let's try another approach.

**Step 3**: Use diagonalization (assuming diagonalizable).

Let $\mathbf{A} = \mathbf{P}\mathbf{D}_A\mathbf{P}^{-1}$, $\mathbf{B} = \mathbf{Q}\mathbf{D}_B\mathbf{Q}^{-1}$, where $\mathbf{D}_A = \text{diag}(\lambda_1, \ldots, \lambda_n)$ and $\mathbf{D}_B = \text{diag}(\mu_1, \ldots, \mu_n)$.

**Step 4**: Compute $\det(\mathbf{AB})$:

$$\det(\mathbf{AB}) = \det(\mathbf{P}\mathbf{D}_A\mathbf{P}^{-1}\mathbf{Q}\mathbf{D}_B\mathbf{Q}^{-1})$$

Using $\det(\mathbf{XY}) = \det(\mathbf{X})\det(\mathbf{Y})$ (which we are proving) would be circular. Let's use the elementary matrix method instead.

**Method 2: Using elementary matrices**

**Step 1**: Any matrix $\mathbf{A}$ can be transformed to an upper triangular matrix $\mathbf{U}$ through elementary row operations:

$$\mathbf{A} = \mathbf{E}_k \cdots \mathbf{E}_2 \mathbf{E}_1 \mathbf{U}$$

Where $\mathbf{E}_i$ are elementary matrices.

**Step 2**: For elementary matrices, directly verify $\det(\mathbf{E}_1\mathbf{E}_2) = \det(\mathbf{E}_1)\det(\mathbf{E}_2)$.

- Swap two rows: $\det = -1$
- Multiply a row by $c$: $\det = c$
- Add a multiple of one row to another: $\det = 1$

The multiplication property holds in all three cases.

**Step 3**: For an upper triangular matrix $\mathbf{U}$, $\det(\mathbf{U}) = \prod_{i} U_{ii}$.

Similarly, $\mathbf{B} = \mathbf{F}_l \cdots \mathbf{F}_1 \mathbf{V}$ ($\mathbf{V}$ upper triangular).

**Step 4**: Therefore, $\det(\mathbf{A})\det(\mathbf{B}) = \det(\mathbf{U})\det(\mathbf{V}) \cdot (\text{product of elementary matrix determinants})$

$\det(\mathbf{AB}) = \det(\mathbf{E}_k \cdots \mathbf{F}_l \cdots \mathbf{U}\mathbf{B}) = \ldots = \det(\mathbf{U})\det(\mathbf{V}) \cdot (\text{same factors})$

$$\boxed{\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})}$$

**Derivation of inverse matrix determinant $\det(\mathbf{A}^{-1}) = \frac{1}{\det(\mathbf{A})}$**:

From $\mathbf{AA}^{-1} = \mathbf{I}$, take the determinant of both sides:

$$\det(\mathbf{AA}^{-1}) = \det(\mathbf{I})$$

$$\det(\mathbf{A})\det(\mathbf{A}^{-1}) = 1$$

$$\boxed{\det(\mathbf{A}^{-1}) = \frac{1}{\det(\mathbf{A})}}$$

### Determinants of Special Matrices

**Diagonal matrix**:

$$
\det(\text{diag}(d_1, \ldots, d_n)) = \prod_{i=1}^n d_i
$$

**Triangular matrix**:

$$
\det\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
0 & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & a_{nn}
\end{bmatrix} = \prod_{i=1}^n a_{ii}
$$

**Orthogonal matrix**:

$$
\det(\mathbf{Q}) = \pm 1
$$

**Block matrix** ($\mathbf{A}$ invertible):

$$
\det\begin{bmatrix} \mathbf{A} & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{bmatrix} = \det(\mathbf{A})\det(\mathbf{D} - \mathbf{CA}^{-1}\mathbf{B})
$$

### Geometric Meaning

The absolute value of the determinant represents the factor by which a linear transformation scales volume:

| Value of $\det(\mathbf{A})$ | Geometric Meaning |
|----------------------------|-------------------|
| $\det(\mathbf{A}) > 0$ | Preserves orientation (positive volume) |
| $\det(\mathbf{A}) < 0$ | Reverses orientation (negative volume) |
| $\det(\mathbf{A}) = 0$ | Dimensionality reduction (zero volume, non-invertible) |
| $\det(\mathbf{A}) = 1$ | Preserves volume (e.g., rotation) |

**2D example**: The determinant represents the area of the parallelogram spanned by the column vectors.

**3D example**: The determinant represents the volume of the parallelepiped spanned by the column vectors.

### Python Implementation

```python
import numpy as np

# Determinant of 2x2 matrix
A_2x2 = np.array([[1, 2],
                  [3, 4]])
det_2x2 = np.linalg.det(A_2x2)
print(f"det(A_2x2) = {det_2x2}")  # -2.0

# Analytic verification: 1*4 - 2*3 = -2

# Determinant of 3x3 matrix
A_3x3 = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 10]])
det_3x3 = np.linalg.det(A_3x3)
print(f"det(A_3x3) = {det_3x3}")  # -3.0

# Determinant of diagonal matrix
D = np.diag([1, 2, 3, 4])
det_D = np.linalg.det(D)
print(f"det(diag) = {det_D}")  # 24 = 1*2*3*4

# Property verification
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# det(AB) = det(A) * det(B)
det_AB = np.linalg.det(A @ B)
det_A_det_B = np.linalg.det(A) * np.linalg.det(B)
print(f"det(AB) = {det_AB:.2f}")
print(f"det(A) * det(B) = {det_A_det_B:.2f}")
print(f"Equal? {np.allclose(det_AB, det_A_det_B)}")  # True

# det(A^T) = det(A)
det_A = np.linalg.det(A)
det_At = np.linalg.det(A.T)
print(f"det(A) = {det_A}, det(A^T) = {det_At}")
print(f"Equal? {np.allclose(det_A, det_At)}")  # True
```

### Determinant and Invertibility

```python
def is_invertible(A, tol=1e-10):
    """Check if a matrix is invertible"""
    det = np.linalg.det(A)
    return abs(det) > tol

# Invertible matrix
A_inv = np.array([[1, 2], [3, 4]])
print(f"A invertible? {is_invertible(A_inv)}")  # True (det = -2)

# Non-invertible matrix (singular matrix)
A_sing = np.array([[1, 2], [2, 4]])
print(f"Singular matrix invertible? {is_invertible(A_sing)}")  # False (det = 0)

# Verification: Second row is 2 times the first row
```

---

## Matrix Rank

### Definition

The **rank** $\text{rank}(\mathbf{A})$ of matrix $\mathbf{A}$ is the maximum number of linearly independent row (or column) vectors.

### Equivalent Definitions

Matrix rank has the following equivalent definitions:

1. **Maximum number of linearly independent row vectors** (row rank)
2. **Maximum number of linearly independent column vectors** (column rank)
3. **Number of non-zero singular values**
4. **Order of highest-order non-zero minor**
5. **Dimension of the matrix image space**: $\text{rank}(\mathbf{A}) = \dim(\text{range}(\mathbf{A}))$

### Properties

| Property | Formula |
|----------|---------|
| Range | $0 \leq \text{rank}(\mathbf{A}_{m \times n}) \leq \min(m, n)$ |
| Transpose invariant | $\text{rank}(\mathbf{A}) = \text{rank}(\mathbf{A}^\top)$ |
| Product | $\text{rank}(\mathbf{AB}) \leq \min(\text{rank}(\mathbf{A}), \text{rank}(\mathbf{B}))$ |
| Addition | $\text{rank}(\mathbf{A} + \mathbf{B}) \leq \text{rank}(\mathbf{A}) + \text{rank}(\mathbf{B})$ |
| With inverse matrix | $\text{rank}(\mathbf{A}^{-1}\mathbf{A}) = \text{rank}(\mathbf{A})$ |

### Full Rank

For an $m \times n$ matrix $\mathbf{A}$:

| Type | Definition | Condition |
|------|-----------|-----------|
| **Full rank** | $\text{rank}(\mathbf{A}) = \min(m, n)$ | Rank reaches maximum possible value |
| **Full column rank** | $\text{rank}(\mathbf{A}) = n$ | Column vectors are linearly independent |
| **Full row rank** | $\text{rank}(\mathbf{A}) = m$ | Row vectors are linearly independent |
| **Full rank square matrix** | $\text{rank}(\mathbf{A}) = n = m$ | Invertible matrix |

### Relationship Between Rank and Determinant

For an $n \times n$ square matrix:

$$
\text{rank}(\mathbf{A}) = n \iff \det(\mathbf{A}) \neq 0 \iff \mathbf{A} \text{ is invertible}
$$

### Computing Rank

**Method 1**: Use Gaussian elimination to reduce to row echelon form; the number of non-zero rows is the rank.

**Method 2**: Count the number of non-zero singular values.

```python
import numpy as np

# Compute matrix rank
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
rank_A = np.linalg.matrix_rank(A)
print(f"rank(A) = {rank_A}")  # 2 (third row is a linear combination of first two)

# Verification: Third row = 2*second row - first row
print(f"Third row: {A[2]}")
print(f"2*second row - first row: {2*A[1] - A[0]}")  # [7, 8, 9]

# Full rank matrix
B = np.array([[1, 0],
              [0, 1]])
rank_B = np.linalg.matrix_rank(B)
print(f"rank(B) = {rank_B}")  # 2 (full rank)

# Compute rank through singular values
U, S, Vt = np.linalg.svd(A)
print(f"Singular values: {S}")
print(f"Number of non-zero singular values: {np.sum(S > 1e-10)}")  # 2

# Relationship between rank and determinant
C = np.array([[1, 2], [3, 4]])
print(f"rank(C) = {np.linalg.matrix_rank(C)}")  # 2
print(f"det(C) = {np.linalg.det(C)}")  # -2 (non-zero, full rank)

D = np.array([[1, 2], [2, 4]])
print(f"rank(D) = {np.linalg.matrix_rank(D)}")  # 1
print(f"det(D) = {np.linalg.det(D)}")  # 0 (zero, not full rank)
```

### Rank Factorization

Any $m \times n$ matrix $\mathbf{A}$ can be decomposed as:

$$
\mathbf{A} = \mathbf{CR}
$$

Where $\mathbf{C} \in \mathbb{R}^{m \times r}$ is a full column rank matrix, $\mathbf{R} \in \mathbb{R}^{r \times n}$ is a full row rank matrix, and $r = \text{rank}(\mathbf{A})$.

---

## Linear Dependence and Independence

### Definitions

**Linear combination**: A linear combination of vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_k$ is:

$$
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k
$$

**Linear dependence**: If there exist non-zero scalars $c_1, c_2, \ldots, c_k$ (not all zero) such that:

$$
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}
$$

Then these vectors are said to be **linearly dependent**.

**Linear independence**: If the above equation holds only when $c_1 = c_2 = \cdots = c_k = 0$, then these vectors are said to be **linearly independent**.

### Determination Methods

1. **Definition method**: Check if the homogeneous system has non-zero solutions
2. **Determinant method**: For $n$ vectors in $n$-dimensional space, if the determinant of the matrix formed by them is non-zero, they are linearly independent
3. **Rank method**: If the rank of the matrix formed by the vectors equals the number of vectors, they are linearly independent

### Properties

- If a vector set contains a zero vector, it is linearly dependent
- If a vector set contains two identical vectors, it is linearly dependent
- If one vector in a set is a linear combination of the others, the set is linearly dependent
- In $n$-dimensional space, there are at most $n$ linearly independent vectors

```python
import numpy as np

def check_linear_independence(vectors):
    """Check if a set of vectors is linearly independent"""
    # Build matrix (vectors as columns)
    A = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(A)
    n_vectors = len(vectors)

    return rank == n_vectors, rank

# Linearly independent vectors
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])

indep, rank = check_linear_independence([v1, v2, v3])
print(f"Unit vectors linearly independent? {indep}, rank = {rank}")  # True, 3

# Linearly dependent vectors
v4 = np.array([1, 2, 3])
v5 = np.array([2, 4, 6])  # v5 = 2 * v4
v6 = np.array([3, 6, 9])  # v6 = 3 * v4

indep, rank = check_linear_independence([v4, v5, v6])
print(f"Dependent vectors linearly independent? {indep}, rank = {rank}")  # False, 1

# Use determinant to determine (square matrix case)
A = np.array([[1, 2, 3],
              [0, 1, 4],
              [5, 6, 0]])
det_A = np.linalg.det(A)
print(f"det(A) = {det_A:.2f}")  # Non-zero means linearly independent
```

---

## Four Fundamental Subspaces

For $\mathbf{A} \in \mathbb{R}^{m \times n}$, there are four important subspaces:

### 1. Column Space (Image Space)

$$
\text{Col}(\mathbf{A}) = \text{range}(\mathbf{A}) = \{\mathbf{Ax} : \mathbf{x} \in \mathbb{R}^n\} \subseteq \mathbb{R}^m
$$

- Space spanned by the column vectors of the matrix
- Dimension: $\dim(\text{Col}(\mathbf{A})) = \text{rank}(\mathbf{A})$

### 2. Row Space

$$
\text{Row}(\mathbf{A}) = \text{Col}(\mathbf{A}^\top) \subseteq \mathbb{R}^n
$$

- Space spanned by the row vectors of the matrix
- Dimension: $\dim(\text{Row}(\mathbf{A})) = \text{rank}(\mathbf{A})$

### 3. Null Space (Kernel)

$$
\text{Null}(\mathbf{A}) = \{\mathbf{x} : \mathbf{Ax} = \mathbf{0}\} \subseteq \mathbb{R}^n
$$

- Solution space of the homogeneous equation $\mathbf{Ax} = \mathbf{0}$
- Dimension: $\dim(\text{Null}(\mathbf{A})) = n - \text{rank}(\mathbf{A})$ (nullity)

### 4. Left Null Space

$$
\text{Null}(\mathbf{A}^\top) = \{\mathbf{y} : \mathbf{A}^\top\mathbf{y} = \mathbf{0}\} \subseteq \mathbb{R}^m
$$

- Dimension: $\dim(\text{Null}(\mathbf{A}^\top)) = m - \text{rank}(\mathbf{A})$

### Fundamental Theorem (Rank-Nullity Theorem)

$$
\text{rank}(\mathbf{A}) + \dim(\text{Null}(\mathbf{A})) = n
$$

$$
\text{rank}(\mathbf{A}) + \dim(\text{Null}(\mathbf{A}^\top)) = m
$$

**Proof of rank-nullity theorem**:

Let $\mathbf{A} \in \mathbb{R}^{m \times n}$, $\text{rank}(\mathbf{A}) = r$.

**Step 1**: Through elementary transformations, $\mathbf{A}$ can be reduced to row echelon form:

$$\mathbf{A} \sim \begin{bmatrix} \mathbf{I}_r & \mathbf{F} \\ \mathbf{0} & \mathbf{0} \end{bmatrix}$$

Where $\mathbf{I}_r$ is an $r \times r$ identity matrix, and $\mathbf{F}$ is an $r \times (n-r)$ matrix.

**Step 2**: Dimension of column space.

The first $r$ columns of the row echelon form are linearly independent (pivot columns), so:

$$\dim(\text{Col}(\mathbf{A})) = r = \text{rank}(\mathbf{A})$$

**Step 3**: Dimension of null space.

Solving $\mathbf{Ax} = \mathbf{0}$ is equivalent to solving the homogeneous system of the row echelon form.

There are $r$ pivot variables and $n - r$ free variables.

Each free variable corresponds to a basis vector of the null space, so:

$$\dim(\text{Null}(\mathbf{A})) = n - r$$

**Step 4**: Combine results:

$$\text{rank}(\mathbf{A}) + \dim(\text{Null}(\mathbf{A})) = r + (n - r) = n$$

$$\boxed{\text{rank}(\mathbf{A}) + \dim(\text{Null}(\mathbf{A})) = n}$$

**Intuitive understanding**:
- $\text{rank}(\mathbf{A})$ = dimension of information preserved
- $\dim(\text{Null}(\mathbf{A}))$ = dimension of information lost
- Their sum equals the input space dimension $n$

### Orthogonal Relationships

- $\text{Row}(\mathbf{A}) \perp \text{Null}(\mathbf{A})$ (orthogonal complement in $\mathbb{R}^n$)
- $\text{Col}(\mathbf{A}) \perp \text{Null}(\mathbf{A}^\top)$ (orthogonal complement in $\mathbb{R}^m$)

```python
import numpy as np

def compute_null_space(A, tol=1e-10):
    """Compute the null space of a matrix"""
    U, S, Vt = np.linalg.svd(A)
    # Find indices of singular values below threshold
    null_mask = S < tol
    # Null space is the corresponding rows of V^T
    null_space = Vt[len(S):, :].T if len(S) < A.shape[1] else Vt[null_mask, :].T
    if null_space.size == 0:
        # No zero singular values, null space only contains zero vector
        return np.zeros((A.shape[1], 0))
    return null_space

A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

# Compute rank
rank = np.linalg.matrix_rank(A)
print(f"rank(A) = {rank}")  # 2

# Verify rank-nullity theorem
n = A.shape[1]
nullity = n - rank
print(f"Nullity = {nullity}")  # 1

# Null space
null_space = compute_null_space(A)
print(f"Null space basis vectors:\n{null_space}")

# Verification: A @ null_vector ≈ 0
if null_space.size > 0:
    print(f"A @ null_space = {A @ null_space[:, 0]}")
```

---

## Applications in Deep Learning

### Invertibility of Weight Matrices

In deep networks, weight matrices are usually not square matrices. Even when determinants are undefined, rank is still important:

- **Full rank weights**: Information can be completely transmitted
- **Low rank weights**: May indicate redundancy or over-parameterization

```python
import numpy as np

# Weights in deep networks
d_in, d_out = 784, 128
W = np.random.randn(d_in, d_out)

rank_W = np.linalg.matrix_rank(W)
print(f"Weight matrix rank: {rank_W}")
print(f"Maximum possible rank: {min(d_in, d_out)}")
print(f"Full rank? {rank_W == min(d_in, d_out)}")

# Low rank initialization (can cause problems)
W_low_rank = np.random.randn(d_in, 10) @ np.random.randn(10, d_out)
rank_low = np.linalg.matrix_rank(W_low_rank)
print(f"Low rank weight rank: {rank_low}")  # At most 10
```

### Over-parameterization and Under-parameterization

```python
def analyze_model_capacity(n_samples, n_features, n_parameters):
    """Analyze model capacity"""
    if n_parameters >= n_samples:
        return "Over-parameterized (may overfit)"
    elif n_parameters == n_features:
        return "Just-determined"
    else:
        return "Under-parameterized (may underfit)"

# Example: MNIST classification
n_samples = 60000  # Number of training samples
n_features = 784   # Number of features

# Single-layer network: 784 -> 10
n_params_single = 784 * 10 + 10
print(f"Single-layer network parameters: {n_params_single}")
print(analyze_model_capacity(n_samples, n_features, n_params_single))

# Two-layer network: 784 -> 256 -> 10
n_params_two = 784 * 256 + 256 + 256 * 10 + 10
print(f"Two-layer network parameters: {n_params_two}")
print(analyze_model_capacity(n_samples, n_features, n_params_two))
```

### Linear Systems and Backpropagation

Backpropagation involves solving linear systems (in a sense):

```python
def linear_layer_gradients(X, W, dL_dY):
    """
    Compute gradients for a linear layer
    Y = XW + b

    Parameters:
        X: (batch, in_features)
        W: (in_features, out_features)
        dL_dY: (batch, out_features) - gradient with respect to output

    Returns:
        dL_dX, dL_dW, dL_db
    """
    # Gradient w.r.t. W: X^T @ dL_dY
    dL_dW = X.T @ dL_dY

    # Gradient w.r.t. X: dL_dY @ W^T
    dL_dX = dL_dY @ W.T

    # Gradient w.r.t. b: sum over batch dimension
    dL_db = np.sum(dL_dY, axis=0)

    return dL_dX, dL_dW, dL_db

# Example
batch, in_feat, out_feat = 32, 784, 128
X = np.random.randn(batch, in_feat)
W = np.random.randn(in_feat, out_feat)
dL_dY = np.random.randn(batch, out_feat)

dL_dX, dL_dW, dL_db = linear_layer_gradients(X, W, dL_dY)
print(f"dL_dX shape: {dL_dX.shape}")  # (32, 784)
print(f"dL_dW shape: {dL_dW.shape}")  # (784, 128)
print(f"dL_db shape: {dL_db.shape}")  # (128,)
```

### Assessing Weight Matrix Health

```python
def analyze_weight_matrix(W, name="W"):
    """Analyze the health of a weight matrix"""
    rank = np.linalg.matrix_rank(W)
    m, n = W.shape

    # Singular value analysis
    U, S, Vt = np.linalg.svd(W, full_matrices=False)

    # Condition number
    cond = S[0] / S[-1] if S[-1] > 1e-10 else np.inf

    print(f"\n{name} analysis:")
    print(f"  Shape: {m} x {n}")
    print(f"  Rank: {rank} / {min(m, n)}")
    print(f"  Condition number: {cond:.2e}")
    print(f"  Max singular value: {S[0]:.4f}")
    print(f"  Min singular value: {S[-1]:.4e}")

    # Judgments
    if rank < min(m, n):
        print(f"  Warning: Matrix is not full rank!")
    if cond > 1e6:
        print(f"  Warning: Condition number too large, possibly numerically unstable!")

    return rank, cond, S

# Healthy weights
W_healthy = np.random.randn(256, 128) * np.sqrt(2.0 / 256)
analyze_weight_matrix(W_healthy, "Healthy weights")

# Unhealthy weights (low rank)
W_lowrank = np.random.randn(256, 10) @ np.random.randn(10, 128)
analyze_weight_matrix(W_lowrank, "Low rank weights")
```

### Equivalence of Linear Layers

```python
def check_linear_layer_equivalence(W1, W2, b1, b2, X, tol=1e-6):
    """Check if two linear layers are equivalent"""
    Y1 = X @ W1 + b1
    Y2 = X @ W2 + b2
    return np.allclose(Y1, Y2, atol=tol)

# Equivalent transformation of linear layers
d = 64
W = np.random.randn(d, d)

# Equivalent layer after orthogonal transformation
Q = np.linalg.qr(np.random.randn(d, d))[0]  # Orthogonal matrix
W_equiv = Q.T @ W @ Q

# Verify: May not be equivalent on some inputs (because transformation is different)
X = np.random.randn(10, d)
Y_original = X @ W
Y_transformed = X @ W_equiv
print(f"Output similar after orthogonal transformation? {np.allclose(Y_original, Y_transformed)}")  # Usually False
```

---

## Summary

This chapter introduced the core concepts of linear systems, determinants, and matrix rank:

| Concept | Definition | Application in Deep Learning |
|---------|-----------|----------------------------|
| Linear systems | $\mathbf{Ax} = \mathbf{b}$ | Solving, least squares, backpropagation |
| Determinants | "Volume scaling factor" of a matrix | Determining invertibility, initialization |
| Matrix rank | Maximum number of linearly independent rows/columns | Analyzing over-parameterization, low-rank approximation |
| Linear dependence | Existence of non-trivial linear combination equaling zero | Understanding redundancy, regularization |
| Null space | Solution space of $\mathbf{Ax} = \mathbf{0}$ | Understanding constraints, gradients |

### Key Formula Summary

| Formula | Meaning |
|---------|---------|
| $\text{rank}(\mathbf{A}) + \dim(\text{Null}(\mathbf{A})) = n$ | Rank-nullity theorem |
| $\det(\mathbf{AB}) = \det(\mathbf{A})\det(\mathbf{B})$ | Multiplicativity of determinants |
| $\text{rank}(\mathbf{A}) = n \iff \det(\mathbf{A}) \neq 0$ | Relationship between full rank and determinant |
| $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$ | Unique solution (when $\det(\mathbf{A}) \neq 0$) |
| $\mathbf{x}^* = \arg\min \|\mathbf{Ax} - \mathbf{b}\|_2^2$ | Least squares solution |

---

**Previous section**: [Chapter 1(a): Vectors and Matrices Basics](01a-vectors-matrices-basics_EN.md)

**Next section**: [Chapter 1(c): Eigenvalues and Matrix Decomposition](01c-eigenvalues-matrix-decomposition_EN.md) - Learn about eigenvalues, eigenvectors, and various matrix decomposition methods.

**Back**: [Math Fundamentals Tutorial Table of Contents](../math-fundamentals.md)
