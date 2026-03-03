# Chapter 1(a): Vectors and Matrices Basics

Linear algebra is the **core mathematical foundation** of deep learning. All computations in neural networks—from the simplest fully connected layers to complex Transformers—are essentially linear algebra operations. This chapter systematically introduces the basics of vectors and matrices, which is the first step in understanding deep learning.

---

## 🎯 Life Analogy: Vectors are GPS Navigation Directions

GPS says "Go 100 meters east, then 50 meters north" - that's a vector!

$$\text{Displacement} = \begin{bmatrix} 100 \text{m (East)} \\ 50 \text{m (North)} \end{bmatrix}$$

**The essence of vectors**: They tell you both "where to go" (direction) and "how far" (magnitude).

| Real-life Example | Vector Representation |
|-------------------|----------------------|
| Wind: Northwest at 20 km/h | $\begin{bmatrix} -14 \\ 14 \end{bmatrix}$ km/h |
| Stock: Up $5, Volume +1000 | $\begin{bmatrix} +5 \\ +1000 \end{bmatrix}$ |

### 📖 Plain English Translation

| Math Term | Plain English |
|-----------|---------------|
| Scalar | A single number (like temperature: 25°C) |
| Vector | A list of numbers with direction (like wind speed AND direction) |
| Matrix | A grid of numbers (like a spreadsheet) |
| Tensor | Multi-dimensional array (like a stack of spreadsheets) |

---

## Table of Contents

1. [Scalars](#scalars)
2. [Vectors](#vectors)
3. [Matrices](#matrices)
4. [Tensors](#tensors)
5. [Vector Operations](#vector-operations)
6. [Matrix Operations](#matrix-operations)
7. [Applications in Deep Learning](#applications-in-deep-learning)
8. [Summary](#summary)

---

## Scalars

### Definition

**Scalars** are single numerical values and are zero-dimensional tensors. They contain only one number and have no direction.

$$
s \in \mathbb{R} \quad \text{(real scalar)}
$$

$$
s \in \mathbb{C} \quad \text{(complex scalar)}
$$

In deep learning, we primarily use real scalars.

### Notation

Scalars are usually represented by lowercase letters:
- Ordinary scalars: $s, t, a, b, c, \ldots$
- Scalars with subscripts: $x_1, x_2, \ldots, x_n$
- Greek letters: $\alpha, \beta, \lambda, \eta, \ldots$

### Scalar Examples in Deep Learning

| Scalar | Meaning | Typical Values |
|--------|---------|----------------|
| $\eta$ | Learning rate | $0.001, 0.01, 0.1$ |
| $\lambda$ | Regularization coefficient | $0.001, 0.01, 0.1$ |
| $L$ | Loss value | Depends on task |
| $\epsilon$ | Numerical stability term | $10^{-8}, 10^{-7}$ |
| $\gamma, \beta$ | BatchNorm scale and shift | Learnable parameters |
| $p$ | Dropout probability | $0.1, 0.5$ |
| $\mu$ | Momentum coefficient | $0.9, 0.99$ |

### Python Implementation

```python
import numpy as np

# Scalars
learning_rate = 0.001  # Learning rate
lambda_reg = 0.01      # Regularization coefficient
loss = 0.5             # Loss value
epsilon = 1e-8         # Numerical stability term

# Scalars in NumPy (0-dimensional array)
scalar = np.array(5.0)
print(scalar.shape)    # () - zero dimensions
print(scalar.ndim)     # 0 - zero dimensions
print(scalar.item())   # 5.0 - Get Python scalar value

# Mathematical operations
a = 3.0
b = 2.0
print(a + b)   # Addition: 5.0
print(a - b)   # Subtraction: 1.0
print(a * b)   # Multiplication: 6.0
print(a / b)   # Division: 1.5
print(a ** b)  # Exponentiation: 9.0
print(np.sqrt(a))  # Square root: 1.732...
print(np.exp(a))   # Exponential: 20.085...
print(np.log(a))   # Natural logarithm: 1.098...
```

---

## Vectors

### Definition

**Vectors** are one-dimensional arrays, which are ordered collections of numbers. Vectors can represent direction and magnitude.

$$
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n
$$

Or written in row vector form (via transpose):

$$
\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top \in \mathbb{R}^n
$$

### Notation Conventions

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}$ | Vector (bold lowercase letter) |
| $\vec{x}$ | Vector (with arrow) |
| $\mathbf{x}^\top$ | Row vector (transpose of column vector) |
| $x_i$ | The $i$-th component of the vector |
| $\mathbf{x} \in \mathbb{R}^n$ | $n$-dimensional real vector |
| $\mathbf{0}$ | Zero vector |
| $\mathbf{1}$ | All-ones vector |

### Special Vectors

**Zero vector**:

$$
\mathbf{0} = \begin{bmatrix} 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix} \in \mathbb{R}^n
$$

**Unit vector** (length is 1):

$$
\mathbf{e}_i = \begin{bmatrix} 0 \\ \vdots \\ 1 \\ \vdots \\ 0 \end{bmatrix} \leftarrow \text{第 } i \text{ 个位置}
$$

**All-ones vector**:

$$
\mathbf{1} = \begin{bmatrix} 1 \\ 1 \\ \vdots \\ 1 \end{bmatrix} \in \mathbb{R}^n
$$

### Geometric Representation of Vectors

In two-dimensional and three-dimensional space, vectors can be represented as arrows starting from the origin:
- **Direction**: The direction the arrow points
- **Length (Magnitude)**: $\|\mathbf{x}\|_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}$

### Vector Examples in Deep Learning

| Vector | Dimension | Application |
|--------|-----------|-------------|
| $\mathbf{x} \in \mathbb{R}^{784}$ | 784 | MNIST image flattened (28×28) |
| $\mathbf{e} \in \mathbb{R}^{512}$ | 512 | Word embedding vector (Word2Vec) |
| $\mathbf{h} \in \mathbb{R}^{256}$ | 256 | RNN hidden state |
| $\mathbf{b} \in \mathbb{R}^{d_{out}}$ | $d_{out}$ | Fully connected layer bias |
| $\mathbf{g} \in \mathbb{R}^n$ | $n$ | Gradient vector |

### Python Implementation

```python
import numpy as np

# Create vectors
x = np.array([1, 2, 3, 4, 5])  # Shape (5,)
print(x.shape)   # (5,)
print(x.ndim)    # 1
print(len(x))    # 5

# Column vector (2D array, single column)
x_col = x.reshape(-1, 1)  # Shape (5, 1)
print(x_col)
# [[1]
#  [2]
#  [3]
#  [4]
#  [5]]

# Row vector (2D array, single row)
x_row = x.reshape(1, -1)  # Shape (1, 5)
print(x_row)
# [[1 2 3 4 5]]

# Access elements (0-indexed)
print(x[0])      # 1 (first element)
print(x[-1])     # 5 (last element)
print(x[1:4])    # [2, 3, 4] (slicing)

# Special vectors
zeros = np.zeros(5)           # Zero vector
ones = np.ones(5)             # All-ones vector
e_i = np.eye(5)[2]            # 3rd unit vector (index 2)
random_vec = np.random.randn(5)  # Random vector (standard normal)

# Vector length (L2 norm)
length = np.linalg.norm(x)
print(f"Vector length: {length}")  # sqrt(1+4+9+16+25) = sqrt(55) ≈ 7.416
```

---

## Matrices

### Definition

**Matrices** are two-dimensional arrays with rows and columns. They are the core objects of linear algebra.

$$
\mathbf{A} = \begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1n} \\
a_{21} & a_{22} & \cdots & a_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1} & a_{m2} & \cdots & a_{mn}
\end{bmatrix} \in \mathbb{R}^{m \times n}
$$

### Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{A}$ | Matrix (bold uppercase letter) |
| $\mathbf{A}_{m \times n}$ | Matrix with $m$ rows and $n$ columns |
| $\mathbf{A}_{i,:}$ or $\mathbf{A}[i, :]$ | The $i$-th row (row vector) |
| $\mathbf{A}_{:,j}$ or $\mathbf{A}[:, j]$ | The $j$-th column (column vector) |
| $A_{ij}$ or $a_{ij}$ | Element at row $i$, column $j$ |
| $\mathbf{A}^\top$ | Transposed matrix |
| $\mathbf{A}^{-1}$ | Inverse matrix |

### Special Matrices

**Identity matrix**:

$$
\mathbf{I}_n = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}, \quad \mathbf{I}\mathbf{A} = \mathbf{A}\mathbf{I} = \mathbf{A}
$$

**Zero matrix**:

$$
\mathbf{O}_{m \times n} = \begin{bmatrix}
0 & 0 & \cdots & 0 \\
0 & 0 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 0
\end{bmatrix}
$$

**Diagonal matrix**:

$$
\mathbf{D} = \text{diag}(d_1, d_2, \ldots, d_n) = \begin{bmatrix}
d_1 & 0 & \cdots & 0 \\
0 & d_2 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & d_n
\end{bmatrix}
$$

**Symmetric matrix**: $\mathbf{A} = \mathbf{A}^\top$

$$
\mathbf{A} = \begin{bmatrix} a & b \\ b & c \end{bmatrix}
$$

**Orthogonal matrix**: $\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$

$$
\mathbf{Q} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

### Matrix Examples in Deep Learning

| Matrix | Shape | Application |
|--------|-------|-------------|
| $\mathbf{I} \in \mathbb{R}^{H \times W}$ | $H \times W$ | Grayscale image |
| $\mathbf{W} \in \mathbb{R}^{d_{in} \times d_{out}}$ | $d_{in} \times d_{out}$ | Fully connected layer weights |
| $\mathbf{X} \in \mathbb{R}^{B \times D}$ | $B \times D$ | Batch data |
| $\mathbf{K} \in \mathbb{R}^{k_h \times k_w}$ | $k_h \times k_w$ | Convolution kernel |
| $\mathbf{H} \in \mathbb{R}^{n \times d}$ | $n \times d$ | Representations in attention mechanism |

### Python Implementation

```python
import numpy as np

# Create matrices
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape (2, 3)
print(A.shape)  # (2, 3)
print(A.ndim)   # 2

# Access elements
print(A[0, 1])    # 2 (row 1, column 2, 0-indexed)
print(A[1, 2])    # 6 (row 2, column 3)

# Access rows
print(A[0, :])    # [1, 2, 3] (row 1)
print(A[1])       # [4, 5, 6] (row 2, abbreviated notation)

# Access columns
print(A[:, 0])    # [1, 4] (column 1)
print(A[:, 1])    # [2, 5] (column 2)

# Submatrix
print(A[0:2, 1:3])  # [[2, 3], [5, 6]]

# Special matrices
I = np.eye(3)           # 3×3 identity matrix
O = np.zeros((2, 3))    # 2×3 zero matrix
D = np.diag([1, 2, 3])  # Diagonal matrix

# Random matrix
random_mat = np.random.randn(3, 4)  # Standard normal distribution

# Transpose
At = A.T
print(At.shape)  # (3, 2)
print(At)
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

---

## Tensors

### Definition

**Tensors** are generalizations of vectors and matrices, representing arrays of arbitrary dimensions.

$$
\mathcal{T} \in \mathbb{R}^{n_1 \times n_2 \times \cdots \times n_k} \quad \text{($k$-dimensional tensor)}
$$

Tensors are the fundamental representation of data in deep learning.

### Dimension Hierarchy

| Dimension | Name | Symbol | Deep Learning Example |
|-----------|------|--------|---------------------|
| 0 | Scalar | $s \in \mathbb{R}$ | Loss value, learning rate, single prediction |
| 1 | Vector | $\mathbf{x} \in \mathbb{R}^n$ | Bias, word vector, feature vector |
| 2 | Matrix | $\mathbf{A} \in \mathbb{R}^{m \times n}$ | Weight matrix, grayscale image, attention scores |
| 3 | 3D Tensor | $\mathcal{T} \in \mathbb{R}^{h \times w \times c}$ | RGB image, sequence data |
| 4 | 4D Tensor | $\mathcal{T} \in \mathbb{R}^{n \times c \times h \times w}$ | Image batch (NCHW format) |
| 5 | 5D Tensor | $\mathcal{T} \in \mathbb{R}^{n \times t \times c \times h \times w}$ | Video batch |

### Tensor Examples in Deep Learning

**Image data (4D tensor)**:

$$
\mathcal{X} \in \mathbb{R}^{N \times C \times H \times W}
$$

- $N$: Batch size
- $C$: Number of channels (1 for grayscale, 3 for RGB)
- $H$: Image height
- $W$: Image width

**Sequence data (3D tensor)**:

$$
\mathcal{X} \in \mathbb{R}^{N \times T \times D}
$$

- $N$: Batch size
- $T$: Sequence length
- $D$: Feature dimension

**Video data (5D tensor)**:

$$
\mathcal{X} \in \mathbb{R}^{N \times T \times C \times H \times W}
$$

- $N$: Batch size
- $T$: Number of frames
- $C, H, W$: Channels, height, width of each frame

### Python Implementation

```python
import numpy as np

# Tensors of different dimensions
scalar = np.array(5.0)                       # 0D scalar
vector = np.array([1, 2, 3])                 # 1D vector
matrix = np.array([[1, 2], [3, 4]])          # 2D matrix

# 3D tensor: RGB image
# Shape: (height, width, channels) - NHWC format
image_nhwc = np.random.randn(224, 224, 3)

# Shape: (channels, height, width) - NCHW format (PyTorch default)
image_nchw = np.random.randn(3, 224, 224)

# 4D tensor: Image batch
# NCHW format: (batch, channels, height, width)
batch_nchw = np.random.randn(32, 3, 224, 224)

# NHWC format: (batch, height, width, channels)
batch_nhwc = np.random.randn(32, 224, 224, 3)

# 5D tensor: Video batch
# Shape: (batch, frames, channels, height, width)
video_batch = np.random.randn(8, 16, 3, 224, 224)

# Print tensor information
tensors = [
    ("Scalar", scalar),
    ("Vector", vector),
    ("Matrix", matrix),
    ("3D Tensor (RGB Image)", image_nhwc),
    ("4D Tensor (Image Batch)", batch_nchw),
    ("5D Tensor (Video Batch)", video_batch)
]

for name, t in tensors:
    print(f"{name}: shape={t.shape}, ndim={t.ndim}")

# Output:
# Scalar: shape=(), ndim=0
# Vector: shape=(3,), ndim=1
# Matrix: shape=(2, 2), ndim=2
# 3D Tensor (RGB Image): shape=(224, 224, 3), ndim=3
# 4D Tensor (Image Batch): shape=(32, 3, 224, 224), ndim=4
# 5D Tensor (Video Batch): shape=(8, 16, 3, 224, 224), ndim=5

# Basic tensor operations
# Get total number of elements in tensor
print(f"Total elements in 4D tensor: {batch_nchw.size}")  # 32 * 3 * 224 * 224 = 4816896

# Reshape
batch_flattened = batch_nchw.reshape(32, -1)  # (32, 3*224*224)
print(f"Flattened shape: {batch_flattened.shape}")  # (32, 150528)

# Swap dimensions (NCHW -> NHWC)
batch_transposed = np.transpose(batch_nchw, (0, 2, 3, 1))
print(f"Transposed shape: {batch_transposed.shape}")  # (32, 224, 224, 3)
```

---

## Vector Operations

### Vector Addition

**Definition**: Element-wise addition of two vectors of the same dimension.

$$
\mathbf{c} = \mathbf{a} + \mathbf{b}, \quad c_i = a_i + b_i
$$

**Properties**:
- **Commutative law**: $\mathbf{a} + \mathbf{b} = \mathbf{b} + \mathbf{a}$
- **Associative law**: $(\mathbf{a} + \mathbf{b}) + \mathbf{c} = \mathbf{a} + (\mathbf{b} + \mathbf{c})$
- **Zero vector**: $\mathbf{a} + \mathbf{0} = \mathbf{a}$
- **Negative vector**: $\mathbf{a} + (-\mathbf{a}) = \mathbf{0}$

**Geometric meaning**: Parallelogram law or triangle law for vectors.

```python
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Vector addition
c = a + b  # [5, 7, 9]

# Verify properties
print(a + b)  # [5, 7, 9]
print(b + a)  # [5, 7, 9] - Commutative law

# Subtraction
d = a - b  # [-3, -3, -3]
```

### Scalar Multiplication

**Definition**: Element-wise multiplication of a scalar and a vector.

$$
\mathbf{y} = \alpha \mathbf{x}, \quad y_i = \alpha x_i
$$

**Properties**:
- Associative law: $\alpha(\beta \mathbf{x}) = (\alpha\beta)\mathbf{x}$
- Distributive law (scalar): $(\alpha + \beta)\mathbf{x} = \alpha\mathbf{x} + \beta\mathbf{x}$
- Distributive law (vector): $\alpha(\mathbf{x} + \mathbf{y}) = \alpha\mathbf{x} + \alpha\mathbf{y}$
- Identity element: $1 \cdot \mathbf{x} = \mathbf{x}$

**Geometric meaning**: Scale the length of the vector without changing direction ($\alpha > 0$) or reverse direction ($\alpha < 0$).

```python
x = np.array([1, 2, 3])
alpha = 2

# Scalar multiplication
y = alpha * x  # [2, 4, 6]

# Negative vector
neg_x = -x  # [-1, -2, -3]

# Scaling application: Normalize to unit length
norm = np.linalg.norm(x)
x_normalized = x / norm  # [0.267, 0.535, 0.802]
```

### Dot Product (Inner Product)

**Definition**: Sum of products of corresponding elements of two vectors.

$$
\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^\top \mathbf{b} = \sum_{i=1}^n a_i b_i
$$

**Geometric meaning**:

$$
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \|\mathbf{b}\| \cos\theta
$$

Where $\theta$ is the angle between the two vectors.

**Properties**:
- Commutative law: $\mathbf{a} \cdot \mathbf{b} = \mathbf{b} \cdot \mathbf{a}$
- Distributive law: $\mathbf{a} \cdot (\mathbf{b} + \mathbf{c}) = \mathbf{a} \cdot \mathbf{b} + \mathbf{a} \cdot \mathbf{c}$
- Positive definiteness: $\mathbf{a} \cdot \mathbf{a} \geq 0$, equality holds if and only if $\mathbf{a} = \mathbf{0}$
- $\alpha(\mathbf{a} \cdot \mathbf{b}) = (\alpha\mathbf{a}) \cdot \mathbf{b} = \mathbf{a} \cdot (\alpha\mathbf{b})$

**Important relationship**:

$$
\mathbf{a} \perp \mathbf{b} \iff \mathbf{a} \cdot \mathbf{b} = 0
$$

Vectors are orthogonal (perpendicular) if and only if their dot product is zero.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product
dot_ab = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
dot_ab = a @ b         # Equivalent notation in Python 3.5+
dot_ab = np.sum(a * b) # Another way

print(f"a · b = {dot_ab}")  # 32

# Verify geometric meaning
cos_theta = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"cos(θ) = {cos_theta:.4f}")  # 0.9746

# Orthogonal vectors
v1 = np.array([1, 0])
v2 = np.array([0, 1])
print(f"Dot product of orthogonal vectors: {np.dot(v1, v2)}")  # 0
```

### Cosine Similarity

**Definition**: Measure the similarity of the direction of two vectors.

$$
\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|} = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}
$$

**Range of values**: $[-1, 1]$

| $\cos\theta$ | Relationship |
|--------------|-------------|
| $1$ | Same direction (completely similar) |
| $0$ | Orthogonal (unrelated) |
| $-1$ | Opposite direction (completely opposite) |

**Applications**: Text similarity, recommendation systems, embedding vector retrieval.

```python
def cosine_similarity(a, b):
    """Compute cosine similarity"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.array([-1, -2, -3])
d = np.array([3, 0, 0])

print(f"Similarity between a and b: {cosine_similarity(a, b):.4f}")  # 0.9746 (very similar)
print(f"Similarity between a and c: {cosine_similarity(a, c):.4f}")  # -1.0000 (completely opposite)
print(f"Similarity between a and d: {cosine_similarity(a, d):.4f}")  # 0.2673 (weak similarity)
```

### Outer Product

**Definition**: Two vectors generate a matrix.

$$
\mathbf{A} = \mathbf{a} \otimes \mathbf{b} = \mathbf{a} \mathbf{b}^\top, \quad A_{ij} = a_i b_j
$$

For $\mathbf{a} \in \mathbb{R}^m, \mathbf{b} \in \mathbb{R}^n$, the outer product $\mathbf{A} \in \mathbb{R}^{m \times n}$.

**Properties**:
- $\mathbf{a} \otimes \mathbf{b} \neq \mathbf{b} \otimes \mathbf{a}$ (generally non-commutative)
- $(\mathbf{a} + \mathbf{b}) \otimes \mathbf{c} = \mathbf{a} \otimes \mathbf{c} + \mathbf{b} \otimes \mathbf{c}$
- $\text{rank}(\mathbf{a} \otimes \mathbf{b}) = 1$ (rank-1 matrix)

```python
a = np.array([1, 2, 3])  # (3,)
b = np.array([4, 5])     # (2,)

# Outer product
outer_ab = np.outer(a, b)
print(outer_ab)
# [[ 4,  5],   # 1 * [4, 5]
#  [ 8, 10],   # 2 * [4, 5]
#  [12, 15]]   # 3 * [4, 5]

# Equivalent notation
outer_ab_alt = a.reshape(-1, 1) @ b.reshape(1, -1)
print(np.allclose(outer_ab, outer_ab_alt))  # True

# Application: Construct attention weights from vectors
query = np.array([1, 0, 1])
key = np.array([0, 1, 0])
# Outer product used to build the foundation of attention matrix
```

### Hadamard Product (Element-wise Product)

**Definition**: Element-wise multiplication of two vectors of the same shape.

$$
\mathbf{c} = \mathbf{a} \odot \mathbf{b}, \quad c_i = a_i b_i
$$

**Properties**:
- Commutative law: $\mathbf{a} \odot \mathbf{b} = \mathbf{b} \odot \mathbf{a}$
- Associative law: $(\mathbf{a} \odot \mathbf{b}) \odot \mathbf{c} = \mathbf{a} \odot (\mathbf{b} \odot \mathbf{c})$
- Distributive law: $\mathbf{a} \odot (\mathbf{b} + \mathbf{c}) = \mathbf{a} \odot \mathbf{b} + \mathbf{a} \odot \mathbf{c}$

**Applications**: Gating mechanisms (LSTM, GRU), attention weight application.

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Hadamard product
hadamard = a * b  # [4, 10, 18]
print(f"a ⊙ b = {hadamard}")

# Note: Don't confuse with dot product
dot = np.dot(a, b)  # 32 (scalar)
print(f"a · b = {dot}")

# Application: Gating mechanism example
hidden = np.array([0.5, -0.3, 0.8])
gate = np.array([1.0, 0.0, 1.0])  # sigmoid output
gated_output = hidden * gate  # [0.5, 0.0, 0.8]
print(f"Gated output: {gated_output}")
```

### Vector Norms

**Definition**: Measure the "size" or "length" of a vector.

**$L_p$ norm**:

$$
\|\mathbf{x}\|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}, \quad p \geq 1
$$

**Commonly used norms**:

| Norm | Formula | Name | Application |
|------|---------|------|-------------|
| $L_1$ | $\|\mathbf{x}\|_1 = \sum_i \|x_i\|$ | Manhattan norm | Sparsity, Lasso |
| $L_2$ | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ | Euclidean norm | Regularization, normalization |
| $L_\infty$ | $\|\mathbf{x}\|_\infty = \max_i \|x_i\|$ | Maximum norm | Uniform convergence |
| $L_0$ | $\|\mathbf{x}\|_0 = \#\{i : x_i \neq 0\}$ | (Pseudo-norm) | Sparsity |

**Properties of norms**:
1. **Non-negativity**: $\|\mathbf{x}\| \geq 0$, equality holds if and only if $\mathbf{x} = \mathbf{0}$
2. **Homogeneity**: $\|\alpha\mathbf{x}\| = |\alpha|\|\mathbf{x}\|$
3. **Triangle inequality**: $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$

```python
x = np.array([3, 4])

# L2 norm (Euclidean norm)
l2 = np.linalg.norm(x)  # 5.0 (Pythagorean theorem: sqrt(9+16)=5)
l2 = np.linalg.norm(x, ord=2)  # Equivalent notation

# L1 norm
l1 = np.linalg.norm(x, ord=1)  # 7.0 (3+4)

# L∞ norm
linf = np.linalg.norm(x, ord=np.inf)  # 4.0 (max(3,4))

print(f"L1 norm: {l1}")     # 7.0
print(f"L2 norm: {l2}")     # 5.0
print(f"L∞ norm: {linf}")   # 4.0

# Normalize to unit vector
x_normalized = x / l2
print(f"Normalized vector: {x_normalized}")  # [0.6, 0.8]
print(f"Norm after normalization: {np.linalg.norm(x_normalized)}")  # 1.0
```

---

## Matrix Operations

### Matrix Addition

**Definition**: Element-wise addition of two matrices of the same shape.

$$
\mathbf{C} = \mathbf{A} + \mathbf{B}, \quad C_{ij} = A_{ij} + B_{ij}
$$

Requires $\mathbf{A}$ and $\mathbf{B}$ to have the same shape.

**Properties**:
- Commutative law: $\mathbf{A} + \mathbf{B} = \mathbf{B} + \mathbf{A}$
- Associative law: $(\mathbf{A} + \mathbf{B}) + \mathbf{C} = \mathbf{A} + (\mathbf{B} + \mathbf{C})$
- Zero matrix: $\mathbf{A} + \mathbf{O} = \mathbf{A}$
- Negative matrix: $\mathbf{A} + (-\mathbf{A}) = \mathbf{O}$

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix addition
C = A + B
# [[ 6,  8],
#  [10, 12]]

# Matrix subtraction
D = A - B
# [[-4, -4],
#  [-4, -4]]
```

### Scalar Multiplication

**Definition**: Element-wise multiplication of a scalar and a matrix.

$$
\mathbf{B} = \alpha \mathbf{A}, \quad B_{ij} = \alpha A_{ij}
$$

```python
A = np.array([[1, 2], [3, 4]])
alpha = 2

# Scalar multiplication
B = alpha * A
# [[2, 4],
#  [6, 8]]
```

### Matrix Multiplication

**Definition**: For $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$:

$$
\mathbf{C} = \mathbf{A}\mathbf{B} \in \mathbb{R}^{m \times n}, \quad C_{ij} = \sum_{l=1}^k A_{il} B_{lj}
$$

**Dimension requirement**: Number of columns in $\mathbf{A}$ must equal number of rows in $\mathbf{B}$.

$$
(m \times \underline{k}) \times (\underline{k} \times n) = (m \times n)
$$

**Important properties**:

| Property | Formula | Description |
|----------|---------|-------------|
| Non-commutative | $\mathbf{AB} \neq \mathbf{BA}$ | Generally |
| Associative law | $(\mathbf{AB})\mathbf{C} = \mathbf{A}(\mathbf{BC})$ | Always holds |
| Distributive law | $\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{AB} + \mathbf{AC}$ | Left distributive |
| Distributive law | $(\mathbf{A} + \mathbf{B})\mathbf{C} = \mathbf{AC} + \mathbf{BC}$ | Right distributive |
| Transpose | $(\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top$ | Order reversed |
| Inverse matrix | $(\mathbf{AB})^{-1} = \mathbf{B}^{-1} \mathbf{A}^{-1}$ | If invertible |

**Derivation of transpose property $(\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top$**:

Let $\mathbf{A} \in \mathbb{R}^{m \times k}$, $\mathbf{B} \in \mathbb{R}^{k \times n}$, then $\mathbf{AB} \in \mathbb{R}^{m \times n}$.

**Step 1**: Write the $(i, j)$-th element of $(\mathbf{AB})^\top$.

$(\mathbf{AB})_{ji} = \sum_{l=1}^k A_{jl} B_{li}$

Therefore:
$$[(\mathbf{AB})^\top]_{ij} = (\mathbf{AB})_{ji} = \sum_{l=1}^k A_{jl} B_{li}$$

**Step 2**: Write the $(i, j)$-th element of $\mathbf{B}^\top \mathbf{A}^\top$.

$$[\mathbf{B}^\top \mathbf{A}^\top]_{ij} = \sum_{l=1}^k (\mathbf{B}^\top)_{il} (\mathbf{A}^\top)_{lj} = \sum_{l=1}^k B_{li} A_{jl} = \sum_{l=1}^k A_{jl} B_{li}$$

**Step 3**: Compare the two equations.

$$[(\mathbf{AB})^\top]_{ij} = [\mathbf{B}^\top \mathbf{A}^\top]_{ij}$$

Since this holds for all $i, j$, we have:
$$\boxed{(\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top}$$

**Derivation of inverse matrix property $(\mathbf{AB})^{-1} = \mathbf{B}^{-1} \mathbf{A}^{-1}$**:

To prove $(\mathbf{AB})^{-1} = \mathbf{B}^{-1} \mathbf{A}^{-1}$, we only need to verify:

$$(\mathbf{AB})(\mathbf{B}^{-1} \mathbf{A}^{-1}) = \mathbf{A}(\mathbf{B}\mathbf{B}^{-1})\mathbf{A}^{-1} = \mathbf{A}\mathbf{I}\mathbf{A}^{-1} = \mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$$

$$(\mathbf{B}^{-1} \mathbf{A}^{-1})(\mathbf{AB}) = \mathbf{B}^{-1}(\mathbf{A}^{-1}\mathbf{A})\mathbf{B} = \mathbf{B}^{-1}\mathbf{I}\mathbf{B} = \mathbf{B}^{-1}\mathbf{B} = \mathbf{I}$$

Therefore, $\mathbf{B}^{-1} \mathbf{A}^{-1}$ is indeed the inverse matrix of $\mathbf{AB}$.

**Time complexity**:
- Naive algorithm: $O(mnk)$
- Strassen algorithm: $O(n^{2.807})$
- Theoretical optimum: $O(n^{2.373})$

```python
A = np.array([[1, 2],
              [3, 4]])  # (2, 2)
B = np.array([[5, 6],
              [7, 8]])  # (2, 2)

# Matrix multiplication
C = A @ B
# [[1*5+2*7, 1*6+2*8],    [[19, 22],
#  [3*5+4*7, 3*6+4*8]]  =  [43, 50]]

# Or using np.matmul
C = np.matmul(A, B)

# Note: * is element-wise multiplication, not matrix multiplication!
elementwise = A * B
# [[ 5, 12],
#  [21, 32]]

# Verify non-commutativity
AB = A @ B
BA = B @ A
print(f"AB = \n{AB}")
print(f"BA = \n{BA}")
print(f"AB == BA? {np.allclose(AB, BA)}")  # False

# Verify transpose property
print(f"(AB)^T = \n{(A @ B).T}")
print(f"B^T A^T = \n{B.T @ A.T}")
print(f"Equal? {np.allclose((A @ B).T, B.T @ A.T)}")  # True
```

### Transpose

**Definition**: Exchange rows and columns of a matrix.

$$
(\mathbf{A}^\top)_{ij} = A_{ji}
$$

**Properties**:
- $(\mathbf{A}^\top)^\top = \mathbf{A}$ (two transpositions restore the original matrix)
- $(\mathbf{A} + \mathbf{B})^\top = \mathbf{A}^\top + \mathbf{B}^\top$
- $(\alpha\mathbf{A})^\top = \alpha\mathbf{A}^\top$
- $(\mathbf{AB})^\top = \mathbf{B}^\top \mathbf{A}^\top$ (order reversed)

```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # (2, 3)

At = A.T  # (3, 2)
# [[1, 4],
#  [2, 5],
#  [3, 6]]

# Transpose of symmetric matrix equals itself
S = np.array([[1, 2],
              [2, 3]])
print(f"Symmetric matrix: S.T == S? {np.allclose(S.T, S)}")  # True

# Row vector transposed to column vector
row = np.array([[1, 2, 3]])  # (1, 3)
col = row.T  # (3, 1)
print(f"Row vector shape: {row.shape}")  # (1, 3)
print(f"Column vector shape: {col.shape}")  # (3, 1)
```

### Inverse Matrix

**Definition**: For a square matrix $\mathbf{A}$, if there exists a matrix $\mathbf{A}^{-1}$ such that:

$$
\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}
$$

Then $\mathbf{A}$ is said to be **invertible**, and $\mathbf{A}^{-1}$ is its inverse matrix.

**Invertibility conditions** (equivalent):
- $\det(\mathbf{A}) \neq 0$ (determinant is non-zero)
- $\mathbf{A}$ is full rank ($\text{rank}(\mathbf{A}) = n$)
- All eigenvalues of $\mathbf{A}$ are non-zero
- Row (column) vectors of $\mathbf{A}$ are linearly independent

**Inverse of 2×2 matrix**:

$$
\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \quad
\mathbf{A}^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

**Derivation of 2×2 matrix inverse formula**:

Let $\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}$, and assume its inverse is $\mathbf{X} = \begin{bmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \end{bmatrix}$.

**Step 1**: From the inverse matrix definition $\mathbf{AX} = \mathbf{I}$, we have:

$$\begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x_{11} & x_{12} \\ x_{21} & x_{22} \end{bmatrix} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$$

**Step 2**: Expand the first column (system of equations satisfied by $x_{11}, x_{21}$):

$$\begin{cases} ax_{11} + bx_{21} = 1 \\ cx_{11} + dx_{21} = 0 \end{cases}$$

From the second equation: $x_{21} = -\frac{c}{d} x_{11}$

Substitute into the first equation: $ax_{11} + b \cdot \left(-\frac{c}{d} x_{11}\right) = 1$

Solution: $x_{11}(ad - bc) = d$, so $x_{11} = \frac{d}{ad - bc}$

Then: $x_{21} = -\frac{c}{d} \cdot \frac{d}{ad-bc} = \frac{-c}{ad - bc}$

**Step 3**: Expand the second column (system of equations satisfied by $x_{12}, x_{22}$):

$$\begin{cases} ax_{12} + bx_{22} = 0 \\ cx_{12} + dx_{22} = 1 \end{cases}$$

From the first equation: $x_{12} = -\frac{b}{a} x_{22}$

Substitute into the second equation: $c \cdot \left(-\frac{b}{a} x_{22}\right) + dx_{22} = 1$

Solution: $x_{22}(ad - bc) = a$, so $x_{22} = \frac{a}{ad - bc}$

Then: $x_{12} = -\frac{b}{a} \cdot \frac{a}{ad-bc} = \frac{-b}{ad - bc}$

**Step 4**: Combine results:

$$\mathbf{A}^{-1} = \frac{1}{ad-bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

Where $\det(\mathbf{A}) = ad - bc$ is the determinant. When $\det(\mathbf{A}) \neq 0$, the inverse matrix exists.

**Properties**:
- $(\mathbf{A}^{-1})^{-1} = \mathbf{A}$
- $(\mathbf{A}^\top)^{-1} = (\mathbf{A}^{-1})^\top$
- $(\mathbf{AB})^{-1} = \mathbf{B}^{-1} \mathbf{A}^{-1}$
- $(\alpha\mathbf{A})^{-1} = \frac{1}{\alpha}\mathbf{A}^{-1}$
- $\det(\mathbf{A}^{-1}) = \frac{1}{\det(\mathbf{A})}$

```python
A = np.array([[1, 2],
              [3, 4]])

# Compute inverse matrix
A_inv = np.linalg.inv(A)
# [[-2. ,  1. ],
#  [ 1.5, -0.5]]

# Verify
print(f"A @ A_inv = \n{A @ A_inv}")
# [[1. 0.]
#  [0. 1.]] (Identity matrix, possible floating-point error)

print(f"A_inv @ A = \n{A_inv @ A}")
# [[1. 0.]
#  [0. 1.]]

# Analytical solution for 2×2 matrix
a, b, c, d = 1, 2, 3, 4
det = a * d - b * c  # -2
A_inv_manual = np.array([[d, -b], [-c, a]]) / det
print(f"Analytical solution: \n{A_inv_manual}")

# Singular matrix (non-invertible)
singular = np.array([[1, 2],
                     [2, 4]])  # det = 0
try:
    np.linalg.inv(singular)
except np.linalg.LinAlgError as e:
    print(f"Singular matrix is not invertible: {e}")
```

**Note**: In numerical computing, directly computing the inverse matrix is inefficient and unstable. Usually, matrix decomposition or solving linear systems is used instead.

### Hadamard Product (Element-wise Product)

**Definition**: Element-wise multiplication of two matrices of the same shape.

$$
\mathbf{C} = \mathbf{A} \odot \mathbf{B}, \quad C_{ij} = A_{ij} B_{ij}
$$

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Hadamard product (element-wise multiplication)
C = A * B
# [[ 5, 12],
#  [21, 32]]

# Compare with matrix multiplication
matmul = A @ B
# [[19, 22],
#  [43, 50]]
```

### Matrix Trace

**Definition**: Sum of the diagonal elements of a square matrix.

$$
\text{tr}(\mathbf{A}) = \sum_{i=1}^n A_{ii}
$$

**Properties**:
- $\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})$
- $\text{tr}(\alpha\mathbf{A}) = \alpha \cdot \text{tr}(\mathbf{A})$
- $\text{tr}(\mathbf{A}^\top) = \text{tr}(\mathbf{A})$
- $\text{tr}(\mathbf{AB}) = \text{tr}(\mathbf{BA})$ (cyclic property)
- $\text{tr}(\mathbf{ABC}) = \text{tr}(\mathbf{BCA}) = \text{tr}(\mathbf{CAB})$

**Relationship with Frobenius norm**:

$$
\|\mathbf{A}\|_F^2 = \text{tr}(\mathbf{A}^\top \mathbf{A})
$$

**Derivation of relationship between Frobenius norm and trace**:

Let $\mathbf{A} \in \mathbb{R}^{m \times n}$, define Frobenius norm as the square root of the sum of squares of all elements.

**Step 1**: Directly compute $\|\mathbf{A}\|_F^2$:

$$\|\mathbf{A}\|_F^2 = \sum_{i=1}^m \sum_{j=1}^n A_{ij}^2$$

**Step 2**: Compute $\mathbf{A}^\top \mathbf{A}$:

$(\mathbf{A}^\top \mathbf{A})_{jk} = \sum_{i=1}^m (\mathbf{A}^\top)_{ji} A_{ik} = \sum_{i=1}^m A_{ij} A_{ik}$

**Step 3**: Compute $\text{tr}(\mathbf{A}^\top \mathbf{A})$ (sum of diagonal elements):

$$\text{tr}(\mathbf{A}^\top \mathbf{A}) = \sum_{j=1}^n (\mathbf{A}^\top \mathbf{A})_{jj} = \sum_{j=1}^n \sum_{i=1}^m A_{ij}^2$$

**Step 4**: Compare the two equations:

$$\|\mathbf{A}\|_F^2 = \sum_{i=1}^m \sum_{j=1}^n A_{ij}^2 = \text{tr}(\mathbf{A}^\top \mathbf{A})$$

$$\boxed{\|\mathbf{A}\|_F^2 = \text{tr}(\mathbf{A}^\top \mathbf{A})}$$

This relationship is very useful for computing regularization losses and gradient norms.

```python
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Trace
trace = np.trace(A)  # 1 + 5 + 9 = 15
print(f"tr(A) = {trace}")

# Verify cyclic property
B = np.array([[1, 2], [3, 4]])
C = np.array([[5, 6], [7, 8]])
print(f"tr(BC) = {np.trace(B @ C)}")    # 39
print(f"tr(CB) = {np.trace(C @ B)}")    # 39
```

---

## Applications in Deep Learning

### Fully Connected Layer

Fully connected layers (linear layers) are essentially matrix multiplication and vector addition:

$$
\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}
$$

Where:
- $\mathbf{X} \in \mathbb{R}^{B \times D_{in}}$: Input batch
- $\mathbf{W} \in \mathbb{R}^{D_{in} \times D_{out}}$: Weight matrix
- $\mathbf{b} \in \mathbb{R}^{D_{out}}$: Bias vector (broadcast to each row)
- $\mathbf{Y} \in \mathbb{R}^{B \times D_{out}}$: Output

```python
import numpy as np

def linear_forward(X, W, b):
    """Fully connected layer forward propagation"""
    return X @ W + b  # b is added to each row via broadcasting

# Parameters
batch_size = 32
input_dim = 784
output_dim = 128

X = np.random.randn(batch_size, input_dim)   # (32, 784)
W = np.random.randn(input_dim, output_dim)   # (784, 128)
b = np.random.randn(output_dim)              # (128,)

Y = linear_forward(X, W, b)
print(f"Output shape: {Y.shape}")  # (32, 128)
```

### Embedding Layer

Embedding layer maps discrete indices to continuous vectors:

$$
\mathbf{E} \in \mathbb{R}^{V \times D}, \quad \text{embedding}(i) = \mathbf{E}[i, :]
$$

```python
def embedding_lookup(E, indices):
    """Embedding lookup"""
    return E[indices]

vocab_size = 10000
embed_dim = 512
E = np.random.randn(vocab_size, embed_dim)  # Embedding matrix

# Lookup
indices = np.array([0, 42, 100])  # Word indices
embedded = embedding_lookup(E, indices)
print(f"Embedding shape: {embedded.shape}")  # (3, 512)
```

### Batch Normalization

Batch normalization involves vector operations:

$$
\hat{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\sqrt{\boldsymbol{\sigma}^2 + \epsilon}}
$$

$$
\mathbf{y} = \gamma \odot \hat{\mathbf{x}} + \beta
$$

```python
def batch_norm(X, gamma, beta, eps=1e-5):
    """
    X: (batch, features)
    gamma, beta: (features,)
    """
    mu = np.mean(X, axis=0)           # Mean
    sigma2 = np.var(X, axis=0)        # Variance
    x_hat = (X - mu) / np.sqrt(sigma2 + eps)  # Standardization
    y = gamma * x_hat + beta          # Scale and shift
    return y

X = np.random.randn(32, 128)
gamma = np.ones(128)
beta = np.zeros(128)
Y = batch_norm(X, gamma, beta)
```

### Attention Mechanism

Attention scores are computed via dot product:

$$
\text{score}(\mathbf{q}, \mathbf{k}) = \frac{\mathbf{q}^\top \mathbf{k}}{\sqrt{d_k}}
$$

Batch form of attention:

$$
\mathbf{A} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)
$$

$$
\mathbf{O} = \mathbf{A}\mathbf{V}
$$

```python
def softmax(X, axis=-1):
    """Numerically stable softmax"""
    X_max = np.max(X, axis=axis, keepdims=True)
    exp_X = np.exp(X - X_max)
    return exp_X / np.sum(exp_X, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Q: (batch, seq_len, d_k)
    K: (batch, seq_len, d_k)
    V: (batch, seq_len, d_v)
    """
    d_k = Q.shape[-1]
    # Compute attention scores: Q @ K^T
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_k)
    # Softmax normalization
    attention_weights = softmax(scores, axis=-1)
    # Weighted sum
    output = attention_weights @ V
    return output, attention_weights

# Example
batch, seq_len, d_k = 2, 10, 64
Q = np.random.randn(batch, seq_len, d_k)
K = np.random.randn(batch, seq_len, d_k)
V = np.random.randn(batch, seq_len, d_k)

output, weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")      # (2, 10, 64)
print(f"Attention weights shape: {weights.shape}")  # (2, 10, 10)
```

### Gradient Computation

Matrix derivatives in backpropagation:

$$
\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top \frac{\partial L}{\partial \mathbf{Y}}
$$

$$
\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \mathbf{W}^\top
$$

**Derivation of linear layer gradient formulas**:

Let the linear layer be $\mathbf{Y} = \mathbf{X}\mathbf{W} + \mathbf{b}$, where $\mathbf{X} \in \mathbb{R}^{B \times D_{in}}$, $\mathbf{W} \in \mathbb{R}^{D_{in} \times D_{out}}$, $\mathbf{Y} \in \mathbb{R}^{B \times D_{out}}$.

**Deriving $\frac{\partial L}{\partial \mathbf{W}}$**:

**Step 1**: Use the chain rule. Let $\frac{\partial L}{\partial Y_{ij}}$ be known (from subsequent layers), need to find $\frac{\partial L}{\partial W_{kl}}$.

$$\frac{\partial L}{\partial W_{kl}} = \sum_{i,j} \frac{\partial L}{\partial Y_{ij}} \cdot \frac{\partial Y_{ij}}{\partial W_{kl}}$$

**Step 2**: Compute $\frac{\partial Y_{ij}}{\partial W_{kl}}$.

Since $Y_{ij} = \sum_{m=1}^{D_{in}} X_{im} W_{mj} + b_j$, we have:

$$\frac{\partial Y_{ij}}{\partial W_{kl}} = X_{ik} \cdot \delta_{jl} = \begin{cases} X_{ik} & \text{if } j = l \\ 0 & \text{otherwise} \end{cases}$$

**Step 3**: Substitute into the sum:

$$\frac{\partial L}{\partial W_{kl}} = \sum_{i=1}^{B} \frac{\partial L}{\partial Y_{il}} \cdot X_{ik} = \sum_{i=1}^{B} X_{ik} \cdot \frac{\partial L}{\partial Y_{il}}$$

**Step 4**: Write in matrix form:

$$\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top \frac{\partial L}{\partial \mathbf{Y}}$$

**Deriving $\frac{\partial L}{\partial \mathbf{X}}$**:

**Step 1**: Similarly, $\frac{\partial L}{\partial X_{kl}} = \sum_{i,j} \frac{\partial L}{\partial Y_{ij}} \cdot \frac{\partial Y_{ij}}{\partial X_{kl}}$

**Step 2**: $\frac{\partial Y_{ij}}{\partial X_{kl}} = W_{lj} \cdot \delta_{ik} = \begin{cases} W_{lj} & \text{if } i = k \\ 0 & \text{otherwise} \end{cases}$

**Step 3**: Substitute into the sum:

$$\frac{\partial L}{\partial X_{kl}} = \sum_{j=1}^{D_{out}} \frac{\partial L}{\partial Y_{kj}} \cdot W_{lj}$$

**Step 4**: Write in matrix form:

$$\frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \mathbf{W}^\top$$

$$\boxed{\frac{\partial L}{\partial \mathbf{W}} = \mathbf{X}^\top \frac{\partial L}{\partial \mathbf{Y}}, \quad \frac{\partial L}{\partial \mathbf{X}} = \frac{\partial L}{\partial \mathbf{Y}} \mathbf{W}^\top}$$

```python
def linear_backward(dL_dY, X, W):
    """Fully connected layer backpropagation"""
    # Gradient w.r.t. W
    dL_dW = X.T @ dL_dY  # (D_in, D_out)
    # Gradient w.r.t. X
    dL_dX = dL_dY @ W.T  # (B, D_in)
    # Gradient w.r.t. b
    dL_db = np.sum(dL_dY, axis=0)  # (D_out,)
    return dL_dX, dL_dW, dL_db

# Backpropagation example
dL_dY = np.random.randn(32, 128)
dL_dX, dL_dW, dL_db = linear_backward(dL_dY, X, W)
print(f"dL/dX shape: {dL_dX.shape}")  # (32, 784)
print(f"dL/dW shape: {dL_dW.shape}")  # (784, 128)
print(f"dL/db shape: {dL_db.shape}")  # (128,)
```

---

## Summary

This chapter introduced the basics of vectors and matrices:

| Concept | Definition | Application in Deep Learning |
|---------|-----------|----------------------------|
| Scalar | Single numerical value, 0D tensor | Learning rate, loss value, regularization coefficient |
| Vector | One-dimensional array | Word embedding, bias, hidden state |
| Matrix | Two-dimensional array | Weights, image, attention scores |
| Tensor | Multi-dimensional array | Image batch, video data, sequence data |
| Dot Product | Sum of products of corresponding vector elements | Attention scores, similarity computation |
| Matrix Multiplication | New matrix formed by row-column dot products | Fully connected layer, linear transformation |
| Transpose | Exchange rows and columns | Gradient computation, dimension matching |
| Inverse Matrix | Satisfies $\mathbf{AA}^{-1} = \mathbf{I}$ | Solving linear systems |

### Key Formula Summary

| Operation | Formula |
|-----------|---------|
| Vector Dot Product | $\mathbf{a} \cdot \mathbf{b} = \sum_i a_i b_i$ |
| Cosine Similarity | $\cos\theta = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$ |
| Matrix Multiplication | $(\mathbf{AB})_{ij} = \sum_k A_{ik} B_{kj}$ |
| Transpose | $(\mathbf{A}^\top)_{ij} = A_{ji}$ |
| $L_2$ Norm | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ |
| Fully Connected Layer | $\mathbf{Y} = \mathbf{XW} + \mathbf{b}$ |

---

**Next section**: [Chapter 1(b): Linear Systems and Matrix Properties](01b-linear-systems-matrix-properties_EN.md) - Learn about solving linear systems, determinants, and matrix rank.

**Back**: [Math Fundamentals Tutorial Table of Contents](../math-fundamentals.md)
