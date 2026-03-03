# Chapter 1: Tensor Basics

## Data is the Blood of Deep Learning...

What is an image? An array of 224×224×3 numbers.

What is text? A sequence of word vectors.

What is sound? A waveform of amplitude over time.

Neural networks cannot see images, hear sounds, or read text. The only thing they can understand is **numbers**. And Tensor is the container that carries these numbers.

Imagine a Rubik's cube—length, width, height, three dimensions. Now, extend this cube infinitely: four dimensions, five dimensions... even a hundred dimensions. This is Tensor.

```
0D: A drop of water    → Scalar (5)
1D: A string of beads  → Vector ([1,2,3])
2D: A chess board      → Matrix ([[1,2],[3,4]])
3D: A photo album      → Tensor ([[[...]]])
4D: A video            → Batch×Time×Height×Width×Channel
```

**Tensor is the myriad forms of numbers in computers.**

---

## 1.1 Why Do We Need Tensor?

**Question**: Doesn't NumPy already have ndarray?

**Answer**: Yes, but Tensor has two key capabilities:

| Capability | NumPy ndarray | Tensor |
|------------|--------------|--------|
| GPU Acceleration | ❌ | ✅ |
| Automatic Differentiation | ❌ | ✅ |

```
Regular computation:       Deep learning computation:
[1,2,3]                    [1,2,3]
  +                          +
[4,5,6]                    [4,5,6]
  =                          =
[5,7,9]                    [5,7,9] → Can remember: this result came from addition!
                             ↓
                       Can automatically compute gradients during backward()
```

---

## 1.2 Simplest Tensor: Starting from Zero

### Goal
Create a Tensor class that can store data.

### Implementation

A Tensor is essentially a **multi-dimensional array**, similar to NumPy's ndarray.

| Dimensions | Name | Example |
|------------|------|---------|
| 0D | Scalar | `5` |
| 1D | Vector | `[1, 2, 3]` |
| 2D | Matrix | `[[1, 2], [3, 4]]` |
| 3D+ | Tensor | Image `[B, C, H, W]` |

## 1.2 Simplest Tensor Implementation

Let's start with the most basic version:

```python
# tensor_v1.py
import numpy as np
from typing import List, Union, Optional

class Tensor:
    """Simplest Tensor implementation"""
    
    def __init__(self, data, requires_grad: bool = False):
        # Convert input to NumPy array
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        
        self.requires_grad = requires_grad
        self.grad = None  # Gradient storage
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


# Test
t = Tensor([1, 2, 3, 4, 5])
print(t)  # Tensor(shape=(5,), requires_grad=False)

m = Tensor([[1, 2], [3, 4]])
print(m)  # Tensor(shape=(2, 2), requires_grad=False)
```

## 1.3 Adding Basic Operations

Deep learning requires many mathematical operations. Let's implement the basics first:

```python
# tensor_v2.py
class Tensor:
    def __init__(self, data, requires_grad: bool = False):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
    
    # ... previous code omitted ...
    
    def __add__(self, other):
        """Addition: t1 + t2"""
        if isinstance(other, (int, float)):
            other_data = other
        else:
            other_data = other.data
        
        result = Tensor(
            self.data + other_data,
            requires_grad=self.requires_grad
        )
        return result
    
    def __mul__(self, other):
        """Multiplication: t1 * t2 (element-wise)"""
        if isinstance(other, (int, float)):
            other_data = other
        else:
            other_data = other.data
        
        result = Tensor(
            self.data * other_data,
            requires_grad=self.requires_grad
        )
        return result
    
    def __neg__(self):
        """Negation: -t"""
        return Tensor(-self.data, requires_grad=self.requires_grad)
    
    def __sub__(self, other):
        """Subtraction: t1 - t2"""
        return self + (-other if isinstance(other, Tensor) else -other)
    
    def __truediv__(self, other):
        """Division: t1 / t2"""
        if isinstance(other, (int, float)):
            return Tensor(self.data / other, requires_grad=self.requires_grad)
        return Tensor(self.data / other.data, requires_grad=self.requires_grad)
    
    def __pow__(self, n):
        """Power: t ** n"""
        return Tensor(self.data ** n, requires_grad=self.requires_grad)


# Test
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

print(a + b)      # [5, 7, 9]
print(a * b)      # [4, 10, 18]
print(a * 2)      # [2, 4, 6]
print(a ** 2)     # [1, 4, 9]
```

## 1.4 Matrix Operations

Matrix multiplication is core to deep learning:

```python
def matmul(self, other):
    """Matrix multiplication: t1 @ t2"""
    if not isinstance(other, Tensor):
        other = Tensor(other)
    
    if self.ndim != 2 or other.ndim != 2:
        raise ValueError("matmul requires 2D tensors")
    
    if self.shape[1] != other.shape[0]:
        raise ValueError(f"Shape mismatch: {self.shape} vs {other.shape}")
    
    result = np.matmul(self.data, other.data)
    return Tensor(result, requires_grad=self.requires_grad)

def __matmul__(self, other):
    return self.matmul(other)


# Test
A = Tensor([[1, 2], [3, 4]])  # 2x2
B = Tensor([[5, 6], [7, 8]])  # 2x2
C = A @ B
print(C.data)
# [[19, 22],
#  [43, 50]]
```

### Matrix Multiplication Intuition

```
A (2x2) @ B (2x2) = C (2x2)

C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 19
C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*6 + 2*8 = 22
...
```

## 1.5 Shape Operations

Deep learning often requires changing tensor shapes:

```python
def reshape(self, new_shape):
    """Change shape"""
    if np.prod(new_shape) != np.prod(self.shape):
        raise ValueError(f"Cannot reshape {self.shape} to {new_shape}")
    
    return Tensor(
        self.data.reshape(new_shape),
        requires_grad=self.requires_grad
    )

def transpose(self, dim0=0, dim1=1):
    """Transpose two dimensions"""
    axes = list(range(self.ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return Tensor(
        np.transpose(self.data, axes),
        requires_grad=self.requires_grad
    )

def flatten(self, start_dim=0):
    """Flatten"""
    new_shape = (-1,) + self.shape[start_dim+1:]
    return self.reshape(new_shape)

def squeeze(self, dim=None):
    """Remove dimensions of size 1"""
    return Tensor(
        np.squeeze(self.data, axis=dim),
        requires_grad=self.requires_grad
    )

def unsqueeze(self, dim):
    """Add a dimension"""
    return Tensor(
        np.expand_dims(self.data, axis=dim),
        requires_grad=self.requires_grad
    )


# Test
t = Tensor([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
print(t.reshape((3, 2)).shape)       # (3, 2)
print(t.reshape((-1,)).shape)        # (6,) - flattened
print(t.transpose().shape)           # (3, 2)
print(t.flatten().shape)             # (6,)
```

## 1.6 Reduction Operations

Sum, mean, etc.:

```python
def sum(self, dim=None, keepdims=False):
    """Sum"""
    result = np.sum(self.data, axis=dim, keepdims=keepdims)
    return Tensor(result, requires_grad=self.requires_grad)

def mean(self, dim=None, keepdims=False):
    """Mean"""
    result = np.mean(self.data, axis=dim, keepdims=keepdims)
    return Tensor(result, requires_grad=self.requires_grad)

def max(self, dim=None, keepdims=False):
    """Maximum"""
    result = np.max(self.data, axis=dim, keepdims=keepdims)
    return Tensor(result, requires_grad=self.requires_grad)

def min(self, dim=None, keepdims=False):
    """Minimum"""
    result = np.min(self.data, axis=dim, keepdims=keepdims)
    return Tensor(result, requires_grad=self.requires_grad)


# Test
t = Tensor([[1, 2, 3], [4, 5, 6]])
print(t.sum())           # 21
print(t.sum(dim=0))      # [5, 7, 9] - sum by column
print(t.sum(dim=1))      # [6, 15] - sum by row
print(t.mean())          # 3.5
```

### Understanding Reduction Dimensions

```
Original data: [[1, 2, 3],
              [4, 5, 6]]
shape: (2, 3)

sum(dim=0) - compress dimension 0 (rows), result (3,):
[1+4, 2+5, 3+6] = [5, 7, 9]

sum(dim=1) - compress dimension 1 (columns), result (2,):
[1+2+3, 4+5+6] = [6, 15]
```

## 1.7 Broadcasting Mechanism

Operations between tensors of different shapes require broadcasting:

```python
def __add__(self, other):
    if isinstance(other, (int, float)):
        other_data = np.array(other)
    elif isinstance(other, Tensor):
        other_data = other.data
    else:
        other_data = other
    
    # NumPy automatically handles broadcasting
    result = self.data + other_data
    return Tensor(result, requires_grad=self.requires_grad)


# Test broadcasting
A = Tensor([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
b = Tensor([10, 20, 30])             # shape (3,)

# b broadcasts to [[10, 20, 30], [10, 20, 30]]
C = A + b
print(C.data)
# [[11, 22, 33],
#  [14, 25, 36]]
```

### Broadcasting Rules

1. Align dimensions from right to left
2. Dimensions must be equal, or one is 1, or doesn't exist
3. Dimensions of size 1 are expanded by copying

```
(2, 3) + (3,)     -> (2, 3) + (1, 3) -> (2, 3)
(2, 3, 4) + (4,)  -> (2, 3, 4) + (1, 1, 4) -> (2, 3, 4)
(2, 3) + (2, 1)   -> (2, 3) + (2, 3) -> (2, 3)
(2, 3) + (2,)     -> Error! Cannot broadcast
```

## 1.8 Factory Methods

Convenient creation of special tensors:

```python
@classmethod
def zeros(cls, shape, requires_grad=False):
    """Create all-zeros tensor"""
    return cls(np.zeros(shape), requires_grad=requires_grad)

@classmethod
def ones(cls, shape, requires_grad=False):
    """Create all-ones tensor"""
    return cls(np.ones(shape), requires_grad=requires_grad)

@classmethod
def randn(cls, shape, requires_grad=False):
    """Create standard normal distribution tensor"""
    return cls(np.random.randn(*shape), requires_grad=requires_grad)

@classmethod
def rand(cls, shape, requires_grad=False):
    """Create uniform distribution tensor [0, 1)"""
    return cls(np.random.rand(*shape), requires_grad=requires_grad)


# Test
z = Tensor.zeros((2, 3))
o = Tensor.ones((3, 3))
r = Tensor.randn((4, 4))  # Random initialization, commonly used for weights
```

## 1.9 Complete Code

```python
# tensor.py - First complete implementation
import numpy as np
from typing import List, Tuple, Union, Optional

class Tensor:
    def __init__(self, data, requires_grad: bool = False):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        
        self.requires_grad = requires_grad
        self.grad = None
    
    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    @property
    def T(self) -> "Tensor":
        return self.transpose()
    
    def item(self) -> float:
        return self.data.item()
    
    # Operations
    def __add__(self, other): ...
    def __radd__(self, other): return self + other
    def __mul__(self, other): ...
    def __rmul__(self, other): return self * other
    def __neg__(self): ...
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return (-self) + other
    def __truediv__(self, other): ...
    def __pow__(self, n): ...
    def __matmul__(self, other): return self.matmul(other)
    
    # Matrix
    def matmul(self, other): ...
    
    # Shape
    def reshape(self, shape): ...
    def transpose(self, dim0=0, dim1=1): ...
    def flatten(self): ...
    
    # Reduction
    def sum(self, dim=None): ...
    def mean(self, dim=None): ...
    
    # Factory methods
    @classmethod
    def zeros(cls, shape): ...
    @classmethod
    def ones(cls, shape): ...
    @classmethod
    def randn(cls, shape): ...
```

## 1.10 Exercises

1. **Implement `sqrt()` method**: Compute element-wise square root

2. **Implement `abs()` method**: Compute absolute value

3. **Implement `clip(min_val, max_val)` method**: Limit values within range

4. **Implement `concatenate(tensors, dim)` function**: Concatenate tensors along specified dimension

5. **Challenge**: Implement `conv2d(x, weight, stride, padding)` (Hint: use im2col)

## Next Chapter

Now we have a Tensor class capable of basic operations. In the next chapter, we will implement **automatic differentiation**, enabling Tensors to automatically compute gradients!

→ [Chapter 2: Automatic Differentiation](02-autograd_EN.md)
