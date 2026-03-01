# 第一章：Tensor 基础

Tensor（张量）是深度学习框架的核心数据结构。本章我们将从零实现一个 Tensor 类。

## 1.1 什么是 Tensor？

Tensor 本质上是一个**多维数组**，类似于 NumPy 的 ndarray。

| 维度 | 名称 | 示例 |
|------|------|------|
| 0D | 标量 | `5` |
| 1D | 向量 | `[1, 2, 3]` |
| 2D | 矩阵 | `[[1, 2], [3, 4]]` |
| 3D+ | 张量 | 图像 `[B, C, H, W]` |

## 1.2 最简单的 Tensor 实现

让我们从最基础的版本开始：

```python
# tensor_v1.py
import numpy as np
from typing import List, Union, Optional

class Tensor:
    """最简单的 Tensor 实现"""
    
    def __init__(self, data, requires_grad: bool = False):
        # 将输入转换为 NumPy 数组
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)
        
        self.requires_grad = requires_grad
        self.grad = None  # 梯度存储
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


# 测试
t = Tensor([1, 2, 3, 4, 5])
print(t)  # Tensor(shape=(5,), requires_grad=False)

m = Tensor([[1, 2], [3, 4]])
print(m)  # Tensor(shape=(2, 2), requires_grad=False)
```

## 1.3 添加基本运算

深度学习需要大量的数学运算，我们先实现最基础的：

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
    
    # ... 省略前面的代码 ...
    
    def __add__(self, other):
        """加法：t1 + t2"""
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
        """乘法：t1 * t2（逐元素）"""
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
        """取负：-t"""
        return Tensor(-self.data, requires_grad=self.requires_grad)
    
    def __sub__(self, other):
        """减法：t1 - t2"""
        return self + (-other if isinstance(other, Tensor) else -other)
    
    def __truediv__(self, other):
        """除法：t1 / t2"""
        if isinstance(other, (int, float)):
            return Tensor(self.data / other, requires_grad=self.requires_grad)
        return Tensor(self.data / other.data, requires_grad=self.requires_grad)
    
    def __pow__(self, n):
        """幂运算：t ** n"""
        return Tensor(self.data ** n, requires_grad=self.requires_grad)


# 测试
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

print(a + b)      # [5, 7, 9]
print(a * b)      # [4, 10, 18]
print(a * 2)      # [2, 4, 6]
print(a ** 2)     # [1, 4, 9]
```

## 1.4 矩阵运算

矩阵乘法是深度学习的核心：

```python
def matmul(self, other):
    """矩阵乘法：t1 @ t2"""
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


# 测试
A = Tensor([[1, 2], [3, 4]])  # 2x2
B = Tensor([[5, 6], [7, 8]])  # 2x2
C = A @ B
print(C.data)
# [[19, 22],
#  [43, 50]]
```

### 矩阵乘法的直觉

```
A (2x2) @ B (2x2) = C (2x2)

C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*5 + 2*7 = 19
C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*6 + 2*8 = 22
...
```

## 1.5 形状操作

深度学习中经常需要改变张量形状：

```python
def reshape(self, new_shape):
    """改变形状"""
    if np.prod(new_shape) != np.prod(self.shape):
        raise ValueError(f"Cannot reshape {self.shape} to {new_shape}")
    
    return Tensor(
        self.data.reshape(new_shape),
        requires_grad=self.requires_grad
    )

def transpose(self, dim0=0, dim1=1):
    """转置两个维度"""
    axes = list(range(self.ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return Tensor(
        np.transpose(self.data, axes),
        requires_grad=self.requires_grad
    )

def flatten(self, start_dim=0):
    """展平"""
    new_shape = (-1,) + self.shape[start_dim+1:]
    return self.reshape(new_shape)

def squeeze(self, dim=None):
    """去除大小为1的维度"""
    return Tensor(
        np.squeeze(self.data, axis=dim),
        requires_grad=self.requires_grad
    )

def unsqueeze(self, dim):
    """增加一个维度"""
    return Tensor(
        np.expand_dims(self.data, axis=dim),
        requires_grad=self.requires_grad
    )


# 测试
t = Tensor([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
print(t.reshape((3, 2)).shape)       # (3, 2)
print(t.reshape((-1,)).shape)        # (6,) - 展平
print(t.transpose().shape)           # (3, 2)
print(t.flatten().shape)             # (6,)
```

## 1.6 归约操作

求和、求均值等：

```python
def sum(self, dim=None, keepdims=False):
    """求和"""
    result = np.sum(self.data, axis=dim, keepdims=keepdims)
    return Tensor(result, requires_grad=self.requires_grad)

def mean(self, dim=None, keepdims=False):
    """求均值"""
    result = np.mean(self.data, axis=dim, keepdims=keepdims)
    return Tensor(result, requires_grad=self.requires_grad)

def max(self, dim=None, keepdims=False):
    """最大值"""
    result = np.max(self.data, axis=dim, keepdims=keepdims)
    return Tensor(result, requires_grad=self.requires_grad)

def min(self, dim=None, keepdims=False):
    """最小值"""
    result = np.min(self.data, axis=dim, keepdims=keepdims)
    return Tensor(result, requires_grad=self.requires_grad)


# 测试
t = Tensor([[1, 2, 3], [4, 5, 6]])
print(t.sum())           # 21
print(t.sum(dim=0))      # [5, 7, 9] - 按列求和
print(t.sum(dim=1))      # [6, 15] - 按行求和
print(t.mean())          # 3.5
```

### 归约操作的维度理解

```
原始数据: [[1, 2, 3],
          [4, 5, 6]]
shape: (2, 3)

sum(dim=0) - 压缩第0维（行），结果 (3,):
[1+4, 2+5, 3+6] = [5, 7, 9]

sum(dim=1) - 压缩第1维（列），结果 (2,):
[1+2+3, 4+5+6] = [6, 15]
```

## 1.7 广播机制

不同形状的张量运算时需要广播：

```python
def __add__(self, other):
    if isinstance(other, (int, float)):
        other_data = np.array(other)
    elif isinstance(other, Tensor):
        other_data = other.data
    else:
        other_data = other
    
    # NumPy 自动处理广播
    result = self.data + other_data
    return Tensor(result, requires_grad=self.requires_grad)


# 测试广播
A = Tensor([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
b = Tensor([10, 20, 30])             # shape (3,)

# b 广播为 [[10, 20, 30], [10, 20, 30]]
C = A + b
print(C.data)
# [[11, 22, 33],
#  [14, 25, 36]]
```

### 广播规则

1. 从右向左对齐维度
2. 维度大小相等，或其中一个为1，或不存在
3. 为1的维度会被复制扩展

```
(2, 3) + (3,)     -> (2, 3) + (1, 3) -> (2, 3)
(2, 3, 4) + (4,)  -> (2, 3, 4) + (1, 1, 4) -> (2, 3, 4)
(2, 3) + (2, 1)   -> (2, 3) + (2, 3) -> (2, 3)
(2, 3) + (2,)     -> 错误！无法广播
```

## 1.8 工厂方法

方便创建特殊张量：

```python
@classmethod
def zeros(cls, shape, requires_grad=False):
    """创建全0张量"""
    return cls(np.zeros(shape), requires_grad=requires_grad)

@classmethod
def ones(cls, shape, requires_grad=False):
    """创建全1张量"""
    return cls(np.ones(shape), requires_grad=requires_grad)

@classmethod
def randn(cls, shape, requires_grad=False):
    """创建标准正态分布张量"""
    return cls(np.random.randn(*shape), requires_grad=requires_grad)

@classmethod
def rand(cls, shape, requires_grad=False):
    """创建均匀分布张量 [0, 1)"""
    return cls(np.random.rand(*shape), requires_grad=requires_grad)


# 测试
z = Tensor.zeros((2, 3))
o = Tensor.ones((3, 3))
r = Tensor.randn((4, 4))  # 随机初始化，常用于权重
```

## 1.9 完整代码

```python
# tensor.py - 第一版完整实现
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
    
    # 运算
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
    
    # 矩阵
    def matmul(self, other): ...
    
    # 形状
    def reshape(self, shape): ...
    def transpose(self, dim0=0, dim1=1): ...
    def flatten(self): ...
    
    # 归约
    def sum(self, dim=None): ...
    def mean(self, dim=None): ...
    
    # 工厂方法
    @classmethod
    def zeros(cls, shape): ...
    @classmethod
    def ones(cls, shape): ...
    @classmethod
    def randn(cls, shape): ...
```

## 1.10 练习

1. **实现 `sqrt()` 方法**：计算逐元素平方根

2. **实现 `abs()` 方法**：计算绝对值

3. **实现 `clip(min_val, max_val)` 方法**：将值限制在范围内

4. **实现 `concatenate(tensors, dim)` 函数**：沿指定维度拼接张量

5. **挑战**：实现 `conv2d(x, weight, stride, padding)`（提示：使用 im2col）

## 下一章

现在我们有了一个能进行基本运算的 Tensor 类。下一章，我们将实现**自动微分**，让 Tensor 能够自动计算梯度！

→ [第二章：自动微分](02-autograd.md)
