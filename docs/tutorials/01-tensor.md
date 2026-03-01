# 第一章：Tensor 基础

## 想象一张超级 Excel 表格...

想象你在用 Excel：

- 一个单元格：`A1 = 5` → **标量（0维）**
- 一行数据：`A1:A5 = [1,2,3,4,5]` → **向量（1维）**
- 一个表格：`A1:C3` 是 3行3列 → **矩阵（2维）**
- 多个工作表：10个工作表，每张 3×3 → **张量（3维）**

**Tensor（张量）就是这样一个"超级 Excel"** —— 它能存储任意维度的数据。

```
Excel 与 Tensor 的对应：

标量 (0D)        向量 (1D)         矩阵 (2D)          张量 (3D)
   5           [1, 2, 3]      [[1,2,3],       [[[1,2],    ← 第0张表
                 不可见            [4,5,6]]           [3,4]],
   单元格         一行              一个表            [[5,6],    ← 第1张表
                                                     [7,8]]]
shape: ()      shape: (3,)     shape: (2,3)     shape: (2,2,2)
```

**一句话总结**：Tensor = 可以有任意维度的数组，是深度学习中所有数据的容器。

---

## 1.1 为什么需要 Tensor？

**问题**：NumPy 不是已经有 ndarray 了吗？

**答案**：是的，但 Tensor 有两个关键能力：

| 能力 | NumPy ndarray | Tensor |
|------|--------------|--------|
| GPU 加速 | ❌ | ✅ |
| 自动求导 | ❌ | ✅ |

```
普通计算：        深度学习计算：
[1,2,3]          [1,2,3]
  +                +
[4,5,6]          [4,5,6]
  =                =
[5,7,9]          [5,7,9] → 还能记住：这个结果来自加法！
                   ↓
              backward() 时能自动计算梯度
```

---

## 1.2 最简单的 Tensor：从零开始

### 目标
创建一个能存储数据的 Tensor 类。

### 实现

```python
# tensor_v1.py - 最简单的版本
import numpy as np

class Tensor:
    """
    Tensor 类：深度学习的基础数据结构

    类比：一个超级 Excel 表格，但能记住数据从哪来
    """

    def __init__(self, data, requires_grad: bool = False):
        """
        初始化 Tensor

        参数:
            data: 数据（列表、NumPy数组、数字）
            requires_grad: 是否需要计算梯度（默认不需要）

        类比:
            requires_grad=True 就像在 Excel 里标记"这个单元格我要追踪"
        """
        # 统一转换为 NumPy 数组，方便计算
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)

        # 梯度相关
        self.requires_grad = requires_grad  # 是否需要梯度
        self.grad = None                    # 存储梯度值

    @property
    def shape(self):
        """形状：每个维度的大小"""
        return self.data.shape

    @property
    def ndim(self):
        """维度数：有几个轴"""
        return self.data.ndim

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"


# ===== 测试 =====
if __name__ == "__main__":
    # 标量
    scalar = Tensor(5)
    print(f"标量: {scalar}")  # Tensor(shape=(), requires_grad=False)

    # 向量
    vector = Tensor([1, 2, 3, 4, 5])
    print(f"向量: {vector}")  # Tensor(shape=(5,), requires_grad=False)

    # 矩阵
    matrix = Tensor([[1, 2], [3, 4]])
    print(f"矩阵: {matrix}")  # Tensor(shape=(2, 2), requires_grad=False)

    # 3D 张量（比如一张 RGB 图片）
    image = Tensor.ones((3, 224, 224))  # 3通道，224×224
    print(f"图像: {image}")  # Tensor(shape=(3, 224, 224), ...)
```

### 理解 shape

```
shape = (2, 3, 4) 的含义：

第0维大小=2  →  有2个 "3×4 的矩阵"
第1维大小=3  →  每个矩阵有3行
第2维大小=4  →  每行有4个元素

总元素数 = 2 × 3 × 4 = 24

可视化：
[                         ← 第0维（选哪个大块）
  [                       ← 第1维（选哪一行）
    [1, 2, 3, 4],         ← 第2维（选哪个元素）
    [5, 6, 7, 8],
    [9, 10, 11, 12]
  ],
  [
    [13, 14, 15, 16],
    [17, 18, 19, 20],
    [21, 22, 23, 24]
  ]
]
```

---

## 1.3 基本运算：让 Tensor 能计算

### 为什么需要这些运算？

神经网络本质上就是**大量的数学运算**：
- 加法：`y = x + b`（偏置）
- 乘法：`y = x * w`（权重）
- 矩阵乘法：`y = W @ x`（线性层）

### 实现

```python
# tensor_v2.py - 添加运算
class Tensor:
    # ... 前面的代码不变 ...

    def __add__(self, other):
        """
        加法: self + other

        类比：Excel 两列相加
        A列 + B列 = C列

        数学：如果 y = a + b，则 dy/da = 1
        """
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
        """
        乘法: self * other（逐元素）

        类比：Excel 两列逐行相乘
        A1*B1, A2*B2, A3*B3...

        数学：如果 y = a * b，则 dy/da = b
        """
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
        """取负: -self"""
        return Tensor(-self.data, requires_grad=self.requires_grad)

    def __sub__(self, other):
        """减法: self - other"""
        return self + (-other if isinstance(other, Tensor) else -other)

    def __truediv__(self, other):
        """除法: self / other"""
        if isinstance(other, (int, float)):
            return Tensor(self.data / other, requires_grad=self.requires_grad)
        return Tensor(self.data / other.data, requires_grad=self.requires_grad)

    def __pow__(self, n):
        """幂运算: self ** n"""
        return Tensor(self.data ** n, requires_grad=self.requires_grad)

    # 反向运算（支持 5 + tensor 这种写法）
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __rsub__(self, other): return (-self) + other


# ===== 测试 =====
a = Tensor([1, 2, 3])
b = Tensor([4, 5, 6])

print(f"a + b = {(a + b).data}")   # [5, 7, 9]
print(f"a * b = {(a * b).data}")   # [4, 10, 18]
print(f"a * 2 = {(a * 2).data}")   # [2, 4, 6]
print(f"a ** 2 = {(a ** 2).data}") # [1, 4, 9]
print(f"5 + a = {(5 + a).data}")   # [6, 7, 8] - 反向运算
```

---

## 1.4 矩阵乘法：神经网络的引擎

### 为什么矩阵乘法这么重要？

**神经网络的一层就是矩阵乘法！**

```
线性层计算：
输入 x: (1, 3)     [0.5, 0.3, 0.2]
权重 W: (3, 2)     [[0.1, 0.2],
                    [0.3, 0.4],
                    [0.5, 0.6]]
偏置 b: (1, 2)     [0.1, 0.1]

输出 y = x @ W + b: (1, 2)

计算过程：
y[0] = 0.5*0.1 + 0.3*0.3 + 0.2*0.5 + 0.1 = 0.05 + 0.09 + 0.1 + 0.1 = 0.34
y[1] = 0.5*0.2 + 0.3*0.4 + 0.2*0.6 + 0.1 = 0.1 + 0.12 + 0.12 + 0.1 = 0.44
```

### 矩阵乘法规则

```
A @ B 有意义的前提：A 的列数 = B 的行数

A: (M, K)
B: (K, N)
--------
C: (M, N)  ← 结果形状

例：A(2,3) @ B(3,4) = C(2,4)

记忆口诀：(M,K) @ (K,N) = (M,N)
         中间的 K 要相同，结果的形状取外边的 M 和 N
```

### 实现

```python
def matmul(self, other):
    """
    矩阵乘法: self @ other

    类比：
    想象 A 的每一行是一个"问题"
    B 的每一列是一个"答案模板"
    结果是每个问题用每个模板得到的"匹配分数"
    """
    if not isinstance(other, Tensor):
        other = Tensor(other)

    if self.ndim != 2 or other.ndim != 2:
        raise ValueError("matmul 需要 2D 张量")

    if self.shape[1] != other.shape[0]:
        raise ValueError(f"形状不匹配: {self.shape} @ {other.shape}")

    result = np.matmul(self.data, other.data)
    return Tensor(result, requires_grad=self.requires_grad)

def __matmul__(self, other):
    return self.matmul(other)


# ===== 测试 =====
A = Tensor([[1, 2], [3, 4]])  # 2×2
B = Tensor([[5, 6], [7, 8]])  # 2×2
C = A @ B

print(f"A @ B = \n{C.data}")
# [[19, 22],   ← [1*5+2*7, 1*6+2*8] = [19, 22]
#  [43, 50]]  ← [3*5+4*7, 3*6+4*8] = [43, 50]
```

### 手算验证

```
C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
       = 1*5 + 2*7
       = 5 + 14 = 19 ✓

C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
       = 1*6 + 2*8
       = 6 + 16 = 22 ✓
```

---

## 1.5 形状操作：改变数据排布

### 为什么需要改变形状？

```
场景1：图像数据
原始: (batch, channels, height, width) = (32, 3, 224, 224)
全连接层需要: (batch, features) = (32, 150528)
需要 flatten

场景2：注意力机制
Q: (batch, seq_len, heads, head_dim)
需要转置为: (batch, heads, seq_len, head_dim)
```

### 常用形状操作

```python
def reshape(self, new_shape):
    """
    改变形状（总元素数必须相同）

    类比：把一箱书重新排列，书总数不变
    (2,3) → (3,2) → (6,) → (1,6) 都可以
    但 (2,3) → (2,4) 不行！6 ≠ 8
    """
    if np.prod(new_shape) != np.prod(self.shape):
        raise ValueError(f"无法从 {self.shape} 变形为 {new_shape}")

    return Tensor(
        self.data.reshape(new_shape),
        requires_grad=self.requires_grad
    )

def transpose(self, dim0=0, dim1=1):
    """
    交换两个维度

    类比：Excel 的行列转置
    原来每行是一个样本，转置后每列是一个样本
    """
    axes = list(range(self.ndim))
    axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    return Tensor(
        np.transpose(self.data, axes),
        requires_grad=self.requires_grad
    )

@property
def T(self):
    """转置的简写"""
    return self.transpose()

def flatten(self, start_dim=0):
    """
    展平：把多个维度合并成一维

    类比：把一张照片"压扁"成一行像素
    (2, 3, 4) → flatten() → (24,)
    """
    new_shape = (-1,) + self.shape[start_dim+1:]
    return self.reshape(new_shape)

def squeeze(self, dim=None):
    """
    去掉大小为1的维度

    类比：拆掉不必要的包装盒
    (1, 3, 1, 4) → squeeze() → (3, 4)
    """
    return Tensor(
        np.squeeze(self.data, axis=dim),
        requires_grad=self.requires_grad
    )

def unsqueeze(self, dim):
    """
    增加一个大小为1的维度

    类比：加一个包装盒
    (3, 4) → unsqueeze(0) → (1, 3, 4)
    """
    return Tensor(
        np.expand_dims(self.data, axis=dim),
        requires_grad=self.requires_grad
    )


# ===== 测试 =====
t = Tensor([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)

print(f"原始: {t.shape}")                  # (2, 3)
print(f"reshape(3,2): {t.reshape((3,2)).shape}")  # (3, 2)
print(f"reshape(-1): {t.reshape((-1,)).shape}")   # (6,) - 展平
print(f"transpose: {t.transpose().shape}")        # (3, 2)
print(f"flatten: {t.flatten().shape}")            # (6,)

# unsqueeze 和 squeeze
x = Tensor([1, 2, 3])  # shape (3,)
y = x.unsqueeze(0)     # shape (1, 3) - 增加批次维度
z = y.squeeze()        # shape (3,) - 去掉大小为1的维度
print(f"unsqueeze: {y.shape}, squeeze: {z.shape}")
```

---

## 1.6 归约操作：汇总数据

### 什么是归约？

**归约 = 把多个值变成一个值**（减少数据量）

```
sum: [1, 2, 3, 4] → 10
mean: [1, 2, 3, 4] → 2.5
max: [1, 2, 3, 4] → 4
```

### 实现

```python
def sum(self, dim=None, keepdims=False):
    """
    求和

    参数:
        dim: 沿哪个维度求和（None表示全部）
        keepdims: 是否保持维度

    类比：Excel 的 SUM 函数
    """
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


# ===== 测试 =====
t = Tensor([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)

print(f"全部求和: {t.sum().data}")         # 21
print(f"按列求和(dim=0): {t.sum(dim=0).data}")  # [5, 7, 9]
print(f"按行求和(dim=1): {t.sum(dim=1).data}")  # [6, 15]
print(f"均值: {t.mean().data}")            # 3.5
```

### 维度理解图解

```
原始数据: [[1, 2, 3],
          [4, 5, 6]]
shape: (2, 3)

sum(dim=0) - 压缩第0维（把行叠起来）:
  想象把两张纸叠在一起，对应位置相加
  [1+4, 2+5, 3+6] = [5, 7, 9]
  结果 shape: (3,)

sum(dim=1) - 压缩第1维（把每行压扁）:
  想象把每一行卷起来
  [1+2+3, 4+5+6] = [6, 15]
  结果 shape: (2,)

记忆：
  dim=0 → 行消失（行被压缩）
  dim=1 → 列消失（列被压缩）
```

---

## 1.7 广播机制：不同形状也能运算

### 什么是广播？

**广播 = 自动扩展小张量，使其能与大张量运算**

```
类比：给全班同学发同一份试卷
学生: 30人（30个数）
试卷: 1份（1个数）
结果: 每个人都拿到同样的试卷（自动复制30份）

数学上：
[1, 2, 3] + 10 → [1, 2, 3] + [10, 10, 10] = [11, 12, 13]
```

### 广播规则

```
规则1: 从右向左对齐维度
规则2: 维度相等，或其中一个为1，或不存在
规则3: 为1的维度会被复制扩展

例子：
(2, 3) + (3,)       ✓
  ↓       ↓
(2, 3) + (1, 3)     ← (3,) 前面补1
  ↓       ↓
(2, 3) + (2, 3)     ← (1, 3) 复制变成 (2, 3)

(2, 3, 4) + (4,)    ✓
(2, 3) + (2, 1)     ✓
(2, 3) + (2,)       ✗ 无法广播！
```

### 代码验证

```python
# NumPy 自动处理广播
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
b = np.array([10, 20, 30])             # (3,)

# b 被广播为 [[10, 20, 30], [10, 20, 30]]
C = A + b
print(C)
# [[11, 22, 33],
#  [14, 25, 36]]
```

---

## 1.8 工厂方法：快速创建张量

```python
@classmethod
def zeros(cls, shape, requires_grad=False):
    """创建全0张量 - 常用于初始化偏置"""
    return cls(np.zeros(shape, dtype=np.float32), requires_grad=requires_grad)

@classmethod
def ones(cls, shape, requires_grad=False):
    """创建全1张量"""
    return cls(np.ones(shape, dtype=np.float32), requires_grad=requires_grad)

@classmethod
def randn(cls, shape, requires_grad=False):
    """
    创建标准正态分布张量 N(0,1)
    常用于初始化权重
    """
    return cls(np.random.randn(*shape).astype(np.float32), requires_grad=requires_grad)

@classmethod
def rand(cls, shape, requires_grad=False):
    """创建 [0,1) 均匀分布张量"""
    return cls(np.random.rand(*shape).astype(np.float32), requires_grad=requires_grad)

@classmethod
def arange(cls, start, end=None, step=1, requires_grad=False):
    """创建等差数列"""
    if end is None:
        end = start
        start = 0
    return cls(np.arange(start, end, step, dtype=np.float32), requires_grad=requires_grad)


# ===== 测试 =====
z = Tensor.zeros((2, 3))
o = Tensor.ones((3, 3))
r = Tensor.randn((4, 4))
a = Tensor.arange(5)

print(f"zeros:\n{z.data}")
print(f"ones:\n{o.data}")
print(f"randn 均值: {r.mean().data:.4f}, 标准差: {r.data.std():.4f}")
print(f"arange: {a.data}")
```

---

## 1.9 常见陷阱

### 陷阱1：形状不匹配

```python
# 错误示例
A = Tensor([[1, 2], [3, 4]])  # (2, 2)
B = Tensor([[1, 2, 3]])       # (1, 3)
# C = A @ B  # 错误！ (2,2) @ (1,3) 不兼容

# 正确做法：检查形状
print(f"A.shape[1] = {A.shape[1]}, B.shape[0] = {B.shape[0]}")
# 2 ≠ 1，无法相乘
```

### 陷阱2：修改了原数据

```python
# 错误示例
a = Tensor([1, 2, 3])
b = a  # 这只是引用，不是复制！
b.data[0] = 999
print(a.data)  # [999, 2, 3] - a 也被改了！

# 正确做法
c = Tensor(a.data.copy())  # 显式复制
```

### 陷阱3：忘记数据类型

```python
# 错误示例
a = Tensor([1, 2, 3])  # float32
b = np.array([1, 2, 3])  # int64
# a.data + b 可能有意想不到的结果

# 正确做法：统一类型
b = np.array([1, 2, 3], dtype=np.float32)
```

---

## 1.10 完整代码

```python
# tensor.py - 第一版完整实现
import numpy as np
from typing import Tuple, Union, Optional, List

class Tensor:
    """
    Tensor 类：深度学习的基础数据结构

    属性:
        data: NumPy 数组存储实际数据
        shape: 张量形状
        ndim: 维度数
        requires_grad: 是否需要梯度
        grad: 梯度值
    """

    def __init__(self, data, requires_grad: bool = False):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32)
        else:
            self.data = np.array(data, dtype=np.float32)

        self.requires_grad = requires_grad
        self.grad = None

    # 属性
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
        """获取标量值"""
        return self.data.item()

    # 算术运算
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

    # 矩阵运算
    def matmul(self, other): ...

    # 形状操作
    def reshape(self, shape): ...
    def transpose(self, dim0=0, dim1=1): ...
    def flatten(self): ...
    def squeeze(self, dim=None): ...
    def unsqueeze(self, dim): ...

    # 归约操作
    def sum(self, dim=None, keepdims=False): ...
    def mean(self, dim=None, keepdims=False): ...
    def max(self, dim=None, keepdims=False): ...
    def min(self, dim=None, keepdims=False): ...

    # 工厂方法
    @classmethod
    def zeros(cls, shape, requires_grad=False): ...
    @classmethod
    def ones(cls, shape, requires_grad=False): ...
    @classmethod
    def randn(cls, shape, requires_grad=False): ...
    @classmethod
    def rand(cls, shape, requires_grad=False): ...
```

---

## 1.11 练习

### 基础练习

1. **实现 `sqrt()` 方法**：计算逐元素平方根
   ```python
   def sqrt(self):
       return Tensor(np.sqrt(self.data), requires_grad=self.requires_grad)
   ```

2. **实现 `abs()` 方法**：计算绝对值

3. **实现 `clip(min_val, max_val)` 方法**：将值限制在范围内

### 进阶练习

4. **实现 `concatenate(tensors, dim)` 函数**：沿指定维度拼接张量

5. **实现 `stack(tensors, dim)` 函数**：沿新维度堆叠张量

### 挑战

6. **实现 `conv2d(x, weight, stride, padding)`**（提示：使用 im2col 技术）

---

## 一句话总结

| 概念 | 一句话 |
|------|--------|
| Tensor | 多维数组，存储深度学习中所有数据 |
| shape | 描述张量每个维度的大小 |
| matmul | 神经网络的核心运算 |
| reshape | 改变数据排布，总元素不变 |
| 广播 | 自动扩展小张量以匹配大张量 |

---

## 下一章

现在我们有了一个能存储数据、进行运算的 Tensor 类。

但是，**它还不能自动计算梯度！**

下一章，我们将实现**自动微分（Autograd）**，让 Tensor 真正拥有"学习"的能力。

→ [第二章：自动微分](02-autograd.md)

```python
# 预告：下一章你将实现这个
x = Tensor([2.0], requires_grad=True)
y = x * x + 2 * x + 1  # y = x² + 2x + 1
y.backward()
print(x.grad)  # [6.] = 2*2 + 2 = 6 （自动计算！）
```
