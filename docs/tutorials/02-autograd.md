# 第二章：自动微分

自动微分（Automatic Differentiation, Autograd）是深度学习框架的核心。本章我们实现反向传播算法。

## 2.1 为什么需要自动微分？

考虑函数 `f(x) = x² + 2x + 1`，我们需要求 `df/dx`：

**手动求导**：`df/dx = 2x + 2`

**数值微分**（不精确）：
```python
def numerical_grad(f, x, eps=1e-5):
    return (f(x + eps) - f(x - eps)) / (2 * eps)
```

**自动微分**（精确、通用）：
```python
x = Tensor([3.0], requires_grad=True)
y = x**2 + 2*x + 1
y.backward()
print(x.grad)  # [8.] = 2*3 + 2
```

## 2.2 计算图

自动微分基于**计算图**：

```
x = 3
  │
  ├──→ (x²) ──┐
  │           │
  └──→ (2x) ──┼──→ (+) ──→ (+1) ──→ y = 16
              │
              └─────────────────────→ (+)
```

前向传播：从输入到输出
反向传播：从输出到输入，传递梯度

## 2.3 链式法则

反向传播的核心是**链式法则**：

```
y = f(g(x))
dy/dx = dy/dg * dg/dx
```

示例：`y = (x + 1)²`
- `g = x + 1`，`y = g²`
- `dy/dg = 2g`
- `dg/dx = 1`
- `dy/dx = dy/dg * dg/dx = 2g * 1 = 2(x+1)`

## 2.4 最简单的实现

让我们给 Tensor 添加梯度追踪：

```python
# tensor_v3.py
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        
        # 用于构建计算图
        self._prev = set(_children)  # 父节点
        self._op = _op               # 创建此张量的操作
        self._backward = lambda: None  # 反向传播函数
    
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad  # d(a+b)/da = 1
            
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += out.grad  # d(a+b)/db = 1
        
        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')
        
        def _backward():
            if self.requires_grad:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += other.data * out.grad  # d(a*b)/da = b
            
            if other.requires_grad:
                if other.grad is None:
                    other.grad = np.zeros_like(other.data)
                other.grad += self.data * out.grad  # d(a*b)/db = a
        
        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out
```

### 梯度公式推导

**加法**：`y = a + b`
- `dy/da = 1`
- `dy/db = 1`
- 所以 `a.grad += out.grad * 1`

**乘法**：`y = a * b`
- `dy/da = b`
- `dy/db = a`
- 所以 `a.grad += out.grad * b`

## 2.5 反向传播

现在实现 `backward()` 方法：

```python
def backward(self):
    # 构建拓扑排序
    topo = []
    visited = set()
    
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    
    build_topo(self)
    
    # 初始化输出梯度为1
    self.grad = np.ones_like(self.data)
    
    # 反向传播（从后往前）
    for node in reversed(topo):
        node._backward()


# 测试
x = Tensor([2.0], requires_grad=True)
y = x * x + 2 * x + 1  # x² + 2x + 1
y.backward()
print(x.grad)  # [6.] = 2*2 + 2 = 6
```

### 拓扑排序

为什么需要拓扑排序？

```
计算图:
  a ──→ c
  │     │
  └──→ b ──→ d
```

反向传播必须**从后向前**：
1. d → c, b
2. c → a
3. b → a

拓扑排序保证：父节点在子节点之后被处理。

## 2.6 更多运算的反向传播

```python
def __neg__(self):
    return self * (-1)

def __sub__(self, other):
    return self + (-other)

def __pow__(self, n):
    out = Tensor(self.data ** n, _children=(self,), _op=f'**{n}')
    
    def _backward():
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            # d(x^n)/dx = n * x^(n-1)
            self.grad += n * (self.data ** (n-1)) * out.grad
    
    out._backward = _backward
    out.requires_grad = self.requires_grad
    return out

def relu(self):
    out = Tensor(np.maximum(0, self.data), _children=(self,), _op='relu')
    
    def _backward():
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            # d(relu(x))/dx = 1 if x > 0 else 0
            self.grad += (self.data > 0).astype(np.float32) * out.grad
    
    out._backward = _backward
    out.requires_grad = self.requires_grad
    return out
```

### 常用导数公式

| 函数 f(x) | 导数 f'(x) |
|-----------|------------|
| `x + c` | `1` |
| `x * c` | `c` |
| `x²` | `2x` |
| `x^n` | `nx^(n-1)` |
| `e^x` | `e^x` |
| `ln(x)` | `1/x` |
| `sin(x)` | `cos(x)` |
| `cos(x)` | `-sin(x)` |
| `ReLU(x)` | `1 if x>0 else 0` |
| `Sigmoid(x)` | `σ(x)(1-σ(x))` |
| `Tanh(x)` | `1-tanh²(x)` |

## 2.7 矩阵运算的梯度

矩阵乘法的梯度是最复杂的：

```python
def matmul(self, other):
    out = Tensor(
        np.matmul(self.data, other.data),
        _children=(self, other),
        _op='@'
    )
    
    def _backward():
        if self.requires_grad:
            if self.grad is None:
                self.grad = np.zeros_like(self.data)
            # d(A @ B) / dA = grad @ B.T
            self.grad += np.matmul(out.grad, other.data.T)
        
        if other.requires_grad:
            if other.grad is None:
                other.grad = np.zeros_like(other.data)
            # d(A @ B) / dB = A.T @ grad
            other.grad += np.matmul(self.data.T, out.grad)
    
    out._backward = _backward
    out.requires_grad = self.requires_grad or other.requires_grad
    return out
```

### 矩阵乘法梯度推导

设 `Y = A @ B`，其中 `A: (M, K)`, `B: (K, N)`, `Y: (M, N)`

对于 `A[i,j]`：
```
∂Y/∂A[i,j] 只影响 Y[i,:] (第i行)
∂loss/∂A[i,j] = Σ_k (∂loss/∂Y[i,k] * ∂Y[i,k]/∂A[i,j])
              = Σ_k grad[i,k] * B[j,k]
              = (grad @ B.T)[i,j]
```

所以 `A.grad = grad @ B.T`

同理 `B.grad = A.T @ grad`

## 2.8 广播的梯度处理

当操作涉及广播时，梯度需要求和：

```python
def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, _children=(self, other), _op='+')
    
    def _backward():
        if self.requires_grad:
            grad = out.grad
            # 处理广播：如果 self 被广播，需要求和
            # 例如 (3,) + (2,3) -> self 的梯度需要沿 axis=0 求和
            self.grad = self.grad + grad if self.grad is not None else grad
        
        if other.requires_grad:
            grad = out.grad
            # 同样处理广播
            other.grad = other.grad + grad if other.grad is not None else grad
    
    out._backward = _backward
    return out
```

### 广播梯度示例

```python
# A: (2, 3), b: (3,)
# C = A + b，b 被广播为 (2, 3)
# C.grad: (2, 3)
# b.grad: 需要沿 axis=0 求和，变成 (3,)
```

## 2.9 完整实现

```python
# autograd.py
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
    
    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data)
        
        self.grad = gradient
        
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        self.grad = None
```

## 2.10 验证梯度正确性

使用数值梯度验证：

```python
def numerical_gradient(f, x, eps=1e-5):
    """计算数值梯度"""
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=['multi_index'])
    
    while not it.finished:
        idx = it.multi_index
        old_val = x.data[idx]
        
        x.data[idx] = old_val + eps
        fxh = f().data
        
        x.data[idx] = old_val - eps
        fxl = f().data
        
        x.data[idx] = old_val
        grad[idx] = (fxh - fxl) / (2 * eps)
        
        it.iternext()
    
    return grad


def check_gradient():
    x = Tensor([2.0, 3.0, 4.0], requires_grad=True)
    
    def f():
        return (x * x).sum()
    
    # 解析梯度
    y = f()
    y.backward()
    analytical = x.grad.copy()
    
    # 数值梯度
    x.grad = None
    numerical = numerical_gradient(f, x)
    
    # 比较
    diff = np.abs(analytical - numerical).max()
    print(f"Analytical: {analytical}")
    print(f"Numerical: {numerical}")
    print(f"Max diff: {diff}")
    
    assert diff < 1e-5, "Gradient check failed!"
    print("✓ Gradient check passed!")


check_gradient()
```

## 2.11 练习

1. **实现 `exp()` 的反向传播**

2. **实现 `log()` 的反向传播**

3. **实现 `sum(dim)` 的反向传播**（提示：梯度需要扩展回原形状）

4. **实现 `softmax()` 的反向传播**（Jacobian 矩阵）

5. **挑战**：实现 `conv2d` 的反向传播

## 2.12 调试技巧

```python
# 打印计算图
def print_graph(tensor, indent=0):
    print("  " * indent + f"{tensor._op or 'input'} {tensor.shape}")
    for child in tensor._prev:
        print_graph(child, indent + 1)

# 检查梯度流动
def check_grad_flow(tensor):
    for node in tensor._prev:
        if node.requires_grad and node.grad is None:
            print(f"⚠ No gradient at {node._op}")
        check_grad_flow(node)
```

## 下一章

现在 Tensor 可以自动计算梯度了！下一章，我们将实现**神经网络模块（Module）**，构建可训练的模型。

→ [第三章：Module 基类](03-nn-module.md)
