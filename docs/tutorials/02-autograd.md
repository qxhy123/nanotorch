# 第二章：自动微分（Autograd）

## 你有没有在山里迷过路？

清晨出发时，阳光明媚。午后，你发现自己身处群山之中，四周都是陌生的风景。

天色渐暗，你必须尽快下山。问题是——**往哪个方向走，能最快到达山脚？**

你环顾四周，发现脚下的坡度似乎指向东北。于是你迈出一步。再环顾四周，调整方向。一步，又一步...直到看见山谷的灯光。

这就是**梯度下降**：每一步，都沿着最陡峭的方向往下走。

```
山的高度 = 损失函数
你的位置 = 模型参数
下坡方向 = 梯度的反方向
山脚 = 最优解
```

但有一个问题：在神经网络这座"万维山"中，手动计算梯度几乎是不可能的。百万个参数，意味着百万个方向。

**自动微分**，就是你的指南针。它替你记住每一步是怎么走来的，然后精确地告诉你：每个方向，坡度是多少。

---

## 2.1 为什么需要自动微分？

### 问题：训练神经网络需要计算梯度

神经网络的训练过程：

```
1. 前向传播：输入 → 网络 → 预测
2. 计算损失：预测 vs 真实值 → 损失
3. 反向传播：损失 → 梯度（告诉每个参数怎么调整）
4. 更新参数：参数 - 学习率 × 梯度
```

**第3步需要计算梯度**，问题是：网络可能有几百万个参数，手动求导是不可能的！

### 三种求导方法对比

| 方法 | 原理 | 优点 | 缺点 |
|------|------|------|------|
| **手动求导** | 人算公式 | 精确 | 太慢，容易错 |
| **数值微分** | `(f(x+h)-f(x-h))/2h` | 简单 | 不精确，慢 |
| **自动微分** | 计算图 + 链式法则 | 精确、快 | 实现复杂 |

### 数值微分示例

```python
def numerical_grad(f, x, eps=1e-5):
    """数值微分：用很小的差分近似导数"""
    return (f(x + eps) - f(x - eps)) / (2 * eps)

# 测试：f(x) = x²，在 x=3 处的导数
f = lambda x: x ** 2
print(numerical_grad(f, 3))  # ≈ 6.0（精确值是 6）

# 问题：如果函数很复杂，需要算很多次 f()
```

### 自动微分：我们要实现的

```python
x = Tensor([3.0], requires_grad=True)
y = x ** 2           # y = x²
y.backward()         # 自动计算梯度
print(x.grad)        # [6.] ← 精确答案！
```

**一句话总结**：自动微分 = 计算机帮你算导数，你只需要写前向计算。

---

## 2.2 计算图：自动微分的地图

### 什么是计算图？

**计算图 = 把计算过程画成图**

```
计算 y = (x + 1)²

从左到右（前向传播）：
  x ──→ (+1) ──→ g=x+1 ──→ (平方) ──→ y=g²
  3      4          4         16

从右到左（反向传播）：
  x ←── (+1) ←── g ←── (平方) ←── y
  2      2         32         1
  ↑
  这是 x 的梯度！
```

### 生活类比：快递配送

```
计算图就像快递追踪：

前向传播（发货）：
北京仓库 → 郑州中转 → 武汉中转 → 长沙送达
   x      →   +1    →   平方   →   y

反向传播（如果包裹破损，追溯责任）：
长沙收件 → 武汉责任 → 郑州责任 → 北京责任
   y     ←   导数   ←   导数   ←   x梯度
```

### 代码：记录计算过程

```python
class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None

        # 记录计算图
        self._prev = set(_children)  # "父节点"：这个张量是谁生成的
        self._op = _op               # 操作名：比如 '+'、'*'、'**2'
        self._backward = lambda: None  # 反向传播函数
```

---

## 2.3 链式法则：反向传播的核心

### 什么是链式法则？

**链式法则 = 复合函数的导数 = 外层导数 × 内层导数**

```
如果 y = f(g(x))，那么：
dy/dx = dy/dg × dg/dx

口诀：外层导数 × 内层导数
```

### 例子：y = (x + 1)²

```
分解：
  g = x + 1   （内层）
  y = g²      （外层）

链式法则：
  dy/dx = dy/dg × dg/dx
        = 2g    × 1
        = 2(x+1) × 1
        = 2(x+1)

当 x = 3 时：
  dy/dx = 2(3+1) = 8

验证：
  y = (3+1)² = 16
  x=3.001: y = (4.001)² = 16.008
  变化率 ≈ (16.008-16)/0.001 = 8 ✓
```

### 图解链式法则

```
前向传播（从左到右）：
    x ──────→ g=x+1 ──────→ y=g²
    3           4            16

反向传播（从右到左，传递梯度）：
    x ←────── g ←───────── y
    ↑         ↑            ↑
  dg/dx=1   dy/dg=8     dy/dy=1
    │         │            │
    └──── 1×8=8 ────┘

x的梯度 = dy/dg × dg/dx = 8 × 1 = 8
```

---

## 2.4 实现加法和乘法的反向传播

### 加法的梯度

```
y = a + b

问：dy/da = ?  dy/db = ?

答：都是 1！

因为：
  d(a+b)/da = 1
  d(a+b)/db = 1

直观理解：
  a 增加多少，y 就增加多少
  b 增加多少，y 也增加多少
```

```python
def __add__(self, other):
    """加法 + 反向传播"""
    other = other if isinstance(other, Tensor) else Tensor(other)

    # 前向：计算结果
    out = Tensor(
        self.data + other.data,
        _children=(self, other),
        _op='+'
    )

    # 反向：定义梯度计算
    def _backward():
        # dy/da = 1, dy/db = 1
        # 所以梯度直接传过去
        if self.requires_grad:
            self.grad = (self.grad or 0) + out.grad
        if other.requires_grad:
            other.grad = (other.grad or 0) + out.grad

    out._backward = _backward
    out.requires_grad = self.requires_grad or other.requires_grad
    return out
```

### 乘法的梯度

```
y = a × b

问：dy/da = ?  dy/db = ?

答：
  dy/da = b  （把 a 当变量，b 当常数）
  dy/db = a  （把 b 当变量，a 当常数）

直观理解：
  a 的变化会被 b 放大
  b 的变化会被 a 放大
```

```python
def __mul__(self, other):
    """乘法 + 反向传播"""
    other = other if isinstance(other, Tensor) else Tensor(other)

    out = Tensor(
        self.data * other.data,
        _children=(self, other),
        _op='*'
    )

    def _backward():
        # dy/da = b, dy/db = a
        if self.requires_grad:
            self.grad = (self.grad or 0) + other.data * out.grad
        if other.requires_grad:
            other.grad = (other.grad or 0) + self.data * out.grad

    out._backward = _backward
    out.requires_grad = self.requires_grad or other.requires_grad
    return out
```

### 为什么用 `+=` 而不是 `=`？

```python
# 场景：a 被使用两次
a = Tensor([2.0], requires_grad=True)
b = a + a  # a 用了两次！

# 反向传播时：
# a.grad = 1 (第一次加法)
# a.grad = 1 (第二次加法)
# 总梯度 = 1 + 1 = 2

# 所以必须累加！
self.grad = (self.grad or 0) + grad_contribution
```

---

## 2.5 实现 backward()：触发反向传播

### 拓扑排序

**问题**：反向传播必须按正确顺序执行。

```
计算图：
  a ──→ c ──→ e (输出)
  │     │
  └─→ d ─┘

错误的顺序：先更新 a，再更新 c → c 的梯度还没算出来！
正确的顺序：e → c/d → a（从输出往回）
```

**拓扑排序**：保证父节点在子节点之后被处理。

```python
def backward(self):
    """触发反向传播"""

    # 1. 构建拓扑排序
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)  # 后序遍历：子节点先加入

    build_topo(self)

    # 2. 初始化输出梯度为1
    self.grad = np.ones_like(self.data)

    # 3. 反向传播（从后往前）
    for node in reversed(topo):
        node._backward()
```

### 图解拓扑排序

```
计算图：
    a ──┬──→ c ──┐
        │        │
        └──→ d ──┼──→ e (输出)
                 │
        b ───────┘

后序遍历顺序：a, b, c, d, e
反序（反向传播顺序）：e, d, c, b, a

这样保证了：
- e 的梯度先算
- d 和 c 的梯度在 e 之后算
- a 和 b 的梯度最后算
```

### 完整示例

```python
# 构建计算图
x = Tensor([2.0], requires_grad=True)
y = x * x + 2 * x + 1  # y = x² + 2x + 1

# 反向传播
y.backward()

# 检查梯度
# dy/dx = 2x + 2 = 2*2 + 2 = 6
print(f"x.grad = {x.grad}")  # [6.]
```

---

## 2.6 更多运算的反向传播

### 幂运算

```python
def __pow__(self, n):
    """幂运算: x ** n"""
    out = Tensor(
        self.data ** n,
        _children=(self,),
        _op=f'**{n}'
    )

    def _backward():
        # d(x^n)/dx = n * x^(n-1)
        if self.requires_grad:
            grad = n * (self.data ** (n-1)) * out.grad
            self.grad = (self.grad or 0) + grad

    out._backward = _backward
    out.requires_grad = self.requires_grad
    return out


# 测试
x = Tensor([3.0], requires_grad=True)
y = x ** 3  # y = x³
y.backward()
print(x.grad)  # [27.] = 3 * 3² = 3 * 9 = 27
```

### ReLU

```python
def relu(self):
    """ReLU: max(0, x)"""
    out = Tensor(
        np.maximum(0, self.data),
        _children=(self,),
        _op='relu'
    )

    def _backward():
        # d(relu(x))/dx = 1 if x > 0 else 0
        if self.requires_grad:
            mask = (self.data > 0).astype(np.float32)
            self.grad = (self.grad or 0) + mask * out.grad

    out._backward = _backward
    out.requires_grad = self.requires_grad
    return out


# 测试
x = Tensor([-2.0, 0.0, 2.0], requires_grad=True)
y = x.relu()
y.sum().backward()
print(x.grad)  # [0., 0., 1.] - 负数和0的梯度是0，正数的梯度是1
```

### 常用导数公式表

| 函数 f(x) | 导数 f'(x) | 记忆技巧 |
|-----------|-----------|---------|
| `x + c` | `1` | 加常数不影响变化率 |
| `x × c` | `c` | 乘常数就是缩放变化率 |
| `x²` | `2x` | 幂函数"降一次幂，乘次数" |
| `x^n` | `nx^(n-1)` | 同上 |
| `e^x` | `e^x` | 自己不变 |
| `ln(x)` | `1/x` | 对数变成分数 |
| `sin(x)` | `cos(x)` | 正弦变余弦 |
| `cos(x)` | `-sin(x)` | 余弦变负正弦 |
| `ReLU(x)` | `1 if x>0 else 0` | 正数通过，负数截断 |
| `σ(x)` (sigmoid) | `σ(1-σ)` | 自己乘(1-自己) |
| `tanh(x)` | `1-tanh²` | 1减自己平方 |

---

## 2.7 矩阵乘法的梯度

### 为什么矩阵乘法特殊？

矩阵乘法是神经网络的核心，但它的梯度计算比较复杂。

```
Y = A @ B

A: (M, K)  输入或权重
B: (K, N)  权重或输入
Y: (M, N)  输出

问题：d(loss)/dA 和 d(loss)/dB 是多少？
```

### 推导过程

```
Y[i,j] = Σ_k A[i,k] × B[k,j]

对于 A[i,k]：
  dY[i,j]/dA[i,k] = B[k,j]  （只有这一个项包含 A[i,k]）

  d(loss)/dA[i,k] = Σ_j d(loss)/dY[i,j] × B[k,j]
                  = Σ_j Y.grad[i,j] × B[k,j]
                  = (Y.grad @ B.T)[i,k]

所以：A.grad = Y.grad @ B.T

同理：B.grad = A.T @ Y.grad
```

### 实现

```python
def matmul(self, other):
    """矩阵乘法 + 反向传播"""
    out = Tensor(
        np.matmul(self.data, other.data),
        _children=(self, other),
        _op='@'
    )

    def _backward():
        if self.requires_grad:
            # A.grad = Y.grad @ B.T
            self.grad = (self.grad or 0) + np.matmul(out.grad, other.data.T)

        if other.requires_grad:
            # B.grad = A.T @ Y.grad
            other.grad = (other.grad or 0) + np.matmul(self.data.T, out.grad)

    out._backward = _backward
    out.requires_grad = self.requires_grad or other.requires_grad
    return out
```

### 形状验证

```
A: (M, K) = (2, 3)
B: (K, N) = (3, 4)
Y: (M, N) = (2, 4)
Y.grad: (2, 4)

A.grad = Y.grad @ B.T
       = (2, 4) @ (4, 3)
       = (2, 3) ✓ 和 A 形状一致

B.grad = A.T @ Y.grad
       = (3, 2) @ (2, 4)
       = (3, 4) ✓ 和 B 形状一致
```

---

## 2.8 广播的梯度处理

### 问题：广播后梯度怎么处理？

```python
A = Tensor([[1, 2, 3],    # shape (2, 3)
            [4, 5, 6]])
b = Tensor([10, 20, 30])  # shape (3,)

C = A + b  # b 被广播为 (2, 3)

# 问题：b 的梯度应该是什么形状？
# C.grad: (2, 3)
# b.grad: ?
```

### 解决：对被广播的维度求和

```
b 广播过程：
[10, 20, 30] → [[10, 20, 30],
               [10, 20, 30]]

反向传播时：
  b 的每个元素"分身"到了多个位置
  梯度需要"合并"回来

b.grad = sum(C.grad, axis=0) = [C.grad[0,:] + C.grad[1,:]]
```

### 实现

```python
def __add__(self, other):
    other = other if isinstance(other, Tensor) else Tensor(other)
    out = Tensor(self.data + other.data, _children=(self, other), _op='+')

    def _backward():
        if self.requires_grad:
            grad = out.grad
            # 处理广播：如果形状不同，需要对被广播的维度求和
            if self.shape != out.shape:
                # 找出被广播的维度，沿这些维度求和
                grad = _unbroadcast(grad, self.shape)
            self.grad = (self.grad or 0) + grad

        if other.requires_grad:
            grad = out.grad
            if other.shape != out.shape:
                grad = _unbroadcast(grad, other.shape)
            other.grad = (other.grad or 0) + grad

    out._backward = _backward
    return out

def _unbroadcast(grad, target_shape):
    """将梯度求和到目标形状"""
    # 增加维度直到维度数匹配
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    # 对大小为1的维度求和
    for i, (grad_dim, target_dim) in enumerate(zip(grad.shape, target_shape)):
        if target_dim == 1 and grad_dim != 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad
```

---

## 2.9 验证梯度正确性

### 用数值梯度检验

```python
def numerical_gradient(f, x, eps=1e-5):
    """计算数值梯度（用于验证）"""
    grad = np.zeros_like(x.data)

    it = np.nditer(x.data, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        old_val = x.data[idx]

        # f(x + eps)
        x.data[idx] = old_val + eps
        fxh = f().data.copy()

        # f(x - eps)
        x.data[idx] = old_val - eps
        fxl = f().data.copy()

        # 恢复原值
        x.data[idx] = old_val

        # 中心差分
        grad[idx] = (fxh - fxl).sum() / (2 * eps)
        it.iternext()

    return grad


def check_gradient():
    """梯度检验"""
    x = Tensor([2.0, 3.0, 4.0], requires_grad=True)

    def f():
        return (x * x).sum()  # f = x1² + x2² + x3²

    # 解析梯度（我们的实现）
    y = f()
    y.backward()
    analytical = x.grad.copy()

    # 数值梯度（基准真值）
    x.grad = None
    numerical = numerical_gradient(f, x)

    # 比较
    print(f"解析梯度: {analytical}")  # [4., 6., 8.]
    print(f"数值梯度: {numerical}")  # [4., 6., 8.]
    print(f"最大差异: {np.abs(analytical - numerical).max()}")

    assert np.allclose(analytical, numerical, atol=1e-5)
    print("✓ 梯度检验通过！")


check_gradient()
```

---

## 2.10 完整代码

```python
# autograd.py
import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=(), _op=''):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, _children=(self, other), _op='+')

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or 0) + out.grad
            if other.requires_grad:
                other.grad = (other.grad or 0) + out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, _children=(self, other), _op='*')

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or 0) + other.data * out.grad
            if other.requires_grad:
                other.grad = (other.grad or 0) + self.data * out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad or other.requires_grad
        return out

    def __pow__(self, n):
        out = Tensor(self.data ** n, _children=(self,), _op=f'**{n}')

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or 0) + n * (self.data ** (n-1)) * out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other

    def relu(self):
        out = Tensor(np.maximum(0, self.data), _children=(self,), _op='relu')

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or 0) + (self.data > 0) * out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def sum(self):
        out = Tensor(self.data.sum(), _children=(self,), _op='sum')

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad or 0) + np.ones_like(self.data) * out.grad

        out._backward = _backward
        out.requires_grad = self.requires_grad
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        self.grad = None
```

---

## 2.11 调试技巧

### 打印计算图

```python
def print_graph(tensor, indent=0):
    """可视化计算图"""
    prefix = "  " * indent
    op = tensor._op or "input"
    print(f"{prefix}└─ {op} {tensor.shape}")
    for child in tensor._prev:
        print_graph(child, indent + 1)


x = Tensor([2.0], requires_grad=True)
y = x * x + 2 * x + 1
print_graph(y)
# └─ + (1,)
#   └─ * (1,)
#     └─ input (1,)
#     └─ input (1,)
#   └─ * (1,)
#     └─ input (1,)
#     └─ input (1,)
```

### 检查梯度流动

```python
def check_grad_flow(tensor):
    """检查梯度是否正确传播"""
    for node in tensor._prev:
        if node.requires_grad and node.grad is None:
            print(f"⚠ 警告：{node._op} 的梯度为 None")
        check_grad_flow(node)
```

### 常见问题排查

```python
# 问题1：梯度为 None
# 原因：忘记设置 requires_grad=True
x = Tensor([1.0])  # requires_grad=False
y = x * 2
y.backward()
print(x.grad)  # None

# 问题2：梯度累积
# 原因：多次 backward 没有清零
x = Tensor([1.0], requires_grad=True)
for _ in range(3):
    y = x * 2
    y.backward()
print(x.grad)  # 6 而不是 2！应该每次 backward 前清零

# 问题3：计算图断开
# 原因：使用了 .data 进行运算
x = Tensor([1.0], requires_grad=True)
y = Tensor(x.data * 2)  # 错误！断开了计算图
y.backward()
print(x.grad)  # None
```

---

## 2.12 练习

### 基础练习

1. **实现 `exp()` 的反向传播**
   - 提示：d(e^x)/dx = e^x

2. **实现 `log()` 的反向传播**
   - 提示：d(ln(x))/dx = 1/x

3. **实现 `mean()` 的反向传播**
   - 提示：梯度平均分配给每个元素

### 进阶练习

4. **实现 `softmax()` 的反向传播**
   - 提示：需要计算 Jacobian 矩阵

5. **实现 `LayerNorm()` 的反向传播**
   - 提示：涉及均值、方差、缩放和平移

### 挑战

6. **实现 `conv2d` 的反向传播**
   - 提示：需要对输入和权重分别计算梯度

---

## 一句话总结

| 概念 | 一句话 |
|------|--------|
| 计算图 | 记录计算过程，像快递追踪 |
| 链式法则 | 外层导数 × 内层导数 |
| 反向传播 | 从输出往回传梯度 |
| 拓扑排序 | 保证父节点在子节点之后处理 |
| 梯度累加 | 一个变量用多次，梯度要加起来 |

---

## 下一章

现在 Tensor 可以自动计算梯度了！

下一章，我们将实现 **Module 基类**，构建可复用的神经网络层。

→ [第三章：Module 基类](03-nn-module.md)

```python
# 预告：下一章你将实现这个
class Linear(Module):
    def __init__(self, in_features, out_features):
        self.W = Tensor.randn((in_features, out_features))
        self.b = Tensor.zeros((out_features,))

    def forward(self, x):
        return x @ self.W + self.b

# 然后就可以像 PyTorch 一样用了！
```
