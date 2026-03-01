# 第四章：激活函数

## 想象一个开关...

你走进一个黑暗的房间，墙上有一个开关：

```
开关状态：
  关 (0) ──→ 按 ──→ 开 (1)
  开 (1) ──→ 按 ──→ 关 (0)

只有两种状态，要么全开，要么全关。
```

**激活函数就像神经网络的"开关"** —— 但更智能，它可以是：
- 渐变调光器（Sigmoid：0到1之间平滑过渡）
- 只往上开的门（ReLU：负数变0，正数保持）
- 波浪形状（Tanh：-1到1之间波动）

---

## 4.1 为什么需要激活函数？

### 问题：没有激活函数 = 只能画直线

```
两层线性网络：
y = W2 @ (W1 @ x + b1) + b2
  = (W2 @ W1) @ x + (W2 @ b1 + b2)
  = W @ x + b                    ← 还是线性的！

无论叠加多少层，最终等价于一层！
```

### 解决：激活函数引入"弯曲"

```
有了激活函数：
y = W2 @ relu(W1 @ x + b1) + b2

relu() 打破了线性关系！
现在网络可以拟合任意复杂的曲线。
```

### 图解：激活函数的作用

```
没有激活函数：              有激活函数：

    │   /                      │  ╱╲
    │  /                       │ ╱  ╲    ╱
    │ /                        │╱    ╲__╱
    └────────→                 └────────→
     只能画直线                  可以画任意曲线
```

**一句话总结**：激活函数让神经网络从"只能画直线"变成"能画任意曲线"。

---

## 4.2 ReLU：最简单最常用

### 什么是 ReLU？

```
ReLU(x) = max(0, x)

简单说：
  x > 0 → 输出 x
  x ≤ 0 → 输出 0

类比：一扇只能往外推的门
  推（正数）→ 门开了
  拉（负数）→ 门不动（还是0）
```

### 图示

```
      ReLU 函数图像

    y │    /
      │   /
      │  /
    0 ├─┴──────→ x
      │
      └ 负数被"截断"成0
```

### 实现

```python
class ReLU(Module):
    """
    ReLU: f(x) = max(0, x)

    优点：计算快，缓解梯度消失
    缺点：负数区域梯度为0（"死亡ReLU"）
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


# Tensor 类中的实现
def relu(self):
    out = Tensor(
        np.maximum(0, self.data),
        _children=(self,),
        _op='relu'
    )

    def _backward():
        if self.requires_grad:
            # 梯度：x>0 传1，x≤0 传0
            mask = (self.data > 0).astype(np.float32)
            self.grad = (self.grad or 0) + mask * out.grad

    out._backward = _backward
    out.requires_grad = self.requires_grad
    return out
```

### 为什么 ReLU 这么流行？

```
1. 计算超快：只需要比较和取最大值
2. 梯度不消失：正数区域梯度恒为1
3. 稀疏激活：约50%神经元输出0，节省计算
```

---

## 4.3 LeakyReLU：解决"死亡"问题

### 问题：死亡 ReLU

```
如果某个神经元始终输出 ≤ 0：
  → 梯度永远为 0
  → 参数永远不更新
  → 神经元"死亡"

坏初始化或大学习率可能导致这个问题
```

### 解决：给负数一点"漏气"

```
LeakyReLU(x) = x           if x > 0
             = α * x       if x ≤ 0  (α 通常=0.01)

图示：
      y │    /
        │   /
        │  /
      0 ├─┴──────→ x
        │  ╲
        │   ╲ 负数也有小梯度（斜率=0.01）
```

### 实现

```python
class LeakyReLU(Module):
    """LeakyReLU：负数区域保留小梯度"""

    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return x.leaky_relu(self.negative_slope)


# Tensor 方法
def leaky_relu(self, alpha=0.01):
    out = Tensor(
        np.where(self.data > 0, self.data, alpha * self.data),
        _children=(self,),
        _op='leaky_relu'
    )

    def _backward():
        if self.requires_grad:
            grad = np.where(self.data > 0, 1.0, alpha)
            self.grad = (self.grad or 0) + grad * out.grad

    out._backward = _backward
    return out
```

---

## 4.4 Sigmoid：压缩到 0-1

### 什么是 Sigmoid？

```
Sigmoid(x) = 1 / (1 + e^(-x))

作用：把任意数字压缩到 (0, 1) 区间

类比：一个永远不彻底的开关
  -∞ → 接近 0（几乎关）
  0  → 恰好 0.5（半开半关）
  +∞ → 接近 1（几乎开）
```

### 图示

```
    Sigmoid 函数图像（S形曲线）

1.0 │          ___
    │       __/
0.5 │----__/--------
    │ __/
0.0 │/
    └───────────────→ x
       -∞     0    +∞
```

### 实现

```python
class Sigmoid(Module):
    """
    Sigmoid: f(x) = 1 / (1 + e^(-x))

    用途：二分类输出层
    问题：梯度消失（最大梯度只有0.25）
    """

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()


# Tensor 方法
def sigmoid(self):
    # 数值稳定版本
    out = Tensor(
        np.where(self.data >= 0,
                 1 / (1 + np.exp(-self.data)),
                 np.exp(self.data) / (1 + np.exp(self.data))),
        _children=(self,),
        _op='sigmoid'
    )

    def _backward():
        if self.requires_grad:
            # 导数 = sigmoid * (1 - sigmoid)
            s = out.data
            self.grad = (self.grad or 0) + s * (1 - s) * out.grad

    out._backward = _backward
    return out
```

---

## 4.5 Tanh：压缩到 -1 到 1

```
Tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))

作用：把任意数字压缩到 (-1, 1) 区间

和 Sigmoid 的区别：
  Sigmoid: (0, 1)   → 只有正数
  Tanh:    (-1, 1)  → 有正有负（零中心）
```

### 图示

```
    Tanh 函数图像

 1.0 │      ___
     │    _/
  0.0 ├──_/_\_────────→ x
     │     -
-1.0 │      ---
     │
```

```python
class Tanh(Module):
    """Tanh：零中心的S形曲线"""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()
```

---

## 4.6 Softmax：多分类必备

### 什么是 Softmax？

```
Softmax 把一组数字变成"概率分布"

输入：[2.0, 1.0, 0.1]
输出：[0.659, 0.242, 0.099]  ← 和为1

公式：softmax(x_i) = e^(x_i) / Σ e^(x_j)

类比：一场选美比赛
  - 每个选手有一个分数
  - Softmax 把分数转成"得票率"
  - 所有得票率加起来 = 100%
```

### 实现

```python
class Softmax(Module):
    """Softmax：把向量变成概率分布"""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(dim=self.dim)


# Tensor 方法
def softmax(self, dim=-1):
    # 数值稳定性：减去最大值
    shifted = self.data - np.max(self.data, axis=dim, keepdims=True)
    exp_x = np.exp(shifted)
    out = exp_x / np.sum(exp_x, axis=dim, keepdims=True)

    result = Tensor(out, _children=(self,), _op='softmax')

    def _backward():
        if self.requires_grad:
            # Softmax 导数（简化形式）
            sum_grad = np.sum(result.grad * out, axis=dim, keepdims=True)
            grad_input = out * (result.grad - sum_grad)
            self.grad = (self.grad or 0) + grad_input

    result._backward = _backward
    result.requires_grad = self.requires_grad
    return result
```

---

## 4.7 GELU：Transformer 的选择

```
GELU = x * Φ(x)  （Φ 是标准正态分布的累积分布函数）

近似：GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715*x³)))

ReLU vs GELU：
  ReLU: 在0处"硬"截断
  GELU: 平滑过渡，处处可导

GELU 在 Transformer（BERT、GPT）中效果更好
```

```python
class GELU(Module):
    """GELU：Transformer 的首选激活函数"""

    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()
```

---

## 4.8 激活函数对比表

| 函数 | 公式 | 输出范围 | 梯度范围 | 常用场景 |
|------|------|---------|---------|----------|
| ReLU | max(0,x) | [0,∞) | {0,1} | **隐藏层默认选择** |
| LeakyReLU | max(αx,x) | (-∞,∞) | {α,1} | 避免死亡ReLU |
| Sigmoid | 1/(1+e⁻ˣ) | (0,1) | (0,0.25) | 二分类输出 |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | (-1,1) | (0,1) | RNN隐藏层 |
| Softmax | eˣᵢ/Σeˣⱼ | (0,1) | 复杂 | **多分类输出** |
| GELU | x·Φ(x) | (-∞,∞) | 连续 | **Transformer** |

### 选择建议

```
隐藏层：
  首选 ReLU
  如果很多神经元"死亡" → 换 LeakyReLU
  Transformer 架构 → 用 GELU

输出层：
  二分类 → Sigmoid
  多分类 → Softmax
  回归 → 不用激活函数（或用恒等函数）
```

---

## 4.9 使用示例

```python
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU, Softmax

# 经典的 MLP 结构
model = Sequential(
    Linear(784, 256),
    ReLU(),           # 隐藏层用 ReLU
    Linear(256, 128),
    ReLU(),
    Linear(128, 10),
    Softmax(dim=-1)   # 输出层用 Softmax（多分类）
)

x = Tensor.randn((32, 784))
probs = model(x)

print(f"输出形状: {probs.shape}")           # (32, 10)
print(f"概率和: {probs.sum(dim=1).data}")   # [1, 1, 1, ...] 每行和为1
print(f"最大概率: {probs.max(dim=1).data}") # 每个样本的预测类别概率
```

---

## 4.10 常见陷阱

### 陷阱1：输出层用错激活函数

```python
# 错误：多分类输出层用了 ReLU
model = Sequential(
    Linear(784, 10),
    ReLU()  # ✗ 输出可能是 [0, 0, 2.5, 0, ...]，不是概率！
)

# 正确：用 Softmax
model = Sequential(
    Linear(784, 10),
    Softmax()  # ✓ 输出是概率分布
)
```

### 陷阱2：Sigmoid 梯度消失

```python
# 问题：深层网络用 Sigmoid
for i in range(10):
    x = Linear(256, 256)(x)
    x = Sigmoid()(x)  # 每层梯度最多×0.25
# 10层后梯度 ≈ 0.25^10 ≈ 0.0000001 → 梯度消失！

# 解决：隐藏层用 ReLU
for i in range(10):
    x = Linear(256, 256)(x)
    x = ReLU()(x)  # 梯度要么0要么1，不会消失
```

---

## 4.11 练习

### 基础练习

1. **实现 ELU**：`f(x) = x if x > 0 else α(e^x - 1)`

2. **实现 Swish/SiLU**：`f(x) = x * sigmoid(x)`（YOLO常用）

3. **实现 PReLU**：α 是可学习参数

### 进阶练习

4. **画出各激活函数曲线**（用 matplotlib）

5. **比较不同激活函数的训练速度**

---

## 一句话总结

| 概念 | 一句话 |
|------|--------|
| 激活函数 | 让神经网络能画曲线，不只是直线 |
| ReLU | 最简单实用：负数变0，正数不变 |
| Sigmoid | 压缩到0-1，用于二分类 |
| Softmax | 压缩成概率分布，和为1 |
| 选择原则 | 隐藏层ReLU，输出层看任务 |

---

## 下一章

现在我们有了激活函数！

但是，怎么告诉网络"预测得对不对"呢？我们需要**损失函数**来衡量预测和真实值的差距。

→ [第五章：损失函数](05-loss.md)

```python
# 预告：下一章你将实现
loss = CrossEntropyLoss()
L = loss(predictions, targets)  # 告诉网络"差多远"
L.backward()  # 计算梯度，告诉网络"怎么改"
```
