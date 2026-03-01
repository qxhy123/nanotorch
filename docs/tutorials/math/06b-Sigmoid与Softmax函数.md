# 第六章（b）：Sigmoid 与 Softmax 函数

Sigmoid 和 Softmax 是深度学习中最重要的输出层激活函数。Sigmoid 用于二分类，Softmax 用于多分类。本节将深入探讨这些函数的数学性质及其在深度学习中的应用。

---

## 目录

1. [Sigmoid 函数](#sigmoid-函数)
2. [Sigmoid 函数族](#sigmoid-函数族)
3. [Softmax 函数](#softmax-函数)
4. [Softmax 与交叉熵](#softmax-与交叉熵)
5. [温度参数](#温度参数)
6. [在深度学习中的应用](#在深度学习中的应用)

---

## Sigmoid 函数

### 定义

**Sigmoid 函数**（也称 Logistic 函数）：

$$
\sigma(x) = \frac{1}{1 + e^{-x}} = \frac{e^x}{e^x + 1}
$$

### 核心性质

1. **值域**：$(0, 1)$，输出可解释为概率

2. **导数**（最重要的性质）：

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

**证明**：

$$
\begin{align}
\sigma'(x) &= \frac{d}{dx}\left(\frac{1}{1+e^{-x}}\right) \\
&= \frac{e^{-x}}{(1+e^{-x})^2} \\
&= \frac{1}{1+e^{-x}} \cdot \frac{e^{-x}}{1+e^{-x}} \\
&= \sigma(x) \cdot (1 - \sigma(x))
\end{align}
$$

3. **对称性**：

$$
\sigma(-x) = 1 - \sigma(x)
$$

4. **逆函数**（Logit 函数）：

$$
\sigma^{-1}(y) = \ln\left(\frac{y}{1-y}\right) = \text{logit}(y)
$$

5. **渐近线**：
   - $\lim_{x \to -\infty} \sigma(x) = 0$
   - $\lim_{x \to +\infty} \sigma(x) = 1$

6. **特殊值**：
   - $\sigma(0) = 0.5$
   - $\sigma'(0) = 0.25$（最大梯度）

### Sigmoid 的优缺点

**优点**：
- 输出范围 $(0, 1)$，适合概率解释
- 处处可微，平滑
- 导数计算简单

**缺点**：
- **梯度消失**：当 $|x|$ 较大时，$\sigma'(x) \to 0$
- **非零中心**：输出恒为正，影响梯度更新方向
- **计算开销**：指数运算较慢

### 数值稳定的实现

```python
import numpy as np

def sigmoid(x):
    """数值稳定的 sigmoid"""
    # 对于大正数和大负数使用不同的计算方式
    pos_mask = x >= 0
    neg_mask = ~pos_mask
    
    result = np.zeros_like(x, dtype=np.float64)
    
    # 正数情况：1 / (1 + exp(-x))
    result[pos_mask] = 1 / (1 + np.exp(-x[pos_mask]))
    
    # 负数情况：exp(x) / (1 + exp(x))
    exp_x = np.exp(x[neg_mask])
    result[neg_mask] = exp_x / (1 + exp_x)
    
    return result

def sigmoid_derivative(x):
    """Sigmoid 的导数"""
    s = sigmoid(x)
    return s * (1 - s)

# 示例
x = np.array([-10, -1, 0, 1, 10])
print(f"Sigmoid: {sigmoid(x)}")
print(f"导数: {sigmoid_derivative(x)}")
```

---

## Sigmoid 函数族

### Hard Sigmoid

计算更快的线性近似：

$$
\text{HardSigmoid}(x) = \max(0, \min(1, 0.2x + 0.5))
$$

```python
def hard_sigmoid(x):
    """Hard Sigmoid - 线性近似"""
    return np.clip(0.2 * x + 0.5, 0, 1)
```

**优点**：计算快速，适合移动端
**缺点**：在转折点不可微

### Swish / SiLU (Sigmoid Linear Unit)

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

**导数**：

$$
\text{Swish}'(x) = \sigma(x) + x \cdot \sigma'(x) = \sigma(x) \left(1 + x - \frac{x}{1 + e^x}\right)
$$

**性质**：
- 非单调（不同于 ReLU）
- 有下界（$-\infty$ 时趋近于 0），无上界
- 平滑
- 在深层网络中通常优于 ReLU

```python
def swish(x):
    """Swish/SiLU 激活函数"""
    return x * sigmoid(x)

def swish_derivative(x):
    """Swish 的导数"""
    s = sigmoid(x)
    return s + x * s * (1 - s)
```

### Mish

$$
\text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x)) = x \cdot \tanh(\text{softplus}(x))
$$

**性质**：
- 比 Swish 更平滑
- 通常在视觉任务上表现更好
- 计算开销更大

```python
def mish(x):
    """Mish 激活函数"""
    return x * np.tanh(np.log1p(np.exp(x)))  # log1p(exp(x)) = log(1 + exp(x))
```

### Sigmoid 函数族比较

| 函数 | 公式 | 特点 |
|------|------|------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | 经典，概率输出 |
| Hard Sigmoid | $\max(0, \min(1, 0.2x+0.5))$ | 快速，不可微 |
| Swish/SiLU | $x \cdot \sigma(x)$ | 非单调，平滑 |
| Mish | $x \cdot \tanh(\ln(1+e^x))$ | 更平滑，计算慢 |

```python
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 1000)

plt.figure(figsize=(12, 6))
plt.plot(x, sigmoid(x), label='Sigmoid', linewidth=2)
plt.plot(x, hard_sigmoid(x), label='Hard Sigmoid', linewidth=2)
plt.plot(x, swish(x), label='Swish/SiLU', linewidth=2)
plt.plot(x, mish(x), label='Mish', linewidth=2)

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Sigmoid 函数族比较')
plt.legend()
plt.grid(True)
plt.xlim([-5, 5])
plt.ylim([-1, 2])

plt.tight_layout()
plt.show()
```

---

## Softmax 函数

### 定义

**Softmax** 将向量 $\mathbf{z} = (z_1, \ldots, z_K)$ 映射到概率分布：

$$
\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad i = 1, \ldots, K
$$

### 核心性质

1. **归一化**：$\sum_{i=1}^K \text{Softmax}(\mathbf{z})_i = 1$

2. **非负性**：$\text{Softmax}(\mathbf{z})_i > 0$

3. **最大值保持**：

$$
\arg\max_i z_i = \arg\max_i \text{Softmax}(\mathbf{z})_i
$$

4. **平移不变性**：

$$
\text{Softmax}(\mathbf{z} + c) = \text{Softmax}(\mathbf{z}), \quad \forall c \in \mathbb{R}
$$

5. **缩放性质**：

$$
\text{Softmax}(\alpha \mathbf{z}) \xrightarrow{\alpha \to \infty} \text{OneHot}(\arg\max \mathbf{z})
$$

### Softmax 的导数

设 $s_i = \text{Softmax}(\mathbf{z})_i$，则：

$$
\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)
$$

其中 $\delta_{ij}$ 是 Kronecker delta（$i=j$ 时为 1，否则为 0）。

**两种情况**：
- $i = j$：$\frac{\partial s_i}{\partial z_i} = s_i(1 - s_i)$
- $i \neq j$：$\frac{\partial s_i}{\partial z_j} = -s_i s_j$

**Jacobian 矩阵**：

$$
\mathbf{J} = \text{diag}(\mathbf{s}) - \mathbf{s} \mathbf{s}^\top
$$

### Log-Softmax

为了避免数值问题，通常使用 log-softmax：

$$
\ln s_i = z_i - \ln\left(\sum_{j=1}^K e^{z_j}\right)
$$

**数值稳定实现**：

$$
\ln s_i = z_i - \max_k z_k - \ln\left(\sum_{j=1}^K e^{z_j - \max_k z_k}\right)
$$

```python
def softmax(x, axis=-1):
    """数值稳定的 Softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def log_softmax(x, axis=-1):
    """数值稳定的 Log-Softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    return x - x_max - np.log(np.sum(np.exp(x - x_max), axis=axis, keepdims=True))
```

---

## Softmax 与交叉熵

### 交叉熵损失

对于真实标签 $y$（one-hot 编码）和预测概率 $\mathbf{s}$：

$$
\mathcal{L} = -\sum_{i=1}^K y_i \ln s_i = -\ln s_{y}
$$

其中 $y$ 是真实类别（one-hot 时只有一个非零）。

### 梯度（交叉熵 + Softmax）

**最优雅的性质**：

$$
\frac{\partial \mathcal{L}}{\partial z_i} = s_i - y_i
$$

这是非常简洁的形式！梯度就是预测概率减去真实标签。

**证明思路**：

$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial z_i} &= -\sum_j y_j \frac{\partial \ln s_j}{\partial z_i} \\
&= -\sum_j y_j \frac{1}{s_j} \frac{\partial s_j}{\partial z_i} \\
&= -\sum_j y_j \frac{1}{s_j} s_j(\delta_{ij} - s_i) \\
&= -\sum_j y_j (\delta_{ij} - s_i) \\
&= s_i - y_i
\end{align}
$$

```python
def softmax_cross_entropy(logits, labels):
    """
    Softmax 交叉熵损失
    
    Args:
        logits: 模型输出（未归一化）
        labels: 真实标签（整数或 one-hot）
    """
    # Log-softmax
    log_probs = log_softmax(logits, axis=-1)
    
    # 交叉熵
    if labels.ndim == logits.ndim:
        # One-hot labels
        return -np.sum(labels * log_probs, axis=-1)
    else:
        # Integer labels
        return -log_probs[np.arange(len(labels)), labels]

# 示例
logits = np.array([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])
labels = np.array([0, 1])  # 真实类别

probs = softmax(logits)
print("Softmax 概率:")
print(probs)

loss = softmax_cross_entropy(logits, labels)
print(f"\n交叉熵损失: {loss}")

# 梯度
def softmax_cross_entropy_gradient(logits, labels):
    """交叉熵 + Softmax 的梯度"""
    probs = softmax(logits)
    if labels.ndim == logits.ndim:
        return probs - labels
    else:
        # One-hot encode labels
        one_hot = np.zeros_like(logits)
        one_hot[np.arange(len(labels)), labels] = 1
        return probs - one_hot

grad = softmax_cross_entropy_gradient(logits, labels)
print(f"\n梯度:\n{grad}")
```

---

## 温度参数

### 定义

带温度参数 $T$ 的 Softmax：

$$
\text{Softmax}_T(\mathbf{z})_i = \frac{e^{z_i/T}}{\sum_{j=1}^K e^{z_j/T}}
$$

### 温度的作用

- **$T > 1$**：分布更**平滑**，概率差异减小
- **$T = 1$**：标准 Softmax
- **$T < 1$**：分布更**尖锐**，概率差异增大
- **$T \to 0$**：趋近于 argmax（one-hot）

```python
def softmax_with_temperature(x, temperature=1.0, axis=-1):
    """带温度参数的 Softmax"""
    x = x / temperature
    return softmax(x, axis=axis)

# 示例：不同温度下的概率分布
logits = np.array([2.0, 1.0, 0.1])

print("不同温度下的概率分布:")
for T in [0.1, 0.5, 1.0, 2.0, 10.0]:
    probs = softmax_with_temperature(logits, T)
    print(f"T={T:4.1f}: {probs}")
```

### 应用场景

1. **知识蒸馏**：使用高温度让学生模型学习教师模型的软标签
2. **强化学习**：控制探索与利用的平衡
3. **文本生成**：控制输出的多样性

---

## 在深度学习中的应用

### 二分类输出层

```python
# 二分类
logits = model(x)
probs = sigmoid(logits)  # 输出概率
loss = binary_cross_entropy(probs, labels)
```

### 多分类输出层

```python
# 多分类（注意：通常不需要显式调用 softmax）
logits = model(x)
loss = cross_entropy_loss(logits, labels)  # 内部有 softmax
```

### 多标签分类

```python
# 多标签分类（每个类别独立）
logits = model(x)
probs = sigmoid(logits)  # 每个类别独立使用 sigmoid
loss = binary_cross_entropy(probs, labels)
```

### 门控机制 (LSTM/GRU)

```python
# LSTM 中的门控
forget_gate = sigmoid(W_f @ h + b_f)
input_gate = sigmoid(W_i @ h + b_i)
output_gate = sigmoid(W_o @ h + b_o)
```

### 注意力权重

```python
# 自注意力中的权重计算
attention_scores = queries @ keys.T / sqrt(d_k)
attention_weights = softmax(attention_scores, axis=-1)
```

### nanotorch 中的使用

```python
from nanotorch import Tensor
from nanotorch.nn import Sigmoid, Softmax, BCELoss, CrossEntropyLoss

# 二分类
sigmoid = Sigmoid()
bce_loss = BCELoss()

logits = Tensor.randn((32, 1))
targets = Tensor(np.random.randint(0, 2, (32, 1)).astype(np.float32))

probs = sigmoid(logits)
loss = bce_loss(probs, targets)

# 多分类
softmax = Softmax(dim=-1)
ce_loss = CrossEntropyLoss()

logits = Tensor.randn((32, 10))
labels = Tensor(np.random.randint(0, 10, 32))

loss = ce_loss(logits, labels)  # 内部包含 softmax
```

---

## 小结

本节介绍了 Sigmoid 和 Softmax 函数：

| 函数 | 公式 | 用途 |
|------|------|------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | 二分类、门控 |
| Swish | $x \cdot \sigma(x)$ | 隐藏层激活 |
| Softmax | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | 多分类输出 |
| Log-Softmax | $z_i - \log\sum_j e^{z_j}$ | 数值稳定 |

**关键要点**：
- Sigmoid 用于二分类，Softmax 用于多分类
- Softmax + 交叉熵的梯度非常简洁：$s_i - y_i$
- 温度参数控制分布的平滑程度
- 数值稳定性是实现的痛点

---

**上一节**：[指数、对数与三角函数](06a-指数对数与三角函数.md)

**下一节**：[ReLU 函数族与激活函数](06c-ReLU函数族与激活函数.md) - 学习 ReLU、LeakyReLU、GELU 等激活函数。

**返回**：[第六章：初等函数](06-elementary-functions.md) | [数学基础教程目录](../math-fundamentals.md)
