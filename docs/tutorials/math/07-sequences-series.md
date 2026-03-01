# 第七章：数列与级数

数列与级数是**深度学习的数学基石**。从学习率衰减策略到序列建模的递推关系，从注意力机制的位置编码到Transformer的序列处理，数列与级数的概念无处不在。本章将系统介绍数列与级数的核心概念及其在深度学习中的应用。

---

## 章节结构

为了便于学习和深入理解，本章分为四个子章节：

### [7.1 数列基础](07a-数列基础.md)

**内容概要**：
- 数列的定义与表示方法
- 等差数列：通项公式、求和公式
- 等比数列：通项公式、求和公式
- 递推数列与递推关系
- 数列的单调性与有界性

**核心概念**：
| 概念 | 公式 | 在深度学习中的应用 |
|------|------|-------------------|
| 等差数列通项 | $a_n = a_1 + (n-1)d$ | 线性学习率衰减 |
| 等比数列通项 | $a_n = a_1 \cdot r^{n-1}$ | 指数学习率衰减、位置编码 |
| 等比数列求和 | $S_n = a_1 \frac{1-r^n}{1-r}$ | RNN 梯度传播分析 |
| 递推关系 | $a_n = f(a_{n-1}, a_{n-2}, \ldots)$ | RNN 隐状态更新 |

**[开始学习 →](07a-数列基础.md)**

---

### [7.2 数列极限](07b-数列极限.md)

**内容概要**：
- 数列极限的 $\epsilon-N$ 定义
- 极限的性质与运算法则
- 收敛数列的判定方法
- Cauchy 收敛准则
- 重要极限：$e$ 的定义、单调有界原理

**核心概念**：
| 概念 | 定义/公式 | 重要性 |
|------|----------|--------|
| 极限定义 | $\forall \epsilon > 0, \exists N, \forall n > N: \|a_n - L\| < \epsilon$ | 严格数学基础 |
| 唯一性 | 收敛数列的极限唯一 | 理论保证 |
| 有界性 | 收敛数列必有界 | 判定工具 |
| Cauchy 准则 | $\forall \epsilon > 0, \exists N, \forall m,n > N: \|a_m - a_n\| < \epsilon$ | 完备性基础 |

**[开始学习 →](07b-数列极限.md)**

---

### [7.3 级数与求和](07c-级数与求和.md)

**内容概要**：
- 数项级数的收敛与发散
- 正项级数的判别法（比较、比值、根值）
- 交错级数与 Leibniz 判别法
- 幂级数与收敛半径
- 常见级数的求和

**核心概念**：
| 级数类型 | 收敛条件 | 应用场景 |
|----------|----------|----------|
| 几何级数 $\sum r^n$ | $\|r\| < 1$ | RNN 长期依赖分析 |
| p-级数 $\sum \frac{1}{n^p}$ | $p > 1$ | 正则化项分析 |
| 调和级数 $\sum \frac{1}{n}$ | 发散 | 学习率调度理论 |
| 泰勒级数 | 收敛半径内 | 函数近似、优化理论 |

**[开始学习 →](07c-级数与求和.md)**

---

### [7.4 数列在深度学习中的应用](07d-数列在深度学习中的应用.md)

**内容概要**：
- 学习率衰减策略（指数衰减、余弦退火）
- RNN 中的序列建模与梯度传播
- Transformer 位置编码（正弦位置编码）
- 自注意力中的 Softmax 序列
- 序列生成与采样策略

**核心概念**：
| 应用 | 数列/级数概念 | 具体形式 |
|------|--------------|----------|
| 指数衰减 | 等比数列 | $\eta_t = \eta_0 \cdot \gamma^t$ |
| 余弦退火 | 三角函数数列 | $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\frac{\pi t}{T}))$ |
| 位置编码 | 正弦/余弦函数 | $PE_{(pos,2i)} = \sin(pos/10000^{2i/d})$ |
| 梯度传播 | 矩阵的几何级数 | $\prod_{t} W_h$ 的特征值分析 |

**[开始学习 →](07d-数列在深度学习中的应用.md)**

---

## 学习路径

```
┌─────────────────────────────────────────────────────────────┐
│                     第七章：数列与级数                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  7.1 数列基础 ──→ 7.2 数列极限 ──→ 7.3 级数与求和 ──→ 7.4 深度学习应用 │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ 等差数列 │   │ ε-N定义  │   │ 收敛判别 │   │ 学习率衰减│ │
│  │ 等比数列 │   │ Cauchy   │   │ 幂级数   │   │ 位置编码  │ │
│  │ 递推关系 │   │ 重要极限 │   │ 泰勒展开 │   │ 梯度传播  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                             │
│  应用: 学习率调度, RNN, Transformer位置编码, 序列建模         │
└─────────────────────────────────────────────────────────────┘
```

## 为什么数列与级数对深度学习重要？

### 1. 训练过程中的数列

```
训练过程 = 参数序列 {θ₁, θ₂, θ₃, ...}
    ↓
学习率序列 {η₁, η₂, η₃, ...}
    ↓
损失序列 {L₁, L₂, L₃, ...} → 期望收敛到最优
```

### 2. 序列建模的核心

| 模型 | 序列处理方式 | 数列概念 |
|------|-------------|----------|
| RNN | 递推更新隐状态 | 递推数列 |
| LSTM | 门控递推关系 | 复杂递推系统 |
| Transformer | 位置编码 | 正弦/余弦数列 |
| Attention | Softmax 权重 | 归一化序列 |

### 3. 理论分析工具

- **梯度消失/爆炸**：分析 $\prod_{t} W_h$ 的极限行为
- **收敛性证明**：损失序列的收敛条件
- **正则化**：级数求和形式的正则项

---

## 核心公式速查

### 等差数列

$$
a_n = a_1 + (n-1)d
$$

$$
S_n = \frac{n(a_1 + a_n)}{2} = \frac{n[2a_1 + (n-1)d]}{2}
$$

### 等比数列

$$
a_n = a_1 \cdot r^{n-1}
$$

$$
S_n = \begin{cases} na_1 & r = 1 \\ a_1 \frac{1-r^n}{1-r} & r \neq 1 \end{cases}
$$

### 无穷等比级数

当 $|r| < 1$ 时：

$$
\sum_{n=0}^{\infty} a_1 r^n = \frac{a_1}{1-r}
$$

### 重要极限

$$
\lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n = e \approx 2.71828
$$

---

## Python 代码示例

### 数列生成

```python
import numpy as np

def arithmetic_sequence(a1, d, n):
    """等差数列"""
    return np.array([a1 + i * d for i in range(n)])

def geometric_sequence(a1, r, n):
    """等比数列"""
    return np.array([a1 * (r ** i) for i in range(n)])

# 示例
print("等差数列:", arithmetic_sequence(1, 2, 10))
print("等比数列:", geometric_sequence(1, 0.5, 10))
```

### 级数求和

```python
def geometric_sum(a1, r, n):
    """等比级数前n项和"""
    if abs(r - 1) < 1e-10:
        return n * a1
    return a1 * (1 - r**n) / (1 - r)

def infinite_geometric_sum(a1, r):
    """无穷等比级数和（当|r| < 1时收敛）"""
    if abs(r) >= 1:
        return float('inf')  # 发散
    return a1 / (1 - r)

# 示例
print("前10项和:", geometric_sum(1, 0.5, 10))
print("无穷级数和:", infinite_geometric_sum(1, 0.5))
```

### 学习率衰减

```python
class ExponentialDecay:
    """指数学习率衰减"""
    def __init__(self, initial_lr, decay_rate, decay_steps):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
    
    def get_lr(self, step):
        return self.initial_lr * (self.decay_rate ** (step / self.decay_steps))

class CosineAnnealing:
    """余弦退火"""
    def __init__(self, initial_lr, min_lr, total_steps):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
    
    def get_lr(self, step):
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
               (1 + np.cos(np.pi * step / self.total_steps))

# 可视化比较
import matplotlib.pyplot as plt

steps = np.arange(0, 1000)
exp_decay = ExponentialDecay(0.1, 0.96, 100)
cosine = CosineAnnealing(0.1, 0.001, 1000)

plt.figure(figsize=(10, 5))
plt.plot(steps, [exp_decay.get_lr(s) for s in steps], label='指数衰减')
plt.plot(steps, [cosine.get_lr(s) for s in steps], label='余弦退火')
plt.xlabel('Step')
plt.ylabel('学习率')
plt.title('学习率衰减策略比较')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 学习建议

1. **从基础开始**：先掌握等差、等比数列，再学习极限和级数
2. **理解收敛性**：收敛是深度学习训练的核心目标
3. **关注应用**：将数列概念与学习率调度、RNN 联系起来
4. **动手实践**：用 Python 实现各种数列和学习率策略
5. **理论联系实际**：理解梯度消失/爆炸的数列本质

---

## 延伸阅读

- [第二章：微积分](02-calculus.md) - 极限与导数
- [第五章：最优化方法](05-optimization.md) - 学习率调度
- [第六章：初等函数](06-elementary-functions.md) - 指数与三角函数

---

**返回**：[数学基础教程目录](../math-fundamentals.md)
