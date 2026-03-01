# 第六章：初等函数

初等函数是**深度学习的构建基石**。从激活函数到损失函数，从概率分布到归一化操作，初等函数无处不在。本章将系统介绍深度学习中常用的数学函数及其性质。

---

## 章节结构

为了便于学习和深入理解，本章分为四个子章节：

### [6.1 指数、对数与三角函数](06a-指数对数与三角函数.md)

**内容概要**：
- 指数函数的定义、性质与泰勒展开
- 对数函数的定义与核心性质
- 三角函数：sin, cos, tan 及其恒等式
- 双曲函数：sinh, cosh, tanh
- Tanh 作为激活函数的特性
- 数值稳定的实现技巧

**核心概念**：
| 函数 | 定义 | 关键性质 | 深度学习应用 |
|------|------|----------|--------------|
| $e^x$ | $\sum_{n=0}^{\infty} \frac{x^n}{n!}$ | $(e^x)' = e^x$ | Softmax、概率 |
| $\ln x$ | $e^y = x$ 的反函数 | $(\ln x)' = 1/x$ | 交叉熵、MLE |
| $\tanh$ | $\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | 零中心、$(-1,1)$ | RNN、GELU |

**[开始学习 →](06a-指数对数与三角函数.md)**

---

### [6.2 Sigmoid 与 Softmax 函数](06b-Sigmoid与Softmax函数.md)

**内容概要**：
- Sigmoid 函数及其导数
- Sigmoid 函数族：Hard Sigmoid、Swish、Mish
- Softmax 函数的定义与性质
- Softmax 的导数（Jacobian 矩阵）
- Softmax 与交叉熵的结合
- 温度参数的作用

**核心概念**：
| 函数 | 公式 | 用途 |
|------|------|------|
| Sigmoid | $\frac{1}{1+e^{-x}}$ | 二分类、门控 |
| Swish | $x \cdot \sigma(x)$ | 隐藏层激活 |
| Softmax | $\frac{e^{z_i}}{\sum_j e^{z_j}}$ | 多分类输出 |

**[开始学习 →](06b-Sigmoid与Softmax函数.md)**

---

### [6.3 ReLU 函数族与激活函数](06c-ReLU函数族与激活函数.md)

**内容概要**：
- ReLU 的定义与"死亡 ReLU"问题
- Leaky ReLU 与 PReLU
- ELU 与 SELU（自归一化）
- GELU（Transformer 标配）
- 激活函数选择指南

**核心概念**：
| 函数 | 核心特点 | 推荐场景 |
|------|----------|----------|
| ReLU | 简单、高效 | CNN、通用 |
| LeakyReLU | 无死亡问题 | 通用 |
| GELU | 平滑、非单调 | Transformer |

**[开始学习 →](06c-ReLU函数族与激活函数.md)**

---

### [6.4 损失函数与归一化](06d-损失函数与归一化.md)

**内容概要**：
- 回归损失：MSE、MAE、Huber Loss
- 分类损失：BCE、交叉熵、Focal Loss
- Batch Normalization 原理与实现
- Layer Normalization（Transformer 必备）
- 其他归一化：InstanceNorm、GroupNorm、RMSNorm

**核心概念**：
| 损失/归一化 | 公式/方法 | 用途 |
|-------------|----------|------|
| MSE | $\frac{1}{n}\sum(y-\hat{y})^2$ | 回归 |
| CE | $-\sum y \ln\hat{y}$ | 分类 |
| BatchNorm | 按 batch 归一化 | CNN |
| LayerNorm | 按 feature 归一化 | Transformer |

**[开始学习 →](06d-损失函数与归一化.md)**

---

## 学习路径

```
┌─────────────────────────────────────────────────────────────┐
│                     第六章：初等函数                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  6.1 指数对数 ──→ 6.2 Sigmoid ──→ 6.3 ReLU ──→ 6.4 损失函数  │
│  与三角函数       与 Softmax      函数族       与归一化        │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ e^x, ln  │   │ Sigmoid  │   │ ReLU     │   │ MSE, CE  │ │
│  │ tanh     │   │ Softmax  │   │ GELU     │   │ BatchNorm│ │
│  │ 数值稳定 │   │ 温度参数  │   │ 激活选择  │   │ LayerNorm│ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                             │
│  应用: 激活函数, 损失函数, 归一化, 概率建模                      │
└─────────────────────────────────────────────────────────────┘
```

## 为什么初等函数对深度学习重要？

### 1. 激活函数

深度网络的每一层都需要非线性激活函数：

```
线性变换: y = Wx + b
    ↓
激活函数: a = σ(y)
    ↓
非线性使深层网络有意义
```

### 2. 损失函数与概率

| 损失函数 | 概率模型 | 激活函数 |
|---------|---------|---------|
| MSE | 高斯分布 | 无 |
| BCE | 伯努利分布 | Sigmoid |
| CE | 类别分布 | Softmax |

### 3. 归一化的必要性

- **内部协变量偏移**：每层输入分布变化
- **梯度消失/爆炸**：深层网络信号衰减
- **归一化解决**：稳定训练、加速收敛

---

## 核心公式速查

### Sigmoid

$$
\sigma(x) = \frac{1}{1 + e^{-x}}, \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

### Softmax

$$
\text{Softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

### ReLU

$$
\text{ReLU}(x) = \max(0, x)
$$

### 交叉熵

$$
\mathcal{L} = -\sum_i y_i \ln \hat{y}_i
$$

---

## Python 代码示例

### 激活函数

```python
import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
```

### 损失函数

```python
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def cross_entropy(logits, labels):
    log_probs = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = log_probs - np.log(np.sum(np.exp(log_probs), axis=-1, keepdims=True))
    return -np.mean(log_probs[np.arange(len(labels)), labels])
```

### BatchNorm

```python
def batch_norm(x, gamma, beta, eps=1e-5):
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta
```

---

## 学习建议

1. **理解函数性质**：导数、值域、特殊点
2. **掌握数值稳定性**：防止溢出和下溢
3. **联系实际应用**：每个函数在哪些场景使用
4. **动手实现**：用 NumPy 实现所有函数
5. **可视化**：画出函数及其导数的图像

---

## 延伸阅读

- [第五章：最优化方法](05-optimization.md) - 梯度下降与优化算法
- [第四章：数理统计](04-statistics.md) - 概率分布与统计推断

---

**返回**：[数学基础教程目录](../math-fundamentals.md)
