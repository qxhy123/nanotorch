# 第五章：最优化方法

最优化是**机器学习的核心引擎**。从最小化损失函数到寻找最优参数，最优化理论为深度学习提供了数学基础和实用算法。本章将系统介绍最优化方法的核心概念及其在深度学习中的应用。

---

## 章节结构

为了便于学习和深入理解，本章分为四个子章节：

### [5.1 优化基础与梯度下降](05a-优化基础与梯度下降.md)

**内容概要**：
- 最优化问题的一般形式与最优性条件
- 凸集、凸函数与凸优化的重要性
- 梯度下降算法原理与实现
- 步长选择策略（固定步长、线搜索、回溯）
- 收敛性分析与条件数的影响
- 随机梯度下降与 Mini-batch

**核心概念**：
| 概念 | 公式 | 在深度学习中的应用 |
|------|------|-------------------|
| 梯度下降 | $\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$ | 参数更新 |
| 凸函数 | $f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta)f(y)$ | 全局最优保证 |
| 强凸性 | $\nabla^2 f \succeq \mu I$ | 线性收敛 |
| 条件数 | $\kappa = L/\mu$ | 收敛速度 |

**[开始学习 →](05a-优化基础与梯度下降.md)**

---

### [5.2 动量法与加速技术](05b-动量法与加速技术.md)

**内容概要**：
- 动量法 (Momentum) 的物理类比与公式
- 动量系数的作用与选择
- Nesterov 加速梯度 (NAG) 的前瞻机制
- 收敛率比较：$O(1/t)$ vs $O(1/t^2)$
- 动量在高条件数问题上的优势

**核心概念**：
| 方法 | 更新公式 | 收敛率（强凸） |
|------|----------|----------------|
| 梯度下降 | $x_{t+1} = x_t - \eta \nabla f$ | $O((1-1/\kappa)^t)$ |
| 标准动量 | $v_{t+1} = \beta v_t + \nabla f$ | $O((1-\sqrt{1/\kappa})^t)$ |
| NAG | $v_{t+1} = \beta v_t + \nabla f(x_t - \eta\beta v_t)$ | $O((1-\sqrt{1/\kappa})^t)$ |

**[开始学习 →](05b-动量法与加速技术.md)**

---

### [5.3 自适应学习率方法](05c-自适应学习率方法.md)

**内容概要**：
- 自适应学习率的动机
- AdaGrad：累积梯度平方调整学习率
- RMSprop：指数移动平均解决学习率衰减
- Adam：动量 + 自适应学习率
- AdamW：解耦权重衰减
- 优化器选择指南

**核心概念**：
| 方法 | 核心思想 | 适用场景 |
|------|----------|----------|
| AdaGrad | 累积梯度平方 | 稀疏数据 |
| RMSprop | EMA 梯度平方 | RNN、非平稳目标 |
| Adam | 动量 + 自适应 | 通用、快速原型 |
| AdamW | Adam + 解耦正则 | Transformer |

**[开始学习 →](05c-自适应学习率方法.md)**

---

### [5.4 学习率调度与高级技巧](05d-学习率调度与高级技巧.md)

**内容概要**：
- 学习率调度的重要性
- 常用策略：Step Decay、Cosine Annealing、Warmup
- Reduce on Plateau 与 Cyclic LR
- 二阶优化方法：牛顿法、BFGS、L-BFGS
- 约束优化与投影梯度下降
- 梯度裁剪、学习率查找器

**核心概念**：
| 策略 | 公式/方法 | 适用场景 |
|------|----------|----------|
| Step Decay | $\eta \cdot \gamma^{\lfloor t/T \rfloor}$ | 通用 |
| Cosine Annealing | $\eta_{\min} + \frac{1}{2}(\eta_0-\eta_{\min})(1+\cos(\pi t/T))$ | 通用 |
| Warmup + Cosine | 先线性增加再余弦退火 | Transformer |

**[开始学习 →](05d-学习率调度与高级技巧.md)**

---

## 学习路径

```
┌─────────────────────────────────────────────────────────────┐
│                     第五章：最优化方法                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  5.1 优化基础 ──→ 5.2 动量法 ──→ 5.3 自适应 ──→ 5.4 学习率调度  │
│  与梯度下降      与加速技术      学习率方法      与高级技巧       │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ 凸优化   │   │ Momentum │   │ AdaGrad  │   │ Cosine   │ │
│  │ 梯度下降 │   │ NAG      │   │ RMSprop  │   │ Warmup   │ │
│  │ 收敛分析 │   │ 收敛率   │   │ Adam     │   │ 二阶方法  │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
│                                                             │
│  应用: SGD, Adam, AdamW, 学习率调度, 梯度裁剪                  │
└─────────────────────────────────────────────────────────────┘
```

## 为什么最优化对深度学习重要？

### 1. 训练即优化

```
训练神经网络 = 寻找最优参数
    ↓
最小化损失函数: min_θ L(θ)
    ↓
使用优化算法迭代更新参数
```

### 2. 优化器的选择影响

| 方面 | 优化器的影响 |
|------|-------------|
| 收敛速度 | Adam 通常比 SGD 快 |
| 泛化性能 | SGD + Momentum 可能更好 |
| 稳定性 | 自适应方法更稳定 |
| 内存占用 | SGD 最小，Adam 需要 2x |

### 3. 学习率调度的必要性

- **训练初期**：需要较大学习率快速下降
- **训练中期**：需要稳定学习
- **训练后期**：需要小学习率精细调优

---

## 核心公式速查

### 梯度下降

$$
\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)
$$

### 动量法

$$
\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\mathbf{x}_t)
$$

### Adam

$$
\mathbf{m}_{t+1} = \beta_1 \mathbf{m}_t + (1-\beta_1) \mathbf{g}_t
$$

$$
\mathbf{v}_{t+1} = \beta_2 \mathbf{v}_t + (1-\beta_2) \mathbf{g}_t^2
$$

### 余弦退火

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\pi t/T))
$$

---

## Python 代码示例

### 梯度下降

```python
import numpy as np

def gradient_descent(f, grad_f, x0, lr=0.01, max_iters=1000):
    x = x0.copy()
    for _ in range(max_iters):
        x = x - lr * grad_f(x)
    return x

# 示例
f = lambda x: x[0]**2 + 2*x[1]**2
grad_f = lambda x: np.array([2*x[0], 4*x[1]])
x0 = np.array([2.0, 2.0])
x_opt = gradient_descent(f, grad_f, x0, lr=0.1)
print(f"最优解: {x_opt}")
```

### Adam 优化器

```python
class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999)):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
    
    def step(self, params, grads):
        self.t += 1
        for key in params:
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key]**2
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-8)
```

### 余弦退火调度器

```python
class CosineAnnealingLR:
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_max = optimizer.lr
        self.epoch = 0
    
    def step(self):
        self.epoch += 1
        self.optimizer.lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
                           (1 + np.cos(np.pi * self.epoch / self.T_max))
```

---

## 学习建议

1. **从基础开始**：先理解梯度下降，再学习高级方法
2. **动手实现**：自己实现 SGD、Momentum、Adam
3. **实验比较**：在同一问题上比较不同优化器
4. **理解权衡**：收敛速度 vs 泛化性能
5. **关注学习率**：这是最重要的超参数

---

## 延伸阅读

- [第四章：数理统计](04-statistics.md) - 参数估计与假设检验
- [第六章：初等函数](06-elementary-functions.md) - 激活函数与损失函数

---

**返回**：[数学基础教程目录](../math-fundamentals.md)

