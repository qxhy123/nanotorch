# 第六章（c）：ReLU 函数族与激活函数

ReLU（Rectified Linear Unit）是深度学习中最常用的激活函数。它简单、高效，并且有效地解决了梯度消失问题。本节将系统介绍 ReLU 及其变体，以及如何选择合适的激活函数。

---

## 目录

1. [ReLU 函数](#relu-函数)
2. [Leaky ReLU 与 PReLU](#leaky-relu-与-prelu)
3. [ELU 与 SELU](#elu-与-selu)
4. [GELU](#gelu)
5. [激活函数选择指南](#激活函数选择指南)
6. [在深度学习中的应用](#在深度学习中的应用)

---

## ReLU 函数

### 🎯 生活类比：音量控制

想象你有一个音量调节器：
- **正数输入**：正常放大（输入3 → 输出3）
- **负数输入**：静音（输入-5 → 输出0）

**ReLU 就像一个"负数静音器"**：
- 把所有负数变成0
- 正数保持不变

```
输入 x     ReLU(x)
───────────────────
-5         0       ← 负数被"静音"
-2         0       ← 负数被"静音"
 0         0       ← 0也是0
 1         1       ← 正数保持
 3         3       ← 正数保持
10        10       ← 正数保持
```

### 📖 为什么 ReLU 这么好用？

1. **计算超级快**：只需要判断 x > 0，不需要指数运算
2. **梯度不消失**（正区间）：梯度恒等于1，不像Sigmoid那样最大只有0.25
3. **稀疏激活**：很多神经元输出0，减少计算量

### ⚠️ ReLU 的问题：死亡神经元

**问题**：如果神经元总是收到负数输入：
- 输出永远是0
- 梯度永远是0
- 参数永远不更新
- 神经元"死亡"了！

**类比**：就像一个学生考试总是不及格（负数），老师就不给他任何反馈（梯度=0），他永远学不会。

### 定义

**ReLU**（Rectified Linear Unit）：

$$
\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

### 核心性质

1. **导数**：

$$
\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x \leq 0 \end{cases}
$$

2. **非负性**：输出 $\geq 0$

3. **稀疏激活**：输入为负时输出为 0，产生稀疏表示

4. **计算高效**：只需比较和 max 操作，无指数运算

### ReLU 的优势

| 优势 | 说明 |
|------|------|
| 计算简单 | 只需要 max 操作 |
| 梯度不消失（正区间） | 梯度恒为 1 |
| 稀疏激活 | 负输入被抑制 |
| 收敛快 | 梯度恒定，无饱和 |

### ReLU 的问题：死亡 ReLU

**死亡 ReLU 问题**：如果神经元的输入总是负数，该神经元将永远输出 0，梯度永远为 0，参数永远无法更新。

**原因**：
- 学习率过大
- 权重初始化不当
- 数据分布问题

```python
import numpy as np

def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU 的导数"""
    return (x > 0).astype(float)

# 示例
x = np.array([-2, -1, 0, 1, 2])
print(f"ReLU: {relu(x)}")
print(f"导数: {relu_derivative(x)}")
```

---

## Leaky ReLU 与 PReLU

### 🎯 生活类比：给"死亡"神经元一根救命稻草

**Leaky ReLU** 解决了 ReLU 的"死亡"问题：
- 正数：正常通过（斜率=1）
- 负数：给一点点"漏气"（斜率=0.01）

**类比**：考试不及格的学生，虽然分数很低（负数），但还是能得到一点点反馈（0.01倍），这样他还有学习的机会！

```
输入 x     ReLU(x)     LeakyReLU(x, 0.1)
────────────────────────────────────────
-5         0           -0.5    ← 给点"漏气"
-2         0           -0.2    ← 给点"漏气"
 0         0            0
 2         2            2
 5         5            5
```

### Leaky ReLU

**定义**：在负区间给一个小的斜率 $\alpha$（通常 $\alpha = 0.01$）：

$$
\text{LeakyReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}
$$

**导数**：

$$
\text{LeakyReLU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}
$$

**优点**：避免"死亡 ReLU"问题，负区间仍有梯度。

```python
def leaky_relu(x, alpha=0.01):
    """Leaky ReLU"""
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    """Leaky ReLU 的导数"""
    return np.where(x > 0, 1, alpha)
```

### PReLU (Parametric ReLU)

**定义**：$\alpha$ 是可学习参数：

$$
\text{PReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}
$$

**优点**：
- 自适应学习最优的负区间斜率
- 在大数据集上效果好

**缺点**：
- 可能过拟合（增加了参数）
- 小数据集上不如 Leaky ReLU 稳定

```python
class PReLU:
    """PReLU 激活函数"""
    
    def __init__(self, shape, alpha=0.25):
        self.alpha = np.full(shape, alpha)  # 可学习参数
    
    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)
    
    def backward(self, x, grad_output):
        # 对 x 的梯度
        grad_x = np.where(x > 0, grad_output, grad_output * self.alpha)
        
        # 对 alpha 的梯度
        grad_alpha = np.where(x <= 0, x * grad_output, 0).sum(axis=0)
        
        return grad_x, grad_alpha
```

---

## ELU 与 SELU

### ELU (Exponential Linear Unit)

**定义**：

$$
\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}
$$

其中 $\alpha > 0$（通常取 1.0）。

**导数**：

$$
\text{ELU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha e^x = \text{ELU}(x) + \alpha & x \leq 0 \end{cases}
$$

**优点**：
- 输出均值接近 0（自归一化）
- 负区间平滑（不同于 ReLU 的折点）
- 对噪声更鲁棒

**缺点**：
- 负区间计算指数，稍慢

```python
def elu(x, alpha=1.0):
    """ELU 激活函数"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    """ELU 的导数"""
    return np.where(x > 0, 1, elu(x, alpha) + alpha)
```

### SELU (Scaled ELU)

**定义**：在 ELU 基础上添加缩放因子：

$$
\text{SELU}(x) = \lambda \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}
$$

其中：
- $\alpha \approx 1.6732632423543772$
- $\lambda \approx 1.0507009873554805$

**自归一化性质**：配合特定的初始化（LeCun Normal），可使网络各层输出自动归一化到均值 0、方差 1。

```python
def selu(x, alpha=1.6732632423543772, scale=1.0507009873554805):
    """SELU 激活函数"""
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))
```

---

## GELU

### 定义

**GELU**（Gaussian Error Linear Unit）：

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(Z \leq x)
$$

其中 $\Phi(x)$ 是标准正态分布的 CDF。

### 近似公式

**Tanh 近似**（最常用）：

$$
\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right]\right)
$$

**Sigmoid 近似**：

$$
\text{GELU}(x) \approx x \cdot \sigma(1.702x)
$$

**GELU Tanh 近似公式的推导思路**：

**第一步**：从精确形式出发。

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

其中 $\text{erf}(x) = \frac{2}{\sqrt{\pi}}\int_0^x e^{-t^2} dt$

**第二步**：利用 $\tanh$ 近似 $\text{erf}$。

已知 $\text{erf}(x) \approx \tanh(ax + bx^3)$，需要确定 $a, b$。

通过 Taylor 展开比较：

$$\text{erf}(x) = \frac{2}{\sqrt{\pi}}\left(x - \frac{x^3}{3} + \frac{x^5}{10} - \cdots\right)$$

$$\tanh(ax + bx^3) = ax + bx^3 - \frac{a^3x^3}{3} + O(x^5)$$

**第三步**：匹配系数。

通过最小化近似误差，得到最优参数：

$$a = \sqrt{\frac{2}{\pi}} \approx 0.7979$$

$$b = 0.044715$$

**第四步**：代入得到最终公式。

$$\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)$$

$$\boxed{\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}\left(x + 0.044715 x^3\right)\right]\right)}$$

**Sigmoid 近似的推导**：

由于 $\tanh(x) = 2\sigma(2x) - 1$，可以转换为 Sigmoid 形式：

$$\text{GELU}(x) \approx x \cdot \sigma(1.702x)$$

这个近似精度略低于 Tanh 近似，但计算更简单。

### 精确计算

使用误差函数 erf：

$$
\text{GELU}(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

### GELU 的性质

- **非单调**：存在一个小的负区域
- **平滑**：处处可微
- **零中心**：输出均值接近 0
- **渐进行为**：$x \to +\infty$ 时趋近于 $x$，$x \to -\infty$ 时趋近于 0

### 为什么 Transformer 用 GELU？

1. **平滑性**：没有 ReLU 的折点
2. **非单调性**：在某些区域允许负值通过
3. **概率解释**：可以看作随机正则化的期望

```python
def gelu(x):
    """GELU 激活函数（Tanh 近似）"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_exact(x):
    """GELU 激活函数（精确，使用 erf）"""
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_sigmoid_approx(x):
    """GELU Sigmoid 近似"""
    return x * (1 / (1 + np.exp(-1.702 * x)))

# 比较
x = np.linspace(-3, 3, 1000)
print(f"GELU 近似与精确的最大差异: {np.max(np.abs(gelu(x) - gelu_exact(x))):.6f}")
```

---

## 激活函数选择指南

### ReLU 函数族比较

| 激活函数 | 公式 | 优点 | 缺点 | 适用场景 |
|----------|------|------|------|----------|
| ReLU | $\max(0, x)$ | 简单、快速 | 死亡 ReLU | 通用 |
| LeakyReLU | $\max(\alpha x, x)$ | 无死亡问题 | 超参数 $\alpha$ | 通用 |
| PReLU | $\max(\alpha x, x)$, $\alpha$ 可学习 | 自适应 | 过拟合风险 | 大数据 |
| ELU | $x$ if $x>0$ else $\alpha(e^x-1)$ | 自归一化 | 指数计算 | 深层网络 |
| SELU | Scaled ELU | 自归一化 | 需要特定初始化 | 自归一化网络 |
| GELU | $x \cdot \Phi(x)$ | 平滑、非单调 | 计算稍慢 | Transformer |

### 可视化比较

```python
import matplotlib.pyplot as plt

x = np.linspace(-3, 3, 1000)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 激活函数
axes[0].plot(x, relu(x), label='ReLU', linewidth=2)
axes[0].plot(x, leaky_relu(x, 0.1), label='LeakyReLU (α=0.1)', linewidth=2)
axes[0].plot(x, elu(x), label='ELU', linewidth=2)
axes[0].plot(x, gelu(x), label='GELU', linewidth=2)

axes[0].set_xlabel('x')
axes[0].set_ylabel('f(x)')
axes[0].set_title('激活函数比较')
axes[0].legend()
axes[0].grid(True)
axes[0].set_xlim([-3, 3])
axes[0].set_ylim([-1, 3])

# 导数
axes[1].plot(x, relu_derivative(x), label="ReLU'", linewidth=2)
axes[1].plot(x, leaky_relu_derivative(x, 0.1), label="LeakyReLU'", linewidth=2)
axes[1].plot(x, elu_derivative(x), label="ELU'", linewidth=2)

axes[1].set_xlabel('x')
axes[1].set_ylabel("f'(x)")
axes[1].set_title('激活函数导数比较')
axes[1].legend()
axes[1].grid(True)
axes[1].set_xlim([-3, 3])

plt.tight_layout()
plt.show()
```

### 选择建议

| 场景 | 推荐激活函数 | 原因 |
|------|--------------|------|
| 隐藏层（通用） | ReLU / GELU | 简单有效 |
| 隐藏层（深层） | ELU / SELU | 自归一化 |
| Transformer | GELU | 平滑、非单调 |
| CNN | ReLU / LeakyReLU | 计算效率 |
| RNN | Tanh | 零中心 |
| 输出层（二分类） | Sigmoid | 输出概率 |
| 输出层（多分类） | Softmax | 输出概率分布 |
| 输出层（回归） | 无 / Linear | 无限制 |

---

## 在深度学习中的应用

### nanotorch 中的激活函数

```python
from nanotorch.nn import ReLU, LeakyReLU, GELU, PReLU, ELU

# ReLU
relu = ReLU()
x = Tensor.randn((32, 128))
y = relu(x)

# LeakyReLU
leaky_relu = LeakyReLU(negative_slope=0.01)
y = leaky_relu(x)

# GELU
gelu = GELU()
y = gelu(x)

# PReLU
prelu = PReLU(num_parameters=128)  # 每个通道一个参数
y = prelu(x)
```

### 完整的神经网络示例

```python
from nanotorch import Tensor
from nanotorch.nn import Linear, ReLU, GELU, BatchNorm1d, Dropout, Sequential

# 使用 ReLU 的网络
model_relu = Sequential(
    Linear(784, 256),
    BatchNorm1d(256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 128),
    BatchNorm1d(128),
    ReLU(),
    Linear(128, 10)
)

# 使用 GELU 的网络（类似 Transformer）
model_gelu = Sequential(
    Linear(784, 256),
    BatchNorm1d(256),
    GELU(),
    Dropout(0.1),
    Linear(256, 128),
    BatchNorm1d(128),
    GELU(),
    Linear(128, 10)
)
```

### 处理死亡 ReLU

```python
def check_dead_neurons(model, x):
    """检查死亡神经元的比例"""
    activations = []
    
    def hook(module, input, output):
        activations.append(output)
    
    # 注册 hook
    for name, module in model.named_modules():
        if isinstance(module, ReLU):
            module.register_forward_hook(hook)
    
    # 前向传播
    model(x)
    
    # 检查死亡神经元
    for i, act in enumerate(activations):
        dead_ratio = (act == 0).mean().item()
        print(f"Layer {i}: 死亡神经元比例 = {dead_ratio:.2%}")

# 解决方法
# 1. 使用 LeakyReLU
# 2. 降低学习率
# 3. 使用更好的初始化（He 初始化）
# 4. 使用 BatchNorm
```

---

## 小结

本节介绍了 ReLU 函数族和激活函数的选择：

| 函数 | 核心特点 | 推荐场景 |
|------|----------|----------|
| ReLU | 简单、高效 | CNN、通用 |
| LeakyReLU | 无死亡问题 | 通用 |
| ELU/SELU | 自归一化 | 深层网络 |
| GELU | 平滑、非单调 | Transformer |

**关键要点**：
- ReLU 是最常用的激活函数
- LeakyReLU 解决死亡 ReLU 问题
- GELU 在 Transformer 中表现优异
- 激活函数选择影响训练效率和模型性能

---

**上一节**：[Sigmoid 与 Softmax 函数](06b-Sigmoid与Softmax函数.md)

**下一节**：[损失函数与归一化](06d-损失函数与归一化.md) - 学习 MSE、交叉熵等损失函数和 BatchNorm、LayerNorm 等归一化技术。

**返回**：[第六章：初等函数](06-elementary-functions.md) | [数学基础教程目录](../math-fundamentals.md)
