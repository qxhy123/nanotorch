# 第五章（b）：动量法与加速技术

梯度下降虽然简单有效，但在某些情况下收敛较慢。本节将介绍动量法和加速技术，这些方法通过利用历史梯度信息来加速收敛，特别是在狭长峡谷（高条件数）的情况下表现优异。

---

## 目录

1. [动量法 (Momentum)](#动量法-momentum)
2. [Nesterov 加速梯度 (NAG)](#nesterov-加速梯度-nag)
3. [收敛率分析](#收敛率分析)
4. [加速技术的直观理解](#加速技术的直观理解)
5. [在深度学习中的应用](#在深度学习中的应用)

---

## 动量法 (Momentum)

### 🎯 生活类比：滚下山坡的球

想象你在山坡上放下一个实心铁球：
- **没有动量**（普通梯度下降）：球每一步都只看脚下，哪里最陡就往哪滚一小步
- **有动量**：球有惯性！它会记住之前的速度，累积"冲劲"

**效果对比**：
- **普通方法**：在峡谷里来回震荡，像醉酒的人
- **动量法**：像滚下山的球，有惯性，能冲过小坑，沿着主要方向加速

**具体场景**：想象一个V形的山谷
- 普通方法：左右震荡，慢慢往下爬
- 动量法：左右晃动会相互抵消，主要往下冲！

### 动机

标准梯度下降的问题：
1. **震荡**：在高条件数问题中，梯度方向可能指向错误的方向
2. **慢收敛**：在平坦区域移动缓慢
3. **陷入局部最优**：缺乏"冲劲"逃离浅层局部最优

### 物理类比

想象一个球从山上滚下来：
- **速度**：累积动能，具有惯性
- **动量**：当前速度 + 历史速度的影响
- **效果**：能够冲过小的凹陷，沿主要方向加速

### 动量法公式

**标准动量法**（PyTorch 风格）：

$$
\begin{align}
\mathbf{v}_{t+1} &= \beta \mathbf{v}_t + \nabla f(\mathbf{x}_t) \\
\mathbf{x}_{t+1} &= \mathbf{x}_t - \eta \mathbf{v}_{t+1}
\end{align}
$$

其中：
- $\beta \in [0, 1)$ 是**动量系数**（通常取 0.9）
- $\mathbf{v}_t$ 是**速度**（累积梯度）

**展开形式**：

$$
\mathbf{v}_t = \sum_{k=0}^{t-1} \beta^{t-1-k} \nabla f(\mathbf{x}_k)
$$

### 动量系数的作用

| $\beta$ 值 | 效果 |
|-----------|------|
| $\beta = 0$ | 退化为标准 GD |
| $\beta = 0.5$ | 弱动量，历史影响较小 |
| $\beta = 0.9$ | 标准动量（推荐） |
| $\beta = 0.99$ | 强动量，可能过度平滑 |

### 指数移动平均解释

动量本质上是梯度的**指数移动平均**（EMA）：

$$
\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1-\beta) \nabla f(\mathbf{x}_t)
$$

有效窗口大小约为 $\frac{1}{1-\beta}$：
- $\beta = 0.9$：约 10 步历史的平均
- $\beta = 0.99$：约 100 步历史的平均

### 动量法的优势

1. **加速收敛**：在一致方向上累积速度
2. **抑制震荡**：在震荡方向上相互抵消
3. **逃离局部最优**：利用惯性冲过浅层凹陷

```python
import numpy as np

def sgd_with_momentum(grad_f, x0, learning_rate=0.01, momentum=0.9, 
                       max_iters=1000, tol=1e-6):
    """
    带动量的随机梯度下降
    
    Args:
        grad_f: 梯度函数
        x0: 初始点
        learning_rate: 学习率
        momentum: 动量系数
        max_iters: 最大迭代次数
        tol: 收敛容差
    """
    x = x0.copy()
    v = np.zeros_like(x)  # 速度
    history = {'x': [x.copy()], 'f': [], 'grad_norm': []}
    
    for i in range(max_iters):
        g = grad_f(x)
        grad_norm = np.linalg.norm(g)
        history['grad_norm'].append(grad_norm)
        
        if grad_norm < tol:
            print(f"在 {i} 次迭代后收敛")
            break
        
        # 动量更新
        v = momentum * v + g
        x = x - learning_rate * v
        
        history['x'].append(x.copy())
    
    return x, history

# 示例：Rosenbrock 函数（经典的测试函数）
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x):
    return np.array([
        -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0]),
        200 * (x[1] - x[0]**2)
    ])

x0 = np.array([-1.0, 1.0])
x_opt, history = sgd_with_momentum(rosenbrock_grad, x0, learning_rate=0.001, momentum=0.9)
print(f"最优解: {x_opt}")
print(f"迭代次数: {len(history['x'])}")
```

---

## Nesterov 加速梯度 (NAG)

### 动机

标准动量法在当前位置计算梯度，但此时已经积累了速度。Nesterov 提出：**在"未来位置"计算梯度**。

### NAG 公式

$$
\begin{align}
\mathbf{x}_{t+1/2} &= \mathbf{x}_t - \eta \beta \mathbf{v}_t \quad \text{（预测位置）} \\
\mathbf{v}_{t+1} &= \beta \mathbf{v}_t + \nabla f(\mathbf{x}_{t+1/2}) \\
\mathbf{x}_{t+1} &= \mathbf{x}_t - \eta \mathbf{v}_{t+1}
\end{align}
$$

**简化形式**：

$$
\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \nabla f(\mathbf{x}_t - \eta \beta \mathbf{v}_t)
$$

### NAG vs 标准动量

```
标准动量：
  当前位置 → 计算梯度 → 应用速度
  
NAG：
  当前位置 → 预测位置 → 计算梯度 → 应用速度
       ↓           ↑
       └───────────┘
         前瞻一步
```

### NAG 的优势

1. **更"聪明"的更新**：在预测位置评估梯度
2. **提前减速**：接近最优时能更快减速
3. **理论保证**：对于凸函数，达到 $O(1/t^2)$ 收敛率

### NAG 可视化

```python
def nesterov_accelerated_gradient(f, grad_f, x0, learning_rate=0.01, 
                                    momentum=0.9, max_iters=1000, tol=1e-6):
    """
    Nesterov 加速梯度
    
    Args:
        f: 目标函数
        grad_f: 梯度函数
        x0: 初始点
        learning_rate: 学习率
        momentum: 动量系数
        max_iters: 最大迭代次数
        tol: 收敛容差
    """
    x = x0.copy()
    v = np.zeros_like(x)
    history = {'x': [x.copy()], 'f': [f(x)]}
    
    for i in range(max_iters):
        # 在"未来位置"计算梯度
        look_ahead = x - learning_rate * momentum * v
        g = grad_f(look_ahead)
        
        if np.linalg.norm(g) < tol:
            print(f"在 {i} 次迭代后收敛")
            break
        
        # NAG 更新
        v = momentum * v + g
        x = x - learning_rate * v
        
        history['x'].append(x.copy())
        history['f'].append(f(x))
    
    return x, history

def compare_momentum_methods():
    """比较标准动量和 NAG"""
    import matplotlib.pyplot as plt
    
    # 定义目标函数
    def f(x):
        return x[0]**2 + 10*x[1]**2
    
    def grad_f(x):
        return np.array([2*x[0], 20*x[1]])
    
    x0 = np.array([2.0, 2.0])
    
    # 标准动量
    _, hist_momentum = sgd_with_momentum(grad_f, x0, 0.05, 0.9, 100)
    
    # NAG
    _, hist_nag = nesterov_accelerated_gradient(f, grad_f, x0, 0.05, 0.9, 100)
    
    # 创建等高线图
    x1_range = np.linspace(-2.5, 2.5, 100)
    x2_range = np.linspace(-1.5, 2.5, 100)
    X1, X2 = np.meshgrid(x1_range, x2_range)
    Z = X1**2 + 10*X2**2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax, traj, title in [
        (axes[0], hist_momentum['x'], '标准动量法'),
        (axes[1], hist_nag['x'], 'Nesterov 加速梯度')
    ]:
        ax.contour(X1, X2, Z, levels=20, cmap='viridis')
        trajectory = np.array(traj)
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'ro-', linewidth=1.5, markersize=3)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], color='green', s=100, zorder=5, label='起点')
        ax.scatter(0, 0, color='red', s=100, zorder=5, label='最优点')
        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()

compare_momentum_methods()
```

---

## 收敛率分析

### 收敛率比较

对于 $\mu$-强凸且 $L$-光滑的函数，条件数 $\kappa = L/\mu$：

| 方法 | 收敛率 | 达到 $\epsilon$ 精度所需迭代 |
|------|--------|------------------------------|
| 梯度下降 | $O((1-1/\kappa)^t)$ | $O(\kappa \log(1/\epsilon))$ |
| 标准动量 | $O((1-\sqrt{1/\kappa})^t)$ | $O(\sqrt{\kappa} \log(1/\epsilon))$ |
| NAG | $O((1-\sqrt{1/\kappa})^t)$ | $O(\sqrt{\kappa} \log(1/\epsilon))$ |

**关键改进**：动量法将依赖从 $\kappa$ 改进到 $\sqrt{\kappa}$！

### 动量法收敛性证明概要

**定理**：对于 $\mu$-强凸且 $L$-光滑的二次函数 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top\mathbf{Q}\mathbf{x}$，动量法在 $\eta = \frac{4}{(\sqrt{L}+\sqrt{\mu})^2}$，$\beta = \frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}}$ 时收敛最快。

**证明步骤**：

**第一步**：将动量法写成二阶递归形式。

动量法：
$$\mathbf{v}_{t+1} = \beta\mathbf{v}_t + \nabla f(\mathbf{x}_t)$$
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\mathbf{v}_{t+1}$$

代入 $\mathbf{v}_{t+1}$：
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta(\beta\mathbf{v}_t + \nabla f(\mathbf{x}_t))$$

利用 $\eta\mathbf{v}_t = \mathbf{x}_t - \mathbf{x}_{t-1}$：
$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\nabla f(\mathbf{x}_t) + \beta(\mathbf{x}_t - \mathbf{x}_{t-1})$$

**第二步**：对于二次函数 $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^\top\mathbf{Q}\mathbf{x}$，有 $\nabla f(\mathbf{x}) = \mathbf{Qx}$。

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta\mathbf{Qx}_t + \beta(\mathbf{x}_t - \mathbf{x}_{t-1})$$

**第三步**：分析特征方向。

设 $\mathbf{q}_i$ 是 $\mathbf{Q}$ 的特征向量，对应特征值 $\lambda_i \in [\mu, L]$。

在特征方向上的误差：$e_t^{(i)} = \mathbf{q}_i^\top(\mathbf{x}_t - \mathbf{x}^*)$

误差递推：
$$e_{t+1}^{(i)} = e_t^{(i)} - \eta\lambda_i e_t^{(i)} + \beta(e_t^{(i)} - e_{t-1}^{(i)})$$

**第四步**：分析特征方程。

这是一个二阶线性递推，特征方程为：
$$r^2 - (1 - \eta\lambda_i + \beta)r + \beta = 0$$

根为：
$$r = \frac{(1-\eta\lambda_i+\beta) \pm \sqrt{(1-\eta\lambda_i+\beta)^2 - 4\beta}}{2}$$

**第五步**：最优参数选择。

为使收敛最快，需要最小化 $\max_i |r_i|$。通过分析得到最优参数：
$$\eta^* = \frac{4}{(\sqrt{L}+\sqrt{\mu})^2}, \quad \beta^* = \frac{\sqrt{L}-\sqrt{\mu}}{\sqrt{L}+\sqrt{\mu}} = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$$

收敛率：
$$|r| \leq \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} = 1 - \frac{2}{\sqrt{\kappa}+1} \approx 1 - \frac{1}{\sqrt{\kappa}}$$

$$\boxed{\text{动量法收敛率} = O\left(\left(1 - \frac{1}{\sqrt{\kappa}}\right)^t\right)}$$

**对比**：梯度下降收敛率为 $O\left(\left(1 - \frac{1}{\kappa}\right)^t\right)$

当 $\kappa = 100$ 时：
- GD：$(0.99)^t$
- 动量：$(0.82)^t$

达到 $10^{-6}$ 精度：
- GD：约 1400 次迭代
- 动量：约 70 次迭代

### 凸函数（非强凸）的收敛率

| 方法 | 收敛率 |
|------|--------|
| 梯度下降 | $O(1/t)$ |
| 标准动量 | $O(1/t)$ |
| NAG | $O(1/t^2)$ |

**NAG 的独特优势**：在一般凸情况下达到 $O(1/t^2)$ 的收敛率。

### 数值示例

```python
def compare_convergence_rates():
    """比较不同方法的收敛率"""
    import matplotlib.pyplot as plt
    
    # 定义目标函数（高条件数）
    def f(x):
        return x[0]**2 + 100*x[1]**2
    
    def grad_f(x):
        return np.array([2*x[0], 200*x[1]])
    
    x0 = np.array([2.0, 2.0])
    
    # 标准梯度下降
    def gradient_descent(grad_f, x0, lr, max_iters=200):
        x = x0.copy()
        history = [f(x)]
        for _ in range(max_iters):
            x = x - lr * grad_f(x)
            history.append(f(x))
        return history
    
    # 标准动量
    _, hist_momentum = sgd_with_momentum(grad_f, x0, 0.01, 0.9, 200)
    
    # NAG
    _, hist_nag = nesterov_accelerated_gradient(f, grad_f, x0, 0.01, 0.9, 200)
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(gradient_descent(grad_f, x0, 0.01), label='梯度下降', linewidth=2)
    plt.plot([f(x) for x in hist_momentum['x']], label='标准动量', linewidth=2)
    plt.plot(hist_nag['f'], label='NAG', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('函数值')
    plt.title('收敛曲线（线性坐标）')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(gradient_descent(grad_f, x0, 0.01), label='梯度下降', linewidth=2)
    plt.semilogy([f(x) for x in hist_momentum['x']], label='标准动量', linewidth=2)
    plt.semilogy(hist_nag['f'], label='NAG', linewidth=2)
    plt.xlabel('迭代次数')
    plt.ylabel('函数值（对数）')
    plt.title('收敛曲线（对数坐标）')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

compare_convergence_rates()
```

---

## 加速技术的直观理解

### 为什么动量有效？

**1. 方向一致性**
- 如果连续梯度方向一致，动量会累积速度
- 如果梯度方向震荡，动量会相互抵消

**2. 震荡抑制**

```
无动量：                    有动量：
  ↑                          ↑
  │  ·  →  ·                 │    ·→→→·
  │    ↗                     │      ↓
  │  ·  ←  ·                 │    ·←←←·
                           
  震荡前进                   平滑前进
```

**3. 峡谷问题**

```
峡谷横截面：

        ╱╲
       ╱  ╲
      ╱    ╲    无动量：震荡下降
     ╱      ╲   有动量：平滑下降
    ╱        ╲
   ╱          ╲
  ╱            ╲
```

### NAG 的前瞻优势

NAG 在预测位置计算梯度，相当于"预判"：

```
当前位置：●
预测位置：  ○
真实梯度：    ↓
NAG 梯度：  ↘ （更准确的方向）
```

如果接近最优，NAG 能更早感知到并减速。

### 动量参数选择指南

| 场景 | 推荐 $\beta$ | 说明 |
|------|--------------|------|
| 一般训练 | 0.9 | 平衡稳定性和速度 |
| 深层网络 | 0.9-0.99 | 更强的累积效应 |
| 嘈杂梯度 | 0.99 | 更强的平滑 |
| 精细调优 | 0.5-0.7 | 更快的响应 |

---

## 在深度学习中的应用

### PyTorch 风格实现

```python
class SGDMomentum:
    """带动量的 SGD 优化器（PyTorch 风格）"""
    
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {k: np.zeros_like(v) for k, v in params.items()}
    
    def step(self, params, grads):
        """执行一步更新"""
        for key in params:
            # 更新速度
            self.velocities[key] = self.momentum * self.velocities[key] + grads[key]
            # 更新参数
            params[key] -= self.lr * self.velocities[key]
    
    def zero_grad(self):
        """清零梯度（占位）"""
        pass


class NAG:
    """Nesterov 加速梯度优化器"""
    
    def __init__(self, params, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {k: np.zeros_like(v) for k, v in params.items()}
    
    def step(self, params, grads_func):
        """执行一步更新（需要梯度函数来计算预测位置的梯度）"""
        for key in params:
            # 计算预测位置
            look_ahead = params[key] - self.lr * self.momentum * self.velocities[key]
            # 在预测位置计算梯度（简化实现）
            # 实际中需要重新前向传播
            params[key] = look_ahead
        
        # 重新计算梯度（这里简化）
        grads = grads_func(params)
        
        for key in params:
            # 更新速度
            self.velocities[key] = self.momentum * self.velocities[key] + grads[key]
            # 从原始位置更新
            params[key] -= self.lr * self.velocities[key]
```

### nanotorch 中的使用

```python
from nanotorch.optim import SGD
from nanotorch import Tensor
from nanotorch.nn import Linear, ReLU, Sequential

# 创建模型
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# 使用带动量的 SGD
optimizer = SGD(
    model.parameters(), 
    lr=0.01, 
    momentum=0.9,
    nesterov=False  # 设为 True 使用 NAG
)

# 训练循环
for epoch in range(100):
    for batch_x, batch_y in train_loader:
        # 前向传播
        output = model(Tensor(batch_x))
        loss = criterion(output, Tensor(batch_y))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 参数更新
        optimizer.step()
```

### 实践建议

1. **默认使用动量**
   - 几乎总是比纯 SGD 更好
   - $\beta = 0.9$ 是安全的默认值

2. **NAG vs 标准动量**
   - NAG 理论收敛更快
   - 实践中差异可能不明显
   - 建议都尝试，选择效果更好的

3. **与学习率调度配合**
   - 动量不影响学习率调度策略
   - 可以与 StepLR、CosineAnnealing 等配合使用

4. **监控训练曲线**
   - 梯度范数突然增大可能需要降低学习率
   - 训练震荡可能需要增大动量或减小学习率

---

## 小结

本节介绍了动量法和加速技术：

| 方法 | 公式 | 收敛率（强凸） | 特点 |
|------|------|----------------|------|
| 梯度下降 | $x_{t+1} = x_t - \eta \nabla f$ | $O((1-1/\kappa)^t)$ | 简单、慢 |
| 标准动量 | $v_{t+1} = \beta v_t + \nabla f$ | $O((1-\sqrt{1/\kappa})^t)$ | 稳定、快 |
| NAG | $v_{t+1} = \beta v_t + \nabla f(x_t - \eta \beta v_t)$ | $O((1-\sqrt{1/\kappa})^t)$ | 前瞻、更快 |

**关键要点**：
- 动量通过累积历史梯度信息加速收敛
- 动量在高条件数问题上效果显著
- NAG 通过前瞻机制进一步改进
- $\beta = 0.9$ 是常用的默认值

---

**上一节**：[优化基础与梯度下降](05a-优化基础与梯度下降.md)

**下一节**：[自适应学习率方法](05c-自适应学习率方法.md) - 学习 AdaGrad、RMSprop、Adam 等自适应优化算法。

**返回**：[第五章：最优化方法](05-optimization.md) | [数学基础教程目录](../math-fundamentals.md)
