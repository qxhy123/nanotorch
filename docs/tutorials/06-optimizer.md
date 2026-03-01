# 第六章：优化器

## 在迷雾中，如何找到山谷...

你被困在一座雾蒙蒙的山上，能见度不足三米。

你看不见全貌，不知道山谷在哪个方向。你只能感知一件事：**脚下的坡度**。

坡度告诉你："这边上升最快。"那么反方向，就是下降最快的路。

于是你迈出一步。再感知坡度。再迈一步。一步，又一步...直到雾气散去，你发现自己已站在山脚。

```
梯度下降的哲学：

  我不知道山长什么样
  但我知道脚下的坡度
  沿着坡度反方向走
  总有一天，会到达山谷
```

**优化器，就是那个"下山策略"。** 它告诉你每一步该怎么走——步子多大、要不要记住之前的方向、遇到平地该怎么办。

有的优化器小心翼翼，步步为营（SGD）。有的优化器聪明伶俐，自适应调整步幅（Adam）。但它们的核心哲学都是同一个：

**跟着梯度，往山下走。**

---

## 6.1 梯度下降基础

### 核心公式

```
θ_new = θ_old - lr × ∂L/∂θ

参数   参数   学习率   梯度
  ↓      ↓       ↓       ↓
往哪走  现在   步子多大   哪边上升
```

### 生活类比

```
下山 = 梯度下降

1. 看脚下坡度（梯度）
2. 往坡度反方向走一步
3. 重复直到平地（损失最小）

学习率 = 步子大小
  太小 → 走太慢，要很久
  太大 → 可能跨过山谷，甚至走上去
```

---

## 6.2 Optimizer 基类

```python
# optimizer.py
from typing import List, Dict, Any
from nanotorch.tensor import Tensor

class Optimizer:
    """
    所有优化器的基类

    职责：用梯度更新参数
    """

    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr
        self.param_groups: List[Dict[str, Any]] = [{
            'params': self.params,
            'lr': lr
        }]

    def zero_grad(self) -> None:
        """
        清零所有参数的梯度

        必须在每次 backward() 之前调用！
        """
        for param in self.params:
            if param.grad is not None:
                param.grad = None

    def step(self) -> None:
        """
        执行一步参数更新

        子类必须实现
        """
        raise NotImplementedError("Subclasses must implement step()")
```

---

## 6.3 SGD：最朴素的走法

### 什么是 SGD？

```
SGD = Stochastic Gradient Descent（随机梯度下降）

最简单：直接沿着梯度反方向走

θ = θ - lr × gradient

"随机"的意思：用一小批数据估计梯度，而不是全部数据
```

### 带动量的 SGD

```
问题：普通 SGD 会震荡

     ╲   ╱╲   ╱
      ╲ ╱  ╲ ╱  ← 来回震荡
       ╲    ╱
        ╲  ╱
         ╲╱
          ↓ 应该直接往下

解决：动量（Momentum）

v = momentum × v + gradient
θ = θ - lr × v

类比：小球下坡
  - 有惯性，会记住之前的方向
  - 震荡会被"平滑"掉
```

### 实现

```python
class SGD(Optimizer):
    """
    随机梯度下降（带动量）

    v = momentum × v + gradient
    θ = θ - lr × v

    参数:
        params: 模型参数
        lr: 学习率
        momentum: 动量系数（0-1，常用0.9）
        weight_decay: L2正则化系数
        nesterov: 是否使用 Nesterov 加速
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.01,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False
    ):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # 动量缓存（记住历史方向）
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        """执行一步更新"""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # 权重衰减（L2 正则化）
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # 动量更新
            if self.momentum != 0:
                v = self.velocities[i]
                v = self.momentum * v + grad  # 累积历史梯度

                if self.nesterov:
                    # Nesterov：先"预走"一步再算梯度
                    grad = grad + self.momentum * v
                else:
                    grad = v

                self.velocities[i] = v

            # 参数更新
            param.data = param.data - self.lr * grad
```

### 动量效果对比

```
无动量：                    有动量：

    ↗️↘️↗️↘️                    ↘️
   ↗️  ↘️  ↗️                    ↘️
  ↗️    ↘️  ↗️                    ↘️
 ↗️      ↘️                       ↘️
↓          ↓                       ⬇️

震荡严重                    平滑收敛
```

---

## 6.4 Adam：自适应学习率

### 为什么 Adam 更好用？

```
SGD 的问题：所有参数用同样的学习率

问题场景：
  - 有的参数梯度大（应该走小步）
  - 有的参数梯度小（应该走大步）
  - 用同样的学习率，很难平衡

Adam 的解决：
  - 每个参数有自己的学习率
  - 根据历史梯度自动调整
```

### Adam 算法

```
1. 记录梯度的一阶矩（均值）
   m = β1 × m + (1-β1) × gradient

2. 记录梯度的二阶矩（方差）
   v = β2 × v + (1-β2) × gradient²

3. 偏差修正（解决初始化偏向0的问题）
   m_hat = m / (1 - β1^t)
   v_hat = v / (1 - β2^t)

4. 更新参数
   θ = θ - lr × m_hat / (√v_hat + ε)
```

### 实现

```python
class Adam(Optimizer):
    """
    Adam: Adaptive Moment Estimation

    结合了动量（一阶矩）和自适应学习率（二阶矩）

    默认参数通常不需要调整！
    lr=0.001, betas=(0.9, 0.999) 对大多数任务都好用
    """

    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # 一阶矩（梯度均值）
        self.m = [np.zeros_like(p.data) for p in self.params]
        # 二阶矩（梯度平方的均值）
        self.v = [np.zeros_like(p.data) for p in self.params]
        # 时间步
        self.t = 0

    def step(self) -> None:
        """执行一步更新"""
        self.t += 1

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad.copy()

            # 权重衰减
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data

            # 更新一阶矩
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad

            # 更新二阶矩
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            # 偏差修正
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # 参数更新
            param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
```

### Adam 为什么好用？

```
1. 自适应学习率
   - 梯度大的参数 → 学习率自动变小
   - 梯度小的参数 → 学习率自动变大

2. 动量加速
   - 一阶矩 m 记录梯度方向
   - 加速收敛

3. 偏差修正
   - 解决初始化时 m、v 偏向 0 的问题
   - 训练初期更稳定

4. 默认参数好用
   - lr=0.001, betas=(0.9, 0.999) 几乎万能
```

---

## 6.5 使用示例

```python
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU
from nanotorch.nn.loss import CrossEntropyLoss
from nanotorch.optim import SGD, Adam
import numpy as np

# 模型
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# 优化器（二选一）
optimizer = Adam(model.parameters(), lr=0.001)
# optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# 损失函数
criterion = CrossEntropyLoss()

# 训练循环
for epoch in range(100):
    # 获取数据
    X = Tensor.randn((32, 784))
    y = Tensor(np.random.randint(0, 10, 32))

    # 1. 前向传播
    logits = model(X)
    loss = criterion(logits, y)

    # 2. 清零梯度（重要！）
    optimizer.zero_grad()

    # 3. 反向传播
    loss.backward()

    # 4. 更新参数
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

---

## 6.6 学习率调度器

### 为什么需要调整学习率？

```
训练过程：
  初期：需要大学习率，快速接近最优
  后期：需要小学习率，精细调整

┌───────────────────────────────┐
│                               │
│    大步走 → 精细调整            │
│   ↘️                          │
│    ↘️                         │
│     ↘️  ↘️                    │
│       ↘️  ↓  ← 小步微调        │
│         ⬇️                    │
└───────────────────────────────┘
```

### StepLR 实现

```python
class StepLR:
    """每 step_size 个 epoch，学习率乘以 gamma"""

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self) -> None:
        """更新学习率"""
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
            print(f"Learning rate: {self.optimizer.lr}")


# 使用
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()  # 更新学习率

# 学习率变化：0.1 → 0.01 → 0.001
```

---

## 6.7 优化器对比

| 优化器 | 特点 | 适用场景 | 调参难度 |
|--------|------|---------|---------|
| SGD | 简单，需要调参 | 凸优化、精细调优 | 😫 难 |
| SGD+Momentum | 加速收敛 | 深度网络、CV | 😐 中等 |
| Adam | 自适应学习率 | **通用默认** | 😊 简单 |
| AdamW | 解耦权重衰减 | Transformer、NLP | 😊 简单 |
| RMSprop | 自适应学习率 | RNN | 😐 中等 |

### 选择建议

```
默认选择：Adam + lr=0.001

特殊情况：
  - CV 任务，追求极致精度 → SGD+Momentum
  - Transformer → AdamW
  - RNN → RMSprop 或 Adam
  - 不确定 → 用 Adam
```

---

## 6.8 常见陷阱

### 陷阱1：忘记 zero_grad()

```python
# 错误：梯度累积
for epoch in range(100):
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()  # 梯度累积！
    optimizer.step()  # 用的是所有历史的梯度之和

# 正确：每次迭代前清零
for epoch in range(100):
    optimizer.zero_grad()  # 清零！
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
```

### 陷阱2：学习率太大

```python
# 问题：loss 不下降或变成 NaN
optimizer = Adam(model.parameters(), lr=1.0)  # 太大！

# 解决：降低学习率
optimizer = Adam(model.parameters(), lr=0.001)  # 正常
```

### 陷阱3：学习率太小

```python
# 问题：训练太慢
optimizer = Adam(model.parameters(), lr=1e-6)  # 太小！

# 解决：增大学习率
optimizer = Adam(model.parameters(), lr=0.001)  # 正常
```

---

## 6.9 练习

### 基础练习

1. **实现 AdamW**：Adam + 解耦权重衰减
   ```python
   # weight_decay 在梯度计算之前应用
   param.data = param.data - lr * weight_decay * param.data
   # 然后再做 Adam 更新
   ```

2. **实现 RMSprop**：
   ```
   v = α × v + (1-α) × grad²
   θ = θ - lr × grad / (√v + ε)
   ```

### 进阶练习

3. **实现 CosineAnnealingLR**：学习率按余弦曲线变化

4. **实现 OneCycleLR**：先增大后减小学习率

---

## 一句话总结

| 概念 | 一句话 |
|------|--------|
| 优化器 | 用梯度更新参数的下山策略 |
| 学习率 | 步子大小，太大震荡太小慢 |
| SGD | 直接沿着梯度走，简单但慢 |
| Momentum | 记住历史方向，平滑震荡 |
| Adam | 自适应学习率，默认首选 |

---

## 下一章

现在我们有了优化器！

下一章，我们将整合所有组件，实现完整的**训练循环**。

→ [第七章：训练循环](07-training.md)

```python
# 预告：下一章你将实现
for epoch in range(epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    # 验证、保存模型、早停...
```
