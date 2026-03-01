# 第六章：优化器

优化器使用梯度更新模型参数，是最小化损失函数的关键。

## 6.1 梯度下降基础

```
θ_new = θ_old - lr * ∂L/∂θ
```

- `θ`：模型参数
- `lr`：学习率
- `∂L/∂θ`：梯度

## 6.2 Optimizer 基类

```python
# optimizer.py
from typing import List, Dict, Any
from nanotorch.tensor import Tensor

class Optimizer:
    """所有优化器的基类"""
    
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = list(params)
        self.lr = lr
        self.param_groups: List[Dict[str, Any]] = []
        
        # 默认参数组
        self.param_groups.append({
            'params': self.params,
            'lr': lr
        })
    
    def zero_grad(self) -> None:
        """清零所有参数的梯度"""
        for param in self.params:
            if param.grad is not None:
                param.grad = None
    
    def step(self) -> None:
        """执行一步参数更新"""
        raise NotImplementedError("Subclasses must implement step()")
```

## 6.3 SGD 实现

```python
class SGD(Optimizer):
    """随机梯度下降
    
    支持动量 (momentum) 和 Nesterov 加速
    
    v = momentum * v + gradient
    θ = θ - lr * v
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
        
        # 动量缓存
        self.velocities = [np.zeros_like(p.data) for p in self.params]
    
    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue
            
            grad = param.grad.copy()
            
            # 权重衰减 (L2 正则化)
            if self.weight_decay != 0:
                grad = grad + self.weight_decay * param.data
            
            # 动量更新
            if self.momentum != 0:
                v = self.velocities[i]
                v = self.momentum * v + grad
                
                if self.nesterov:
                    grad = grad + self.momentum * v
                else:
                    grad = v
                
                self.velocities[i] = v
            
            # 参数更新
            param.data = param.data - self.lr * grad
```

### 动量的作用

```
无动量：每次更新方向可能大幅变化
有动量：积累历史梯度，平滑更新方向
```

## 6.4 Adam 实现

```python
class Adam(Optimizer):
    """Adam: Adaptive Moment Estimation
    
    m = β1 * m + (1 - β1) * grad     # 一阶矩估计
    v = β2 * v + (1 - β2) * grad²    # 二阶矩估计
    
    m_hat = m / (1 - β1^t)           # 偏差修正
    v_hat = v / (1 - β2^t)
    
    θ = θ - lr * m_hat / (√v_hat + ε)
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
        
        # 矩估计缓存
        self.m = [np.zeros_like(p.data) for p in self.params]  # 一阶矩
        self.v = [np.zeros_like(p.data) for p in self.params]  # 二阶矩
        self.t = 0  # 时间步
    
    def step(self) -> None:
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

1. **自适应学习率**：每个参数有不同的学习率
2. **动量**：加速收敛
3. **偏差修正**：解决初始化偏差问题

## 6.5 使用示例

```python
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU
from nanotorch.nn.loss import CrossEntropyLoss
from nanotorch.optim import SGD, Adam

# 模型
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# 优化器
optimizer = Adam(model.parameters(), lr=0.001)
criterion = CrossEntropyLoss()

# 训练循环
for epoch in range(100):
    # 前向传播
    X, y = get_batch()
    logits = model(X)
    loss = criterion(logits, y)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 参数更新
    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## 6.6 学习率调度器

```python
class StepLR:
    """每 step_size 个 epoch，学习率乘以 gamma"""
    
    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0
    
    def step(self) -> None:
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma


# 使用
optimizer = SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    train(...)
    scheduler.step()  # 更新学习率
```

## 6.7 优化器对比

| 优化器 | 特点 | 适用场景 |
|--------|------|----------|
| SGD | 简单，需要调参 | 凸优化 |
| SGD+Momentum | 加速收敛 | 深度网络 |
| Adam | 自适应学习率 | 通用默认选择 |
| AdamW | 解耦权重衰减 | Transformer |
| RMSprop | 自适应学习率 | RNN |

## 6.8 练习

1. **实现 AdamW**：Adam + 解耦权重衰减

2. **实现 RMSprop**：`v = α*v + (1-α)*grad²`

3. **实现 CosineAnnealingLR** 调度器

## 下一章

下一章，我们将整合所有组件，实现完整的**训练循环**。

→ [第七章：训练循环](07-training.md)
