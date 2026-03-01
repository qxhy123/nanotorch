# 教程 15：高级主题 (Advanced Topics)

## 目录

1. [概述](#概述)
2. [梯度裁剪](#梯度裁剪)
3. [学习率预热](#学习率预热)
4. [模型序列化](#模型序列化)
5. [梯度检查](#梯度检查)
6. [训练技巧](#训练技巧)
7. [调试技巧](#调试技巧)
8. [总结](#总结)

---

## 概述

本教程涵盖 nanotorch 中的一些高级功能和实用技巧：

- **梯度裁剪**：防止梯度爆炸
- **学习率预热**：稳定训练初期
- **模型序列化**：保存和加载模型
- **梯度检查**：验证自动微分实现
- **训练/调试技巧**：最佳实践

---

## 梯度裁剪

### 为什么需要梯度裁剪

在训练过程中，梯度可能变得非常大（梯度爆炸），导致：
- 参数更新过大，破坏已学习的特征
- 损失变为 NaN/Inf
- 训练不稳定

### 实现

```python
# nanotorch/utils.py

from typing import Iterable
import numpy as np
from nanotorch.tensor import Tensor

def clip_grad_norm_(
    parameters: Iterable[Tensor],
    max_norm: float,
    norm_type: float = 2.0
) -> float:
    """按范数裁剪梯度。
    
    如果梯度范数超过 max_norm，则按比例缩放梯度。
    
    Args:
        parameters: 参数迭代器
        max_norm: 最大范数
        norm_type: 范数类型 (1, 2, 或 inf)
    
    Returns:
        裁剪前的总梯度范数
    """
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad.flatten())
    
    if len(grads) == 0:
        return 0.0
    
    total_grad = np.concatenate(grads)
    
    if norm_type == float('inf'):
        total_norm = np.abs(total_grad).max()
    else:
        total_norm = np.linalg.norm(total_grad, ord=norm_type)
    
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        for param in parameters:
            if param.grad is not None:
                param.grad = param.grad * clip_coef
    
    return float(total_norm)


def clip_grad_value_(
    parameters: Iterable[Tensor],
    clip_value: float
) -> None:
    """按值裁剪梯度。"""
    for param in parameters:
        if param.grad is not None:
            param.grad = np.clip(param.grad, -clip_value, clip_value)


def get_grad_norm_(
    parameters: Iterable[Tensor],
    norm_type: float = 2.0
) -> float:
    """获取梯度范数（不裁剪）。"""
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad.flatten())
    
    if len(grads) == 0:
        return 0.0
    
    total_grad = np.concatenate(grads)
    
    if norm_type == float('inf'):
        return float(np.abs(total_grad).max())
    else:
        return float(np.linalg.norm(total_grad, ord=norm_type))
```

### 使用示例

```python
from nanotorch.utils import clip_grad_norm_, get_grad_norm_

# 训练循环
for x, y in dataloader:
    optimizer.zero_grad()
    
    output = model(Tensor(x))
    loss = criterion(output, Tensor(y))
    loss.backward()
    
    # 梯度裁剪
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
    print(f"Gradient norm: {grad_norm:.4f}")
    
    optimizer.step()
```

---

## 学习率预热

### 为什么需要预热

在训练初期：
- 模型参数是随机初始化的
- 大的学习率可能导致训练不稳定
- 预热（Warmup）逐渐增加学习率，稳定训练初期

### nanotorch 提供的预热调度器

```python
# nanotorch/optim/lr_scheduler.py

class LinearWarmup:
    """线性预热调度器。"""
    
    def __init__(self, optimizer, warmup_epochs: int, start_lr: float = 0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = optimizer.lr
        self.current_epoch = 0
    
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            alpha = self.current_epoch / self.warmup_epochs
            lr = self.start_lr + alpha * (self.target_lr - self.start_lr)
            self.optimizer.lr = lr
        self.current_epoch += 1


class CosineWarmupScheduler:
    """余弦预热 + 余弦衰减。"""
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        max_epochs: int,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.current_epoch = 0
    
    def step(self):
        import math
        if self.current_epoch < self.warmup_epochs:
            alpha = self.current_epoch / self.warmup_epochs
            lr = self.min_lr + alpha * (self.base_lr - self.min_lr)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        
        self.optimizer.lr = lr
        self.current_epoch += 1
```

### 使用示例

```python
from nanotorch.optim import AdamW, CosineWarmupScheduler

optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = CosineWarmupScheduler(
    optimizer,
    warmup_epochs=5,
    max_epochs=100,
    min_lr=1e-6
)

for epoch in range(100):
    train_one_epoch(model, dataloader, optimizer)
    scheduler.step()
    print(f"Epoch {epoch}, LR: {optimizer.lr:.6f}")
```

---

## 模型序列化

### 保存和加载模型

```python
# 保存模型
state_dict = model.state_dict()
np.savez('model.npz', **state_dict)

# 加载模型
state_dict = dict(np.load('model.npz'))
model.load_state_dict(state_dict)
```

### 保存训练检查点

```python
def save_checkpoint(model, optimizer, epoch, loss, path):
    """保存完整的训练检查点。"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    np.savez(path, **checkpoint)

def load_checkpoint(model, path):
    """加载训练检查点。"""
    checkpoint = dict(np.load(path, allow_pickle=True))
    model.load_state_dict(checkpoint['model_state_dict'].item())
    return checkpoint['epoch'], checkpoint['loss']
```

---

## 梯度检查

### 原理

使用有限差分法验证自动微分的正确性：

```
数值梯度 ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
```

### 实现

```python
def gradient_check(
    func: callable,
    inputs: list,
    eps: float = 1e-5,
    atol: float = 1e-4,
    rtol: float = 1e-3
) -> bool:
    """使用有限差分验证梯度。"""
    # 计算解析梯度
    loss = func(inputs)
    loss.backward()
    
    analytic_grads = [inp.grad.copy() if inp.grad is not None else None for inp in inputs]
    
    # 计算数值梯度
    for i, inp in enumerate(inputs):
        if analytic_grads[i] is None:
            continue
        
        numerical_grad = np.zeros_like(inp.data)
        it = np.nditer(inp.data, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            original = inp.data[idx]
            
            # f(x + eps)
            inp.data[idx] = original + eps
            loss_plus = func([Tensor(inp.data) if j == i else inputs[j] for j in range(len(inputs))])
            
            # f(x - eps)
            inp.data[idx] = original - eps
            loss_minus = func([Tensor(inp.data) if j == i else inputs[j] for j in range(len(inputs))])
            
            # 数值梯度
            numerical_grad[idx] = (loss_plus.item() - loss_minus.item()) / (2 * eps)
            
            # 恢复原值
            inp.data[idx] = original
            it.iternext()
        
        # 比较梯度
        diff = np.abs(analytic_grads[i] - numerical_grad)
        max_diff = diff.max()
        max_grad = max(np.abs(analytic_grads[i]).max(), np.abs(numerical_grad).max())
        
        if max_diff > atol and max_diff / (max_grad + 1e-8) > rtol:
            print(f"Gradient check failed for input {i}!")
            return False
    
    print("Gradient check passed!")
    return True
```

### 使用示例

```python
from nanotorch import Tensor
from nanotorch.nn import Linear, MSELoss
from nanotorch.utils import gradient_check

linear = Linear(10, 5)

def compute_loss(inputs):
    x, y = inputs
    output = linear(x)
    loss = MSELoss()(output, y)
    return loss

x = Tensor.randn((4, 10), requires_grad=True)
y = Tensor.randn((4, 5))

gradient_check(compute_loss, [x, y])
```

---

## 训练技巧

### 早停（Early Stopping）

```python
class EarlyStopping:
    """早停机制。"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# 使用
early_stop = EarlyStopping(patience=15)

for epoch in range(epochs):
    train_loss = train(model, dataloader, optimizer)
    val_loss = validate(model, val_loader)
    
    if early_stop(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

### 指数移动平均（EMA）

```python
class EMA:
    """参数的指数移动平均。"""
    
    def __init__(self, model, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        
        for name, param in model.named_parameters():
            self.shadow[name] = param.data.copy()
    
    def update(self, model):
        for name, param in model.named_parameters():
            self.shadow[name] = (
                self.decay * self.shadow[name] + (1 - self.decay) * param.data
            )
    
    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            param.data = self.shadow[name].copy()

# 使用
ema = EMA(model, decay=0.999)

for epoch in range(epochs):
    train(...)
    ema.update(model)

# 推理时使用 EMA 参数
ema.apply_shadow(model)
evaluate(model)
```

---

## 调试技巧

### 检查梯度流

```python
def check_gradients(model):
    """检查梯度是否正常。"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = np.linalg.norm(param.grad)
            has_nan = np.isnan(param.grad).any()
            has_inf = np.isinf(param.grad).any()
            
            print(f"{name}: grad_norm={grad_norm:.6f}, nan={has_nan}, inf={has_inf}")
```

### 检查模型输出

```python
def check_output(output, name="output"):
    """检查输出是否正常。"""
    data = output.data if hasattr(output, 'data') else output
    
    print(f"{name}:")
    print(f"  shape: {data.shape}")
    print(f"  mean: {data.mean():.6f}")
    print(f"  std: {data.std():.6f}")
    print(f"  min: {data.min():.6f}")
    print(f"  max: {data.max():.6f}")
    print(f"  has_nan: {np.isnan(data).any()}")
    print(f"  has_inf: {np.isinf(data).any()}")
```

### 可视化计算图

```python
def print_computation_graph(tensor, indent=0):
    """打印计算图。"""
    prefix = "  " * indent
    print(f"{prefix}Tensor(shape={tensor.shape}, requires_grad={tensor.requires_grad})")
    
    if hasattr(tensor, '_op') and tensor._op is not None:
        print(f"{prefix}  op: {tensor._op}")
    
    if hasattr(tensor, '_parents'):
        for parent in tensor._parents:
            print_computation_graph(parent, indent + 2)
```

---

## 总结

本教程介绍了 nanotorch 的高级功能和实用技巧：

| 功能 | 作用 |
|------|------|
| **clip_grad_norm_** | 按范数裁剪梯度 |
| **clip_grad_value_** | 按值裁剪梯度 |
| **LinearWarmup** | 线性预热 |
| **CosineWarmupScheduler** | 余弦预热+衰减 |
| **gradient_check** | 验证梯度正确性 |
| **EarlyStopping** | 早停机制 |
| **EMA** | 指数移动平均 |

### 训练最佳实践

1. **梯度裁剪**：RNN 和 Transformer 训练必备
2. **学习率预热**：大模型训练稳定性的关键
3. **早停**：防止过拟合
4. **EMA**：提高模型泛化能力
5. **梯度检查**：实现新操作时验证正确性

### 调试技巧

1. 检查梯度范数和 NaN/Inf
2. 监控激活值分布
3. 使用小数据集先过拟合
4. 逐步增加模型复杂度

---

**恭喜！** 你已完成 nanotorch 教程系列！

通过这些教程，你学到了：
- Tensor 和自动微分的核心原理
- 如何实现各种神经网络层
- 数据加载和增强
- 优化器和调度器
- 训练和调试技巧

现在你可以：
1. 阅读 nanotorch 源代码加深理解
2. 尝试实现更多功能（如分布式训练、混合精度）
3. 将这些知识应用到 PyTorch 实际项目中

**参考资源**：
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n: CNNs for Visual Recognition](http://cs231n.stanford.edu/)
