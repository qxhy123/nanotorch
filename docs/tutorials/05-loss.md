# 第五章：损失函数

损失函数衡量模型预测与真实标签的差距，是训练的核心。

## 5.1 损失函数的作用

```
预测 y_pred → 损失函数 → 标量 L
              ↑
           真实 y_true
```

训练目标：**最小化 L**

## 5.2 MSE 均方误差

```python
class MSE(Module):
    """均方误差: L = mean((y_pred - y_true)²)"""
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        return (diff * diff).mean()
```

**导数**：
```
∂L/∂y_pred = 2 * (y_pred - y_true) / n
```

**使用场景**：回归问题

## 5.3 CrossEntropyLoss 交叉熵

最常用的分类损失：

```python
class CrossEntropyLoss(Module):
    """交叉熵损失（内置 Softmax）
    
    L = -sum(y_true * log(softmax(y_pred)))
    
    对于独热标签: L = -log(softmax(y_pred)[正确类别])
    """
    
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # logits: (batch, num_classes)
        # target: (batch,) - 类别索引
        
        # 数值稳定的 softmax
        shifted = logits - logits.max(dim=1, keepdims=True)
        exp_x = shifted.exp()
        softmax = exp_x / exp_x.sum(dim=1, keepdims=True)
        
        # 提取正确类别的概率
        batch_size = logits.shape[0]
        correct_probs = softmax[np.arange(batch_size), target.data.astype(int)]
        
        # 负对数似然
        loss = -correct_probs.log().mean()
        
        return loss
```

**数学推导**：

设 $p = \text{softmax}(\text{logits})$, $y = \text{target}$（独热）

$$L = -\sum_i y_i \cdot \log(p_i)$$

$$\frac{\partial L}{\partial \text{logits}_i} = p_i - y_i$$

**简洁形式**：梯度 = `softmax(logits) - one_hot(target)`

## 5.4 完整 CrossEntropyLoss 实现

```python
class CrossEntropyLoss(Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        # 数值稳定的 log_softmax
        shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
        log_softmax = shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        
        # 提取正确类别的 log 概率
        batch_size = logits.shape[0]
        target_indices = target.data.astype(np.int64)
        correct_log_probs = log_softmax[np.arange(batch_size), target_indices]
        
        # 计算损失
        if self.reduction == 'mean':
            loss = -np.mean(correct_log_probs)
        elif self.reduction == 'sum':
            loss = -np.sum(correct_log_probs)
        else:
            loss = -correct_log_probs
        
        # 创建输出张量并设置反向传播
        out = Tensor(loss, _children=(logits,), _op='cross_entropy')
        out.requires_grad = logits.requires_grad
        
        def _backward():
            if logits.requires_grad:
                # 梯度 = softmax - one_hot
                softmax = np.exp(log_softmax)
                grad = softmax.copy()
                grad[np.arange(batch_size), target_indices] -= 1
                grad /= batch_size  # 因为 mean reduction
                
                if logits.grad is None:
                    logits.grad = grad * out.grad
                else:
                    logits.grad += grad * out.grad
        
        out._backward = _backward
        return out
```

## 5.5 BCELoss 二元交叉熵

```python
class BCELoss(Module):
    """二元交叉熵: L = -[y*log(p) + (1-y)*log(1-p)]"""
    
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        eps = 1e-7
        y_pred = y_pred.clip(eps, 1 - eps)
        
        loss = -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log())
        return loss.mean()
```

## 5.6 使用示例

```python
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU
from nanotorch.nn.loss import CrossEntropyLoss

# 模型
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# 损失函数
criterion = CrossEntropyLoss()

# 模拟数据
X = Tensor.randn((32, 784))
y = Tensor(np.random.randint(0, 10, 32))

# 前向传播
logits = model(X)
loss = criterion(logits, y)

# 反向传播
loss.backward()

print(f"Loss: {loss.item():.4f}")
```

## 5.7 损失函数对比

| 损失函数 | 公式 | 使用场景 |
|---------|------|----------|
| MSE | mean((y-ŷ)²) | 回归 |
| MAE/L1 | mean(|y-ŷ|) | 回归（鲁棒） |
| CrossEntropy | -Σy*log(p) | 多分类 |
| BCE | -[y*log(p)+(1-y)*log(1-p)] | 二分类 |
| BCEWithLogits | BCE(sigmoid(x), y) | 二分类（数值稳定） |

## 5.8 练习

1. **实现 L1Loss**：`mean(|y_pred - y_true|)`

2. **实现 SmoothL1Loss**（Huber Loss）

3. **实现 NLLLoss**：负对数似然（不带 Softmax）

## 下一章

下一章，我们将实现**优化器**，使用梯度更新参数。

→ [第六章：优化器](06-optimizer.md)
