# 第五章：损失函数

## 想象你在考试...

考试结束后：
- 你估分 85，实际 90 → 差 5 分
- 你估分 60，实际 90 → 差 30 分

**损失函数就是"评分标准"** —— 衡量你的预测和真实答案差多远。

```
预测 vs 真实

  你的答案:  [猫, 狗, 鸟, 猫]
  正确答案:  [猫, 猫, 鸟, 狗]
                    ↑  ↑
                   错  错

损失函数: "你错了2个，损失=2"
```

---

## 5.1 损失函数的作用

### 训练的本质

```
训练循环：

┌─────────────────────────────────────┐
│                                     │
│   预测 → 损失函数 → 损失值(标量)      │
│              ↑                      │
│           真实标签                   │
│                                     │
│   目标：让损失值尽可能小              │
│                                     │
└─────────────────────────────────────┘
```

### 两大类损失函数

```
回归问题（预测数值）：
  - MSE: 均方误差
  - MAE: 平均绝对误差

分类问题（预测类别）：
  - CrossEntropy: 交叉熵（多分类）
  - BCE: 二元交叉熵（二分类）
```

**一句话总结**：损失函数告诉网络"错得有多离谱"，指导网络往哪里改进。

---

## 5.2 MSE：均方误差

### 什么是 MSE？

```
MSE = mean((y_pred - y_true)²)

类比：射箭比赛
  - 每箭离靶心的距离 → 差值
  - 平方 → 让远距离更"痛"
  - 平均 → 总体表现

例子：
  预测: [2.5, 0.0, 2.1]
  真实: [3.0, -0.5, 2.0]

  差值: [-0.5, 0.5, 0.1]
  平方: [0.25, 0.25, 0.01]
  均值: (0.25 + 0.25 + 0.01) / 3 = 0.17
```

### 实现

```python
class MSELoss(Module):
    """
    均方误差: L = mean((y_pred - y_true)²)

    用途：回归问题
    特点：对大误差惩罚更重（平方效应）
    """

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true
        return (diff * diff).mean()


# 测试
loss_fn = MSELoss()
pred = Tensor([2.5, 0.0, 2.1])
true = Tensor([3.0, -0.5, 2.0])
loss = loss_fn(pred, true)
print(f"MSE Loss: {loss.item():.4f}")  # 0.17
```

### 导数

```
L = mean((y_pred - y_true)²)

∂L/∂y_pred = 2 * (y_pred - y_true) / n

梯度指向"远离真实值的方向"
反向传播会沿着梯度的反方向调整
```

---

## 5.3 CrossEntropy：多分类神器

### 什么是交叉熵？

```
CrossEntropy 衡量两个概率分布的"距离"

公式：L = -Σ y_true * log(y_pred)

对于 one-hot 标签（只有一个类别是1）：
L = -log(y_pred[正确类别])

例子：
  预测概率: [0.1, 0.7, 0.2]  ← 猫,狗,鸟
  真实标签: [0,   1,   0  ]  ← 是狗

  L = -log(0.7) = 0.357

  如果预测概率是 [0.1, 0.2, 0.7]（猜鸟了）：
  L = -log(0.2) = 1.609  ← 损失更大！
```

### 直观理解

```
-log(x) 的图像：

  损失 │╲
       │ ╲
       │  ╲
    0  │   ╲________
       └────────────→ 预测概率
         0    0.5    1

特点：
  - 预测概率接近1 → 损失接近0（猜对了）
  - 预测概率接近0 → 损失爆炸到∞（猜错了）
```

### 实现

```python
class CrossEntropyLoss(Module):
    """
    交叉熵损失（内置 Softmax）

    公式：L = -log(softmax(logits)[正确类别])

    特点：
    1. 数值稳定（使用 log_softmax 技巧）
    2. 梯度简洁：softmax(logits) - one_hot(target)
    """

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        """
        参数:
            logits: (batch, num_classes) - 未经 softmax 的原始输出
            target: (batch,) - 类别索引

        返回:
            标量损失值
        """
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

        def _backward():
            if logits.requires_grad:
                # 梯度 = softmax - one_hot（超简洁！）
                softmax = np.exp(log_softmax)
                grad = softmax.copy()
                grad[np.arange(batch_size), target_indices] -= 1
                grad /= batch_size

                logits.grad = (logits.grad or 0) + grad * out.grad

        out._backward = _backward
        out.requires_grad = logits.requires_grad
        return out
```

### 为什么梯度这么简洁？

```
数学推导：

设 p = softmax(logits), y = one_hot(target)

L = -Σ y_i * log(p_i)

∂L/∂logits_i = p_i - y_i

惊人地简洁！
这意味着：预测概率减去真实标签就是梯度
```

---

## 5.4 BCE：二分类专用

### 什么是二元交叉熵？

```
BCE = -[y * log(p) + (1-y) * log(1-p)]

两种情况：
  y=1 时：L = -log(p)     → p 越接近 1 越好
  y=0 时：L = -log(1-p)   → p 越接近 0 越好

例子：
  预测: 0.9（90%是正类）
  真实: 1（确实是正类）
  L = -log(0.9) = 0.105 ← 损失小

  预测: 0.1（10%是正类）
  真实: 1（实际是正类）
  L = -log(0.1) = 2.303 ← 损失大
```

### 实现

```python
class BCELoss(Module):
    """
    二元交叉熵

    公式：L = -[y*log(p) + (1-y)*log(1-p)]

    用途：二分类问题
    注意：输入应该是经过 Sigmoid 的概率
    """

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        eps = 1e-7  # 防止 log(0)
        y_pred_clipped = np.clip(y_pred.data, eps, 1 - eps)

        loss = -(y_true.data * np.log(y_pred_clipped) +
                 (1 - y_true.data) * np.log(1 - y_pred_clipped))

        return Tensor(np.mean(loss))
```

---

## 5.5 使用示例

```python
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU
from nanotorch.nn.loss import CrossEntropyLoss
import numpy as np

# 模型
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# 损失函数
criterion = CrossEntropyLoss()

# 模拟数据
X = Tensor.randn((32, 784))                # 32张图片
y = Tensor(np.random.randint(0, 10, 32))   # 32个标签

# 前向传播
logits = model(X)          # (32, 10)
loss = criterion(logits, y)

print(f"Loss: {loss.item():.4f}")

# 反向传播
loss.backward()

# 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} 梯度范数: {np.linalg.norm(param.grad):.4f}")
```

---

## 5.6 损失函数对比

| 损失函数 | 公式 | 使用场景 | 输出范围 |
|---------|------|---------|---------|
| MSE | mean((y-ŷ)²) | **回归** | [0, ∞) |
| MAE | mean(\|y-ŷ\|) | 回归（鲁棒） | [0, ∞) |
| CrossEntropy | -Σy·log(p) | **多分类** | [0, ∞) |
| BCE | -[y·log(p)+(1-y)·log(1-p)] | **二分类** | [0, ∞) |
| SmoothL1 | 分段函数 | 目标检测 | [0, ∞) |

### 选择指南

```
问题类型 → 损失函数

回归问题：
  一般 → MSE
  有异常值 → MAE 或 Huber Loss

分类问题：
  二分类 → BCE + Sigmoid
  多分类 → CrossEntropy（已内置 Softmax）
```

---

## 5.7 常见陷阱

### 陷阱1：CrossEntropy 之前加了 Softmax

```python
# 错误：重复使用 Softmax
model = Sequential(
    Linear(784, 10),
    Softmax(),        # ← 不需要！
)
loss = CrossEntropyLoss()(model(x), y)  # CrossEntropy 已经内置 Softmax

# 正确：直接输出 logits
model = Sequential(
    Linear(784, 10),  # 输出原始分数
)
loss = CrossEntropyLoss()(model(x), y)  # 内部会做 Softmax
```

### 陷阱2：BCE 输入没过 Sigmoid

```python
# 错误：BCE 输入是 logits
loss = BCELoss()(model(x), y)  # model(x) 可能是负数或大于1

# 正确：先过 Sigmoid
probs = Sigmoid()(model(x))
loss = BCELoss()(probs, y)

# 或使用 BCEWithLogitsLoss（内置 Sigmoid）
loss = BCEWithLogitsLoss()(model(x), y)
```

### 陷阱3：标签格式错误

```python
# CrossEntropy 标签格式
# 正确：(batch,) - 类别索引
y = Tensor([2, 0, 1, 3])  # 4个样本的类别

# 错误：(batch, num_classes) - one-hot
y = Tensor([[0,0,1,0], [1,0,0,0], ...])  # 不需要！
```

---

## 5.8 练习

### 基础练习

1. **实现 L1Loss（MAE）**：`mean(|y_pred - y_true|)`

2. **实现 SmoothL1Loss（Huber Loss）**：
   ```python
   if |x| < 1: 0.5 * x²
   else: |x| - 0.5
   ```

3. **实现 NLLLoss**：负对数似然（输入是 log_softmax）

### 进阶练习

4. **实现 Focal Loss**：解决类别不平衡
   ```
   FL = -α * (1-p)^γ * log(p)
   ```

5. **实现 Label Smoothing**：软化标签防止过拟合

---

## 一句话总结

| 概念 | 一句话 |
|------|--------|
| 损失函数 | 衡量预测和真实的差距 |
| MSE | 差值平方平均，用于回归 |
| CrossEntropy | -log(预测概率)，用于多分类 |
| BCE | 二分类专用，配合 Sigmoid |
| 梯度 | 预测减真实，简洁优美 |

---

## 下一章

现在我们有了损失函数！

但损失只是告诉我们"差多远"，真正更新参数还需要**优化器**。

→ [第六章：优化器](06-optimizer.md)

```python
# 预告：下一章你将实现
optimizer = Adam(model.parameters(), lr=0.001)

# 训练循环
optimizer.zero_grad()  # 清零梯度
loss.backward()        # 计算梯度
optimizer.step()       # 更新参数
```
