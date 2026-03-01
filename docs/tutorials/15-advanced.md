# 第十五章：高级主题

## 训练神经网络，像是在照顾一个孩子...

你需要耐心，需要技巧，需要懂得察言观色。

有时候，它学得太猛，梯度爆炸，你需要温和地告诉它："慢一点，别急。"——这是**梯度裁剪**。

有时候，它刚起步，什么都不懂，你需要让它从简单的开始，慢慢加量——这是**学习率预热**。

有时候，它学过头了，开始死记硬背，你需要及时喊停——这是**早停**。

```
训练的智慧：

  梯度裁剪：
    梯度太大 → 限制住
    防止参数更新太猛，跑飞了

  学习率预热：
    刚开始 → 小学习率
    慢慢增加 → 让模型先适应

  早停：
    验证损失不再下降 → 停止训练
    防止过拟合，见好就收
```

**训练技巧，是经验与智慧的结合。** 它们让模型学得更稳、更快、更好。

---

## 15.1 梯度裁剪

### 问题：梯度爆炸

```
深层网络或 RNN 训练时：

正常梯度：[0.1, 0.2, -0.1, ...]
爆炸梯度：[100, 200, -50, ...]

后果：
  - 参数更新太大
  - 损失变成 NaN
  - 训练崩溃
```

### 解决：梯度裁剪

```
梯度裁剪：限制梯度的大小

clip_grad_norm_: 按范数裁剪
  如果 ||梯度|| > max_norm:
    梯度 = 梯度 × (max_norm / ||梯度||)

类比：限速
  超速了 → 降速到限速值
  没超速 → 保持原速
```

### 实现

```python
def clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    """
    按范数裁剪梯度

    Args:
        parameters: 模型参数
        max_norm: 最大范数（常用1.0或5.0）

    Returns:
        裁剪前的梯度范数
    """
    # 收集所有梯度
    grads = []
    for param in parameters:
        if param.grad is not None:
            grads.append(param.grad.flatten())

    if len(grads) == 0:
        return 0.0

    # 计算总梯度范数
    total_grad = np.concatenate(grads)
    total_norm = np.linalg.norm(total_grad, ord=norm_type)

    # 计算裁剪系数
    clip_coef = max_norm / (total_norm + 1e-6)

    # 如果需要裁剪
    if clip_coef < 1:
        for param in parameters:
            if param.grad is not None:
                param.grad = param.grad * clip_coef

    return float(total_norm)
```

### 使用

```python
from nanotorch.utils import clip_grad_norm_

# 训练循环
for x, y in dataloader:
    optimizer.zero_grad()

    output = model(Tensor(x))
    loss = criterion(output, Tensor(y))
    loss.backward()

    # 梯度裁剪（在 optimizer.step() 之前）
    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    if grad_norm > 1.0:
        print(f"Gradient clipped! Original norm: {grad_norm:.2f}")
```

### 什么时候用？

```
必须用：
  - RNN / LSTM 训练
  - Transformer 训练
  - 深层网络

可选：
  - 普通CNN
  - 浅层网络

经验值：
  - RNN: max_norm=1.0 或 5.0
  - Transformer: max_norm=1.0
```

---

## 15.2 学习率预热

### 问题：训练初期不稳定

```
训练刚开始时：
  - 参数是随机初始化的
  - 特征还没有意义
  - 大学习率可能导致震荡

类比：冷车启动
  - 冬天早上，引擎是冷的
  - 直接踩油门 → 引擎损伤
  - 先热车 → 运行顺畅
```

### 解决：Warmup

```
Warmup（预热）：

普通训练：
  学习率 = 0.001（一直不变）

有预热：
  Epoch 1:  学习率 = 0.0001
  Epoch 2:  学习率 = 0.0002
  Epoch 3:  学习率 = 0.0003
  ...
  Epoch 10: 学习率 = 0.001  ← 到达目标
  Epoch 11: 学习率 = 0.001
  ...
```

### 实现

```python
class LinearWarmup:
    """
    线性预热

    学习率从 start_lr 线性增加到目标学习率
    """

    def __init__(self, optimizer, warmup_epochs, start_lr=0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.target_lr = optimizer.lr
        self.current_epoch = 0

    def step(self):
        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：线性增加
            alpha = self.current_epoch / self.warmup_epochs
            self.optimizer.lr = self.start_lr + alpha * (self.target_lr - self.start_lr)
        self.current_epoch += 1
```

### 使用

```python
from nanotorch.optim import Adam
from nanotorch.utils import LinearWarmup

optimizer = Adam(model.parameters(), lr=0.001)
warmup = LinearWarmup(optimizer, warmup_epochs=5, start_lr=0.0001)

for epoch in range(100):
    train_one_epoch(...)

    warmup.step()  # 更新学习率
    print(f"Epoch {epoch}, LR: {optimizer.lr:.6f}")
```

### 余弦预热+衰减

```python
class CosineWarmupScheduler:
    """
    余弦预热 + 余弦衰减

    最常用的 Transformer 调度器
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.current_epoch = 0

    def step(self):
        import math

        if self.current_epoch < self.warmup_epochs:
            # 预热阶段：线性增加
            alpha = self.current_epoch / self.warmup_epochs
            lr = self.min_lr + alpha * (self.base_lr - self.min_lr)
        else:
            # 衰减阶段：余弦衰减
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        self.optimizer.lr = lr
        self.current_epoch += 1
```

---

## 15.3 早停

### 问题：过拟合

```
训练过程：

Epoch 1-20:  训练loss ↓，验证loss ↓  ← 学习中
Epoch 21-50: 训练loss ↓，验证loss →  ← 开始过拟合
Epoch 51+:   训练loss ↓，验证loss ↑  ← 过拟合严重

要在验证loss不再下降时停止！
```

### 实现

```python
class EarlyStopping:
    """
    早停机制

    如果验证损失连续 patience 次没改善，就停止训练
    """

    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        """检查是否应该停止"""
        if val_loss < self.best_loss - self.min_delta:
            # 有改善
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            # 没改善
            self.counter += 1
            if self.counter >= self.patience:
                return True  # 应该停止
            return False
```

### 使用

```python
early_stop = EarlyStopping(patience=15)
best_loss = float('inf')

for epoch in range(100):
    train_loss = train(model, train_loader, optimizer)
    val_loss = validate(model, val_loader)

    # 保存最佳模型
    if val_loss < best_loss:
        best_loss = val_loss
        save_model(model, 'best_model.npz')

    # 检查是否早停
    if early_stop(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 15.4 模型保存与加载

### 保存模型

```python
# 保存参数
def save_model(model, path):
    state_dict = model.state_dict()
    np.savez(path, **state_dict)

# 保存检查点（包含训练状态）
def save_checkpoint(model, optimizer, epoch, loss, path):
    checkpoint = {
        'epoch': epoch,
        'model_state': model.state_dict(),
        'loss': loss,
    }
    np.savez(path, **checkpoint)
```

### 加载模型

```python
# 加载参数
def load_model(model, path):
    state_dict = dict(np.load(path))
    model.load_state_dict(state_dict)
    return model

# 加载检查点
def load_checkpoint(model, path):
    checkpoint = dict(np.load(path, allow_pickle=True))
    model.load_state_dict(checkpoint['model_state'].item())
    return checkpoint['epoch'], checkpoint['loss']
```

---

## 15.5 训练调试技巧

### 检查梯度

```python
def check_gradients(model):
    """检查梯度是否正常"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = np.linalg.norm(param.grad)
            has_nan = np.isnan(param.grad).any()
            has_inf = np.isinf(param.grad).any()

            print(f"{name}:")
            print(f"  norm: {grad_norm:.6f}")
            print(f"  has_nan: {has_nan}")
            print(f"  has_inf: {has_inf}")
```

### 先用小数据集验证

```python
# 调试技巧：先用10个样本
small_dataset = TensorDataset(X[:10], y[:10])
small_loader = DataLoader(small_dataset, batch_size=10)

# 应该能快速过拟合（loss → 0）
for epoch in range(100):
    loss = train_one_epoch(model, small_loader, optimizer)
    print(f"Epoch {epoch}, Loss: {loss:.6f}")

# 如果 loss 不下降 → 模型或代码有问题
```

### 监控训练指标

```python
def train_with_logging(model, train_loader, val_loader, epochs):
    history = {'train_loss': [], 'val_loss': [], 'grad_norm': []}

    for epoch in range(epochs):
        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer)
        grad_norm = get_grad_norm(model.parameters())

        # 验证
        val_loss = validate(model, val_loader)

        # 记录
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['grad_norm'].append(grad_norm)

        # 打印
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Grad Norm: {grad_norm:.4f}")

    return history
```

---

## 15.6 训练技巧总结

### 必用技巧

```
1. 梯度裁剪
   clip_grad_norm_(model.parameters(), max_norm=1.0)

2. 学习率调度
   scheduler.step() 每个 epoch 后

3. 早停
   防止过拟合

4. 保存最佳模型
   根据验证损失
```

### 调试流程

```
1. 先用小数据集过拟合
   确保代码正确

2. 检查梯度
   没有 NaN/Inf

3. 监控损失曲线
   应该平滑下降

4. 逐步增加数据/模型复杂度
```

---

## 15.7 一句话总结

| 技巧 | 作用 | 何时用 |
|------|------|--------|
| 梯度裁剪 | 防止梯度爆炸 | RNN、Transformer |
| 学习率预热 | 稳定训练初期 | 大模型、大学习率 |
| 早停 | 防止过拟合 | 训练时间长时 |
| 小数据验证 | 检查代码 | 调试时 |

---

## 恭喜！

你已经完成了 nanotorch 教程系列！

```
你学到了：

┌─────────────────────────────────────────┐
│                                         │
│  ① Tensor：数据载体 + 自动微分          │
│  ② Autograd：计算图 + 链式法则          │
│  ③ Module：参数管理 + 模块组合          │
│  ④ Layer：Linear, Conv, RNN, Attention │
│  ⑤ Activation：ReLU, Sigmoid, Softmax  │
│  ⑥ Loss：MSE, CrossEntropy, BCE        │
│  ⑦ Optimizer：SGD, Adam, AdamW         │
│  ⑧ Training：完整训练循环               │
│  ⑨ Data：Dataset, DataLoader           │
│  ⑩ Init：Xavier, Kaiming               │
│  ⑪ Advanced：梯度裁剪、早停             │
│                                         │
└─────────────────────────────────────────┘
```

### 下一步

1. 阅读 nanotorch 源代码
2. 实现更多功能（混合精度、分布式）
3. 应用到实际项目

**你已经理解了深度学习框架的核心原理！** 🎉
