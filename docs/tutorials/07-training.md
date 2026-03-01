# 第七章：完整训练循环

## 想象你在训练一只小狗...

训练小狗的过程：
1. 给指令（输入）
2. 小狗反应（前向传播）
3. 给奖励/纠正（计算损失）
4. 小狗记住教训（反向传播+更新）

```
训练循环：

  给指令 → 小狗动作 → 对了吗？ → 记住教训
    ↑                              │
    └──────── 重复很多遍 ──────────┘

深度学习也一样：
  输入数据 → 模型预测 → 算损失 → 更新参数
    ↑                              │
    └──────── 重复很多遍 ──────────┘
```

**训练循环就是让模型"学习"的整个过程** —— 一遍遍地尝试、犯错、改进。

---

## 7.1 训练循环的本质

### 五步法

```
for epoch in range(num_epochs):
    for batch in dataloader:
        ① 前向传播：predictions = model(inputs)
        ② 计算损失：loss = criterion(predictions, targets)
        ③ 清零梯度：optimizer.zero_grad()
        ④ 反向传播：loss.backward()
        ⑤ 更新参数：optimizer.step()
```

### 为什么要循环？

```
一次不够：

第1次：模型瞎猜，loss=2.5
第2次：学到一点，loss=2.0
第3次：继续进步，loss=1.5
...
第100次：学得不错，loss=0.1

学习需要重复！
就像背单词，看一遍记不住。
```

---

## 7.2 完整训练代码

### 训练函数

```python
import numpy as np
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU, Dropout
from nanotorch.nn.loss import CrossEntropyLoss
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.utils import clip_grad_norm_

def train(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 100,
    lr: float = 0.001,
    save_path: str = 'model.npz'
):
    """
    完整的训练函数

    想象：
      model = 学生
      train_loader = 课本
      val_loader = 模拟考试
      num_epochs = 复习几轮
      optimizer = 学习方法
    """
    # 初始化
    criterion = CrossEntropyLoss()  # 评分标准
    optimizer = Adam(model.parameters(), lr=lr)  # 学习方法
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)  # 学习计划

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # ========== 训练阶段（学生看书学习）==========
        model.train()  # 切换到训练模式
        train_loss = 0.0
        num_batches = 0

        for X_batch, y_batch in train_loader:
            # 转换为 Tensor
            X = Tensor(X_batch)
            y = Tensor(y_batch)

            # ① 前向传播：学生做题
            logits = model(X)
            loss = criterion(logits, y)

            # ③ 清零梯度（清除上次的记忆）
            optimizer.zero_grad()

            # ④ 反向传播：学生看答案，理解错哪了
            loss.backward()

            # 梯度裁剪（防止"学疯了"）
            clip_grad_norm_(model.parameters(), max_norm=1.0)

            # ⑤ 更新参数：学生记住教训
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches

        # ========== 验证阶段（模拟考试）==========
        model.eval()  # 切换到评估模式
        val_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in val_loader:
            X = Tensor(X_batch)
            y = Tensor(y_batch)

            logits = model(X)
            loss = criterion(logits, y)

            val_loss += loss.item()

            # 计算准确率
            predictions = np.argmax(logits.data, axis=1)
            correct += np.sum(predictions == y.data)
            total += len(y.data)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        # 记录历史
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state = model.state_dict()
            np.savez(save_path, **state)
            print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")

        # 更新学习率
        scheduler.step()

    return history
```

---

## 7.3 训练流程图解

```
┌─────────────────────────────────────────────────────────────┐
│                      一个 Epoch                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Batch 1    │ -> │   Batch 2    │ -> │   Batch N    │  │
│  │  前向+反向   │    │  前向+反向   │    │  前向+反向   │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                             │
│                      ↓ 遍历完所有 Batch                      │
│                                                             │
│              ┌─────────────────────┐                        │
│              │     验证阶段        │                        │
│              │  evaluate on val    │                        │
│              └─────────────────────┘                        │
│                      ↓                                      │
│              ┌─────────────────────┐                        │
│              │   保存最佳模型？    │                        │
│              │   更新学习率？      │                        │
│              └─────────────────────┘                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘

重复 num_epochs 次
```

---

## 7.4 使用 DataLoader

### 为什么需要 DataLoader？

```
数据太多，一次吃不下：

10000 张图片 → 分成 313 个 batch（每个 32 张）
             → 每次只处理 32 张
             → 313 次处理完一轮
```

### 代码示例

```python
from nanotorch import DataLoader, TensorDataset
import numpy as np

# 准备数据
X_train = np.random.randn(1000, 784).astype(np.float32)  # 1000张图片
y_train = np.random.randint(0, 10, 1000).astype(np.int64)  # 1000个标签

X_val = np.random.randn(200, 784).astype(np.float32)
y_val = np.random.randint(0, 10, 200).astype(np.int64)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# DataLoader 做了什么？
# 1. 把数据分成小批次（每批32个）
# 2. shuffle=True：打乱顺序（训练时重要！）
# 3. 迭代时自动返回 (X_batch, y_batch)
```

---

## 7.5 模型保存与加载

### 为什么保存模型？

```
训练好的模型 = 学生学到的知识

训练很慢（可能几小时/几天）
保存后可以：
  1. 下次直接用，不用重新训练
  2. 部署到生产环境
  3. 继续训练（迁移学习）
```

### 代码

```python
# 保存模型
def save_model(model, path: str):
    """把模型参数保存到文件"""
    state = model.state_dict()  # 获取所有参数
    np.savez(path, **state)

# 加载模型
def load_model(model, path: str):
    """从文件恢复模型参数"""
    state = dict(np.load(path))
    model.load_state_dict(state)
    return model

# 推理（使用模型）
def predict(model, X):
    """用模型做预测"""
    model.eval()  # 评估模式
    with no_grad():  # 不计算梯度（省内存）
        X = Tensor(X)
        logits = model(X)
        predictions = np.argmax(logits.data, axis=1)
    return predictions

# 使用
model = load_model(model, 'model.npz')
predictions = predict(model, X_test)
```

---

## 7.6 训练技巧

### 1. 学习率调度

```
学习率 = 学习的速度

初期：大步走，快速接近目标
后期：小步走，精细调整

┌───────────────────────────────┐
│    大步走 → 精细调整            │
│   ↘️                          │
│    ↘️                         │
│     ↘️  ↘️                    │
│       ↘️  ↓  ← 小步微调        │
│         ⬇️                    │
└───────────────────────────────┘
```

| 调度器 | 策略 | 适用场景 |
|--------|------|----------|
| StepLR | 每 N 步乘以 0.1 | 简单任务 |
| CosineAnnealingLR | 余弦曲线下降 | Transformer |
| ReduceLROnPlateau | 损失不降就减 | 不确定收敛速度 |

### 2. 梯度裁剪

```python
from nanotorch.utils import clip_grad_norm_

# 梯度裁剪：防止梯度爆炸
clip_grad_norm_(model.parameters(), max_norm=1.0)

# 类比：限制学习速度，防止"学疯了"
# 梯度太大 → 裁剪到 1.0
# 梯度正常 → 不变
```

### 3. 早停（Early Stopping）

```python
def train_with_early_stopping(model, train_loader, val_loader, patience=10):
    """
    如果连续 patience 次验证损失没改善，就停止训练

    类比：
      学生连续 10 次模拟考没进步 → 停止复习
      避免过拟合（死记硬背）
    """
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader)
        val_loss = validate(model, val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_model(model)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
```

---

## 7.7 train() vs eval() 模式

```
model.train()  vs  model.eval()

训练模式：
  - Dropout：随机丢弃神经元
  - BatchNorm：用当前 batch 统计

评估模式：
  - Dropout：不丢弃
  - BatchNorm：用全局统计

类比：
  train() = 平时练习（有随机干扰）
  eval()  = 正式考试（稳定发挥）
```

---

## 7.8 完整示例：MNIST 分类

```python
# examples/mnist_simple.py
import numpy as np
from nanotorch import Tensor, DataLoader, TensorDataset
from nanotorch.nn import Sequential, Linear, ReLU, Dropout
from nanotorch.nn.loss import CrossEntropyLoss
from nanotorch.optim import Adam

# 加载 MNIST 数据（这里用随机数据演示）
def load_mnist():
    X_train = np.random.randn(60000, 784).astype(np.float32)
    y_train = np.random.randint(0, 10, 60000).astype(np.int64)
    X_test = np.random.randn(10000, 784).astype(np.float32)
    y_test = np.random.randint(0, 10, 10000).astype(np.int64)
    return X_train, y_train, X_test, y_test

# 主函数
def main():
    # 数据
    X_train, y_train, X_test, y_test = load_mnist()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=64, shuffle=False
    )

    # 模型
    model = Sequential(
        Linear(784, 512),
        ReLU(),
        Dropout(0.2),
        Linear(512, 256),
        ReLU(),
        Dropout(0.2),
        Linear(256, 10)
    )

    print(f"Model parameters: {sum(p.data.size for p in model.parameters()):,}")

    # 训练
    history = train(
        model, train_loader, test_loader,
        num_epochs=20, lr=0.001
    )

    print(f"\nFinal accuracy: {history['val_acc'][-1]:.4f}")

if __name__ == '__main__':
    main()
```

---

## 7.9 常见陷阱

### 陷阱1：忘记 zero_grad()

```python
# 错误：梯度会累积
for epoch in range(100):
    loss = criterion(model(X), y)
    loss.backward()  # 梯度叠加在上一次的基础上！
    optimizer.step()

# 正确：每次清零
for epoch in range(100):
    optimizer.zero_grad()  # 先清零！
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
```

### 陷阱2：训练/评估模式混淆

```python
# 错误：验证时没切换模式
val_loss = validate(model, val_loader)  # model 还在 train 模式

# 正确
model.eval()
val_loss = validate(model, val_loader)
model.train()  # 切回去
```

### 陷阱3：过拟合

```
过拟合 = 死记硬背

表现：
  - 训练 loss 一直降
  - 验证 loss 先降后升

解决：
  - Dropout
  - 早停
  - 数据增强
  - 减小模型
```

---

## 7.10 一句话总结

| 概念 | 一句话 |
|------|--------|
| 训练循环 | 重复：前向→损失→反向→更新 |
| Epoch | 遍历一遍所有数据 |
| Batch | 一次处理的一小批数据 |
| DataLoader | 自动分批和打乱数据 |
| zero_grad | 清除上次的梯度 |
| model.train/eval | 切换训练/评估模式 |
| 早停 | 验证不改善就停止 |

---

## 恭喜！

你现在理解了深度学习的完整流程：

```
┌─────────────────────────────────────────────────┐
│                                                 │
│   ① Tensor：数据载体 + 自动微分                  │
│   ② Autograd：计算图 + 链式法则                  │
│   ③ Module：参数管理 + 模块组合                  │
│   ④ Layer：Linear, Conv, RNN, Transformer      │
│   ⑤ Activation：ReLU, Sigmoid, Softmax         │
│   ⑥ Loss：衡量预测与真实的差距                   │
│   ⑦ Optimizer：梯度下降更新参数                  │
│   ⑧ Training：整合所有组件 ← 你在这里           │
│                                                 │
└─────────────────────────────────────────────────┘
```

## 下一步

- 阅读 [第八章：数据增强](08-transforms.md)：让数据变多变样
- 学习 [第九章：卷积层](09-conv.md)：处理图像的神器
- 查看 `examples/` 目录的完整示例
- 尝试用 nanotorch 实现你自己的项目！

---

**恭喜！** 你已经掌握了深度学习框架的核心原理！🎉
