# 第七章：完整训练循环

本章我们将所有组件整合，实现一个完整的训练流程。

## 7.1 训练循环结构

```
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. 前向传播
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        
        # 2. 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 3. 参数更新
        optimizer.step()
    
    # 4. 验证
    val_loss = validate(model, val_loader)
    
    # 5. 保存模型
    if val_loss < best_loss:
        save_model(model)
```

## 7.2 完整训练代码

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
    # 初始化
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        # ========== 训练阶段 ==========
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for X_batch, y_batch in train_loader:
            # 转换为 Tensor
            X = Tensor(X_batch)
            y = Tensor(y_batch)
            
            # 前向传播
            logits = model(X)
            loss = criterion(logits, y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 参数更新
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        
        # ========== 验证阶段 ==========
        model.eval()
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

## 7.3 使用 DataLoader

```python
from nanotorch import DataLoader, TensorDataset
import numpy as np

# 准备数据
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 1000).astype(np.int64)

X_val = np.random.randn(200, 784).astype(np.float32)
y_val = np.random.randint(0, 10, 200).astype(np.int64)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 创建模型
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Dropout(0.2),
    Linear(256, 128),
    ReLU(),
    Dropout(0.2),
    Linear(128, 10)
)

# 训练
history = train(
    model,
    train_loader,
    val_loader,
    num_epochs=50,
    lr=0.001
)
```

## 7.4 模型加载与推理

```python
# 加载模型
def load_model(model, path: str):
    state = dict(np.load(path))
    model.load_state_dict(state)
    return model

# 推理
def predict(model, X):
    model.eval()
    with no_grad():
        X = Tensor(X)
        logits = model(X)
        predictions = np.argmax(logits.data, axis=1)
    return predictions

# 使用
model = load_model(model, 'model.npz')
predictions = predict(model, X_test)
```

## 7.5 训练技巧

### 学习率调度的选择

| 调度器 | 适用场景 |
|--------|----------|
| StepLR | 简单任务 |
| CosineAnnealingLR | Transformer |
| ReduceLROnPlateau | 不确定收敛速度时 |

### 梯度裁剪

```python
from nanotorch.utils import clip_grad_norm_

# 防止梯度爆炸
clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 早停（Early Stopping）

```python
def train_with_early_stopping(model, train_loader, val_loader, patience=10):
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

## 7.6 完整示例：MNIST 分类

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

## 7.7 总结

恭喜你完成了 nanotorch 教程！你现在理解了：

1. **Tensor**：多维数组 + 自动微分
2. **Autograd**：计算图 + 链式法则
3. **Module**：参数管理 + 模块组合
4. **Layer**：Linear, Conv, RNN, Transformer
5. **Loss**：衡量预测与真实的差距
6. **Optimizer**：梯度下降更新参数
7. **Training**：整合所有组件

## 下一步

- 阅读 [第八章：数据增强](08-transforms.md)：图像变换和数据增强
- 学习 [第九章：卷积层](09-conv.md)：Conv2D、转置卷积
- 查看 `examples/` 目录的完整示例
- 尝试用 nanotorch 实现你自己的项目！

---

**恭喜！** 你已经掌握了深度学习框架的核心原理！🎉
