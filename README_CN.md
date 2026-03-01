# nanotorch

从零开始实现的精简版 PyTorch，专为教育目的设计。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

## 概述

nanotorch 是一个轻量级的 PyTorch 核心功能实现，完全从零开始，仅依赖 NumPy。它提供了：

- **张量**：支持自动微分和 85+ 种操作
- **神经网络层**：Linear, Conv1D/2D/3D, ConvTranspose2D/3D, RNN/LSTM/GRU, Transformer
- **归一化层**：BatchNorm1d/2d/3d, LayerNorm, GroupNorm, InstanceNorm1d/2d/3d
- **激活函数**：ReLU, GELU, SiLU, LeakyReLU, ELU, PReLU, Softplus 等
- **池化层**：MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
- **损失函数**：MSE, L1Loss, SmoothL1Loss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, NLLLoss
- **优化器**：SGD, Adam, AdamW, RMSprop, Adagrad
- **学习率调度器**：StepLR, CosineAnnealingLR, LinearWarmup, CosineWarmupScheduler 等
- **数据工具**：DataLoader, Dataset, TensorDataset, random_split
- **数据增强**：RandomCrop, RandomFlip, ColorJitter, RandomErasing 等
- **模型序列化**：保存/加载 state dict

## 安装

```bash
# 克隆仓库
git clone https://github.com/your-username/nanotorch.git
cd nanotorch

# 使用 uv 安装（推荐）
uv venv
source .venv/bin/activate
uv sync

# 或使用 pip 安装
pip install -e .
```

## 快速开始

### 基础神经网络

```python
import numpy as np
from nanotorch import Tensor
from nanotorch.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from nanotorch.optim import SGD

# 创建模型
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)

# 示例数据
X = Tensor.randn((100, 784))
y = Tensor(np.random.randint(0, 10, (100,)))

# 训练配置
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(100):
    predictions = model(X)
    loss = criterion(predictions, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### 使用 DataLoader

```python
from nanotorch import DataLoader, TensorDataset
import numpy as np

# 创建数据集
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 1000).astype(np.int64)
dataset = TensorDataset(X_train, y_train)

# 创建数据加载器
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练
for X_batch, y_batch in loader:
    X_tensor = Tensor(X_batch)
    y_tensor = Tensor(y_batch)
    # ... 训练步骤
```

### RNN / LSTM / GRU

```python
from nanotorch.nn import LSTM, Linear
from nanotorch import Tensor

# 创建 LSTM 模型
lstm = LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
fc = Linear(128, 10)

# 前向传播
x = Tensor.randn((32, 10, 64))  # (batch, seq_len, input_size)
output, (h_n, c_n) = lstm(x)
output = fc(output[:, -1, :])  # 使用最后一个隐藏状态
```

### Transformer

```python
from nanotorch.nn import TransformerEncoderLayer, TransformerEncoder, Embedding

# 创建 Transformer 编码器
encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048)
encoder = TransformerEncoder(encoder_layer, num_layers=6)

# 嵌入层
embedding = Embedding(num_embeddings=10000, embedding_dim=512)

# 前向传播
tokens = Tensor(np.random.randint(0, 10000, (32, 100)))  # (batch, seq_len)
x = embedding(tokens)
output = encoder(x)
```

### 学习率预热

```python
from nanotorch.optim import AdamW, CosineWarmupScheduler

optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = CosineWarmupScheduler(
    optimizer, 
    warmup_epochs=5,  # 预热轮数
    max_epochs=100
)

for epoch in range(100):
    train(...)
    scheduler.step()
```

### 数据增强

```python
from nanotorch.transforms import (
    Compose, ToFloat, Normalize, 
    RandomHorizontalFlip, RandomCrop, ColorJitter
)

transform = Compose([
    ToFloat(),
    RandomHorizontalFlip(p=0.5),
    RandomCrop(size=224),
    ColorJitter(brightness=0.2, contrast=0.2),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 应用到图像
image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
augmented = transform(image)
```

## 可用组件

### 神经网络层

| 层 | 描述 |
|---|------|
| `Linear` | 全连接层 |
| `Conv1D/2D/3D` | 卷积层 |
| `ConvTranspose2D/3D` | 转置卷积 |
| `Embedding` | 词嵌入层 |
| `RNN` | 循环神经网络 |
| `LSTM` | 长短期记忆网络 |
| `GRU` | 门控循环单元 |
| `TransformerEncoder` | Transformer 编码器 |
| `MultiheadAttention` | 多头注意力 |

### 归一化层

| 层 | 描述 |
|---|------|
| `BatchNorm1d/2d/3d` | 批归一化 |
| `LayerNorm` | 层归一化 |
| `GroupNorm` | 组归一化 |
| `InstanceNorm1d/2d/3d` | 实例归一化 |

### 池化层

| 层 | 描述 |
|---|------|
| `MaxPool1d/2d/3d` | 最大池化 |
| `AvgPool1d/2d/3d` | 平均池化 |
| `AdaptiveAvgPool2d` | 自适应平均池化 |
| `AdaptiveMaxPool2d` | 自适应最大池化 |

### 激活函数

| 激活函数 | 描述 |
|---------|------|
| `ReLU` | 修正线性单元 |
| `LeakyReLU` | 带泄露的 ReLU |
| `GELU` | 高斯误差线性单元 |
| `SiLU` | Sigmoid 线性单元 (Swish) |
| `PReLU` | 参数化 ReLU |
| `Sigmoid` | Sigmoid 激活 |
| `Tanh` | 双曲正切 |
| `Softmax` | Softmax 激活 |
| `ELU` | 指数线性单元 |
| `Softplus` | Softplus 激活 |

### 损失函数

| 损失函数 | 描述 |
|---------|------|
| `MSE` | 均方误差 |
| `L1Loss` | 平均绝对误差 |
| `SmoothL1Loss` | Huber 损失 |
| `CrossEntropyLoss` | 交叉熵损失 |
| `BCELoss` | 二元交叉熵 |
| `BCEWithLogitsLoss` | 带 Sigmoid 的二元交叉熵 |
| `NLLLoss` | 负对数似然损失 |

### 优化器

| 优化器 | 描述 |
|-------|------|
| `SGD` | 随机梯度下降（支持 momentum, nesterov） |
| `Adam` | Adam 优化器 |
| `AdamW` | 带解耦权重衰减的 Adam |
| `RMSprop` | RMSprop 优化器 |
| `Adagrad` | Adagrad 优化器 |

### 学习率调度器

| 调度器 | 描述 |
|-------|------|
| `StepLR` | 每 step_size 轮衰减 gamma 倍 |
| `MultiStepLR` | 在指定轮次衰减 |
| `ExponentialLR` | 指数衰减 |
| `CosineAnnealingLR` | 余弦退火 |
| `LinearWarmup` | 线性预热 |
| `WarmupScheduler` | 预热 + 任意调度器 |
| `CosineWarmupScheduler` | 预热 + 余弦退火 |
| `ReduceLROnPlateau` | 指标停滞时降低学习率 |

### 数据工具

| 类 | 描述 |
|---|------|
| `Dataset` | 数据集基类 |
| `TensorDataset` | 张量数据集 |
| `DataLoader` | 批量数据加载器 |
| `Subset` | 数据集子集 |
| `random_split` | 随机划分数据集 |

### 数据增强

| 变换 | 描述 |
|-----|------|
| `Compose` | 组合多个变换 |
| `Normalize` | 归一化 |
| `RandomHorizontalFlip` | 随机水平翻转 |
| `RandomVerticalFlip` | 随机垂直翻转 |
| `RandomCrop` | 随机裁剪 |
| `CenterCrop` | 中心裁剪 |
| `RandomResizedCrop` | 随机裁剪并缩放 |
| `ColorJitter` | 颜色抖动 |
| `RandomErasing` | 随机擦除 |
| `GaussianBlur` | 高斯模糊 |

### 初始化函数

| 函数 | 描述 |
|-----|------|
| `xavier_uniform_` | Xavier/Glorot 均匀分布 |
| `xavier_normal_` | Xavier/Glorot 正态分布 |
| `kaiming_uniform_` | Kaiming/He 均匀分布 |
| `kaiming_normal_` | Kaiming/He 正态分布 |
| `trunc_normal_` | 截断正态分布 |
| `orthogonal_` | 正交矩阵初始化 |
| `sparse_` | 稀疏初始化 |
| `zeros_` / `ones_` | 常数初始化 |

## 张量操作

```python
from nanotorch import Tensor

t = Tensor.randn((2, 3, 4))

# 形状操作
t.reshape((6, 4))
t.flatten(start_dim=1)
t.transpose(0, 1)
t.squeeze()
t.expand(4, 3, 4)
t.repeat(2, 1, 1)

# 数学运算
t + t
t * t
t.matmul(t.transpose(0, 1))
t.sum(dim=1)
t.mean(dim=0)
t.softmax(dim=-1)

# 分割与排序
t.split(split_size=2, dim=0)
t.chunk(chunks=2, dim=0)
values, indices = t.topk(k=2, dim=-1)
values, indices = t.sort(dim=-1, descending=True)
```

## 梯度工具

```python
from nanotorch.utils import clip_grad_norm_, clip_grad_value_, get_grad_norm_

# 裁剪梯度范数
total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)

# 裁剪梯度值
clip_grad_value_(model.parameters(), clip_value=0.5)

# 获取梯度范数
norm = get_grad_norm_(model.parameters(), norm_type=2.0)
```

## 模型序列化

```python
# 保存模型
state_dict = model.state_dict()
np.savez('model.npz', **state_dict)

# 加载模型
state_dict = dict(np.load('model.npz'))
model.load_state_dict(state_dict)
```

## 示例

参见 `examples/` 目录：

| 示例 | 描述 |
|-----|------|
| `simple_neural_net.py` | 基础神经网络 |
| `mnist_classifier.py` | MNIST CNN 分类器 |
| `mini_gpt.py` | 字符级 GPT |
| `chat_llm.py` | 简单的 Transformer 聊天机器人 |
| `autograd_demo.py` | 自动微分演示 |
| `conv2d_training.py` | CNN 训练示例 |

## 项目结构

```
nanotorch/
├── nanotorch/
│   ├── tensor.py          # 带自动微分的张量
│   ├── autograd.py        # 自动微分引擎
│   ├── utils.py           # 工具函数和初始化
│   ├── nn/                # 神经网络模块
│   │   ├── linear.py
│   │   ├── conv.py
│   │   ├── rnn.py         # RNN/LSTM/GRU
│   │   ├── transformer.py # Transformer 组件
│   │   ├── attention.py
│   │   ├── embedding.py
│   │   ├── pooling.py
│   │   ├── normalization.py
│   │   ├── activation.py
│   │   ├── loss.py
│   │   └── dropout.py
│   ├── optim/             # 优化器
│   │   ├── sgd.py
│   │   ├── adam.py
│   │   ├── adamw.py
│   │   ├── rmsprop.py
│   │   ├── adagrad.py
│   │   └── lr_scheduler.py
│   ├── data/              # 数据工具
│   │   └── __init__.py    # DataLoader, Dataset
│   └── transforms/        # 数据增强
│       └── __init__.py
├── tests/                 # 测试套件 (199 个测试)
├── examples/              # 示例脚本
├── docs/                  # 文档
└── pyproject.toml
```

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 运行特定测试
python -m pytest tests/test_tensor.py -v
python -m pytest tests/test_nn.py -v

# 带覆盖率运行
python -m pytest tests/ --cov=nanotorch
```

## 限制

- 仅支持 CPU（无 GPU 加速）
- 操作集比 PyTorch 有限
- 不支持分布式训练
- 转置卷积的 groups > 1 未实现

## 贡献

欢迎贡献！请：

1. 遵循现有代码风格
2. 为新功能添加测试
3. 更新文档
4. 确保所有测试通过

## 许可证

MIT 许可证。详见 [LICENSE](LICENSE)。

## 致谢

- 灵感来源于 PyTorch、micrograd 和 tinygrad
- 专为深度学习框架的教育用途设计

## 引用

```bibtex
@software{nanotorch,
  title = {nanotorch: 从零开始的精简版 PyTorch 实现},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/nanotorch}
}
```

---

[English Documentation](README.md)
