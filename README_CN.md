# nanotorch

从零开始实现的精简版 PyTorch，附带交互式可视化，专为教育目的设计。

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
- **🆕 交互式 Web 可视化**：交互式探索 Transformer 架构

## Web 可视化

nanotorch 包含一个交互式 Web 应用，用于可视化和理解 Transformer 架构。

### 功能特性

- **概览仪表板**：快速统计和 Transformer 流程可视化
- **3D 结构视图**：Transformer 架构的交互式 3D 模型
- **嵌入可视化**：词嵌入、位置编码和语义算术演示
- **注意力探索**：多头注意力、QKV 分解、注意力模式
- **逐层分解**：通过 Transformer 层的逐步计算
- **数据流图**：展示张量在网络中流动的桑基图
- **训练监控**：损失曲线、梯度流、权重分布和性能分析
- **推理过程**：自回归生成、束搜索、采样策略
- **分词工具**：字符/词/BPE 分词方法对比

### 运行 Web 应用

```bash
# 从项目根目录
cd frontend
npm install
npm run dev

# 或构建生产版本
npm run build
npm run preview
```

然后在浏览器中打开 `http://localhost:5173`。

### 可视化标签页

| 标签页 | 描述 |
|--------|------|
| **概览** | 模型配置、Transformer 流程、快速统计 |
| **结构** | 3D 架构可视化、模型对比 |
| **嵌入** | 词嵌入、位置编码、语义算术 |
| **注意力** | 注意力矩阵、多头分析、QKV 分解 |
| **分步视图** | 分步注意力计算与流程图 |
| **层** | 详细的层可视化及中间结果 |
| **数据流** | 张量变换的桑基图 |
| **分词** | Token 到文本的映射、词表浏览器 |
| **推理** | 采样策略、束搜索、生成可视化 |
| **训练** | 损失曲线、梯度、权重、模型性能分析 |

## 安装

```bash
# 克隆仓库
git clone https://github.com/qxhy123/nanotorch.git
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

### 激活函数

| 激活函数 | 描述 |
|---------|------|
| `ReLU` | 修正线性单元 |
| `GELU` | 高斯误差线性单元 |
| `SiLU` | Sigmoid 线性单元 (Swish) |
| `LeakyReLU` | 带泄露的 ReLU |
| `Softmax` | Softmax 激活 |

### 损失函数

| 损失函数 | 描述 |
|---------|------|
| `MSE` | 均方误差 |
| `L1Loss` | 平均绝对误差 |
| `CrossEntropyLoss` | 交叉熵损失 |
| `BCELoss` | 二元交叉熵 |
| `BCEWithLogitsLoss` | 带 Sigmoid 的二元交叉熵 |

### 优化器

| 优化器 | 描述 |
|-------|------|
| `SGD` | 随机梯度下降 |
| `Adam` | Adam 优化器 |
| `AdamW` | 带解耦权重衰减的 Adam |
| `RMSprop` | RMSprop 优化器 |

## 项目结构

```
nanotorch/
├── nanotorch/              # 核心库
│   ├── tensor.py          # 带自动微分的张量
│   ├── autograd.py        # 自动微分引擎
│   ├── nn/                # 神经网络模块
│   │   ├── transformer.py # Transformer 组件
│   │   ├── attention.py
│   │   └── ...
│   ├── optim/             # 优化器与调度器
│   ├── data/              # 数据工具
│   └── tokenizer/         # 分词器实现
├── frontend/              # Web 可视化应用
│   ├── src/
│   │   ├── components/   # React 组件
│   │   │   └── visualization/
│   │   │       ├── attention/   # 注意力可视化
│   │   │       ├── embedding/   # 嵌入可视化
│   │   │       ├── training/    # 训练可视化
│   │   │       ├── inference/   # 推理可视化
│   │   │       └── ...
│   │   └── stores/       # 状态管理
│   └── package.json
├── backend/               # 可视化后端 API
│   └── app/
│       └── api/
│           └── routes/   # API 端点
├── tests/                 # 测试套件
├── examples/              # 示例脚本
├── docs/                  # 文档
└── pyproject.toml
```

## 测试

```bash
# 运行所有测试
python -m pytest tests/ -v

# 带覆盖率运行
python -m pytest tests/ --cov=nanotorch
```

## 截图

### Web 可视化

![概览](docs/screenshots/overview.png)
*概览仪表板与 Transformer 流程*

![注意力](docs/screenshots/attention.png)
*多头注意力可视化*

![训练](docs/screenshots/training.png)
*训练指标与损失曲线*

## 限制

- 仅支持 CPU（无 GPU 加速）
- 操作集比 PyTorch 有限
- 不支持分布式训练

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
- Web 可视化使用 React、TypeScript 和 Recharts 构建

## 引用

```bibtex
@software{nanotorch,
  title = {nanotorch: 从零开始的精简版 PyTorch 实现（附带交互式可视化）},
  author = {qxhy123},
  year = {2026},
  url = {https://github.com/qxhy123/nanotorch}
}
```

---

[English Documentation](README.md)
