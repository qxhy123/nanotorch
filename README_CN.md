# nanotorch

一个以教学为目标、基于 NumPy 从零实现的 PyTorch 风格库；同一仓库中还附带了一个 Transformer 可视化应用。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

## 概述

nanotorch 仓库现在按两条主线组织，并在文档中保持一致：

### Core Python Library

`nanotorch` Python 包聚焦于可读、可学习的 PyTorch 风格核心实现。

- 带反向传播能力的 `Tensor`
- 基于 `Function.apply(...)` 的算子定义，以及统一的 backward 遍历机制
- 线性层、卷积、池化、归一化、注意力、Embedding、RNN、Transformer 等神经网络模块
- SGD、Adam、AdamW、RMSprop、Adagrad，以及 StepLR、CosineAnnealingLR、warmup 等调度器
- `Dataset`、`DataLoader`、`TensorDataset`、`random_split` 等数据工具
- tokenizer、transforms，以及仓库内的部分实验性子系统
- 默认以 CPU 路径为主；若额外安装兼容的 CuPy，则可使用 CUDA / device 相关能力

### Visualization App

同一仓库还包含一个 Transformer 可视化应用，由前端和后端共同组成。它是仓库级配套工具，不属于发布到 PyPI 的 `nanotorch` 包元数据本体。

- 前端：交互式查看 embedding、attention、layer flow、tokenization、inference、training metrics
- 后端：用 FastAPI 封装 nanotorch 的 Transformer 组件并提供可视化数据
- 启动方式：完整体验需要同时启动 frontend 和 backend

可视化专项说明请查看 [`QUICKSTART_CN.md`](./QUICKSTART_CN.md) 和 [`README_VISUALIZATION_CN.md`](./README_VISUALIZATION_CN.md)。

## 安装

```bash
git clone https://github.com/qxhy123/nanotorch.git
cd nanotorch

uv venv
source .venv/bin/activate
uv sync
```

或以 editable 模式安装：

```bash
pip install -e .
```

如果需要 CUDA 路径，请额外安装与环境匹配的 CuPy。

## 快速开始

### 训练一个小网络

```python
import numpy as np
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from nanotorch.optim import SGD

model = Sequential(
    Linear(4, 16),
    ReLU(),
    Linear(16, 3),
)

inputs = Tensor.randn((8, 4))
targets = Tensor(np.random.randint(0, 3, size=(8,)))
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

logits = model(inputs)
loss = criterion(logits, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 直接计算梯度

```python
from nanotorch import Tensor

x = Tensor([2.0], requires_grad=True)
y = x * x + 3 * x + 1
y.backward()

print(x.grad.numpy())
```

## 可视化应用

在仓库根目录分别启动前后端：

```bash
# 终端 1
./start-backend.sh

# 终端 2
./start-frontend.sh
```

然后访问 `http://localhost:5173`。

如果你更希望手动启动，或需要 API 请求示例，请查看 [`QUICKSTART_CN.md`](./QUICKSTART_CN.md)。如果你想看可视化模块的详细能力和架构说明，请查看 [`README_VISUALIZATION_CN.md`](./README_VISUALIZATION_CN.md)。

## 包能力概览

### 核心模块

- `nanotorch.tensor`：张量运算、梯度跟踪、`no_grad`
- `nanotorch.autograd`：`Function`、`FunctionContext`、统一的 `backward(...)`
- `nanotorch.nn`：层、损失函数、注意力、Transformer、RNN 等构件
- `nanotorch.optim`：优化器与学习率调度器
- `nanotorch.data`：dataset、sampler、dataloader
- `nanotorch.transforms`：预处理与数据增强工具
- `nanotorch.tokenizer`：char / word / BPE tokenizer
- `nanotorch.device` / `nanotorch.backend`：CPU/CUDA 与后端抽象

### 仓库级扩展内容

仓库中还包含文档、示例、benchmark、可视化前后端，以及目标检测、生成模型等实验性方向。

## 项目结构

```text
nanotorch/
├── nanotorch/              # Python package
├── docs/                   # 设计文档、API 文档、教程
├── tests/                  # 测试
├── examples/               # 示例脚本
├── benchmarks/             # 微基准
├── frontend/               # 可视化前端
├── backend/                # 可视化后端
├── QUICKSTART.md           # Visualization quick start (English default)
├── QUICKSTART_CN.md        # 可视化快速启动说明（中文）
└── README_VISUALIZATION_CN.md # 可视化专项文档（中文）
```

## 文档

- [`docs/design.md`](./docs/design.md)：架构与 autograd 设计
- [`docs/api.md`](./docs/api.md)：API 参考与示例
- [`docs/autograd_boundaries.md`](./docs/autograd_boundaries.md)：autograd 统一后仍保留的原始数组边界说明
- [`QUICKSTART_CN.md`](./QUICKSTART_CN.md)：可视化应用的中文快速启动说明

## 测试

```bash
python -m pytest tests/ -v
```

## 当前限制

- 项目首先服务于教学与实验，不是 PyTorch 的完整替代品。
- 默认路径以 CPU 为主；CUDA 依赖可选的 CuPy，且不同子系统的覆盖程度不完全一致。
- 仓库中的高级或实验性模块成熟度低于核心的 tensor / autograd / nn / optim 主路径。

## License

MIT。
