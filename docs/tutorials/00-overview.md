# nanotorch 系列教程：从零开始打造深度学习框架

## 教程简介

本系列教程将带你从零开始实现一个完整的深度学习框架——nanotorch。通过这个系列，你将深入理解：

- **张量（Tensor）** 的底层实现与数学运算
- **自动微分（Autograd）** 的工作原理
- **神经网络层** 的设计与实现
- **优化器** 如何更新模型参数
- **卷积、循环、注意力** 等核心机制

## 为什么学习这个？

1. **深入理解 PyTorch**：了解 PyTorch 内部是如何工作的
2. **巩固深度学习基础**：从数学到代码的完整实现
3. **提升编程能力**：学习设计 API 和组织代码结构
4. **面试加分项**：展示你对底层原理的理解

## 前置知识

- Python 编程基础
- NumPy 数组操作
- 基础线性代数（矩阵运算）
- 基础微积分（导数、链式法则）
- 深度学习基础概念（神经网络、反向传播）

> 💡 **提示**：如果你需要复习数学基础，可以先阅读 [数学基础：深度学习必备知识](math-fundamentals.md)，涵盖线性代数、微积分、概率论、最优化等内容。

## 教程目录

### 数学基础（选修）

| 教程 | 主题 | 内容 |
|------|------|------|
| [math-fundamentals.md](math-fundamentals.md) | 数学基础 | 线性代数、微积分、概率论、最优化 |

### 第一部分：核心基础

| 教程 | 主题 | 内容 |
|------|------|------|
| [01-tensor.md](01-tensor.md) | Tensor 基础 | 张量数据结构、运算、形状操作 |
| [02-autograd.md](02-autograd.md) | 自动微分 | 计算图、反向传播、梯度计算 |

### 第二部分：神经网络模块

| 教程 | 主题 | 内容 |
|------|------|------|
| [03-nn-module.md](03-nn-module.md) | Module 基类 | 参数管理、模块组合、Sequential |
| [04-activation.md](04-activation.md) | 激活函数 | ReLU、Sigmoid、Softmax 等 |
| [05-loss.md](05-loss.md) | 损失函数 | MSE、CrossEntropy、BCE |
| [06-optimizer.md](06-optimizer.md) | 优化器 | SGD、Adam、学习率调度 |

### 第三部分：训练与数据

| 教程 | 主题 | 内容 |
|------|------|------|
| [07-training.md](07-training.md) | 训练循环 | 完整训练流程、验证、保存模型 |
| [08-transforms.md](08-transforms.md) | 数据增强 | 图像变换、归一化、数据增强 |

### 第四部分：高级层

| 教程 | 主题 | 内容 |
|------|------|------|
| [09-conv.md](09-conv.md) | 卷积层 | Conv1D/2D/3D、转置卷积 |
| [10-normalization.md](10-normalization.md) | 归一化 | BatchNorm、LayerNorm、GroupNorm |
| [11-rnn.md](11-rnn.md) | 循环网络 | RNN、LSTM、GRU |
| [12-transformer.md](12-transformer.md) | Transformer | 注意力机制、位置编码、多头注意力 |

### 第五部分：进阶主题

| 教程 | 主题 | 内容 |
|------|------|------|
| [13-dataloader.md](13-dataloader.md) | 数据加载 | Dataset、DataLoader、采样器 |
| [14-init.md](14-init.md) | 参数初始化 | Xavier、Kaiming、正交初始化 |
| [15-advanced.md](15-advanced.md) | 高级主题 | 梯度裁剪、学习率预热、调试技巧 |

## 项目结构

```
nanotorch/
├── nanotorch/                 # 核心库
│   ├── __init__.py
│   ├── tensor.py             # 张量实现
│   ├── autograd.py           # 自动微分
│   ├── utils.py              # 工具函数（梯度裁剪、初始化等）
│   ├── nn/                   # 神经网络模块
│   │   ├── __init__.py
│   │   ├── module.py         # 模块基类
│   │   ├── linear.py         # 全连接层
│   │   ├── conv.py           # 卷积层
│   │   ├── activation.py     # 激活函数
│   │   ├── loss.py           # 损失函数
│   │   ├── dropout.py        # Dropout
│   │   ├── pooling.py        # 池化层
│   │   ├── normalization.py  # 归一化层
│   │   ├── rnn.py            # RNN/LSTM/GRU
│   │   ├── attention.py      # 注意力机制
│   │   ├── transformer.py    # Transformer
│   │   └── embedding.py      # 嵌入层
│   ├── optim/                # 优化器
│   │   ├── __init__.py
│   │   ├── optimizer.py      # 优化器基类
│   │   ├── sgd.py            # SGD
│   │   ├── adam.py           # Adam
│   │   ├── adamw.py          # AdamW
│   │   ├── rmsprop.py        # RMSprop
│   │   ├── adagrad.py        # Adagrad
│   │   └── lr_scheduler.py   # 学习率调度器
│   ├── data/                 # 数据加载
│   │   └── __init__.py       # Dataset, DataLoader
│   └── transforms/           # 数据增强
│       └── __init__.py       # 图像变换
├── tests/                    # 测试（边学边写）
│   ├── test_tensor.py
│   ├── test_autograd.py
│   ├── test_nn.py
│   ├── test_optim.py
│   └── ...
├── examples/                 # 示例代码
│   ├── simple_neural_net.py
│   ├── mnist_classifier.py
│   ├── mini_gpt.py
│   └── ...
├── benchmarks/               # 性能基准测试
├── docs/                     # 文档
│   ├── tutorials/            # 本教程
│   ├── api.md
│   └── design.md
├── README.md                 # 英文文档
├── README_CN.md              # 中文文档
└── pyproject.toml            # 项目配置
```

## 学习建议

### 边学边写

每学完一章，建议：

1. **自己实现一遍**：不要只看代码，亲手敲出来
2. **写测试用例**：验证你的实现是否正确
3. **调试和实验**：打印中间结果，理解数据流动
4. **与 PyTorch 对比**：用同样的数据验证输出是否一致

### 调试技巧

```python
# 打印张量形状，理解维度变化
print(f"输入形状: {x.shape}")
print(f"输出形状: {y.shape}")

# 打印梯度，理解反向传播
print(f"权重梯度: {w.grad}")

# 与 NumPy 对比，验证计算
expected = np_function(x.data)
actual = y.data
print(f"差异: {np.abs(expected - actual).max()}")
```

## 资源推荐

### 必读

- [PyTorch 官方文档](https://pytorch.org/docs/)：API 参考
- [深度学习](https://www.deeplearningbook.org/)：Goodfellow 等著
- [自动微分](https://en.wikipedia.org/wiki/Automatic_differentiation)：维基百科

### 开源参考

- [micrograd](https://github.com/karpathy/micrograd)：Karpathy 的微型 autograd
- [tinygrad](https://github.com/tinygrad/tinygrad)：小型深度学习框架
- [PyTorch 源码](https://github.com/pytorch/pytorch)：官方实现

## 开始学习

准备好了吗？让我们从 [第一章：Tensor 基础](01-tensor.md) 开始！

```python
# 你的第一个 nanotorch 代码
from nanotorch import Tensor

x = Tensor([1, 2, 3, 4, 5])
print(x)  # Tensor(shape=(5,), requires_grad=False)

y = x * 2 + 1
print(y)  # Tensor(shape=(5,), ...)
```
