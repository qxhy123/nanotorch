# 教程 10：归一化层 (Normalization Layers)

## 目录

1. [概述](#概述)
2. [为什么需要归一化](#为什么需要归一化)
3. [BatchNorm 实现](#batchnorm-实现)
4. [LayerNorm 实现](#layernorm-实现)
5. [GroupNorm 实现](#groupnorm-实现)
6. [InstanceNorm 实现](#instancenorm-实现)
7. [各归一化方法的比较](#各归一化方法的比较)
8. [使用示例](#使用示例)
9. [总结](#总结)

---

## 概述

归一化层是现代深度神经网络的核心组件，它们能够：
- **加速训练**：允许使用更大的学习率
- **稳定训练**：减少内部协变量偏移（Internal Covariate Shift）
- **正则化效果**：BatchNorm 具有一定的正则化作用

nanotorch 实现了以下归一化层：
- **BatchNorm1d/2d/3d**：批归一化
- **LayerNorm**：层归一化
- **GroupNorm**：组归一化
- **InstanceNorm1d/2d/3d**：实例归一化

---

## 为什么需要归一化

### 内部协变量偏移

在深层网络中，每层的输入分布会随着前层参数更新而变化，这导致：
1. 每层需要不断适应新的输入分布
2. 学习率需要设得很小
3. 训练不稳定，收敛慢

### 归一化的作用

```
归一化前:                   归一化后:
    ┌─────┐                     ┌─────┐
    │ 很大 │                     │ 均值0│
    │ 方差 │                     │ 方差1│
    └─────┘                     └─────┘
   输入分布不稳定              输入分布稳定
```

通过归一化，我们将每层的输入稳定在均值0、方差1附近，使训练更加稳定。

---

## BatchNorm 实现

### 原理

BatchNorm 对**一个批次内**的样本在**每个通道**上分别计算均值和方差：

```
输入: (N, C, H, W)
对每个通道 c:
    mean = mean(x[:, c, :, :])  # 在 N, H, W 上求平均
    var = var(x[:, c, :, :])
    x_norm[:, c, :, :] = (x[:, c, :, :] - mean) / sqrt(var + eps)
    output[:, c, :, :] = gamma * x_norm[:, c, :, :] + beta
```

### 训练与推理的区别

| 阶段 | 使用的统计量 |
|------|--------------|
| 训练 | 当前 batch 的均值和方差 |
| 推理 | 训练时累积的 running mean/var |

### 基类实现

```python
# nanotorch/nn/normalization.py

class _BatchNorm(Module):
    """BatchNorm 基类"""
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # 可学习参数
        self.gamma = None  # 缩放参数
        self.beta = None   # 平移参数
        if self.affine:
            self.gamma = Tensor.ones((num_features,), requires_grad=True)
            self.beta = Tensor.zeros((num_features,), requires_grad=True)
        
        # 运行时统计量
        self.running_mean = None
        self.running_var = None
        if self.track_running_stats:
            self.running_mean = Tensor.zeros((num_features,), requires_grad=False)
            self.running_var = Tensor.ones((num_features,), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        N, C = x.shape[0], x.shape[1]
        # 在批次和空间维度上求均值
        axes = (0,) + tuple(range(2, x.ndim))
        
        if self.training and self.track_running_stats:
            # 计算当前 batch 的统计量
            mean = x.mean(axis=axes, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
            
            # 更新 running 统计量
            mean_squeezed = mean.squeeze(axis=axes)
            var_squeezed = var.squeeze(axis=axes)
            
            if self.momentum is not None:
                self.running_mean.data = (
                    1 - self.momentum
                ) * self.running_mean.data + self.momentum * mean_squeezed.data
                self.running_var.data = (
                    1 - self.momentum
                ) * self.running_var.data + self.momentum * var_squeezed.data
            
            # 使用 batch 统计量归一化
            x_normalized = (x - mean) / (var + self.eps) ** 0.5
        else:
            # 使用 running 统计量
            broadcast_shape = (1, C) + (1,) * (x.ndim - 2)
            mean = self.running_mean.reshape(broadcast_shape)
            var = self.running_var.reshape(broadcast_shape)
            x_normalized = (x - mean) / (var + self.eps) ** 0.5
        
        # 仿射变换
        if self.affine:
            broadcast_shape = (1, C) + (1,) * (x.ndim - 2)
            gamma_reshaped = self.gamma.reshape(broadcast_shape)
            beta_reshaped = self.beta.reshape(broadcast_shape)
            x_normalized = gamma_reshaped * x_normalized + beta_reshaped
        
        return x_normalized
```

### BatchNorm2d 实现

```python
class BatchNorm2d(_BatchNorm):
    """2D Batch Normalization.
    
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W)
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
    
    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(f"BatchNorm2d expects 4D input, got {x.ndim}D")
```

### 使用示例

```python
from nanotorch import Tensor
from nanotorch.nn import BatchNorm2d

# 创建 BatchNorm 层
bn = BatchNorm2d(num_features=64)

# 训练模式
bn.train()
x = Tensor.randn((16, 64, 32, 32))
output = bn(x)

# 推理模式
bn.eval()
x_test = Tensor.randn((1, 64, 32, 32))
output_test = bn(x_test)  # 使用 running statistics
```

---

## LayerNorm 实现

### 原理

LayerNorm 对**每个样本**在**特征维度**上归一化，不依赖 batch 统计量：

```
输入: (N, C, H, W)
对每个样本 n:
    mean = mean(x[n, :, :, :])  # 在 C, H, W 上求平均
    var = var(x[n, :, :, :])
    x_norm[n, :, :, :] = (x[n, :, :, :] - mean) / sqrt(var + eps)
```

### 实现

```python
class LayerNorm(Module):
    """Layer Normalization.
    
    在最后一个（或最后几个）维度上归一化。
    常用于 Transformer 和 NLP 任务。
    
    Shape:
        - Input: (*, normalized_shape)
        - Output: (*, normalized_shape)
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        self.gamma = None
        self.beta = None
        if self.elementwise_affine:
            self.gamma = Tensor.ones(normalized_shape, requires_grad=True)
            self.beta = Tensor.zeros(normalized_shape, requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # 使用 LayerNormFunction 进行前向传播
        return LayerNormFunction.apply(
            x, self.normalized_shape, self.gamma, self.beta, self.eps
        )
```

### 使用示例

```python
from nanotorch.nn import LayerNorm

# 对最后一个维度归一化
ln = LayerNorm(normalized_shape=512)

# Transformer 中的使用
x = Tensor.randn((32, 100, 512))  # (batch, seq_len, hidden_dim)
output = ln(x)

# 对最后两个维度归一化
ln_2d = LayerNorm(normalized_shape=(64, 64))
x_2d = Tensor.randn((8, 3, 64, 64))
output_2d = ln_2d(x_2d)
```

---

## GroupNorm 实现

### 原理

GroupNorm 将通道分成若干组，在每组的通道和空间维度上归一化：

```
输入: (N, C, H, W), groups = G
将 C 个通道分成 G 组，每组 C/G 个通道
对每个样本 n 和每组 g:
    mean = mean(x[n, g*C/G:(g+1)*C/G, :, :])
    var = var(x[n, g*C/G:(g+1)*C/G, :, :])
```

### 实现

```python
class GroupNorm(Module):
    """Group Normalization.
    
    将通道分组后在组内归一化。
    介于 LayerNorm 和 InstanceNorm 之间。
    
    Args:
        num_groups: 组数
        num_channels: 通道数（必须能被 num_groups 整除）
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        self.gamma = None
        self.beta = None
        if self.affine:
            self.gamma = Tensor.ones((num_channels,), requires_grad=True)
            self.beta = Tensor.zeros((num_channels,), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return GroupNormFunction.apply(
            x, self.num_groups, self.gamma, self.beta, self.eps
        )
```

### 使用示例

```python
from nanotorch.nn import GroupNorm

# 32 个通道分成 8 组
gn = GroupNorm(num_groups=8, num_channels=32)

x = Tensor.randn((16, 32, 64, 64))
output = gn(x)

# 特例：GroupNorm(num_groups=1, ...) = LayerNorm (通道维度)
# 特例：GroupNorm(num_groups=C, ...) = InstanceNorm
```

---

## InstanceNorm 实现

### 原理

InstanceNorm 对每个样本的每个通道独立归一化：

```
输入: (N, C, H, W)
对每个样本 n 和每个通道 c:
    mean = mean(x[n, c, :, :])
    var = var(x[n, c, :, :])
```

### 实现

```python
class InstanceNorm2d(_InstanceNorm):
    """Instance Normalization for 2D inputs.
    
    常用于风格迁移等任务。
    等价于 GroupNorm(num_groups=num_features)。
    
    Shape:
        - Input: (N, C, H, W) or (C, H, W)
        - Output: same as input
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,  # InstanceNorm 默认不学习参数
        track_running_stats: bool = False,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            num_spatial_dims=2,
        )
```

---

## 各归一化方法的比较

### 归一化维度对比

假设输入形状为 `(N, C, H, W)`：

```
BatchNorm:    对 (N, H, W) 求统计量，每个通道独立
LayerNorm:    对 (C, H, W) 求统计量，每个样本独立
InstanceNorm: 对 (H, W) 求统计量，每个样本的每个通道独立
GroupNorm:    对 (C/G, H, W) 求统计量，每组独立
```

### 可视化对比

```
输入张量 (N=2, C=4, H=2, W=2):

BatchNorm (跨 N, H, W):  LayerNorm (跨 C, H, W):
┌───┬───┐                ┌───┬───┐
│ N │ N │                │C=1│C=2│
│ 0 │ 1 │                │   │   │
├───┼───┤                ├───┼───┤
│ N │ N │                │C=3│C=4│
│ 0 │ 1 │                │   │   │
└───┴───┘                └───┴───┘

GroupNorm (G=2):         InstanceNorm:
┌─────┬─────┐            ┌───┬───┐
│G=1  │G=2  │            │N=0│N=1│
│C=1,2│C=3,4│            │每个│每个│
└─────┴─────┘            │通道│通道│
                         │独立│独立│
                         └───┴───┘
```

### 选择指南

| 场景 | 推荐归一化 | 原因 |
|------|-----------|------|
| CNN 图像分类 | BatchNorm | 批量大时效果好 |
| 小批量/内存受限 | GroupNorm | 不依赖批量大小 |
| RNN/NLP | LayerNorm | 处理变长序列 |
| 风格迁移 | InstanceNorm | 保留内容，忽略风格 |
| 目标检测/分割 | GroupNorm 或 SyncBN | 批量通常较小 |
| Transformer | LayerNorm | 标准 Transformer 架构 |

---

## 使用示例

### CNN 中的 BatchNorm

```python
from nanotorch import Tensor
from nanotorch.nn import Conv2D, BatchNorm2d, ReLU, Sequential

# 标准 CNN 块
def conv_block(in_ch, out_ch):
    return Sequential(
        Conv2D(in_ch, out_ch, kernel_size=3, padding=1),
        BatchNorm2d(out_ch),
        ReLU(),
    )

block = conv_block(64, 128)
x = Tensor.randn((4, 64, 32, 32))
output = block(x)
```

### Transformer 中的 LayerNorm

```python
from nanotorch.nn import Linear, LayerNorm, Dropout, ReLU

class TransformerBlock:
    def __init__(self, d_model=512, d_ff=2048):
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.ff = Sequential(
            Linear(d_model, d_ff),
            ReLU(),
            Linear(d_ff, d_model),
        )
    
    def __call__(self, x):
        # Pre-norm 架构
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x
```

### ResNet 中的 GroupNorm

```python
class ResBlock:
    def __init__(self, channels, groups=32):
        self.conv1 = Conv2D(channels, channels, 3, padding=1)
        self.gn1 = GroupNorm(groups, channels)
        self.conv2 = Conv2D(channels, channels, 3, padding=1)
        self.gn2 = GroupNorm(groups, channels)
    
    def __call__(self, x):
        identity = x
        out = self.gn1(self.conv1(x)).relu()
        out = self.gn2(self.conv2(out))
        return (out + identity).relu()
```

---

## 总结

本教程介绍了 nanotorch 中的四种归一化层：

| 归一化 | 归一化维度 | 特点 | 适用场景 |
|--------|-----------|------|----------|
| **BatchNorm** | 批次+空间 | 依赖批次大小 | CNN，大批量训练 |
| **LayerNorm** | 特征+空间 | 不依赖批次 | Transformer，NLP |
| **GroupNorm** | 分组内 | 不依赖批次 | 小批量，检测分割 |
| **InstanceNorm** | 单通道空间 | 不依赖批次 | 风格迁移 |

### 关键要点

1. **BatchNorm** 训练和推理行为不同（running statistics）
2. **LayerNorm** 适合处理变长序列
3. **GroupNorm** 是 BatchNorm 在小批量下的良好替代
4. **InstanceNorm** 常用于生成任务

### 下一步

在 [教程 11：循环神经网络](11-rnn.md) 中，我们将学习如何实现 RNN、LSTM 和 GRU。

---

**参考资源**：
- [Batch Normalization: Accelerating Deep Network Training](https://arxiv.org/abs/1502.03167)
- [Layer Normalization](https://arxiv.org/abs/1607.06450)
- [Group Normalization](https://arxiv.org/abs/1803.08494)
- [Instance Normalization](https://arxiv.org/abs/1607.08022)
