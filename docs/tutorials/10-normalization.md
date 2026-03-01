# 第十章：归一化层

## 一个合唱团的和谐秘密...

合唱团的每个人都在唱歌。但问题来了——

男高音嗓门大，盖过了女低音。
女中音音调飘，带偏了整个声部。
有人气息不稳，时强时弱。

如果每个人都按自己的节奏来，结果只能是噪音。

于是，指挥家站了出来。他不断调整："男高音，轻一点。女低音，大声些。整体，再稳一点。"

**归一化层，就是神经网络里的"指挥家"。**

在深层网络中，每一层的输出都可能偏离"正轨"——有的数值爆炸，有的数值消失。后面的层就会不知所措："我该学什么？"

归一化层会在每一步都把数据"拉回来"，让它们回到稳定的分布。

```
没有归一化：
  第 1 层输出：[0.001, 0.002, ...]
  第 2 层输出：[0.0001, 0.0002, ...]
  第 3 层输出：[0.00001, ...]  ← 消失了

有归一化：
  每层输出：均值 ≈ 0，方差 ≈ 1
  稳定、可控、可学习
```

**归一化，是深度网络的稳定器。** 它让训练更平稳，收敛更快。

---

## 10.1 为什么需要归一化？

### 问题：内部协变量偏移

```
深度网络的困境：

第1层输出 → 第2层 → 第3层 → ... → 第N层
    ↓           ↓         ↓
  分布变化   分布变化   分布变化

问题：
  - 每层的"输入分布"不断变化
  - 后面的层要不断适应
  - 训练不稳定，收敛慢
```

### 生活类比

```
就像你在考试：

场景1：每次考试难度不一样
  第1次：很简单，平均90分
  第2次：超难，平均50分
  第3次：简单，平均85分
  → 你很难调整学习策略

场景2：每次考试标准化
  第1次：标准化后平均70分
  第2次：标准化后平均70分
  第3次：标准化后平均70分
  → 容易看出你的进步
```

### 解决：归一化

```
归一化的作用：

输入 [很大的数, 很小的数, ...]
        ↓
   减去均值，除以标准差
        ↓
输出 [接近0的数, 接近0的数, ...]

每层输入都稳定 → 训练更稳定
```

---

## 10.2 BatchNorm：按批次归一化

### 原理

```
BatchNorm 对每个通道，在批次内统计：

输入：(N, C, H, W) = (16, 64, 32, 32)
      批次  通道  高   宽

对每个通道 c：
  - 计算这16张图在通道c上的均值和方差
  - 用这个统计量归一化

形象理解：
  假设通道1代表"红色"
  → 看这16张图的红色平均多亮
  → 调整到标准亮度
```

### 图示

```
输入数据 (N=4, C=3):

通道0 (红):          通道1 (绿):         通道2 (蓝):
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ 255 200 180 │     │ 100 120 90  │     │ 50  60  40  │
│ 220 190 210 │     │ 110 100 130 │     │ 55  45  70  │
│ ...         │     │ ...         │     │ ...         │
└─────────────┘     └─────────────┘     └─────────────┘
      ↓                   ↓                   ↓
  mean=200           mean=110            mean=50
  std=30             std=15              std=10
      ↓                   ↓                   ↓
   归一化             归一化              归一化
```

### 训练 vs 推理

```
训练时：
  - 用当前 batch 的均值/方差
  - 同时更新 running_mean/running_var

推理时：
  - 用训练时累积的 running_mean/running_var
  - 因为推理可能只有1个样本，无法计算统计量
```

### 实现

```python
class BatchNorm2d(Module):
    """
    2D 批归一化

    类比：
      训练时 = 现场指挥，根据当前情况调整
      推理时 = 用之前的经验调整
    """

    def __init__(
        self,
        num_features: int,      # 通道数
        eps: float = 1e-5,      # 防止除0
        momentum: float = 0.1,  # running 统计量更新速度
        affine: bool = True,    # 是否学习 gamma 和 beta
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数：缩放和平移
        if affine:
            self.gamma = Tensor.ones((num_features,), requires_grad=True)  # 缩放
            self.beta = Tensor.zeros((num_features,), requires_grad=True)  # 平移

        # 运行时统计量（不是参数，不参与梯度）
        self.running_mean = Tensor.zeros((num_features,))
        self.running_var = Tensor.ones((num_features,))

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.shape

        if self.training:
            # 训练模式：用当前 batch 统计量
            # 在 N, H, W 维度上求均值和方差
            mean = x.mean(axis=(0, 2, 3), keepdims=True)
            var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)

            # 更新 running 统计量
            self.running_mean.data = (1 - self.momentum) * self.running_mean.data + \
                                     self.momentum * mean.squeeze().data
            self.running_var.data = (1 - self.momentum) * self.running_var.data + \
                                    self.momentum * var.squeeze().data
        else:
            # 推理模式：用 running 统计量
            mean = self.running_mean.reshape(1, C, 1, 1)
            var = self.running_var.reshape(1, C, 1, 1)

        # 归一化
        x_norm = (x - mean) / (var + self.eps) ** 0.5

        # 缩放和平移（让网络自己决定最佳分布）
        if hasattr(self, 'gamma'):
            x_norm = self.gamma.reshape(1, C, 1, 1) * x_norm + \
                     self.beta.reshape(1, C, 1, 1)

        return x_norm
```

### 使用

```python
from nanotorch.nn import BatchNorm2d

# 创建 BatchNorm
bn = BatchNorm2d(num_features=64)  # 64个通道

# 训练
bn.train()
output = bn(x_train)

# 推理
bn.eval()
output = bn(x_test)
```

---

## 10.3 LayerNorm：按层归一化

### 原理

```
LayerNorm 对每个样本的所有特征归一化：

输入：(N, C, H, W)
对每个样本 n：
  - 计算这个样本所有特征的均值和方差
  - 不依赖 batch 大小

形象理解：
  每个人自己调整，不管别人
  我看我自己各个科目的分数，调整到平均
```

### BatchNorm vs LayerNorm

```
BatchNorm (跨样本):        LayerNorm (跨特征):

┌─────────────────┐        ┌─────────────────┐
│ 样本1 │ 样本2   │        │ 样本1           │
│  ↓    │  ↓      │        │  所有特征一起   │
│ 统计  │ 统计    │        │  ↓              │
└─────────────────┘        │  统计           │
                           └─────────────────┘
每个通道单独统计             每个样本单独统计

需要大 batch                不依赖 batch
```

### 实现

```python
class LayerNorm(Module):
    """
    层归一化

    常用于：Transformer、NLP 任务
    优点：不依赖 batch 大小，适合变长序列
    """

    def __init__(
        self,
        normalized_shape: int,   # 最后一维的大小
        eps: float = 1e-5,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps

        # 可学习参数
        self.gamma = Tensor.ones((normalized_shape,), requires_grad=True)
        self.beta = Tensor.zeros((normalized_shape,), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, seq_len, hidden_dim)
        # 在最后一个维度上归一化

        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)

        x_norm = (x - mean) / (var + self.eps) ** 0.5

        return self.gamma * x_norm + self.beta
```

### 使用

```python
from nanotorch.nn import LayerNorm

# Transformer 中常用
ln = LayerNorm(normalized_shape=512)  # hidden_dim = 512

x = Tensor.randn((32, 100, 512))  # (batch, seq_len, hidden_dim)
output = ln(x)
```

---

## 10.4 GroupNorm：分组归一化

### 原理

```
GroupNorm 把通道分成几组，每组内归一化：

输入：(N, C, H, W)，C=32，分成8组
→ 每组 4 个通道
→ 在每组的 4 个通道 + H + W 上统计

介于 LayerNorm 和 InstanceNorm 之间
```

### 图解对比

```
输入形状：(N, C, H, W)，假设 C=6

BatchNorm:  对每个通道，跨 (N, H, W) 统计
            通道1: [所有样本的通道1的所有像素]
            通道2: [所有样本的通道2的所有像素]
            ...

LayerNorm:  对每个样本，跨 (C, H, W) 统计
            样本1: [所有通道的所有像素]
            样本2: [所有通道的所有像素]
            ...

GroupNorm:  对每个样本的每组，跨 (C/G, H, W) 统计
            样本1-组1: [通道1,2的所有像素]
            样本1-组2: [通道3,4的所有像素]
            样本1-组3: [通道5,6的所有像素]
            ...

InstanceNorm: 对每个样本的每个通道，跨 (H, W) 统计
            样本1-通道1: [通道1的所有像素]
            样本1-通道2: [通道2的所有像素]
            ...
```

### 实现

```python
class GroupNorm(Module):
    """
    分组归一化

    优点：不依赖 batch 大小
    常用于：目标检测、分割（batch 通常较小）
    """

    def __init__(
        self,
        num_groups: int,      # 组数
        num_channels: int,    # 通道数（必须能被组数整除）
        eps: float = 1e-5,
    ):
        super().__init__()
        assert num_channels % num_groups == 0

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps

        self.gamma = Tensor.ones((num_channels,), requires_grad=True)
        self.beta = Tensor.zeros((num_channels,), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        G = self.num_groups

        # 重塑为 (N, G, C/G, H, W)
        x = x.reshape(N, G, C // G, H, W)

        # 在 (C/G, H, W) 维度上统计
        mean = x.mean(axis=(2, 3, 4), keepdims=True)
        var = ((x - mean) ** 2).mean(axis=(2, 3, 4), keepdims=True)

        x_norm = (x - mean) / (var + self.eps) ** 0.5

        # 恢复形状
        x_norm = x_norm.reshape(N, C, H, W)

        return self.gamma.reshape(1, C, 1, 1) * x_norm + \
               self.beta.reshape(1, C, 1, 1)
```

### 使用

```python
from nanotorch.nn import GroupNorm

# 32 个通道分成 8 组
gn = GroupNorm(num_groups=8, num_channels=32)

x = Tensor.randn((16, 32, 64, 64))
output = gn(x)

# 特例：
# GroupNorm(num_groups=1, ...) ≈ LayerNorm
# GroupNorm(num_groups=C, ...) = InstanceNorm
```

---

## 10.5 归一化方法对比

### 选择指南

```
场景                          推荐
──────────────────────────────────────
CNN 图像分类，batch 大        BatchNorm
CNN，batch 小（<8）           GroupNorm
Transformer / NLP             LayerNorm
风格迁移                      InstanceNorm
目标检测/分割                 GroupNorm
```

### 对比表

| 归一化 | 统计维度 | 依赖 batch | 适用场景 |
|--------|---------|-----------|----------|
| BatchNorm | (N, H, W) | 是 | CNN，大批量 |
| LayerNorm | (C, H, W) | 否 | Transformer |
| GroupNorm | (C/G, H, W) | 否 | 小批量 |
| InstanceNorm | (H, W) | 否 | 风格迁移 |

### 可视化

```
假设输入是 (N=3, C=4, H=2, W=2) 的张量

不同颜色的区域表示一起归一化的元素：

BatchNorm (每通道):
  ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐
  │ C=0   │  │ C=1   │  │ C=2   │  │ C=3   │
  │ 跨NHW │  │ 跨NHW │  │ 跨NHW │  │ 跨NHW │
  └───────┘  └───────┘  └───────┘  └───────┘

LayerNorm (每样本):
  ┌─────────────────────────────────────┐
  │ 样本 0: 跨 C, H, W                  │
  └─────────────────────────────────────┘
  ┌─────────────────────────────────────┐
  │ 样本 1: 跨 C, H, W                  │
  └─────────────────────────────────────┘

GroupNorm (每样本每组，假设2组):
  ┌─────────────┐  ┌─────────────┐
  │ 样本0 组0   │  │ 样本0 组1   │
  │ C=0,1       │  │ C=2,3       │
  └─────────────┘  └─────────────┘
```

---

## 10.6 常见陷阱

### 陷阱1：训练/推理模式混淆

```python
# 错误：推理时没切换模式
model.eval()
output = bn(x)  # 如果 bn 还在 training 模式，会用当前 batch 统计

# 正确
bn.eval()  # 单独设置
# 或
model.eval()  # 整个模型设置
```

### 陷阱2：BatchNorm 用小 batch

```python
# 问题：batch=1 或 2 时，统计量不稳定
bn = BatchNorm2d(64)
x = Tensor.randn((2, 64, 32, 32))  # batch=2 太小

# 解决：用 GroupNorm
gn = GroupNorm(num_groups=32, num_channels=64)
```

### 陷阱3：BatchNorm 后再用 Dropout

```python
# 通常不需要，BatchNorm 本身有正则化效果
model = Sequential(
    Conv2D(64, 128),
    BatchNorm2d(128),
    # Dropout(0.5),  # 通常不需要
    ReLU(),
)
```

---

## 10.7 一句话总结

| 概念 | 一句话 |
|------|--------|
| 归一化 | 让每层输出稳定，加速训练 |
| BatchNorm | 按批次统计，适合大 batch |
| LayerNorm | 按样本统计，适合 Transformer |
| GroupNorm | 按组统计，适合小 batch |
| 训练/推理 | 训练用当前统计，推理用历史统计 |

---

## 下一章

现在我们学会了归一化！

下一章，我们将学习**循环神经网络** —— 处理序列数据的经典架构。

→ [第十一章：循环神经网络](11-rnn.md)

```python
# 预告：下一章你将学到
lstm = LSTM(input_size=64, hidden_size=128)
# 记住历史信息，处理变长序列
```
