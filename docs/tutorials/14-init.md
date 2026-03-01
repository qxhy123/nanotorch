# 教程 14：参数初始化 (Initialization)

## 目录

1. [概述](#概述)
2. [为什么初始化重要](#为什么初始化重要)
3. [常量初始化](#常量初始化)
4. [Xavier/Glorot 初始化](#xavierglorot-初始化)
5. [Kaiming/He 初始化](#kaiminghe-初始化)
6. [正交初始化](#正交初始化)
7. [截断正态分布](#截断正态分布)
8. [使用示例](#使用示例)
9. [总结](#总结)

---

## 概述

参数初始化是深度学习中被低估但极其重要的环节。好的初始化可以：
- **加速收敛**：减少训练时间
- **避免梯度消失/爆炸**：稳定训练
- **提高最终性能**：找到更好的局部最优

nanotorch 提供了多种初始化方法：

| 方法 | 适用场景 |
|------|----------|
| `zeros_`, `ones_`, `constant_` | 偏置初始化 |
| `xavier_uniform_`, `xavier_normal_` | Tanh/Sigmoid 激活 |
| `kaiming_uniform_`, `kaiming_normal_` | ReLU 族激活 |
| `orthogonal_` | RNN/深度网络 |
| `trunc_normal_` | Transformer |
| `sparse_` | 稀疏连接 |

---

## 为什么初始化重要

### 问题：梯度消失/爆炸

```
深层网络中的梯度传播:
∂L/∂W_1 = ∂L/∂W_L × ∏_{i=2}^{L} ∂h_i/∂h_{i-1}

如果每层的梯度 < 1: 指数衰减 (消失)
如果每层的梯度 > 1: 指数增长 (爆炸)
```

### 全零初始化的问题

```python
# 错误示例
W = np.zeros((in_features, out_features))  # 所有神经元相同
# 前向传播：所有输出相同
# 反向传播：所有梯度相同
# 结果：对称性无法打破，网络无法学习
```

### 太大/太小的问题

```
权重太大:    激活值饱和 → 梯度接近 0
权重太小:    激活值太小 → 梯度接近 0
权重适中:    激活值合理 → 梯度有效传播
```

---

## 常量初始化

### 实现

```python
# nanotorch/utils.py

def zeros_(tensor: Tensor) -> Tensor:
    """用 0 填充张量。"""
    tensor.data.fill(0)
    return tensor

def ones_(tensor: Tensor) -> Tensor:
    """用 1 填充张量。"""
    tensor.data.fill(1)
    return tensor

def constant_(tensor: Tensor, value: float) -> Tensor:
    """用常量值填充张量。"""
    tensor.data.fill(value)
    return tensor

def eye_(tensor: Tensor) -> Tensor:
    """用单位矩阵填充 2D 张量。"""
    assert tensor.ndim == 2, "eye_ only supports 2D tensors"
    tensor.data = np.eye(tensor.shape[0], tensor.shape[1], dtype=tensor.data.dtype)
    return tensor

def dirac_(tensor: Tensor) -> Tensor:
    """用 Dirac delta 函数填充卷积核。
    
    保持前向传播的信号强度不变。
    常用于初始化卷积层。
    """
    if tensor.ndim < 2:
        raise ValueError("dirac_ requires at least 2D tensor")
    
    tensor.data.fill(0)
    out_channels = tensor.shape[0]
    in_channels = tensor.shape[1]
    
    # 设置中心位置为 1
    min_channels = min(out_channels, in_channels)
    for c in range(min_channels):
        center_idx = tuple(
            s // 2 if i >= 2 else c
            for i, s in enumerate(tensor.shape)
        )
        tensor.data[center_idx] = 1.0
    
    return tensor
```

### 使用

```python
from nanotorch import Tensor
from nanotorch.utils import zeros_, ones_, constant_

# 初始化偏置
bias = Tensor(np.zeros(128))
zeros_(bias)

# 初始化权重
weight = Tensor(np.empty((256, 128)))
constant_(weight, 0.5)
```

---

## Xavier/Glorot 初始化

### 原理

Xavier 初始化假设激活函数是线性的（如 Tanh），目标是使**前向传播和反向传播的方差保持一致**。

```
方差分析:
- 输入 x: Var(x)
- 权重 W: Var(W) 
- 输出 y = Wx: Var(y) = n_in * Var(W) * Var(x)

为了 Var(y) = Var(x):
Var(W) = 1 / n_in

为了反向传播方差一致:
Var(W) = 1 / n_out

折中:
Var(W) = 2 / (n_in + n_out)
```

### 实现

```python
def xavier_uniform_(
    tensor: Tensor,
    gain: float = 1.0
) -> Tensor:
    """Xavier 均匀分布初始化。
    
    也称为 Glorot 初始化。
    
    权重从均匀分布 U(-a, a) 中采样，其中:
    a = gain * sqrt(6 / (fan_in + fan_out))
    
    Args:
        tensor: 要初始化的张量
        gain: 缩放因子
    
    Returns:
        初始化后的张量
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    a = math.sqrt(3.0) * std  # uniform(-a, a) 的方差为 a^2/3
    
    tensor.data = np.random.uniform(-a, a, tensor.shape).astype(tensor.data.dtype)
    return tensor

def xavier_normal_(
    tensor: Tensor,
    gain: float = 1.0
) -> Tensor:
    """Xavier 正态分布初始化。
    
    权重从正态分布 N(0, std^2) 中采样，其中:
    std = gain * sqrt(2 / (fan_in + fan_out))
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    
    tensor.data = np.random.normal(0, std, tensor.shape).astype(tensor.data.dtype)
    return tensor

def _calculate_fan_in_and_fan_out(tensor: Tensor) -> Tuple[int, int]:
    """计算 fan_in 和 fan_out。
    
    - Linear: fan_in = in_features, fan_out = out_features
    - Conv2D: fan_in = in_channels * kernel_h * kernel_w
              fan_out = out_channels * kernel_h * kernel_w
    """
    if tensor.ndim < 2:
        raise ValueError("Cannot calculate fan_in and fan_out for tensor with less than 2 dimensions")
    
    if tensor.ndim == 2:
        # Linear: (out_features, in_features)
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        # Conv: (out_channels, in_channels, *kernel_size)
        receptive_field_size = 1
        for s in tensor.shape[2:]:
            receptive_field_size *= s
        fan_in = tensor.shape[1] * receptive_field_size
        fan_out = tensor.shape[0] * receptive_field_size
    
    return fan_in, fan_out
```

### 使用

```python
from nanotorch.nn import Linear, Tanh
from nanotorch.utils import xavier_normal_

# Xavier 初始化适合 Tanh/Sigmoid
linear = Linear(128, 64)
xavier_normal_(linear.weight)  # 初始化权重
```

---

## Kaiming/He 初始化

### 原理

Kaiming 初始化专门为 **ReLU 族激活函数**设计。由于 ReLU 会将一半的输入置为 0，需要额外的补偿：

```
ReLU 的方差分析:
- 输入经过 ReLU 后，只有一半的神经元激活
- 为了保持方差，需要将权重的方差加倍

Kaiming 均匀：

$$a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in}}} \quad \text{(fan\_in 模式)}$$

$$a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_out}}} \quad \text{(fan\_out 模式)}$$

对于 ReLU，$\text{gain} = \sqrt{2}$

### 实现

```python
def kaiming_uniform_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu"
) -> Tensor:
    """Kaiming 均匀分布初始化。
    
    也称为 He 初始化。
    
    适用于 ReLU、LeakyReLU、PReLU 等激活函数。
    
    Args:
        tensor: 要初始化的张量
        a: LeakyReLU 的负斜率（用于计算 gain）
        mode: 'fan_in' 或 'fan_out'
        nonlinearity: 激活函数类型
    
    Returns:
        初始化后的张量
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std
    
    tensor.data = np.random.uniform(-bound, bound, tensor.shape).astype(tensor.data.dtype)
    return tensor

def kaiming_normal_(
    tensor: Tensor,
    a: float = 0,
    mode: str = "fan_in",
    nonlinearity: str = "leaky_relu"
) -> Tensor:
    """Kaiming 正态分布初始化。
    
    权重从 N(0, std^2) 采样，其中:
    std = gain / sqrt(fan)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    fan = fan_in if mode == "fan_in" else fan_out
    
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    
    tensor.data = np.random.normal(0, std, tensor.shape).astype(tensor.data.dtype)
    return tensor

def calculate_gain(nonlinearity: str, param: float = 0) -> float:
    """计算激活函数的 gain 值。"""
    if nonlinearity == "linear" or nonlinearity == "sigmoid":
        return 1.0
    elif nonlinearity == "tanh":
        return 5.0 / 3
    elif nonlinearity == "relu":
        return math.sqrt(2.0)
    elif nonlinearity == "leaky_relu":
        return math.sqrt(2.0 / (1 + param ** 2))
    elif nonlinearity == "selu":
        return 3.0 / 4
    else:
        raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")
```

### 使用

```python
from nanotorch.nn import Linear, ReLU, Conv2D
from nanotorch.utils import kaiming_normal_, zeros_

# Kaiming 初始化适合 ReLU
linear = Linear(128, 64)
kaiming_normal_(linear.weight, nonlinearity="relu")
zeros_(linear.bias)

# 卷积层
conv = Conv2D(3, 64, kernel_size=3)
kaiming_normal_(conv.weight, nonlinearity="relu")
zeros_(conv.bias)

# LeakyReLU
from nanotorch.nn import LeakyReLU
lrelu = LeakyReLU(negative_slope=0.01)
linear2 = Linear(64, 32)
kaiming_normal_(linear2.weight, a=0.01, nonlinearity="leaky_relu")
```

---

## 正交初始化

### 原理

正交初始化使权重矩阵的行（或列）相互正交：
- 前向传播：保持信号强度
- 反向传播：梯度不相关
- 特别适合 RNN，防止梯度消失/爆炸

### 实现

```python
def orthogonal_(
    tensor: Tensor,
    gain: float = 1.0
) -> Tensor:
    """正交矩阵初始化。
    
    使用 QR 分解生成正交矩阵。
    
    Args:
        tensor: 要初始化的张量（至少 2D）
        gain: 缩放因子
    
    Returns:
        初始化后的张量
    """
    if tensor.ndim < 2:
        raise ValueError("orthogonal_ requires at least 2D tensor")
    
    # 展平除最后两维外的所有维度
    original_shape = tensor.shape
    if tensor.ndim > 2:
        flat_shape = (np.prod(original_shape[:-2]), *original_shape[-2:])
    else:
        flat_shape = original_shape
    
    rows, cols = flat_shape[-2], flat_shape[-1]
    
    # 生成随机矩阵
    flat_tensor = np.random.randn(*flat_shape).astype(np.float32)
    
    # QR 分解
    if rows >= cols:
        q, r = np.linalg.qr(flat_tensor.reshape(-1, cols))
        q = q.reshape(flat_shape)
    else:
        # 转置后做 QR
        flat_tensor = flat_tensor.transpose(-1, -2)
        q, r = np.linalg.qr(flat_tensor.reshape(-1, rows))
        q = q.reshape(flat_shape[:-2] + (cols, rows))
        q = q.transpose(-1, -2)
    
    # 应用 gain
    q = q * gain
    
    tensor.data = q.reshape(original_shape).astype(tensor.data.dtype)
    return tensor
```

### 使用

```python
from nanotorch.nn import LSTM
from nanotorch.utils import orthogonal_, zeros_

# RNN 使用正交初始化
lstm = LSTM(input_size=64, hidden_size=128)

for name, param in lstm.named_parameters():
    if 'weight' in name:
        orthogonal_(param)
    elif 'bias' in name:
        zeros_(param)
```

---

## 截断正态分布

### 原理

截断正态分布将采样值限制在 `[mean - 2*std, mean + 2*std]` 范围内，避免极端值。常用于 Transformer 初始化。

### 实现

```python
def trunc_normal_(
    tensor: Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0
) -> Tensor:
    """截断正态分布初始化。
    
    从 N(mean, std^2) 采样，但限制在 [a, b] 范围内。
    
    Args:
        tensor: 要初始化的张量
        mean: 正态分布均值
        std: 正态分布标准差
        a: 截断下界（以 std 为单位）
        b: 截断上界（以 std 为单位）
    
    Returns:
        初始化后的张量
    """
    # 生成截断正态分布
    lower = mean + a * std
    upper = mean + b * std
    
    # 使用拒绝采样
    size = tensor.shape
    result = np.zeros(size)
    
    remaining = np.ones(size, dtype=bool)
    while remaining.any():
        # 生成候选值
        candidates = np.random.normal(mean, std, size)
        
        # 接受在范围内的值
        valid = (candidates >= lower) & (candidates <= upper)
        result = np.where(remaining & valid, candidates, result)
        remaining = remaining & ~valid
    
    tensor.data = result.astype(tensor.data.dtype)
    return tensor
```

### 使用

```python
from nanotorch.nn import Linear
from nanotorch.utils import trunc_normal_

# Transformer 风格初始化
linear = Linear(512, 512)
trunc_normal_(linear.weight, std=0.02)  # Transformer 常用 std=0.02
```

---

## 稀疏初始化

### 实现

```python
def sparse_(
    tensor: Tensor,
    sparsity: float,
    std: float = 0.01
) -> Tensor:
    """稀疏初始化。
    
    大部分权重为 0，只有少部分非零。
    
    Args:
        tensor: 要初始化的张量
        sparsity: 稀疏度（0 的比例）
        std: 非零值的标准差
    
    Returns:
        初始化后的张量
    """
    tensor.data.fill(0)
    
    # 随机选择非零位置
    total_elements = tensor.data.size
    num_nonzero = int(total_elements * (1 - sparsity))
    
    flat_indices = np.random.choice(total_elements, num_nonzero, replace=False)
    flat_tensor = tensor.data.flatten()
    flat_tensor[flat_indices] = np.random.normal(0, std, num_nonzero)
    
    tensor.data = flat_tensor.reshape(tensor.shape)
    return tensor
```

---

## 使用示例

### 初始化辅助函数

```python
from nanotorch import Tensor
from nanotorch.nn import Module, Linear, Conv2D, ReLU, BatchNorm2d
from nanotorch.utils import kaiming_normal_, xavier_normal_, zeros_, ones_

def init_weights(module: Module, init_type: str = "kaiming"):
    """统一的权重初始化函数。"""
    
    for name, param in module.named_parameters():
        if 'weight' in name:
            if init_type == "kaiming":
                kaiming_normal_(param, nonlinearity="relu")
            elif init_type == "xavier":
                xavier_normal_(param)
            elif init_type == "normal":
                param.data = np.random.normal(0, 0.02, param.shape).astype(np.float32)
        elif 'bias' in name:
            zeros_(param)

def init_bn(module: Module):
    """初始化 BatchNorm 层。"""
    for name, param in module.named_parameters():
        if 'weight' in name or 'gamma' in name:
            ones_(param)
        elif 'bias' in name or 'beta' in name:
            zeros_(param)
```

### CNN 初始化

```python
class SimpleCNN(Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = Conv2D(3, 64, kernel_size=3, padding=1)
        self.bn1 = BatchNorm2d(64)
        self.conv2 = Conv2D(64, 128, kernel_size=3, padding=1)
        self.bn2 = BatchNorm2d(128)
        self.fc = Linear(128 * 28 * 28, num_classes)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, Conv2D):
                kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    zeros_(module.bias)
            elif isinstance(module, BatchNorm2d):
                ones_(module.weight)
                zeros_(module.bias)
            elif isinstance(module, Linear):
                kaiming_normal_(module.weight, nonlinearity="relu")
                zeros_(module.bias)
```

### Transformer 初始化

```python
def init_transformer(module):
    """Transformer 风格初始化。"""
    for name, param in module.named_parameters():
        if 'weight' in name:
            if 'layernorm' in name.lower() or 'norm' in name.lower():
                ones_(param)
            else:
                trunc_normal_(param, std=0.02)
        elif 'bias' in name:
            zeros_(param)
```

### ResNet 初始化

```python
def init_resnet(module):
    """ResNet 风格初始化。
    
    - 卷积层：Kaiming normal
    - BN 层：weight=1, bias=0
    - 最后的 FC：特殊初始化
    """
    for name, module in module.named_modules():
        if isinstance(module, Conv2D):
            kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, BatchNorm2d):
            ones_(module.weight)
            zeros_(module.bias)
        elif isinstance(module, Linear):
            # 最后的 FC 层初始化为较小的值
            module.weight.data = np.random.normal(0, 0.01, module.weight.shape).astype(np.float32)
            zeros_(module.bias)
```

---

## 总结

本教程介绍了 nanotorch 中的参数初始化方法：

| 方法 | 公式 | 适用场景 |
|------|------|----------|
| **zeros/ones/constant** | 常量 | 偏置 |
| **Xavier** | $\sqrt{\frac{2}{n_{in}+n_{out}}}$ | Tanh/Sigmoid |
| **Kaiming** | $\sqrt{\frac{2}{n_{in}}}$ | ReLU 族 |
| **Orthogonal** | QR 分解 | RNN |
| **Trunc Normal** | 截断正态 | Transformer |

### 关键要点

1. **ReLU 使用 Kaiming**，**Tanh 使用 Xavier**
2. **偏置通常初始化为 0**
3. **BatchNorm**：weight=1, bias=0
4. **RNN**：正交初始化防止梯度问题
5. **Transformer**：截断正态，std=0.02

### 下一步

在 [教程 15：高级主题](15-advanced.md) 中，我们将探讨一些高级主题，包括梯度裁剪、混合精度训练等。

---

**参考资源**：
- [Understanding the difficulty of training deep feedforward neural networks (Xavier)](http://proceedings.mlr.press/v9/glorot10a.html)
- [Delving Deep into Rectifiers (Kaiming)](https://arxiv.org/abs/1502.01852)
- [All you need is a good init](https://arxiv.org/abs/1511.06422)
