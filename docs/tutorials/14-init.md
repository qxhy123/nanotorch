# 第十四章：参数初始化

## 想象你在准备一场射箭比赛...

你要让100个射手同时射箭：
- 如果大家都站在同一个位置 → 箭都落在同一个地方
- 如果大家随机分布 → 箭散落在靶场各处
- 如果分布得当 → 覆盖整个靶面

```
全零初始化：
  所有人站在同一位置
  → 所有箭落在同一点
  → 没法知道哪里更准

好的初始化：
  每个人站不同位置
  → 箭散布开来
  → 能看出哪里更准，往哪调整
```

**初始化就是给参数"选起点"** —— 好的起点让训练更快收敛。

---

## 14.1 为什么初始化很重要？

### 问题1：全零初始化

```python
# 错误：全零
W = np.zeros((128, 64))

问题：
  - 所有神经元输出相同
  - 所有梯度相同
  - 更新后所有权重还相同
  → 对称性无法打破，网络学不到东西
```

### 问题2：权重太大

```python
# 问题：数值太大
W = np.random.randn(128, 64) * 10

前向传播：
  输出 = 输入 × W → 很大的数
  经过激活函数 → 饱和（输出接近0或1）
  梯度 → 接近0
→ 梯度消失，学不动
```

### 问题3：权重太小

```python
# 问题：数值太小
W = np.random.randn(128, 64) * 0.001

前向传播：
  输出 = 输入 × W → 很小的数
  逐层传播 → 越来越小
  梯度 → 也越来越小
→ 梯度消失，学不动
```

### 解决：恰当的初始化

```
好的初始化：

权重不大不小，刚好让：
  - 前向传播：信号不衰减
  - 反向传播：梯度不消失

类比：射箭
  - 太近：箭扎不到靶心
  - 太远：箭飞不到靶子
  - 刚好：箭能覆盖靶面
```

---

## 14.2 Xavier 初始化

### 原理

```
Xavier 初始化适合 Tanh/Sigmoid 激活

目标：前向和反向传播时，方差保持不变

推导：
  输入 x，权重 W，输出 y = W @ x

  Var(y) = n_in × Var(W) × Var(x)

  要使 Var(y) = Var(x)：
  Var(W) = 1 / n_in

  反向传播同理：
  Var(W) = 1 / n_out

  折中：
  Var(W) = 2 / (n_in + n_out)
```

### 公式

```
Xavier 均匀分布：
  W ~ U(-a, a)
  a = sqrt(6 / (n_in + n_out))

Xavier 正态分布：
  W ~ N(0, std²)
  std = sqrt(2 / (n_in + n_out))
```

### 实现

```python
def xavier_normal_(tensor, gain=1.0):
    """
    Xavier 正态初始化

    适合：Tanh、Sigmoid 激活函数
    """
    # 计算 fan_in 和 fan_out
    if tensor.ndim == 2:
        # Linear: (out_features, in_features)
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        # Conv: (out_channels, in_channels, *kernel)
        receptive_field = np.prod(tensor.shape[2:])
        fan_in = tensor.shape[1] * receptive_field
        fan_out = tensor.shape[0] * receptive_field

    std = gain * np.sqrt(2.0 / (fan_in + fan_out))

    tensor.data = np.random.normal(0, std, tensor.shape).astype(np.float32)
    return tensor
```

### 使用

```python
from nanotorch.nn import Linear, Tanh
from nanotorch.utils import xavier_normal_

# Xavier 适合 Tanh
linear = Linear(128, 64)
xavier_normal_(linear.weight)  # 初始化权重
```

---

## 14.3 Kaiming 初始化

### 原理

```
Kaiming 初始化适合 ReLU 族激活

ReLU 会把一半输入变成0，所以需要额外补偿

Var(y) = (n_in/2) × Var(W) × Var(x)

要使 Var(y) = Var(x)：
Var(W) = 2 / n_in
```

### 公式

```
Kaiming 均匀分布：
  W ~ U(-a, a)
  a = sqrt(6 / n_in)  (for ReLU)

Kaiming 正态分布：
  W ~ N(0, std²)
  std = sqrt(2 / n_in)  (for ReLU)
```

### 实现

```python
def kaiming_normal_(tensor, a=0, mode='fan_in', nonlinearity='relu'):
    """
    Kaiming 正态初始化

    适合：ReLU、LeakyReLU 激活函数

    Args:
        a: LeakyReLU 的负斜率
        mode: 'fan_in' 或 'fan_out'
        nonlinearity: 激活函数类型
    """
    # 计算 fan_in 和 fan_out
    if tensor.ndim == 2:
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        receptive_field = np.prod(tensor.shape[2:])
        fan_in = tensor.shape[1] * receptive_field
        fan_out = tensor.shape[0] * receptive_field

    fan = fan_in if mode == 'fan_in' else fan_out

    # 计算 gain
    if nonlinearity == 'relu':
        gain = np.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        gain = np.sqrt(2.0 / (1 + a**2))
    else:
        gain = 1.0

    std = gain / np.sqrt(fan)

    tensor.data = np.random.normal(0, std, tensor.shape).astype(np.float32)
    return tensor
```

### 使用

```python
from nanotorch.nn import Linear, ReLU, Conv2D
from nanotorch.utils import kaiming_normal_, zeros_

# Kaiming 适合 ReLU
linear = Linear(128, 64)
kaiming_normal_(linear.weight, nonlinearity='relu')
zeros_(linear.bias)  # 偏置初始化为0

# 卷积层也用 Kaiming
conv = Conv2D(3, 64, kernel_size=3)
kaiming_normal_(conv.weight, nonlinearity='relu')
zeros_(conv.bias)
```

---

## 14.4 初始化选择指南

### 按激活函数选择

```
激活函数          推荐初始化
─────────────────────────────
ReLU            Kaiming
LeakyReLU       Kaiming
PReLU           Kaiming
Tanh            Xavier
Sigmoid         Xavier
GELU            Xavier 或 Kaiming
```

### 按网络类型选择

```
网络类型         推荐初始化
─────────────────────────────
CNN (用ReLU)     Kaiming
RNN/LSTM         正交初始化
Transformer      截断正态 (std=0.02)
ResNet           Kaiming + 特殊处理
```

---

## 14.5 其他初始化方法

### 正交初始化

```python
def orthogonal_(tensor, gain=1.0):
    """
    正交矩阵初始化

    适合：RNN、深度网络
    好处：防止梯度消失/爆炸
    """
    # 生成随机矩阵
    flat_shape = (tensor.shape[0], np.prod(tensor.shape[1:]))
    random_matrix = np.random.randn(*flat_shape)

    # QR 分解得到正交矩阵
    q, r = np.linalg.qr(random_matrix)

    # 应用 gain
    tensor.data = (q * gain).reshape(tensor.shape).astype(np.float32)
    return tensor
```

### 截断正态

```python
def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """
    截断正态分布初始化

    适合：Transformer (std=0.02)
    好处：避免极端值
    """
    lower = mean + a * std
    upper = mean + b * std

    # 采样直到都在范围内
    data = np.random.normal(mean, std, tensor.shape)
    while np.any((data < lower) | (data > upper)):
        mask = (data < lower) | (data > upper)
        data[mask] = np.random.normal(mean, std, mask.sum())

    tensor.data = data.astype(np.float32)
    return tensor
```

---

## 14.6 偏置初始化

```python
# 偏置通常初始化为 0
bias = np.zeros(out_features)

# 特殊情况：
# BatchNorm/LayerNorm 的 gamma = 1, beta = 0
gamma = np.ones(num_features)
beta = np.zeros(num_features)
```

---

## 14.7 完整示例

```python
from nanotorch.nn import Linear, ReLU, Conv2D, BatchNorm2d, Sequential
from nanotorch.utils import kaiming_normal_, zeros_, ones_

def init_weights(model):
    """统一的初始化函数"""
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                zeros_(module.bias)

        elif isinstance(module, Conv2D):
            kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                zeros_(module.bias)

        elif isinstance(module, BatchNorm2d):
            ones_(module.weight)
            zeros_(module.bias)

# 使用
model = Sequential(
    Conv2D(3, 64, 3),
    BatchNorm2d(64),
    ReLU(),
    Linear(64, 10)
)

init_weights(model)
```

---

## 14.8 常见陷阱

### 陷阱1：忘记初始化

```python
# 问题：使用默认随机初始化
linear = Linear(128, 64)
# 权重是随机的，可能不合适

# 正确：显式初始化
linear = Linear(128, 64)
kaiming_normal_(linear.weight, nonlinearity='relu')
```

### 陷阱2：激活函数和初始化不匹配

```python
# 错误：ReLU 用 Xavier
linear = Linear(128, 64)
xavier_normal_(linear.weight)  # 不匹配！
relu = ReLU()

# 正确：ReLU 用 Kaiming
kaiming_normal_(linear.weight, nonlinearity='relu')
```

### 陷阱3：偏置也用复杂初始化

```python
# 不必要
kaiming_normal_(linear.bias)  # 偏置不需要

# 简单就好
zeros_(linear.bias)  # 通常为0
```

---

## 14.9 一句话总结

| 激活函数 | 初始化 | 原因 |
|----------|--------|------|
| ReLU | Kaiming | 补偿一半被置0 |
| Tanh | Xavier | 保持方差 |
| RNN | 正交 | 防止梯度消失 |
| Transformer | 截断正态 | 避免极端值 |

```
简单记忆：
  ReLU → Kaiming
  Tanh → Xavier
  偏置 → 零
```

---

## 下一章

现在我们学会了初始化！

下一章，我们将学习**高级主题** —— 梯度裁剪、学习率预热等实用技巧。

→ [第十五章：高级主题](15-advanced.md)

```python
# 预告：下一章你将学到
clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
# 防止梯度爆炸
```
