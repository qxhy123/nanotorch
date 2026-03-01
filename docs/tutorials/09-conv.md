# 教程 09：卷积层 (Convolution Layers)

## 目录

1. [概述](#概述)
2. [卷积的基本概念](#卷积的基本概念)
3. [Conv1D 实现](#conv1d-实现)
4. [Conv2D 实现](#conv2d-实现)
5. [Conv3D 实现](#conv3d-实现)
6. [转置卷积](#转置卷积)
7. [梯度计算](#梯度计算)
8. [使用示例](#使用示例)
9. [总结](#总结)

---

## 概述

卷积神经网络（CNN）是深度学习中处理图像、音频和序列数据的核心架构。卷积层通过**局部连接**和**权重共享**两个关键特性，大大减少了参数数量，同时保持了强大的特征提取能力。

本教程将详细介绍如何在 nanotorch 中实现卷积层，包括：
- Conv1D：一维卷积，用于序列数据
- Conv2D：二维卷积，用于图像处理
- Conv3D：三维卷积，用于视频或体积数据
- ConvTranspose2D/3D：转置卷积（反卷积）

---

## 卷积的基本概念

### 什么是卷积？

在深度学习中，卷积操作是将一个小的**卷积核（kernel/filter）**在输入数据上滑动，在每个位置计算点积并求和。

```
输入图像 (5x5):          卷积核 (3x3):
[1  2  3  4  5]         [1  0  1]
[6  7  8  9  10]        [0  1  0]
[11 12 13 14 15]        [1  0  1]
[16 17 18 19 20]
[21 22 23 24 25]

在位置 (0,0) 的计算:
1*1 + 2*0 + 3*1 +
6*0 + 7*1 + 8*0 +
11*1 + 12*0 + 13*1 = 1 + 3 + 7 + 11 + 13 = 35
```

### 关键参数

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `kernel_size` | 卷积核大小 | - |
| `stride` | 步长，滑动距离 | 1 |
| `padding` | 边缘填充 | 0 |
| `dilation` | 膨胀率，增大感受野 | 1 |
| `groups` | 分组卷积 | 1 |
| `bias` | 是否添加偏置 | True |

### 输出尺寸计算

对于输入尺寸 $H_{in} \times W_{in}$，输出尺寸为：

$$H_{out} = \left\lfloor \frac{H_{in} + 2 \times padding - dilation \times (kernel\_size - 1) - 1}{stride} \right\rfloor + 1$$

---

## Conv1D 实现

### 数据布局

Conv1D 处理的输入形状为 `(N, C_in, L)`：
- N：批次大小
- C_in：输入通道数
- L：序列长度

### 实现代码

```python
# nanotorch/nn/conv.py

class Conv1D(Module):
    """1D convolution layer.
    
    Applies a 1D convolution over an input signal composed of several input planes.
    
    Args:
        in_channels: Number of channels in the input signal.
        out_channels: Number of channels produced by the convolution.
        kernel_size: Size of the convolving kernel.
        stride: Stride of the convolution. Default: 1.
        padding: Zero-padding added to both sides. Default: 0.
        dilation: Spacing between kernel elements. Default: 1.
        bias: If True, adds a learnable bias. Default: True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # 权重形状: (out_channels, in_channels, kernel_size)
        weight_shape = (out_channels, in_channels, kernel_size)
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True
        )
        
        self.bias = None
        if bias:
            self.bias = Tensor(
                np.zeros((out_channels, 1), dtype=np.float32),
                requires_grad=True
            )
        
        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Conv1D expects 3D input (N, C, L), got {x.ndim}D")
        
        return Conv1DFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )
```

### 使用示例

```python
from nanotorch import Tensor
from nanotorch.nn import Conv1D

# 创建 Conv1D 层
conv = Conv1D(in_channels=16, out_channels=32, kernel_size=3, padding=1)

# 输入: (batch_size, in_channels, length)
x = Tensor.randn((8, 16, 100))

# 前向传播
output = conv(x)
print(output.shape)  # (8, 32, 100)
```

---

## Conv2D 实现

### 数据布局

Conv2D 处理的输入形状为 `(N, C_in, H, W)`：
- N：批次大小
- C_in：输入通道数
- H：高度
- W：宽度

### 实现代码

```python
class Conv2D(Module):
    """2D convolution layer.
    
    Applies a 2D convolution over an input signal composed of several input planes.
    
    Shape:
        - Input: (N, C_in, H_in, W_in)
        - Output: (N, C_out, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        
        # 统一转换为元组
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # 权重形状: (out_channels, in_channels, kernel_height, kernel_width)
        weight_shape = (out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True
        )
        
        self.bias = None
        if bias:
            self.bias = Tensor(
                np.zeros((out_channels, 1, 1), dtype=np.float32),
                requires_grad=True
            )
        
        self.register_parameter("weight", self.weight)
        if self.bias is not None:
            self.register_parameter("bias", self.bias)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Conv2D expects 4D input (N, C, H, W), got {x.ndim}D")
        
        return Conv2DFunction.apply(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation
        )
```

### 使用示例

```python
from nanotorch import Tensor
from nanotorch.nn import Conv2D

# 创建 Conv2D 层
conv = Conv2D(
    in_channels=3,      # RGB 图像
    out_channels=64,    # 输出特征图数
    kernel_size=3,      # 3x3 卷积核
    stride=1,
    padding=1           # 保持尺寸不变
)

# 输入: (batch_size, channels, height, width)
x = Tensor.randn((16, 3, 224, 224))

# 前向传播
output = conv(x)
print(output.shape)  # (16, 64, 224, 224)
```

### 简单实现（教学用途）

为了更好地理解卷积操作，我们提供一个简化的实现：

```python
def _conv2d_forward(self, x: Tensor, H_out: int, W_out: int) -> Tensor:
    """朴素的卷积实现（用于教学）"""
    N, C_in, H_in, W_in = x.shape
    C_out = self.out_channels
    K_H, K_W = self.kernel_size
    
    output = Tensor.zeros((N, C_out, H_out, W_out), requires_grad=x.requires_grad)
    
    # 滑动窗口卷积
    for n in range(N):           # 批次维度
        for c_out in range(C_out):  # 输出通道
            for h_out in range(H_out):  # 高度
                for w_out in range(W_out):  # 宽度
                    # 计算输入窗口位置
                    h_start = h_out * self.stride
                    w_start = w_out * self.stride
                    h_end = h_start + K_H
                    w_end = w_start + K_W
                    
                    # 提取输入窗口
                    window = x.data[n, :, h_start:h_end, w_start:w_end]
                    
                    # 获取对应的权重
                    weight_slice = self.weight.data[c_out]
                    
                    # 计算卷积和
                    conv_sum = np.sum(window * weight_slice)
                    
                    output.data[n, c_out, h_out, w_out] = conv_sum
    
    return output
```

---

## Conv3D 实现

### 数据布局

Conv3D 处理的输入形状为 `(N, C_in, D, H, W)`：
- N：批次大小
- C_in：输入通道数
- D：深度（时间帧或体数据深度）
- H：高度
- W：宽度

### 使用场景

- **视频处理**：D 为时间维度
- **医学影像**：CT/MRI 体数据
- **3D 物体识别**：点云或体素数据

### 实现代码

```python
class Conv3D(Module):
    """3D convolution layer.
    
    Shape:
        - Input: (N, C_in, D_in, H_in, W_in)
        - Output: (N, C_out, D_out, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        
        # 统一转换为三元组
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # 权重形状: (out_channels, in_channels, kD, kH, kW)
        weight_shape = (out_channels, in_channels, *kernel_size)
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True
        )
        
        # ... bias 初始化
```

---

## 转置卷积

### 概念

转置卷积（Transposed Convolution），也称为**反卷积（Deconvolution）**或**分数步长卷积**，用于**上采样**操作。

与普通卷积不同，转置卷积将低分辨率特征图转换为高分辨率输出。

```
普通卷积 (下采样):    转置卷积 (上采样):
[H, W] -> [H/2, W/2]  [H, W] -> [2H, 2W]
```

### 输出尺寸计算

对于转置卷积：

$$H_{out} = (H_{in} - 1) \times stride - 2 \times padding + dilation \times (kernel\_size - 1) + output\_padding + 1$$

### ConvTranspose2D 实现

```python
class ConvTranspose2D(Module):
    """2D transposed convolution layer.
    
    Also known as deconvolution or fractionally-strided convolution.
    Used for upsampling in segmentation networks, GANs, etc.
    
    Shape:
        - Input: (N, C_in, H_in, W_in)
        - Output: (N, C_out, H_out, W_out)
    
    Note:
        Weight shape is (in_channels, out_channels, K_H, K_W) for ConvTranspose2D,
        different from Conv2D's (out_channels, in_channels, K_H, K_W).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        
        # 注意：权重形状与 Conv2D 相反！
        # ConvTranspose2D: (in_channels, out_channels, K_H, K_W)
        weight_shape = (in_channels, out_channels, kernel_size[0], kernel_size[1])
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True
        )
        # ...
```

### 使用示例

```python
from nanotorch.nn import ConvTranspose2D

# 上采样层
upsample = ConvTranspose2D(
    in_channels=256,
    out_channels=128,
    kernel_size=4,
    stride=2,
    padding=1
)

# 输入: (N, C, H, W) -> 输出: (N, C, 2H, 2W)
x = Tensor.randn((8, 256, 32, 32))
output = upsample(x)
print(output.shape)  # (8, 128, 64, 64)
```

---

## 梯度计算

卷积层的梯度计算是自动微分的关键部分。

### 输入梯度

对于输出梯度 $\frac{\partial L}{\partial Y}$，输入梯度为：

$$\frac{\partial L}{\partial X} = \text{conv}(\frac{\partial L}{\partial Y}, W^{flipped})$$

其中 $W^{flipped}$ 是旋转180度的权重。

### 权重梯度

$$\frac{\partial L}{\partial W} = \text{conv}(X, \frac{\partial L}{\partial Y})$$

### autograd.py 中的实现

```python
# nanotorch/autograd.py

class Conv2DFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation):
        # 保存用于反向传播
        ctx.save_for_backward(x, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        
        # 执行前向卷积
        output = conv2d_forward(x.data, weight.data, bias, stride, padding, dilation)
        return Tensor(output, requires_grad=x.requires_grad or weight.requires_grad)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        
        # 计算输入梯度
        grad_input = conv2d_backward_input(grad_output, weight.data, ctx.stride, ctx.padding)
        
        # 计算权重梯度
        grad_weight = conv2d_backward_weight(grad_output, x.data, ctx.stride, ctx.padding)
        
        return grad_input, grad_weight, None, None, None, None
```

---

## 使用示例

### 构建简单的 CNN

```python
from nanotorch import Tensor
from nanotorch.nn import Conv2D, BatchNorm2d, ReLU, MaxPool2d, Linear, Sequential

# 简单的 CNN 模型
class SimpleCNN:
    def __init__(self, num_classes=10):
        self.features = Sequential(
            Conv2D(3, 32, kernel_size=3, padding=1),   # 224 -> 224
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2, 2),                            # 224 -> 112
            
            Conv2D(32, 64, kernel_size=3, padding=1),  # 112 -> 112
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2, 2),                            # 112 -> 56
            
            Conv2D(64, 128, kernel_size=3, padding=1), # 56 -> 56
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(2, 2),                            # 56 -> 28
        )
        
        self.classifier = Sequential(
            Linear(128 * 28 * 28, 512),
            ReLU(),
            Linear(512, num_classes)
        )
    
    def __call__(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x
    
    def parameters(self):
        return self.features.parameters() + self.classifier.parameters()

# 使用模型
model = SimpleCNN()
x = Tensor.randn((4, 3, 224, 224))
output = model(x)
print(output.shape)  # (4, 10)
```

### 使用不同参数

```python
# 空洞卷积 (Dilated Convolution) - 增大感受野
conv_dilated = Conv2D(64, 64, kernel_size=3, dilation=2, padding=2)

# 大步长卷积 - 快速下采样
conv_stride = Conv2D(64, 128, kernel_size=3, stride=2, padding=1)

# 1x1 卷积 - 通道变换
conv_1x1 = Conv2D(256, 64, kernel_size=1)
```

---

## 总结

本教程详细介绍了 nanotorch 中卷积层的实现：

1. **Conv1D**：处理序列数据，形状 `(N, C, L)`
2. **Conv2D**：处理图像数据，形状 `(N, C, H, W)`
3. **Conv3D**：处理视频/体积数据，形状 `(N, C, D, H, W)`
4. **ConvTranspose2D/3D**：用于上采样的转置卷积

关键参数：
- `kernel_size`：卷积核大小
- `stride`：步长
- `padding`：填充
- `dilation`：膨胀率

### 下一步

在 [教程 10：归一化层](10-normalization.md) 中，我们将学习如何实现 BatchNorm、LayerNorm、GroupNorm 等归一化层，这些层对稳定训练至关重要。

---

**参考资源**：
- [CS231n: Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)
- [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285)
