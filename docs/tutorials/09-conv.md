# 第九章：卷积层

## 人类的眼睛，是如何看世界的...

你正在欣赏一幅油画。

你不会把整幅画当作一堆无意义的像素。你的眼睛会聚焦：这里是一抹红色，那里是一笔蓝色；左边有曲线，右边有棱角。

你看到的是**局部特征**——边缘、颜色、纹理——然后大脑把它们组合成完整的画面。

卷积神经网络（CNN）也在做同样的事。

它拿着一个小窗口——3×3 或 5×5 大小——在图像上滑动。每个位置，它都在问："这里有边缘吗？有角吗？有纹理吗？"

```
卷积的智慧：

  一张图 = 224 × 224 × 3 = 150,528 个数字

  不用一次性处理全部
  而是用小窗口逐块扫描
  提取局部特征，层层抽象

  第一层：边缘、角点
  第二层：纹理、形状
  第三层：部件、对象
  第四层：完整的猫、狗、汽车
```

**卷积，是机器视觉的基石。** 它教会机器像人类一样，从局部到整体地理解图像。

---

## 9.1 为什么需要卷积？

### 问题：全连接层处理图像太笨

```
一张 224x224 的彩色图片：
  - 像素数：224 × 224 × 3 = 150,528
  - 连接到 1000 个神经元：150,528 × 1000 = 1.5 亿参数！

问题：
  1. 参数太多，训练慢
  2. 没利用图像的空间结构
  3. 位置变了就不认识
```

### 解决：卷积的两大神器

```
1. 局部连接：
   每个神经元只看一小块区域（比如 3×3）

2. 权重共享：
   同一个放大镜扫描整张图

效果：
  - 参数量：3 × 3 × 3 × 64 = 1,728（少了上万倍！）
  - 位置不变：猫在左上角或右下角都能识别
```

---

## 9.2 卷积是怎么工作的？

### 单步计算

```
输入图像 (5×5)：         卷积核 (3×3)：
                         ┌─────────┐
[1  2  3  4  5]          │ 1  0  1 │
[6  7  8  9  10]         │ 0  1  0 │
[11 12 13 14 15]         │ 1  0  1 │
[16 17 18 19 20]         └─────────┘
[21 22 23 24 25]

第1步：卷积核覆盖左上角 3×3 区域

[1  2  3]       [1  0  1]
[6  7  8]   ⊙   [0  1  0]   = 1×1 + 2×0 + 3×1 + 6×0 + 7×1 + 8×0 + 11×1 + 12×0 + 13×1
[11 12 13]      [1  0  1]

              = 1 + 0 + 3 + 0 + 7 + 0 + 11 + 0 + 13 = 35

⊙ = 对应位置相乘再相加
```

### 滑动过程

```
卷积核在图像上滑动：

位置1: 位置2: ... 位置n:
┌─────┐    ┌─────┐       ┌─────┐
│1 2 3│    │  2 3│       │  3 4│
│6 7  │ →  │6 7  │ → ... →│7 8  │
│11 12│    │11 12│       │12 13│
└─────┘    └─────┘       └─────┘
  ↓          ↓             ↓
  35         40            45

输出：[35, 40, 45, ...]
```

### 图解卷积

```
输入 (H×W)          卷积核 (K×K)         输出 (H'×W')

┌─────────────┐     ┌───────┐           ┌─────────┐
│             │     │ w w w │           │         │
│   ┌─────┐   │  ⊙  │ w w w │  =        │ o o o   │
│   │扫描 │   │     │ w w w │           │ o o o   │
│   └─────┘   │     └───────┘           │         │
│             │                         └─────────┘
└─────────────┘

扫描完 → 输出是一个更小的图（特征图）
```

---

## 9.3 关键参数

### 步长（Stride）

```
Stride = 1：每次移动1格
Stride = 2：每次移动2格

Stride=1:           Stride=2:
[1 2 3 4]           [1 2 3 4]
[● ● ○ ○]           [●   ○   ]
[○ ○ ● ●]           [    ●   ]
 移1格               移2格

步长越大，输出越小
```

### 填充（Padding）

```
问题：卷积后图像变小了
  输入 5×5 → 卷积 3×3 → 输出 3×3

解决：边缘补0

原始：      补0后：
[1 2 3]     [0 0 0 0 0]
[4 5 6] →  [0 1 2 3 0]
[7 8 9]     [0 4 5 6 0]
            [0 7 8 9 0]
            [0 0 0 0 0]

补0后：输入 5×5 → 输出还是 5×5
```

### 输出尺寸计算

```
输出大小 = (输入大小 + 2×padding - 卷积核大小) / stride + 1

例子：
  输入：224×224
  卷积核：3×3
  padding：1
  stride：1

  输出 = (224 + 2×1 - 3) / 1 + 1 = 224

  → 尺寸不变！这是常用的配置
```

---

## 9.4 Conv2D 实现

### 数据布局

```
Conv2D 输入形状：(N, C, H, W)
  N = batch size（一次处理几张图）
  C = channels（通道数，RGB=3）
  H = height（高度）
  W = width（宽度）

例子：
  16 张 RGB 图片，每张 224×224
  形状：(16, 3, 224, 224)
```

### 实现代码

```python
class Conv2D(Module):
    """
    2D 卷积层

    类比：
      - 输入：一张或多张图片
      - 卷积核：多个"放大镜"
      - 输出：多张特征图（每个放大镜产生一张）

    权重形状：(out_channels, in_channels, kernel_h, kernel_w)

    例子：
      in_channels=3 (RGB)
      out_channels=64 (64个不同的放大镜)
      kernel_size=3 (3×3的放大镜)
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

        # 权重：64个 3×3×3 的滤波器 = 64×3×3×3 个参数
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

# 创建卷积层
conv = Conv2D(
    in_channels=3,      # RGB 图像
    out_channels=64,    # 64 个特征图
    kernel_size=3,      # 3×3 卷积核
    stride=1,
    padding=1           # 保持尺寸不变
)

# 输入: (batch_size, channels, height, width)
x = Tensor.randn((16, 3, 224, 224))

# 前向传播
output = conv(x)
print(output.shape)  # (16, 64, 224, 224)
```

---

## 9.5 朴素实现（教学用）

```python
def conv2d_simple(x, weight, stride=1, padding=0):
    """
    朴素的卷积实现（帮助理解）

    四重循环：
      - n: 第几张图
      - c_out: 第几个输出通道
      - h: 高度方向滑动
      - w: 宽度方向滑动
    """
    N, C_in, H_in, W_in = x.shape
    C_out, _, K_H, K_W = weight.shape

    # 计算输出尺寸
    H_out = (H_in + 2*padding - K_H) // stride + 1
    W_out = (W_in + 2*padding - K_W) // stride + 1

    # 填充
    if padding > 0:
        x_padded = np.pad(x, ((0,0), (0,0), (padding,padding), (padding,padding)))
    else:
        x_padded = x

    output = np.zeros((N, C_out, H_out, W_out))

    # 滑动窗口
    for n in range(N):              # 每张图
        for c_out in range(C_out):  # 每个输出通道
            for h in range(H_out):  # 高度滑动
                for w in range(W_out):  # 宽度滑动
                    # 计算窗口位置
                    h_start = h * stride
                    w_start = w * stride

                    # 提取窗口
                    window = x_padded[n, :, h_start:h_start+K_H, w_start:w_start+K_W]

                    # 卷积计算
                    output[n, c_out, h, w] = np.sum(window * weight[c_out])

    return output
```

---

## 9.6 Conv1D：处理序列

```
Conv1D 用于序列数据（文本、音频）

输入形状：(N, C, L)
  N = batch size
  C = channels（词向量维度）
  L = length（序列长度）

滑动方向：只有一维（从左到右）

文本例子：
  输入："我爱学习"（4个词）
  卷积核：3个词的窗口
  滑动：[我爱学] → [爱学习]
```

```python
class Conv1D(Module):
    """
    1D 卷积层

    用于：文本分类、时间序列、音频处理

    输入：(N, C_in, L)
    输出：(N, C_out, L')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()

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
```

---

## 9.7 Conv3D：处理视频

```
Conv3D 用于体积数据（视频、CT扫描）

输入形状：(N, C, D, H, W)
  N = batch size
  C = channels
  D = depth（时间/深度）
  H = height
  W = width

视频例子：
  D = 16帧
  每帧 224×224
  输入：(N, 3, 16, 224, 224)

滑动方向：三维（时间+空间）
```

```python
class Conv3D(Module):
    """
    3D 卷积层

    用于：视频分析、医学影像、3D物体识别

    输入：(N, C_in, D_in, H_in, W_in)
    输出：(N, C_out, D_out, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        # 权重形状: (out_channels, in_channels, kD, kH, kW)
        weight_shape = (out_channels, in_channels, *kernel_size)
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True
        )
```

---

## 9.8 转置卷积：上采样

### 什么是转置卷积？

```
普通卷积：下采样（变小）
  224×224 → 112×112

转置卷积：上采样（变大）
  112×112 → 224×224

用途：
  - 图像分割：恢复分辨率
  - GAN：生成大图
  - 超分辨率：图像放大
```

### 图示

```
普通卷积：                 转置卷积：

┌─────────┐               ┌───┐
│  大图   │  → 卷积 →     │小图│
└─────────┘               └───┘

┌───┐                     ┌─────────┐
│小图│  → 转置卷积 →      │  大图   │
└───┘                     └─────────┘
```

### 实现

```python
class ConvTranspose2D(Module):
    """
    2D 转置卷积（反卷积）

    用于上采样：将低分辨率特征图变大

    注意：权重形状与 Conv2D 相反！
      Conv2D: (out_channels, in_channels, K_H, K_W)
      ConvTranspose2D: (in_channels, out_channels, K_H, K_W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = (3, 3),
        stride: int = 2,      # 通常 stride=2（放大2倍）
        padding: int = 1,
        output_padding: int = 0,
    ) -> None:
        super().__init__()

        # 注意：权重形状与 Conv2D 相反
        weight_shape = (in_channels, out_channels, kernel_size[0], kernel_size[1])
        self.weight = Tensor(
            np.random.randn(*weight_shape).astype(np.float32) * 0.1,
            requires_grad=True
        )
```

### 使用

```python
# 上采样层：32×32 → 64×64
upsample = ConvTranspose2D(
    in_channels=256,
    out_channels=128,
    kernel_size=4,
    stride=2,
    padding=1
)

x = Tensor.randn((8, 256, 32, 32))
output = upsample(x)
print(output.shape)  # (8, 128, 64, 64)
```

---

## 9.9 特殊卷积

### 1×1 卷积

```
1×1 卷积：只改变通道数，不改空间尺寸

用途：
  - 降维/升维
  - 增加非线性
  - 通道混合

例子：
  输入：(N, 512, 28, 28)
  1×1 卷积：(N, 64, 28, 28)
  参数：512×64 = 32,768（很少！）
```

```python
conv_1x1 = Conv2D(512, 64, kernel_size=1)  # 降维
```

### 空洞卷积（Dilated Convolution）

```
空洞卷积：卷积核中间"跳过"一些位置

普通 3×3：          空洞 3×3（dilation=2）：
┌─────────┐         ┌───────────────┐
│ w w w │           │ w   w   w     │
│ w w w │           │   \   /       │
│ w w w │           │ w   w   w     │
└─────────┘         │   /   \       │
                    │ w   w   w     │
感受野：3×3          └───────────────┘
                    感受野：5×5

好处：不增加参数，增大感受野
```

```python
conv_dilated = Conv2D(64, 64, kernel_size=3, dilation=2, padding=2)
```

### 深度可分离卷积

```
普通卷积：每个滤波器看所有通道
深度可分离：每个通道单独看

普通卷积参数：3×3×3×64 = 1,728
深度可分离参数：3×3×3 + 1×1×3×64 = 27 + 192 = 219

少了8倍！
```

---

## 9.10 梯度计算

### 输入梯度

```
反向传播：已知 ∂L/∂output，求 ∂L/∂input

∂L/∂input = conv(∂L/∂output, W_flipped)

W_flipped = 权重旋转180度
```

### 权重梯度

```
∂L/∂W = conv(input, ∂L/∂output)

用输入和输出梯度做卷积，得到权重梯度
```

---

## 9.11 构建 CNN

```python
from nanotorch import Tensor
from nanotorch.nn import Conv2D, BatchNorm2d, ReLU, MaxPool2d, Linear, Sequential

class SimpleCNN:
    """
    简单的 CNN 分类器

    结构：
      Conv → BN → ReLU → Pool（重复3次）
      Flatten → FC → ReLU → FC
    """

    def __init__(self, num_classes=10):
        self.features = Sequential(
            # Block 1: 224 → 112
            Conv2D(3, 32, kernel_size=3, padding=1),
            BatchNorm2d(32),
            ReLU(),
            MaxPool2d(2, 2),

            # Block 2: 112 → 56
            Conv2D(32, 64, kernel_size=3, padding=1),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(2, 2),

            # Block 3: 56 → 28
            Conv2D(64, 128, kernel_size=3, padding=1),
            BatchNorm2d(128),
            ReLU(),
            MaxPool2d(2, 2),
        )

        self.classifier = Sequential(
            Linear(128 * 28 * 28, 512),
            ReLU(),
            Linear(512, num_classes)
        )

    def __call__(self, x):
        x = self.features(x)
        x = x.flatten(start_dim=1)  # 展平
        x = self.classifier(x)
        return x

    def parameters(self):
        return self.features.parameters() + self.classifier.parameters()

# 使用
model = SimpleCNN()
x = Tensor.randn((4, 3, 224, 224))
output = model(x)
print(output.shape)  # (4, 10)
```

---

## 9.12 常见陷阱

### 陷阱1：padding 不对导致尺寸不整

```python
# 问题：224 / 2 = 112，但 225 / 2 = 112.5
# 解决：确保输入尺寸能被 stride 整除

# 正确配置
Conv2D(64, 128, kernel_size=3, stride=2, padding=1)
# 224 → 112（刚好整除）
```

### 陷阱2：通道数配置错误

```python
# 错误：前后通道数不匹配
Conv2D(3, 64, ...)   # 输出 64 通道
Conv2D(32, 128, ...) # 期望输入 32 通道 ← 错！

# 正确
Conv2D(3, 64, ...)    # 输出 64 通道
Conv2D(64, 128, ...)  # 输入 64 通道 ← 对！
```

### 陷阱3：忘记 bias

```python
# 有些情况下不需要 bias
Conv2D(..., bias=False)  # 后面跟 BatchNorm 时可以不要 bias
```

---

## 9.13 卷积层对比

| 层 | 输入维度 | 用途 | 例子 |
|----|---------|------|------|
| Conv1D | (N,C,L) | 序列 | 文本、音频 |
| Conv2D | (N,C,H,W) | 图像 | 图片分类 |
| Conv3D | (N,C,D,H,W) | 体积 | 视频、CT |
| ConvTranspose2D | (N,C,H,W) | 上采样 | 分割、GAN |

---

## 9.14 练习

### 基础练习

1. 手动计算一个 3×3 卷积的输出

2. 实现 `kernel_size=5, stride=2, padding=2` 的卷积，计算输出尺寸

3. 实现 `AvgPool2d`（平均池化）

### 进阶练习

4. 实现深度可分离卷积

5. 实现分组卷积（groups>1）

---

## 一句话总结

| 概念 | 一句话 |
|------|--------|
| 卷积 | 拿着放大镜扫描图像 |
| 卷积核 | 那个放大镜（找特征的模板） |
| stride | 每次移动几格 |
| padding | 边缘补0保持尺寸 |
| Conv2D | 处理图像的卷积 |
| 转置卷积 | 反向卷积，用于放大 |

---

## 下一章

现在我们学会了卷积！

下一章，我们将学习**归一化层** —— 让训练更稳定的秘诀。

→ [第十章：归一化层](10-normalization.md)

```python
# 预告：下一章你将学到
BatchNorm2d(64)   # 批归一化
LayerNorm(768)    # 层归一化（Transformer 用）
GroupNorm(32, 64) # 分组归一化
```
