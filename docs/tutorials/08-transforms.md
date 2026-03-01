# 第八章：数据增强

## 一张猫的照片，能变成多少张？...

你有一张猫的照片。正面照，光线恰好，猫咪端坐在画面中央。

如果只让模型看这一张照片，它会怎么理解"猫"？

"猫就是这样的：正面、光线好、端坐在中间。"

于是，当它看到侧面猫、昏暗处的猫、跑动中的猫——就懵了。

**数据增强，就是让一张图变出千张图。**

翻转一下，猫还是猫。旋转一下，猫还是猫。调亮一点，猫还是猫。裁剪局部，猫还是猫。

```
原始数据：
  1000 张照片 → 模型死记硬背

数据增强：
  1000 张 → 翻转 → 2000 张
          → 旋转 → 4000 张
          → 裁剪 → 8000 张
          → 变色 → 16000 张

  模型学到的是"猫的本质"
  而不是"这张照片长什么样"
```

**数据增强，是防止过拟合的疫苗。** 它教会模型：形式万变，本质不变。

---

## 8.1 为什么需要数据增强？

### 问题：数据太少

```
你只有 100 张猫的照片

模型会"死记硬背"：
  "这张图位置(x=10,y=20)、颜色(灰)、大小(大)的是猫"

遇到新猫就不认识了！
```

### 解决：数据增强

```
原始图片 → 变换 → 增强图片

    🐱    →  翻转  →  🐱（镜像）
    🐱    →  旋转  →  🐱（斜的）
    🐱    →  裁剪  →  🐱（局部）
    🐱    →  变色  →  🐱（灰度）

一张图 → 很多变体 → 模型学到"猫的本质"
```

### 效果对比

```
不用数据增强：
  训练：100% 准确率（死记硬背）
  测试：60% 准确率（遇到新图就不行）

用数据增强：
  训练：85% 准确率（学的是规律）
  测试：82% 准确率（泛化能力强！）
```

---

## 8.2 Transform 基类

### 组合变换

```python
# transforms/__init__.py
from typing import Callable, List
import numpy as np

class Compose:
    """
    组合多个变换

    类比：流水线工厂
      原料 → 工序1 → 工序2 → 工序3 → 成品

    图像 → 翻转 → 裁剪 → 归一化 → 输出
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, img: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


# 使用
transform = Compose([
    RandomHorizontalFlip(p=0.5),  # 50% 概率翻转
    RandomCrop(size=224),         # 随机裁剪
    Normalize(mean=[0.5], std=[0.5])  # 归一化
])

augmented_img = transform(original_img)
```

---

## 8.3 基础变换

### 归一化（Normalize）

```
为什么要归一化？

原始数据：[0, 255] 范围
  问题：数值太大，训练不稳定

归一化后：[-1, 1] 或 [0, 1] 范围
  好处：数值小，训练稳定
```

```python
class Normalize:
    """
    标准化图像

    output = (input - mean) / std

    类比：把分数换算成标准分
      原始：语文150分制，数学100分制
      标准化后：都是均值0，标准差1
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return (img - self.mean) / self.std


# ImageNet 标准化（ImageNet 数据集的均值和标准差）
normalize = Normalize(
    mean=[0.485, 0.456, 0.406],  # RGB 三通道均值
    std=[0.229, 0.224, 0.225]    # RGB 三通道标准差
)
```

### 类型转换

```python
class ToFloat:
    """转换为 float32（0-1 范围）"""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32) / 255.0


class ToUint8:
    """转换为 uint8（0-255 范围）"""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return (img * 255).astype(np.uint8)
```

---

## 8.4 几何变换

### 随机水平翻转

```
原始图片：        水平翻转后：

  🚗→                ←🚗
 朝右              朝左

猫还是猫，只是方向变了
```

```python
class RandomHorizontalFlip:
    """
    以概率 p 随机水平翻转

    为什么有用？
      - 猫朝左朝右都是猫
      - 让模型学会"左右对称"
    """

    def __init__(self, p: float = 0.5):
        self.p = p  # 翻转概率

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            # img: (H, W, C) 或 (H, W)
            return np.flip(img, axis=1).copy()
        return img
```

### 随机垂直翻转

```python
class RandomVerticalFlip:
    """以概率 p 随机垂直翻转"""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return np.flip(img, axis=0).copy()
        return img
```

### 随机裁剪

```
原始图片 (256x256)：      随机裁剪 (224x224)：

┌────────────────┐       ┌──────────┐
│  🌳    🌳      │       │  🌳      │
│      🐱        │   →   │    🐱    │
│  🌳    🌳      │       │  🌳      │
└────────────────┘       └──────────┘

每次裁剪不同位置 → 模型学会关注不同区域
```

```python
class RandomCrop:
    """
    随机裁剪到指定大小

    好处：
      - 模型学会物体不一定在中心
      - 相当于增加了训练样本
    """

    def __init__(self, size: tuple):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(f"Crop size {self.size} larger than image {(h, w)}")

        # 随机选择裁剪起点
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)

        if img.ndim == 3:
            return img[i:i+th, j:j+tw, :].copy()
        return img[i:i+th, j:j+tw].copy()
```

### 中心裁剪

```python
class CenterCrop:
    """
    中心裁剪

    用于验证/测试：确定性裁剪，每次结果一样
    """

    def __init__(self, size: tuple):
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        th, tw = self.size

        i = (h - th) // 2
        j = (w - tw) // 2

        if img.ndim == 3:
            return img[i:i+th, j:j+tw, :].copy()
        return img[i:i+th, j:j+tw].copy()
```

### 随机缩放裁剪（RandomResizedCrop）

```
ImageNet 训练的标准增强：

1. 随机选择区域（面积 8%-100%）
2. 随机宽高比（3:4 到 4:3）
3. 裁剪并缩放到 224x224

┌────────────────────┐
│    ┌──────┐        │
│    │ 裁剪 │        │  →  缩放到 224x224
│    └──────┘        │
└────────────────────┘
```

```python
class RandomResizedCrop:
    """
    随机裁剪并缩放到指定大小

    ImageNet 训练最常用的增强！
    """

    def __init__(
        self,
        size: tuple,
        scale: tuple = (0.08, 1.0),  # 裁剪面积比例
        ratio: tuple = (3./4., 4./3.)  # 宽高比
    ):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        area = h * w

        for _ in range(10):  # 最多尝试10次
            target_area = np.random.uniform(*self.scale) * area
            aspect_ratio = np.random.uniform(*self.ratio)

            tw = int(round(np.sqrt(target_area * aspect_ratio)))
            th = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < tw <= w and 0 < th <= h:
                i = np.random.randint(0, h - th + 1)
                j = np.random.randint(0, w - tw + 1)

                if img.ndim == 3:
                    cropped = img[i:i+th, j:j+tw, :]
                else:
                    cropped = img[i:i+th, j:j+tw]

                return self._resize(cropped, self.size)

        # 失败则中心裁剪
        return CenterCrop(self.size)(img)

    def _resize(self, img: np.ndarray, size: tuple) -> np.ndarray:
        th, tw = size
        h, w = img.shape[:2]

        y_indices = (np.arange(th) * h / th).astype(int)
        x_indices = (np.arange(tw) * w / tw).astype(int)

        if img.ndim == 3:
            return img[np.ix_(y_indices, x_indices, [0,1,2])]
        return img[np.ix_(y_indices, x_indices)]
```

---

## 8.5 颜色变换

### 颜色抖动（ColorJitter）

```
原始图片：      抖动后：

  🌈              🌈（更亮/更暗）
  彩色            灰一点/鲜艳一点

模拟不同光照条件
```

```python
class ColorJitter:
    """
    随机调整亮度、对比度、饱和度

    模拟真实世界的变化：
      - 不同时间的光照
      - 不同的相机设置
    """

    def __init__(
        self,
        brightness: float = 0.0,  # 亮度变化范围
        contrast: float = 0.0,    # 对比度变化范围
        saturation: float = 0.0   # 饱和度变化范围
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.copy().astype(np.float32)

        # 亮度调整
        if self.brightness > 0:
            factor = np.random.uniform(
                max(0, 1 - self.brightness),
                1 + self.brightness
            )
            img = img * factor

        # 对比度调整
        if self.contrast > 0:
            factor = np.random.uniform(
                max(0, 1 - self.contrast),
                1 + self.contrast
            )
            mean = img.mean()
            img = (img - mean) * factor + mean

        # 饱和度调整（仅RGB图像）
        if self.saturation > 0 and img.ndim == 3:
            factor = np.random.uniform(
                max(0, 1 - self.saturation),
                1 + self.saturation
            )
            gray = np.mean(img, axis=2, keepdims=True)
            img = gray + (img - gray) * factor

        return np.clip(img, 0, 255).astype(np.uint8)
```

### 随机灰度

```python
class RandomGrayscale:
    """
    以概率 p 转换为灰度图

    为什么？
      - 有些场景本来就是灰色的
      - 让模型学会关注形状，不只依赖颜色
    """

    def __init__(self, p: float = 0.1):
        self.p = p

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p and img.ndim == 3:
            gray = np.mean(img, axis=2, keepdims=True)
            return np.repeat(gray, 3, axis=2).astype(img.dtype)
        return img
```

---

## 8.6 高级变换

### 随机擦除（RandomErasing）

```
原始图片：         随机擦除后：

  🐱🐱🐱              🐱■🐱
  完整的猫           部分被遮挡

模拟物体被遮挡的情况
```

```python
class RandomErasing:
    """
    随机擦除图像区域

    模拟遮挡，让模型学会：
      - 不依赖某个特定区域
      - 根据部分信息也能识别
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: tuple = (0.02, 0.33),  # 擦除区域大小
        ratio: tuple = (0.3, 3.3),    # 宽高比
        value: float = 0.0            # 填充值
    ):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() > self.p:
            return img

        h, w = img.shape[:2]
        area = h * w

        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            aspect_ratio = np.random.uniform(*self.ratio)

            ew = int(round(np.sqrt(target_area * aspect_ratio)))
            eh = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < ew <= w and 0 < eh <= h:
                i = np.random.randint(0, h - eh + 1)
                j = np.random.randint(0, w - ew + 1)

                img = img.copy()
                if img.ndim == 3:
                    img[i:i+eh, j:j+ew, :] = self.value
                else:
                    img[i:i+eh, j:j+ew] = self.value
                return img

        return img
```

---

## 8.7 训练 vs 验证的变换

```
训练时：用随机增强

  原图 → 随机翻转 → 随机裁剪 → 颜色抖动 → 归一化
         ↑ 每次不同

验证时：只用确定性变换

  原图 → 中心裁剪 → 归一化
         ↑ 每次相同
```

```python
from nanotorch.transforms import (
    Compose, ToFloat, Normalize,
    RandomHorizontalFlip, RandomCrop, CenterCrop,
    ColorJitter, RandomErasing
)

# 训练变换（有随机增强）
train_transform = Compose([
    ToFloat(),
    RandomHorizontalFlip(p=0.5),      # 随机翻转
    RandomCrop(size=224),             # 随机裁剪
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    RandomErasing(p=0.5),             # 随机擦除
])

# 验证变换（确定性）
val_transform = Compose([
    ToFloat(),
    CenterCrop(size=224),             # 中心裁剪
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 8.8 完整使用示例

```python
from nanotorch import DataLoader
from nanotorch.transforms import Compose, ToFloat, Normalize, RandomHorizontalFlip

# 定义变换
train_transform = Compose([
    ToFloat(),
    RandomHorizontalFlip(p=0.5),
    Normalize(mean=[0.5], std=[0.5]),
])

# 自定义 Dataset 应用变换
class TransformDataset:
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        return x, y

# 使用
train_dataset = TransformDataset(X_train, y_train, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

---

## 8.9 变换对比表

| 变换 | 作用 | 适用场景 | 训练/验证 |
|------|------|----------|-----------|
| RandomHorizontalFlip | 左右翻转 | 通用 | 训练 |
| RandomCrop | 随机裁剪 | 通用 | 训练 |
| RandomResizedCrop | 裁剪+缩放 | ImageNet | 训练 |
| ColorJitter | 颜色扰动 | 通用 | 训练 |
| RandomErasing | 随机遮挡 | 遮挡鲁棒 | 训练 |
| CenterCrop | 中心裁剪 | 通用 | 验证 |
| Normalize | 归一化 | **必须** | 两者 |

---

## 8.10 常见陷阱

### 陷阱1：验证时用了随机增强

```python
# 错误：验证时每次结果不一样
val_transform = Compose([
    RandomCrop(224),  # 随机！
    Normalize(...),
])

# 正确：验证用确定性变换
val_transform = Compose([
    CenterCrop(224),  # 确定性
    Normalize(...),
])
```

### 陷阱2：归一化参数不对

```python
# 错误：用了 ImageNet 的参数处理 MNIST
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet

# 正确：MNIST 是灰度图，单通道
Normalize(mean=[0.1307], std=[0.3081])  # MNIST
```

### 陷阱3：标签也跟着变了

```python
# 注意：某些任务标签也要变！
# 目标检测：框的坐标要跟着翻转
# 语义分割：分割图也要同步变换
```

---

## 8.11 练习

### 基础练习

1. **实现 RandomRotation**：随机旋转图像 ±30°

2. **实现 GaussianBlur**：高斯模糊

3. **实现 Cutout**：固定大小的随机遮挡

### 进阶练习

4. **实现 MixUp**：两张图混合
   ```
   new_image = 0.7 * image1 + 0.3 * image2
   new_label = 0.7 * label1 + 0.3 * label2
   ```

5. **实现 AutoAugment**：自动搜索最佳增强策略

---

## 一句话总结

| 概念 | 一句话 |
|------|--------|
| 数据增强 | 一张图变多张，学得更扎实 |
| 几何变换 | 翻转、裁剪、旋转（位置变了） |
| 颜色变换 | 亮度、对比度、饱和度（颜色变了） |
| 训练增强 | 随机变换，每次不同 |
| 验证增强 | 确定性变换，每次相同 |

---

## 下一章

现在我们的数据变多变样了！

下一章，我们将学习**卷积层** —— 处理图像的核心组件。

→ [第九章：卷积层](09-conv.md)

```python
# 预告：下一章你将学到
conv = Conv2D(in_channels=3, out_channels=64, kernel_size=3)
# 用一个 3x3 的小窗口滑过整张图
# 提取图像的局部特征
```
