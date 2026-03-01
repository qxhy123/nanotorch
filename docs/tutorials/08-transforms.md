# 第八章：数据增强

数据增强（Data Augmentation）通过对训练数据进行变换，增加数据多样性，提升模型泛化能力。

## 8.1 为什么需要数据增强？

```
原始图像 → 随机变换 → 增强图像
   ↓
一张图片 → 多种变体 → 增加训练样本量
```

**好处**：
- 防止过拟合
- 提升模型泛化能力
- 有效增加训练数据量

## 8.2 Transform 基类

```python
# transforms/__init__.py
from typing import Callable, List
import numpy as np

class Compose:
    """组合多个变换"""
    
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
```

## 8.3 基础变换

### 归一化

```python
class Normalize:
    """标准化图像
    
    output = (input - mean) / std
    """
    
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return (img - self.mean) / self.std


# ImageNet 标准化
normalize = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### 类型转换

```python
class ToFloat:
    """转换为 float32"""
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32)


class ToUint8:
    """转换为 uint8"""
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.uint8)
```

## 8.4 几何变换

### 随机水平翻转

```python
class RandomHorizontalFlip:
    """以概率 p 随机水平翻转"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
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

```python
class RandomCrop:
    """随机裁剪到指定大小"""
    
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
    """中心裁剪"""
    
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

### 随机缩放裁剪

```python
class RandomResizedCrop:
    """随机裁剪并缩放到指定大小
    
    常用于 ImageNet 训练
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
                
                # 缩放到目标大小
                return self._resize(cropped, self.size)
        
        # 失败则中心裁剪
        return CenterCrop(self.size)(img)
    
    def _resize(self, img: np.ndarray, size: tuple) -> np.ndarray:
        # 简单的最近邻缩放
        th, tw = size
        h, w = img.shape[:2]
        
        y_indices = (np.arange(th) * h / th).astype(int)
        x_indices = (np.arange(tw) * w / tw).astype(int)
        
        if img.ndim == 3:
            return img[np.ix_(y_indices, x_indices, [0,1,2])]
        return img[np.ix_(y_indices, x_indices)]
```

## 8.5 颜色变换

### 颜色抖动

```python
class ColorJitter:
    """随机调整亮度、对比度、饱和度"""
    
    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.copy().astype(np.float32)
        
        # 亮度
        if self.brightness > 0:
            factor = np.random.uniform(
                max(0, 1 - self.brightness),
                1 + self.brightness
            )
            img = img * factor
        
        # 对比度
        if self.contrast > 0:
            factor = np.random.uniform(
                max(0, 1 - self.contrast),
                1 + self.contrast
            )
            mean = img.mean()
            img = (img - mean) * factor + mean
        
        # 饱和度（仅RGB图像）
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
    """以概率 p 转换为灰度图"""
    
    def __init__(self, p: float = 0.1):
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p and img.ndim == 3:
            gray = np.mean(img, axis=2, keepdims=True)
            return np.repeat(gray, 3, axis=2).astype(img.dtype)
        return img
```

## 8.6 高级变换

### 随机擦除

```python
class RandomErasing:
    """随机擦除图像区域
    
    用于模拟遮挡，提升鲁棒性
    """
    
    def __init__(
        self,
        p: float = 0.5,
        scale: tuple = (0.02, 0.33),
        ratio: tuple = (0.3, 3.3),
        value: float = 0.0
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

## 8.7 使用示例

### 训练时增强

```python
from nanotorch.transforms import (
    Compose, ToFloat, Normalize,
    RandomHorizontalFlip, RandomCrop,
    ColorJitter, RandomErasing
)

# 训练变换
train_transform = Compose([
    ToFloat(),
    RandomHorizontalFlip(p=0.5),
    RandomCrop(size=224),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    RandomErasing(p=0.5),
])

# 应用变换
image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
augmented = train_transform(image)
```

### 验证时变换

```python
# 验证变换（不做随机增强）
val_transform = Compose([
    ToFloat(),
    CenterCrop(size=224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## 8.8 完整训练示例

```python
from nanotorch import DataLoader, TensorDataset
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

## 8.9 变换对比

| 变换 | 用途 | 适用场景 |
|------|------|----------|
| RandomHorizontalFlip | 水平翻转 | 通用 |
| RandomCrop | 随机裁剪 | 通用 |
| RandomResizedCrop | 裁剪+缩放 | ImageNet |
| ColorJitter | 颜色扰动 | 通用 |
| RandomErasing | 随机擦除 | 通用、遮挡鲁棒 |

## 8.10 练习

1. **实现 RandomRotation**：随机旋转图像

2. **实现 GaussianBlur**：高斯模糊

3. **实现 Cutout**：随机遮挡方块

## 下一章

下一章，我们将介绍**卷积层**，用于图像和序列处理。

→ [第九章：卷积层](09-conv.md)
