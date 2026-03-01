# Chapter 8: Data Augmentation

Data augmentation increases data diversity by transforming training data, improving model generalization.

## 8.1 Why Do We Need Data Augmentation?

```
Original Image → Random Transform → Augmented Image
    ↓
One Image → Multiple Variants → Increase training sample count
```

**Benefits**:
- Prevents overfitting
- Improves model generalization
- Effectively increases training data volume

## 8.2 Transform Base Class

```python
# transforms/__init__.py
from typing import Callable, List
import numpy as np

class Compose:
    """Compose multiple transforms"""
    
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

## 8.3 Basic Transforms

### Normalization

```python
class Normalize:
    """Normalize image
    
    output = (input - mean) / std
    """
    
    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return (img - self.mean) / self.std


# ImageNet normalization
normalize = Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
```

### Type Conversion

```python
class ToFloat:
    """Convert to float32"""
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32)


class ToUint8:
    """Convert to uint8"""
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.uint8)
```

## 8.4 Geometric Transforms

### Random Horizontal Flip

```python
class RandomHorizontalFlip:
    """Randomly flip horizontally with probability p"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            # img: (H, W, C) or (H, W)
            return np.flip(img, axis=1).copy()
        return img
```

### Random Vertical Flip

```python
class RandomVerticalFlip:
    """Randomly flip vertically with probability p"""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return np.flip(img, axis=0).copy()
        return img
```

### Random Crop

```python
class RandomCrop:
    """Randomly crop to specified size"""
    
    def __init__(self, size: tuple):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        th, tw = self.size
        
        if h < th or w < tw:
            raise ValueError(f"Crop size {self.size} larger than image {(h, w)}")
        
        # Randomly select crop starting point
        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)
        
        if img.ndim == 3:
            return img[i:i+th, j:j+tw, :].copy()
        return img[i:i+th, j:j+tw].copy()
```

### Center Crop

```python
class CenterCrop:
    """Center crop"""
    
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

### Random Resized Crop

```python
class RandomResizedCrop:
    """Randomly crop and resize to specified size
    
    Commonly used for ImageNet training
    """
    
    def __init__(
        self,
        size: tuple,
        scale: tuple = (0.08, 1.0),  # Crop area ratio
        ratio: tuple = (3./4., 4./3.)  # Aspect ratio
    ):
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.scale = scale
        self.ratio = ratio
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        area = h * w
        
        for _ in range(10):  # Try at most 10 times
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
                
                # Resize to target size
                return self._resize(cropped, self.size)
        
        # Fallback to center crop
        return CenterCrop(self.size)(img)
    
    def _resize(self, img: np.ndarray, size: tuple) -> np.ndarray:
        # Simple nearest-neighbor resize
        th, tw = size
        h, w = img.shape[:2]
        
        y_indices = (np.arange(th) * h / th).astype(int)
        x_indices = (np.arange(tw) * w / tw).astype(int)
        
        if img.ndim == 3:
            return img[np.ix_(y_indices, x_indices, [0,1,2])]
        return img[np.ix_(y_indices, x_indices)]
```

## 8.5 Color Transforms

### Color Jitter

```python
class ColorJitter:
    """Randomly adjust brightness, contrast, saturation"""
    
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
        
        # Brightness
        if self.brightness > 0:
            factor = np.random.uniform(
                max(0, 1 - self.brightness),
                1 + self.brightness
            )
            img = img * factor
        
        # Contrast
        if self.contrast > 0:
            factor = np.random.uniform(
                max(0, 1 - self.contrast),
                1 + self.contrast
            )
            mean = img.mean()
            img = (img - mean) * factor + mean
        
        # Saturation (RGB images only)
        if self.saturation > 0 and img.ndim == 3:
            factor = np.random.uniform(
                max(0, 1 - self.saturation),
                1 + self.saturation
            )
            gray = np.mean(img, axis=2, keepdims=True)
            img = gray + (img - gray) * factor
        
        return np.clip(img, 0, 255).astype(np.uint8)
```

### Random Grayscale

```python
class RandomGrayscale:
    """Convert to grayscale with probability p"""
    
    def __init__(self, p: float = 0.1):
        self.p = p
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p and img.ndim == 3:
            gray = np.mean(img, axis=2, keepdims=True)
            return np.repeat(gray, 3, axis=2).astype(img.dtype)
        return img
```

## 8.6 Advanced Transforms

### Random Erasing

```python
class RandomErasing:
    """Randomly erase image regions
    
    Used to simulate occlusion and improve robustness
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

## 8.7 Usage Example

### Training Augmentation

```python
from nanotorch.transforms import (
    Compose, ToFloat, Normalize,
    RandomHorizontalFlip, RandomCrop,
    ColorJitter, RandomErasing
)

# Training transforms
train_transform = Compose([
    ToFloat(),
    RandomHorizontalFlip(p=0.5),
    RandomCrop(size=224),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    RandomErasing(p=0.5),
])

# Apply transforms
image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
augmented = train_transform(image)
```

### Validation Transforms

```python
# Validation transforms (no random augmentation)
val_transform = Compose([
    ToFloat(),
    CenterCrop(size=224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

## 8.8 Complete Training Example

```python
from nanotorch import DataLoader, TensorDataset
from nanotorch.transforms import Compose, ToFloat, Normalize, RandomHorizontalFlip

# Define transforms
train_transform = Compose([
    ToFloat(),
    RandomHorizontalFlip(p=0.5),
    Normalize(mean=[0.5], std=[0.5]),
])

# Custom Dataset with transforms
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

# Usage
train_dataset = TransformDataset(X_train, y_train, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
```

## 8.9 Transform Comparison

| Transform | Purpose | Use Case |
|-----------|---------|----------|
| RandomHorizontalFlip | Horizontal flip | General |
| RandomCrop | Random crop | General |
| RandomResizedCrop | Crop + resize | ImageNet |
| ColorJitter | Color perturbation | General |
| RandomErasing | Random erasing | General, occlusion robustness |

## 8.10 Exercises

1. **Implement RandomRotation**: Randomly rotate image

2. **Implement GaussianBlur**: Gaussian blur

3. **Implement Cutout**: Random block occlusion

## Next Chapter

In the next chapter, we will introduce **convolution layers** for image and sequence processing.

→ [Chapter 9: Convolution Layers](09-conv.md)
