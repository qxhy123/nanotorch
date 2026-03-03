# Chapter 8: Data Augmentation

## How Many Photos Can One Cat Photo Become?...

You have a photo of a cat. Front view, perfect lighting, the cat sits squarely in the center of the frame.

If you only let the model see this one photo, how will it understand "cat"?

"A cat looks like this: front view, good lighting, sitting in the center."

So when it sees a side-view cat, a cat in dim light, a running cat—it's completely lost.

**Data augmentation is about turning one image into thousands.**

Flip it, it's still a cat. Rotate it, it's still a cat. Brighten it, it's still a cat. Crop a part, it's still a cat.

```
Original data:
  1000 photos → Model memorizes them

Data augmentation:
  1000 photos → Flip → 2000 photos
              → Rotate → 4000 photos
              → Crop → 8000 photos
              → Color change → 16000 photos

  The model learns "the essence of cat"
  Not "what this specific photo looks like"
```

**Data augmentation is the vaccine against overfitting.** It teaches the model: forms may change, essence remains.

---

## 8.1 Why Do We Need Data Augmentation?

### Problem: Too Little Data

```
You only have 100 cat photos

The model will "rote memorize":
  "An image at position (x=10,y=20), gray color, large size is a cat"

Won't recognize new cats!
```

### Solution: Data Augmentation

```
Original image → Transform → Augmented image

    🐱    →  Flip   →  🐱 (mirrored)
    🐱    →  Rotate →  🐱 (tilted)
    🐱    →  Crop   →  🐱 (partial)
    🐱    →  Color  →  🐱 (grayscale)

One image → Many variants → Model learns "essence of cat"
```

### Effect Comparison

```
Without data augmentation:
  Training: 100% accuracy (rote memorization)
  Testing: 60% accuracy (fails on new images)

With data augmentation:
  Training: 85% accuracy (learns patterns)
  Testing: 82% accuracy (strong generalization!)
```

---

## 8.2 Transform Base Class

### Composing Transforms

```python
# transforms/__init__.py
from typing import Callable, List
import numpy as np

class Compose:
    """
    Compose multiple transforms

    Analogy: Assembly line factory
      Raw material → Process 1 → Process 2 → Process 3 → Product

    Image → Flip → Crop → Normalize → Output
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


# Usage
transform = Compose([
    RandomHorizontalFlip(p=0.5),  # 50% chance to flip
    RandomCrop(size=224),         # Random crop
    Normalize(mean=[0.5], std=[0.5])  # Normalize
])

augmented_img = transform(original_img)
```

---

## 8.3 Basic Transforms

### Normalization

```
Why normalize?

Original data: [0, 255] range
  Problem: Values too large, training unstable

After normalization: [-1, 1] or [0, 1] range
  Benefit: Small values, stable training
```

```python
class Normalize:
    """
    Normalize image

    output = (input - mean) / std

    Analogy: Converting scores to standard scores
      Original: Chinese 150-point scale, Math 100-point scale
      After standardization: Both have mean 0, std 1
    """

    def __init__(self, mean: List[float], std: List[float]):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return (img - self.mean) / self.std


# ImageNet normalization (mean and std from ImageNet dataset)
normalize = Normalize(
    mean=[0.485, 0.456, 0.406],  # RGB channel means
    std=[0.229, 0.224, 0.225]    # RGB channel stds
)
```

### Type Conversion

```python
class ToFloat:
    """Convert to float32 (0-1 range)"""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return img.astype(np.float32) / 255.0


class ToUint8:
    """Convert to uint8 (0-255 range)"""

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return (img * 255).astype(np.uint8)
```

---

## 8.4 Geometric Transforms

### Random Horizontal Flip

```
Original image:       After horizontal flip:

  🚗→                    ←🚗
 Facing right          Facing left

A cat is still a cat, just the direction changed
```

```python
class RandomHorizontalFlip:
    """
    Randomly flip horizontally with probability p

    Why useful?
      - A cat facing left or right is still a cat
      - Teaches the model "left-right symmetry"
    """

    def __init__(self, p: float = 0.5):
        self.p = p  # Flip probability

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

```
Original image (256x256):    Random crop (224x224):

┌────────────────┐          ┌──────────┐
│  🌳    🌳      │          │  🌳      │
│      🐱        │    →     │    🐱    │
│  🌳    🌳      │          │  🌳      │
└────────────────┘          └──────────┘

Each crop at different position → Model learns to focus on different areas
```

```python
class RandomCrop:
    """
    Randomly crop to specified size

    Benefits:
      - Model learns objects aren't always centered
      - Effectively increases training samples
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
    """
    Center crop

    For validation/testing: Deterministic crop, same result every time
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

### Random Resized Crop

```
Standard ImageNet training augmentation:

1. Randomly select area (8%-100% of image)
2. Random aspect ratio (3:4 to 4:3)
3. Crop and resize to 224x224

┌────────────────────┐
│    ┌──────┐        │
│    │ Crop │        │  →  Resize to 224x224
│    └──────┘        │
└────────────────────┘
```

```python
class RandomResizedCrop:
    """
    Randomly crop and resize to specified size

    The most commonly used augmentation for ImageNet training!
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

                return self._resize(cropped, self.size)

        # Fallback to center crop
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

## 8.5 Color Transforms

### Color Jitter

```
Original image:      After jitter:

  🌈                  🌈 (brighter/darker)
  Colorful            Slightly gray/more vivid

Simulates different lighting conditions
```

```python
class ColorJitter:
    """
    Randomly adjust brightness, contrast, saturation

    Simulates real-world variations:
      - Different lighting at different times
      - Different camera settings
    """

    def __init__(
        self,
        brightness: float = 0.0,  # Brightness range
        contrast: float = 0.0,    # Contrast range
        saturation: float = 0.0   # Saturation range
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.copy().astype(np.float32)

        # Brightness adjustment
        if self.brightness > 0:
            factor = np.random.uniform(
                max(0, 1 - self.brightness),
                1 + self.brightness
            )
            img = img * factor

        # Contrast adjustment
        if self.contrast > 0:
            factor = np.random.uniform(
                max(0, 1 - self.contrast),
                1 + self.contrast
            )
            mean = img.mean()
            img = (img - mean) * factor + mean

        # Saturation adjustment (RGB images only)
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
    """
    Convert to grayscale with probability p

    Why?
      - Some scenes are naturally gray
      - Teaches model to focus on shape, not just color
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

## 8.6 Advanced Transforms

### Random Erasing

```
Original image:       After random erasing:

  🐱🐱🐱                🐱■🐱
  Complete cat         Partially occluded

Simulates object occlusion
```

```python
class RandomErasing:
    """
    Randomly erase image regions

    Simulates occlusion, teaches model:
      - Not to rely on any specific region
      - To recognize from partial information
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: tuple = (0.02, 0.33),  # Erase area size
        ratio: tuple = (0.3, 3.3),    # Aspect ratio
        value: float = 0.0            # Fill value
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

## 8.7 Training vs Validation Transforms

```
During training: Use random augmentation

  Original → Random flip → Random crop → Color jitter → Normalize
             ↑ Different each time

During validation: Only deterministic transforms

  Original → Center crop → Normalize
             ↑ Same each time
```

```python
from nanotorch.transforms import (
    Compose, ToFloat, Normalize,
    RandomHorizontalFlip, RandomCrop, CenterCrop,
    ColorJitter, RandomErasing
)

# Training transforms (with random augmentation)
train_transform = Compose([
    ToFloat(),
    RandomHorizontalFlip(p=0.5),      # Random flip
    RandomCrop(size=224),             # Random crop
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    RandomErasing(p=0.5),             # Random erase
])

# Validation transforms (deterministic)
val_transform = Compose([
    ToFloat(),
    CenterCrop(size=224),             # Center crop
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 8.8 Complete Usage Example

```python
from nanotorch import DataLoader
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

---

## 8.9 Transform Comparison Table

| Transform | Purpose | Use Case | Train/Val |
|-----------|---------|----------|-----------|
| RandomHorizontalFlip | Left-right flip | General | Training |
| RandomCrop | Random crop | General | Training |
| RandomResizedCrop | Crop + resize | ImageNet | Training |
| ColorJitter | Color perturbation | General | Training |
| RandomErasing | Random occlusion | Occlusion robustness | Training |
| CenterCrop | Center crop | General | Validation |
| Normalize | Normalize | **Required** | Both |

---

## 8.10 Common Pitfalls

### Pitfall 1: Using random augmentation during validation

```python
# Wrong: Results differ each time during validation
val_transform = Compose([
    RandomCrop(224),  # Random!
    Normalize(...),
])

# Correct: Use deterministic transforms for validation
val_transform = Compose([
    CenterCrop(224),  # Deterministic
    Normalize(...),
])
```

### Pitfall 2: Wrong normalization parameters

```python
# Wrong: Using ImageNet params for MNIST
Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet

# Correct: MNIST is grayscale, single channel
Normalize(mean=[0.1307], std=[0.3081])  # MNIST
```

### Pitfall 3: Labels not transformed accordingly

```python
# Note: For some tasks, labels must change too!
# Object detection: Bounding box coordinates must flip
# Semantic segmentation: Segmentation map must transform synchronously
```

---

## 8.11 Exercises

### Basic Exercises

1. **Implement RandomRotation**: Randomly rotate image ±30°

2. **Implement GaussianBlur**: Gaussian blur

3. **Implement Cutout**: Fixed-size random occlusion

### Advanced Exercises

4. **Implement MixUp**: Mix two images
   ```
   new_image = 0.7 * image1 + 0.3 * image2
   new_label = 0.7 * label1 + 0.3 * label2
   ```

5. **Implement AutoAugment**: Automatically search for best augmentation policy

---

## Summary in One Sentence

| Concept | One Sentence |
|---------|--------------|
| Data Augmentation | One image becomes many, learning is more solid |
| Geometric Transform | Flip, crop, rotate (position changes) |
| Color Transform | Brightness, contrast, saturation (color changes) |
| Training Augmentation | Random transforms, different each time |
| Validation Augmentation | Deterministic transforms, same each time |

---

## Next Chapter

Now our data is more varied!

In the next chapter, we'll learn about **convolution layers** — the core component for processing images.

→ [Chapter 9: Convolution Layers](09-conv.md)

```python
# Preview: What you'll learn next chapter
conv = Conv2D(in_channels=3, out_channels=64, kernel_size=3)
# Slide a 3x3 small window across the entire image
# Extract local features from the image
```
