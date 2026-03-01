# YOLO v8 Object Detection Model Implementation Tutorial

This tutorial provides a comprehensive guide to implementing YOLO v8 (Ultralytics, 2023) object detection model from scratch using nanotorch.

## Table of Contents

1. [Overview](#overview)
2. [YOLO v8 Core Improvements](#yolo-v8-core-improvements)
3. [Model Architecture](#model-architecture)
4. [Loss Function](#loss-function)
5. [Training Process](#training-process)
6. [Code Examples](#code-examples)
7. [Common Issues](#common-issues)
8. [Summary](#summary)

---

## Overview

YOLO v8 is the latest YOLO version released by Ultralytics in 2023. It improves upon YOLO v5, especially in detection head and loss function design.

### Main Features of YOLO v8

1. **C2f Module**: Faster CSP Bottleneck structure
2. **Anchor-free Detection**: No predefined anchor boxes needed
3. **Decoupled Head**: Separate classification and regression
4. **DFL Loss**: Distribution Focal Loss
5. **Multi-size Models**: n(nano), s(small), m(medium), l(large), x(xlarge)
6. **Mosaic Augmentation**: Training data augmentation

### nanotorch YOLO v8 Implementation Modules

```
nanotorch/detection/yolo_v8/
├── __init__.py        # Module exports
├── yolo_v8_model.py   # Model architecture (C2f, Bottleneck, Backbone, Neck, DetectHead)
└── yolo_v8_loss.py    # Loss function (YOLOv8Loss, encode_targets_v8, decode_predictions_v8)

examples/yolo_v8/
└── demo.py            # Training and inference demo

tests/detection/yolo_v8/
├── test_yolov8_model.py  # Unit tests
└── test_v8_integration.py  # Integration tests
```

---

## YOLO v8 Core Improvements

### C2f Module (CSP Bottleneck with 2 convolutions)

C2f is YOLO v8's core building block, more efficient than YOLO v5's C3:

```
Input Feature (c_in)
    │
    └─── Conv1x1 → c_out/2 ──┬──→ Bottleneck×N ──┐
                              │                    │
                              └────────────────────┤
                                                   │
                         Concat(x, x1) ←───────────┘
                               │
                          Conv1x1 → Output (c_out)
```

C2f improvements over C3:
- **Fewer parameters**: Uses 2 convolutions instead of 3
- **Better gradient flow**: Cleaner branch structure
- **Higher efficiency**: Reduced computation

### Anchor-free Detection

YOLO v8 adopts anchor-free detection strategy:

| Feature | Anchor-based (YOLO v5) | Anchor-free (YOLO v8) |
|---------|------------------------|----------------------|
| Prediction | tx, ty, tw, th + offset | Direct center point and size |
| Hyperparameters | Need anchor box sizes | None needed |
| Generalization | Depends on data distribution | Better generalization |

### Decoupled Head

```
Feature Map
    │
    ├──→ Classification Branch → Conv → Conv → Class scores
    │
    └──→ Regression Branch → Conv → Conv → Bounding box + DFL
```

Advantages of decoupled head:
- **Independent Optimization**: Classification and regression optimized separately
- **Better Features**: Specialized feature extraction
- **Higher Accuracy**: Reduced task conflict

---

## Model Architecture

### 1. ConvBN (Basic Convolution Block)

```python
from nanotorch.detection.yolo_v8 import ConvBN
from nanotorch.tensor import Tensor
import numpy as np

conv_bn = ConvBN(64, 128, k=3, s=1)
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = conv_bn(x)
print(y.shape)  # (1, 128, 52, 52)
```

### 2. Bottleneck Module

```python
from nanotorch.detection.yolo_v8 import Bottleneck

bottleneck = Bottleneck(128, shortcut=True)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = bottleneck(x)
print(y.shape)  # (1, 128, 52, 52)
```

Bottleneck structure:
```
Input (c)
    │
    ├──→ Conv1x1 → c/2 → Conv3x3 → c
    │                        │
    └────────────────────────┴──→ Add → Output (c)
```

### 3. C2f Module

```python
from nanotorch.detection.yolo_v8 import C2f

c2f = C2f(128, 256, n=2, shortcut=True)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = c2f(x)
print(y.shape)  # (1, 256, 52, 52)
```

### 4. Backbone Network

```python
from nanotorch.detection.yolo_v8 import Backbone

backbone = Backbone(in_ch=3)
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
features = backbone(x)

print(features['scale1'].shape)  # (1, 512, 20, 20) - stride=32
print(features['scale2'].shape)  # (1, 256, 40, 40) - stride=16
print(features['scale3'].shape)  # (1, 128, 80, 80) - stride=8
```

Backbone structure:
```
Input (3×640×640)
    ↓ Stem: Conv3×3(s2) → 32 channels
    ↓ Stage1: Conv3×3(s2) → C2f → 64 channels
    ↓ Stage2: Conv3×3(s2) → C2f → 128 channels → Output s3 (stride=8)
    ↓ Stage3: Conv3×3(s2) → C2f → 256 channels → Output s2 (stride=16)
    ↓ Stage4: Conv3×3(s2) → C2f → 512 channels → Output s1 (stride=32)
```

### 5. Neck Network (PANet)

```python
from nanotorch.detection.yolo_v8 import Neck

neck = Neck(channels=[128, 256, 512])
neck_out = neck(features)

print(neck_out['p3'].shape)  # (1, 128, 80, 80)
print(neck_out['p4'].shape)  # (1, 256, 40, 40)
print(neck_out['p5'].shape)  # (1, 512, 20, 20)
```

### 6. DetectHead (Decoupled Detection Head)

```python
from nanotorch.detection.yolo_v8 import DetectHead

head = DetectHead(256, num_classes=80)
x = Tensor(np.random.randn(1, 256, 40, 40).astype(np.float32))
y = head(x)
print(y.shape)  # (1, 85, 40, 40)
```

### 7. Complete YOLOv8 Model

```python
from nanotorch.detection.yolo_v8 import YOLOv8, build_yolov8

# Method 1: Create directly
model = YOLOv8(num_classes=80, input_size=640)

# Method 2: Use factory function
model = build_yolov8(num_classes=80, input_size=640)

# Forward pass
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
output = model(x)

print(output['small'].shape)   # (1, 85, 20, 20) - Large object detection
print(output['medium'].shape)  # (1, 85, 40, 40) - Medium object detection
print(output['large'].shape)   # (1, 85, 80, 80) - Small object detection
```

---

## Loss Function

### YOLO v8 Loss Function Design

YOLO v8's loss function contains three parts:

$$
L = \lambda_{box} L_{box} + \lambda_{cls} L_{cls} + \lambda_{dfl} L_{dfl}
$$

### Bounding Box Loss (CIoU)

$$
L_{box} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$

### Classification Loss (BCE)

$$
L_{cls} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left[ y_{ic} \log(\hat{p}_{ic}) + (1 - y_{ic}) \log(1 - \hat{p}_{ic}) \right]
$$

### DFL (Distribution Focal Loss)

DFL models bounding box regression as a distribution problem:

**Distribution Modeling**:

$$
\hat{y} = \sum_{n=0}^{N-1} P(n) \cdot n, \quad N = \text{reg\_max}
$$

Where $P(n)$ is the predicted probability distribution.

**DFL Loss**:

$$
L_{dfl} = -\sum_{n=0}^{N-1} \left[ y_n \log(\hat{P}(n)) \right]
$$

Where:
- Traditional method: Directly predict (x, y, w, h)
- DFL method: Predict distribution, take expected value

**Target Distribution Construction**:

For target value $y \in [l, l+1]$, where $l = \lfloor y \rfloor$:

$$
y_l = l + 1 - y, \quad y_{l+1} = y - l
$$

### C2f Module

C2f module feature flow:

$$
x_{out} = \text{Concat}(x_{split}, \text{BottleNeck}_1(x_{split}), \ldots, \text{BottleNeck}_k(x_{split}))
$$

### YOLOv8Loss

YOLO v8's loss function contains three parts:

```python
from nanotorch.detection.yolo_v8 import YOLOv8Loss

loss_fn = YOLOv8Loss(
    num_classes=80,
    reg_max=16,         # DFL maximum value
    lambda_box=7.5,     # Bounding box loss weight
    lambda_cls=0.5,     # Classification loss weight
    lambda_dfl=1.5      # DFL loss weight
)

# Compute loss
predictions = {
    'small': Tensor(np.random.randn(2, 85, 20, 20).astype(np.float32) * 0.1),
    'medium': Tensor(np.random.randn(2, 85, 40, 40).astype(np.float32) * 0.1),
    'large': Tensor(np.random.randn(2, 85, 80, 80).astype(np.float32) * 0.1)
}

targets = [
    {'boxes': np.array([[100, 100, 200, 200]], dtype=np.float32),
     'labels': np.array([0], dtype=np.int64)},
]

loss, loss_dict = loss_fn(predictions, targets)
print(f"Total Loss: {loss.item():.4f}")
print(f"Box Loss: {loss_dict['box_loss']:.4f}")
print(f"Cls Loss: {loss_dict['cls_loss']:.4f}")
print(f"DFL Loss: {loss_dict['dfl_loss']:.4f}")
```

### DFL (Distribution Focal Loss)

DFL models bounding box regression as a distribution problem:
- Traditional method: Directly predict (x, y, w, h)
- DFL method: Predict distribution, take expected value

```
Bounding box offset = Σ(P(n) * n), n ∈ [0, reg_max]
```

### Encoding and Decoding Functions

```python
from nanotorch.detection.yolo_v8 import encode_targets_v8, decode_predictions_v8

# Encode targets
boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
labels = np.array([0], dtype=np.int64)
targets = encode_targets_v8(boxes, labels, grid_sizes=[80, 40, 20])

# Decode predictions
boxes, scores, class_ids = decode_predictions_v8(
    predictions=output['large'],
    conf_threshold=0.25,
    num_classes=80
)
```

---

## Training Process

### Complete Training Example

```python
from nanotorch.tensor import Tensor
from nanotorch.optim import AdamW
from nanotorch.optim.lr_scheduler import CosineWarmupScheduler
from nanotorch.detection.yolo_v8 import build_yolov8, YOLOv8Loss
import numpy as np

# Create model
model = build_yolov8(num_classes=80, input_size=640)

# Loss function and optimizer
loss_fn = YOLOv8Loss(num_classes=80)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=3, max_epochs=50)

# Training loop
for epoch in range(50):
    model.train()
    total_loss = 0
    
    for batch_idx in range(100):
        # Generate random data
        images = Tensor(np.random.randn(4, 3, 640, 640).astype(np.float32))
        targets = [{
            'boxes': np.random.rand(3, 4).astype(np.float32) * 600,
            'labels': np.random.randint(0, 80, 3).astype(np.int64)
        } for _ in range(4)]
        
        optimizer.zero_grad()
        output = model(images)
        
        # Compute loss
        loss, loss_dict = loss_fn(output, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / 100
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

---

## Code Examples

### Running Demo

```bash
# View model architecture
python examples/yolo_v8/demo.py --mode arch

# Train model
python examples/yolo_v8/demo.py --mode train --epochs 5

# Inference demo
python examples/yolo_v8/demo.py --mode inference

# Complete workflow
python examples/yolo_v8/demo.py --mode both
```

### Demo Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | arch | Run mode (arch/train/inference/both) |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 2 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--image-size` | 640 | Input image size |
| `--num-classes` | 80 | Number of classes |

### Running Tests

```bash
# Unit tests
python -m pytest tests/detection/yolo_v8/test_yolov8_model.py -v

# Integration tests
python -m pytest tests/detection/yolo_v8/test_v8_integration.py -v

# All v8 tests
python -m pytest tests/detection/yolo_v8/ -v
```

---

## YOLO v7 vs YOLO v8 Comparison

| Feature | YOLO v7 | YOLO v8 |
|---------|---------|---------|
| Developer | WongKinYiu | Ultralytics |
| Core Module | E-ELAN | C2f |
| Detection Head | Coupled | Decoupled |
| Anchors | Anchor-based | Anchor-free |
| Loss Function | CIoU + BCE | DFL + BCE + CIoU |
| Ease of Use | Medium | High |

---

## Common Issues

### 1. What's the difference between C2f and C3?

| Feature | C3 | C2f |
|---------|-----|-----|
| Convolution Count | 3 | 2 |
| Branch Structure | Fixed | Flexible |
| Parameter Count | More | Fewer |
| Computation Efficiency | Medium | Higher |

### 2. Why use Anchor-free?

- **Simpler**: No need to design anchor boxes
- **More General**: Doesn't depend on data distribution
- **More Accurate**: Reduced hyperparameter tuning

### 3. Advantages of DFL Loss?

- **More Stable**: Distribution modeling is more robust
- **More Precise**: Higher bounding box regression accuracy
- **End-to-End**: No post-processing needed

---

## Summary

This tutorial covered the complete process of implementing YOLO v8 using nanotorch:

1. **Core Improvements**: C2f module, Anchor-free detection, decoupled head
2. **Model Architecture**: Backbone + Neck (PANet) + Decoupled Head
3. **Loss Function**: DFL + BCE + CIoU
4. **Training & Inference**: Complete training and inference workflow

YOLO v8 is currently one of the most popular object detection models, with both ease of use and performance at high levels.

---

## References

1. **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics

2. **YOLOv8 Documentation**: https://docs.ultralytics.com/

3. **nanotorch YOLO v7 Tutorial**: `/docs/tutorials/23-yolov7_EN.md`

4. **nanotorch YOLO v9 Tutorial**: `/docs/tutorials/25-yolov9_EN.md`
