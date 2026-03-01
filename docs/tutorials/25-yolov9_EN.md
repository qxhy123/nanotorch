# YOLO v9 Object Detection Model Implementation Tutorial

This tutorial provides a comprehensive guide to implementing YOLO v9 (Programmable Gradient Information, 2024) object detection model from scratch using nanotorch.

## Table of Contents

1. [Overview](#overview)
2. [YOLO v9 Core Improvements](#yolo-v9-core-improvements)
3. [Model Architecture](#model-architecture)
4. [Loss Function](#loss-function)
5. [Training Process](#training-process)
6. [Code Examples](#code-examples)
7. [Common Issues](#common-issues)
8. [Summary](#summary)

---

## Overview

YOLO v9 is developed by WongKinYiu and released in 2024. It solves the information bottleneck problem in deep networks through Programmable Gradient Information (PGI) and Generalized Efficient Layer Aggregation Network (GELAN), achieving significant performance improvements.

### Main Features of YOLO v9

1. **GELAN (Generalized Efficient Layer Aggregation Network)**: Generalized efficient layer aggregation network
2. **PGI (Programmable Gradient Information)**: Programmable gradient information
3. **RepConv Reparameterization**: Multi-branch during training, single-branch during inference
4. **Information Bottleneck Solution**: Feature preservation in deep networks
5. **Auxiliary Branch**: Reversible function branch

### nanotorch YOLO v9 Implementation Modules

```
nanotorch/detection/yolo_v9/
├── __init__.py        # Module exports
├── yolo_v9_model.py   # Model architecture (GELAN, RepConv, Backbone, Head, YOLOv9)
└── yolo_v9_loss.py    # Loss function (YOLOv9Loss, encode_targets_v9, decode_predictions_v9)

examples/yolo_v9/
└── demo.py            # Training and inference demo

tests/detection/yolo_v9/
├── test_yolov9_model.py  # Unit tests
└── test_v9_integration.py  # Integration tests
```

---

## YOLO v9 Core Improvements

### GELAN (Generalized Efficient Layer Aggregation Network)

GELAN is YOLO v9's core building block, a generalized version of ELAN:

```
Input Feature (c_in)
    │
    ├─── Conv1x1 → mid_channels ────────────────────┐
    │                                                │
    └─── Conv1x1 → mid_channels → Conv3x3×N ────────┤
                                                      │
                         Concat(y1, y2) ←─────────────┘
                               │
                          Conv1x1 → Output (c_out)
```

Advantages of GELAN:
- **Flexibility**: Supports arbitrary computation blocks
- **Scalability**: Easy to add new module types
- **Efficiency**: Maintains ELAN's efficiency

### PGI (Programmable Gradient Information)

PGI solves the information bottleneck problem in deep networks:

```
Input
  │
  ├──→ Main Branch → Deep Features → Information Loss
  │                          │
  └──→ Auxiliary Branch ─────┴──→ Gradient Compensation
```

Key components of PGI:
1. **Auxiliary Reversible Branch**: Maintains information integrity
2. **Multi-level Auxiliary Information**: Gradient supervision at different scales
3. **Programmability**: Flexible gradient propagation configuration

### RepConv (Reparameterization Convolution)

```python
# During training
y = Conv3x3(x) + Conv1x1(x)  # Multi-branch

# During inference (after reparameterization)
y = Conv3x3_fused(x)  # Single-branch
```

---

## Model Architecture

### 1. ConvBN (Basic Convolution Block)

```python
from nanotorch.detection.yolo_v9 import ConvBN
from nanotorch.tensor import Tensor
import numpy as np

conv_bn = ConvBN(64, 128, k=3, s=1)
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = conv_bn(x)
print(y.shape)  # (1, 128, 52, 52)
```

### 2. RepConv (Reparameterization Convolution)

```python
from nanotorch.detection.yolo_v9 import RepConv

rep_conv = RepConv(64, 128, k=3, s=1)
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = rep_conv(x)
print(y.shape)  # (1, 128, 52, 52)
```

RepConv structure:
```
Input
    │
    ├──→ Conv3×3 → BN → y1
    │
    └──→ Conv1×1 → BN → y2
              │
              └──→ Add → SiLU → Output
```

### 3. GELAN Module

```python
from nanotorch.detection.yolo_v9 import GELAN

gelan = GELAN(128, 256, n=2)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = gelan(x)
print(y.shape)  # (1, 256, 52, 52)
```

### 4. Backbone Network

```python
from nanotorch.detection.yolo_v9 import Backbone

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
    ↓ Stem: Conv3×3(s2) → Conv3×3(s2)
    ↓ Stage1: GELAN(64→64)
    ↓ Down1: Conv3×3(s2)
    ↓ Stage2: GELAN(128→128) → Output s3 (stride=8)
    ↓ Down2: Conv3×3(s2)
    ↓ Stage3: GELAN(256→256) → Output s2 (stride=16)
    ↓ Down3: Conv3×3(s2)
    ↓ Stage4: GELAN(512→512) → Output s1 (stride=32)
```

### 5. Complete YOLOv9 Model

```python
from nanotorch.detection.yolo_v9 import YOLOv9, build_yolov9

# Method 1: Create directly
model = YOLOv9(num_classes=80, input_size=640)

# Method 2: Use factory function
model = build_yolov9(num_classes=80, input_size=640)

# Forward pass
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
output = model(x)

print(output['small'].shape)   # (1, 85, 20, 20) - Large object detection
print(output['medium'].shape)  # (1, 85, 40, 40) - Medium object detection
print(output['large'].shape)   # (1, 85, 80, 80) - Small object detection
```

---

## Loss Function

### YOLO v9 Loss Function Design

YOLO v9's loss function:

$$
L = \lambda_{box} L_{box} + \lambda_{obj} L_{obj} + \lambda_{noobj} L_{noobj} + \lambda_{class} L_{class}
$$

### Bounding Box Loss

$$
L_{box} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$

### GELAN Module Feature Aggregation

GELAN (Generalized Efficient Layer Aggregation Network):

$$
\text{Output} = \text{Conv}_{out}(\text{Concat}(\text{Branch}_1, \text{Branch}_2, \ldots, \text{Branch}_k))
$$

Each branch computation:

$$
\text{Branch}_i = \text{Conv}_{3 \times 3}^{(i)}(x)
$$

### PGI (Programmable Gradient Information)

PGI information flow:

$$
\text{Info}_{preserved} = \text{Main Branch} + \text{Auxiliary Branch}
$$

Auxiliary branch loss:

$$
L_{aux} = \lambda_{aux} \cdot L_{task}(\text{Aux}_{output}, \text{target})
$$

### YOLOv9Loss

```python
from nanotorch.detection.yolo_v9 import YOLOv9Loss

loss_fn = YOLOv9Loss(
    num_classes=80,
    lambda_box=5.0,
    lambda_obj=1.0,
    lambda_noobj=0.5,
    lambda_class=1.0
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
```

---

## Training Process

### Complete Training Example

```python
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import CosineAnnealingLR
from nanotorch.detection.yolo_v9 import build_yolov9, YOLOv9Loss
import numpy as np

# Create model
model = build_yolov9(num_classes=80, input_size=640)

# Loss function and optimizer
loss_fn = YOLOv9Loss(num_classes=80)
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=50)

# Training loop
for epoch in range(50):
    model.train()
    total_loss = 0
    
    for batch_idx in range(100):
        images = Tensor(np.random.randn(4, 3, 640, 640).astype(np.float32))
        targets = [{
            'boxes': np.random.rand(3, 4).astype(np.float32) * 600,
            'labels': np.random.randint(0, 80, 3).astype(np.int64)
        } for _ in range(4)]
        
        optimizer.zero_grad()
        output = model(images)
        loss, loss_dict = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss/100:.4f}")
```

---

## Code Examples

### Running Demo

```bash
# View model architecture
python examples/yolo_v9/demo.py --mode arch

# Train model
python examples/yolo_v9/demo.py --mode train --epochs 5

# Inference demo
python examples/yolo_v9/demo.py --mode inference

# Complete workflow
python examples/yolo_v9/demo.py --mode both
```

### Running Tests

```bash
# Unit tests
python -m pytest tests/detection/yolo_v9/test_yolov9_model.py -v

# Integration tests
python -m pytest tests/detection/yolo_v9/test_v9_integration.py -v

# All v9 tests
python -m pytest tests/detection/yolo_v9/ -v
```

---

## YOLO v8 vs YOLO v9 Comparison

| Feature | YOLO v8 | YOLO v9 |
|---------|---------|---------|
| Developer | Ultralytics | WongKinYiu |
| Core Module | C2f | GELAN |
| Gradient Optimization | None | PGI |
| Reparameterization | None | RepConv |
| Information Bottleneck | Not Solved | Solved |

---

## Common Issues

### 1. How does PGI solve information bottleneck?

PGI through auxiliary reversible branch:
- Maintains input information integrity
- Provides precise gradient signals
- Avoids information loss in deep networks

### 2. GELAN improvements over ELAN?

| Feature | ELAN | GELAN |
|---------|------|-------|
| Computation Block | Fixed | Replaceable |
| Flexibility | Low | High |
| Scalability | Limited | Strong |

### 3. When to reparameterize RepConv?

- **During Training**: Multi-branch structure, rich features
- **Before Inference**: Fused to single branch, improved speed

---

## Summary

This tutorial covered the complete process of implementing YOLO v9 using nanotorch:

1. **Core Improvements**: GELAN, PGI, RepConv
2. **Model Architecture**: Backbone + Head
3. **Loss Function**: Bounding box loss + confidence loss + classification loss
4. **Training & Inference**: Complete training and inference workflow

YOLO v9 maintains excellent feature representation in deep networks by solving the information bottleneck problem, making it one of the most advanced object detection models currently available.

---

## References

1. **YOLOv9 Paper**: "YOLOv9: Learning What You Want to Learn Using Programmable Gradient Information"

2. **GitHub Repository**: https://github.com/WongKinYiu/yolov9

3. **nanotorch YOLO v8 Tutorial**: `/docs/tutorials/24-yolov8_EN.md`

4. **nanotorch YOLO v10 Tutorial**: `/docs/tutorials/26-yolov10_EN.md`
