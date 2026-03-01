# YOLO v11 Object Detection Model Implementation Tutorial

## The Latest Evolution...

YOLO doesn't stand still.

Every year, sometimes twice a year, a new version arrives. Each brings improvements—some revolutionary, some evolutionary. Together, they tell a story of continuous progress.

**YOLO v11 is the latest chapter in this ongoing story.**

```
The YOLO v11 Philosophy:

  YOLO v8 was about:
    "Let's make anchor-free work beautifully."

  YOLO v9 was about:
    "Let's understand information bottlenecks."

  YOLO v10 was about:
    "Let's eliminate NMS."

  YOLO v11 is about:
    "Let's make everything a little bit better."

  The improvements:
    - C3k2 module → Faster feature extraction
    - Enhanced PANet → Better multi-scale fusion
    - Improved detection head → Cleaner predictions
    - Optimized training → Faster convergence

  Not revolutionary.
  But evolution compounds.
  And v11 is the state of the art.
```

**YOLO v11 is maturity in action.** It doesn't need to reinvent the wheel—it just needs to make the wheel rounder, faster, more reliable. The architecture builds on v8's anchor-free foundation, adds insights from v9 and v10, and refines everything for the best accuracy-speed tradeoff yet.

In this tutorial, we'll implement YOLO v11 from scratch. We'll see how C3k2 improves on C2f, how the enhanced feature pyramid captures multi-scale information, and how the latest YOLO continues the tradition of making object detection faster and more accurate.

---

## Table of Contents

1. [Overview](#overview)
2. [YOLO v11 Core Improvements](#yolo-v11-core-improvements)
3. [Model Architecture](#model-architecture)
4. [Loss Function](#loss-function)
5. [Training Process](#training-process)
6. [Code Examples](#code-examples)
7. [Common Issues](#common-issues)
8. [Summary](#summary)

---

## Overview

YOLO v11 is the latest YOLO version released by Ultralytics in 2024. It improves upon YOLO v8's architecture, achieving better accuracy-speed balance.

### Main Features of YOLO v11

1. **C3k2 Module**: Faster C3k structure
2. **Enhanced Feature Extraction**: Improved network design
3. **SPPF Module**: Fast Spatial Pyramid Pooling
4. **PANet Neck**: Path Aggregation Network
5. **Multi-size Models**: n(nano), s(small), m(medium), l(large), x(xlarge)
6. **SOTA Performance**: Latest accuracy-speed balance

### nanotorch YOLO v11 Implementation Modules

```
nanotorch/detection/yolo_v11/
├── __init__.py        # Module exports
├── yolo_v11_model.py  # Model architecture (C3k2, SPPF, Bottleneck, Backbone, Neck, DetectHead)
└── yolo_v11_loss.py   # Loss function (YOLOv11Loss, encode_targets_v11, decode_predictions_v11)

examples/yolo_v11/
└── demo.py            # Training and inference demo

tests/detection/yolo_v11/
├── test_yolov11_model.py  # Unit tests
└── test_v11_integration.py  # Integration tests
```

---

## YOLO v11 Core Improvements

### C3k2 Module (Faster CSP Bottleneck with 2 convolutions)

C3k2 is YOLO v11's core building block, more efficient than C3k:

```
Input Feature (c_in)
    │
    ├─── Conv1x1 → c_mid ─────────────────────┐
    │                                          │
    └─── Conv1x1 → c_mid → Bottleneck×N ──────┤
                                                │
                     Concat(y1, y2) ←───────────┘
                               │
                          Conv1x1 → Output (c_out)
```

C3k2 improvements over C3k:
- **Faster Computation**: Optimized parallel structure
- **Better Gradient Flow**: Dual-branch design
- **Higher Efficiency**: More efficient parameter utilization

### Bottleneck Module

```
Input (c)
    │
    ├──→ Conv1x1 → c/2 → Conv3x3 → c
    │                        │
    └────────────────────────┴──→ Add → Output (c)
```

### SPPF Module (Spatial Pyramid Pooling Fast)

```
Input Feature
    │
    ↓ Conv1×1
    │
    ├──→ y1 (original)
    │
    ├──→ y2 = MaxPool(y1)
    │
    ├──→ y3 = MaxPool(y2)
    │
    └──→ y4 = MaxPool(y3)
           │
           ↓ Concat(y1, y2, y3, y4)
           │
           ↓ Conv1×1
           │
           ↓ Output
```

SPPF advantages:
- **Multi-scale Receptive Field**: Captures context at different scales
- **Efficient Implementation**: Consecutive pooling replaces parallel pooling
- **Parameter Efficiency**: Fewer parameters

---

## Model Architecture

### 1. ConvBN (Basic Convolution Block)

```python
from nanotorch.detection.yolo_v11 import ConvBN
from nanotorch.tensor import Tensor
import numpy as np

conv_bn = ConvBN(64, 128, k=3, s=1)
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = conv_bn(x)
print(y.shape)  # (1, 128, 52, 52)
```

### 2. Bottleneck Module

```python
from nanotorch.detection.yolo_v11 import Bottleneck

bottleneck = Bottleneck(128, shortcut=True)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = bottleneck(x)
print(y.shape)  # (1, 128, 52, 52)
```

### 3. C3k2 Module

```python
from nanotorch.detection.yolo_v11 import C3k2

c3k2 = C3k2(128, 256, n=2, shortcut=True)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = c3k2(x)
print(y.shape)  # (1, 256, 52, 52)
```

### 4. SPPF Module

```python
from nanotorch.detection.yolo_v11 import SPPF

sppf = SPPF(512, 512, k=5)
x = Tensor(np.random.randn(1, 512, 20, 20).astype(np.float32))
y = sppf(x)
print(y.shape)  # (1, 512, 20, 20)
```

### 5. Backbone Network

```python
from nanotorch.detection.yolo_v11 import Backbone

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
    ↓ Stage1: Conv3×3(s2) → C3k2 → 64 channels
    ↓ Stage2: Conv3×3(s2) → C3k2 → 128 channels → Output s3 (stride=8)
    ↓ Stage3: Conv3×3(s2) → C3k2 → 256 channels → Output s2 (stride=16)
    ↓ Stage4: Conv3×3(s2) → C3k2 → SPPF → 512 channels → Output s1 (stride=32)
```

### 6. Neck Network (PANet)

```python
from nanotorch.detection.yolo_v11 import Neck

neck = Neck(channels=[128, 256, 512])
neck_out = neck(features)

print(neck_out['p3'].shape)  # (1, 128, 80, 80)
print(neck_out['p4'].shape)  # (1, 256, 40, 40)
print(neck_out['p5'].shape)  # (1, 512, 20, 20)
```

### 7. Complete YOLOv11 Model

```python
from nanotorch.detection.yolo_v11 import YOLOv11, build_yolov11

# Method 1: Create directly
model = YOLOv11(num_classes=80, input_size=640)

# Method 2: Use factory function
model = build_yolov11(num_classes=80, input_size=640)

# Forward pass
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
output = model(x)

print(output['small'].shape)   # (1, 85, 20, 20) - Large object detection
print(output['medium'].shape)  # (1, 85, 40, 40) - Medium object detection
print(output['large'].shape)   # (1, 85, 80, 80) - Small object detection
```

---

## Loss Function

### YOLO v11 Loss Function Design

YOLO v11 inherits v8's loss design:

$$
L = \lambda_{box} L_{box} + \lambda_{cls} L_{cls} + \lambda_{dfl} L_{dfl}
$$

### Bounding Box Loss (CIoU)

$$
L_{box} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$

Where:
- $\rho$: Euclidean distance between predicted and ground truth box centers
- $c$: Diagonal length of smallest enclosing box
- $v = \frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h})^2$
- $\alpha = \frac{v}{1-\text{IoU}+v}$

### Classification Loss (BCE)

$$
L_{cls} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left[ y_{ic} \log(\hat{p}_{ic}) + (1 - y_{ic}) \log(1 - \hat{p}_{ic}) \right]
$$

### DFL (Distribution Focal Loss)

$$
\hat{y} = \sum_{n=0}^{N-1} P(n) \cdot n, \quad N = \text{reg\_max}
$$

$$
L_{dfl} = -\sum_{n=0}^{N-1} \left[ y_n \log(\hat{P}(n)) \right]
$$

### C3k2 Module

C3k2 is an optimized version of C3k:

$$
\text{Output} = \text{Conv}_{out}(\text{Concat}(x, \text{BottleNeck}_1(x), \text{BottleNeck}_2(x)))
$$

### C2PSA Module

C2PSA (C2 with Partial Self-Attention):

$$
\text{Output} = \text{Conv}(\text{PartialSA}(\text{Conv}(x)))
$$

Partial self-attention reduces computation:

$$
\text{PartialSA}(x) = \text{Concat}(\text{SA}(x_{part}), x_{rest})
$$

### YOLOv11Loss

```python
from nanotorch.detection.yolo_v11 import YOLOv11Loss

loss_fn = YOLOv11Loss(
    num_classes=80,
    lambda_box=7.5,
    lambda_cls=0.5,
    lambda_dfl=1.5
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
from nanotorch.optim import AdamW
from nanotorch.optim.lr_scheduler import CosineWarmupScheduler
from nanotorch.detection.yolo_v11 import build_yolov11, YOLOv11Loss
import numpy as np

# Create model
model = build_yolov11(num_classes=80, input_size=640)

# Loss function and optimizer
loss_fn = YOLOv11Loss(num_classes=80)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=3, max_epochs=50)

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
python examples/yolo_v11/demo.py --mode arch

# Train model
python examples/yolo_v11/demo.py --mode train --epochs 5

# Inference demo
python examples/yolo_v11/demo.py --mode inference

# Complete workflow
python examples/yolo_v11/demo.py --mode both
```

### Running Tests

```bash
# Unit tests
python -m pytest tests/detection/yolo_v11/test_yolov11_model.py -v

# Integration tests
python -m pytest tests/detection/yolo_v11/test_v11_integration.py -v

# All v11 tests
python -m pytest tests/detection/yolo_v11/ -v
```

---

## YOLO v10 vs YOLO v11 Comparison

| Feature | YOLO v10 | YOLO v11 |
|---------|----------|----------|
| Developer | Tsinghua University | Ultralytics |
| Core Module | C2fCIB | C3k2 |
| NMS | Not Required | Required |
| SPPF | No | Yes |
| Neck | None | PANet |
| Accuracy | High | Higher |

---

## Common Issues

### 1. What's the difference between C3k2 and C2f?

| Feature | C2f | C3k2 |
|---------|-----|------|
| Branch Count | 2 | 2 |
| Middle Channels | c_out/2 | min(c_in, c_out) |
| Bottleneck | Simple | More flexible |
| Efficiency | High | Higher |

### 2. Why does YOLO v11 need NMS?

YOLO v11 maintains traditional detection paradigm:
- May produce multiple overlapping prediction boxes
- Requires NMS for post-processing
- Similar workflow to YOLO v8

### 3. What is the purpose of SPPF?

- **Increased Receptive Field**: Captures more context information
- **Multi-scale Features**: Fuses features at different scales
- **Improved Accuracy**: Helps with large object detection

---

## Summary

This tutorial covered the complete process of implementing YOLO v11 using nanotorch:

1. **Core Improvements**: C3k2 module, SPPF, PANet Neck
2. **Model Architecture**: Backbone + Neck + DetectHead
3. **Loss Function**: DFL + BCE + CIoU
4. **Training & Inference**: Complete training and inference workflow

YOLO v11 is Ultralytics' latest masterpiece, achieving new heights in both accuracy and speed, making it one of the most advanced object detection models currently available.

---

## References

1. **Ultralytics YOLOv11**: https://github.com/ultralytics/ultralytics

2. **YOLOv11 Documentation**: https://docs.ultralytics.com/

3. **nanotorch YOLO v10 Tutorial**: `/docs/tutorials/26-yolov10_EN.md`

4. **nanotorch YOLO v8 Tutorial**: `/docs/tutorials/24-yolov8_EN.md`
