# YOLO v10 Object Detection Model Implementation Tutorial

## The End of NMS...

For a decade, object detection had a dirty secret.

The model would output hundreds of predictions. Many would overlap—multiple boxes around the same object. We needed to clean them up. So we added Non-Maximum Suppression (NMS): a post-processing step that wasn't part of the model, wasn't learned, but was absolutely necessary.

**YOLO v10 asked: what if we didn't need NMS at all?**

```
The NMS Problem:

  Traditional detection:
    Model outputs predictions → Many overlapping boxes
    NMS post-processing → Keep best, discard rest
    → Not end-to-end
    → Not differentiable
    → Extra hyperparameters (IoU threshold)
    → Can fail with crowded objects

  YOLO v10's insight:
    "NMS exists because training allows duplicate predictions.
     Change the training, and we don't need NMS at inference."

  The solution:
    Consistent Dual Assignments
    → One-to-many during training (like before)
    → One-to-one during inference (no duplicates needed)
    → Same predictions, no post-processing

  The result:
    True end-to-end detection.
    No NMS. No thresholds. No post-processing.
```

**YOLO v10 is YOLO finally becoming truly end-to-end.** It's not just about removing a step—it's about recognizing that post-processing is a crutch, a sign that the model wasn't trained to do exactly what we want it to do. Fix the training, and inference becomes cleaner.

In this tutorial, we'll implement YOLO v10 from scratch. We'll see how consistent dual assignments work, how to train without NMS dependency, and what it means for object detection to be truly end-to-end.

---

## Table of Contents

1. [Overview](#overview)
2. [YOLO v10 Core Improvements](#yolo-v10-core-improvements)
3. [Model Architecture](#model-architecture)
4. [Loss Function](#loss-function)
5. [Training Process](#training-process)
6. [Code Examples](#code-examples)
7. [Common Issues](#common-issues)
8. [Summary](#summary)

---

## Overview

YOLO v10 is developed by Tsinghua University and released in 2024. Its biggest innovation is eliminating Non-Maximum Suppression (NMS) post-processing, achieving true end-to-end detection.

### Main Features of YOLO v10

1. **NMS-free Inference**: No post-processing needed, end-to-end detection
2. **Consistent Dual Assignments**: Same assignment strategy for training and inference
3. **SCDown**: Spatial-Channel Downsampling
4. **C2fCIB**: Concat-based Inverted Bottleneck block
5. **Efficiency-Accuracy Driven**: Optimized model design

### nanotorch YOLO v10 Implementation Modules

```
nanotorch/detection/yolo_v10/
├── __init__.py        # Module exports
├── yolo_v10_model.py  # Model architecture (SCDown, C2fCIB, Backbone, Head, YOLOv10)
└── yolo_v10_loss.py   # Loss function (YOLOv10Loss, encode_targets_v10, decode_predictions_v10)

examples/yolo_v10/
└── demo.py            # Training and inference demo

tests/detection/yolo_v10/
├── test_yolov10_model.py  # Unit tests
└── test_v10_integration.py  # Integration tests
```

---

## YOLO v10 Core Improvements

### NMS-free Detection

Traditional YOLO requires NMS post-processing:
```
Prediction Boxes → Confidence Filtering → NMS → Final Results
```

YOLO v10 achieves end-to-end detection:
```
Prediction Boxes → Direct Output → Final Results (No NMS needed)
```

Advantages:
- **Faster Inference**: No post-processing step
- **Simpler Deployment**: End-to-end pipeline
- **Consistent Performance**: Training and inference are consistent

### Consistent Dual Assignments

YOLO v10 uses consistent one-to-many assignment strategy:

| Stage | Traditional YOLO | YOLO v10 |
|-------|------------------|----------|
| Training | One-to-many assignment | Consistent one-to-many |
| Inference | One-to-one (requires NMS) | Consistent one-to-many |
| Consistency | Inconsistent | Fully consistent |

### SCDown (Spatial-Channel Downsampling)

Efficient downsampling module:

```
Input (c_in)
    │
    ↓ Conv3×3 (groups=c_in, stride=2) → Spatial downsampling
    ↓ Conv1×1 → Channel transformation
    ↓
Output (c_out)
```

Advantages:
- **Less Computation**: Separates spatial and channel operations
- **Better Features**: Preserves more information

### C2fCIB Module

```
Input (c_in)
    │
    └──→ Conv1×1 → c_mid ──┬──→ Conv3×3 ──┐
                             │              │
                             ├──→ Conv3×3 ──┤
                             │              │
                             └──→ Conv3×3 ──┤
                                            │
               Concat(cv1, block1, ..., block_n)
                               │
                          Conv1×1 → Output (c_out)
```

---

## Model Architecture

### 1. ConvBN (Basic Convolution Block)

```python
from nanotorch.detection.yolo_v10 import ConvBN
from nanotorch.tensor import Tensor
import numpy as np

conv_bn = ConvBN(64, 128, k=3, s=1)
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = conv_bn(x)
print(y.shape)  # (1, 128, 52, 52)
```

### 2. SCDown (Spatial-Channel Downsampling)

```python
from nanotorch.detection.yolo_v10 import SCDown

sc_down = SCDown(64, 128, k=3, s=2)
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = sc_down(x)
print(y.shape)  # (1, 128, 26, 26)
```

### 3. C2fCIB Module

```python
from nanotorch.detection.yolo_v10 import C2fCIB

c2fcib = C2fCIB(128, 256, n=2)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = c2fcib(x)
print(y.shape)  # (1, 256, 52, 52)
```

### 4. Backbone Network

```python
from nanotorch.detection.yolo_v10 import Backbone

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
    ↓ Stage1: SCDown → C2fCIB → 64 channels
    ↓ Stage2: SCDown → C2fCIB → 128 channels → Output s3 (stride=8)
    ↓ Stage3: SCDown → C2fCIB → 256 channels → Output s2 (stride=16)
    ↓ Stage4: SCDown → C2fCIB → 512 channels → Output s1 (stride=32)
```

### 5. Complete YOLOv10 Model

```python
from nanotorch.detection.yolo_v10 import YOLOv10, build_yolov10

# Method 1: Create directly
model = YOLOv10(num_classes=80, input_size=640)

# Method 2: Use factory function
model = build_yolov10(num_classes=80, input_size=640)

# Forward pass
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
output = model(x)

print(output['small'].shape)   # (1, 85, 20, 20) - Large object detection
print(output['medium'].shape)  # (1, 85, 40, 40) - Medium object detection
print(output['large'].shape)   # (1, 85, 80, 80) - Small object detection
```

---

## Loss Function

### YOLO v10 Loss Function Design

YOLO v10 uses consistent dual assignment loss:

$$
L = \lambda_{box} L_{box} + \lambda_{cls} L_{cls} + \lambda_{obj} L_{obj}
$$

### Consistent Dual Assignment

YOLO v10 uses consistent assignment strategy during training and inference:

**One-to-Many Assignment (Training)**:

$$
L_{train} = \sum_{i=1}^{N} \sum_{j=1}^{M} \mathbb{1}_{ij} \cdot L_{task}(pred_i, gt_j)
$$

**One-to-One Assignment (Inference)**:

$$
L_{inference} = \sum_{i=1}^{N} \mathbb{1}_{i \to j^*} \cdot L_{task}(pred_i, gt_{j^*})
$$

Where $j^* = \arg\max_j \text{IoU}(pred_i, gt_j)$.

### Bounding Box Loss

$$
L_{box} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$

### NMS-free Design

YOLO v10 eliminates NMS through consistent training:

$$
\text{Score} = \text{Confidence} \times \text{ClassProb}
$$

Direct output, no post-processing suppression needed.

### YOLOv10Loss

```python
from nanotorch.detection.yolo_v10 import YOLOv10Loss

loss_fn = YOLOv10Loss(
    num_classes=80,
    lambda_box=5.0,
    lambda_cls=1.0,
    lambda_obj=1.0
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
from nanotorch.detection.yolo_v10 import build_yolov10, YOLOv10Loss
import numpy as np

# Create model
model = build_yolov10(num_classes=80, input_size=640)

# Loss function and optimizer
loss_fn = YOLOv10Loss(num_classes=80)
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
python examples/yolo_v10/demo.py --mode arch

# Train model
python examples/yolo_v10/demo.py --mode train --epochs 5

# Inference demo
python examples/yolo_v10/demo.py --mode inference

# Complete workflow
python examples/yolo_v10/demo.py --mode both
```

### Running Tests

```bash
# Unit tests
python -m pytest tests/detection/yolo_v10/test_yolov10_model.py -v

# Integration tests
python -m pytest tests/detection/yolo_v10/test_v10_integration.py -v

# All v10 tests
python -m pytest tests/detection/yolo_v10/ -v
```

---

## YOLO v9 vs YOLO v10 Comparison

| Feature | YOLO v9 | YOLO v10 |
|---------|---------|----------|
| Developer | WongKinYiu | Tsinghua University |
| NMS | Required | Not Required |
| Core Module | GELAN | C2fCIB |
| Downsampling | Conv | SCDown |
| Inference Speed | Medium | Faster |

---

## Common Issues

### 1. Why doesn't YOLO v10 need NMS?

YOLO v10 uses consistent dual assignment strategy:
- During training: Each target matches multiple prediction boxes
- During inference: Directly outputs highest confidence prediction
- No post-processing needed to remove duplicate boxes

### 2. Advantages of SCDown?

| Feature | Regular Conv Downsampling | SCDown |
|---------|---------------------------|--------|
| Computation | High | Low |
| Parameters | More | Fewer |
| Feature Quality | Medium | Better |

### 3. Difference between C2fCIB and C2f?

| Feature | C2f | C2fCIB |
|---------|-----|--------|
| Connection Method | Simple concat | Sequential concat |
| Feature Reuse | Limited | Full |
| Parameter Efficiency | Medium | Higher |

---

## Summary

This tutorial covered the complete process of implementing YOLO v10 using nanotorch:

1. **Core Improvements**: NMS-free, consistent dual assignments, SCDown, C2fCIB
2. **Model Architecture**: Backbone + Head
3. **Loss Function**: Consistent dual assignment loss
4. **Training & Inference**: Complete training and inference workflow

YOLO v10 is the first YOLO version to achieve true end-to-end detection, eliminating NMS post-processing, achieving excellent balance in both speed and accuracy.

---

## References

1. **YOLOv10 Paper**: "YOLOv10: Real-Time End-to-End Object Detection"

2. **GitHub Repository**: https://github.com/THU-MIG/yolov10

3. **nanotorch YOLO v9 Tutorial**: `/docs/tutorials/25-yolov9_EN.md`

4. **nanotorch YOLO v11 Tutorial**: `/docs/tutorials/27-yolov11_EN.md`
