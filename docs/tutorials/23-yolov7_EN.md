# YOLO v7 Object Detection Model Implementation Tutorial

## The Return of the Creator...

After YOLO v3, Joseph Redmon left the field. The torch passed to others—Ultralytics, Meituan, research teams around the world.

Then in 2022, WongKinYiu—a researcher who had been quietly improving YOLO for years—released v7. It wasn't from Redmon, but it carried the same spirit: push the boundaries of what's possible.

**YOLO v7 asked: can we have accuracy without inference cost?**

```
The Trainable Bag-of-Freebies:

  The dilemma:
    Better accuracy → More computation
    Faster inference → Lower accuracy
    Can we escape this tradeoff?

  YOLO v7's insight:
    "What if we add complexity during training,
     but remove it during inference?"

  The techniques:
    - Auxiliary training heads → Extra supervision during training
    - E-ELAN architecture → Efficient feature aggregation
    - Conv reparameterization → Multi-branch training, single-branch inference

  The result:
    Training is more expensive
    But inference is fast
    And accuracy is state-of-the-art
```

**YOLO v7 is the art of having your cake and eating it too.** The auxiliary heads guide training but disappear at inference. The multi-branch architecture enables better gradients but fuses into simple convolutions. Every training technique that adds cost has a corresponding escape hatch for deployment.

In this tutorial, we'll implement YOLO v7 from scratch. We'll see how E-ELAN aggregates features efficiently, how auxiliary heads improve learning, and how the trainable bag-of-freebies philosophy maximizes accuracy at zero inference cost.

---

## Table of Contents

1. [Overview](#overview)
2. [YOLO v7 Core Improvements](#yolo-v7-core-improvements)
3. [Model Architecture](#model-architecture)
4. [Loss Function](#loss-function)
5. [Training Process](#training-process)
6. [Code Examples](#code-examples)
7. [Common Issues](#common-issues)
8. [Summary](#summary)

---

## Overview

YOLO v7 is developed by WongKinYiu, being an important milestone in the YOLO series. Through "Trainable Bag-of-Freebies" strategy, it significantly improves accuracy without increasing inference cost.

### Main Features of YOLO v7

1. **E-ELAN (Extended Efficient Layer Aggregation Network)**: Extended efficient layer aggregation network
2. **Model Scaling Techniques**: Compound scaling strategy, adjusting both depth and width
3. **Auxiliary Training Head**: Uses auxiliary head during training, removed during inference
4. **Convolution Reparameterization**: Multi-branch during training, fused to single branch during inference
5. **YOLOv7-W6/E6/D6**: Multiple size variants

### nanotorch YOLO v7 Implementation Modules

```
nanotorch/detection/yolo_v7/
├── __init__.py        # Module exports
├── yolo_v7_model.py   # Model architecture (ELAN, Backbone, Neck, Head, YOLOv7)
└── yolo_v7_loss.py    # Loss function (YOLOv7Loss, encode_targets_v7, decode_predictions_v7)

examples/yolo_v7/
└── demo.py            # Training and inference demo

tests/detection/yolo_v7/
├── test_yolov7_model.py  # Unit tests
└── test_v7_integration.py  # Integration tests
```

---

## YOLO v7 Core Improvements

### E-ELAN (Extended Efficient Layer Aggregation Network)

E-ELAN is YOLO v7's core building block, achieving stronger feature representation through efficient layer aggregation:

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

Advantages over regular ELAN:
- **Optimized Gradient Flow**: Better gradient propagation paths
- **Feature Reuse**: Intermediate features are effectively utilized
- **Parameter Efficiency**: Stronger expressive power with same parameters

### Auxiliary Training Head

YOLO v7 uses auxiliary detection heads during training:
- **Lead Head**: Main detection head for final predictions
- **Auxiliary Head**: Helps deep feature learning

```
Backbone Feature
       │
       ├──→ Auxiliary Head → Aux Loss
       │
       └──→ Neck → Lead Head → Main Loss
```

### Reparameterization Convolution (RepConv)

Different structures for training and inference:
- **Training**: 3×3 Conv + 1×1 Conv + Identity (three branches)
- **Inference**: Fused into single 3×3 Conv (structural reparameterization)

---

## Model Architecture

### 1. ConvBN (Basic Convolution Block)

ConvBN is YOLO v7's basic building unit:

```python
from nanotorch.detection.yolo_v7 import ConvBN
from nanotorch.tensor import Tensor
import numpy as np

conv_bn = ConvBN(64, 128, k=3, s=1)
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = conv_bn(x)
print(y.shape)  # (1, 128, 52, 52)
```

ConvBN contains three components:
1. **Conv2D**: 2D convolution layer
2. **BatchNorm2d**: Batch normalization layer
3. **SiLU**: Smooth activation function

### 2. ELAN Module

```python
from nanotorch.detection.yolo_v7 import ELAN

elan = ELAN(128, 256, n=2)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = elan(x)
print(y.shape)  # (1, 256, 52, 52)
```

ELAN parameters:
- `c_in`: Input channels
- `c_out`: Output channels
- `n`: Number of Bottleneck blocks

### 3. Backbone Network

```python
from nanotorch.detection.yolo_v7 import Backbone

backbone = Backbone(in_ch=3)
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
features = backbone(x)

print(features['scale1'].shape)  # (1, 512, 20, 20) - Large object features
print(features['scale2'].shape)  # (1, 256, 40, 40) - Medium object features
print(features['scale3'].shape)  # (1, 128, 80, 80) - Small object features
```

Backbone structure:
```
Input (3×640×640)
    ↓ Stem: Conv3×3(s2) → Conv3×3(s2)
    ↓ Stage1: ELAN(64→64)
    ↓ Down1: Conv3×3(s2)
    ↓ Stage2: ELAN(128→128) → Output s3 (stride=8)
    ↓ Down2: Conv3×3(s2)
    ↓ Stage3: ELAN(256→256) → Output s2 (stride=16)
    ↓ Down3: Conv3×3(s2)
    ↓ Stage4: ELAN(512→512) → Output s1 (stride=32)
```

### 4. Neck Network (PANet)

```python
from nanotorch.detection.yolo_v7 import Neck

neck = Neck(channels=[128, 256, 512])
# Use backbone output features
neck_out = neck(features)
print(neck_out['p3'].shape)  # (1, 128, 80, 80)
print(neck_out['p4'].shape)  # (1, 256, 40, 40)
print(neck_out['p5'].shape)  # (1, 512, 20, 20)
```

Neck uses PANet (Path Aggregation Network) structure:
- **Top-down Pathway**: Upsampling + fusion
- **Bottom-up Pathway**: Downsampling + fusion

### 5. Complete YOLOv7 Model

```python
from nanotorch.detection.yolo_v7 import YOLOv7, build_yolov7

# Method 1: Create directly
model = YOLOv7(num_classes=80, input_size=640)

# Method 2: Use factory function
model = build_yolov7(num_classes=80, input_size=640)

# Forward pass
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
output = model(x)

print(output['small'].shape)   # (1, 85, 20, 20) - Large object detection
print(output['medium'].shape)  # (1, 85, 40, 40) - Medium object detection
print(output['large'].shape)   # (1, 85, 80, 80) - Small object detection
```

Output format explanation:
- 85 = 4 (bbox: tx, ty, tw, th) + 1 (confidence) + 80 (classes)

---

## Loss Function

### YOLO v7 Loss Function Design

YOLO v7's loss function contains four parts:

$$
L = \lambda_{box} L_{box} + \lambda_{obj} L_{obj} + \lambda_{noobj} L_{noobj} + \lambda_{class} L_{class}
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

### Confidence Loss (BCE)

$$
L_{obj} = -\frac{1}{N_{pos}} \sum_{i \in pos} \left[ \hat{C}_i \log(C_i) + (1 - \hat{C}_i) \log(1 - C_i) \right]
$$

$$
L_{noobj} = -\frac{\lambda_{noobj}}{N_{neg}} \sum_{i \in neg} \left[ \hat{C}_i \log(C_i) + (1 - \hat{C}_i) \log(1 - C_i) \right]
$$

### Classification Loss (BCE)

$$
L_{class} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left[ y_{ic} \log(\hat{p}_{ic}) + (1 - y_{ic}) \log(1 - \hat{p}_{ic}) \right]
$$

### E-ELAN Architecture

E-ELAN (Extended Efficient Layer Aggregation Network) feature fusion:

$$
\text{Output} = \text{Concat}(\text{Branch}_1, \text{Branch}_2, \ldots, \text{Branch}_k)
$$

### RepConv Reparameterization

Training compound convolution:

$$
y = \text{Conv}_{3 \times 3}(x) + \text{Conv}_{1 \times 1}(x) + \text{BN}(x)
$$

Inference fused to single convolution:

$$
y = \text{Conv}_{fused}(x)
$$

### YOLOv7Loss

```python
from nanotorch.detection.yolo_v7 import YOLOv7Loss

loss_fn = YOLOv7Loss(
    num_classes=80,
    lambda_box=5.0,      # Bounding box loss weight
    lambda_obj=1.0,      # Object confidence loss weight
    lambda_noobj=0.5,    # No-object confidence loss weight
    lambda_class=1.0     # Classification loss weight
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
print(f"Obj Loss: {loss_dict['obj_loss']:.4f}")
print(f"NoObj Loss: {loss_dict['noobj_loss']:.4f}")
print(f"Class Loss: {loss_dict['class_loss']:.4f}")
```

### Encoding and Decoding Functions

```python
from nanotorch.detection.yolo_v7 import encode_targets_v7, decode_predictions_v7

# Encode targets
boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
labels = np.array([0], dtype=np.int64)
targets = encode_targets_v7(boxes, labels, grid_sizes=[80, 40, 20])

# Decode predictions
boxes, scores, class_ids = decode_predictions_v7(
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
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import CosineAnnealingLR
from nanotorch.detection.yolo_v7 import build_yolov7, YOLOv7Loss
import numpy as np

# Create model
model = build_yolov7(num_classes=80, input_size=224)

# Loss function and optimizer
loss_fn = YOLOv7Loss(num_classes=80)
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    
    for batch_idx in range(10):
        # Generate random data
        images = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        targets = [{
            'boxes': np.random.rand(2, 4).astype(np.float32) * 200,
            'labels': np.random.randint(0, 80, 2).astype(np.int64)
        }]
        
        optimizer.zero_grad()
        output = model(images)
        
        # Compute loss
        loss, loss_dict = loss_fn(output, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / 10
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
```

---

## Code Examples

### Running Demo

```bash
# View model architecture
python examples/yolo_v7/demo.py --mode arch

# Train model
python examples/yolo_v7/demo.py --mode train --epochs 5

# Inference demo
python examples/yolo_v7/demo.py --mode inference

# Complete workflow (training + inference)
python examples/yolo_v7/demo.py --mode both
```

### Demo Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--mode` | arch | Run mode (arch/train/inference/both) |
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 2 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--image-size` | 224 | Input image size |
| `--num-classes` | 80 | Number of classes |

### Running Tests

```bash
# Unit tests
python -m pytest tests/detection/yolo_v7/test_yolov7_model.py -v

# Integration tests
python -m pytest tests/detection/yolo_v7/test_v7_integration.py -v

# All v7 tests
python -m pytest tests/detection/yolo_v7/ -v
```

---

## YOLO v6 vs YOLO v7 Comparison

| Feature | YOLO v6 | YOLO v7 |
|---------|---------|---------|
| Developer | Meituan | WongKinYiu |
| Core Module | RepVGG Block | E-ELAN |
| Training Strategy | Standard | Auxiliary Training Head |
| Model Scaling | Fixed | Compound Scaling |
| Reparameterization | Backbone | Optional |

---

## Common Issues

### 1. What's the difference between E-ELAN and C3 modules?

| Feature | C3 (YOLOv5) | E-ELAN (YOLOv7) |
|---------|-------------|-----------------|
| Structure | CSP + Bottleneck | Dual-branch + Multi-level aggregation |
| Gradient Flow | General | Better |
| Parameter Efficiency | Medium | Higher |

### 2. How does auxiliary training head work?

Auxiliary head provides additional supervision signals during training:
- Helps shallow feature learning
- Does not affect inference speed (removed during inference)
- Improves accuracy by ~1-2%

### 3. How to choose model size?

- **YOLOv7-Tiny**: Edge devices, extreme speed
- **YOLOv7**: Standard version, balanced speed and accuracy
- **YOLOv7-W6**: Larger model, higher accuracy
- **YOLOv7-E6/D6**: Highest accuracy requirements

---

## Summary

This tutorial covered the complete process of implementing YOLO v7 using nanotorch:

1. **Core Improvements**: E-ELAN, auxiliary training head, model scaling
2. **Model Architecture**: Backbone + Neck (PANet) + Head
3. **Loss Function**: Bounding box loss + confidence loss + classification loss
4. **Training & Inference**: Complete training and inference workflow

YOLO v7 achieves SOTA performance without increasing inference cost through "freebies" strategy, making it an excellent choice for industrial deployment.

---

## References

1. **YOLOv7 Paper**: "YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors"

2. **GitHub Repository**: https://github.com/WongKinYiu/yolov7

3. **nanotorch YOLO v6 Tutorial**: `/docs/tutorials/22-yolov6_EN.md`

4. **nanotorch YOLO v8 Tutorial**: `/docs/tutorials/24-yolov8_EN.md`
