# YOLO v5 Object Detection Model Implementation Tutorial

## Engineering Excellence...

YOLO v5 is different.

It wasn't a research paper. It wasn't an academic contribution. It was built by a company—Ultralytics—with a different goal: make YOLO easy to use, easy to deploy, and production-ready.

**Sometimes the best research isn't about new ideas—it's about making existing ideas work beautifully.**

```
The YOLO v5 Philosophy:

  Research-focused (YOLO v1-v4):
    "Here's our novel contribution"
    "Here's our experimental results"
    "Here's the code (good luck running it)"

  Production-focused (YOLO v5):
    "Here's a pip-installable package"
    "Here's auto-anchor computation"
    "Here's model export to ONNX, TensorRT, CoreML"
    "Here's five sizes: nano, small, medium, large, xlarge"

  The same architecture, but engineered for impact:
    C3 modules instead of CSP blocks
    SPPF for faster spatial pooling
    SiLU activation throughout
    Mosaic augmentation by default
```

**YOLO v5 is YOLO for everyone.** You don't need to be a researcher to use it. You don't need to understand every architectural choice. You pip install, you train, you deploy. That simplicity—and the engineering behind it—is its own kind of innovation.

In this tutorial, we'll implement YOLO v5 from scratch. We'll see how the C3 module improves on CSP, how SPPF speeds up spatial pooling, and how the different model sizes (n, s, m, l, x) trade off accuracy and speed for different applications.

---

## Table of Contents

1. [Overview](#overview)
2. [YOLO v5 Core Improvements](#yolo-v5-core-improvements)
3. [Model Architecture](#model-architecture)
4. [Loss Function](#loss-function)
5. [Training Process](#training-process)
6. [Code Examples](#code-examples)

---

## Overview

YOLO v5 is released by Ultralytics, known for its ease of use and engineering optimizations.

### Main Features of YOLO v5

1. **C3 Module**: CSP Bottleneck with 3 convolutions
2. **SPPF**: Fast Spatial Pyramid Pooling
3. **SiLU Activation**: Smooth activation function
4. **Multi-size Models**: n(nano), s(small), m(medium), l(large), x(xlarge)
5. **PANet Neck**: Path Aggregation Network
6. **Auto Anchor**: Automatic anchor calculation

### nanotorch YOLO v5 Implementation Modules

```
nanotorch/detection/yolo_v5/
├── __init__.py        # Module exports
├── yolo_v5_model.py   # Model architecture (C3, SPPF, Backbone, Neck, YOLOv5)
└── yolo_v5_loss.py    # Loss function (YOLOv5Loss, encode_targets_v5, decode_predictions_v5)

examples/yolo_v5/
└── demo.py            # Training and inference demo

tests/detection/yolo_v5/
├── test_yolov5_model.py  # Unit tests
└── test_integration.py   # Integration tests
```

---

## YOLO v5 Core Improvements

### C3 Module (CSP Bottleneck with 3 convolutions)

C3 is YOLOv5's core building block:

```
Input Feature
    ├─── Conv1x1 → mid_channels → Bottleneck×N →┐
    │                                            ├→ Concat → Conv1x1 → Output
    └─── Conv1x1 → mid_channels ────────────────┘
```

### SPPF (Spatial Pyramid Pooling Fast)

SPPF achieves multi-scale features through consecutive pooling:

```
Input Feature
    ↓
Conv 1x1
    ↓
┌─────────────────────────────────┐
│  y1 = Original feature          │
│  y2 = MaxPool(y1)              │
│  y3 = MaxPool(y2)              │
│  y4 = MaxPool(y3)              │
└─────────────────────────────────┘
    ↓
Concat(y1, y2, y3, y4)
    ↓
Conv 1x1
```

### Multi-size Models

YOLOv5 offers five sizes:

| Version | depth_multiple | width_multiple | Parameters |
|---------|----------------|----------------|------------|
| Nano (n) | 0.33 | 0.25 | ~1.9M |
| Small (s) | 0.33 | 0.50 | ~7.2M |
| Medium (m) | 0.67 | 0.75 | ~21M |
| Large (l) | 1.00 | 1.00 | ~46M |
| XLarge (x) | 1.33 | 1.25 | ~86M |

---

## Model Architecture

### 1. ConvBN (Basic Convolution Block)

```python
from nanotorch.detection.yolo_v5 import ConvBN
from nanotorch.tensor import Tensor
import numpy as np

conv_bn = ConvBN(64, 128, kernel_size=3, stride=1)
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = conv_bn(x)
print(y.shape)  # (1, 128, 52, 52)
```

### 2. C3 Module

```python
from nanotorch.detection.yolo_v5 import C3

c3 = C3(128, 256, num_blocks=3)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = c3(x)
print(y.shape)  # (1, 256, 52, 52)
```

### 3. SPPF Module

```python
from nanotorch.detection.yolo_v5 import SPPF

sppf = SPPF(512, 512, kernel_size=5)
x = Tensor(np.random.randn(1, 512, 13, 13).astype(np.float32))
y = sppf(x)
print(y.shape)  # (1, 512, 13, 13)
```

### 4. Complete YOLOv5 Model

```python
from nanotorch.detection.yolo_v5 import YOLOv5, YOLOv5Nano, YOLOv5Small, build_yolov5

# Method 1: Create specific version
model = YOLOv5Nano(num_classes=80, input_size=640)
model = YOLOv5Small(num_classes=80, input_size=640)

# Method 2: Specify version
model = YOLOv5(num_classes=80, input_size=640, version='s')

# Method 3: Use factory function
model = build_yolov5('s', num_classes=80, input_size=640)

# Forward pass
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
output = model(x)

print(output['small'].shape)   # Large object detection
print(output['medium'].shape)  # Medium object detection
print(output['large'].shape)   # Small object detection
```

---

## Loss Function

### YOLO v5 Loss Function Design

YOLO v5 uses a loss function combination similar to YOLO v4:

$$
L = \lambda_{box} L_{CIoU} + \lambda_{obj} L_{obj} + \lambda_{noobj} L_{noobj} + \lambda_{class} L_{class}
$$

### Bounding Box Regression Loss (CIoU)

$$
L_{box} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$

### Object Confidence Loss

$$
L_{obj} = -\frac{1}{N_{pos}} \sum_{i \in pos} \left[ \hat{C}_i \log(C_i) + (1 - \hat{C}_i) \log(1 - C_i) \right]
$$

$$
L_{noobj} = -\frac{1}{N_{neg}} \sum_{i \in neg} \left[ \hat{C}_i \log(C_i) + (1 - \hat{C}_i) \log(1 - C_i) \right]
$$

### Classification Loss (BCE)

$$
L_{class} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left[ y_{ic} \log(\hat{p}_{ic}) + (1 - y_{ic}) \log(1 - \hat{p}_{ic}) \right]
$$

### AutoAnchor Automatic Anchor Calculation

YOLO v5 automatically computes optimal anchors:

$$
\text{Anchor}_{opt} = \arg\min_{k} \sum_{i=1}^{N} \min_{j=1}^{K} (1 - \text{IoU}(box_i, anchor_j))
$$

### Mosaic Data Augmentation

Mosaic stitches 4 images together:

$$
I_{out} = \text{Concat}(I_1[0:H/2, 0:W/2], I_2[0:H/2, W/2:W], I_3[H/2:H, 0:W/2], I_4[H/2:H, W/2:W])
$$

### Using the Loss Function

```python
from nanotorch.detection.yolo_v5 import YOLOv5Loss, YOLOv5LossSimple

# Create loss function
loss_fn = YOLOv5Loss(
    num_classes=80,
    lambda_box=5.0,
    lambda_obj=1.0,
    lambda_noobj=0.5,
    lambda_class=1.0
)

# Compute loss
predictions = {
    'small': Tensor(np.random.randn(2, 255, 20, 20).astype(np.float32) * 0.1),
    'medium': Tensor(np.random.randn(2, 255, 40, 40).astype(np.float32) * 0.1),
    'large': Tensor(np.random.randn(2, 255, 80, 80).astype(np.float32) * 0.1)
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

```python
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v5 import build_yolov5
import numpy as np

# Create model
model = build_yolov5('n', num_classes=80, input_size=224)

# Optimizer and scheduler
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    
    for _ in range(10):
        images = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        
        optimizer.zero_grad()
        output = model(images)
        
        # MSE loss
        loss = 0.0
        for pred in output.values():
            diff = pred - Tensor(np.zeros_like(pred.data))
            loss += (diff * diff).mean().item()
        
        loss_tensor = Tensor(loss, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()
        
        total_loss += loss
    
    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss/10:.4f}")
```

---

## Code Examples

### Running Demo

```bash
# View architecture
python examples/yolo_v5/demo.py --mode arch

# Training (using nano version)
python examples/yolo_v5/demo.py --mode train --version n

# Training (using small version)
python examples/yolo_v5/demo.py --mode train --version s --epochs 5

# Inference
python examples/yolo_v5/demo.py --mode inference

# Complete workflow
python examples/yolo_v5/demo.py --mode both
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 2 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--image-size` | 224 | Input size |
| `--version` | n | Model version (n/s/m/l/x) |
| `--num-classes` | 80 | Number of classes |

### Running Tests

```bash
# Unit tests
python -m pytest tests/detection/yolo_v5/test_yolov5_model.py -v

# Integration tests
python -m pytest tests/detection/yolo_v5/test_integration.py -v
```

---

## YOLO v4 vs YOLO v5 Comparison

| Feature | YOLO v4 | YOLO v5 |
|---------|---------|---------|
| Developer | Alexey Bochkovskiy | Ultralytics |
| Activation | Mish | SiLU |
| Pooling Module | SPP | SPPF |
| Bottleneck | CSPResBlock | C3 |
| Model Sizes | Fixed | Multi-size (n/s/m/l/x) |
| Ease of Use | Medium | High |

---

## Common Issues

### 1. How to choose model version?

- **Nano (n)**: Edge devices, high real-time requirements
- **Small (s)**: Balance of speed and accuracy
- **Medium (m)**: General application scenarios
- **Large (l)**: Accuracy priority
- **XLarge (x)**: Highest accuracy

### 2. Advantages of SPPF over SPP?

- Less computation
- Same receptive field
- Faster inference

### 3. Why use SiLU activation?

- Smooth nonlinearity
- Unbounded above, bounded below
- Better gradient flow than ReLU

---

## Summary

This tutorial covered the complete process of implementing YOLO v5 using nanotorch:

1. **Core Improvements**: C3 module, SPPF, SiLU activation
2. **Model Architecture**: Multi-size model design
3. **Loss Function**: Bounding box loss + confidence loss + classification loss
4. **Training & Inference**: Complete training and inference workflow

---

## References

1. **Ultralytics YOLOv5**: https://github.com/ultralytics/yolov5

2. **nanotorch YOLO v4 Tutorial**: `/docs/tutorials/20-yolov4_EN.md`

3. **nanotorch YOLO v3 Tutorial**: `/docs/tutorials/19-yolov3_EN.md`
