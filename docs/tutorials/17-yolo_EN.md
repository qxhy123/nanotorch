# YOLO v12 Object Detection Model Implementation Tutorial

## Imagine you're a painter...

Your friend hands you a photo: "Help me find what's in the painting."

You glance at it: "An orange cat lying on the sofa in the bottom left corner, a vase by the window with three sunflowers, and in the distance... is that the Eiffel Tower?"

Your friend is amazed: "How did you do that so fast?"

You smile: "I looked once, and I knew where everything was."

This is the miracle of human vision—one glance, and you see it all.

```
The clumsiness of traditional methods:
  First, scan every corner with a magnifying glass
  Mark all "suspicious areas"
  Then identify each one: "What is this?"
  Like an old pedant, slow and methodical

The elegance of YOLO:
  One glance, everything in its place
  Position and category, revealed simultaneously
  Like a seasoned connoisseur
  "This is a cat, that is a flower"
```

**YOLO - You Only Look Once**, just one glance, and all is known.

---

## Table of Contents

1. [Overview](#overview)
2. [YOLO v12 Architecture Innovations](#yolo-v12-architecture-innovations)
3. [Core Component Implementation](#core-component-implementation)
4. [Complete Model Assembly](#complete-model-assembly)
5. [Training Process](#training-process)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Code Examples](#code-examples)

---

## Overview

YOLO (You Only Look Once) is a landmark work in object detection. Version 12 introduces an attention-centric architecture design, significantly improving detection accuracy.

### nanotorch YOLO Implementation Modules

```
nanotorch/detection/
├── bbox.py           # Bounding box utilities
├── iou.py            # IoU variants (IoU, GIoU, DIoU, CIoU, SIoU)
├── nms.py            # Non-Maximum Suppression
├── layers.py         # YOLO base modules (Conv, C2f, SPPF, Attention)
├── yolo_backbone.py  # R-ELAN backbone network
├── yolo_neck.py      # PANet/FPN feature fusion network
├── yolo_head.py      # Anchor-free detection head
└── losses.py         # Loss functions (CIoU Loss, DFL Loss, VFL Loss)
```

---

## YOLO v12 Architecture Innovations

### 1. Attention-Centric Architecture

YOLO v12 abandons the pure CNN architecture of traditional YOLO, introducing **Area Attention (A²)** modules:

```
Traditional CNN:  Local Receptive Field → Local features
Area Attention:   Global Receptive Field → Global context
```

The core idea of Area Attention is to partition the feature map into multiple regions and compute self-attention within each region:

```
Feature map (H, W) → Partition into 4 regions (H/2, W/2) → Compute Self-Attention within each region
```

Complexity reduced from $O((HW)^2)$ to $O(4 \times (HW/4)^2) = O((HW)^2/4)$

### 2. R-ELAN (Residual Efficient Layer Aggregation Network)

R-ELAN is the core module of YOLO v12's backbone network with the following features:

- **Efficient Feature Aggregation**: Aggregates features from different levels through multiple paths
- **Residual Connection**: Adds residual connections at block level with scaling factor 0.1
- **Simplified Structure**: More concise and efficient than ELAN

```
Input → Conv 1x1 → Split → [Conv 3x3] × n → Concat → Conv 1x1 → + Input × 0.1
```

### 3. Anchor-Free Detection Head

YOLO v12 adopts an anchor-free design:

```
Traditional Anchor-Based:
  Predefined anchors → Predict offsets → Decode to detection boxes

Anchor-Free:
  Directly predict distances from grid center to four boundaries → Decode to detection boxes
```

Advantages:
- Reduced hyperparameters (no need to design anchors)
- Better generalization capability
- Simplified training process

---

## Core Component Implementation

### Bounding Box Format Conversion

```python
from nanotorch.detection.bbox import xyxy_to_xywh, xywh_to_xyxy

# [x1, y1, x2, y2] → [cx, cy, w, h]
boxes_xyxy = np.array([[10, 20, 50, 60]], dtype=np.float32)
boxes_xywh = xyxy_to_xywh(boxes_xyxy)
# Result: [[30, 40, 40, 40]] (center point + width/height)
```

### IoU Variants

```python
from nanotorch.detection.iou import iou, giou, diou, ciou

boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
boxes2 = np.array([[5, 5, 15, 15]], dtype=np.float32)

# Standard IoU
iou_value = iou(boxes1, boxes2)  # 0.14

# GIoU (handles non-overlapping cases)
giou_value = giou(boxes1, boxes2)  # -0.19 ~ 1.0

# DIoU (considers center distance)
diou_value = diou(boxes1, boxes2)

# CIoU (adds aspect ratio consistency)
ciou_value = ciou(boxes1, boxes2)
```

### CIoU Loss Mathematical Formula

$$
\text{CIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2} - \alpha v
$$

Where:
- $\rho^2$: Squared Euclidean distance between predicted and ground truth box centers
- $c^2$: Squared diagonal length of the smallest enclosing box
- $v = \frac{4}{\pi^2}\left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2$
- $\alpha = \frac{v}{1-\text{IoU}+v}$

### IoU Calculation Formula

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|A \cap B|}{|A \cup B|}
$$

For two bounding boxes $A = (x_1^A, y_1^A, x_2^A, y_2^A)$ and $B = (x_1^B, y_1^B, x_2^B, y_2^B)$:

$$
\text{IoU}(A, B) = \frac{\max(0, x_2^{\text{int}} - x_1^{\text{int}}) \times \max(0, y_2^{\text{int}} - y_1^{\text{int}})}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}
$$

Intersection coordinates:
- $x_1^{\text{int}} = \max(x_1^A, x_1^B)$
- $y_1^{\text{int}} = \max(y_1^A, y_1^B)$
- $x_2^{\text{int}} = \min(x_2^A, x_2^B)$
- $y_2^{\text{int}} = \min(y_2^A, y_2^B)$

### Coordinate Encoding Formula

YOLO uses relative coordinates instead of absolute coordinates:

$$
b_x = \sigma(t_x) + c_x
$$

$$
b_y = \sigma(t_y) + c_y
$$

$$
b_w = p_w \cdot e^{t_w}
$$

$$
b_h = p_h \cdot e^{t_h}
$$

Where:
- $(b_x, b_y, b_w, b_h)$: Predicted box center coordinates and width/height
- $(t_x, t_y, t_w, t_h)$: Network outputs
- $(c_x, c_y)$: Current grid cell's top-left coordinates
- $(p_w, p_h)$: Anchor width and height
- $\sigma$: Sigmoid function

### Confidence Definition

$$
\text{Confidence} = \Pr(\text{Object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}
$$

### NMS (Non-Maximum Suppression)

```python
from nanotorch.detection.nms import nms, batched_nms

boxes = np.array([
    [10, 10, 50, 50],
    [12, 12, 52, 52],  # Highly overlaps with first box
    [100, 100, 150, 150]
], dtype=np.float32)

scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)

# Standard NMS
keep = nms(boxes, scores, iou_threshold=0.5)
# Result: [0, 2] - Kept highest scoring overlapping box

# Class-wise NMS (boxes of different classes don't suppress each other)
class_ids = np.array([0, 0, 1], dtype=np.int64)
keep = batched_nms(boxes, scores, class_ids, iou_threshold=0.5)
```

### YOLO Base Modules

#### Conv Module

YOLO's basic convolution unit: Conv2D → BatchNorm → SiLU

```python
from nanotorch.detection.layers import Conv

conv = Conv(64, 128, kernel_size=3, stride=2)
x = Tensor(np.random.randn(1, 64, 32, 32).astype(np.float32))
y = conv(x)
# Output: (1, 128, 16, 16) - stride=2 halves spatial dimensions
```

#### C2f Module (CSP Bottleneck)

```python
from nanotorch.detection.layers import C2f

c2f = C2f(128, 128, num_bottlenecks=3)
x = Tensor(np.random.randn(1, 128, 16, 16).astype(np.float32))
y = c2f(x)
# Output: (1, 128, 16, 16)
```

C2f structure:
```
Input → Conv 1x1 → Split → [Bottleneck × n] → Concat → Conv 1x1 → Output
                      ↓              ↑
                      └──────────────┘
```

#### SPPF (Spatial Pyramid Pooling - Fast)

```python
from nanotorch.detection.layers import SPPF

sppf = SPPF(256, 256, kernel_size=5)
x = Tensor(np.random.randn(1, 256, 8, 8).astype(np.float32))
y = sppf(x)
# Output: (1, 256, 8, 8) - Spatial size unchanged, increased receptive field
```

SPPF is equivalent to SPP(k=5, 9, 13), but uses 3 consecutive MaxPool(k=5), more efficient.

---

## Complete Model Assembly

### Backbone

```python
from nanotorch.detection import build_backbone

# Build backbones of different scales
backbone_n = build_backbone(model_size='n')  # Nano - smallest
backbone_s = build_backbone(model_size='s')  # Small
backbone_m = build_backbone(model_size='m')  # Medium
backbone_l = build_backbone(model_size='l')  # Large
backbone_x = build_backbone(model_size='x')  # Extra Large

# Forward pass
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
features = backbone_n(x)

# Output feature maps at three scales
print(features['p3'].shape)  # (1, C3, 80, 80)  - stride 8
print(features['p4'].shape)  # (1, C4, 40, 40)  - stride 16
print(features['p5'].shape)  # (1, C5, 20, 20)  - stride 32
```

### Neck (Feature Fusion Network)

```python
from nanotorch.detection import build_neck

# PANet (Path Aggregation Network)
neck = build_neck(
    neck_type='panet',
    in_channels=backbone.out_channels,
    num_blocks=3
)

# Forward pass
fused_features = neck(features)
```

PANet structure:
```
P5 ─────────────────→ Conv → Upsample →┐
                                         ↓ Concat → C2f → P4'
P4 ────────────────────────────────────→┘
                                         ↓ Conv → Upsample →┐
                                                             ↓ Concat → C2f → P3'
P3 ────────────────────────────────────────────────────────→┘
                                                             ↓
                        P3' → Downsample → Concat ← P4' ←───┘
                                         ↓ C2f → P4''
                        P4'' → Downsample → Concat ← P5
                                         ↓ C2f → P5''
```

### Detection Head

```python
from nanotorch.detection import build_head

head = build_head(
    head_type='decoupled',
    in_channels=neck.out_channels,
    num_classes=80
)

# Forward pass
predictions = head(fused_features)

# Predictions at each scale
for scale_name, (box_pred, cls_pred) in predictions.items():
    print(f"{scale_name}: box={box_pred.shape}, cls={cls_pred.shape}")
```

---

## Training Process

### Loss Function

YOLO v12 uses a composite loss:

```python
from nanotorch.detection.losses import YOLOLoss

loss_fn = YOLOLoss(
    num_classes=80,
    reg_max=16,
    box_weight=7.5,   # Bounding box loss weight
    cls_weight=0.5,   # Classification loss weight
    dfl_weight=1.5    # DFL loss weight
)

total_loss, loss_dict = loss_fn(predictions, targets, image_size)
```

Composite loss formula:
$$
L_{total} = \lambda_{box} \cdot L_{box} + \lambda_{cls} \cdot L_{cls} + \lambda_{dfl} \cdot L_{dfl}
$$

### Data Augmentation

```python
from examples.yolo.data import (
    SyntheticDetectionDataset,
    MosaicAugmentation,
    RandomHorizontalFlip,
    LetterboxResize,
    create_yolo_dataloader
)

# Create synthetic dataset (for testing)
dataset = SyntheticDetectionDataset(
    num_samples=1000,
    image_size=(640, 640),
    num_classes=10
)

# Create DataLoader
dataloader = create_yolo_dataloader(
    dataset,
    batch_size=16,
    image_size=640
)

# Data augmentation
augmentations = [
    RandomHorizontalFlip(p=0.5),
    LetterboxResize(target_size=640)
]
```

### Training Script

```python
from examples.yolo.train import YOLOModel, Trainer

# Create model
model = YOLOModel(
    num_classes=10,
    model_size='s',
    use_attention=True
)

# Create trainer
trainer = Trainer(
    model=model,
    train_loader=dataloader,
    num_classes=10,
    lr=1e-3
)

# Training
for epoch in range(num_epochs):
    avg_loss = trainer.train_epoch(epoch)
    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
```

---

## Evaluation Metrics

### mAP (mean Average Precision)

```python
from examples.yolo.evaluate import DetectionMetrics, compute_ap

# Create evaluator
metrics = DetectionMetrics(num_classes=10)

# Update predictions
metrics.update(
    pred_boxes=pred_boxes,      # (N, 4)
    pred_scores=pred_scores,    # (N,)
    pred_labels=pred_labels,    # (N,)
    gt_boxes=gt_boxes,          # (M, 4)
    gt_labels=gt_labels         # (M,)
)

# Compute metrics
results = metrics.compute()
print(f"mAP@0.5: {results['mAP50']:.4f}")
print(f"mAP@0.5:0.95: {results['mAP']:.4f}")
```

### AP Calculation Method

Using VOC standard 11-point interpolation or COCO all-point interpolation:

```python
# All-point interpolation AP
recalls = np.array([0.1, 0.2, 0.3, ..., 1.0])
precisions = np.array([1.0, 0.9, 0.8, ..., 0.5])
ap = compute_ap(recalls, precisions)
```

---

## Code Examples

### Complete Inference Example

```python
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.detection import (
    build_backbone,
    build_neck,
    build_head,
    batched_nms
)

# Build model
backbone = build_backbone(model_size='s', use_attention=False)
neck = build_neck('panet', backbone.out_channels)
head = build_head('decoupled', neck.out_channels, num_classes=80)

# Prepare input
image = np.random.randn(1, 3, 640, 640).astype(np.float32)
x = Tensor(image)

# Forward pass
features = backbone(x)
features = neck(features)
predictions = head(features)

# Post-processing
all_boxes = []
all_scores = []
all_labels = []

for scale, (box_pred, cls_pred) in predictions.items():
    # Decode predictions...
    pass

# NMS
keep = batched_nms(all_boxes, all_scores, all_labels, iou_threshold=0.45)
final_boxes = all_boxes[keep]
final_scores = all_scores[keep]
final_labels = all_labels[keep]
```

### Run Tests

```bash
# Run unit tests
python -m pytest tests/detection/ -v

# Run training example
python examples/yolo/train.py

# Run evaluation example
python examples/yolo/evaluate.py
```

---

## Model Size Comparison

| Model | Size (MB) | mAP@0.5:0.95 | FPS (V100) |
|-------|-----------|--------------|------------|
| YOLOv12-n | 5.2 | 37.8 | 120 |
| YOLOv12-s | 15.4 | 45.2 | 95 |
| YOLOv12-m | 32.1 | 51.5 | 65 |
| YOLOv12-l | 52.3 | 54.8 | 45 |
| YOLOv12-x | 78.9 | 56.4 | 32 |

---

## Summary

This tutorial covered the complete process of implementing YOLO v12 using nanotorch:

1. **Core Components**: Bounding box handling, IoU calculation, NMS
2. **Network Modules**: Conv, C2f, SPPF, Area Attention
3. **Model Architecture**: R-ELAN Backbone + PANet Neck + Anchor-Free Head
4. **Training Process**: Loss functions, data augmentation, optimization strategies
5. **Evaluation Metrics**: mAP calculation, precision-recall curves

After this tutorial, you should be able to:
- Understand YOLO v12's architectural design
- Build object detection models using nanotorch
- Customize training and evaluation workflows

---

## References

1. YOLO v12 Paper: Attention-Centric Object Detection
2. CIoU Paper: https://arxiv.org/abs/1911.08287
3. DFL Paper: https://arxiv.org/abs/2006.04388
4. nanotorch Documentation: `/docs`
