# YOLO v4 Object Detection Model Implementation Tutorial

This tutorial provides a comprehensive guide to implementing YOLO v4 (You Only Look Once, 2020) object detection model from scratch using nanotorch.

## Table of Contents

1. [Overview](#overview)
2. [YOLO v4 Core Improvements](#yolo-v4-core-improvements)
3. [Model Architecture](#model-architecture)
4. [Loss Function](#loss-function)
5. [Data Preparation](#data-preparation)
6. [Training Process](#training-process)
7. [Inference and Post-processing](#inference-and-post-processing)
8. [Code Examples](#code-examples)

---

## Overview

YOLO v4 is the fourth version of the YOLO series, significantly improving detection accuracy while maintaining real-time performance.

### Main Contributions of YOLO v4

1. **CSPDarknet53 Backbone**: Uses Cross Stage Partial connections
2. **SPP Module**: Spatial Pyramid Pooling for enhanced receptive field
3. **PANet**: Path Aggregation Network for better feature fusion
4. **Mish Activation**: Smooth non-monotonic activation function
5. **CIoU Loss**: Complete IoU loss for better bounding box regression
6. **Bag of Freebies (BoF)**: Techniques that only increase training cost
7. **Bag of Specials (BoS)**: Modules that add minimal inference cost

### nanotorch YOLO v4 Implementation Modules

```
nanotorch/detection/yolo_v4/
├── __init__.py        # Module exports
├── yolo_v4_model.py   # Model architecture (CSPDarknet53, PANet, YOLOHead, YOLOv4, YOLOv4Tiny)
└── yolo_v4_loss.py    # Loss function (YOLOv4Loss, YOLOv4LossSimple, encode_targets_v4, decode_predictions_v4)

examples/yolo_v4/
└── demo.py            # Training and inference demo

tests/detection/yolo_v4/
├── test_yolov4_model.py  # Unit tests
└── test_integration.py   # Integration tests
```

---

## YOLO v4 Core Improvements

### CSPDarknet53 Backbone Network

CSP (Cross Stage Partial) connections split the feature map into two parts, reducing computation while maintaining gradient flow:

```
Input Feature
    ├─── Branch 1: Direct connection
    │
    └─── Branch 2: Through residual blocks
              ↓
         Concat(Branch 1, Branch 2)
              ↓
           Merge Conv
```

### SPP (Spatial Pyramid Pooling)

SPP module captures multi-scale context through max pooling at different scales:

```
Input Feature
    ↓
Conv 1x1
    ↓
┌───────────────────────────────────────┐
│  MaxPool 5x5 →──────────────────────┐ │
│  MaxPool 9x9 →────────────────────┐ │ │
│  MaxPool 13x13 →───────────────┐  │ │ │
│  Original Feature →──────────┐ │  │ │ │
└──────────────────────────────┼─┼──┼─┼─┘
                               ↓ ↓  ↓ ↓
                           Concat
                               ↓
                          Conv 1x1
```

### PANet (Path Aggregation Network)

PANet improves feature fusion:

```
Backbone Output:
  scale1 (512 channels, 13×13)  ← Large objects
  scale2 (512 channels, 26×26)  ← Medium objects
  scale3 (256 channels, 52×52)  ← Small objects

PANet Processing:
┌─────────────────────────────────────────────────────┐
│ Top-down pathway:                                    │
│   scale1 → Conv → Upsample → Concat(scale2) → p4   │
│   p4 → Conv → Upsample → Concat(scale3) → p3       │
├─────────────────────────────────────────────────────┤
│ Bottom-up pathway:                                   │
│   p3 → Downsample → Concat(p4) → n4                │
│   n4 → Downsample → Concat(p5) → n5                │
└─────────────────────────────────────────────────────┘
```

### Mish Activation Function

Mish is a smooth non-monotonic activation function:

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

```python
from nanotorch.detection.yolo_v4 import Mish
from nanotorch.tensor import Tensor
import numpy as np

mish = Mish()
x = Tensor(np.array([-2, -1, 0, 1, 2], dtype=np.float32))
y = mish(x)
print(y.data)  # Smooth activation values
```

---

## Model Architecture

### 1. ConvBNMish (Basic Convolution Block)

```python
from nanotorch.detection.yolo_v4 import ConvBNMish
from nanotorch.tensor import Tensor
import numpy as np

conv_bn = ConvBNMish(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1
)

x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = conv_bn(x)
print(y.shape)  # (1, 128, 52, 52)
```

### 2. CSPResBlock (CSP Residual Block)

```python
from nanotorch.detection.yolo_v4 import CSPResBlock

csp_block = CSPResBlock(
    in_channels=256,
    out_channels=256,
    num_blocks=2
)

x = Tensor(np.random.randn(1, 256, 52, 52).astype(np.float32))
y = csp_block(x)
print(y.shape)  # (1, 256, 52, 52)
```

### 3. SPP Module

```python
from nanotorch.detection.yolo_v4 import SPP

spp = SPP(
    in_channels=512,
    out_channels=512,
    pool_sizes=[5, 9, 13]
)

x = Tensor(np.random.randn(1, 512, 13, 13).astype(np.float32))
y = spp(x)
print(y.shape)  # (1, 512, 13, 13)
```

### 4. CSPDarknet53 Backbone Network

```python
from nanotorch.detection.yolo_v4 import CSPDarknet53

backbone = CSPDarknet53(in_channels=3)

x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
features = backbone(x)

print(features['scale1'].shape)  # (1, 512, 13, 13)
print(features['scale2'].shape)  # (1, 512, 26, 26)
print(features['scale3'].shape)  # (1, 256, 52, 52)
```

### 5. Complete YOLOv4 Model

```python
from nanotorch.detection.yolo_v4 import YOLOv4, YOLOv4Tiny, build_yolov4

# Method 1: Create complete model
model = YOLOv4(num_classes=80, input_size=416)

# Method 2: Create lightweight version
tiny_model = YOLOv4Tiny(num_classes=80, input_size=416)

# Method 3: Use factory function
model = build_yolov4('full', num_classes=80, input_size=416)
tiny_model = build_yolov4('tiny', num_classes=80, input_size=416)

# Forward pass
x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
output = model(x)

print(output['small'].shape)   # (1, 255, 13, 13)
print(output['medium'].shape)  # (1, 255, 26, 26)
print(output['large'].shape)   # (1, 255, 52, 52)
```

---

## Loss Function

### CIoU Loss

YOLO v4 uses CIoU (Complete IoU) loss instead of MSE for bounding box regression:

$$\text{CIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2} - \alpha v$$

Where:
- $\rho$: Euclidean distance between box centers
- $c$: Diagonal length of smallest enclosing box
- $v$: Aspect ratio consistency
- $\alpha$: Trade-off parameter

### Complete CIoU Formula

$$
\mathcal{L}_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$

Aspect ratio consistency:

$$
v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2
$$

Trade-off parameter:

$$
\alpha = \frac{v}{1 - \text{IoU} + v}
$$

### Total Loss Function

$$
L = \lambda_{box} L_{CIoU} + \lambda_{obj} L_{obj} + \lambda_{noobj} L_{noobj} + \lambda_{class} L_{class}
$$

### Mish Activation Function Formula

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

Derivative:

$$
\text{Mish}'(x) = \frac{e^x}{1 + e^x} \cdot \tanh(\text{softplus}(x)) + \frac{x}{1 + e^x} \cdot \text{sech}^2(\text{softplus}(x))
$$

### CmBN (Cross mini-Batch Normalization)

CmBN aggregates statistics across multiple batches:

$$
\mu_t = \frac{1}{m \times k} \sum_{i=1}^{k} \sum_{j=1}^{m} x_{t-i,j}
$$

$$
\sigma_t^2 = \frac{1}{m \times k} \sum_{i=1}^{k} \sum_{j=1}^{m} (x_{t-i,j} - \mu_t)^2
$$

Where $k$ is the number of aggregated batches.

### Using the Loss Function

```python
from nanotorch.detection.yolo_v4 import YOLOv4Loss, YOLOv4LossSimple

# Create loss function
loss_fn = YOLOv4Loss(
    num_classes=80,
    ignore_threshold=0.5,
    lambda_coord=5.0,
    lambda_obj=1.0,
    lambda_noobj=0.5,
    lambda_class=1.0
)

# Prepare predictions and targets
predictions = {
    'small': Tensor(np.random.randn(2, 255, 13, 13).astype(np.float32) * 0.1),
    'medium': Tensor(np.random.randn(2, 255, 26, 26).astype(np.float32) * 0.1),
    'large': Tensor(np.random.randn(2, 255, 52, 52).astype(np.float32) * 0.1)
}

targets = [
    {'boxes': np.array([[100, 100, 200, 200]], dtype=np.float32), 
     'labels': np.array([0], dtype=np.int64)},
    {'boxes': np.array([[50, 50, 150, 150]], dtype=np.float32),
     'labels': np.array([1], dtype=np.int64)}
]

# Compute loss
loss, loss_dict = loss_fn(predictions, targets)

print(f"Total Loss: {loss.item():.4f}")
print(f"Coord Loss: {loss_dict['coord_loss']:.4f}")
print(f"Obj Loss: {loss_dict['obj_loss']:.4f}")
```

### Target Encoding and Decoding

```python
from nanotorch.detection.yolo_v4 import encode_targets_v4, decode_predictions_v4

# YOLO v4 anchors
anchors = [
    (12, 16), (19, 36), (40, 28),
    (36, 75), (76, 55), (72, 146),
    (142, 110), (192, 243), (459, 401)
]

# Encoding
boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
labels = np.array([0], dtype=np.int64)

targets = encode_targets_v4(
    boxes=boxes,
    labels=labels,
    anchors=anchors,
    grid_sizes=[13, 26, 52],
    num_classes=80,
    image_size=416
)

print(targets['scale_0'].shape)  # (3, 85, 13, 13)
print(targets['scale_1'].shape)  # (3, 85, 26, 26)
print(targets['scale_2'].shape)  # (3, 85, 52, 52)

# Decoding
predictions = np.random.randn(3, 85, 13, 13).astype(np.float32) * 0.1
scale_anchors = [(142, 110), (192, 243), (459, 401)]

boxes, scores, class_ids = decode_predictions_v4(
    predictions,
    anchors=scale_anchors,
    conf_threshold=0.5,
    num_classes=80,
    image_size=416
)
```

---

## Data Preparation

### SyntheticCOCODataset

```python
from examples.yolo_v4.demo import SyntheticCOCODataset, create_dataloader

# Create dataset
dataset = SyntheticCOCODataset(
    num_samples=1000,
    image_size=416,
    num_classes=80,
    max_objects=5
)

sample = dataset[0]
print(sample['image'].shape)     # (3, 416, 416)
print(sample['boxes'].shape)     # (N, 4)
print(sample['labels'].shape)    # (N,)

# Create DataLoader
dataloader = create_dataloader(
    num_samples=100,
    batch_size=4,
    image_size=416,
    num_classes=80
)

for batch in dataloader:
    images = np.stack([item['image'] for item in batch])
    print(images.shape)  # (4, 3, 416, 416)
    break
```

---

## Training Process

```python
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v4 import YOLOv4Tiny
from examples.yolo_v4.demo import create_dataloader
import numpy as np

# Create model
model = YOLOv4Tiny(num_classes=80, input_size=224)

# Optimizer and scheduler
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Dataloader
dataloader = create_dataloader(
    num_samples=50,
    batch_size=2,
    image_size=224,
    num_classes=80
)

# Training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        images = np.stack([item['image'] for item in batch])
        images = Tensor(images)
        
        optimizer.zero_grad()
        output = model(images)
        
        # MSE loss
        loss = 0.0
        for scale_name, pred in output.items():
            diff = pred - Tensor(np.zeros_like(pred.data))
            loss += (diff * diff).mean().item()
        
        loss_tensor = Tensor(loss, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()
        
        total_loss += loss
    
    scheduler.step()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

---

## Inference and Post-processing

```python
from nanotorch.detection.yolo_v4 import YOLOv4Tiny
from nanotorch.detection.nms import nms
from nanotorch.tensor import Tensor
import numpy as np

# Load model
model = YOLOv4Tiny(num_classes=80, input_size=416)
model.eval()

# Prepare input
image = np.random.randn(1, 3, 416, 416).astype(np.float32)
x = Tensor(image)

# Forward pass
output = model(x)
print(output['small'].shape)  # (1, 255, 13, 13)

# Apply NMS
boxes = np.array([[100, 100, 200, 200], [105, 105, 205, 205]], dtype=np.float32)
scores = np.array([0.9, 0.8], dtype=np.float32)

keep = nms(boxes, scores, iou_threshold=0.5)
print(f"Kept detection boxes: {keep}")
```

---

## Code Examples

### Complete Training Example

```python
"""
YOLO v4 complete training example
"""
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v4 import YOLOv4Tiny, YOLOv4Loss

def train_yolov4():
    # Hyperparameters
    num_epochs = 10
    batch_size = 4
    learning_rate = 1e-4
    image_size = 224
    num_classes = 80
    
    # Create model
    model = YOLOv4Tiny(num_classes=num_classes, input_size=image_size)
    
    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # Simulated data
        for _ in range(10):
            images = Tensor(np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32))
            
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
        avg_loss = total_loss / 10
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

if __name__ == "__main__":
    model = train_yolov4()
    print("Training complete!")
```

### Running Tests

```bash
# Run YOLO v4 unit tests
python -m pytest tests/detection/yolo_v4/test_yolov4_model.py -v

# Run integration tests
python -m pytest tests/detection/yolo_v4/test_integration.py -v

# Run all YOLO v4 tests
python -m pytest tests/detection/yolo_v4/ -v
```

---

## Quick Start: Demo Script

### View Model Architecture

```bash
python examples/yolo_v4/demo.py --mode arch
```

### Training Model

```bash
# Default uses YOLOv4Tiny
python examples/yolo_v4/demo.py --mode train

# Custom parameters
python examples/yolo_v4/demo.py --mode train \
    --epochs 5 \
    --batch-size 4 \
    --num-samples 50

# Use full YOLOv4 (requires more memory)
python examples/yolo_v4/demo.py --mode train --full --batch-size 1
```

### Inference Demo

```bash
python examples/yolo_v4/demo.py --mode inference
```

### Complete Workflow

```bash
python examples/yolo_v4/demo.py --mode both
```

---

## YOLO v3 vs YOLO v4 Comparison

| Feature | YOLO v3 | YOLO v4 |
|---------|---------|---------|
| Backbone | Darknet-53 | CSPDarknet-53 |
| Activation | LeakyReLU | Mish |
| Feature Fusion | FPN | PANet |
| Pooling Module | None | SPP |
| Bounding Box Loss | MSE | CIoU Loss |
| Data Augmentation | Basic | Mosaic, Mixup, etc. |

---

## Model Parameter Comparison

| Model | Input Size | Parameters | Memory |
|-------|------------|------------|--------|
| YOLOv4Tiny | 416×416 | ~6M | ~2GB |
| YOLOv4 | 416×416 | ~64M | ~8GB |

---

## Common Issues

### 1. Why does YOLO v4 use CSP connections?

Advantages of CSP (Cross Stage Partial) connections:
- Reduces computation by ~20%
- Mitigates gradient vanishing
- Maintains or improves accuracy

### 2. What improvements does PANet have over FPN?

PANet improvements:
- Adds bottom-up pathway
- Better feature fusion
- Improved small object detection

### 3. Advantages of CIoU Loss?

CIoU Loss considers:
- Overlap area
- Center point distance
- Aspect ratio consistency
- Better convergence than MSE

---

## Summary

This tutorial covered the complete process of implementing YOLO v4 using nanotorch:

1. **Core Improvements**: CSPDarknet53, PANet, SPP, Mish activation
2. **Model Architecture**: Backbone + Feature Fusion + Detection Head
3. **Loss Function**: CIoU loss + BCE loss
4. **Training & Inference**: Complete training and inference workflow

After this tutorial, you should be able to:
- Understand YOLO v4's architecture design
- Build complex object detection models using nanotorch
- Understand the role of CSP connections and PANet

---

## References

1. **YOLO v4 Paper**: "YOLOv4: Optimal Speed and Accuracy of Object Detection" (2020)
   - https://arxiv.org/abs/2004.10934

2. **CSPNet**: "CSPNet: A New Backbone that can Enhance Learning Capability of CNN" (2020)
   - https://arxiv.org/abs/1911.11929

3. **PANet**: "Path Aggregation Network for Instance Segmentation" (2018)
   - https://arxiv.org/abs/1803.01534

4. **Mish**: "Mish: A Self Regularized Non-Monotonic Activation Function" (2019)
   - https://arxiv.org/abs/1908.08681

5. **nanotorch YOLO v3 Tutorial**: `/docs/tutorials/19-yolov3_EN.md`
