# YOLO v1 Object Detection Model Implementation Tutorial

## The Beginning of a Revolution...

In 2015, a paper titled "You Only Look Once" changed object detection forever.

Before YOLO, detection was slow. Models would propose hundreds of candidate regions, then classify each one separately. It worked, but it was like reading a book by examining one word at a time.

**YOLO asked: why not read the whole page at once?**

```
The YOLO v1 Breakthrough:

  Traditional detection:
    1. Generate region proposals (slow)
    2. Classify each region (slow)
    3. Refine bounding boxes (slow)
    → 0.05 frames per second

  YOLO v1:
    1. Look at the whole image once
    2. Predict all boxes simultaneously
    3. Done
    → 45 frames per second

  The insight:
    Detection is just regression.
    Each grid cell predicts what it contains.
    No proposals needed. No slow pipelines.
```

**YOLO v1 wasn't perfect, but it was a paradigm shift.** It showed that object detection could be unified, elegant, and fast. The accuracy wasn't state-of-the-art, but the speed made real-time detection possible for the first time.

In this tutorial, we'll implement YOLO v1 from scratch. We'll see how grid-based prediction works, how the loss function balances localization and classification, and how a single forward pass can predict all objects in an image. This is where the YOLO story begins.

---

## Table of Contents

1. [Overview](#overview)
2. [YOLO v1 Core Concepts](#yolo-v1-core-concepts)
3. [Model Architecture](#model-architecture)
4. [Loss Function](#loss-function)
5. [Data Preparation](#data-preparation)
6. [Training Process](#training-process)
7. [Inference and Post-processing](#inference-and-post-processing)
8. [Code Examples](#code-examples)

---

## Overview

YOLO v1 is a groundbreaking work in object detection, first modeling object detection as a **single regression problem**, enabling real-time object detection.

### Main Contributions of YOLO v1

1. **Single-stage Detection**: Formulates object detection as a regression problem, eliminating region proposal generation
2. **Real-time Performance**: Achieves 45 FPS on VOC 2007 test set
3. **Global Reasoning**: Uses the entire image for prediction, reducing false positives from background

### nanotorch YOLO v1 Implementation Modules

```
nanotorch/detection/yolo_v1/
├── __init__.py        # Module exports
├── yolo_v1_model.py   # Model architecture (Darknet, YOLOv1, YOLOv1Tiny)
└── yolo_v1_loss.py    # Loss function (YOLOv1Loss, encode_targets, decode_predictions)

examples/yolo_v1/
└── data.py            # Data loading (SyntheticVOCDataset, YOLOv1Transform)

tests/detection/yolo_v1/
├── test_yolov1_model.py  # Unit tests (27 tests)
└── test_integration.py   # Integration tests (17 tests)
```

---

## YOLO v1 Core Concepts

### Grid-based Detection

YOLO v1 divides the input image into S×S grid cells (S=7 in the original paper):

```
Input Image (448×448)
    ↓
Divide into 7×7 grid
    ↓
Each grid cell responsible for detecting:
  - Objects whose center falls within the grid
  - Predicting B bounding boxes (B=2 in original paper)
  - Predicting C class probabilities (C=20 for VOC dataset)
```

### Output Tensor Format

Model output shape is (N, S, S, B×5+C):

```
Output per grid cell (30 values total):
┌─────────────────────────────────────────────────────────────┐
│ Box 1 (5)    │ Box 2 (5)    │ Class Probs (20)              │
│ x, y, w, h, c│ x, y, w, h, c│ p0, p1, p2, ..., p19          │
└─────────────────────────────────────────────────────────────┘
      ↑              ↑              ↑
   1st bbox      2nd bbox     20 class probs

- x, y: Bounding box center coordinates relative to grid cell (0~1)
- w, h: Bounding box width/height relative to entire image (0~1)
- c: Confidence = Pr(Object) × IoU(pred, truth)
```

### Darknet Backbone Network

YOLO v1 uses Darknet-24 as the backbone:

```
Input (3, 448, 448)
    ↓
Conv 7×7, s=2 → MaxPool 2×2, s=2
Conv 3×3 → Conv 3×3 → MaxPool
...(24 conv layers total)
    ↓
Feature Map (1024, 14, 14)
    ↓
Flatten → FC 4096 → FC 1470
    ↓
Output (7×7×30 = 1470)
```

---

## Model Architecture

### 1. ConvBlock (Basic Convolution Block)

Each convolution block contains: Conv2D + LeakyReLU

```python
from nanotorch.detection.yolo_v1 import ConvBlock
from nanotorch.tensor import Tensor
import numpy as np

# Create convolution block
conv_block = ConvBlock(
    in_channels=3,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3
)

# Forward pass
x = Tensor(np.random.randn(1, 3, 448, 448).astype(np.float32))
y = conv_block(x)
print(y.shape)  # (1, 64, 224, 224)
```

### 2. Darknet Backbone Network

```python
from nanotorch.detection.yolo_v1 import Darknet

# Create Darknet backbone
backbone = Darknet(in_channels=3)

# Forward pass
x = Tensor(np.random.randn(1, 3, 448, 448).astype(np.float32))
features = backbone(x)
print(features.shape)  # (1, 1024, 7, 7)
```

Darknet network structure:

```
Layer  Type        Filters    Size/Stride    Output
─────────────────────────────────────────────────────
0      Conv        64         7 × 7 / 2      (224, 224)
1      MaxPool                2 × 2 / 2      (112, 112)
2      Conv        192        3 × 3 / 1      (112, 112)
3      MaxPool                2 × 2 / 2      (56, 56)
4      Conv        128        1 × 1 / 1      (56, 56)
5      Conv        256        3 × 3 / 1      (56, 56)
6      Conv        256        1 × 1 / 1      (56, 56)
7      Conv        512        3 × 3 / 1      (56, 56)
8      MaxPool                2 × 2 / 2      (28, 28)
...    ...         ...        ...            ...
23     Conv        1024       3 × 3 / 1      (7, 7)
```

### 3. YOLOv1Head (Detection Head)

```python
from nanotorch.detection.yolo_v1 import YOLOv1Head

head = YOLOv1Head(
    in_channels=1024,
    hidden_dim=4096,
    S=7,
    B=2,
    C=20
)

x = Tensor(np.random.randn(1, 1024, 7, 7).astype(np.float32))
output = head(x)
print(output.shape)  # (1, 1470)
```

### 4. YOLOv1 Complete Model

```python
from nanotorch.detection.yolo_v1 import YOLOv1, YOLOv1Tiny, build_yolov1

# Method 1: Create complete model directly
model = YOLOv1(
    input_size=448,
    S=7,
    B=2,
    C=20
)

# Method 2: Create tiny version (for testing)
tiny_model = YOLOv1Tiny(
    input_size=224,
    S=7,
    B=2,
    C=20
)

# Method 3: Use factory function
model = build_yolov1('full', input_size=448, S=7, B=2, C=20)
tiny_model = build_yolov1('tiny', input_size=224, S=7, B=2, C=20)

# Forward pass
x = Tensor(np.random.randn(1, 3, 448, 448).astype(np.float32))
output = model(x)
print(output.shape)  # (1, 1470)

# Use predict method for structured output
result = model.predict(x)
print(result['reshaped'].shape)  # (1, 7, 7, 30)
```

---

## Loss Function

### YOLO v1 Loss Function Design

YOLO v1 uses **weighted sum-squared error loss**:

$$
L = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]
$$

$$
+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2
+ \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2
$$

$$
+ \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2
$$

Where:
- $\lambda_{coord} = 5.0$: Coordinate loss weight
- $\lambda_{noobj} = 0.5$: No-object confidence loss weight
- $\mathbb{1}_{ij}^{obj}$: Indicates grid i's j-th bounding box is responsible for detecting an object
- $S = 7$: Grid size
- $B = 2$: Number of bounding boxes predicted per grid
- $C = 20$: Number of classes

### Loss Function Breakdown

**Coordinate Loss** (using square root to reduce impact of large boxes):

$$
L_{coord} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]
$$

**Confidence Loss**:

$$
L_{conf} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 + \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2
$$

**Classification Loss**:

$$
L_{class} = \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2
$$

### Output Tensor Structure

$$
\text{Output} \in \mathbb{R}^{S \times S \times (B \times 5 + C)} = \mathbb{R}^{7 \times 7 \times 30}
$$

Each grid cell predicts:
- 2 bounding boxes: $(x, y, w, h, \text{confidence}) \times 2 = 10$ values
- 20 class probabilities: $C = 20$ values
- Total: $10 + 20 = 30$ values

### Using the Loss Function

```python
from nanotorch.detection.yolo_v1 import YOLOv1Loss, YOLOv1LossSimple
from nanotorch.tensor import Tensor

# Create loss function
loss_fn = YOLOv1Loss(
    S=7,
    B=2,
    C=20,
    coord_weight=5.0,
    noobj_weight=0.5
)

# Prepare predictions and targets
predictions = Tensor(np.random.randn(2, 7, 7, 30).astype(np.float32) * 0.1)
targets = Tensor(np.zeros((2, 7, 7, 30), dtype=np.float32))

# Compute loss
loss, loss_dict = loss_fn(predictions, targets)

print(f"Total Loss: {loss.item():.4f}")
print(f"Coord Loss: {loss_dict['coord_loss']:.4f}")
print(f"Obj Conf Loss: {loss_dict['obj_conf_loss']:.4f}")
print(f"Noobj Conf Loss: {loss_dict['noobj_conf_loss']:.4f}")
print(f"Class Loss: {loss_dict['class_loss']:.4f}")
```

### Simplified Loss Function (for testing)

```python
# Simplified MSE loss for quick testing
simple_loss_fn = YOLOv1LossSimple()

predictions = Tensor(np.random.randn(2, 1470).astype(np.float32))
targets = Tensor(np.zeros((2, 1470), dtype=np.float32))

loss, loss_dict = simple_loss_fn(predictions, targets)
# Note: YOLOv1LossSimple returns (float, dict), does not support backward
```

### Target Encoding and Decoding

```python
from nanotorch.detection.yolo_v1 import encode_targets, decode_predictions
import numpy as np

# Encoding: Convert bounding boxes to YOLO format
boxes = np.array([
    [100, 100, 200, 200],   # [x1, y1, x2, y2]
    [250, 250, 350, 350]
], dtype=np.float32)
labels = np.array([0, 5], dtype=np.int64)  # Class indices

target = encode_targets(
    boxes=boxes,
    labels=labels,
    S=7,
    B=2,
    C=20,
    image_size=448
)
print(target.shape)  # (7, 7, 30)

# Decoding: Convert predictions to bounding boxes
predictions = np.random.randn(7, 7, 30).astype(np.float32) * 0.1
boxes, scores, class_ids = decode_predictions(
    predictions,
    conf_threshold=0.5,
    image_size=448
)

print(f"Detected {len(boxes)} objects")
```

---

## Data Preparation

### SyntheticVOCDataset

Synthetic VOC format dataset for testing:

```python
from examples.yolo_v1.data import SyntheticVOCDataset

# Create synthetic dataset
dataset = SyntheticVOCDataset(
    num_samples=1000,
    image_size=448,
    S=7,
    B=2,
    C=20,
    max_objects=5,
    min_objects=1
)

# Get sample
sample = dataset[0]
print(sample['image'].shape)     # (448, 448, 3)
print(sample['target'].shape)    # (7, 7, 30)
print(sample['boxes'].shape)     # (N, 4)
print(sample['labels'].shape)    # (N,)

# VOC 20 classes
print(dataset.VOC_CLASSES)
# ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
#  'bus', 'car', 'cat', 'chair', 'cow', 
#  'diningtable', 'dog', 'horse', 'motorbike', 'person', 
#  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

### YOLOv1Transform (Data Preprocessing)

```python
from examples.yolo_v1.data import YOLOv1Transform

transform = YOLOv1Transform(
    image_size=448,
    S=7,
    B=2,
    C=20
)

# Apply transform
sample = dataset[0]
transformed = transform(sample)
```

### YOLOv1Collate (Batching)

```python
from examples.yolo_v1.data import YOLOv1Collate

collate = YOLOv1Collate(S=7, B=2, C=20)

# Batch multiple samples
samples = [dataset[i] for i in range(4)]
batch = collate(samples)

print(batch['images'].shape)   # (4, 3, 448, 448)
print(batch['targets'].shape)  # (4, 7, 7, 30)
```

### Creating DataLoader

```python
from examples.yolo_v1.data import create_synthetic_dataloader

# One-liner DataLoader creation
dataloader = create_synthetic_dataloader(
    num_samples=1000,
    batch_size=8,
    image_size=448,
    S=7,
    B=2,
    C=20,
    shuffle=True
)

# Iterate data
for batch in dataloader:
    images = batch['images']    # (8, 3, 448, 448)
    targets = batch['targets']  # (8, 7, 7, 30)
    break
```

---

## Training Process

### Basic Training Loop

```python
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.detection.yolo_v1 import YOLOv1, YOLOv1Loss
from examples.yolo_v1.data import create_synthetic_dataloader

# Create model and optimizer
model = YOLOv1(input_size=448, S=7, B=2, C=20)
optimizer = Adam(model.parameters(), lr=1e-4)

# Create dataloader
dataloader = create_synthetic_dataloader(
    num_samples=100,
    batch_size=4,
    image_size=448
)

# Training loop
for epoch in range(10):
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        # Prepare data
        images = Tensor(batch['images'])
        targets = Tensor(batch['targets'])
        
        # Forward pass
        optimizer.zero_grad()
        output = model(images)
        output_reshaped = output.reshape((images.shape[0], 7, 7, 30))
        
        # MSE loss for gradient computation
        # (YOLOv1Loss doesn't support backward, uses NumPy internally)
        diff = output_reshaped - targets
        loss = (diff * diff).mean()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
```

### Using YOLOv1Loss for Training Monitoring

```python
from nanotorch.detection.yolo_v1 import YOLOv1Loss

loss_fn = YOLOv1Loss(S=7, B=2, C=20)

# Monitor individual losses during training
output = model(images)
output_reshaped = output.reshape((images.shape[0], 7, 7, 30))

# Compute YOLO loss (for monitoring, doesn't support backward)
yolo_loss, loss_dict = loss_fn(output_reshaped, targets)

print(f"Coord Loss: {loss_dict['coord_loss']:.4f}")
print(f"Object Conf Loss: {loss_dict['obj_conf_loss']:.4f}")
print(f"No-Object Conf Loss: {loss_dict['noobj_conf_loss']:.4f}")
print(f"Class Loss: {loss_dict['class_loss']:.4f}")
```

---

## Inference and Post-processing

### Model Inference

```python
from nanotorch.detection.yolo_v1 import YOLOv1, decode_predictions
from nanotorch.tensor import Tensor
import numpy as np

# Load trained model
model = YOLOv1(input_size=448, S=7, B=2, C=20)
# model.load_state_dict(...)  # Load weights

# Prepare input image
image = np.random.randn(1, 3, 448, 448).astype(np.float32)
x = Tensor(image)

# Forward pass
output = model(x)
output_reshaped = output.reshape((1, 7, 7, 30))

# Decode predictions
predictions = output_reshaped.data[0]  # (7, 7, 30)
boxes, scores, class_ids = decode_predictions(
    predictions,
    conf_threshold=0.5,
    image_size=448
)

print(f"Detected {len(boxes)} objects")
for i in range(len(boxes)):
    print(f"Box: {boxes[i]}, Score: {scores[i]:.2f}, Class: {class_ids[i]}")
```

### Non-Maximum Suppression (NMS)

```python
from nanotorch.detection.nms import nms

# Assume multiple overlapping detection boxes
boxes = np.array([
    [100, 100, 200, 200],
    [105, 105, 205, 205],  # Highly overlaps with first
    [300, 300, 400, 400]
], dtype=np.float32)

scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)

# Apply NMS
keep = nms(boxes, scores, iou_threshold=0.5)
print(f"Kept indices: {keep}")  # [0, 2]

final_boxes = boxes[keep]
final_scores = scores[keep]
```

---

## Code Examples

### Complete Training Example

```python
"""
Complete YOLO v1 training example
"""
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam, SGD
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v1 import (
    YOLOv1,
    YOLOv1Loss,
    decode_predictions
)
from examples.yolo_v1.data import create_synthetic_dataloader

def train_yolov1():
    # Hyperparameters
    num_epochs = 50
    batch_size = 8
    learning_rate = 1e-4
    image_size = 448
    S, B, C = 7, 2, 20
    
    # Create model
    model = YOLOv1(input_size=image_size, S=S, B=B, C=C)
    
    # Create optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Create loss function (for monitoring)
    loss_fn = YOLOv1Loss(S=S, B=B, C=C)
    
    # Create dataloader
    dataloader = create_synthetic_dataloader(
        num_samples=200,
        batch_size=batch_size,
        image_size=image_size,
        S=S, B=B, C=C,
        shuffle=True
    )
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            images = Tensor(batch['images'])
            targets = Tensor(batch['targets'])
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(images)
            output_reshaped = output.reshape((images.shape[0], S, S, B*5+C))
            
            # MSE loss (for backward)
            diff = output_reshaped - targets
            loss = (diff * diff).mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Print statistics
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_lr():.6f}")
    
    return model

if __name__ == "__main__":
    model = train_yolov1()
    print("Training complete!")
```

### Inference Example

```python
"""
YOLO v1 inference example
"""
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.detection.yolo_v1 import YOLOv1, decode_predictions

def inference_example():
    # Load model
    model = YOLOv1(input_size=448, S=7, B=2, C=20)
    model.eval()
    
    # Prepare input (in practice, load real image)
    image = np.random.randn(1, 3, 448, 448).astype(np.float32)
    x = Tensor(image)
    
    # Forward pass
    output = model(x)
    output_reshaped = output.reshape((1, 7, 7, 30))
    
    # Decode predictions
    predictions = output_reshaped.data[0]
    boxes, scores, class_ids = decode_predictions(
        predictions,
        conf_threshold=0.3,
        image_size=448
    )
    
    # VOC class names
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # Print detection results
    print(f"Detected {len(boxes)} objects:")
    for i in range(len(boxes)):
        class_name = VOC_CLASSES[class_ids[i]]
        x1, y1, x2, y2 = boxes[i]
        print(f"  {class_name}: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), Confidence: {scores[i]:.2f}")

if __name__ == "__main__":
    inference_example()
```

### Running Tests

```bash
# Run YOLO v1 unit tests
python -m pytest tests/detection/yolo_v1/test_yolov1_model.py -v

# Run integration tests
python -m pytest tests/detection/yolo_v1/test_integration.py -v

# Run all YOLO v1 tests
python -m pytest tests/detection/yolo_v1/ -v
```

---

## Quick Start: Demo Script

We provide a complete demo script to demonstrate YOLO v1 training and inference workflow.

### View Model Architecture

```bash
python examples/yolo_v1/demo.py --mode arch
```

Example output:
```
============================================================
YOLO v1 Architecture Demo
============================================================

1. Full YOLOv1 Model:
   Parameters: 271,703,550
   Input shape: (1, 3, 448, 448)
   Output shape: (1, 1470)

2. YOLOv1Tiny Model (for testing):
   Parameters: 42,175,326
   Input shape: (1, 3, 224, 224)
   Output shape: (1, 1470)

3. Output Format:
   Grid size: 7×7 = 49 cells
   Boxes per cell: 2
   Classes: 20
   Output per cell: 2×5 + 20 = 30 values
   Total output: 7×7×30 = 1470 values
```

### Encode/Decode Demo

```bash
python examples/yolo_v1/demo.py --mode encode
```

Example output:
```
============================================================
YOLO v1 Encode/Decode Demo
============================================================

Input boxes:
  Object 1: bus at [100. 100. 200. 200.]
  Object 2: person at [300. 300. 400. 400.]

Encoded target shape: (7, 7, 30)
Non-zero elements: 22
Grid cells with objects: [(2, 2), (5, 5)]

Decoded boxes (4 total):
  bus: box=[100. 100. 200. 200.], score=1.00
  person: box=[300. 300. 400. 400.], score=1.00
```

### Training Model

```bash
# Default uses YOLOv1Tiny (recommended)
python examples/yolo_v1/demo.py --mode train

# Custom parameters
python examples/yolo_v1/demo.py --mode train \
    --epochs 5 \
    --batch-size 4 \
    --num-samples 50

# Use full YOLOv1 (requires 8GB+ memory)
python examples/yolo_v1/demo.py --mode train --full --batch-size 1
```

> ⚠️ **Memory Warning**: Full YOLOv1 has 271M parameters, 448×448 input requires ~8GB memory.
> If you encounter SIGKILL errors, use `--tiny` or reduce `--batch-size`.

Training parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 3 | Number of training epochs |
| `--batch-size` | 2 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--lr-step` | 5 | Learning rate decay step |
| `--image-size` | 224 | Input image size |
| `--num-samples` | 20 | Number of training samples |
| `--tiny` | True | Use YOLOv1Tiny (default) |
| `--full` | - | Use full YOLOv1 |

### Inference Demo

```bash
# Run inference after training
python examples/yolo_v1/demo.py --mode inference

# Custom inference parameters
python examples/yolo_v1/demo.py --mode inference \
    --num-inference 10 \
    --conf-threshold 0.3 \
    --nms-threshold 0.5
```

Inference parameters:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--num-inference` | 5 | Number of inference images |
| `--conf-threshold` | 0.3 | Confidence threshold |
| `--nms-threshold` | 0.5 | NMS IoU threshold |

### Complete Workflow (Training + Inference)

```bash
# Run complete demo: architecture → encode/decode → train → inference
python examples/yolo_v1/demo.py --mode both
```

### Demo Script Full Parameters

```bash
python examples/yolo_v1/demo.py --help
```

```
usage: demo.py [-h] [--mode {train,inference,both,encode,arch}]
               [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
               [--lr-step LR_STEP] [--image-size IMAGE_SIZE]
               [--num-samples NUM_SAMPLES]
               [--num-inference NUM_INFERENCE]
               [--conf-threshold CONF_THRESHOLD]
               [--nms-threshold NMS_THRESHOLD] [--tiny] [--full]

YOLO v1 Demo

options:
  --mode {train,inference,both,encode,arch}
                        Demo mode
  --epochs EPOCHS       Number of training epochs
  --batch-size BATCH_SIZE
                        Batch size
  --lr LR               Learning rate
  --lr-step LR_STEP     LR scheduler step
  --image-size IMAGE_SIZE
                        Input image size
  --num-samples NUM_SAMPLES
                        Number of training samples
  --num-inference NUM_INFERENCE
                        Number of inference images
  --conf-threshold CONF_THRESHOLD
                        Confidence threshold
  --nms-threshold NMS_THRESHOLD
                        NMS IoU threshold
  --tiny                Use YOLOv1Tiny (default: True)
  --full                Use full YOLOv1 (requires more memory)
```

---

## Model Parameter Comparison

| Model | Input Size | Parameters | Memory |
|-------|------------|------------|--------|
| YOLOv1Tiny | 224×224 | ~42M | ~2GB |
| YOLOv1 | 448×448 | ~272M | ~8GB |

---

## Common Issues

### 1. Why do I get SIGKILL errors?

Full YOLOv1 model has 271M parameters and requires significant memory. Solutions:
- Use `--tiny` (default)
- Reduce `--batch-size` (e.g., `--batch-size 1`)
- Reduce `--num-samples`

### 2. Why doesn't YOLOv1Loss support backward?

YOLOv1Loss uses NumPy internally for loss computation, returning a Tensor not connected to the computation graph. Use manual MSE loss for gradient computation during training.

### 3. How to handle negative width/height predictions?

YOLO v1 uses sqrt(w) and sqrt(h) for loss computation. Negative predictions cause NaN. Solutions:
- Use sigmoid to constrain output range
- Add regularization to prevent extreme predictions

### 4. How to improve detection accuracy?

- Increase training data
- Use data augmentation (Mosaic, MixUp)
- Adjust loss weights
- Use deeper backbone network

---

## Summary

This tutorial covered the complete process of implementing YOLO v1 using nanotorch:

1. **Core Concepts**: Grid-based detection, single-stage detection, global reasoning
2. **Model Architecture**: Darknet backbone + FC detection head
3. **Loss Function**: Weighted sum-squared error with coordinate, confidence, classification losses
4. **Data Processing**: Synthetic dataset, preprocessing, batching
5. **Training & Inference**: Complete training loop, prediction decoding, NMS post-processing

After this tutorial, you should be able to:
- Understand YOLO v1's design philosophy
- Build object detection models using nanotorch
- Customize training and evaluation workflows

---

## References

1. **YOLO v1 Paper**: "You Only Look Once: Unified, Real-Time Object Detection" (CVPR 2016)
   - https://arxiv.org/abs/1506.02640

2. **Darknet Official Implementation**: https://github.com/pjreddie/darknet

3. **PASCAL VOC Dataset**: http://host.robots.ox.ac.uk/pascal/VOC/

4. **nanotorch Documentation**: `/docs`
