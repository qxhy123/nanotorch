# YOLO v6 Object Detection Model Implementation Tutorial

## Training and Inference Are Two Worlds...

During training, you have the luxury of time—you can stack complex structures, let the model learn more thoroughly.

During inference, you count every penny—every millisecond is a cost, every multiplication must be carefully calculated.

**Can you have the best of both worlds?**

YOLO v6 provides the answer: **Structural Reparameterization**.

During training, it's like a lush tree with three branches drawing nutrients together—3x3 convolution, 1x1 convolution, identity mapping—each doing its part, complementing each other.

During inference, this tree quietly transforms, all branches merge into one, becoming a solid trunk—a single 3x3 convolution, traveling light, fast as lightning.

```
Training: Intricate as brocade, layer upon layer
Inference: Simple as a sword, striking with precision

This is not compromise, but wisdom
Doing the right thing at the right time
```

**YOLO v6 — From Meituan, born for production**, industrial-grade elegance.

---

## Overview

YOLO v6 is released by Meituan Technical Team, focusing on industrial applications.

### Main Features of YOLO v6

1. **RepVGG-style Backbone**: Multi-branch during training, single-branch during inference
2. **Decoupled Head**: Separate classification and regression
3. **SiLU Activation**: Smooth activation function
4. **Efficient Training and Inference**: Industrial-grade optimization

### nanotorch Implementation Modules

```
nanotorch/detection/yolo_v6/
├── __init__.py
├── yolo_v6_model.py   # RepVGGBlock, DecoupledHead, YOLOv6
└── yolo_v6_loss.py    # YOLOv6Loss, encode/decode

examples/yolo_v6/
└── demo.py

tests/detection/yolo_v6/
├── test_yolov6_model.py
└── test_integration.py
```

## Model Architecture

### RepVGG Block

RepVGG uses reparameterization technique:
- Training: 3x3 conv + 1x1 conv + identity
- Inference: Can be fused into single 3x3 conv

```python
from nanotorch.detection.yolo_v6 import RepVGGBlock
from nanotorch.tensor import Tensor
import numpy as np

block = RepVGGBlock(64, 128, stride=1)
x = Tensor(np.random.randn(1, 64, 56, 56).astype(np.float32))
y = block(x)
print(y.shape)  # (1, 128, 56, 56)
```

### Decoupled Head

Classification and regression use independent branches:

```python
from nanotorch.detection.yolo_v6 import DecoupledHead

head = DecoupledHead(256, num_anchors=1, num_classes=80)
x = Tensor(np.random.randn(1, 256, 13, 13).astype(np.float32))
y = head(x)
print(y.shape)  # (1, 85, 13, 13)
```

## Loss Function

### YOLO v6 Loss Function Design

YOLO v6 uses Anchor-Free design, loss function includes:

$$
L = \lambda_{box} L_{box} + \lambda_{obj} L_{obj} + \lambda_{class} L_{class}
$$

### Bounding Box Loss

YOLO v6 uses SIoU or GIoU loss:

**GIoU Loss**:

$$
L_{GIoU} = 1 - \text{GIoU} = 1 - \left( \text{IoU} - \frac{|C - (A \cup B)|}{|C|} \right)
$$

Where $C$ is the smallest convex set containing both boxes.

**SIoU Loss** (considering angle):

$$
L_{SIoU} = 1 - \text{IoU} + \frac{\Delta + \Omega}{2}
$$

Angle loss:

$$
\Delta = 1 - \sum_{t \in (x,y)} \exp\left(-\gamma \cdot \sin^2\left(\frac{\pi}{2} - \theta_t\right)\right)
$$

Distance loss:

$$
\Omega = \sum_{t \in (x,y)} \left(1 - \exp\left(-\frac{c_t}{\sigma}\right)\right)
$$

### Decoupled Detection Head

YOLO v6 separates classification and regression branches:

$$
\text{Output}_{cls} = \text{Sigmoid}(\text{Conv}_{cls}(\text{Feature}))
$$

$$
\text{Output}_{reg} = \text{Conv}_{reg}(\text{Feature})
$$

### Anchor-Free Prediction

Directly predict center point and width/height:

$$
b_x = 2 \cdot \sigma(t_x) - 0.5 + c_x
$$

$$
b_y = 2 \cdot \sigma(t_y) - 0.5 + c_y
$$

$$
b_w = 2 \cdot \sigma(t_w)^2 \cdot a_w
$$

$$
b_h = 2 \cdot \sigma(t_h)^2 \cdot a_h
$$

### Complete Model

```python
from nanotorch.detection.yolo_v6 import YOLOv6, YOLOv6Nano, YOLOv6Small, build_yolov6

model = YOLOv6Nano(num_classes=80, input_size=640)
model = build_yolov6('s', num_classes=80, input_size=640)

x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
output = model(x)

print(output['small'].shape)
print(output['medium'].shape)
print(output['large'].shape)
```

## Training Process

```python
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.detection.yolo_v6 import build_yolov6
import numpy as np

model = build_yolov6('n', num_classes=80, input_size=224)
optimizer = Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    images = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
    optimizer.zero_grad()
    output = model(images)
    loss = sum(np.mean(pred.data ** 2) for pred in output.values())
    Tensor(loss, requires_grad=True).backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
```

## Running Demo

```bash
# View architecture
python examples/yolo_v6/demo.py --mode arch

# Training
python examples/yolo_v6/demo.py --mode train --version n

# Inference
python examples/yolo_v6/demo.py --mode inference

# Complete workflow
python examples/yolo_v6/demo.py --mode both
```

## Running Tests

```bash
python -m pytest tests/detection/yolo_v6/ -v
```

## YOLO v5 vs YOLO v6 Comparison

| Feature | YOLO v5 | YOLO v6 |
|---------|---------|---------|
| Developer | Ultralytics | Meituan |
| Backbone | CSPDarknet | RepVGG |
| Detection Head | Coupled | Decoupled |
| Anchors | Anchor-based | Anchor-free |

## References

1. **YOLOv6 Paper**: "YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications"
2. **GitHub**: https://github.com/meituan/YOLOv6
