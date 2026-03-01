# YOLO v6 目标检测模型实现教程

## 想象你在经营一家工厂...

你需要流水线：
- 训练时可以复杂（多分支，效果好）
- 推理时必须简单（单分支，速度快）
- 量产时效率第一（结构重参数化）

```
传统方法：
  训练和推理用同样的结构
  要想快，就得简化结构
  简化了，效果就差了

YOLO v6 的创新（RepVGG）：
  训练时：3 个分支（3×3 + 1×1 + 恒等）
  推理时：合并成 1 个 3×3 卷积
  效果好 + 速度快，两全其美
```

**YOLO v6 = 工业级部署专家** —— 美团出品，为生产而生。

---

## 概述

YOLO v6 是美团技术团队发布的 YOLO 版本，专注于工业应用。

### YOLO v6 的主要特点

1. **RepVGG 风格骨干网络**: 训练时多分支，推理时单分支
2. **解耦头 (Decoupled Head)**: 分类和回归分离
3. **SiLU 激活函数**: 平滑的激活函数
4. **高效训练和推理**: 工业级优化

### nanotorch 实现模块

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

## 模型架构

### RepVGG Block

RepVGG 使用重参数化技术：
- 训练时：3x3 conv + 1x1 conv + identity
- 推理时：可融合为单个 3x3 conv

```python
from nanotorch.detection.yolo_v6 import RepVGGBlock
from nanotorch.tensor import Tensor
import numpy as np

block = RepVGGBlock(64, 128, stride=1)
x = Tensor(np.random.randn(1, 64, 56, 56).astype(np.float32))
y = block(x)
print(y.shape)  # (1, 128, 56, 56)
```

### 解耦头 (Decoupled Head)

分类和回归使用独立的分支：

```python
from nanotorch.detection.yolo_v6 import DecoupledHead

head = DecoupledHead(256, num_anchors=1, num_classes=80)
x = Tensor(np.random.randn(1, 256, 13, 13).astype(np.float32))
y = head(x)
print(y.shape)  # (1, 85, 13, 13)
```

## 损失函数

### YOLO v6 损失函数设计

YOLO v6 使用 Anchor-Free 设计，损失函数包括：

$$
L = \lambda_{box} L_{box} + \lambda_{obj} L_{obj} + \lambda_{class} L_{class}
$$

### 边界框损失

YOLO v6 使用 SIoU 或 GIoU 损失：

**GIoU 损失**：

$$
L_{GIoU} = 1 - \text{GIoU} = 1 - \left( \text{IoU} - \frac{|C - (A \cup B)|}{|C|} \right)
$$

其中 $C$ 是包含两个框的最小凸集。

**SIoU 损失**（考虑角度）：

$$
L_{SIoU} = 1 - \text{IoU} + \frac{\Delta + \Omega}{2}
$$

角度损失：

$$
\Delta = 1 - \sum_{t \in (x,y)} \exp\left(-\gamma \cdot \sin^2\left(\frac{\pi}{2} - \theta_t\right)\right)
$$

距离损失：

$$
\Omega = \sum_{t \in (x,y)} \left(1 - \exp\left(-\frac{c_t}{\sigma}\right)\right)
$$

### 解耦检测头

YOLO v6 将分类和回归分支分离：

$$
\text{Output}_{cls} = \text{Sigmoid}(\text{Conv}_{cls}(\text{Feature}))
$$

$$
\text{Output}_{reg} = \text{Conv}_{reg}(\text{Feature})
$$

### Anchor-Free 预测

直接预测中心点和宽高：

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

### 完整模型

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

## 训练流程

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

## 运行 Demo

```bash
# 查看架构
python examples/yolo_v6/demo.py --mode arch

# 训练
python examples/yolo_v6/demo.py --mode train --version n

# 推理
python examples/yolo_v6/demo.py --mode inference

# 完整流程
python examples/yolo_v6/demo.py --mode both
```

## 运行测试

```bash
python -m pytest tests/detection/yolo_v6/ -v
```

## YOLO v5 vs YOLO v6 对比

| 特性 | YOLO v5 | YOLO v6 |
|------|---------|---------|
| 开发者 | Ultralytics | 美团 |
| 骨干网络 | CSPDarknet | RepVGG |
| 检测头 | 耦合 | 解耦 |
| 锚框 | 有锚框 | Anchor-free |

## 参考资料

1. **YOLOv6 论文**: "YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications"
2. **GitHub**: https://github.com/meituan/YOLOv6
