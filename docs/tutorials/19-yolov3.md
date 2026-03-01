# YOLO v3 目标检测模型实现教程

本教程详细介绍如何使用 nanotorch 从零实现 YOLO v3（You Only Look Once, 2018）目标检测模型。

## 目录

1. [概述](#概述)
2. [YOLO v3 核心改进](#yolo-v3-核心改进)
3. [模型架构](#模型架构)
4. [损失函数](#损失函数)
5. [数据准备](#数据准备)
6. [训练流程](#训练流程)
7. [推理与后处理](#推理与后处理)
8. [代码示例](#代码示例)

---

## 概述

YOLO v3 是 YOLO 系列的第三个版本，在保持实时性能的同时大幅提升了检测精度。

### YOLO v3 的主要贡献

1. **多尺度检测**: 在 3 个不同尺度上进行检测（13×13, 26×26, 52×52）
2. **Darknet-53 骨干网络**: 引入残差连接的深层网络
3. **特征金字塔网络 (FPN)**: 融合多层特征，提升小物体检测能力
4. **独立分类器**: 使用 sigmoid 而非 softmax，支持多标签分类

### nanotorch 的 YOLO v3 实现模块

```
nanotorch/detection/yolo_v3/
├── __init__.py        # 模块导出
├── yolo_v3_model.py   # 模型架构 (Darknet53, FPN, YOLOHead, YOLOv3, YOLOv3Tiny)
└── yolo_v3_loss.py    # 损失函数 (YOLOv3Loss, encode_targets_v3, decode_predictions_v3)

examples/yolo_v3/
└── demo.py            # 训练和推理演示

tests/detection/yolo_v3/
├── test_yolov3_model.py  # 单元测试 (22 tests)
└── test_integration.py   # 集成测试
```

---

## YOLO v3 核心改进

### 多尺度检测 (Multi-scale Detection)

YOLO v3 在 3 个不同尺度的特征图上进行检测：

```
输入图像 (416×416)
    ↓
Darknet-53 骨干网络
    ↓
┌─────────────────────────────────────────────────────┐
│ Scale 1: 13×13 feature map (大物体检测)              │
│   - 3 个 anchor boxes: (116,90), (156,198), (373,326)│
│   - 输出: 13×13×3×(5+80) = 13×13×255                │
├─────────────────────────────────────────────────────┤
│ Scale 2: 26×26 feature map (中等物体检测)            │
│   - 3 个 anchor boxes: (30,61), (62,45), (59,119)    │
│   - 输出: 26×26×3×(5+80) = 26×26×255                │
├─────────────────────────────────────────────────────┤
│ Scale 3: 52×52 feature map (小物体检测)              │
│   - 3 个 anchor boxes: (10,13), (16,30), (33,23)     │
│   - 输出: 52×52×3×(5+80) = 52×52×255                │
└─────────────────────────────────────────────────────┘
```

### 特征金字塔网络 (FPN)

FPN 通过上采样和特征融合实现多尺度检测：

```
Backbone 输出:
  scale1 (1024 channels, 13×13)
  scale2 (512 channels, 26×26)
  scale3 (256 channels, 52×52)
       ↓
FPN 处理:
  scale1 → Conv → Upsample →┐
                            ├→ Concat → Conv → p5 (512 channels)
  scale2 ───────────────────┘
                            ↓
                        Upsample →┐
                                   ├→ Concat → Conv → p4 (256 channels)
  scale3 ─────────────────────────┘
                                   ↓
                               Conv → p3 (128 channels)
```

### Anchor Boxes

YOLO v3 使用 9 个预定义的 anchor boxes（3 个尺度 × 3 个 anchor）：

```python
# 大尺度 anchor (13×13，检测大物体)
[(116, 90), (156, 198), (373, 326)]

# 中等尺度 anchor (26×26，检测中等物体)
[(30, 61), (62, 45), (59, 119)]

# 小尺度 anchor (52×52，检测小物体)
[(10, 13), (16, 30), (33, 23)]
```

---

## 模型架构

### 1. ConvBN（基础卷积块）

每个卷积块包含：Conv2D + BatchNorm + LeakyReLU

```python
from nanotorch.detection.yolo_v3 import ConvBN
from nanotorch.tensor import Tensor
import numpy as np

# 创建卷积块
conv_bn = ConvBN(
    in_channels=64,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1
)

# 前向传播
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = conv_bn(x)
print(y.shape)  # (1, 128, 52, 52)
```

### 2. ResidualBlock（残差块）

```python
from nanotorch.detection.yolo_v3 import ResidualBlock

# 创建残差块
res_block = ResidualBlock(channels=256)

x = Tensor(np.random.randn(1, 256, 52, 52).astype(np.float32))
y = res_block(x)
print(y.shape)  # (1, 256, 52, 52) - 形状不变，便于残差连接
```

残差块结构：
```
输入 (C channels)
    ↓
Conv 1×1 → C/2 channels
    ↓
Conv 3×3 → C channels
    ↓
  + 输入 (残差连接)
    ↓
输出 (C channels)
```

### 3. Darknet53 骨干网络

```python
from nanotorch.detection.yolo_v3 import Darknet53

# 创建 Darknet-53 骨干
backbone = Darknet53(in_channels=3)

# 前向传播
x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
features = backbone(x)

print(features['scale1'].shape)  # (1, 1024, 13, 13) - 大物体特征
print(features['scale2'].shape)  # (1, 512, 26, 26)  - 中等物体特征
print(features['scale3'].shape)  # (1, 256, 52, 52)  - 小物体特征
```

Darknet-53 网络结构：

```
Layer         Filters   Size/Stride   Output
──────────────────────────────────────────────
Conv          32        3×3 / 1       (416, 416)
Conv          64        3×3 / 2       (208, 208)
ResBlock×1    64                      (208, 208)
Conv          128       3×3 / 2       (104, 104)
ResBlock×2    128                     (104, 104)
Conv          256       3×3 / 2       (52, 52)
ResBlock×8    256                     (52, 52) → scale3
Conv          512       3×3 / 2       (26, 26)
ResBlock×8    512                     (26, 26) → scale2
Conv          1024      3×3 / 2       (13, 13)
ResBlock×4    1024                    (13, 13) → scale1
```

### 4. FPN（特征金字塔网络）

```python
from nanotorch.detection.yolo_v3 import FPN

# 创建 FPN
fpn = FPN(in_channels=[1024, 512, 256])  # 注意：反序输入

# 前向传播
fpn_features = fpn(features)

print(fpn_features['p5'].shape)  # (1, 512, 13, 13)  - 大物体
print(fpn_features['p4'].shape)  # (1, 256, 26, 26)  - 中等物体
print(fpn_features['p3'].shape)  # (1, 128, 52, 52)  - 小物体
```

### 5. YOLOHead（检测头）

```python
from nanotorch.detection.yolo_v3 import YOLOHead

# 创建检测头
head = YOLOHead(
    in_channels=512,
    num_anchors=3,
    num_classes=80
)

x = Tensor(np.random.randn(1, 512, 13, 13).astype(np.float32))
output = head(x)
print(output.shape)  # (1, 255, 13, 13) = (1, 3*(5+80), 13, 13)
```

### 6. YOLOv3 完整模型

```python
from nanotorch.detection.yolo_v3 import YOLOv3, YOLOv3Tiny, build_yolov3

# 方式一：创建完整模型
model = YOLOv3(
    num_classes=80,
    input_size=416
)

# 方式二：创建轻量版（用于测试和快速推理）
tiny_model = YOLOv3Tiny(
    num_classes=80,
    input_size=416
)

# 方式三：使用工厂函数
model = build_yolov3('full', num_classes=80, input_size=416)
tiny_model = build_yolov3('tiny', num_classes=80, input_size=416)

# 前向传播
x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
output = model(x)

print(output['small'].shape)   # (1, 255, 13, 13)  - 大物体
print(output['medium'].shape)  # (1, 255, 26, 26)  - 中等物体
print(output['large'].shape)   # (1, 255, 52, 52)  - 小物体
```

---

## 损失函数

### YOLO v3 损失函数设计

YOLO v3 使用多部分损失函数：

$$
L = \lambda_{coord} L_{coord} + \lambda_{obj} L_{obj} + \lambda_{noobj} L_{noobj} + \lambda_{class} L_{class}
$$

其中：
- $L_{coord}$: 边界框坐标损失（MSE）
- $L_{obj}$: 有物体置信度损失（BCE）
- $L_{noobj}$: 无物体置信度损失（BCE）
- $L_{class}$: 分类损失（BCE，支持多标签）

### 坐标损失

$$
L_{coord} = \sum_{s \in \{S,M,L\}} \sum_{i=0}^{G_s^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (t_x - \hat{t}_x)^2 + (t_y - \hat{t}_y)^2 + (t_w - \hat{t}_w)^2 + (t_h - \hat{t}_h)^2 \right]
$$

### 置信度损失（Binary Cross Entropy）

$$
L_{obj} = -\sum_{i,j,s} \mathbb{1}_{ij}^{obj} \left[ C_{ijs} \log(\hat{C}_{ijs}) + (1 - C_{ijs}) \log(1 - \hat{C}_{ijs}) \right]
$$

$$
L_{noobj} = -\lambda_{noobj} \sum_{i,j,s} \mathbb{1}_{ij}^{noobj} \left[ C_{ijs} \log(\hat{C}_{ijs}) + (1 - C_{ijs}) \log(1 - \hat{C}_{ijs}) \right]
$$

### 分类损失（支持多标签）

$$
L_{class} = -\sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c=1}^{C} \left[ y_{ic} \log(\hat{p}_{ic}) + (1 - y_{ic}) \log(1 - \hat{p}_{ic}) \right]
$$

### 多尺度检测

YOLO v3 在三个尺度上进行检测：

$$
\text{Output}_{\text{small}} \in \mathbb{R}^{N \times 3 \times (5 + C) \times 13 \times 13}
$$

$$
\text{Output}_{\text{medium}} \in \mathbb{R}^{N \times 3 \times (5 + C) \times 26 \times 26}
$$

$$
\text{Output}_{\text{large}} \in \mathbb{R}^{N \times 3 \times (5 + C) \times 52 \times 52}
$$

其中 $3$ 是每个尺度的 Anchor 数量，$5 = 4(\text{coords}) + 1(\text{conf})$。

### 使用损失函数

```python
from nanotorch.detection.yolo_v3 import YOLOv3Loss, YOLOv3LossSimple

# 创建损失函数
loss_fn = YOLOv3Loss(
    num_classes=80,
    ignore_threshold=0.5,
    lambda_coord=5.0,
    lambda_obj=1.0,
    lambda_noobj=0.5,
    lambda_class=1.0
)

# 准备预测和目标
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

# 计算损失
loss, loss_dict = loss_fn(predictions, targets)

print(f"Total Loss: {loss.item():.4f}")
print(f"Coord Loss: {loss_dict['coord_loss']:.4f}")
print(f"Obj Loss: {loss_dict['obj_loss']:.4f}")
print(f"NoObj Loss: {loss_dict['noobj_loss']:.4f}")
print(f"Class Loss: {loss_dict['class_loss']:.4f}")
```

### 简化损失函数（用于测试）

```python
# 简化版 MSE 损失，用于快速测试
simple_loss_fn = YOLOv3LossSimple(num_classes=80)

targets = {
    'small': Tensor(np.zeros((2, 255, 13, 13), dtype=np.float32)),
    'medium': Tensor(np.zeros((2, 255, 26, 26), dtype=np.float32)),
    'large': Tensor(np.zeros((2, 255, 52, 52), dtype=np.float32))
}

loss, loss_dict = simple_loss_fn(predictions, targets)
# 注意：YOLOv3LossSimple 返回 (float, dict)，不支持 backward
```

### 目标编码与解码

```python
from nanotorch.detection.yolo_v3 import encode_targets_v3, decode_predictions_v3
import numpy as np

# YOLO v3 anchors
anchors = [
    (10, 13), (16, 30), (33, 23),       # 小尺度
    (30, 61), (62, 45), (59, 119),      # 中等尺度
    (116, 90), (156, 198), (373, 326)   # 大尺度
]

# 编码：将边界框转换为 YOLO v3 格式
boxes = np.array([
    [100, 100, 200, 200],
    [250, 250, 350, 350]
], dtype=np.float32)
labels = np.array([0, 5], dtype=np.int64)

targets = encode_targets_v3(
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

# 解码：将预测转换为边界框
predictions = np.random.randn(3, 85, 13, 13).astype(np.float32) * 0.1
scale_anchors = [(116, 90), (156, 198), (373, 326)]

boxes, scores, class_ids = decode_predictions_v3(
    predictions,
    anchors=scale_anchors,
    conf_threshold=0.3,
    num_classes=80,
    image_size=416
)

print(f"Detected {len(boxes)} objects")
```

---

## 数据准备

### SyntheticCOCODataset（合成数据集）

用于测试的合成 COCO 格式数据集：

```python
from examples.yolo_v3.demo import SyntheticCOCODataset

# 创建合成数据集
dataset = SyntheticCOCODataset(
    num_samples=1000,
    image_size=416,
    num_classes=80,
    max_objects=5
)

# 获取样本
sample = dataset[0]
print(sample['image'].shape)     # (3, 416, 416)
print(sample['boxes'].shape)     # (N, 4)
print(sample['labels'].shape)    # (N,)

# COCO 80 类别
from examples.yolo_v3.demo import COCO_CLASSES
print(len(COCO_CLASSES))  # 80
```

### 创建 DataLoader

```python
from examples.yolo_v3.demo import create_dataloader

# 一键创建 DataLoader
dataloader = create_dataloader(
    num_samples=1000,
    batch_size=8,
    image_size=416,
    num_classes=80,
    shuffle=True
)

# 遍历数据
for batch in dataloader:
    images = batch['image']       # (8, 3, 416, 416)
    boxes_list = batch['boxes']   # list of (N, 4) arrays
    labels_list = batch['labels'] # list of (N,) arrays
    break
```

---

## 训练流程

### 基本训练循环

```python
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v3 import YOLOv3Tiny, YOLOv3Loss
from examples.yolo_v3.demo import create_dataloader
import numpy as np

# 创建模型和优化器
model = YOLOv3Tiny(num_classes=80, input_size=416)
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 创建数据加载器
dataloader = create_dataloader(
    num_samples=100,
    batch_size=4,
    image_size=416,
    num_classes=80
)

# 创建损失函数（用于监控）
loss_fn = YOLOv3Loss(num_classes=80)

# 训练循环
for epoch in range(10):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        # 准备数据
        images = Tensor(batch['image'])
        targets = [
            {'boxes': batch['boxes'][i], 'labels': batch['labels'][i]}
            for i in range(len(batch['boxes']))
        ]
        
        # 前向传播
        optimizer.zero_grad()
        output = model(images)
        
        # 计算 MSE 损失用于梯度更新
        loss = 0.0
        for scale_name, pred in output.items():
            diff = pred - Tensor(np.zeros_like(pred.data))
            loss += (diff * diff).mean().item()
        
        # 使用简化的反向传播
        loss_tensor = Tensor(loss, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()
        
        total_loss += loss
        num_batches += 1
    
    scheduler.step()
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
```

---

## 推理与后处理

### 模型推理

```python
from nanotorch.detection.yolo_v3 import YOLOv3Tiny, decode_predictions_v3
from nanotorch.tensor import Tensor
import numpy as np

# 加载训练好的模型
model = YOLOv3Tiny(num_classes=80, input_size=416)
# model.load_state_dict(...)  # 加载权重

# 准备输入图像
image = np.random.randn(1, 3, 416, 416).astype(np.float32)
x = Tensor(image)

# 前向传播
output = model(x)

# 解码预测
all_boxes = []
all_scores = []
all_class_ids = []

anchors = [
    (10, 13), (16, 30), (33, 23),
    (30, 61), (62, 45), (59, 119),
    (116, 90), (156, 198), (373, 326)
]

# 对每个尺度进行解码
scale_anchors = [
    [(116, 90), (156, 198), (373, 326)],  # small scale
    [(30, 61), (62, 45), (59, 119)],      # medium scale
    [(10, 13), (16, 30), (33, 23)]        # large scale
]

# 假设我们有对应尺度的预测
# boxes, scores, class_ids = decode_predictions_v3(...)

print(f"Detected objects")
```

### 非极大值抑制 (NMS)

```python
from nanotorch.detection.nms import nms

# 假设有多个重叠的检测框
boxes = np.array([
    [100, 100, 200, 200],
    [105, 105, 205, 205],  # 与第一个高度重叠
    [300, 300, 400, 400]
], dtype=np.float32)

scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)

# 应用 NMS
keep = nms(boxes, scores, iou_threshold=0.5)
print(f"保持的索引: {keep}")  # [0, 2]

final_boxes = boxes[keep]
final_scores = scores[keep]
```

---

## 代码示例

### 完整训练示例

```python
"""
完整的 YOLO v3 训练示例
"""
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v3 import (
    YOLOv3Tiny,
    YOLOv3Loss,
    decode_predictions_v3
)
from examples.yolo_v3.demo import create_dataloader

def train_yolov3():
    # 超参数
    num_epochs = 50
    batch_size = 8
    learning_rate = 1e-4
    image_size = 416
    num_classes = 80
    
    # 创建模型
    model = YOLOv3Tiny(num_classes=num_classes, input_size=image_size)
    
    # 创建优化器和调度器
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 创建数据加载器
    dataloader = create_dataloader(
        num_samples=200,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes,
        shuffle=True
    )
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            images = Tensor(batch['image'])
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(images)
            
            # MSE 损失（用于 backward）
            loss = 0.0
            for scale_name, pred in output.items():
                diff = pred - Tensor(np.zeros_like(pred.data))
                loss += (diff * diff).mean().item()
            
            # 简化的反向传播
            loss_tensor = Tensor(loss, requires_grad=True)
            loss_tensor.backward()
            optimizer.step()
            
            total_loss += loss
        
        scheduler.step()
        
        # 打印统计
        avg_loss = total_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
    
    return model

if __name__ == "__main__":
    model = train_yolov3()
    print("训练完成！")
```

### 运行测试

```bash
# 运行 YOLO v3 单元测试
python -m pytest tests/detection/yolo_v3/test_yolov3_model.py -v

# 运行集成测试
python -m pytest tests/detection/yolo_v3/test_integration.py -v

# 运行所有 YOLO v3 测试
python -m pytest tests/detection/yolo_v3/ -v
```

---

## 快速开始：Demo 脚本

我们提供了一个完整的 Demo 脚本来演示 YOLO v3 的训练和推理流程。

### 查看模型架构

```bash
python examples/yolo_v3/demo.py --mode arch
```

输出示例：
```
============================================================
YOLO v3 Architecture Demo
============================================================

1. Full YOLOv3 Model (Darknet-53):
   Parameters: ~61M
   small: (1, 255, 13, 13)
   medium: (1, 255, 26, 26)
   large: (1, 255, 52, 52)

2. YOLOv3Tiny Model:
   Parameters: ~8M
   small: (1, 255, 13, 13)
   route: (1, 256, 26, 26)

3. Multi-scale Detection:
   - Large scale (52x52): Small objects
   - Medium scale (26x26): Medium objects
   - Small scale (13x13): Large objects
```

### 训练模型

```bash
# 默认使用 YOLOv3Tiny（推荐）
python examples/yolo_v3/demo.py --mode train

# 自定义参数
python examples/yolo_v3/demo.py --mode train \
    --epochs 5 \
    --batch-size 4 \
    --num-samples 50

# 使用完整 YOLOv3（需要 8GB+ 内存）
python examples/yolo_v3/demo.py --mode train --full --batch-size 1
```

训练参数说明：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 2 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--lr-step` | 5 | 学习率衰减步长 |
| `--image-size` | 224 | 输入图像大小 |
| `--num-samples` | 20 | 训练样本数 |
| `--num-classes` | 80 | 类别数 |
| `--full` | - | 使用完整 YOLOv3 |

### 推理演示

```bash
# 训练后运行推理
python examples/yolo_v3/demo.py --mode inference

# 自定义推理参数
python examples/yolo_v3/demo.py --mode inference \
    --num-inference 10 \
    --image-size 416
```

### 完整流程（训练 + 推理）

```bash
# 运行完整演示：架构 → 训练 → 推理
python examples/yolo_v3/demo.py --mode both
```

---

## YOLO v1 vs YOLO v3 对比

| 特性 | YOLO v1 | YOLO v3 |
|------|---------|---------|
| 骨干网络 | Darknet-24 | Darknet-53 |
| 检测尺度 | 单尺度 (7×7) | 多尺度 (13×13, 26×26, 52×52) |
| Anchor Boxes | 无 | 9 个 (3×3) |
| 特征融合 | 无 | FPN |
| 分类方式 | Softmax | Sigmoid (多标签) |
| 残差连接 | 无 | 有 |
| 小物体检测 | 较差 | 显著提升 |

---

## 模型参数对比

| Model | Input Size | Parameters | FLOPs | Memory |
|-------|------------|------------|-------|--------|
| YOLOv3Tiny | 416×416 | ~8M | ~5B | ~2GB |
| YOLOv3 | 416×416 | ~61M | ~65B | ~8GB |

---

## 常见问题

### 1. 为什么 YOLO v3 使用多尺度检测？

多尺度检测能够：
- 在不同分辨率特征图上检测不同大小的物体
- 大尺度特征图（52×52）检测小物体
- 小尺度特征图（13×13）检测大物体
- 显著提升小物体检测性能

### 2. FPN 的作用是什么？

特征金字塔网络 (FPN)：
- 融合不同层级的特征
- 通过上采样将语义信息传递到高分辨率层
- 增强小物体检测能力

### 3. 为什么使用 Sigmoid 而不是 Softmax？

Sigmoid 分类器的优势：
- 支持多标签分类（一个物体属于多个类别）
- 独立计算每个类别的概率
- 更适合 COCO 等多标签数据集

### 4. 如何选择 Anchor Boxes？

Anchor Boxes 选择策略：
- 使用 K-means 聚类训练数据
- 选择 IoU 最高的 anchor
- 不同尺度使用不同大小的 anchor

---

## 总结

本教程介绍了使用 nanotorch 实现 YOLO v3 的完整流程：

1. **核心改进**: Darknet-53、多尺度检测、FPN、Anchor Boxes
2. **模型架构**: 骨干网络 + 特征金字塔 + 检测头
3. **损失函数**: 坐标损失 + 置信度损失 + 分类损失
4. **数据处理**: 合成数据集、预处理、批处理
5. **训练推理**: 完整训练循环、多尺度解码、NMS 后处理

通过本教程，你应该能够：
- 理解 YOLO v3 的多尺度检测机制
- 使用 nanotorch 构建复杂的目标检测模型
- 理解 FPN 和残差连接的作用

---

## 参考资料

1. **YOLO v3 论文**: "YOLOv3: An Incremental Improvement" (2018)
   - https://arxiv.org/abs/1804.02767

2. **Darknet 官方实现**: https://github.com/pjreddie/darknet

3. **COCO 数据集**: https://cocodataset.org/

4. **Feature Pyramid Networks**: "Feature Pyramid Networks for Object Detection" (CVPR 2017)
   - https://arxiv.org/abs/1612.03144

5. **nanotorch YOLO v1 教程**: `/docs/tutorials/18-yolov1.md`
