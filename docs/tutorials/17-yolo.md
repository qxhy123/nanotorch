# YOLO v12 目标检测模型实现教程

## 想象你在玩"找不同"游戏...

一张复杂的图片摆在你面前：
- 你需要找出所有隐藏的物品
- 要说出每个物品是什么、在哪里
- 还要快速完成，不能慢吞吞

```
传统方法（两阶段）：
  先找可能的位置 → 再识别是什么
  就像：先圈出可疑区域 → 再仔细看圈里有什么
  问题：慢，要看好几遍

YOLO 方法（一阶段）：
  一眼看完整张图 → 同时说出所有物品
  就像：扫一眼就知道"猫在左下角，狗在右上角"
  优势：快，只看一遍
```

**YOLO 就是"一眼识别"** —— 看一次就能找出图中所有的物体。

---

## 目录

1. [概述](#概述)
2. [YOLO v12 架构创新](#yolo-v12-架构创新)
3. [核心组件实现](#核心组件实现)
4. [完整模型组装](#完整模型组装)
5. [训练流程](#训练流程)
6. [评估指标](#评估指标)
7. [代码示例](#代码示例)

---

## 概述

YOLO (You Only Look Once) 是目标检测领域的里程碑式工作，v12 版本引入了以注意力为核心的架构设计，显著提升了检测精度。

### nanotorch 的 YOLO 实现模块

```
nanotorch/detection/
├── bbox.py           # 边界框工具函数
├── iou.py            # IoU 变体 (IoU, GIoU, DIoU, CIoU, SIoU)
├── nms.py            # 非极大值抑制
├── layers.py         # YOLO 基础模块 (Conv, C2f, SPPF, Attention)
├── yolo_backbone.py  # R-ELAN 骨干网络
├── yolo_neck.py      # PANet/FPN 特征融合网络
├── yolo_head.py      # Anchor-free 检测头
└── losses.py         # 损失函数 (CIoU Loss, DFL Loss, VFL Loss)
```

---

## YOLO v12 架构创新

### 1. 以注意力为核心的架构

YOLO v12 摒弃了传统 YOLO 的纯 CNN 架构，引入 **Area Attention (A²)** 模块：

```
传统 CNN:  Local Receptive Field → 局部特征
Area Attention:  Global Receptive Field → 全局上下文
```

Area Attention 的核心思想是将特征图划分为多个区域，在每个区域内执行自注意力计算：

```
特征图 (H, W) → 划分为 4 个区域 (H/2, W/2) → 每个区域内计算 Self-Attention
```

复杂度从 $O((HW)^2)$ 降低到 $O(4 \times (HW/4)^2) = O((HW)^2/4)$

### 2. R-ELAN (Residual Efficient Layer Aggregation Network)

R-ELAN 是 YOLO v12 的骨干网络核心模块，具有以下特点：

- **高效特征聚合**: 通过多条路径聚合不同层级的特征
- **残差连接**: 在块级别添加残差连接，缩放因子 0.1
- **结构简化**: 相比 ELAN 更加简洁高效

```
输入 → Conv 1x1 → Split → [Conv 3x3] × n → Concat → Conv 1x1 → + 输入 × 0.1
```

### 3. Anchor-Free 检测头

YOLO v12 采用无锚框设计：

```
传统 Anchor-Based:
  预设锚框 → 预测偏移量 → 解码得到检测框

Anchor-Free:
  直接预测网格中心到四个边界的距离 → 解码得到检测框
```

优势：
- 减少超参数（无需设计锚框）
- 更好的泛化能力
- 简化训练流程

---

## 核心组件实现

### 边界框格式转换

```python
from nanotorch.detection.bbox import xyxy_to_xywh, xywh_to_xyxy

# [x1, y1, x2, y2] → [cx, cy, w, h]
boxes_xyxy = np.array([[10, 20, 50, 60]], dtype=np.float32)
boxes_xywh = xyxy_to_xywh(boxes_xyxy)
# 结果: [[30, 40, 40, 40]] (中心点 + 宽高)
```

### IoU 变体

```python
from nanotorch.detection.iou import iou, giou, diou, ciou

boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
boxes2 = np.array([[5, 5, 15, 15]], dtype=np.float32)

# 标准 IoU
iou_value = iou(boxes1, boxes2)  # 0.14

# GIoU (处理非重叠情况)
giou_value = giou(boxes1, boxes2)  # -0.19 ~ 1.0

# DIoU (考虑中心距离)
diou_value = diou(boxes1, boxes2)

# CIoU (加入宽高比一致性)
ciou_value = ciou(boxes1, boxes2)
```

### CIoU Loss 数学公式

$$
\text{CIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2} - \alpha v
$$

其中：
- $\rho^2$: 预测框与真实框中心点的欧氏距离平方
- $c^2$: 最小包围框对角线长度平方
- $v = \frac{4}{\pi^2}\left(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h}\right)^2$
- $\alpha = \frac{v}{1-\text{IoU}+v}$

### IoU 计算公式

$$
\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|A \cap B|}{|A \cup B|}
$$

对于两个边界框 $A = (x_1^A, y_1^A, x_2^A, y_2^A)$ 和 $B = (x_1^B, y_1^B, x_2^B, y_2^B)$：

$$
\text{IoU}(A, B) = \frac{\max(0, x_2^{\text{int}} - x_1^{\text{int}}) \times \max(0, y_2^{\text{int}} - y_1^{\text{int}})}{\text{Area}(A) + \text{Area}(B) - \text{Area}(A \cap B)}
$$

其中交集坐标：
- $x_1^{\text{int}} = \max(x_1^A, x_1^B)$
- $y_1^{\text{int}} = \max(y_1^A, y_1^B)$
- $x_2^{\text{int}} = \min(x_2^A, x_2^B)$
- $y_2^{\text{int}} = \min(y_2^A, y_2^B)$

### 坐标编码公式

YOLO 使用相对坐标而非绝对坐标：

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

其中：
- $(b_x, b_y, b_w, b_h)$: 预测框的中心坐标和宽高
- $(t_x, t_y, t_w, t_h)$: 网络输出
- $(c_x, c_y)$: 当前网格的左上角坐标
- $(p_w, p_h)$: Anchor 的宽高
- $\sigma$: Sigmoid 函数

### 置信度定义

$$
\text{Confidence} = \Pr(\text{Object}) \times \text{IoU}_{\text{pred}}^{\text{truth}}
$$

### NMS (非极大值抑制)

```python
from nanotorch.detection.nms import nms, batched_nms

boxes = np.array([
    [10, 10, 50, 50],
    [12, 12, 52, 52],  # 与第一个框高度重叠
    [100, 100, 150, 150]
], dtype=np.float32)

scores = np.array([0.9, 0.8, 0.85], dtype=np.float32)

# 标准 NMS
keep = nms(boxes, scores, iou_threshold=0.5)
# 结果: [0, 2] - 保留了最高分的重叠框

# 分类别 NMS (不同类别的框互不抑制)
class_ids = np.array([0, 0, 1], dtype=np.int64)
keep = batched_nms(boxes, scores, class_ids, iou_threshold=0.5)
```

### YOLO 基础模块

#### Conv 模块

YOLO 的基本卷积单元：Conv2D → BatchNorm → SiLU

```python
from nanotorch.detection.layers import Conv

conv = Conv(64, 128, kernel_size=3, stride=2)
x = Tensor(np.random.randn(1, 64, 32, 32).astype(np.float32))
y = conv(x)
# 输出: (1, 128, 16, 16) - stride=2 使空间尺寸减半
```

#### C2f 模块 (CSP Bottleneck)

```python
from nanotorch.detection.layers import C2f

c2f = C2f(128, 128, num_bottlenecks=3)
x = Tensor(np.random.randn(1, 128, 16, 16).astype(np.float32))
y = c2f(x)
# 输出: (1, 128, 16, 16)
```

C2f 结构：
```
输入 → Conv 1x1 → Split → [Bottleneck × n] → Concat → Conv 1x1 → 输出
                      ↓              ↑
                      └──────────────┘
```

#### SPPF (Spatial Pyramid Pooling - Fast)

```python
from nanotorch.detection.layers import SPPF

sppf = SPPF(256, 256, kernel_size=5)
x = Tensor(np.random.randn(1, 256, 8, 8).astype(np.float32))
y = sppf(x)
# 输出: (1, 256, 8, 8) - 空间尺寸不变，增加感受野
```

SPPF 等价于 SPP(k=5, 9, 13)，但使用 3 个连续的 MaxPool(k=5) 实现，更高效。

---

## 完整模型组装

### Backbone (骨干网络)

```python
from nanotorch.detection import build_backbone

# 构建不同规模的骨干网络
backbone_n = build_backbone(model_size='n')  # Nano - 最小
backbone_s = build_backbone(model_size='s')  # Small
backbone_m = build_backbone(model_size='m')  # Medium
backbone_l = build_backbone(model_size='l')  # Large
backbone_x = build_backbone(model_size='x')  # Extra Large

# 前向传播
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
features = backbone_n(x)

# 输出三个尺度的特征图
print(features['p3'].shape)  # (1, C3, 80, 80)  - stride 8
print(features['p4'].shape)  # (1, C4, 40, 40)  - stride 16
print(features['p5'].shape)  # (1, C5, 20, 20)  - stride 32
```

### Neck (特征融合网络)

```python
from nanotorch.detection import build_neck

# PANet (Path Aggregation Network)
neck = build_neck(
    neck_type='panet',
    in_channels=backbone.out_channels,
    num_blocks=3
)

# 前向传播
fused_features = neck(features)
```

PANet 结构：
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

### Detection Head (检测头)

```python
from nanotorch.detection import build_head

head = build_head(
    head_type='decoupled',
    in_channels=neck.out_channels,
    num_classes=80
)

# 前向传播
predictions = head(fused_features)

# 每个尺度的预测
for scale_name, (box_pred, cls_pred) in predictions.items():
    print(f"{scale_name}: box={box_pred.shape}, cls={cls_pred.shape}")
```

---

## 训练流程

### 损失函数

YOLO v12 使用组合损失：

```python
from nanotorch.detection.losses import YOLOLoss

loss_fn = YOLOLoss(
    num_classes=80,
    reg_max=16,
    box_weight=7.5,   # 边界框损失权重
    cls_weight=0.5,   # 分类损失权重
    dfl_weight=1.5    # DFL 损失权重
)

total_loss, loss_dict = loss_fn(predictions, targets, image_size)
```

组合损失公式：
$$
L_{total} = \lambda_{box} \cdot L_{box} + \lambda_{cls} \cdot L_{cls} + \lambda_{dfl} \cdot L_{dfl}
$$

### 数据增强

```python
from examples.yolo.data import (
    SyntheticDetectionDataset,
    MosaicAugmentation,
    RandomHorizontalFlip,
    LetterboxResize,
    create_yolo_dataloader
)

# 创建合成数据集（测试用）
dataset = SyntheticDetectionDataset(
    num_samples=1000,
    image_size=(640, 640),
    num_classes=10
)

# 创建 DataLoader
dataloader = create_yolo_dataloader(
    dataset,
    batch_size=16,
    image_size=640
)

# 数据增强
augmentations = [
    RandomHorizontalFlip(p=0.5),
    LetterboxResize(target_size=640)
]
```

### 训练脚本

```python
from examples.yolo.train import YOLOModel, Trainer

# 创建模型
model = YOLOModel(
    num_classes=10,
    model_size='s',
    use_attention=True
)

# 创建训练器
trainer = Trainer(
    model=model,
    train_loader=dataloader,
    num_classes=10,
    lr=1e-3
)

# 训练
for epoch in range(num_epochs):
    avg_loss = trainer.train_epoch(epoch)
    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
```

---

## 评估指标

### mAP (mean Average Precision)

```python
from examples.yolo.evaluate import DetectionMetrics, compute_ap

# 创建评估器
metrics = DetectionMetrics(num_classes=10)

# 更新预测结果
metrics.update(
    pred_boxes=pred_boxes,      # (N, 4)
    pred_scores=pred_scores,    # (N,)
    pred_labels=pred_labels,    # (N,)
    gt_boxes=gt_boxes,          # (M, 4)
    gt_labels=gt_labels         # (M,)
)

# 计算指标
results = metrics.compute()
print(f"mAP@0.5: {results['mAP50']:.4f}")
print(f"mAP@0.5:0.95: {results['mAP']:.4f}")
```

### AP 计算方法

使用 VOC 标准的 11 点插值法或 COCO 的全插值法：

```python
# 全插值法 AP
recalls = np.array([0.1, 0.2, 0.3, ..., 1.0])
precisions = np.array([1.0, 0.9, 0.8, ..., 0.5])
ap = compute_ap(recalls, precisions)
```

---

## 代码示例

### 完整推理示例

```python
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.detection import (
    build_backbone,
    build_neck,
    build_head,
    batched_nms
)

# 构建模型
backbone = build_backbone(model_size='s', use_attention=False)
neck = build_neck('panet', backbone.out_channels)
head = build_head('decoupled', neck.out_channels, num_classes=80)

# 准备输入
image = np.random.randn(1, 3, 640, 640).astype(np.float32)
x = Tensor(image)

# 前向传播
features = backbone(x)
features = neck(features)
predictions = head(features)

# 后处理
all_boxes = []
all_scores = []
all_labels = []

for scale, (box_pred, cls_pred) in predictions.items():
    # 解码预测...
    pass

# NMS
keep = batched_nms(all_boxes, all_scores, all_labels, iou_threshold=0.45)
final_boxes = all_boxes[keep]
final_scores = all_scores[keep]
final_labels = all_labels[keep]
```

### 运行测试

```bash
# 运行单元测试
python -m pytest tests/detection/ -v

# 运行训练示例
python examples/yolo/train.py

# 运行评估示例
python examples/yolo/evaluate.py
```

---

## 模型规模对比

| Model | Size (MB) | mAP@0.5:0.95 | FPS (V100) |
|-------|-----------|--------------|------------|
| YOLOv12-n | 5.2 | 37.8 | 120 |
| YOLOv12-s | 15.4 | 45.2 | 95 |
| YOLOv12-m | 32.1 | 51.5 | 65 |
| YOLOv12-l | 52.3 | 54.8 | 45 |
| YOLOv12-x | 78.9 | 56.4 | 32 |

---

## 总结

本教程介绍了使用 nanotorch 实现 YOLO v12 的完整流程：

1. **核心组件**: 边界框处理、IoU 计算、NMS
2. **网络模块**: Conv、C2f、SPPF、Area Attention
3. **模型架构**: R-ELAN Backbone + PANet Neck + Anchor-Free Head
4. **训练流程**: 损失函数、数据增强、优化策略
5. **评估指标**: mAP 计算、精度-召回曲线

通过本教程，你应该能够：
- 理解 YOLO v12 的架构设计
- 使用 nanotorch 构建目标检测模型
- 自定义训练和评估流程

---

## 参考资料

1. YOLO v12 Paper: Attention-Centric Object Detection
2. CIoU Paper: https://arxiv.org/abs/1911.08287
3. DFL Paper: https://arxiv.org/abs/2006.04388
4. nanotorch Documentation: `/docs`
