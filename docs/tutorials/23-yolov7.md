# YOLO v7 目标检测模型实现教程

本教程详细介绍如何使用 nanotorch 从零实现 YOLO v7（Trainable Bag-of-Freebies, 2022）目标检测模型。

## 目录

1. [概述](#概述)
2. [YOLO v7 核心改进](#yolo-v7-核心改进)
3. [模型架构](#模型架构)
4. [损失函数](#损失函数)
5. [训练流程](#训练流程)
6. [代码示例](#代码示例)
7. [常见问题](#常见问题)
8. [总结](#总结)

---

## 概述

YOLO v7 由 WongKinYiu 开发，是 YOLO 系列的重要里程碑。它通过"可训练的免费赠品"（Trainable Bag-of-Freebies）策略，在不增加推理成本的情况下显著提升精度。

### YOLO v7 的主要特点

1. **E-ELAN (Extended Efficient Layer Aggregation Network)**: 扩展的高效层聚合网络
2. **模型缩放技术**: 复合缩放策略，同时调整深度和宽度
3. **辅助训练头**: 训练时使用辅助头，推理时移除
4. **卷积重参数化**: 训练时多分支，推理时融合为单分支
5. **YOLOv7-W6/E6/D6**: 多种尺寸变体

### nanotorch 的 YOLO v7 实现模块

```
nanotorch/detection/yolo_v7/
├── __init__.py        # 模块导出
├── yolo_v7_model.py   # 模型架构 (ELAN, Backbone, Neck, Head, YOLOv7)
└── yolo_v7_loss.py    # 损失函数 (YOLOv7Loss, encode_targets_v7, decode_predictions_v7)

examples/yolo_v7/
└── demo.py            # 训练和推理演示

tests/detection/yolo_v7/
├── test_yolov7_model.py  # 单元测试
└── test_v7_integration.py  # 集成测试
```

---

## YOLO v7 核心改进

### E-ELAN (Extended Efficient Layer Aggregation Network)

E-ELAN 是 YOLO v7 的核心构建块，通过高效的层聚合实现更强的特征表示：

```
输入特征 (c_in)
    │
    ├─── Conv1x1 → mid_channels ────────────────────┐
    │                                                │
    └─── Conv1x1 → mid_channels → Conv3x3×N ────────┤
                                                     │
                        Concat(y1, y2) ←─────────────┘
                              │
                         Conv1x1 → 输出 (c_out)
```

与普通 ELAN 相比，E-ELAN 的优势：
- **梯度流优化**: 更好的梯度传播路径
- **特征重用**: 中间特征被有效利用
- **参数效率**: 相同参数量下更强的表达能力

### 辅助训练头 (Auxiliary Head)

YOLO v7 在训练时使用辅助检测头：
- **Lead Head**: 主检测头，用于最终预测
- **Auxiliary Head**: 辅助头，帮助深层特征学习

```
Backbone Feature
       │
       ├──→ Auxiliary Head → Aux Loss
       │
       └──→ Neck → Lead Head → Main Loss
```

### 重参数化卷积 (RepConv)

训练和推理使用不同的结构：
- **训练时**: 3×3 Conv + 1×1 Conv + Identity（三个分支）
- **推理时**: 融合为单个 3×3 Conv（结构重参数化）

---

## 模型架构

### 1. ConvBN（基础卷积块）

ConvBN 是 YOLO v7 的基础构建单元：

```python
from nanotorch.detection.yolo_v7 import ConvBN
from nanotorch.tensor import Tensor
import numpy as np

conv_bn = ConvBN(64, 128, k=3, s=1)
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = conv_bn(x)
print(y.shape)  # (1, 128, 52, 52)
```

ConvBN 包含三个组件：
1. **Conv2D**: 2D 卷积层
2. **BatchNorm2d**: 批归一化层
3. **SiLU**: 平滑激活函数

### 2. ELAN 模块

```python
from nanotorch.detection.yolo_v7 import ELAN

elan = ELAN(128, 256, n=2)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = elan(x)
print(y.shape)  # (1, 256, 52, 52)
```

ELAN 参数说明：
- `c_in`: 输入通道数
- `c_out`: 输出通道数
- `n`: Bottleneck 块的数量

### 3. Backbone 网络

```python
from nanotorch.detection.yolo_v7 import Backbone

backbone = Backbone(in_ch=3)
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
features = backbone(x)

print(features['scale1'].shape)  # (1, 512, 20, 20) - 大物体特征
print(features['scale2'].shape)  # (1, 256, 40, 40) - 中等物体特征
print(features['scale3'].shape)  # (1, 128, 80, 80) - 小物体特征
```

Backbone 结构：
```
Input (3×640×640)
    ↓ Stem: Conv3×3(s2) → Conv3×3(s2)
    ↓ Stage1: ELAN(64→64)
    ↓ Down1: Conv3×3(s2)
    ↓ Stage2: ELAN(128→128) → 输出 s3 (stride=8)
    ↓ Down2: Conv3×3(s2)
    ↓ Stage3: ELAN(256→256) → 输出 s2 (stride=16)
    ↓ Down3: Conv3×3(s2)
    ↓ Stage4: ELAN(512→512) → 输出 s1 (stride=32)
```

### 4. Neck 网络（PANet）

```python
from nanotorch.detection.yolo_v7 import Neck

neck = Neck(channels=[128, 256, 512])
# 使用 backbone 输出的特征
neck_out = neck(features)
print(neck_out['p3'].shape)  # (1, 128, 80, 80)
print(neck_out['p4'].shape)  # (1, 256, 40, 40)
print(neck_out['p5'].shape)  # (1, 512, 20, 20)
```

Neck 采用 PANet（Path Aggregation Network）结构：
- **自顶向下路径**: 上采样 + 融合
- **自底向上路径**: 下采样 + 融合

### 5. 完整 YOLOv7 模型

```python
from nanotorch.detection.yolo_v7 import YOLOv7, build_yolov7

# 方式一：直接创建
model = YOLOv7(num_classes=80, input_size=640)

# 方式二：使用工厂函数
model = build_yolov7(num_classes=80, input_size=640)

# 前向传播
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
output = model(x)

print(output['small'].shape)   # (1, 85, 20, 20) - 大物体检测
print(output['medium'].shape)  # (1, 85, 40, 40) - 中等物体检测
print(output['large'].shape)   # (1, 85, 80, 80) - 小物体检测
```

输出格式说明：
- 85 = 4 (bbox: tx, ty, tw, th) + 1 (confidence) + 80 (classes)

---

## 损失函数

### YOLO v7 损失函数设计

YOLO v7 的损失函数包含四个部分：

$$
L = \lambda_{box} L_{box} + \lambda_{obj} L_{obj} + \lambda_{noobj} L_{noobj} + \lambda_{class} L_{class}
$$

### 边界框损失（CIoU）

$$
L_{box} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$

其中：
- $\rho$: 预测框与真实框中心点的欧氏距离
- $c$: 最小包围框的对角线长度
- $v = \frac{4}{\pi^2}(\arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h})^2$
- $\alpha = \frac{v}{1-\text{IoU}+v}$

### 置信度损失（BCE）

$$
L_{obj} = -\frac{1}{N_{pos}} \sum_{i \in pos} \left[ \hat{C}_i \log(C_i) + (1 - \hat{C}_i) \log(1 - C_i) \right]
$$

$$
L_{noobj} = -\frac{\lambda_{noobj}}{N_{neg}} \sum_{i \in neg} \left[ \hat{C}_i \log(C_i) + (1 - \hat{C}_i) \log(1 - C_i) \right]
$$

### 分类损失（BCE）

$$
L_{class} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left[ y_{ic} \log(\hat{p}_{ic}) + (1 - y_{ic}) \log(1 - \hat{p}_{ic}) \right]
$$

### E-ELAN 架构

E-ELAN (Extended Efficient Layer Aggregation Network) 的特征融合：

$$
\text{Output} = \text{Concat}(\text{Branch}_1, \text{Branch}_2, \ldots, \text{Branch}_k)
$$

### RepConv 重参数化

训练时的复合卷积：

$$
y = \text{Conv}_{3 \times 3}(x) + \text{Conv}_{1 \times 1}(x) + \text{BN}(x)
$$

推理时融合为单个卷积：

$$
y = \text{Conv}_{fused}(x)
$$

### YOLOv7Loss

```python
from nanotorch.detection.yolo_v7 import YOLOv7Loss

loss_fn = YOLOv7Loss(
    num_classes=80,
    lambda_box=5.0,      # 边界框损失权重
    lambda_obj=1.0,      # 目标置信度损失权重
    lambda_noobj=0.5,    # 非目标置信度损失权重
    lambda_class=1.0     # 分类损失权重
)

# 计算损失
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

### 编码和解码函数

```python
from nanotorch.detection.yolo_v7 import encode_targets_v7, decode_predictions_v7

# 编码目标
boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
labels = np.array([0], dtype=np.int64)
targets = encode_targets_v7(boxes, labels, grid_sizes=[80, 40, 20])

# 解码预测
boxes, scores, class_ids = decode_predictions_v7(
    predictions=output['large'],
    conf_threshold=0.25,
    num_classes=80
)
```

---

## 训练流程

### 完整训练示例

```python
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import CosineAnnealingLR
from nanotorch.detection.yolo_v7 import build_yolov7, YOLOv7Loss
import numpy as np

# 创建模型
model = build_yolov7(num_classes=80, input_size=224)

# 损失函数和优化器
loss_fn = YOLOv7Loss(num_classes=80)
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# 训练循环
for epoch in range(10):
    model.train()
    total_loss = 0
    
    for batch_idx in range(10):
        # 生成随机数据
        images = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        targets = [{
            'boxes': np.random.rand(2, 4).astype(np.float32) * 200,
            'labels': np.random.randint(0, 80, 2).astype(np.int64)
        }]
        
        optimizer.zero_grad()
        output = model(images)
        
        # 计算损失
        loss, loss_dict = loss_fn(output, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / 10
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
```

---

## 代码示例

### 运行 Demo

```bash
# 查看模型架构
python examples/yolo_v7/demo.py --mode arch

# 训练模型
python examples/yolo_v7/demo.py --mode train --epochs 5

# 推理演示
python examples/yolo_v7/demo.py --mode inference

# 完整流程（训练 + 推理）
python examples/yolo_v7/demo.py --mode both
```

### Demo 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | arch | 运行模式 (arch/train/inference/both) |
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 2 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--image-size` | 224 | 输入图像大小 |
| `--num-classes` | 80 | 类别数 |

### 运行测试

```bash
# 单元测试
python -m pytest tests/detection/yolo_v7/test_yolov7_model.py -v

# 集成测试
python -m pytest tests/detection/yolo_v7/test_v7_integration.py -v

# 所有 v7 测试
python -m pytest tests/detection/yolo_v7/ -v
```

---

## YOLO v6 vs YOLO v7 对比

| 特性 | YOLO v6 | YOLO v7 |
|------|---------|---------|
| 开发者 | 美团 | WongKinYiu |
| 核心模块 | RepVGG Block | E-ELAN |
| 训练策略 | 标准训练 | 辅助训练头 |
| 模型缩放 | 固定 | 复合缩放 |
| 重参数化 | 骨干网络 | 可选 |

---

## 常见问题

### 1. E-ELAN 和 C3 模块有什么区别？

| 特性 | C3 (YOLOv5) | E-ELAN (YOLOv7) |
|------|-------------|-----------------|
| 结构 | CSP + Bottleneck | 双分支 + 多级聚合 |
| 梯度流 | 一般 | 更优 |
| 参数效率 | 中等 | 更高 |

### 2. 辅助训练头如何工作？

辅助头在训练时提供额外的监督信号：
- 帮助浅层特征学习
- 不影响推理速度（推理时移除）
- 提升约 1-2% 的精度

### 3. 如何选择模型尺寸？

- **YOLOv7-Tiny**: 边缘设备，极致速度
- **YOLOv7**: 标准版本，平衡速度和精度
- **YOLOv7-W6**: 更大模型，更高精度
- **YOLOv7-E6/D6**: 最高精度需求

---

## 总结

本教程介绍了使用 nanotorch 实现 YOLO v7 的完整流程：

1. **核心改进**: E-ELAN、辅助训练头、模型缩放
2. **模型架构**: Backbone + Neck (PANet) + Head
3. **损失函数**: 边界框损失 + 置信度损失 + 分类损失
4. **训练推理**: 完整的训练和推理流程

YOLO v7 通过"免费赠品"策略，在不增加推理成本的情况下实现了 SOTA 性能，是工业部署的优秀选择。

---

## 参考资料

1. **YOLOv7 论文**: "YOLOv7: Trainable Bag-of-Freebies Sets New State-of-the-Art for Real-Time Object Detectors"

2. **GitHub 仓库**: https://github.com/WongKinYiu/yolov7

3. **nanotorch YOLO v6 教程**: `/docs/tutorials/22-yolov6.md`

4. **nanotorch YOLO v8 教程**: `/docs/tutorials/24-yolov8.md`
