# YOLO v8 目标检测模型实现教程

## 多年来，锚框像一道无形的枷锁...

它告诉模型："目标应该长这样，大小应该差不多。"

这很好用，却也带来束缚——每换一个数据集，都要重新设计锚框。小物体多的场景？调锚框。长宽比特殊的物体？调锚框。调来调去，永无止境。

**能不能，彻底挣脱这道枷锁？**

YOLO v8 说：可以。

它抛弃了锚框，选择了一条更自由的路——**Anchor-Free**。不再预设任何框，直接预测目标的中心点和宽高。

这就像从"填空题"变成了"简答题"——不再受限于给定的选项，而是自由作答。

```
有锚框：
  "目标应该长这样"
  猜测、试探、调整
  永远在追赶数据的步伐

无锚框：
  "目标在哪里？多大？"
  直接回答，干净利落
  一套参数，走遍天下
```

**YOLO v8 —— 挣脱枷锁的自由**，不再猜，直接答。

---

## 目录

1. [概述](#概述)
2. [YOLO v8 核心改进](#yolo-v8-核心改进)
3. [模型架构](#模型架构)
4. [损失函数](#损失函数)
5. [训练流程](#训练流程)
6. [代码示例](#代码示例)
7. [常见问题](#常见问题)
8. [总结](#总结)

---

## 概述

YOLO v8 是 Ultralytics 发布的最新 YOLO 版本，于 2023 年推出。它在 YOLO v5 的基础上进行了多项改进，特别是在检测头和损失函数方面。

### YOLO v8 的主要特点

1. **C2f 模块**: 更快的 CSP Bottleneck 结构
2. **Anchor-free 检测**: 无需预设锚框
3. **解耦头 (Decoupled Head)**: 分类和回归分离
4. **DFL 损失**: Distribution Focal Loss
5. **多尺寸模型**: n(nano), s(small), m(medium), l(large), x(xlarge)
6. **Mosaic 增强**: 训练时数据增强

### nanotorch 的 YOLO v8 实现模块

```
nanotorch/detection/yolo_v8/
├── __init__.py        # 模块导出
├── yolo_v8_model.py   # 模型架构 (C2f, Bottleneck, Backbone, Neck, DetectHead)
└── yolo_v8_loss.py    # 损失函数 (YOLOv8Loss, encode_targets_v8, decode_predictions_v8)

examples/yolo_v8/
└── demo.py            # 训练和推理演示

tests/detection/yolo_v8/
├── test_yolov8_model.py  # 单元测试
└── test_v8_integration.py  # 集成测试
```

---

## YOLO v8 核心改进

### C2f 模块 (CSP Bottleneck with 2 convolutions)

C2f 是 YOLO v8 的核心构建块，比 YOLO v5 的 C3 更高效：

```
输入特征 (c_in)
    │
    └─── Conv1x1 → c_out/2 ──┬──→ Bottleneck×N ──┐
                             │                    │
                             └────────────────────┤
                                                  │
                        Concat(x, x1) ←───────────┘
                              │
                         Conv1x1 → 输出 (c_out)
```

C2f 相比 C3 的改进：
- **更少的参数**: 使用 2 个卷积而非 3 个
- **更好的梯度流**: 分支结构更简洁
- **更高的效率**: 计算量减少

### Anchor-free 检测

YOLO v8 采用无锚框检测策略：

| 特性 | 有锚框 (YOLO v5) | 无锚框 (YOLO v8) |
|------|------------------|------------------|
| 预测内容 | tx, ty, tw, th + offset | 直接预测中心点和尺寸 |
| 超参数 | 需要设置锚框尺寸 | 无需预设 |
| 泛化能力 | 依赖数据分布 | 更好的泛化 |

### 解耦头 (Decoupled Head)

```
特征图
    │
    ├──→ 分类分支 → Conv → Conv → 类别分数
    │
    └──→ 回归分支 → Conv → Conv → 边界框 + DFL
```

解耦头的优势：
- **独立优化**: 分类和回归各自优化
- **更好的特征**: 专门的特征提取
- **更高精度**: 减少任务冲突

---

## 模型架构

### 1. ConvBN（基础卷积块）

```python
from nanotorch.detection.yolo_v8 import ConvBN
from nanotorch.tensor import Tensor
import numpy as np

conv_bn = ConvBN(64, 128, k=3, s=1)
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = conv_bn(x)
print(y.shape)  # (1, 128, 52, 52)
```

### 2. Bottleneck 模块

```python
from nanotorch.detection.yolo_v8 import Bottleneck

bottleneck = Bottleneck(128, shortcut=True)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = bottleneck(x)
print(y.shape)  # (1, 128, 52, 52)
```

Bottleneck 结构：
```
输入 (c)
    │
    ├──→ Conv1x1 → c/2 → Conv3x3 → c
    │                        │
    └────────────────────────┴──→ Add → 输出 (c)
```

### 3. C2f 模块

```python
from nanotorch.detection.yolo_v8 import C2f

c2f = C2f(128, 256, n=2, shortcut=True)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = c2f(x)
print(y.shape)  # (1, 256, 52, 52)
```

### 4. Backbone 网络

```python
from nanotorch.detection.yolo_v8 import Backbone

backbone = Backbone(in_ch=3)
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
features = backbone(x)

print(features['scale1'].shape)  # (1, 512, 20, 20) - stride=32
print(features['scale2'].shape)  # (1, 256, 40, 40) - stride=16
print(features['scale3'].shape)  # (1, 128, 80, 80) - stride=8
```

Backbone 结构：
```
Input (3×640×640)
    ↓ Stem: Conv3×3(s2) → 32 channels
    ↓ Stage1: Conv3×3(s2) → C2f → 64 channels
    ↓ Stage2: Conv3×3(s2) → C2f → 128 channels → 输出 s3 (stride=8)
    ↓ Stage3: Conv3×3(s2) → C2f → 256 channels → 输出 s2 (stride=16)
    ↓ Stage4: Conv3×3(s2) → C2f → 512 channels → 输出 s1 (stride=32)
```

### 5. Neck 网络（PANet）

```python
from nanotorch.detection.yolo_v8 import Neck

neck = Neck(channels=[128, 256, 512])
neck_out = neck(features)

print(neck_out['p3'].shape)  # (1, 128, 80, 80)
print(neck_out['p4'].shape)  # (1, 256, 40, 40)
print(neck_out['p5'].shape)  # (1, 512, 20, 20)
```

### 6. DetectHead（解耦检测头）

```python
from nanotorch.detection.yolo_v8 import DetectHead

head = DetectHead(256, num_classes=80)
x = Tensor(np.random.randn(1, 256, 40, 40).astype(np.float32))
y = head(x)
print(y.shape)  # (1, 85, 40, 40)
```

### 7. 完整 YOLOv8 模型

```python
from nanotorch.detection.yolo_v8 import YOLOv8, build_yolov8

# 方式一：直接创建
model = YOLOv8(num_classes=80, input_size=640)

# 方式二：使用工厂函数
model = build_yolov8(num_classes=80, input_size=640)

# 前向传播
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
output = model(x)

print(output['small'].shape)   # (1, 85, 20, 20) - 大物体检测
print(output['medium'].shape)  # (1, 85, 40, 40) - 中等物体检测
print(output['large'].shape)   # (1, 85, 80, 80) - 小物体检测
```

---

## 损失函数

### YOLO v8 损失函数设计

YOLO v8 的损失函数包含三个部分：

$$
L = \lambda_{box} L_{box} + \lambda_{cls} L_{cls} + \lambda_{dfl} L_{dfl}
$$

### 边界框损失（CIoU）

$$
L_{box} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$

### 分类损失（BCE）

$$
L_{cls} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left[ y_{ic} \log(\hat{p}_{ic}) + (1 - y_{ic}) \log(1 - \hat{p}_{ic}) \right]
$$

### DFL (Distribution Focal Loss)

DFL 将边界框回归建模为分布问题：

**分布建模**：

$$
\hat{y} = \sum_{n=0}^{N-1} P(n) \cdot n, \quad N = \text{reg\_max}
$$

其中 $P(n)$ 是预测的概率分布。

**DFL 损失**：

$$
L_{dfl} = -\sum_{n=0}^{N-1} \left[ y_n \log(\hat{P}(n)) \right]
$$

其中：
- 传统方法: 直接预测 $(x, y, w, h)$
- DFL 方法: 预测分布，取期望值

**目标分布构建**：

对于目标值 $y \in [l, l+1]$，其中 $l = \lfloor y \rfloor$：

$$
y_l = l + 1 - y, \quad y_{l+1} = y - l
$$

### C2f 模块

C2f 模块的特征流：

$$
x_{out} = \text{Concat}(x_{split}, \text{BottleNeck}_1(x_{split}), \ldots, \text{BottleNeck}_k(x_{split}))
$$

### YOLOv8Loss

YOLO v8 的损失函数包含三个部分：

```python
from nanotorch.detection.yolo_v8 import YOLOv8Loss

loss_fn = YOLOv8Loss(
    num_classes=80,
    reg_max=16,         # DFL 最大值
    lambda_box=7.5,     # 边界框损失权重
    lambda_cls=0.5,     # 分类损失权重
    lambda_dfl=1.5      # DFL 损失权重
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
print(f"Cls Loss: {loss_dict['cls_loss']:.4f}")
print(f"DFL Loss: {loss_dict['dfl_loss']:.4f}")
```

### DFL (Distribution Focal Loss)

DFL 将边界框回归建模为分布问题：
- 传统方法: 直接预测 (x, y, w, h)
- DFL 方法: 预测分布，取期望值

```
边界框偏移 = Σ(P(n) * n), n ∈ [0, reg_max]
```

### 编码和解码函数

```python
from nanotorch.detection.yolo_v8 import encode_targets_v8, decode_predictions_v8

# 编码目标
boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
labels = np.array([0], dtype=np.int64)
targets = encode_targets_v8(boxes, labels, grid_sizes=[80, 40, 20])

# 解码预测
boxes, scores, class_ids = decode_predictions_v8(
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
from nanotorch.optim import AdamW
from nanotorch.optim.lr_scheduler import CosineWarmupScheduler
from nanotorch.detection.yolo_v8 import build_yolov8, YOLOv8Loss
import numpy as np

# 创建模型
model = build_yolov8(num_classes=80, input_size=640)

# 损失函数和优化器
loss_fn = YOLOv8Loss(num_classes=80)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=3, max_epochs=50)

# 训练循环
for epoch in range(50):
    model.train()
    total_loss = 0
    
    for batch_idx in range(100):
        # 生成随机数据
        images = Tensor(np.random.randn(4, 3, 640, 640).astype(np.float32))
        targets = [{
            'boxes': np.random.rand(3, 4).astype(np.float32) * 600,
            'labels': np.random.randint(0, 80, 3).astype(np.int64)
        } for _ in range(4)]
        
        optimizer.zero_grad()
        output = model(images)
        
        # 计算损失
        loss, loss_dict = loss_fn(output, targets)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    avg_loss = total_loss / 100
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
```

---

## 代码示例

### 运行 Demo

```bash
# 查看模型架构
python examples/yolo_v8/demo.py --mode arch

# 训练模型
python examples/yolo_v8/demo.py --mode train --epochs 5

# 推理演示
python examples/yolo_v8/demo.py --mode inference

# 完整流程
python examples/yolo_v8/demo.py --mode both
```

### Demo 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | arch | 运行模式 (arch/train/inference/both) |
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 2 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--image-size` | 640 | 输入图像大小 |
| `--num-classes` | 80 | 类别数 |

### 运行测试

```bash
# 单元测试
python -m pytest tests/detection/yolo_v8/test_yolov8_model.py -v

# 集成测试
python -m pytest tests/detection/yolo_v8/test_v8_integration.py -v

# 所有 v8 测试
python -m pytest tests/detection/yolo_v8/ -v
```

---

## YOLO v7 vs YOLO v8 对比

| 特性 | YOLO v7 | YOLO v8 |
|------|---------|---------|
| 开发者 | WongKinYiu | Ultralytics |
| 核心模块 | E-ELAN | C2f |
| 检测头 | 耦合 | 解耦 |
| 锚框 | 有锚框 | Anchor-free |
| 损失函数 | CIoU + BCE | DFL + BCE + CIoU |
| 易用性 | 中等 | 高 |

---

## 常见问题

### 1. C2f 和 C3 有什么区别？

| 特性 | C3 | C2f |
|------|-----|-----|
| 卷积数 | 3 | 2 |
| 分支结构 | 固定 | 灵活 |
| 参数量 | 较多 | 较少 |
| 计算效率 | 中等 | 更高 |

### 2. 为什么使用 Anchor-free？

- **更简单**: 无需设计锚框
- **更通用**: 不依赖数据分布
- **更准确**: 减少超参数调优

### 3. DFL 损失的优势？

- **更稳定**: 分布建模更鲁棒
- **更精确**: 边界框回归精度更高
- **端到端**: 无需后处理

---

## 总结

本教程介绍了使用 nanotorch 实现 YOLO v8 的完整流程：

1. **核心改进**: C2f 模块、Anchor-free 检测、解耦头
2. **模型架构**: Backbone + Neck (PANet) + Decoupled Head
3. **损失函数**: DFL + BCE + CIoU
4. **训练推理**: 完整的训练和推理流程

YOLO v8 是目前最流行的目标检测模型之一，其易用性和性能都达到了很高水平。

---

## 参考资料

1. **Ultralytics YOLOv8**: https://github.com/ultralytics/ultralytics

2. **YOLOv8 文档**: https://docs.ultralytics.com/

3. **nanotorch YOLO v7 教程**: `/docs/tutorials/23-yolov7.md`

4. **nanotorch YOLO v9 教程**: `/docs/tutorials/25-yolov9.md`
