# YOLO v5 目标检测模型实现教程

本教程详细介绍如何使用 nanotorch 从零实现 YOLO v5（Ultralytics YOLO, 2020）目标检测模型。

## 目录

1. [概述](#概述)
2. [YOLO v5 核心改进](#yolo-v5-核心改进)
3. [模型架构](#模型架构)
4. [损失函数](#损失函数)
5. [训练流程](#训练流程)
6. [代码示例](#代码示例)

---

## 概述

YOLO v5 是 Ultralytics 发布的 YOLO 系列，以易用性和工程优化著称。

### YOLO v5 的主要特点

1. **C3 模块**: CSP Bottleneck with 3 convolutions
2. **SPPF**: 快速空间金字塔池化
3. **SiLU 激活函数**: 平滑的激活函数
4. **多尺寸模型**: n(nano), s(small), m(medium), l(large), x(xlarge)
5. **PANet Neck**: 路径聚合网络
6. **Auto Anchor**: 自动锚框计算

### nanotorch 的 YOLO v5 实现模块

```
nanotorch/detection/yolo_v5/
├── __init__.py        # 模块导出
├── yolo_v5_model.py   # 模型架构 (C3, SPPF, Backbone, Neck, YOLOv5)
└── yolo_v5_loss.py    # 损失函数 (YOLOv5Loss, encode_targets_v5, decode_predictions_v5)

examples/yolo_v5/
└── demo.py            # 训练和推理演示

tests/detection/yolo_v5/
├── test_yolov5_model.py  # 单元测试
└── test_integration.py   # 集成测试
```

---

## YOLO v5 核心改进

### C3 模块 (CSP Bottleneck with 3 convolutions)

C3 是 YOLOv5 的核心构建块：

```
输入特征
    ├─── Conv1x1 → mid_channels → Bottleneck×N →┐
    │                                            ├→ Concat → Conv1x1 → 输出
    └─── Conv1x1 → mid_channels ────────────────┘
```

### SPPF (Spatial Pyramid Pooling Fast)

SPPF 通过连续池化实现多尺度特征：

```
输入特征
    ↓
Conv 1x1
    ↓
┌─────────────────────────────────┐
│  y1 = 原始特征                  │
│  y2 = MaxPool(y1)              │
│  y3 = MaxPool(y2)              │
│  y4 = MaxPool(y3)              │
└─────────────────────────────────┘
    ↓
Concat(y1, y2, y3, y4)
    ↓
Conv 1x1
```

### 多尺寸模型

YOLOv5 提供五种尺寸：

| 版本 | depth_multiple | width_multiple | 参数量 |
|------|----------------|----------------|--------|
| Nano (n) | 0.33 | 0.25 | ~1.9M |
| Small (s) | 0.33 | 0.50 | ~7.2M |
| Medium (m) | 0.67 | 0.75 | ~21M |
| Large (l) | 1.00 | 1.00 | ~46M |
| XLarge (x) | 1.33 | 1.25 | ~86M |

---

## 模型架构

### 1. ConvBN（基础卷积块）

```python
from nanotorch.detection.yolo_v5 import ConvBN
from nanotorch.tensor import Tensor
import numpy as np

conv_bn = ConvBN(64, 128, kernel_size=3, stride=1)
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = conv_bn(x)
print(y.shape)  # (1, 128, 52, 52)
```

### 2. C3 模块

```python
from nanotorch.detection.yolo_v5 import C3

c3 = C3(128, 256, num_blocks=3)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = c3(x)
print(y.shape)  # (1, 256, 52, 52)
```

### 3. SPPF 模块

```python
from nanotorch.detection.yolo_v5 import SPPF

sppf = SPPF(512, 512, kernel_size=5)
x = Tensor(np.random.randn(1, 512, 13, 13).astype(np.float32))
y = sppf(x)
print(y.shape)  # (1, 512, 13, 13)
```

### 4. YOLOv5 完整模型

```python
from nanotorch.detection.yolo_v5 import YOLOv5, YOLOv5Nano, YOLOv5Small, build_yolov5

# 方式一：创建特定版本
model = YOLOv5Nano(num_classes=80, input_size=640)
model = YOLOv5Small(num_classes=80, input_size=640)

# 方式二：指定版本
model = YOLOv5(num_classes=80, input_size=640, version='s')

# 方式三：使用工厂函数
model = build_yolov5('s', num_classes=80, input_size=640)

# 前向传播
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
output = model(x)

print(output['small'].shape)   # 大物体检测
print(output['medium'].shape)  # 中等物体检测
print(output['large'].shape)   # 小物体检测
```

---

## 损失函数

### YOLO v5 损失函数设计

YOLO v5 使用与 YOLO v4 类似的损失函数组合：

$$
L = \lambda_{box} L_{CIoU} + \lambda_{obj} L_{obj} + \lambda_{noobj} L_{noobj} + \lambda_{class} L_{class}
$$

### 边界框回归损失 (CIoU)

$$
L_{box} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$

### 目标置信度损失

$$
L_{obj} = -\frac{1}{N_{pos}} \sum_{i \in pos} \left[ \hat{C}_i \log(C_i) + (1 - \hat{C}_i) \log(1 - C_i) \right]
$$

$$
L_{noobj} = -\frac{1}{N_{neg}} \sum_{i \in neg} \left[ \hat{C}_i \log(C_i) + (1 - \hat{C}_i) \log(1 - C_i) \right]
$$

### 分类损失（BCE）

$$
L_{class} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left[ y_{ic} \log(\hat{p}_{ic}) + (1 - y_{ic}) \log(1 - \hat{p}_{ic}) \right]
$$

### AutoAnchor 自动锚框

YOLO v5 自动计算最佳 Anchor：

$$
\text{Anchor}_{opt} = \arg\min_{k} \sum_{i=1}^{N} \min_{j=1}^{K} (1 - \text{IoU}(box_i, anchor_j))
$$

### Mosaic 数据增强

Mosaic 将 4 张图像拼接：

$$
I_{out} = \text{Concat}(I_1[0:H/2, 0:W/2], I_2[0:H/2, W/2:W], I_3[H/2:H, 0:W/2], I_4[H/2:H, W/2:W])
$$

### 使用损失函数

```python
from nanotorch.detection.yolo_v5 import YOLOv5Loss, YOLOv5LossSimple

# 创建损失函数
loss_fn = YOLOv5Loss(
    num_classes=80,
    lambda_box=5.0,
    lambda_obj=1.0,
    lambda_noobj=0.5,
    lambda_class=1.0
)

# 计算损失
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

## 训练流程

```python
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v5 import build_yolov5
import numpy as np

# 创建模型
model = build_yolov5('n', num_classes=80, input_size=224)

# 优化器和调度器
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 训练循环
for epoch in range(10):
    model.train()
    total_loss = 0
    
    for _ in range(10):
        images = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        
        optimizer.zero_grad()
        output = model(images)
        
        # MSE 损失
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

## 代码示例

### 运行 Demo

```bash
# 查看架构
python examples/yolo_v5/demo.py --mode arch

# 训练（使用 nano 版本）
python examples/yolo_v5/demo.py --mode train --version n

# 训练（使用 small 版本）
python examples/yolo_v5/demo.py --mode train --version s --epochs 5

# 推理
python examples/yolo_v5/demo.py --mode inference

# 完整流程
python examples/yolo_v5/demo.py --mode both
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 2 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--image-size` | 224 | 输入大小 |
| `--version` | n | 模型版本 (n/s/m/l/x) |
| `--num-classes` | 80 | 类别数 |

### 运行测试

```bash
# 单元测试
python -m pytest tests/detection/yolo_v5/test_yolov5_model.py -v

# 集成测试
python -m pytest tests/detection/yolo_v5/test_integration.py -v
```

---

## YOLO v4 vs YOLO v5 对比

| 特性 | YOLO v4 | YOLO v5 |
|------|---------|---------|
| 开发者 | Alexey Bochkovskiy | Ultralytics |
| 激活函数 | Mish | SiLU |
| 池化模块 | SPP | SPPF |
| Bottleneck | CSPResBlock | C3 |
| 模型尺寸 | 固定 | 多尺寸 (n/s/m/l/x) |
| 易用性 | 中等 | 高 |

---

## 常见问题

### 1. 如何选择模型版本？

- **Nano (n)**: 边缘设备，实时性要求高
- **Small (s)**: 平衡速度和精度
- **Medium (m)**: 一般应用场景
- **Large (l)**: 精度优先
- **XLarge (x)**: 最高精度

### 2. SPPF 相比 SPP 的优势？

- 更少的计算量
- 相同的感受野
- 更快的推理速度

### 3. 为什么使用 SiLU 激活？

- 平滑的非线性
- 无上界，有下界
- 比 ReLU 更好的梯度流

---

## 总结

本教程介绍了使用 nanotorch 实现 YOLO v5 的完整流程：

1. **核心改进**: C3 模块、SPPF、SiLU 激活
2. **模型架构**: 多尺寸模型设计
3. **损失函数**: 边界框损失 + 置信度损失 + 分类损失
4. **训练推理**: 完整的训练和推理流程

---

## 参考资料

1. **Ultralytics YOLOv5**: https://github.com/ultralytics/yolov5

2. **nanotorch YOLO v4 教程**: `/docs/tutorials/20-yolov4.md`

3. **nanotorch YOLO v3 教程**: `/docs/tutorials/19-yolov3.md`
