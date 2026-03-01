# YOLO v4 目标检测模型实现教程

本教程详细介绍如何使用 nanotorch 从零实现 YOLO v4（You Only Look Once, 2020）目标检测模型。

## 目录

1. [概述](#概述)
2. [YOLO v4 核心改进](#yolo-v4-核心改进)
3. [模型架构](#模型架构)
4. [损失函数](#损失函数)
5. [数据准备](#数据准备)
6. [训练流程](#训练流程)
7. [推理与后处理](#推理与后处理)
8. [代码示例](#代码示例)

---

## 概述

YOLO v4 是 YOLO 系列的第四个版本，在保持实时性能的同时显著提升了检测精度。

### YOLO v4 的主要贡献

1. **CSPDarknet53 骨干网络**: 使用 Cross Stage Partial 连接
2. **SPP 模块**: 空间金字塔池化，增强感受野
3. **PANet**: 路径聚合网络，更好的特征融合
4. **Mish 激活函数**: 平滑的非单调激活函数
5. **CIoU Loss**: 完整 IoU 损失，更好的边界框回归
6. **Bag of Freebies (BoF)**: 仅增加训练成本的技巧
7. **Bag of Specials (BoS)**: 增加少量推理成本的模块

### nanotorch 的 YOLO v4 实现模块

```
nanotorch/detection/yolo_v4/
├── __init__.py        # 模块导出
├── yolo_v4_model.py   # 模型架构 (CSPDarknet53, PANet, YOLOHead, YOLOv4, YOLOv4Tiny)
└── yolo_v4_loss.py    # 损失函数 (YOLOv4Loss, YOLOv4LossSimple, encode_targets_v4, decode_predictions_v4)

examples/yolo_v4/
└── demo.py            # 训练和推理演示

tests/detection/yolo_v4/
├── test_yolov4_model.py  # 单元测试
└── test_integration.py   # 集成测试
```

---

## YOLO v4 核心改进

### CSPDarknet53 骨干网络

CSP (Cross Stage Partial) 连接将特征图分成两部分，减少计算量的同时保持梯度流：

```
输入特征
    ├─── 分支1: 直接连接
    │
    └─── 分支2: 通过残差块
              ↓
         Concat(分支1, 分支2)
              ↓
           Merge Conv
```

### SPP (Spatial Pyramid Pooling)

SPP 模块通过不同尺度的最大池化捕获多尺度上下文：

```
输入特征
    ↓
Conv 1x1
    ↓
┌───────────────────────────────────────┐
│  MaxPool 5x5 →──────────────────────┐ │
│  MaxPool 9x9 →────────────────────┐ │ │
│  MaxPool 13x13 →───────────────┐  │ │ │
│  原始特征 →──────────────────┐ │  │ │ │
└──────────────────────────────┼─┼──┼─┼─┘
                               ↓ ↓  ↓ ↓
                           Concat
                               ↓
                          Conv 1x1
```

### PANet (Path Aggregation Network)

PANet 改进特征融合：

```
Backbone 输出:
  scale1 (512 channels, 13×13)  ← 大物体
  scale2 (512 channels, 26×26)  ← 中等物体
  scale3 (256 channels, 52×52)  ← 小物体

PANet 处理:
┌─────────────────────────────────────────────────────┐
│ Top-down pathway (自顶向下):                         │
│   scale1 → Conv → Upsample → Concat(scale2) → p4   │
│   p4 → Conv → Upsample → Concat(scale3) → p3       │
├─────────────────────────────────────────────────────┤
│ Bottom-up pathway (自底向上):                        │
│   p3 → Downsample → Concat(p4) → n4                │
│   n4 → Downsample → Concat(p5) → n5                │
└─────────────────────────────────────────────────────┘
```

### Mish 激活函数

Mish 是一个平滑的非单调激活函数：

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

```python
from nanotorch.detection.yolo_v4 import Mish
from nanotorch.tensor import Tensor
import numpy as np

mish = Mish()
x = Tensor(np.array([-2, -1, 0, 1, 2], dtype=np.float32))
y = mish(x)
print(y.data)  # 平滑的激活值
```

---

## 模型架构

### 1. ConvBNMish（基础卷积块）

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

### 2. CSPResBlock（CSP 残差块）

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

### 3. SPP 模块

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

### 4. CSPDarknet53 骨干网络

```python
from nanotorch.detection.yolo_v4 import CSPDarknet53

backbone = CSPDarknet53(in_channels=3)

x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
features = backbone(x)

print(features['scale1'].shape)  # (1, 512, 13, 13)
print(features['scale2'].shape)  # (1, 512, 26, 26)
print(features['scale3'].shape)  # (1, 256, 52, 52)
```

### 5. YOLOv4 完整模型

```python
from nanotorch.detection.yolo_v4 import YOLOv4, YOLOv4Tiny, build_yolov4

# 方式一：创建完整模型
model = YOLOv4(num_classes=80, input_size=416)

# 方式二：创建轻量版
tiny_model = YOLOv4Tiny(num_classes=80, input_size=416)

# 方式三：使用工厂函数
model = build_yolov4('full', num_classes=80, input_size=416)
tiny_model = build_yolov4('tiny', num_classes=80, input_size=416)

# 前向传播
x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
output = model(x)

print(output['small'].shape)   # (1, 255, 13, 13)
print(output['medium'].shape)  # (1, 255, 26, 26)
print(output['large'].shape)   # (1, 255, 52, 52)
```

---

## 损失函数

### CIoU Loss

YOLO v4 使用 CIoU (Complete IoU) 损失替代 MSE 进行边界框回归：

$$\text{CIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2} - \alpha v$$

其中：
- $d$: 两个框中心点的距离
- $c$: 最小包围框的对角线长度
- $v$: 宽高比一致性
- $\alpha$: 权衡参数

### CIoU 完整公式

$$
\mathcal{L}_{CIoU} = 1 - \text{IoU} + \frac{\rho^2(b, b^{gt})}{c^2} + \alpha v
$$

其中宽高比一致性：

$$
v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2
$$

权衡参数：

$$
\alpha = \frac{v}{1 - \text{IoU} + v}
$$

### 总损失函数

$$
L = \lambda_{box} L_{CIoU} + \lambda_{obj} L_{obj} + \lambda_{noobj} L_{noobj} + \lambda_{class} L_{class}
$$

### Mish 激活函数公式

$$\text{Mish}(x) = x \cdot \tanh(\text{softplus}(x)) = x \cdot \tanh(\ln(1 + e^x))$$

导数：

$$
\text{Mish}'(x) = \frac{e^x}{1 + e^x} \cdot \tanh(\text{softplus}(x)) + \frac{x}{1 + e^x} \cdot \text{sech}^2(\text{softplus}(x))
$$

### CmBN (Cross mini-Batch Normalization)

CmBN 聚合多个 batch 的统计量：

$$
\mu_t = \frac{1}{m \times k} \sum_{i=1}^{k} \sum_{j=1}^{m} x_{t-i,j}
$$

$$
\sigma_t^2 = \frac{1}{m \times k} \sum_{i=1}^{k} \sum_{j=1}^{m} (x_{t-i,j} - \mu_t)^2
$$

其中 $k$ 是聚合的 batch 数量。

### 使用损失函数

```python
from nanotorch.detection.yolo_v4 import YOLOv4Loss, YOLOv4LossSimple

# 创建损失函数
loss_fn = YOLOv4Loss(
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
```

### 目标编码与解码

```python
from nanotorch.detection.yolo_v4 import encode_targets_v4, decode_predictions_v4

# YOLO v4 anchors
anchors = [
    (12, 16), (19, 36), (40, 28),
    (36, 75), (76, 55), (72, 146),
    (142, 110), (192, 243), (459, 401)
]

# 编码
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

# 解码
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

## 数据准备

### SyntheticCOCODataset

```python
from examples.yolo_v4.demo import SyntheticCOCODataset, create_dataloader

# 创建数据集
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

# 创建 DataLoader
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

## 训练流程

```python
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v4 import YOLOv4Tiny
from examples.yolo_v4.demo import create_dataloader
import numpy as np

# 创建模型
model = YOLOv4Tiny(num_classes=80, input_size=224)

# 优化器和调度器
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 数据加载器
dataloader = create_dataloader(
    num_samples=50,
    batch_size=2,
    image_size=224,
    num_classes=80
)

# 训练循环
for epoch in range(5):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        images = np.stack([item['image'] for item in batch])
        images = Tensor(images)
        
        optimizer.zero_grad()
        output = model(images)
        
        # MSE 损失
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

## 推理与后处理

```python
from nanotorch.detection.yolo_v4 import YOLOv4Tiny
from nanotorch.detection.nms import nms
from nanotorch.tensor import Tensor
import numpy as np

# 加载模型
model = YOLOv4Tiny(num_classes=80, input_size=416)
model.eval()

# 准备输入
image = np.random.randn(1, 3, 416, 416).astype(np.float32)
x = Tensor(image)

# 前向传播
output = model(x)
print(output['small'].shape)  # (1, 255, 13, 13)

# 应用 NMS
boxes = np.array([[100, 100, 200, 200], [105, 105, 205, 205]], dtype=np.float32)
scores = np.array([0.9, 0.8], dtype=np.float32)

keep = nms(boxes, scores, iou_threshold=0.5)
print(f"保持的检测框: {keep}")
```

---

## 代码示例

### 完整训练示例

```python
"""
YOLO v4 完整训练示例
"""
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v4 import YOLOv4Tiny, YOLOv4Loss

def train_yolov4():
    # 超参数
    num_epochs = 10
    batch_size = 4
    learning_rate = 1e-4
    image_size = 224
    num_classes = 80
    
    # 创建模型
    model = YOLOv4Tiny(num_classes=num_classes, input_size=image_size)
    
    # 优化器和调度器
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # 模拟数据
        for _ in range(10):
            images = Tensor(np.random.randn(batch_size, 3, image_size, image_size).astype(np.float32))
            
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
        avg_loss = total_loss / 10
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

if __name__ == "__main__":
    model = train_yolov4()
    print("训练完成！")
```

### 运行测试

```bash
# 运行 YOLO v4 单元测试
python -m pytest tests/detection/yolo_v4/test_yolov4_model.py -v

# 运行集成测试
python -m pytest tests/detection/yolo_v4/test_integration.py -v

# 运行所有 YOLO v4 测试
python -m pytest tests/detection/yolo_v4/ -v
```

---

## 快速开始：Demo 脚本

### 查看模型架构

```bash
python examples/yolo_v4/demo.py --mode arch
```

### 训练模型

```bash
# 默认使用 YOLOv4Tiny
python examples/yolo_v4/demo.py --mode train

# 自定义参数
python examples/yolo_v4/demo.py --mode train \
    --epochs 5 \
    --batch-size 4 \
    --num-samples 50

# 使用完整 YOLOv4（需要更多内存）
python examples/yolo_v4/demo.py --mode train --full --batch-size 1
```

### 推理演示

```bash
python examples/yolo_v4/demo.py --mode inference
```

### 完整流程

```bash
python examples/yolo_v4/demo.py --mode both
```

---

## YOLO v3 vs YOLO v4 对比

| 特性 | YOLO v3 | YOLO v4 |
|------|---------|---------|
| 骨干网络 | Darknet-53 | CSPDarknet-53 |
| 激活函数 | LeakyReLU | Mish |
| 特征融合 | FPN | PANet |
| 池化模块 | 无 | SPP |
| 边界框损失 | MSE | CIoU Loss |
| 数据增强 | 基础 | Mosaic, Mixup 等 |

---

## 模型参数对比

| Model | Input Size | Parameters | Memory |
|-------|------------|------------|--------|
| YOLOv4Tiny | 416×416 | ~6M | ~2GB |
| YOLOv4 | 416×416 | ~64M | ~8GB |

---

## 常见问题

### 1. 为什么 YOLO v4 使用 CSP 连接？

CSP (Cross Stage Partial) 连接的优势：
- 减少计算量约 20%
- 缓解梯度消失问题
- 保持或提升精度

### 2. PANet 相比 FPN 有什么改进？

PANet 的改进：
- 增加自底向上的路径
- 更好的特征融合
- 提升小物体检测能力

### 3. CIoU Loss 的优势？

CIoU Loss 考虑：
- 重叠面积
- 中心点距离
- 宽高比一致性
- 比 MSE 更好的收敛性

---

## 总结

本教程介绍了使用 nanotorch 实现 YOLO v4 的完整流程：

1. **核心改进**: CSPDarknet53、PANet、SPP、Mish 激活
2. **模型架构**: 骨干网络 + 特征融合 + 检测头
3. **损失函数**: CIoU 损失 + BCE 损失
4. **训练推理**: 完整的训练和推理流程

通过本教程，你应该能够：
- 理解 YOLO v4 的架构设计
- 使用 nanotorch 构建复杂的目标检测模型
- 理解 CSP 连接和 PANet 的作用

---

## 参考资料

1. **YOLO v4 论文**: "YOLOv4: Optimal Speed and Accuracy of Object Detection" (2020)
   - https://arxiv.org/abs/2004.10934

2. **CSPNet**: "CSPNet: A New Backbone that can Enhance Learning Capability of CNN" (2020)
   - https://arxiv.org/abs/1911.11929

3. **PANet**: "Path Aggregation Network for Instance Segmentation" (2018)
   - https://arxiv.org/abs/1803.01534

4. **Mish**: "Mish: A Self Regularized Non-Monotonic Activation Function" (2019)
   - https://arxiv.org/abs/1908.08681

5. **nanotorch YOLO v3 教程**: `/docs/tutorials/19-yolov3.md`
