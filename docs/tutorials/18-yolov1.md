# YOLO v1 目标检测模型实现教程

## 想象你在玩"大家来找茬"...

一张复杂的图片，你需要：
- 找出图中所有的物品
- 说出每个物品的位置和类别
- 越快越好

```
传统方法（R-CNN）：
  第一步：找候选区域（2000个框）
  第二步：每个框单独识别
  就像：用2000个放大镜逐个看
  问题：太慢了！

YOLO v1（开创性方法）：
  把图片分成 7×7 的网格
  每个网格负责检测中心在该网格的物体
  一次前向传播就完成所有检测
  就像：一眼扫完整张图
```

**YOLO = You Only Look Once** —— 只看一眼，就能找出所有物体。

---

## 目录

1. [概述](#概述)
2. [YOLO v1 核心思想](#yolo-v1-核心思想)
3. [模型架构](#模型架构)
4. [损失函数](#损失函数)
5. [数据准备](#数据准备)
6. [训练流程](#训练流程)
7. [推理与后处理](#推理与后处理)
8. [代码示例](#代码示例)

---

## 概述

YOLO v1 是目标检测领域的开创性工作，首次将目标检测建模为**单次回归问题**，实现了实时目标检测。

### YOLO v1 的主要贡献

1. **单阶段检测**: 将目标检测作为回归问题，无需候选区域生成
2. **实时性能**: 在 VOC 2007 测试集上达到 45 FPS
3. **全局推理**: 使用整张图像进行预测，减少背景误检

### nanotorch 的 YOLO v1 实现模块

```
nanotorch/detection/yolo_v1/
├── __init__.py        # 模块导出
├── yolo_v1_model.py   # 模型架构 (Darknet, YOLOv1, YOLOv1Tiny)
└── yolo_v1_loss.py    # 损失函数 (YOLOv1Loss, encode_targets, decode_predictions)

examples/yolo_v1/
└── data.py            # 数据加载 (SyntheticVOCDataset, YOLOv1Transform)

tests/detection/yolo_v1/
├── test_yolov1_model.py  # 单元测试 (27 tests)
└── test_integration.py   # 集成测试 (17 tests)
```

---

## YOLO v1 核心思想

### 网格划分 (Grid-based Detection)

YOLO v1 将输入图像划分为 S×S 的网格（原论文 S=7）：

```
输入图像 (448×448)
    ↓
划分 7×7 网格
    ↓
每个网格单元负责检测：
  - 中心落在该网格内的物体
  - 预测 B 个边界框（原论文 B=2）
  - 预测 C 个类别概率（VOC 数据集 C=20）
```

### 输出张量格式

模型输出形状为 (N, S, S, B×5+C)：

```
每个网格单元的输出（共 30 个值）:
┌─────────────────────────────────────────────────────────────┐
│ Box 1 (5)    │ Box 2 (5)    │ Class Probs (20)              │
│ x, y, w, h, c│ x, y, w, h, c│ p0, p1, p2, ..., p19          │
└─────────────────────────────────────────────────────────────┘
      ↑              ↑              ↑
   第1个边界框    第2个边界框    20个类别概率

- x, y: 边界框中心相对于网格单元的坐标 (0~1)
- w, h: 边界框宽高相对于整张图像的比例 (0~1)
- c: 置信度 = Pr(Object) × IoU(pred, truth)
```

### Darknet 骨干网络

YOLO v1 使用 Darknet-24 作为骨干网络：

```
输入 (3, 448, 448)
    ↓
Conv 7×7, s=2 → MaxPool 2×2, s=2
Conv 3×3 → Conv 3×3 → MaxPool
...（共 24 层卷积）
    ↓
特征图 (1024, 14, 14)
    ↓
Flatten → FC 4096 → FC 1470
    ↓
输出 (7×7×30 = 1470)
```

---

## 模型架构

### 1. ConvBlock（基础卷积块）

每个卷积块包含：Conv2D + LeakyReLU

```python
from nanotorch.detection.yolo_v1 import ConvBlock
from nanotorch.tensor import Tensor
import numpy as np

# 创建卷积块
conv_block = ConvBlock(
    in_channels=3,
    out_channels=64,
    kernel_size=7,
    stride=2,
    padding=3
)

# 前向传播
x = Tensor(np.random.randn(1, 3, 448, 448).astype(np.float32))
y = conv_block(x)
print(y.shape)  # (1, 64, 224, 224)
```

### 2. Darknet 骨干网络

```python
from nanotorch.detection.yolo_v1 import Darknet

# 创建 Darknet 骨干
backbone = Darknet(in_channels=3)

# 前向传播
x = Tensor(np.random.randn(1, 3, 448, 448).astype(np.float32))
features = backbone(x)
print(features.shape)  # (1, 1024, 7, 7)
```

Darknet 网络结构：

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

### 3. YOLOv1Head（检测头）

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

### 4. YOLOv1 完整模型

```python
from nanotorch.detection.yolo_v1 import YOLOv1, YOLOv1Tiny, build_yolov1

# 方式一：直接创建完整模型
model = YOLOv1(
    input_size=448,
    S=7,
    B=2,
    C=20
)

# 方式二：创建轻量版（用于测试）
tiny_model = YOLOv1Tiny(
    input_size=224,
    S=7,
    B=2,
    C=20
)

# 方式三：使用工厂函数
model = build_yolov1('full', input_size=448, S=7, B=2, C=20)
tiny_model = build_yolov1('tiny', input_size=224, S=7, B=2, C=20)

# 前向传播
x = Tensor(np.random.randn(1, 3, 448, 448).astype(np.float32))
output = model(x)
print(output.shape)  # (1, 1470)

# 使用 predict 方法获取结构化输出
result = model.predict(x)
print(result['reshaped'].shape)  # (1, 7, 7, 30)
```

---

## 损失函数

### YOLO v1 损失函数设计

YOLO v1 使用**加权和方误差损失**：

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

其中：
- $\lambda_{coord} = 5.0$: 坐标损失权重
- $\lambda_{noobj} = 0.5$: 无物体置信度损失权重
- $\mathbb{1}_{ij}^{obj}$: 表示网格 i 的第 j 个边界框负责检测物体
- $S = 7$: 网格大小
- $B = 2$: 每个网格预测的边界框数量
- $C = 20$: 类别数量

### 损失函数分解

**坐标损失**（使用平方根以减小大框的影响）：

$$
L_{coord} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]
$$

**置信度损失**：

$$
L_{conf} = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2 + \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2
$$

**分类损失**：

$$
L_{class} = \sum_{i=0}^{S^2} \mathbb{1}_{i}^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2
$$

### 输出张量结构

$$
\text{Output} \in \mathbb{R}^{S \times S \times (B \times 5 + C)} = \mathbb{R}^{7 \times 7 \times 30}
$$

每个网格单元预测：
- 2 个边界框：$(x, y, w, h, \text{confidence}) \times 2 = 10$ 个值
- 20 个类别概率：$C = 20$ 个值
- 总计：$10 + 20 = 30$ 个值

### 使用损失函数

```python
from nanotorch.detection.yolo_v1 import YOLOv1Loss, YOLOv1LossSimple
from nanotorch.tensor import Tensor

# 创建损失函数
loss_fn = YOLOv1Loss(
    S=7,
    B=2,
    C=20,
    coord_weight=5.0,
    noobj_weight=0.5
)

# 准备预测和目标
predictions = Tensor(np.random.randn(2, 7, 7, 30).astype(np.float32) * 0.1)
targets = Tensor(np.zeros((2, 7, 7, 30), dtype=np.float32))

# 计算损失
loss, loss_dict = loss_fn(predictions, targets)

print(f"Total Loss: {loss.item():.4f}")
print(f"Coord Loss: {loss_dict['coord_loss']:.4f}")
print(f"Obj Conf Loss: {loss_dict['obj_conf_loss']:.4f}")
print(f"Noobj Conf Loss: {loss_dict['noobj_conf_loss']:.4f}")
print(f"Class Loss: {loss_dict['class_loss']:.4f}")
```

### 简化损失函数（用于测试）

```python
# 简化版 MSE 损失，用于快速测试
simple_loss_fn = YOLOv1LossSimple()

predictions = Tensor(np.random.randn(2, 1470).astype(np.float32))
targets = Tensor(np.zeros((2, 1470), dtype=np.float32))

loss, loss_dict = simple_loss_fn(predictions, targets)
# 注意：YOLOv1LossSimple 返回 (float, dict)，不支持 backward
```

### 目标编码与解码

```python
from nanotorch.detection.yolo_v1 import encode_targets, decode_predictions
import numpy as np

# 编码：将边界框转换为 YOLO 格式
boxes = np.array([
    [100, 100, 200, 200],   # [x1, y1, x2, y2]
    [250, 250, 350, 350]
], dtype=np.float32)
labels = np.array([0, 5], dtype=np.int64)  # 类别索引

target = encode_targets(
    boxes=boxes,
    labels=labels,
    S=7,
    B=2,
    C=20,
    image_size=448
)
print(target.shape)  # (7, 7, 30)

# 解码：将预测转换为边界框
predictions = np.random.randn(7, 7, 30).astype(np.float32) * 0.1
boxes, scores, class_ids = decode_predictions(
    predictions,
    conf_threshold=0.5,
    image_size=448
)

print(f"Detected {len(boxes)} objects")
```

---

## 数据准备

### SyntheticVOCDataset（合成数据集）

用于测试的合成 VOC 格式数据集：

```python
from examples.yolo_v1.data import SyntheticVOCDataset

# 创建合成数据集
dataset = SyntheticVOCDataset(
    num_samples=1000,
    image_size=448,
    S=7,
    B=2,
    C=20,
    max_objects=5,
    min_objects=1
)

# 获取样本
sample = dataset[0]
print(sample['image'].shape)     # (448, 448, 3)
print(sample['target'].shape)    # (7, 7, 30)
print(sample['boxes'].shape)     # (N, 4)
print(sample['labels'].shape)    # (N,)

# VOC 20 类别
print(dataset.VOC_CLASSES)
# ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
#  'bus', 'car', 'cat', 'chair', 'cow', 
#  'diningtable', 'dog', 'horse', 'motorbike', 'person', 
#  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

### YOLOv1Transform（数据预处理）

```python
from examples.yolo_v1.data import YOLOv1Transform

transform = YOLOv1Transform(
    image_size=448,
    S=7,
    B=2,
    C=20
)

# 应用变换
sample = dataset[0]
transformed = transform(sample)
```

### YOLOv1Collate（批处理）

```python
from examples.yolo_v1.data import YOLOv1Collate

collate = YOLOv1Collate(S=7, B=2, C=20)

# 批处理多个样本
samples = [dataset[i] for i in range(4)]
batch = collate(samples)

print(batch['images'].shape)   # (4, 3, 448, 448)
print(batch['targets'].shape)  # (4, 7, 7, 30)
```

### 创建 DataLoader

```python
from examples.yolo_v1.data import create_synthetic_dataloader

# 一键创建 DataLoader
dataloader = create_synthetic_dataloader(
    num_samples=1000,
    batch_size=8,
    image_size=448,
    S=7,
    B=2,
    C=20,
    shuffle=True
)

# 遍历数据
for batch in dataloader:
    images = batch['images']    # (8, 3, 448, 448)
    targets = batch['targets']  # (8, 7, 7, 30)
    break
```

---

## 训练流程

### 基本训练循环

```python
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.detection.yolo_v1 import YOLOv1, YOLOv1Loss
from examples.yolo_v1.data import create_synthetic_dataloader

# 创建模型和优化器
model = YOLOv1(input_size=448, S=7, B=2, C=20)
optimizer = Adam(model.parameters(), lr=1e-4)

# 创建数据加载器
dataloader = create_synthetic_dataloader(
    num_samples=100,
    batch_size=4,
    image_size=448
)

# 训练循环
for epoch in range(10):
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        # 准备数据
        images = Tensor(batch['images'])
        targets = Tensor(batch['targets'])
        
        # 前向传播
        optimizer.zero_grad()
        output = model(images)
        output_reshaped = output.reshape((images.shape[0], 7, 7, 30))
        
        # 使用 MSE 损失进行梯度计算
        # (YOLOv1Loss 不支持 backward，因为内部使用 NumPy)
        diff = output_reshaped - targets
        loss = (diff * diff).mean()
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
```

### 使用 YOLOv1Loss 监控训练

```python
from nanotorch.detection.yolo_v1 import YOLOv1Loss

loss_fn = YOLOv1Loss(S=7, B=2, C=20)

# 在训练循环中监控各部分损失
output = model(images)
output_reshaped = output.reshape((images.shape[0], 7, 7, 30))

# 计算 YOLO 损失（用于监控，不支持 backward）
yolo_loss, loss_dict = loss_fn(output_reshaped, targets)

print(f"Coord Loss: {loss_dict['coord_loss']:.4f}")
print(f"Object Conf Loss: {loss_dict['obj_conf_loss']:.4f}")
print(f"No-Object Conf Loss: {loss_dict['noobj_conf_loss']:.4f}")
print(f"Class Loss: {loss_dict['class_loss']:.4f}")
```

---

## 推理与后处理

### 模型推理

```python
from nanotorch.detection.yolo_v1 import YOLOv1, decode_predictions
from nanotorch.tensor import Tensor
import numpy as np

# 加载训练好的模型
model = YOLOv1(input_size=448, S=7, B=2, C=20)
# model.load_state_dict(...)  # 加载权重

# 准备输入图像
image = np.random.randn(1, 3, 448, 448).astype(np.float32)
x = Tensor(image)

# 前向传播
output = model(x)
output_reshaped = output.reshape((1, 7, 7, 30))

# 解码预测
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
完整的 YOLO v1 训练示例
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
    # 超参数
    num_epochs = 50
    batch_size = 8
    learning_rate = 1e-4
    image_size = 448
    S, B, C = 7, 2, 20
    
    # 创建模型
    model = YOLOv1(input_size=image_size, S=S, B=B, C=C)
    
    # 创建优化器和调度器
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 创建损失函数（用于监控）
    loss_fn = YOLOv1Loss(S=S, B=B, C=C)
    
    # 创建数据加载器
    dataloader = create_synthetic_dataloader(
        num_samples=200,
        batch_size=batch_size,
        image_size=image_size,
        S=S, B=B, C=C,
        shuffle=True
    )
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch in dataloader:
            images = Tensor(batch['images'])
            targets = Tensor(batch['targets'])
            
            optimizer.zero_grad()
            
            # 前向传播
            output = model(images)
            output_reshaped = output.reshape((images.shape[0], S, S, B*5+C))
            
            # MSE 损失（用于 backward）
            diff = output_reshaped - targets
            loss = (diff * diff).mean()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # 打印统计
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_lr():.6f}")
    
    return model

if __name__ == "__main__":
    model = train_yolov1()
    print("训练完成！")
```

### 推理示例

```python
"""
YOLO v1 推理示例
"""
import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.detection.yolo_v1 import YOLOv1, decode_predictions

def inference_example():
    # 加载模型
    model = YOLOv1(input_size=448, S=7, B=2, C=20)
    model.eval()
    
    # 准备输入（实际应用中应加载真实图像）
    image = np.random.randn(1, 3, 448, 448).astype(np.float32)
    x = Tensor(image)
    
    # 前向传播
    output = model(x)
    output_reshaped = output.reshape((1, 7, 7, 30))
    
    # 解码预测
    predictions = output_reshaped.data[0]
    boxes, scores, class_ids = decode_predictions(
        predictions,
        conf_threshold=0.3,
        image_size=448
    )
    
    # VOC 类别名称
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    # 打印检测结果
    print(f"检测到 {len(boxes)} 个物体:")
    for i in range(len(boxes)):
        class_name = VOC_CLASSES[class_ids[i]]
        x1, y1, x2, y2 = boxes[i]
        print(f"  {class_name}: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), 置信度: {scores[i]:.2f}")

if __name__ == "__main__":
    inference_example()
```

### 运行测试

```bash
# 运行 YOLO v1 单元测试
python -m pytest tests/detection/yolo_v1/test_yolov1_model.py -v

# 运行集成测试
python -m pytest tests/detection/yolo_v1/test_integration.py -v

# 运行所有 YOLO v1 测试
python -m pytest tests/detection/yolo_v1/ -v
```

---

## 快速开始：Demo 脚本

我们提供了一个完整的 Demo 脚本来演示 YOLO v1 的训练和推理流程。

### 查看模型架构

```bash
python examples/yolo_v1/demo.py --mode arch
```

输出示例：
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

### 编码/解码演示

```bash
python examples/yolo_v1/demo.py --mode encode
```

输出示例：
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

### 训练模型

```bash
# 默认使用 YOLOv1Tiny（推荐）
python examples/yolo_v1/demo.py --mode train

# 自定义参数
python examples/yolo_v1/demo.py --mode train \
    --epochs 5 \
    --batch-size 4 \
    --num-samples 50

# 使用完整 YOLOv1（需要 8GB+ 内存）
python examples/yolo_v1/demo.py --mode train --full --batch-size 1
```

> ⚠️ **内存警告**：完整 YOLOv1 有 2.7 亿参数，448×448 输入需要约 8GB 内存。
> 如果遇到 SIGKILL 错误，请使用 `--tiny` 或减小 `--batch-size`。

训练参数说明：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 3 | 训练轮数 |
| `--batch-size` | 2 | 批大小 |
| `--lr` | 1e-4 | 学习率 |
| `--lr-step` | 5 | 学习率衰减步长 |
| `--image-size` | 224 | 输入图像大小 |
| `--num-samples` | 20 | 训练样本数 |
| `--tiny` | True | 使用 YOLOv1Tiny（默认） |
| `--full` | - | 使用完整 YOLOv1 |

### 推理演示

```bash
# 训练后运行推理
python examples/yolo_v1/demo.py --mode inference

# 自定义推理参数
python examples/yolo_v1/demo.py --mode inference \
    --num-inference 10 \
    --conf-threshold 0.3 \
    --nms-threshold 0.5
```

推理参数说明：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--num-inference` | 5 | 推理图像数量 |
| `--conf-threshold` | 0.3 | 置信度阈值 |
| `--nms-threshold` | 0.5 | NMS IoU 阈值 |

### 完整流程（训练 + 推理）

```bash
# 运行完整演示：架构 → 编码/解码 → 训练 → 推理
python examples/yolo_v1/demo.py --mode both
```

### Demo 脚本完整参数

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

## 模型参数对比

| Model | Input Size | Parameters | Memory |
|-------|------------|------------|--------|
| YOLOv1Tiny | 224×224 | ~42M | ~2GB |
| YOLOv1 | 448×448 | ~272M | ~8GB |

---

## 常见问题

### 1. 为什么会出现 SIGKILL 错误？

完整 YOLOv1 模型有 2.7 亿参数，需要较大内存。解决方案：
- 使用 `--tiny`（默认）
- 减小 `--batch-size`（如 `--batch-size 1`）
- 减小 `--num-samples`

### 2. 为什么 YOLOv1Loss 不支持 backward？

YOLOv1Loss 内部使用 NumPy 计算损失，返回的 Tensor 没有连接到计算图。训练时请使用手动 MSE 损失进行梯度计算。

### 2. 如何处理负的宽高预测？

YOLO v1 使用 sqrt(w) 和 sqrt(h) 计算损失，如果预测值为负会导致 NaN。解决方案：
- 使用 sigmoid 限制输出范围
- 添加正则化防止极端预测

### 3. 如何提高检测精度？

- 增加训练数据
- 使用数据增强（Mosaic, MixUp）
- 调整损失权重
- 使用更深的骨干网络

---

## 总结

本教程介绍了使用 nanotorch 实现 YOLO v1 的完整流程：

1. **核心思想**: 网格划分、单阶段检测、全局推理
2. **模型架构**: Darknet 骨干 + FC 检测头
3. **损失函数**: 加权和方误差，包含坐标、置信度、分类损失
4. **数据处理**: 合成数据集、预处理、批处理
5. **训练推理**: 完整训练循环、解码预测、NMS 后处理

通过本教程，你应该能够：
- 理解 YOLO v1 的设计思想
- 使用 nanotorch 构建目标检测模型
- 自定义训练和评估流程

---

## 参考资料

1. **YOLO v1 论文**: "You Only Look Once: Unified, Real-Time Object Detection" (CVPR 2016)
   - https://arxiv.org/abs/1506.02640

2. **Darknet 官方实现**: https://github.com/pjreddie/darknet

3. **PASCAL VOC 数据集**: http://host.robots.ox.ac.uk/pascal/VOC/

4. **nanotorch 文档**: `/docs`
