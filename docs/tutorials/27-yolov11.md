# YOLO v11 目标检测模型实现教程

## 有些进步，不是革命，而是进化...

iPhone 每年发布新款，外观变化不大，芯片却快了 20%。

跑鞋每年迭代，样子差不多，重量却轻了几克。

这世上大多数进步，都是这样——在已臻完美的道路上，再往前挪一小步。

YOLO v11 就是这样一小步。

它没有惊天动地的新架构，也没有革命性的新思路。它只是在 YOLO v8 的基础上，把每个细节都打磨得更好一点——

C3k2 模块取代了 C2f，更快一些。
特征提取更精细，精度更高一些。
相同的速度，精度提升；相同的精度，速度更快。

```
v8 是一栋好房子
v11 是装修后的房子

墙体没变，格局依旧
但窗户更明亮，地板更平整
住起来，更舒服
```

**YOLO v11 —— 稳中求进的智慧**，完美，再完美一点。

---

## 目录

1. [概述](#概述)
2. [YOLO v11 核心改进](#yolo-v11-核心改进)
3. [模型架构](#模型架构)
4. [损失函数](#损失函数)
5. [训练流程](#训练流程)
6. [代码示例](#代码示例)
7. [常见问题](#常见问题)
8. [总结](#总结)

---

## 概述

YOLO v11 是 Ultralytics 于 2024 年发布的最新 YOLO 版本。它在 YOLO v8 的基础上进行了架构优化，实现了更好的精度-速度平衡。

### YOLO v11 的主要特点

1. **C3k2 模块**: 更快的 C3k 结构
2. **增强的特征提取**: 改进的网络设计
3. **SPPF 模块**: 快速空间金字塔池化
4. **PANet Neck**: 路径聚合网络
5. **多尺寸模型**: n(nano), s(small), m(medium), l(large), x(xlarge)
6. **SOTA 性能**: 最新的精度-速度平衡

### nanotorch 的 YOLO v11 实现模块

```
nanotorch/detection/yolo_v11/
├── __init__.py        # 模块导出
├── yolo_v11_model.py  # 模型架构 (C3k2, SPPF, Bottleneck, Backbone, Neck, DetectHead)
└── yolo_v11_loss.py   # 损失函数 (YOLOv11Loss, encode_targets_v11, decode_predictions_v11)

examples/yolo_v11/
└── demo.py            # 训练和推理演示

tests/detection/yolo_v11/
├── test_yolov11_model.py  # 单元测试
└── test_v11_integration.py  # 集成测试
```

---

## YOLO v11 核心改进

### C3k2 模块 (Faster CSP Bottleneck with 2 convolutions)

C3k2 是 YOLO v11 的核心构建块，比 C3k 更高效：

```
输入特征 (c_in)
    │
    ├─── Conv1x1 → c_mid ─────────────────────┐
    │                                          │
    └─── Conv1x1 → c_mid → Bottleneck×N ──────┤
                                               │
                    Concat(y1, y2) ←───────────┘
                              │
                         Conv1x1 → 输出 (c_out)
```

C3k2 相比 C3k 的改进：
- **更快的计算**: 优化的并行结构
- **更好的梯度流**: 双分支设计
- **更高的效率**: 参数利用更充分

### Bottleneck 模块

```
输入 (c)
    │
    ├──→ Conv1x1 → c/2 → Conv3x3 → c
    │                        │
    └────────────────────────┴──→ Add → 输出 (c)
```

### SPPF 模块 (Spatial Pyramid Pooling Fast)

```
输入特征
    │
    ↓ Conv1×1
    │
    ├──→ y1 (原始)
    │
    ├──→ y2 = MaxPool(y1)
    │
    ├──→ y3 = MaxPool(y2)
    │
    └──→ y4 = MaxPool(y3)
           │
           ↓ Concat(y1, y2, y3, y4)
           │
           ↓ Conv1×1
           │
           ↓ 输出
```

SPPF 优势：
- **多尺度感受野**: 捕获不同尺度的上下文
- **高效实现**: 连续池化替代并行池化
- **参数效率**: 更少的参数量

---

## 模型架构

### 1. ConvBN（基础卷积块）

```python
from nanotorch.detection.yolo_v11 import ConvBN
from nanotorch.tensor import Tensor
import numpy as np

conv_bn = ConvBN(64, 128, k=3, s=1)
x = Tensor(np.random.randn(1, 64, 52, 52).astype(np.float32))
y = conv_bn(x)
print(y.shape)  # (1, 128, 52, 52)
```

### 2. Bottleneck 模块

```python
from nanotorch.detection.yolo_v11 import Bottleneck

bottleneck = Bottleneck(128, shortcut=True)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = bottleneck(x)
print(y.shape)  # (1, 128, 52, 52)
```

### 3. C3k2 模块

```python
from nanotorch.detection.yolo_v11 import C3k2

c3k2 = C3k2(128, 256, n=2, shortcut=True)
x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
y = c3k2(x)
print(y.shape)  # (1, 256, 52, 52)
```

### 4. SPPF 模块

```python
from nanotorch.detection.yolo_v11 import SPPF

sppf = SPPF(512, 512, k=5)
x = Tensor(np.random.randn(1, 512, 20, 20).astype(np.float32))
y = sppf(x)
print(y.shape)  # (1, 512, 20, 20)
```

### 5. Backbone 网络

```python
from nanotorch.detection.yolo_v11 import Backbone

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
    ↓ Stage1: Conv3×3(s2) → C3k2 → 64 channels
    ↓ Stage2: Conv3×3(s2) → C3k2 → 128 channels → 输出 s3 (stride=8)
    ↓ Stage3: Conv3×3(s2) → C3k2 → 256 channels → 输出 s2 (stride=16)
    ↓ Stage4: Conv3×3(s2) → C3k2 → SPPF → 512 channels → 输出 s1 (stride=32)
```

### 6. Neck 网络（PANet）

```python
from nanotorch.detection.yolo_v11 import Neck

neck = Neck(channels=[128, 256, 512])
neck_out = neck(features)

print(neck_out['p3'].shape)  # (1, 128, 80, 80)
print(neck_out['p4'].shape)  # (1, 256, 40, 40)
print(neck_out['p5'].shape)  # (1, 512, 20, 20)
```

### 7. 完整 YOLOv11 模型

```python
from nanotorch.detection.yolo_v11 import YOLOv11, build_yolov11

# 方式一：直接创建
model = YOLOv11(num_classes=80, input_size=640)

# 方式二：使用工厂函数
model = build_yolov11(num_classes=80, input_size=640)

# 前向传播
x = Tensor(np.random.randn(1, 3, 640, 640).astype(np.float32))
output = model(x)

print(output['small'].shape)   # (1, 85, 20, 20) - 大物体检测
print(output['medium'].shape)  # (1, 85, 40, 40) - 中等物体检测
print(output['large'].shape)   # (1, 85, 80, 80) - 小物体检测
```

---

## 损失函数

### YOLO v11 损失函数设计

YOLO v11 继承了 v8 的损失设计：

$$
L = \lambda_{box} L_{box} + \lambda_{cls} L_{cls} + \lambda_{dfl} L_{dfl}
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

### 分类损失（BCE）

$$
L_{cls} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left[ y_{ic} \log(\hat{p}_{ic}) + (1 - y_{ic}) \log(1 - \hat{p}_{ic}) \right]
$$

### DFL (Distribution Focal Loss)

$$
\hat{y} = \sum_{n=0}^{N-1} P(n) \cdot n, \quad N = \text{reg\_max}
$$

$$
L_{dfl} = -\sum_{n=0}^{N-1} \left[ y_n \log(\hat{P}(n)) \right]
$$

### C3k2 模块

C3k2 是 C3k 的优化版本：

$$
\text{Output} = \text{Conv}_{out}(\text{Concat}(x, \text{BottleNeck}_1(x), \text{BottleNeck}_2(x)))
$$

### C2PSA 模块

C2PSA (C2 with Partial Self-Attention)：

$$
\text{Output} = \text{Conv}(\text{PartialSA}(\text{Conv}(x)))
$$

部分自注意力减少计算量：

$$
\text{PartialSA}(x) = \text{Concat}(\text{SA}(x_{part}), x_{rest})
$$

### YOLOv11Loss

```python
from nanotorch.detection.yolo_v11 import YOLOv11Loss

loss_fn = YOLOv11Loss(
    num_classes=80,
    lambda_box=7.5,
    lambda_cls=0.5,
    lambda_dfl=1.5
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
```

---

## 训练流程

### 完整训练示例

```python
from nanotorch.tensor import Tensor
from nanotorch.optim import AdamW
from nanotorch.optim.lr_scheduler import CosineWarmupScheduler
from nanotorch.detection.yolo_v11 import build_yolov11, YOLOv11Loss
import numpy as np

# 创建模型
model = build_yolov11(num_classes=80, input_size=640)

# 损失函数和优化器
loss_fn = YOLOv11Loss(num_classes=80)
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
scheduler = CosineWarmupScheduler(optimizer, warmup_epochs=3, max_epochs=50)

# 训练循环
for epoch in range(50):
    model.train()
    total_loss = 0
    
    for batch_idx in range(100):
        images = Tensor(np.random.randn(4, 3, 640, 640).astype(np.float32))
        targets = [{
            'boxes': np.random.rand(3, 4).astype(np.float32) * 600,
            'labels': np.random.randint(0, 80, 3).astype(np.int64)
        } for _ in range(4)]
        
        optimizer.zero_grad()
        output = model(images)
        loss, loss_dict = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {total_loss/100:.4f}")
```

---

## 代码示例

### 运行 Demo

```bash
# 查看模型架构
python examples/yolo_v11/demo.py --mode arch

# 训练模型
python examples/yolo_v11/demo.py --mode train --epochs 5

# 推理演示
python examples/yolo_v11/demo.py --mode inference

# 完整流程
python examples/yolo_v11/demo.py --mode both
```

### 运行测试

```bash
# 单元测试
python -m pytest tests/detection/yolo_v11/test_yolov11_model.py -v

# 集成测试
python -m pytest tests/detection/yolo_v11/test_v11_integration.py -v

# 所有 v11 测试
python -m pytest tests/detection/yolo_v11/ -v
```

---

## YOLO v10 vs YOLO v11 对比

| 特性 | YOLO v10 | YOLO v11 |
|------|----------|----------|
| 开发者 | 清华大学 | Ultralytics |
| 核心模块 | C2fCIB | C3k2 |
| NMS | 不需要 | 需要 |
| SPPF | 无 | 有 |
| Neck | 无 | PANet |
| 精度 | 高 | 更高 |

---

## 常见问题

### 1. C3k2 和 C2f 有什么区别？

| 特性 | C2f | C3k2 |
|------|-----|------|
| 分支数 | 2 | 2 |
| 中间通道 | c_out/2 | min(c_in, c_out) |
| Bottleneck | 简单 | 更灵活 |
| 效率 | 高 | 更高 |

### 2. 为什么 YOLO v11 需要 NMS？

YOLO v11 保持了传统的检测范式：
- 可能产生多个重叠的预测框
- 需要 NMS 进行后处理
- 与 YOLO v8 类似的流程

### 3. SPPF 的作用？

- **增大感受野**: 捕获更多上下文信息
- **多尺度特征**: 融合不同尺度的特征
- **提高精度**: 对大物体检测有帮助

---

## 总结

本教程介绍了使用 nanotorch 实现 YOLO v11 的完整流程：

1. **核心改进**: C3k2 模块、SPPF、PANet Neck
2. **模型架构**: Backbone + Neck + DetectHead
3. **损失函数**: DFL + BCE + CIoU
4. **训练推理**: 完整的训练和推理流程

YOLO v11 是 Ultralytics 的最新力作，在精度和速度上都达到了新的高度，是目前最先进的目标检测模型之一。

---

## 参考资料

1. **Ultralytics YOLOv11**: https://github.com/ultralytics/ultralytics

2. **YOLOv11 文档**: https://docs.ultralytics.com/

3. **nanotorch YOLO v10 教程**: `/docs/tutorials/26-yolov10.md`

4. **nanotorch YOLO v8 教程**: `/docs/tutorials/24-yolov8.md`
