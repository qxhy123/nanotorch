# 第十三章：数据加载

## 一百万张图片，怎么喂给模型？...

想象你要训练一个图像分类器，数据集有一百万张图片。

你不能把一百万张图一次性塞进内存——那需要上百 GB 的空间，再大的服务器也扛不住。

你也不能一张一张地喂——那样训练太慢，GPU 会饿得发慌。

**你需要分批。** 每次取 32 张或 64 张，打包成一个批次，喂给模型。处理完一批，再取下一批。

这就是 DataLoader 的工作。

```
数据加载的艺术：

  一百万张图片
       ↓
  分成 31,250 个批次（每批 32 张）
       ↓
  每个批次：
    - 随机打乱顺序（防止模型记住顺序）
    - 填充/裁剪到相同大小（方便批处理）
    - 转换成 Tensor（喂给 GPU）
       ↓
  流水线般地送入模型
```

**DataLoader，是数据与模型之间的桥梁。** 它让海量数据变得可管理，让训练变得高效。

---

## 13.1 为什么需要 DataLoader？

### 问题：数据太大

```
你有100万张训练图片

一次全部加载：
  内存需求：100万 × 3通道 × 224×224 = 150GB
  → 内存爆炸！

分批加载：
  每批32张：32 × 3 × 224×224 = 4.8MB
  → 轻松处理
```

### DataLoader 做了什么

```
DataLoader 的职责：

1. 分批：把数据分成小批次
2. 打乱：随机打乱顺序（训练时）
3. 迭代：一个一个批次返回

原始数据 → DataLoader → 迭代器
  [1000个]            → 批次1 [32个]
                      → 批次2 [32个]
                      → 批次3 [32个]
                      → ...
```

---

## 13.2 Dataset：数据容器

### 基类

```python
class Dataset:
    """
    数据集抽象基类

    类比：一本菜谱
      - __len__：一共多少道菜
      - __getitem__：第i道菜是什么
    """

    def __len__(self) -> int:
        """返回数据集大小"""
        raise NotImplementedError

    def __getitem__(self, index: int):
        """获取第 index 个样本"""
        raise NotImplementedError
```

### TensorDataset：最简单的数据集

```python
class TensorDataset(Dataset):
    """
    张量数据集

    把多个数组打包成一个数据集

    类比：把食材和配方绑在一起
    """

    def __init__(self, *tensors):
        # 所有张量的第一维必须相同
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors)
        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def __getitem__(self, index: int):
        # 返回所有张量在 index 位置的数据
        return tuple(t[index] for t in self.tensors)
```

### 使用

```python
import numpy as np
from nanotorch.data import TensorDataset

# 创建数据
X = np.random.randn(1000, 784)  # 1000张图片
y = np.random.randint(0, 10, 1000)  # 1000个标签

# 打包成数据集
dataset = TensorDataset(X, y)

# 访问单个样本
x_sample, y_sample = dataset[0]
print(x_sample.shape)  # (784,)
print(y_sample)        # 标量

# 数据集大小
print(len(dataset))    # 1000
```

---

## 13.3 DataLoader：分批迭代器

### 核心功能

```
DataLoader 参数：

dataset：数据集
batch_size：每批多少个（常用32、64、128）
shuffle：是否打乱（训练时True，验证时False）
drop_last：是否丢弃最后不完整的批次
```

### 实现

```python
class DataLoader:
    """
    数据加载器

    类比：服务员
      - 从厨房(dataset)取菜
      - 按桌数(batch_size)分配
      - 按随机顺序(shuffle)上菜
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        """迭代返回批次数据"""
        # 获取索引
        indices = list(range(len(self.dataset)))

        # 打乱
        if self.shuffle:
            np.random.shuffle(indices)

        # 分批
        batch = []
        for idx in indices:
            batch.append(self.dataset[idx])

            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []

        # 处理最后不完整的批次
        if batch and not self.drop_last:
            yield self._collate(batch)

    def __len__(self) -> int:
        """返回批次数"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _collate(self, batch):
        """把样本列表整理成批量数组"""
        if isinstance(batch[0], tuple):
            # 多个返回值（X, y）
            return tuple(
                np.stack([sample[i] for sample in batch], axis=0)
                for i in range(len(batch[0]))
            )
        else:
            # 单个返回值
            return np.stack(batch, axis=0)
```

### 使用

```python
from nanotorch.data import DataLoader, TensorDataset

# 创建数据集
dataset = TensorDataset(X, y)

# 创建数据加载器
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# 训练循环
for batch_x, batch_y in loader:
    # batch_x: (32, 784)
    # batch_y: (32,)
    print(batch_x.shape, batch_y.shape)
```

---

## 13.4 为什么要打乱数据？

```
不打乱：
  第1批：样本1-32（都是猫）
  第2批：样本33-64（都是狗）
  ...
  问题：模型学"这批是猫，那批是狗"
       而不是学"什么是猫，什么是狗"

打乱：
  第1批：猫、狗、鸟、猫、狗...
  第2批：鸟、猫、狗、狗、猫...
  好处：每批都有各种类别，学习更稳定
```

---

## 13.5 数据集划分

### 训练/验证/测试

```
数据集划分：

全部数据 (100%)
    │
    ├── 训练集 (70%) → 训练模型
    │
    ├── 验证集 (15%) → 调参、选模型
    │
    └── 测试集 (15%) → 最终评估

类比：
  训练集 = 课本（学习用）
  验证集 = 模拟考（调试用）
  测试集 = 高考（最终评价）
```

### random_split

```python
def random_split(dataset, lengths):
    """
    随机划分数据集

    Args:
        dataset: 原数据集
        lengths: 各部分比例或数量

    Example:
        random_split(dataset, [0.7, 0.15, 0.15])  # 比例
        random_split(dataset, [7000, 1500, 1500])  # 数量
    """
    total = len(dataset)

    # 比例转数量
    if any(l < 1 for l in lengths):
        lengths = [int(l * total) for l in lengths]

    # 确保总数正确
    lengths[-1] = total - sum(lengths[:-1])

    # 随机打乱索引
    indices = np.random.permutation(total).tolist()

    # 划分
    subsets = []
    offset = 0
    for length in lengths:
        subset_indices = indices[offset:offset + length]
        subsets.append(Subset(dataset, subset_indices))
        offset += length

    return subsets
```

### 使用

```python
from nanotorch.data import TensorDataset, random_split, DataLoader

# 创建数据集
dataset = TensorDataset(X, y)

# 划分
train_set, val_set, test_set = random_split(
    dataset,
    [0.7, 0.15, 0.15]
)

print(len(train_set))  # 700
print(len(val_set))    # 150
print(len(test_set))   # 150

# 创建加载器
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)  # 验证不打乱
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
```

---

## 13.6 自定义 Dataset

### 从文件加载

```python
class ImageDataset(Dataset):
    """
    图像数据集

    从磁盘加载图片
    """

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # 加载图片
        image = self._load_image(self.image_paths[index])
        label = self.labels[index]

        # 应用变换
        if self.transform:
            image = self.transform(image)

        return image, label

    def _load_image(self, path):
        # 实际应用中使用 PIL 或 cv2
        # 这里简化处理
        return np.random.randn(224, 224, 3).astype(np.float32)


# 使用
dataset = ImageDataset(
    image_paths=['img1.jpg', 'img2.jpg', ...],
    labels=[0, 1, ...],
    transform=transform_pipeline
)
loader = DataLoader(dataset, batch_size=32)
```

---

## 13.7 完整训练流程

```python
import numpy as np
from nanotorch import Tensor
from nanotorch.data import TensorDataset, random_split, DataLoader
from nanotorch.nn import Linear, ReLU, Sequential
from nanotorch.nn.loss import CrossEntropyLoss
from nanotorch.optim import Adam

# 1. 准备数据
X = np.random.randn(5000, 784).astype(np.float32)
y = np.random.randint(0, 10, 5000).astype(np.int64)

# 2. 创建数据集并划分
dataset = TensorDataset(X, y)
train_set, val_set = random_split(dataset, [0.8, 0.2])

# 3. 创建数据加载器
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# 4. 定义模型
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 5. 训练
for epoch in range(10):
    # 训练
    model.train()
    for batch_x, batch_y in train_loader:
        x = Tensor(batch_x)
        y = Tensor(batch_y)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    correct = 0
    total = 0
    for batch_x, batch_y in val_loader:
        x = Tensor(batch_x)
        output = model(x)
        pred = np.argmax(output.data, axis=1)
        correct += np.sum(pred == batch_y)
        total += len(batch_y)

    acc = correct / total
    print(f"Epoch {epoch+1}, Val Acc: {acc:.4f}")
```

---

## 13.8 常见陷阱

### 陷阱1：验证集也打乱

```python
# 错误
val_loader = DataLoader(val_set, shuffle=True)

# 正确：验证集不需要打乱
val_loader = DataLoader(val_set, shuffle=False)
```

### 陷阱2：batch_size 太大

```python
# 问题：GPU 内存不足
loader = DataLoader(dataset, batch_size=1024)  # 太大

# 解决：减小 batch_size
loader = DataLoader(dataset, batch_size=32)  # 合适
```

### 陷阱3：忘记转换为 Tensor

```python
# 错误：numpy 数组直接喂给模型
for batch_x, batch_y in loader:
    output = model(batch_x)  # 类型错误！

# 正确：转换为 Tensor
for batch_x, batch_y in loader:
    x = Tensor(batch_x)
    output = model(x)
```

---

## 13.9 一句话总结

| 概念 | 一句话 |
|------|--------|
| Dataset | 数据容器，提供索引访问 |
| DataLoader | 分批、打乱、迭代 |
| batch_size | 每批多少个样本 |
| shuffle | 是否打乱顺序 |
| random_split | 划分训练/验证集 |

---

## 下一章

现在我们学会了数据加载！

下一章，我们将学习**参数初始化** —— 好的初始化让训练事半功倍。

→ [第十四章：参数初始化](14-init.md)

```python
# 预告：下一章你将学到
kaiming_normal_(linear.weight)  # ReLU 用 Kaiming
xavier_normal_(linear.weight)   # Tanh 用 Xavier
```
