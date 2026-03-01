# 教程 13：数据加载 (DataLoader)

## 目录

1. [概述](#概述)
2. [Dataset 基类](#dataset-基类)
3. [TensorDataset](#tensordataset)
4. [Sampler](#sampler)
5. [DataLoader 实现](#dataloader-实现)
6. [数据集拆分](#数据集拆分)
7. [使用示例](#使用示例)
8. [总结](#总结)

---

## 概述

在训练深度学习模型时，高效的数据加载至关重要。nanotorch 提供了类似 PyTorch 的数据加载工具：

- **Dataset**：数据集抽象基类
- **TensorDataset**：简单的张量数据集包装器
- **DataLoader**：批量加载、打乱、并行处理
- **Sampler**：控制样本采样顺序
- **random_split**：随机划分数据集

---

## Dataset 基类

### 抽象接口

```python
# nanotorch/data/__init__.py

from abc import ABC, abstractmethod

class Dataset(ABC):
    """数据集抽象基类。
    
    所有自定义数据集都应该继承此类并实现:
    - __len__: 返回数据集大小
    - __getitem__: 根据索引获取单个样本
    """

    @abstractmethod
    def __len__(self) -> int:
        """返回数据集中的样本数量。"""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int):
        """获取指定索引的样本。
        
        Args:
            index: 样本索引
        
        Returns:
            单个样本（可以是任何类型）
        """
        raise NotImplementedError

    def __iter__(self):
        """支持迭代。"""
        for i in range(len(self)):
            yield self[i]
```

### 自定义数据集示例

```python
import numpy as np
from nanotorch.data import Dataset

class ImageDataset(Dataset):
    """自定义图像数据集。"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # 加载图像
        image = self._load_image(self.image_paths[index])
        label = self.labels[index]
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def _load_image(self, path):
        # 简化示例：实际应使用 PIL 或 cv2
        return np.random.randn(224, 224, 3).astype(np.float32)
```

---

## TensorDataset

### 实现

```python
class TensorDataset(Dataset):
    """简单的张量数据集包装器。
    
    将多个张量打包成数据集，每个样本是这些张量对应索引的元组。
    
    Args:
        *tensors: 任意数量的张量，第一维大小必须相同
    
    Example:
        >>> x = np.random.randn(100, 10)
        >>> y = np.random.randint(0, 2, 100)
        >>> dataset = TensorDataset(x, y)
        >>> sample_x, sample_y = dataset[0]
    """

    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors), \
            "所有张量的第一维大小必须相同"
        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range")
        
        # 返回所有张量在 index 位置的数据
        return tuple(t[index] for t in self.tensors)
```

### 使用示例

```python
import numpy as np
from nanotorch.data import TensorDataset

# 创建数据
X = np.random.randn(1000, 784).astype(np.float32)  # 1000 个样本，784 个特征
y = np.random.randint(0, 10, 1000).astype(np.int64)  # 1000 个标签

# 创建数据集
dataset = TensorDataset(X, y)

# 访问单个样本
x_sample, y_sample = dataset[0]
print(x_sample.shape)  # (784,)
print(y_sample)        # 标量

# 数据集大小
print(len(dataset))    # 1000
```

---

## Sampler

### 基类

```python
class Sampler(ABC):
    """采样器基类。
    
    控制 DataLoader 中样本的采样顺序。
    """

    @abstractmethod
    def __iter__(self):
        """返回样本索引的迭代器。"""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """返回样本数量。"""
        raise NotImplementedError
```

### SequentialSampler

```python
class SequentialSampler(Sampler):
    """顺序采样器。
    
    按顺序返回索引 [0, 1, 2, ..., n-1]
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self) -> int:
        return len(self.data_source)
```

### RandomSampler

```python
class RandomSampler(Sampler):
    """随机采样器。
    
    随机打乱索引顺序。
    """

    def __init__(self, data_source, replacement: bool = False, num_samples: Optional[int] = None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples or len(data_source)

    def __iter__(self):
        n = len(self.data_source)
        
        if self.replacement:
            # 有放回采样
            indices = np.random.randint(0, n, self.num_samples)
        else:
            # 无放回采样（打乱）
            indices = np.random.permutation(n)[:self.num_samples]
        
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples
```

### BatchSampler

```python
class BatchSampler(Sampler):
    """批量采样器。
    
    将索引打包成批次。
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        # 处理最后一个不完整的批次
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size
```

---

## DataLoader 实现

### 核心实现

```python
# nanotorch/data/__init__.py

class DataLoader:
    """数据加载器。
    
    将数据集包装成可迭代的批量数据。
    
    Args:
        dataset: 数据集对象
        batch_size: 批次大小
        shuffle: 是否打乱数据
        sampler: 自定义采样器
        batch_sampler: 自定义批量采样器
        drop_last: 是否丢弃最后一个不完整批次
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[BatchSampler] = None,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # 确定采样器
        if batch_sampler is not None:
            # 使用自定义批量采样器
            self.batch_sampler = batch_sampler
        elif sampler is not None:
            # 使用自定义采样器
            self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        else:
            # 使用默认采样器
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
            self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self):
        """迭代返回批次数据。"""
        for batch_indices in self.batch_sampler:
            # 获取批次中所有样本
            batch = [self.dataset[idx] for idx in batch_indices]
            
            # 整理批次数据
            yield self._collate_fn(batch)

    def __len__(self) -> int:
        """返回批次数。"""
        return len(self.batch_sampler)

    def _collate_fn(self, batch):
        """将批次样本整理成批量张量。
        
        Args:
            batch: 样本列表，每个样本可能是元组
        
        Returns:
            整理后的批量数据
        """
        if isinstance(batch[0], tuple):
            # 如果样本是元组，分别堆叠每个元素
            return tuple(
                np.stack([sample[i] for sample in batch], axis=0)
                for i in range(len(batch[0]))
            )
        else:
            # 单个张量
            return np.stack(batch, axis=0)
```

### 使用示例

```python
import numpy as np
from nanotorch.data import DataLoader, TensorDataset

# 创建数据
X = np.random.randn(1000, 784).astype(np.float32)
y = np.random.randint(0, 10, 1000).astype(np.int64)

# 创建数据集和数据加载器
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# 训练循环
for batch_idx, (batch_x, batch_y) in enumerate(loader):
    print(f"Batch {batch_idx}: X shape = {batch_x.shape}, y shape = {batch_y.shape}")
    # Batch 0: X shape = (32, 784), y shape = (32,)
    # ...

print(f"Total batches: {len(loader)}")  # 31 (1000 / 32 = 31.25)
```

---

## 数据集拆分

### Subset

```python
class Subset(Dataset):
    """数据集子集。
    
    从原数据集中选择特定索引的样本。
    """

    def __init__(self, dataset: Dataset, indices: list):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]
```

### random_split

```python
def random_split(
    dataset: Dataset,
    lengths: list,
    generator: Optional[np.random.Generator] = None
) -> list:
    """随机划分数据集。
    
    Args:
        dataset: 原数据集
        lengths: 各子集的大小列表
        generator: 随机数生成器
    
    Returns:
        Subset 列表
    
    Example:
        >>> train_set, val_set, test_set = random_split(dataset, [0.7, 0.15, 0.15])
    """
    # 计算实际长度（支持比例）
    total = len(dataset)
    if any(l < 1 for l in lengths):
        lengths = [int(l * total) for l in lengths]
    
    # 确保长度之和等于数据集大小
    lengths[-1] = total - sum(lengths[:-1])
    
    # 随机打乱索引
    if generator is not None:
        indices = generator.permutation(total).tolist()
    else:
        indices = np.random.permutation(total).tolist()
    
    # 划分索引
    subsets = []
    offset = 0
    for length in lengths:
        subset_indices = indices[offset:offset + length]
        subsets.append(Subset(dataset, subset_indices))
        offset += length
    
    return subsets
```

### 使用示例

```python
from nanotorch.data import TensorDataset, random_split, DataLoader

# 创建数据集
dataset = TensorDataset(
    np.random.randn(10000, 784).astype(np.float32),
    np.random.randint(0, 10, 10000).astype(np.int64)
)

# 划分数据集
train_set, val_set, test_set = random_split(
    dataset,
    [0.7, 0.15, 0.15]  # 70%, 15%, 15%
)

print(f"Train: {len(train_set)}")  # 7000
print(f"Val: {len(val_set)}")      # 1500
print(f"Test: {len(test_set)}")    # 1500

# 创建数据加载器
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
```

---

## 使用示例

### 完整训练流程

```python
import numpy as np
from nanotorch import Tensor
from nanotorch.data import TensorDataset, random_split, DataLoader
from nanotorch.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from nanotorch.optim import Adam

# 1. 准备数据
X = np.random.randn(5000, 784).astype(np.float32)
y = np.random.randint(0, 10, 5000).astype(np.int64)

# 2. 创建数据集并划分
dataset = TensorDataset(X, y)
train_set, val_set, test_set = random_split(dataset, [0.7, 0.15, 0.15])

# 3. 创建数据加载器
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# 4. 定义模型
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 5. 训练循环
for epoch in range(10):
    # 训练阶段
    model.train()  # 设置为训练模式
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        x = Tensor(batch_x)
        y = Tensor(batch_y)
        
        # 前向传播
        output = model(x)
        loss = criterion(output, y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # 验证阶段
    model.eval()  # 设置为评估模式
    val_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in val_loader:
        x = Tensor(batch_x)
        y = Tensor(batch_y)
        
        output = model(x)
        loss = criterion(output, y)
        val_loss += loss.item()
        
        # 计算准确率
        predictions = np.argmax(output.data, axis=1)
        correct += np.sum(predictions == batch_y)
        total += len(batch_y)
    
    print(f"Epoch {epoch+1}: "
          f"Train Loss = {train_loss / len(train_loader):.4f}, "
          f"Val Loss = {val_loss / len(val_loader):.4f}, "
          f"Val Acc = {correct / total * 100:.2f}%")
```

### 自定义 Dataset

```python
class CSVDataet(Dataset):
    """从 CSV 文件加载数据。"""

    def __init__(self, csv_path, transform=None):
        # 简化示例：实际应使用 pandas
        self.data = self._load_csv(csv_path)
        self.transform = transform

    def _load_csv(self, path):
        # 模拟 CSV 加载
        return {
            'features': np.random.randn(1000, 10).astype(np.float32),
            'labels': np.random.randint(0, 2, 1000).astype(np.int64)
        }

    def __len__(self) -> int:
        return len(self.data['labels'])

    def __getitem__(self, index: int):
        x = self.data['features'][index]
        y = self.data['labels'][index]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


# 使用自定义数据集
dataset = CSVDataset('data/train.csv')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### 可变长度序列处理

```python
class SequenceDataset(Dataset):
    """处理可变长度序列。"""

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index]


def collate_variable_length(batch):
    """自定义 collate 函数处理可变长度序列。"""
    sequences, labels = zip(*batch)
    
    # 获取最大长度
    max_len = max(len(seq) for seq in sequences)
    
    # 填充序列
    padded_sequences = np.zeros((len(sequences), max_len), dtype=np.float32)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    
    labels = np.array(labels, dtype=np.int64)
    
    return padded_sequences, labels


# 使用自定义 collate
dataset = SequenceDataset(
    [np.random.randn(np.random.randint(10, 50)).astype(np.float32) for _ in range(100)],
    np.random.randint(0, 2, 100).astype(np.int64)
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)
# 注意：需要修改 DataLoader 支持 collate_fn 参数
```

---

## 总结

本教程介绍了 nanotorch 的数据加载系统：

| 组件 | 作用 |
|------|------|
| **Dataset** | 数据集抽象基类 |
| **TensorDataset** | 简单的张量包装 |
| **Sampler** | 控制采样顺序 |
| **DataLoader** | 批量加载、打乱 |
| **random_split** | 划分数据集 |

### 关键要点

1. **Dataset** 定义如何获取单个样本
2. **DataLoader** 负责批量和打乱
3. **Sampler** 可以自定义采样策略
4. **random_split** 方便划分训练/验证/测试集

### 下一步

在 [教程 14：参数初始化](14-init.md) 中，我们将学习各种参数初始化方法，这对训练稳定性至关重要。

---

**参考资源**：
- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html)
- [Deep Learning Data Pipelines](https://cs230.stanford.edu/blog/datapipeline/)
