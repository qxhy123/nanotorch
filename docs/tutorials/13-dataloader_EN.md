# Tutorial 13: Data Loading (DataLoader)

## Table of Contents

1. [Overview](#overview)
2. [Dataset Base Class](#dataset-base-class)
3. [TensorDataset](#tensordataset)
4. [Sampler](#sampler)
5. [DataLoader Implementation](#dataloader-implementation)
6. [Dataset Splitting](#dataset-splitting)
7. [Usage Examples](#usage-examples)
8. [Summary](#summary)

---

## Overview

Efficient data loading is crucial when training deep learning models. nanotorch provides PyTorch-like data loading utilities:

- **Dataset**: Abstract base class for datasets
- **TensorDataset**: Simple tensor dataset wrapper
- **DataLoader**: Batch loading, shuffling, parallel processing
- **Sampler**: Controls sample ordering
- **random_split**: Randomly split datasets

---

## Dataset Base Class

### Abstract Interface

```python
# nanotorch/data/__init__.py

from abc import ABC, abstractmethod

class Dataset(ABC):
    """Abstract base class for datasets.
    
    All custom datasets should inherit from this class and implement:
    - __len__: Return dataset size
    - __getitem__: Get single sample by index
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index: int):
        """Get sample at specified index.
        
        Args:
            index: Sample index
        
        Returns:
            Single sample (can be any type)
        """
        raise NotImplementedError

    def __iter__(self):
        """Support iteration."""
        for i in range(len(self)):
            yield self[i]
```

### Custom Dataset Example

```python
import numpy as np
from nanotorch.data import Dataset

class ImageDataset(Dataset):
    """Custom image dataset."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        # Load image
        image = self._load_image(self.image_paths[index])
        label = self.labels[index]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

    def _load_image(self, path):
        # Simplified example: should use PIL or cv2 in practice
        return np.random.randn(224, 224, 3).astype(np.float32)
```

---

## TensorDataset

### Implementation

```python
class TensorDataset(Dataset):
    """Simple tensor dataset wrapper.
    
    Packages multiple tensors into a dataset, where each sample is a tuple
    of the corresponding indices from these tensors.
    
    Args:
        *tensors: Any number of tensors, first dimension must have same size
    
    Example:
        >>> x = np.random.randn(100, 10)
        >>> y = np.random.randint(0, 2, 100)
        >>> dataset = TensorDataset(x, y)
        >>> sample_x, sample_y = dataset[0]
    """

    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors), \
            "All tensors must have the same size in the first dimension"
        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def __getitem__(self, index: int):
        if index < 0:
            index += len(self)
        
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range")
        
        # Return data at index position for all tensors
        return tuple(t[index] for t in self.tensors)
```

### Usage Example

```python
import numpy as np
from nanotorch.data import TensorDataset

# Create data
X = np.random.randn(1000, 784).astype(np.float32)  # 1000 samples, 784 features
y = np.random.randint(0, 10, 1000).astype(np.int64)  # 1000 labels

# Create dataset
dataset = TensorDataset(X, y)

# Access single sample
x_sample, y_sample = dataset[0]
print(x_sample.shape)  # (784,)
print(y_sample)        # scalar

# Dataset size
print(len(dataset))    # 1000
```

---

## Sampler

### Base Class

```python
class Sampler(ABC):
    """Base class for samplers.
    
    Controls the sampling order of samples in DataLoader.
    """

    @abstractmethod
    def __iter__(self):
        """Return an iterator of sample indices."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""
        raise NotImplementedError
```

### SequentialSampler

```python
class SequentialSampler(Sampler):
    """Sequential sampler.
    
    Returns indices in order [0, 1, 2, ..., n-1]
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
    """Random sampler.
    
    Randomly shuffles the order of indices.
    """

    def __init__(self, data_source, replacement: bool = False, num_samples: Optional[int] = None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples or len(data_source)

    def __iter__(self):
        n = len(self.data_source)
        
        if self.replacement:
            # Sampling with replacement
            indices = np.random.randint(0, n, self.num_samples)
        else:
            # Sampling without replacement (shuffle)
            indices = np.random.permutation(n)[:self.num_samples]
        
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.num_samples
```

### BatchSampler

```python
class BatchSampler(Sampler):
    """Batch sampler.
    
    Packages indices into batches.
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
        
        # Handle last incomplete batch
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size
```

---

## DataLoader Implementation

### Core Implementation

```python
# nanotorch/data/__init__.py

class DataLoader:
    """Data loader.
    
    Wraps a dataset into an iterable of batched data.
    
    Args:
        dataset: Dataset object
        batch_size: Batch size
        shuffle: Whether to shuffle data
        sampler: Custom sampler
        batch_sampler: Custom batch sampler
        drop_last: Whether to drop the last incomplete batch
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
        
        # Determine sampler
        if batch_sampler is not None:
            # Use custom batch sampler
            self.batch_sampler = batch_sampler
        elif sampler is not None:
            # Use custom sampler
            self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        else:
            # Use default sampler
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
            self.batch_sampler = BatchSampler(sampler, batch_size, drop_last)

    def __iter__(self):
        """Iterate and return batch data."""
        for batch_indices in self.batch_sampler:
            # Get all samples in batch
            batch = [self.dataset[idx] for idx in batch_indices]
            
            # Collate batch data
            yield self._collate_fn(batch)

    def __len__(self) -> int:
        """Return number of batches."""
        return len(self.batch_sampler)

    def _collate_fn(self, batch):
        """Collate batch samples into batched tensors.
        
        Args:
            batch: List of samples, each sample may be a tuple
        
        Returns:
            Collated batch data
        """
        if isinstance(batch[0], tuple):
            # If sample is a tuple, stack each element separately
            return tuple(
                np.stack([sample[i] for sample in batch], axis=0)
                for i in range(len(batch[0]))
            )
        else:
            # Single tensor
            return np.stack(batch, axis=0)
```

### Usage Example

```python
import numpy as np
from nanotorch.data import DataLoader, TensorDataset

# Create data
X = np.random.randn(1000, 784).astype(np.float32)
y = np.random.randint(0, 10, 1000).astype(np.int64)

# Create dataset and data loader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch_idx, (batch_x, batch_y) in enumerate(loader):
    print(f"Batch {batch_idx}: X shape = {batch_x.shape}, y shape = {batch_y.shape}")
    # Batch 0: X shape = (32, 784), y shape = (32,)
    # ...

print(f"Total batches: {len(loader)}")  # 31 (1000 / 32 = 31.25)
```

---

## Dataset Splitting

### Subset

```python
class Subset(Dataset):
    """Dataset subset.
    
    Selects specific indices from the original dataset.
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
    """Randomly split dataset.
    
    Args:
        dataset: Original dataset
        lengths: List of sizes for each subset
        generator: Random number generator
    
    Returns:
        List of Subsets
    
    Example:
        >>> train_set, val_set, test_set = random_split(dataset, [0.7, 0.15, 0.15])
    """
    # Calculate actual lengths (supports ratios)
    total = len(dataset)
    if any(l < 1 for l in lengths):
        lengths = [int(l * total) for l in lengths]
    
    # Ensure length sum equals dataset size
    lengths[-1] = total - sum(lengths[:-1])
    
    # Randomly shuffle indices
    if generator is not None:
        indices = generator.permutation(total).tolist()
    else:
        indices = np.random.permutation(total).tolist()
    
    # Split indices
    subsets = []
    offset = 0
    for length in lengths:
        subset_indices = indices[offset:offset + length]
        subsets.append(Subset(dataset, subset_indices))
        offset += length
    
    return subsets
```

### Usage Example

```python
from nanotorch.data import TensorDataset, random_split, DataLoader

# Create dataset
dataset = TensorDataset(
    np.random.randn(10000, 784).astype(np.float32),
    np.random.randint(0, 10, 10000).astype(np.int64)
)

# Split dataset
train_set, val_set, test_set = random_split(
    dataset,
    [0.7, 0.15, 0.15]  # 70%, 15%, 15%
)

print(f"Train: {len(train_set)}")  # 7000
print(f"Val: {len(val_set)}")      # 1500
print(f"Test: {len(test_set)}")    # 1500

# Create data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
```

---

## Usage Examples

### Complete Training Pipeline

```python
import numpy as np
from nanotorch import Tensor
from nanotorch.data import TensorDataset, random_split, DataLoader
from nanotorch.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from nanotorch.optim import Adam

# 1. Prepare data
X = np.random.randn(5000, 784).astype(np.float32)
y = np.random.randint(0, 10, 5000).astype(np.int64)

# 2. Create dataset and split
dataset = TensorDataset(X, y)
train_set, val_set, test_set = random_split(dataset, [0.7, 0.15, 0.15])

# 3. Create data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# 4. Define model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 5. Training loop
for epoch in range(10):
    # Training phase
    model.train()  # Set to training mode
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        x = Tensor(batch_x)
        y = Tensor(batch_y)
        
        # Forward pass
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation phase
    model.eval()  # Set to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in val_loader:
        x = Tensor(batch_x)
        y = Tensor(batch_y)
        
        output = model(x)
        loss = criterion(output, y)
        val_loss += loss.item()
        
        # Calculate accuracy
        predictions = np.argmax(output.data, axis=1)
        correct += np.sum(predictions == batch_y)
        total += len(batch_y)
    
    print(f"Epoch {epoch+1}: "
          f"Train Loss = {train_loss / len(train_loader):.4f}, "
          f"Val Loss = {val_loss / len(val_loader):.4f}, "
          f"Val Acc = {correct / total * 100:.2f}%")
```

### Custom Dataset

```python
class CSVDataset(Dataset):
    """Load data from CSV file."""

    def __init__(self, csv_path, transform=None):
        # Simplified example: should use pandas in practice
        self.data = self._load_csv(csv_path)
        self.transform = transform

    def _load_csv(self, path):
        # Simulate CSV loading
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


# Use custom dataset
dataset = CSVDataset('data/train.csv')
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### Variable-Length Sequence Handling

```python
class SequenceDataset(Dataset):
    """Handle variable-length sequences."""

    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int):
        return self.sequences[index], self.labels[index]


def collate_variable_length(batch):
    """Custom collate function for variable-length sequences."""
    sequences, labels = zip(*batch)
    
    # Get maximum length
    max_len = max(len(seq) for seq in sequences)
    
    # Pad sequences
    padded_sequences = np.zeros((len(sequences), max_len), dtype=np.float32)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    
    labels = np.array(labels, dtype=np.int64)
    
    return padded_sequences, labels


# Use custom collate
dataset = SequenceDataset(
    [np.random.randn(np.random.randint(10, 50)).astype(np.float32) for _ in range(100)],
    np.random.randint(0, 2, 100).astype(np.int64)
)

loader = DataLoader(dataset, batch_size=16, shuffle=True)
# Note: Need to modify DataLoader to support collate_fn parameter
```

---

## Summary

This tutorial introduced nanotorch's data loading system:

| Component | Function |
|-----------|----------|
| **Dataset** | Abstract base class for datasets |
| **TensorDataset** | Simple tensor wrapper |
| **Sampler** | Controls sampling order |
| **DataLoader** | Batch loading, shuffling |
| **random_split** | Split datasets |

### Key Points

1. **Dataset** defines how to get individual samples
2. **DataLoader** handles batching and shuffling
3. **Sampler** allows custom sampling strategies
4. **random_split** conveniently splits train/val/test sets

### Next Steps

In [Tutorial 14: Parameter Initialization](14-init.md), we will learn about various parameter initialization methods, which are crucial for training stability.

---

**References**:
- [PyTorch DataLoader Documentation](https://pytorch.org/docs/stable/data.html)
- [Deep Learning Data Pipelines](https://cs230.stanford.edu/blog/datapipeline/)
