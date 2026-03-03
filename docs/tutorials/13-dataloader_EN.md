# Tutorial 13: Data Loading

## A Million Images, How to Feed Them to the Model?...

Imagine you want to train an image classifier with a dataset of one million images.

You can't stuff all one million images into memory at once—that would require hundreds of GB of space, which even the largest servers can't handle.

You can't feed them one by one either—that would make training too slow, leaving the GPU starving.

**You need to batch.** Take 32 or 64 images at a time, package them into a batch, and feed them to the model. Process one batch, then fetch the next.

This is what DataLoader does.

```
The Art of Data Loading:

  One million images
       ↓
  Split into 31,250 batches (32 per batch)
       ↓
  Each batch:
    - Randomly shuffle order (prevent model from memorizing sequence)
    - Pad/crop to same size (for batch processing)
    - Convert to Tensor (feed to GPU)
       ↓
  Flow into the model like a pipeline
```

**DataLoader is the bridge between data and model.** It makes massive data manageable and training efficient.

---

## 13.1 Why Do We Need DataLoader?

### Problem: Data is Too Large

```
You have 1 million training images

Load all at once:
  Memory requirement: 1M × 3 channels × 224×224 = 150GB
  → Memory explosion!

Load in batches:
  Each batch of 32: 32 × 3 × 224×224 = 4.8MB
  → Easy to handle
```

### What DataLoader Does

```
DataLoader's responsibilities:

1. Batching: Split data into small batches
2. Shuffling: Randomly shuffle order (during training)
3. Iteration: Return batches one by one

Raw data → DataLoader → Iterator
  [1000 items]        → Batch 1 [32 items]
                      → Batch 2 [32 items]
                      → Batch 3 [32 items]
                      → ...
```

---

## 13.2 Dataset: Data Container

### Base Class

```python
class Dataset:
    """
    Dataset abstract base class

    Analogy: A cookbook
      - __len__: How many dishes total
      - __getitem__: What is the i-th dish
    """

    def __len__(self) -> int:
        """Return dataset size"""
        raise NotImplementedError

    def __getitem__(self, index: int):
        """Get the index-th sample"""
        raise NotImplementedError
```

### TensorDataset: Simplest Dataset

```python
class TensorDataset(Dataset):
    """
    Tensor dataset

    Packages multiple arrays into one dataset

    Analogy: Binding ingredients and recipes together
    """

    def __init__(self, *tensors):
        # All tensors must have the same first dimension
        assert all(tensors[0].shape[0] == t.shape[0] for t in tensors)
        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def __getitem__(self, index: int):
        # Return data at index position for all tensors
        return tuple(t[index] for t in self.tensors)
```

### Usage

```python
import numpy as np
from nanotorch.data import TensorDataset

# Create data
X = np.random.randn(1000, 784)  # 1000 images
y = np.random.randint(0, 10, 1000)  # 1000 labels

# Package into dataset
dataset = TensorDataset(X, y)

# Access single sample
x_sample, y_sample = dataset[0]
print(x_sample.shape)  # (784,)
print(y_sample)        # scalar

# Dataset size
print(len(dataset))    # 1000
```

---

## 13.3 DataLoader: Batch Iterator

### Core Parameters

```
DataLoader parameters:

dataset: Dataset
batch_size: How many per batch (commonly 32, 64, 128)
shuffle: Whether to shuffle (True for training, False for validation)
drop_last: Whether to drop the last incomplete batch
```

### Implementation

```python
class DataLoader:
    """
    Data loader

    Analogy: A waiter
      - Takes dishes from kitchen (dataset)
      - Distributes by table count (batch_size)
      - Serves in random order (shuffle)
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
        """Iterate and return batch data"""
        # Get indices
        indices = list(range(len(self.dataset)))

        # Shuffle
        if self.shuffle:
            np.random.shuffle(indices)

        # Batch
        batch = []
        for idx in indices:
            batch.append(self.dataset[idx])

            if len(batch) == self.batch_size:
                yield self._collate(batch)
                batch = []

        # Handle last incomplete batch
        if batch and not self.drop_last:
            yield self._collate(batch)

    def __len__(self) -> int:
        """Return number of batches"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _collate(self, batch):
        """Organize sample list into batch arrays"""
        if isinstance(batch[0], tuple):
            # Multiple return values (X, y)
            return tuple(
                np.stack([sample[i] for sample in batch], axis=0)
                for i in range(len(batch[0]))
            )
        else:
            # Single return value
            return np.stack(batch, axis=0)
```

### Usage

```python
from nanotorch.data import DataLoader, TensorDataset

# Create dataset
dataset = TensorDataset(X, y)

# Create data loader
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# Training loop
for batch_x, batch_y in loader:
    # batch_x: (32, 784)
    # batch_y: (32,)
    print(batch_x.shape, batch_y.shape)
```

---

## 13.4 Why Shuffle Data?

```
Without shuffling:
  Batch 1: Samples 1-32 (all cats)
  Batch 2: Samples 33-64 (all dogs)
  ...
  Problem: Model learns "this batch is cats, that batch is dogs"
           Instead of learning "what is a cat, what is a dog"

With shuffling:
  Batch 1: cat, dog, bird, cat, dog...
  Batch 2: bird, cat, dog, dog, cat...
  Benefit: Each batch has various classes, learning is more stable
```

---

## 13.5 Dataset Splitting

### Train/Validation/Test

```
Dataset splitting:

All data (100%)
    │
    ├── Training set (70%) → Train model
    │
    ├── Validation set (15%) → Tune hyperparameters, select model
    │
    └── Test set (15%) → Final evaluation

Analogy:
  Training set = Textbook (for learning)
  Validation set = Practice exam (for tuning)
  Test set = Final exam (for final evaluation)
```

### random_split

```python
def random_split(dataset, lengths):
    """
    Randomly split dataset

    Args:
        dataset: Original dataset
        lengths: Ratios or counts for each part

    Example:
        random_split(dataset, [0.7, 0.15, 0.15])  # Ratios
        random_split(dataset, [7000, 1500, 1500])  # Counts
    """
    total = len(dataset)

    # Convert ratios to counts
    if any(l < 1 for l in lengths):
        lengths = [int(l * total) for l in lengths]

    # Ensure total is correct
    lengths[-1] = total - sum(lengths[:-1])

    # Randomly shuffle indices
    indices = np.random.permutation(total).tolist()

    # Split
    subsets = []
    offset = 0
    for length in lengths:
        subset_indices = indices[offset:offset + length]
        subsets.append(Subset(dataset, subset_indices))
        offset += length

    return subsets
```

### Usage

```python
from nanotorch.data import TensorDataset, random_split, DataLoader

# Create dataset
dataset = TensorDataset(X, y)

# Split
train_set, val_set, test_set = random_split(
    dataset,
    [0.7, 0.15, 0.15]
)

print(len(train_set))  # 700
print(len(val_set))    # 150
print(len(test_set))   # 150

# Create loaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)  # Don't shuffle validation
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
```

---

## 13.6 Custom Dataset

### Loading from Files

```python
class ImageDataset(Dataset):
    """
    Image dataset

    Loads images from disk
    """

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

        # Apply transform
        if self.transform:
            image = self.transform(image)

        return image, label

    def _load_image(self, path):
        # In practice, use PIL or cv2
        # Simplified here
        return np.random.randn(224, 224, 3).astype(np.float32)


# Usage
dataset = ImageDataset(
    image_paths=['img1.jpg', 'img2.jpg', ...],
    labels=[0, 1, ...],
    transform=transform_pipeline
)
loader = DataLoader(dataset, batch_size=32)
```

---

## 13.7 Complete Training Pipeline

```python
import numpy as np
from nanotorch import Tensor
from nanotorch.data import TensorDataset, random_split, DataLoader
from nanotorch.nn import Linear, ReLU, Sequential
from nanotorch.nn.loss import CrossEntropyLoss
from nanotorch.optim import Adam

# 1. Prepare data
X = np.random.randn(5000, 784).astype(np.float32)
y = np.random.randint(0, 10, 5000).astype(np.int64)

# 2. Create dataset and split
dataset = TensorDataset(X, y)
train_set, val_set = random_split(dataset, [0.8, 0.2])

# 3. Create data loaders
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# 4. Define model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# 5. Train
for epoch in range(10):
    # Train
    model.train()
    for batch_x, batch_y in train_loader:
        x = Tensor(batch_x)
        y = Tensor(batch_y)

        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validate
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

## 13.8 Common Pitfalls

### Pitfall 1: Shuffling Validation Set

```python
# Wrong
val_loader = DataLoader(val_set, shuffle=True)

# Correct: Validation set doesn't need shuffling
val_loader = DataLoader(val_set, shuffle=False)
```

### Pitfall 2: batch_size Too Large

```python
# Problem: GPU memory insufficient
loader = DataLoader(dataset, batch_size=1024)  # Too large

# Solution: Reduce batch_size
loader = DataLoader(dataset, batch_size=32)  # Appropriate
```

### Pitfall 3: Forgetting to Convert to Tensor

```python
# Wrong: Numpy array directly fed to model
for batch_x, batch_y in loader:
    output = model(batch_x)  # Type error!

# Correct: Convert to Tensor
for batch_x, batch_y in loader:
    x = Tensor(batch_x)
    output = model(x)
```

---

## 13.9 One-Line Summary

| Concept | One Line |
|---------|----------|
| Dataset | Data container, provides index access |
| DataLoader | Batch, shuffle, iterate |
| batch_size | How many samples per batch |
| shuffle | Whether to shuffle order |
| random_split | Split train/validation sets |

---

## Next Chapter

Now we've learned data loading!

Next chapter, we'll learn **parameter initialization** — good initialization makes training twice as effective.

→ [Chapter 14: Parameter Initialization](14-init.md)

```python
# Preview: What you'll learn in the next chapter
kaiming_normal_(linear.weight)  # Kaiming for ReLU
xavier_normal_(linear.weight)   # Xavier for Tanh
```
