# nanotorch

A minimal PyTorch implementation from scratch, designed for educational purposes.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

## Overview

nanotorch is a lightweight implementation of core PyTorch functionality built entirely from scratch using only NumPy. It provides:

- **Tensors** with automatic differentiation and 85+ operations
- **Neural Network Layers**: Linear, Conv1D/2D/3D, ConvTranspose2D/3D, RNN/LSTM/GRU, Transformer
- **Normalization**: BatchNorm1d/2d/3d, LayerNorm, GroupNorm, InstanceNorm1d/2d/3d
- **Activation Functions**: ReLU, GELU, SiLU, LeakyReLU, ELU, PReLU, Softplus, etc.
- **Pooling**: MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
- **Loss Functions**: MSE, L1Loss, SmoothL1Loss, CrossEntropyLoss, BCELoss, BCEWithLogitsLoss, NLLLoss
- **Optimizers**: SGD, Adam, AdamW, RMSprop, Adagrad
- **LR Schedulers**: StepLR, CosineAnnealingLR, LinearWarmup, CosineWarmupScheduler, etc.
- **Data Utilities**: DataLoader, Dataset, TensorDataset, random_split
- **Data Augmentation**: RandomCrop, RandomFlip, ColorJitter, RandomErasing, etc.
- **Model Serialization**: save/load state dicts

## Installation

```bash
# Clone repository
git clone https://github.com/your-username/nanotorch.git
cd nanotorch

# Install with uv (recommended)
uv venv
source .venv/bin/activate
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Basic Neural Network

```python
import numpy as np
from nanotorch import Tensor
from nanotorch.nn import Linear, ReLU, Sequential, CrossEntropyLoss
from nanotorch.optim import SGD

# Create model
model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)

# Sample data
X = Tensor.randn((100, 784))
y = Tensor(np.random.randint(0, 10, (100,)))

# Training setup
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(100):
    predictions = model(X)
    loss = criterion(predictions, y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

### Using DataLoader

```python
from nanotorch import DataLoader, TensorDataset
import numpy as np

# Create dataset
X_train = np.random.randn(1000, 784).astype(np.float32)
y_train = np.random.randint(0, 10, 1000).astype(np.int64)
dataset = TensorDataset(X_train, y_train)

# Create data loader
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training
for X_batch, y_batch in loader:
    X_tensor = Tensor(X_batch)
    y_tensor = Tensor(y_batch)
    # ... training step
```

### RNN / LSTM / GRU

```python
from nanotorch.nn import LSTM, Linear
from nanotorch import Tensor

# Create LSTM model
lstm = LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
fc = Linear(128, 10)

# Forward pass
x = Tensor.randn((32, 10, 64))  # (batch, seq_len, input_size)
output, (h_n, c_n) = lstm(x)
output = fc(output[:, -1, :])  # Use last hidden state
```

### Transformer

```python
from nanotorch.nn import TransformerEncoderLayer, TransformerEncoder, Embedding

# Create transformer encoder
encoder_layer = TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=2048)
encoder = TransformerEncoder(encoder_layer, num_layers=6)

# Embedding layer
embedding = Embedding(num_embeddings=10000, embedding_dim=512)

# Forward pass
tokens = Tensor(np.random.randint(0, 10000, (32, 100)))  # (batch, seq_len)
x = embedding(tokens)
output = encoder(x)
```

### Learning Rate Warmup

```python
from nanotorch.optim import AdamW, CosineWarmupScheduler

optimizer = AdamW(model.parameters(), lr=1e-3)
scheduler = CosineWarmupScheduler(
    optimizer, 
    warmup_epochs=5, 
    max_epochs=100
)

for epoch in range(100):
    train(...)
    scheduler.step()
```

### Data Augmentation

```python
from nanotorch.transforms import (
    Compose, ToFloat, Normalize, 
    RandomHorizontalFlip, RandomCrop, ColorJitter
)

transform = Compose([
    ToFloat(),
    RandomHorizontalFlip(p=0.5),
    RandomCrop(size=224),
    ColorJitter(brightness=0.2, contrast=0.2),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply to image
image = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
augmented = transform(image)
```

## Available Components

### Neural Network Layers

| Layer | Description |
|-------|-------------|
| `Linear` | Fully connected layer |
| `Conv1D/2D/3D` | Convolution layers |
| `ConvTranspose2D/3D` | Transposed convolution |
| `Embedding` | Token embedding layer |
| `RNN` | Vanilla RNN |
| `LSTM` | Long Short-Term Memory |
| `GRU` | Gated Recurrent Unit |
| `TransformerEncoder` | Transformer encoder |
| `MultiheadAttention` | Multi-head attention |

### Normalization Layers

| Layer | Description |
|-------|-------------|
| `BatchNorm1d/2d/3d` | Batch normalization |
| `LayerNorm` | Layer normalization |
| `GroupNorm` | Group normalization |
| `InstanceNorm1d/2d/3d` | Instance normalization |

### Pooling Layers

| Layer | Description |
|-------|-------------|
| `MaxPool1d/2d/3d` | Max pooling |
| `AvgPool1d/2d/3d` | Average pooling |
| `AdaptiveAvgPool2d` | Adaptive average pooling |
| `AdaptiveMaxPool2d` | Adaptive max pooling |

### Activation Functions

| Activation | Description |
|------------|-------------|
| `ReLU` | Rectified Linear Unit |
| `LeakyReLU` | Leaky ReLU |
| `GELU` | Gaussian Error Linear Unit |
| `SiLU` | Sigmoid Linear Unit (Swish) |
| `PReLU` | Parametric ReLU |
| `Sigmoid` | Sigmoid activation |
| `Tanh` | Hyperbolic tangent |
| `Softmax` | Softmax activation |
| `ELU` | Exponential Linear Unit |
| `Softplus` | Softplus activation |

### Loss Functions

| Loss | Description |
|------|-------------|
| `MSE` | Mean Squared Error |
| `L1Loss` | Mean Absolute Error |
| `SmoothL1Loss` | Huber Loss |
| `CrossEntropyLoss` | Cross Entropy |
| `BCELoss` | Binary Cross Entropy |
| `BCEWithLogitsLoss` | BCE with sigmoid |
| `NLLLoss` | Negative Log Likelihood |

### Optimizers

| Optimizer | Description |
|-----------|-------------|
| `SGD` | Stochastic Gradient Descent (with momentum, nesterov) |
| `Adam` | Adam optimizer |
| `AdamW` | Adam with decoupled weight decay |
| `RMSprop` | RMSprop optimizer |
| `Adagrad` | Adagrad optimizer |

### Learning Rate Schedulers

| Scheduler | Description |
|-----------|-------------|
| `StepLR` | Decay by gamma every step_size epochs |
| `MultiStepLR` | Decay at specific milestones |
| `ExponentialLR` | Exponential decay |
| `CosineAnnealingLR` | Cosine annealing |
| `LinearWarmup` | Linear warmup only |
| `WarmupScheduler` | Warmup + any scheduler |
| `CosineWarmupScheduler` | Warmup + cosine annealing |
| `ReduceLROnPlateau` | Reduce when metric stagnates |

### Data Utilities

| Class | Description |
|-------|-------------|
| `Dataset` | Base dataset class |
| `TensorDataset` | Dataset wrapping tensors |
| `DataLoader` | Batch data loader with shuffling |
| `Subset` | Subset of a dataset |
| `random_split` | Randomly split dataset |

### Data Augmentation

| Transform | Description |
|-----------|-------------|
| `Compose` | Chain multiple transforms |
| `Normalize` | Normalize with mean/std |
| `RandomHorizontalFlip` | Random horizontal flip |
| `RandomVerticalFlip` | Random vertical flip |
| `RandomCrop` | Random crop |
| `CenterCrop` | Center crop |
| `RandomResizedCrop` | Random crop + resize |
| `ColorJitter` | Random brightness/contrast/saturation |
| `RandomErasing` | Random region erasing |
| `GaussianBlur` | Gaussian blur |

### Initialization Functions

| Function | Description |
|----------|-------------|
| `xavier_uniform_` | Xavier/Glorot uniform |
| `xavier_normal_` | Xavier/Glorot normal |
| `kaiming_uniform_` | Kaiming/He uniform |
| `kaiming_normal_` | Kaiming/He normal |
| `trunc_normal_` | Truncated normal |
| `orthogonal_` | Orthogonal matrix |
| `sparse_` | Sparse initialization |
| `zeros_` / `ones_` | Constant initialization |

## Tensor Operations

```python
from nanotorch import Tensor

t = Tensor.randn((2, 3, 4))

# Shape operations
t.reshape((6, 4))
t.flatten(start_dim=1)
t.transpose(0, 1)
t.squeeze()
t.expand(4, 3, 4)
t.repeat(2, 1, 1)

# Math operations
t + t
t * t
t.matmul(t.transpose(0, 1))
t.sum(dim=1)
t.mean(dim=0)
t.softmax(dim=-1)

# Splitting & sorting
t.split(split_size=2, dim=0)
t.chunk(chunks=2, dim=0)
values, indices = t.topk(k=2, dim=-1)
values, indices = t.sort(dim=-1, descending=True)
```

## Gradient Utilities

```python
from nanotorch.utils import clip_grad_norm_, clip_grad_value_, get_grad_norm_

# Clip gradient norm
total_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)

# Clip gradient values
clip_grad_value_(model.parameters(), clip_value=0.5)

# Get gradient norm
norm = get_grad_norm_(model.parameters(), norm_type=2.0)
```

## Model Serialization

```python
# Save model
state_dict = model.state_dict()
np.savez('model.npz', **state_dict)

# Load model
state_dict = dict(np.load('model.npz'))
model.load_state_dict(state_dict)
```

## Examples

See `examples/` directory:

| Example | Description |
|---------|-------------|
| `simple_neural_net.py` | Basic neural network |
| `mnist_classifier.py` | CNN for MNIST |
| `mini_gpt.py` | Character-level GPT |
| `chat_llm.py` | Simple chatbot with Transformer |
| `autograd_demo.py` | Autograd demonstrations |
| `conv2d_training.py` | CNN training example |

## Project Structure

```
nanotorch/
├── nanotorch/
│   ├── tensor.py          # Tensor with autograd
│   ├── autograd.py        # Autograd engine
│   ├── utils.py           # Utilities & initialization
│   ├── nn/                # Neural network modules
│   │   ├── linear.py
│   │   ├── conv.py
│   │   ├── rnn.py         # RNN/LSTM/GRU
│   │   ├── transformer.py # Transformer components
│   │   ├── attention.py
│   │   ├── embedding.py
│   │   ├── pooling.py
│   │   ├── normalization.py
│   │   ├── activation.py
│   │   ├── loss.py
│   │   └── dropout.py
│   ├── optim/             # Optimizers
│   │   ├── sgd.py
│   │   ├── adam.py
│   │   ├── adamw.py
│   │   ├── rmsprop.py
│   │   ├── adagrad.py
│   │   └── lr_scheduler.py
│   ├── data/              # Data utilities
│   │   └── __init__.py    # DataLoader, Dataset
│   └── transforms/        # Data augmentation
│       └── __init__.py
├── tests/                 # Test suite (199 tests)
├── examples/              # Example scripts
├── docs/                  # Documentation
└── pyproject.toml
```

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific tests
python -m pytest tests/test_tensor.py -v
python -m pytest tests/test_nn.py -v

# Run with coverage
python -m pytest tests/ --cov=nanotorch
```

## Limitations

- CPU-only (no GPU support)
- Limited operations compared to PyTorch
- No distributed training
- Groups > 1 not implemented for transposed convolutions

## Contributing

Contributions welcome! Please:

1. Follow existing code style
2. Add tests for new functionality
3. Update documentation
4. Ensure all tests pass

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Inspired by PyTorch, micrograd, and tinygrad
- Designed for educational use in understanding deep learning frameworks

## Citation

```bibtex
@software{nanotorch,
  title = {nanotorch: A minimal PyTorch implementation from scratch},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/nanotorch}
}
```

---

[中文文档 (Chinese Documentation)](README_CN.md)
