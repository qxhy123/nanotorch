# nanotorch

An educational PyTorch-inspired library built from scratch on NumPy, with a companion Transformer visualization app in the same repository.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

## Overview

nanotorch has two aligned tracks in one repository:

### Core Python Library

The `nanotorch` package focuses on readable implementations of PyTorch-style building blocks for learning and experimentation.

- `Tensor` with reverse-mode autograd
- `Function.apply(...)`-based operation definitions with shared backward traversal
- Neural network modules for linear layers, convolutions, pooling, normalization, attention, embeddings, RNNs, and Transformers
- Optimizers and schedulers including SGD, Adam, AdamW, RMSprop, Adagrad, StepLR, CosineAnnealingLR, and warmup helpers
- Data utilities such as `Dataset`, `DataLoader`, `TensorDataset`, and `random_split`
- Tokenizers, transforms, and selected experimental subsystems in the repository
- CPU-first execution with optional CuPy-backed CUDA/device helpers when that dependency is installed

### Visualization App

This repository also includes a Transformer visualization application made of a frontend and backend. It is a companion app for exploring model internals, not part of the published `nanotorch` Python package metadata.

- Frontend: interactive views for embeddings, attention, layer flow, tokenization, inference, and training metrics
- Backend: FastAPI endpoints that run nanotorch Transformer components and serve visualization data
- Startup: both frontend and backend are required for the full app experience

For visualization-specific setup, see [`QUICKSTART.md`](./QUICKSTART.md) and [`README_VISUALIZATION.md`](./README_VISUALIZATION.md).

## Installation

```bash
git clone https://github.com/qxhy123/nanotorch.git
cd nanotorch

uv venv
source .venv/bin/activate
uv sync
```

Or install the package in editable mode:

```bash
pip install -e .
```

Optional CUDA support requires a compatible CuPy package installed separately.

## Quick Start

### Train a small network

```python
import numpy as np
from nanotorch import Tensor
from nanotorch.nn import Sequential, Linear, ReLU, CrossEntropyLoss
from nanotorch.optim import SGD

model = Sequential(
    Linear(4, 16),
    ReLU(),
    Linear(16, 3),
)

inputs = Tensor.randn((8, 4))
targets = Tensor(np.random.randint(0, 3, size=(8,)))
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.01)

logits = model(inputs)
loss = criterion(logits, targets)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Compute gradients directly

```python
from nanotorch import Tensor

x = Tensor([2.0], requires_grad=True)
y = x * x + 3 * x + 1
y.backward()

print(x.grad.numpy())
```

## Visualization App

Start both services from the repository root:

```bash
# terminal 1
./start-backend.sh

# terminal 2
./start-frontend.sh
```

Then open `http://localhost:5173`.

If you prefer manual startup or want API request examples, use [`QUICKSTART.md`](./QUICKSTART.md). For the visualization architecture and feature inventory, use [`README_VISUALIZATION.md`](./README_VISUALIZATION.md).

## Package Highlights

### Core modules

- `nanotorch.tensor`: tensor operations, gradient tracking, `no_grad`
- `nanotorch.autograd`: `Function`, `FunctionContext`, shared `backward(...)`
- `nanotorch.nn`: layers, losses, attention, Transformer and RNN building blocks
- `nanotorch.optim`: optimizers and learning-rate schedulers
- `nanotorch.data`: datasets, samplers, dataloaders
- `nanotorch.transforms`: image-style preprocessing and augmentation helpers
- `nanotorch.tokenizer`: char, word, and BPE tokenizers
- `nanotorch.device` / `nanotorch.backend`: CPU/CUDA and backend abstractions

### Repository extras

The repository also contains documentation, examples, benchmarks, a visualization backend/frontend, and selected experimental areas such as detection and generative model work.

## Project Structure

```text
nanotorch/
├── nanotorch/              # Python package
├── docs/                   # Design notes, API docs, tutorials
├── tests/                  # Test suite
├── examples/               # Example scripts
├── benchmarks/             # Micro-benchmarks
├── frontend/               # Visualization frontend
├── backend/                # Visualization backend
├── QUICKSTART.md           # Visualization quick start
└── README_VISUALIZATION.md # Visualization-specific guide
```

## Documentation

- [`docs/design.md`](./docs/design.md): architecture and autograd design
- [`docs/api.md`](./docs/api.md): public API reference and examples
- [`docs/autograd_boundaries.md`](./docs/autograd_boundaries.md): intentional raw-array boundaries after autograd unification
- [`QUICKSTART.md`](./QUICKSTART.md): frontend/backend startup for the visualization app

## Testing

```bash
python -m pytest tests/ -v
```

## Limitations

- The project is educational first, not a drop-in replacement for PyTorch.
- CPU execution is the default path; CUDA support depends on optional CuPy installation and coverage varies by subsystem.
- Some advanced or experimental modules in the repository are less mature than the core tensor/autograd/nn/optim path.

## License

MIT.
