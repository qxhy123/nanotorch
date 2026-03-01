# nanotorch Tutorial Series: Building a Deep Learning Framework from Scratch

## Tutorial Introduction

This tutorial series will guide you through implementing a complete deep learning framework—nanotorch—from scratch. Through this series, you will deeply understand:

- **Tensor** underlying implementation and mathematical operations
- **Autograd** working principles
- **Neural Network Layers** design and implementation
- **Optimizers** how to update model parameters
- **Core mechanisms** like convolution, recurrence, and attention

## Why Learn This?

1. **Deep Understanding of PyTorch**: Learn how PyTorch works internally
2. **Reinforce Deep Learning Fundamentals**: Complete implementation from math to code
3. **Improve Programming Skills**: Learn to design APIs and organize code structure
4. **Interview Advantage**: Demonstrate your understanding of underlying principles

## Prerequisites

- Python programming basics
- NumPy array operations
- Basic linear algebra (matrix operations)
- Basic calculus (derivatives, chain rule)
- Deep learning fundamentals (neural networks, backpropagation)

> 💡 **Tip**: If you need to review math fundamentals, start with [Math Fundamentals: Deep Learning Essentials](math-fundamentals_EN.md), covering linear algebra, calculus, probability theory, and optimization.

## Tutorial Directory

### Math Fundamentals (Optional)

| Tutorial | Topic | Content |
|----------|-------|---------|
| [math-fundamentals_EN.md](math-fundamentals_EN.md) | Math Fundamentals | Linear algebra, calculus, probability theory, optimization |

### Part 1: Core Foundations

| Tutorial | Topic | Content |
|----------|-------|---------|
| [01-tensor_EN.md](01-tensor_EN.md) | Tensor Basics | Tensor data structure, operations, shape manipulation |
| [02-autograd_EN.md](02-autograd_EN.md) | Automatic Differentiation | Computational graphs, backpropagation, gradient computation |

### Part 2: Neural Network Modules

| Tutorial | Topic | Content |
|----------|-------|---------|
| [03-nn-module_EN.md](03-nn-module_EN.md) | Module Base Class | Parameter management, module composition, Sequential |
| [04-activation_EN.md](04-activation_EN.md) | Activation Functions | ReLU, Sigmoid, Softmax, etc. |
| [05-loss_EN.md](05-loss_EN.md) | Loss Functions | MSE, CrossEntropy, BCE |
| [06-optimizer_EN.md](06-optimizer_EN.md) | Optimizers | SGD, Adam, learning rate scheduling |

### Part 3: Training and Data

| Tutorial | Topic | Content |
|----------|-------|---------|
| [07-training_EN.md](07-training_EN.md) | Training Loop | Complete training process, validation, model saving |
| [08-transforms_EN.md](08-transforms_EN.md) | Data Augmentation | Image transforms, normalization, data augmentation |

### Part 4: Advanced Layers

| Tutorial | Topic | Content |
|----------|-------|---------|
| [09-conv_EN.md](09-conv_EN.md) | Convolution Layers | Conv1D/2D/3D, transposed convolution |
| [10-normalization_EN.md](10-normalization_EN.md) | Normalization | BatchNorm, LayerNorm, GroupNorm |
| [11-rnn_EN.md](11-rnn_EN.md) | Recurrent Networks | RNN, LSTM, GRU |
| [12-transformer_EN.md](12-transformer_EN.md) | Transformer | Attention mechanism, positional encoding, multi-head attention |

### Part 5: Advanced Topics

| Tutorial | Topic | Content |
|----------|-------|---------|
| [13-dataloader_EN.md](13-dataloader_EN.md) | Data Loading | Dataset, DataLoader, samplers |
| [14-init_EN.md](14-init_EN.md) | Parameter Initialization | Xavier, Kaiming, orthogonal initialization |
| [15-advanced_EN.md](15-advanced_EN.md) | Advanced Topics | Gradient clipping, learning rate warmup, debugging techniques |

## Project Structure

```
nanotorch/
├── nanotorch/                 # Core library
│   ├── __init__.py
│   ├── tensor.py             # Tensor implementation
│   ├── autograd.py           # Automatic differentiation
│   ├── utils.py              # Utility functions (gradient clipping, initialization, etc.)
│   ├── nn/                   # Neural network modules
│   │   ├── __init__.py
│   │   ├── module.py         # Module base class
│   │   ├── linear.py         # Fully connected layer
│   │   ├── conv.py           # Convolution layers
│   │   ├── activation.py     # Activation functions
│   │   ├── loss.py           # Loss functions
│   │   ├── dropout.py        # Dropout
│   │   ├── pooling.py        # Pooling layers
│   │   ├── normalization.py  # Normalization layers
│   │   ├── rnn.py            # RNN/LSTM/GRU
│   │   ├── attention.py      # Attention mechanism
│   │   ├── transformer.py    # Transformer
│   │   └── embedding.py      # Embedding layer
│   ├── optim/                # Optimizers
│   │   ├── __init__.py
│   │   ├── optimizer.py      # Optimizer base class
│   │   ├── sgd.py            # SGD
│   │   ├── adam.py           # Adam
│   │   ├── adamw.py          # AdamW
│   │   ├── rmsprop.py        # RMSprop
│   │   ├── adagrad.py        # Adagrad
│   │   └── lr_scheduler.py   # Learning rate schedulers
│   ├── data/                 # Data loading
│   │   └── __init__.py       # Dataset, DataLoader
│   └── transforms/           # Data augmentation
│       └── __init__.py       # Image transforms
├── tests/                    # Tests (write while learning)
│   ├── test_tensor.py
│   ├── test_autograd.py
│   ├── test_nn.py
│   ├── test_optim.py
│   └── ...
├── examples/                 # Example code
│   ├── simple_neural_net.py
│   ├── mnist_classifier.py
│   ├── mini_gpt.py
│   └── ...
├── benchmarks/               # Performance benchmarks
├── docs/                     # Documentation
│   ├── tutorials/            # This tutorial
│   ├── api.md
│   └── design.md
├── README.md                 # English documentation
├── README_CN.md              # Chinese documentation
└── pyproject.toml            # Project configuration
```

## Learning Suggestions

### Write While Learning

After completing each chapter, it's recommended to:

1. **Implement it yourself**: Don't just read the code, type it out
2. **Write test cases**: Verify your implementation is correct
3. **Debug and experiment**: Print intermediate results to understand data flow
4. **Compare with PyTorch**: Use the same data to verify outputs match

### Debugging Tips

```python
# Print tensor shapes to understand dimension changes
print(f"Input shape: {x.shape}")
print(f"Output shape: {y.shape}")

# Print gradients to understand backpropagation
print(f"Weight gradient: {w.grad}")

# Compare with NumPy to verify computation
expected = np_function(x.data)
actual = y.data
print(f"Difference: {np.abs(expected - actual).max()}")
```

## Recommended Resources

### Must Read

- [PyTorch Official Documentation](https://pytorch.org/docs/): API reference
- [Deep Learning](https://www.deeplearningbook.org/): by Goodfellow et al.
- [Automatic Differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation): Wikipedia

### Open Source References

- [micrograd](https://github.com/karpathy/micrograd): Karpathy's tiny autograd
- [tinygrad](https://github.com/tinygrad/tinygrad): Small deep learning framework
- [PyTorch Source](https://github.com/pytorch/pytorch): Official implementation

## Start Learning

Ready? Let's begin with [Chapter 1: Tensor Basics](01-tensor_EN.md)!

```python
# Your first nanotorch code
from nanotorch import Tensor

x = Tensor([1, 2, 3, 4, 5])
print(x)  # Tensor(shape=(5,), requires_grad=False)

y = x * 2 + 1
print(y)  # Tensor(shape=(5,), ...)
```
