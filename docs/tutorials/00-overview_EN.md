# nanotorch Tutorial Series: Building a Deep Learning Framework from Scratch

## Have You Ever Wondered...

When you type these lines of code, what's happening beneath the surface?

```python
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 2 + 1
y.sum().backward()
print(x.grad)  # tensor([2., 2.])
```

Numbers flowing, gradients computing, parameters updating—all like magic. But behind every magic trick lies precision mechanics.

This tutorial series will guide you through **building a deep learning framework from scratch**. When you write the final line of code, PyTorch will no longer be a black box to you—it will be a precision machine you've taken apart and understood completely.

```
Learning to use tools → You are a craftsman
Understanding how tools work → You are an engineer
Building tools yourself → You are a creator
```

---

## The Feynman Method: The Best Way to Learn is to Teach

This tutorial follows the core philosophy of the Feynman learning method:

1. **Use simple language** - Explain things so a 12-year-old can understand
2. **Use life analogies** - Compare complex concepts to everyday examples
3. **Explain "why"** - Not just "what," but why it's needed
4. **Visual understanding** - Use diagrams and animations to build intuition

---

## Building a House: An Analogy

Think of a deep learning framework like **building a house**:

```
┌─────────────────────────────────────────────────────┐
│                    Your Model                        │
│  ┌─────────────────────────────────────────────┐   │
│  │              Neural Network Layers (nn)      │   │
│  │    Linear · Conv2d · LSTM · Transformer     │   │
│  └─────────────────────────────────────────────┘   │
│                      ↓↑                            │
│  ┌─────────────────────────────────────────────┐   │
│  │              Autograd (Automatic Diff)       │   │
│  │       Compute gradients, enable learning     │   │
│  └─────────────────────────────────────────────┘   │
│                      ↓↑                            │
│  ┌─────────────────────────────────────────────┐   │
│  │               Tensor                         │   │
│  │      Store data, like the foundation         │   │
│  └─────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

| Building Concept | Deep Learning Concept | Purpose |
|------------------|----------------------|---------|
| Foundation | **Tensor** | Store all data (numbers, images, text) |
| Utilities (water/electric) | **Autograd** | Automatically compute gradients, let information flow |
| Rooms and Floors | **Neural Network Layers** | Process data, extract features |
| Decoration | **Activation Functions** | Add non-linearity, make models more flexible |
| Property Management | **Optimizer** | Update parameters, make the model better |

---

## Learning Roadmap

```
                    ┌─────────────┐
                    │   Goal:     │
                    │ Understand  │
                    │ PyTorch     │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         ↓                 ↓                 ↓
    ┌─────────┐      ┌─────────┐      ┌─────────┐
    │ Phase 1 │      │ Phase 2 │      │ Phase 3 │
    │  Core   │      │ Neural  │      │ Advanced│
    │ Basics  │      │ Networks│      │ Apps    │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
    ┌────┴────┐      ┌────┴────┐      ┌────┴────┐
    │ Tensor  │      │ Module  │      │ Conv    │
    │ Autograd│      │ Loss    │      │ RNN     │
    └─────────┘      │ Optim   │      │ Transf. │
                     └─────────┘      │ YOLO    │
                                      └─────────┘
```

### Phase 1: Core Foundations (Essential)

| Tutorial | Topic | One-Line Understanding | Life Analogy |
|----------|-------|------------------------|--------------|
| [01-tensor](01-tensor_EN.md) | Tensor | Multi-dimensional arrays, store all data | A super powerful Excel spreadsheet |
| [02-autograd](02-autograd_EN.md) | Automatic Differentiation | Automatically compute "how to adjust parameters" | GPS recalculating your route |

### Phase 2: Neural Networks (Core)

| Tutorial | Topic | One-Line Understanding | Life Analogy |
|----------|-------|------------------------|--------------|
| [03-nn-module](03-nn-module_EN.md) | Module | Parent class for all layers | Universal interface for LEGO bricks |
| [04-activation](04-activation_EN.md) | Activation Functions | Decide if a neuron "fires" | Switch/dimmer |
| [05-loss](05-loss_EN.md) | Loss Functions | Measure gap between prediction and truth | Exam grading rubric |
| [06-optimizer](06-optimizer_EN.md) | Optimizers | Decide how to update parameters | Choosing the best path up a mountain |

### Phase 3: Training Pipeline

| Tutorial | Topic | One-Line Understanding |
|----------|-------|------------------------|
| [07-training](07-training_EN.md) | Training Loop | Complete training, validation, testing flow |
| [08-transforms](08-transforms_EN.md) | Data Augmentation | Make data more diverse |

### Phase 4: Advanced Layers

| Tutorial | Topic | One-Line Understanding | Life Analogy |
|----------|-------|------------------------|--------------|
| [09-conv](09-conv_EN.md) | Convolution Layers | Scan images with small windows | Flashlight illuminating a room |
| [10-normalization](10-normalization_EN.md) | Normalization | Make data distribution more stable | Standardized grading |
| [11-rnn](11-rnn_EN.md) | Recurrent Networks | Process sequential data | Remembering what you read before |
| [12-transformer](12-transformer_EN.md) | Transformer | Attention mechanism | Focusing on key points in a meeting |

### Phase 5: Advanced Topics

| Tutorial | Topic | One-Line Understanding |
|----------|-------|------------------------|
| [13-dataloader](13-dataloader_EN.md) | Data Loading | Batch, shuffle, parallel loading |
| [14-init](14-init_EN.md) | Parameter Initialization | A good start is half the battle |
| [15-advanced](15-advanced_EN.md) | Advanced Techniques | Gradient clipping, learning rate warmup |

### Phase 6: Object Detection (YOLO)

| Tutorial | Topic | Core Innovation |
|----------|-------|-----------------|
| [17-yolo](17-yolo_EN.md) | YOLO Overview | Object detection fundamentals |
| [18-yolov1](18-yolov1_EN.md) | YOLO v1 | The pioneer: end-to-end detection |
| [18.5-yolov2](18.5-yolov2_EN.md) | YOLO v2 | BatchNorm + Anchor |
| [19-yolov3](19-yolov3_EN.md) | YOLO v3 | Multi-scale detection |
| [20-yolov4](20-yolov4_EN.md) | YOLO v4 | Bag of Freebies |
| [21-yolov5](21-yolov5_EN.md) | YOLO v5 | Engineering improvements |
| [22-yolov6](22-yolov6_EN.md) | YOLO v6 | RepVGG + Decoupled Head |
| [23-yolov7](23-yolov7_EN.md) | YOLO v7 | E-ELAN |
| [24-yolov8](24-yolov8_EN.md) | YOLO v8 | Anchor-free |
| [25-yolov9](25-yolov9_EN.md) | YOLO v9 | GELAN |
| [26-yolov10](26-yolov10_EN.md) | YOLO v10 | NMS-free |
| [27-yolov11](27-yolov11_EN.md) | YOLO v11 | C3k2 + C2PSA |

---

## Prerequisites Check

Before starting, make sure you know:

```
✓ Python basics (classes, functions, list comprehensions)
✓ NumPy basics (array operations, shapes)
? Linear algebra (matrix multiplication)
? Calculus (derivatives, chain rule)
? Deep learning concepts (neural networks, backpropagation)
```

**Not familiar?** No problem! Every place that needs math, I'll explain intuitively.

**Math Fundamentals Elective**: [Math Fundamentals Tutorial](math-fundamentals_EN.md) covers linear algebra, calculus, and probability theory.

---

## Learning Method: The Four Steps

### 1. Understand the Concept (What & Why)
First understand what the concept is and why it's needed.

### 2. Read the Code (How)
Read the implementation, understand what each line does.

### 3. Write It Yourself (Do)
**Don't copy-paste**. Type it out yourself—your fingers have memory.

### 4. Test and Verify (Verify)
Write test cases, compare results with PyTorch.

```python
# Verify your implementation
import torch
import nanotorch as nt

# Same input
x_torch = torch.tensor([1.0, 2.0])
x_nt = nt.Tensor([1.0, 2.0])

# Compare outputs
print(torch_out := x_torch * 2 + 1)
print(nt_out := x_nt * 2 + 1)
print(f"Difference: {torch_out - nt_out.data}")  # Should be close to 0
```

---

## Debugging is Your Friend

When you encounter problems:

```python
# 1. Print shapes - the most common error is shape mismatch
print(f"x.shape = {x.shape}")
print(f"y.shape = {y.shape}")

# 2. Print data types
print(f"x.dtype = {x.data.dtype}")

# 3. Check gradient flow
if x.requires_grad and x.grad is None:
    print("Warning: gradient didn't reach here!")

# 4. Compare with expected
expected = np_function(x.data)
actual = y.data
diff = np.abs(expected - actual).max()
print(f"Max difference: {diff}")  # Should be < 1e-5
```

---

## Project Structure

```
nanotorch/
├── nanotorch/           # The core library you'll implement
│   ├── tensor.py        # Chapter 1: Tensor
│   ├── autograd.py      # Chapter 2: Automatic Differentiation
│   ├── nn/              # Neural network modules
│   │   ├── module.py    # Chapter 3: Base class
│   │   ├── linear.py    # Fully connected layer
│   │   ├── conv.py      # Chapter 9: Convolution
│   │   └── ...
│   └── optim/           # Optimizers
│       └── ...
├── tests/               # Test files
└── docs/tutorials/      # This tutorial
```

---

## Recommended Resources

### Must Read
- [PyTorch Official Documentation](https://pytorch.org/docs/) - API reference
- [Deep Learning](https://www.deeplearningbook.org/) - by Goodfellow

### Open Source References
- [micrograd](https://github.com/karpathy/micrograd) - Karpathy's tiny autograd
- [tinygrad](https://github.com/tinygrad/tinygrad) - Small framework

---

## Ready to Start?

> "If you cannot explain it simply, you do not understand it well enough." — Feynman

Let's begin with [Chapter 1: Tensor Basics](01-tensor_EN.md) and understand the building blocks of deep learning in the simplest way possible!

```python
# Your first nanotorch code
from nanotorch import Tensor

x = Tensor([1, 2, 3, 4, 5])
print(f"This is my first tensor: {x.shape}")

# After completing this series, you'll understand everything behind this code
```
