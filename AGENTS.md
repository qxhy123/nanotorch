# nanotorch - Agent Knowledge Base

**Generated:** 2026-02-09  
**Project:** nanotorch - minimal PyTorch implementation from scratch  
**Goal:** Educational implementation of core PyTorch concepts: tensors, autograd, nn modules, optimizers

---

## QUICK START

### Environment Setup
```bash
# Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv sync
```

### Core Commands
```bash
# Run tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_tensor.py -v

# Run examples
python examples/simple_neural_net.py
```

---

## BUILD & TEST COMMANDS

### Package Management
```bash
# Install dependencies
uv sync

# Install development dependencies  
uv sync --group dev

# Update lock file
uv lock
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_tensor.py -v
python -m pytest tests/test_autograd.py -v
python -m pytest tests/test_nn.py -v
python -m pytest tests/test_optim.py -v

# Run tests with coverage
python -m pytest tests/ --cov=nanotorch --cov-report=html

# Run tests matching pattern
python -m pytest tests/ -v -k "tensor"

# Run with detailed output
python -m pytest tests/test_tensor.py -v -s
```

### Development & Debugging
```bash
# Run benchmark
python benchmarks/tensor_operations.py

# Run memory profiling
python -m memory_profiler examples/simple_neural_net.py

# Run type checking
python -m mypy nanotorch/ --strict

# Run linting
black --check nanotorch/
flake8 nanotorch/
```

---

## CODE STYLE GUIDELINES

### Import Organization
```python
# Standard library imports
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Union, Any, Callable
from collections import OrderedDict
from contextlib import contextmanager

# Third-party imports (minimal dependencies)
import numpy as np
from numpy.typing import NDArray

# Local imports
from nanotorch.tensor import Tensor
from nanotorch.autograd import Function, backward
from nanotorch.nn import Module, Linear, ReLU, MSE
from nanotorch.optim import SGD, Adam
```

### Formatting Conventions
- **Line length**: 88-100 characters (black-compatible)
- **Indentation**: 4 spaces (no tabs)
- **Quotes**: Double quotes for strings, single quotes for characters
- **Type hints**: Mandatory for function arguments and return values
- **Docstrings**: Google style with Args/Returns/Raises sections
- **Imports**: Grouped as standard library → third-party → local imports

### Naming Conventions
```python
# Classes: PascalCase
class Tensor:
class Linear:
class SGD:

# Functions/Methods: snake_case
def compute_gradient():
def backward_pass():

# Variables: snake_case
learning_rate = 0.001
weight_tensor = Tensor(...)

# Constants: UPPER_SNAKE_CASE  
MAX_TENSOR_DIMS = 8
FLOAT32_EPSILON = 1e-7

# Private: _leading_underscore
_private_helper_function()
```

### Error Handling
```python
# Use specific exceptions
def reshape(self, new_shape):
    if np.prod(new_shape) != np.prod(self.shape):
        raise ValueError(
            f"Cannot reshape tensor of shape {self.shape} to {new_shape}. "
            f"Total elements must match."
        )
    
    try:
        data = self.data.reshape(new_shape)
    except ValueError as e:
        raise RuntimeError(f"Failed to reshape tensor: {e}")
    
    return Tensor(data, requires_grad=self.requires_grad)

# Use context managers for resources
with no_grad():
    # operations that shouldn't track gradients
    output = model(input_data)

# Validate inputs early
def add(t1, t2):
    if t1.shape != t2.shape:
        raise RuntimeError(f"Shape mismatch: {t1.shape} vs {t2.shape}")
    # ... implementation
```

### Type Hints (Mandatory)
```python
from typing import Optional, Union, List, Dict, Tuple, Any, Callable

def matmul(
    t1: Tensor,
    t2: Tensor,
    transpose_a: bool = False,
    transpose_b: bool = False
) -> Tensor:
    """Matrix multiplication with optional transposition."""
    # Validate inputs
    if t1.ndim != 2 or t2.ndim != 2:
        raise ValueError("matmul requires 2D tensors")
    
    # Implementation
    return result

def backward(
    tensor: Tensor,
    gradient: Optional[NDArray] = None
) -> None:
    """Perform backward pass through computational graph."""
    if gradient is None:
        gradient = np.ones_like(tensor.data)
    # ... implementation
```

---

## ARCHITECTURE & EXTENSION GUIDES

### Project Structure
```
nanotorch/
├── nanotorch/           # Core library
│   ├── tensor.py       # Tensor class with operations
│   ├── autograd.py     # Automatic differentiation engine
│   ├── nn/             # Neural network modules
│   │   ├── __init__.py
│   │   ├── module.py   # Base Module class
│   │   ├── linear.py   # Linear layer
│   │   ├── activation.py # Activation functions
│   │   └── loss.py     # Loss functions
│   ├── optim/          # Optimizers
│   │   ├── __init__.py
│   │   ├── optimizer.py # Base Optimizer class
│   │   ├── sgd.py      # SGD optimizer
│   │   └── adam.py     # Adam optimizer
│   └── utils.py        # Utility functions
├── tests/              # Test suite
│   ├── test_tensor.py
│   ├── test_autograd.py
│   ├── test_nn.py
│   ├── test_optim.py
│   └── conftest.py
├── examples/           # Example scripts
│   ├── simple_neural_net.py
│   ├── mnist_classifier.py
│   └── autograd_demo.py
├── benchmarks/         # Performance benchmarks
│   ├── tensor_operations.py
│   └── memory_usage.py
└── docs/              # Documentation
    ├── design.md
    └── api.md
```

### Core Design Principles
1. **Educational Focus**: Clear, readable code that teaches PyTorch internals
2. **Minimal Dependencies**: Only NumPy as external dependency
3. **Explicit over Implicit**: Clear computational graph building and traversal
4. **Pythonic API**: Follow PyTorch-like API where educational value exists
5. **Test-Driven**: Comprehensive test suite for all core functionality

### Tensor Implementation Guidelines
```python
class Tensor:
    def __init__(self, data, requires_grad=False, _op=None, _parents=()):
        self.data = np.asarray(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._op = _op  # Operation that created this tensor
        self._parents = _parents  # Parent tensors in computational graph
        
        if requires_grad:
            self.zero_grad()
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
    
    def backward(self, gradient=None):
        """Implement backward pass through computational graph."""
        if gradient is None:
            gradient = np.ones_like(self.data)
        
        if self.grad is None:
            self.grad = gradient
        else:
            self.grad += gradient
        
        if self._op is not None:
            self._op.backward(gradient, self._parents)
```

### Adding New Operations
1. **Create operation class** that inherits from `Function`:
```python
class Add(Function):
    @staticmethod
    def forward(ctx, t1, t2):
        result = Tensor(t1.data + t2.data, _op=Add, _parents=(t1, t2))
        if t1.requires_grad or t2.requires_grad:
            result.requires_grad = True
        return result
    
    @staticmethod
    def backward(ctx, grad_output, parents):
        t1, t2 = parents
        if t1.requires_grad:
            t1.backward(grad_output)
        if t2.requires_grad:
            t2.backward(grad_output)
```

2. **Add convenience function** to tensor class:
```python
class Tensor:
    def add(self, other):
        return Add.apply(self, other)
    
    def __add__(self, other):
        return self.add(other)
```

### Adding New Neural Network Layers
1. **Create layer class** that inherits from `Module`:
```python
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights and biases
        scale = math.sqrt(2.0 / in_features)
        self.weight = Tensor(
            np.random.randn(in_features, out_features) * scale,
            requires_grad=True
        )
        self.bias = Tensor(
            np.zeros((1, out_features)),
            requires_grad=True
        )
        
        self.register_parameter('weight', self.weight)
        self.register_parameter('bias', self.bias)
    
    def forward(self, x):
        return x.matmul(self.weight).add(self.bias)
```

---

## DEVELOPMENT WORKFLOW

### Testing Changes
```bash
# Run all tests to ensure no regression
python -m pytest tests/ -v

# Run specific test for the feature you're implementing
python -m pytest tests/test_nn.py::TestLinear -v

# Run benchmarks to check performance impact
python benchmarks/tensor_operations.py --op matmul
```

### Performance Considerations
- **Vectorization**: Use NumPy operations instead of Python loops
- **Memory**: Be careful with tensor copies; use views where possible
- **Graph Size**: Computational graphs can grow large; implement pruning
- **Gradient Accumulation**: Support gradient accumulation for training

### Code Quality Checks
```bash
# Type checking
python -m mypy nanotorch/ --strict

# Linting
black --check nanotorch/
flake8 nanotorch/

# Import sorting
isort --check-only nanotorch/

# Security scanning (if applicable)
bandit -r nanotorch/
```

---

## ANTI-PATTERNS TO AVOID

### ❌ Forbidden Patterns
- **Never** use `as any` or ignore type errors
- **Never** modify tensor data in-place without proper gradient tracking
- **Never** create memory leaks by holding references to large tensors
- **Never** use global state in autograd engine
- **Never** implement operations without proper gradient computation

### ✅ Preferred Patterns
- **Always** implement forward and backward passes for operations
- **Always** validate tensor shapes before operations
- **Always** use NumPy's broadcasting rules consistently
- **Always** clear gradients before backward pass in training loops
- **Always** write tests for new operations

### Project-Specific Constraints
- **No PyTorch dependency**: This is a from-scratch implementation
- **Educational clarity**: Code should be readable and instructive
- **Minimal abstraction**: Avoid over-engineering; focus on core concepts
- **Numerical stability**: Handle edge cases like division by zero, NaN propagation

---

## TROUBLESHOOTING

### Common Issues
```bash
# Memory issues with large tensors
# Use smaller batch sizes or implement gradient checkpointing

# Numerical instability
# Add epsilon to denominators, use stable implementations

# Slow performance
# Profile with: python -m cProfile -o profile.stats examples/simple_neural_net.py

# Import errors
source .venv/bin/activate
uv sync
```

### Debugging Tips
1. **Visualize computational graph**:
```python
def print_graph(tensor, indent=0):
    print(" " * indent + f"Tensor(shape={tensor.shape}, op={tensor._op})")
    for parent in tensor._parents:
        print_graph(parent, indent + 2)
```

2. **Check gradient correctness** with finite differences:
```python
def gradient_check(func, tensor, eps=1e-5):
    """Compare autograd gradient with finite differences."""
    analytic_grad = tensor.grad
    
    # Compute numerical gradient
    numerical_grad = np.zeros_like(tensor.data)
    it = np.nditer(tensor.data, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        original = tensor.data[idx]
        
        tensor.data[idx] = original + eps
        loss_plus = func()
        
        tensor.data[idx] = original - eps
        loss_minus = func()
        
        numerical_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
        tensor.data[idx] = original
        
        it.iternext()
    
    diff = np.abs(analytic_grad - numerical_grad).max()
    return diff < 1e-7, diff
```

---

## CONTRIBUTION GUIDELINES

### Pull Request Requirements
1. **Educational Value**: Changes should improve understanding of PyTorch internals
2. **Test Coverage**: New features must include comprehensive tests
3. **Documentation**: Update relevant docstrings and examples
4. **Performance**: Benchmark critical operations
5. **API Consistency**: Follow existing PyTorch-like API patterns

### Implementation Checklist
- [ ] Forward pass correctly implemented
- [ ] Backward pass correctly implemented with gradient checking
- [ ] Shape validation and error handling
- [ ] Unit tests with edge cases
- [ ] Example usage in examples/
- [ ] Documentation in docstrings

### Code Review Focus Areas
1. **Numerical correctness**: Gradient checking passing
2. **Memory efficiency**: No unnecessary tensor copies
3. **API design**: Consistent with PyTorch patterns
4. **Error handling**: Clear error messages for common mistakes
5. **Test coverage**: Edge cases and typical usage

---

**Educational Goal**: This project aims to teach how PyTorch works internally, not to replace PyTorch for production use.
**Reference**: PyTorch documentation, micrograd, tinygrad implementations
**License**: MIT
